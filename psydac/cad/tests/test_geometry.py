#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from itertools import product

import pytest
import numpy as np
from mpi4py import MPI

from sympde.topology import Domain, Line, Square, Cube, Mapping, IdentityMapping

from psydac.cad.geometry             import Geometry, export_nurbs_to_hdf5, refine_nurbs
from psydac.cad.geometry             import import_geopdes_to_nurbs
from psydac.cad.cad                  import elevate, refine
from psydac.cad.gallery              import quart_circle
from psydac.cad.tests.test_cad       import assert_same_geometry, make_curve, make_surface
from psydac.mapping.discrete         import SplineMapping, NurbsMapping
from psydac.mapping.discrete_gallery import discrete_mapping
from psydac.fem.splines              import SplineSpace
from psydac.fem.tensor               import TensorFemSpace
from psydac.utilities.utils          import refine_array_1d
from psydac.ddm.cart                 import (DomainDecomposition,
                                             MultiPatchDomainDecomposition)


import psydac.cad.mesh as mesh_mod

base_dir = os.path.dirname(os.path.realpath(__file__))

# Unit patch of each logical dimension
PATCHES = {1: Line, 2: Square}
#==============================================================================
def assert_geometries_equal(geo: Geometry, ref: Geometry) -> None:
    """Check that two geometries are equal, comparing their patches by position."""
    assert geo.ldim == ref.ldim
    assert geo.pdim == ref.pdim
    assert list(geo.ncells.values()) == list(ref.ncells.values())
    assert [list(p) for p in geo.periodic.values()] == [list(p) for p in ref.periodic.values()]
    assert len(geo.mappings) == len(ref.mappings)

    for F, F_ref in zip(geo.mappings.values(), ref.mappings.values()):
        assert type(F) is type(F_ref)
        assert list(F.space.degree) == list(F_ref.space.degree)
        assert all(np.array_equal(k, k_ref) for k, k_ref in zip(F.space.knots, F_ref.space.knots))

        fields     = [*F    .fields, F    .weights_field] if isinstance(F, NurbsMapping) else F    .fields
        fields_ref = [*F_ref.fields, F_ref.weights_field] if isinstance(F, NurbsMapping) else F_ref.fields
        assert all(np.array_equal(f.coeffs.toarray(), f_ref.coeffs.toarray())
                   for f, f_ref in zip(fields, fields_ref))

def check_round_trips(geo: Geometry, mapping: SplineMapping, tmp_path) -> Geometry:
    """Check that a single-patch geometry is preserved by export/read and
    by from_discrete_mapping, and return the geometry read from file."""
    filename = str(tmp_path / 'geo.h5')
    geo.export(filename)
    geo_read = Geometry.from_file(filename)
    assert_geometries_equal(geo_read, geo)

    # A geometry read from file is exported again without changes
    filename_again = str(tmp_path / 'geo_again.h5')
    geo_read.export(filename_again)
    assert_geometries_equal(Geometry.from_file(filename_again), geo)

    assert_geometries_equal(Geometry.from_discrete_mapping(mapping), geo)

    return geo_read

def make_identity_mapping(ddm: DomainDecomposition, degree: list[int], shift: float = 0.0) -> SplineMapping:
    """Create a spline mapping on the decomposition, equal to the identity
    shifted along x. Its control points are given by the Greville abscissae."""
    spaces = [SplineSpace(degree=p, grid=np.linspace(0.0, 1.0, n + 1))
              for p, n in zip(degree, ddm.ncells)]
    grevilles = [W.greville for W in spaces]
    grevilles[0] = grevilles[0] + shift
    points = np.stack(np.meshgrid(*grevilles, indexing='ij'), axis=-1)
    return SplineMapping.from_control_points(TensorFemSpace(ddm, *spaces), points)

#==============================================================================
@pytest.mark.parametrize('ldim', [1, 2])
def test_geometry_identity_spline(ldim: int, tmp_path) -> None:

    ddm = DomainDecomposition([2, 3][:ldim], [False] * ldim)
    mapping = make_identity_mapping(ddm, degree=[2] * ldim)
    domain = Mapping('F', dim=ldim)(PATCHES[ldim](name='Omega'))

    geo = Geometry(domain, ddm=ddm, pdim=ldim, mappings={domain.name: mapping})
    geo_read = check_round_trips(geo, mapping, tmp_path)

    # The mapping read from file is the identity
    F_read, = geo_read.mappings.values()
    eta = list(product(np.linspace(0.0, 1.0, 5), repeat=ldim))
    assert np.allclose([F_read(*e) for e in eta], eta, rtol=0, atol=1e-15)

#==============================================================================
@pytest.mark.parametrize('make_mapping', [
    lambda: make_curve(nurbs=True),
    lambda: make_curve(nurbs=True, pdim=2),
    lambda: make_curve(nurbs=True, pdim=3),
    lambda: make_surface(pdim=3),
], ids=['line', 'curve_2d', 'curve_3d', 'surface_3d'])
def test_geometry_nurbs_manifold(make_mapping, tmp_path) -> None:

    mapping = elevate(make_mapping(), axis=0, times=1)
    mapping = refine(mapping, axis=0, values=[0.1, 0.5])
    ldim, pdim = mapping.ldim, mapping.pdim
    domain = Mapping('F', ldim=ldim, pdim=pdim)(PATCHES[ldim](name='Omega'))

    geo = Geometry(domain, ddm=mapping.space.domain_decomposition, pdim=pdim,
                   mappings={domain.name: mapping})
    geo_read = check_round_trips(geo, mapping, tmp_path)

    # The mapping read from file is the original one
    F_read, = geo_read.mappings.values()
    assert_same_geometry(F_read, mapping)

#==============================================================================
def test_geometry_nurbs_quarter_annulus(tmp_path) -> None:

    # create a nurbs mapping
    rmin, rmax = 0.5, 1.0
    degrees, knots, points, weights = quart_circle( rmin=rmin, rmax=rmax, center=None )

    # Create tensor spline space, distributed
    spaces = [SplineSpace( knots=k, degree=p ) for k,p in zip(knots, degrees)]

    ncells   = [len(space.breaks)-1 for space in spaces]
    domain_decomposition = DomainDecomposition(ncells=ncells, periods=[False]*2, comm=None)

    space = TensorFemSpace( domain_decomposition, *spaces )

    mapping = NurbsMapping.from_control_points_weights( space, points, weights )

    mapping = elevate( mapping, axis=0, times=1 )
    mapping = refine( mapping, axis=0, values=[0.3, 0.6, 0.8] )

    # create a topological domain
    F      = Mapping('F', dim=2)
    domain = F(Square(name='Omega'))

    # associate the mapping to the topological domain
    mappings = {domain.name: mapping}

    # create a geometry from a topological domain and the dict of mappings
    geo = Geometry(domain, ddm=mapping.space.domain_decomposition, pdim=2,
                   mappings=mappings)

    geo_read = check_round_trips(geo, mapping, tmp_path)

    # The mapping read from file is a quarter annulus, radial along axis 1
    F_read = list(geo_read.mappings.values())[0]
    t = np.linspace(0.0, 1.0, 9)
    assert np.allclose([np.hypot(*F_read(e1, 0.0)) for e1 in t], rmin, rtol=0, atol=1e-14)
    assert np.allclose([np.hypot(*F_read(e1, 1.0)) for e1 in t], rmax, rtol=0, atol=1e-14)

#==============================================================================
@pytest.mark.mpi
def test_geometry_with_mpi_dims_mask(mpi_tmp_path):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    mpi_dims_mask = [False, True, False]  # We will verify that this has an effect
    ncells = [4, 2*size, 8]  # Each process should have two cells along x2
    degree = [2, 2, 2]

    expected_starts = (0, 2 * rank, 0)
    expected_ends   = (3, 2 * rank + 1, 7)
    
    # create an identity mapping
    mapping = discrete_mapping('identity', ncells=ncells, degree=degree)

    # create a topological domain
    F = Mapping('F', dim=3)
    domain = F(Cube(name='Omega'))

    # associate the mapping to the topological domain
    mappings = {domain.name: mapping}

    # Create a geometry from a topological domain and the dict of mappings
    # Here we allow for any distribution of the domain: mpi_dims_mask is not passed
    geo = Geometry(domain, ddm=mapping.space.domain_decomposition, pdim=3,
                   mappings=mappings)
    filename = str(mpi_tmp_path / 'geo_mpi_dims.h5')
    geo.export(filename)

    # Read geometry file in parallel, but using mpi_dims_mask
    geo_from_file = Geometry.from_file(filename=filename, comm=comm, mpi_dims_mask=mpi_dims_mask)

    # Verify that the domain is distributed as expected
    assert geo_from_file.ddm.starts == expected_starts
    assert geo_from_file.ddm.ends   == expected_ends

# ==============================================================================
@pytest.mark.mpi
def test_from_discrete_mapping():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    mpi_dims_mask = [False, False, True]  # We will verify that this has an effect
    ncells = [4, 8, 2 * size]  # Each process should have two cells along x3
    degree = [3, 3, 2]  # Each process must own at least `degree` cells

    expected_starts = (0, 0, 2 * rank)
    expected_ends   = (3, 7, 2 * rank + 1)

    # Create a mapping using mpi_dims_mask
    mapping = discrete_mapping('identity', ncells=ncells, degree=degree,
                               mpi_dims_mask=mpi_dims_mask)

    # The geometry uses the domain decomposition of the mapping
    geo_from_mapping = Geometry.from_discrete_mapping(mapping)
    assert geo_from_mapping.ddm is mapping.space.domain_decomposition

    # Verify that the domain is distributed as expected
    assert geo_from_mapping.ddm.starts == expected_starts
    assert geo_from_mapping.ddm.ends   == expected_ends

# ==============================================================================
@pytest.mark.mpi
def test_from_topological_domain():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    mpi_dims_mask = [False, True, False]  # We will verify that this has an effect
    ncells = [4, 2 * size, 8]  # Each process should have two cells along x2

    expected_starts = (0, 2 * rank, 0)
    expected_ends   = (3, 2 * rank + 1, 7)

    # Create a topological domain
    F = Mapping('F', dim=3)
    domain = F(Cube(name='Omega'))

    # Create geometry from topological domain using mpi_dims_mask
    geo_from_domain = Geometry.from_topological_domain(domain, ncells, comm=comm, mpi_dims_mask=mpi_dims_mask)

    # Verify that the domain is distributed as expected
    assert geo_from_domain.ddm.starts == expected_starts
    assert geo_from_domain.ddm.ends   == expected_ends

# ==============================================================================
@pytest.mark.parametrize('ldim', [1, 2])
@pytest.mark.parametrize('npatches', [1, 2])
def test_geometry_init_without_mappings(npatches: int, ldim: int) -> None:

    domain = make_domain(npatches, ldim=ldim)

    ncells = [3, 5][:ldim]
    if npatches == 1:
        ddm = DomainDecomposition(ncells, [False] * ldim)
    else:
        ddm = MultiPatchDomainDecomposition([ncells] * npatches,
                                            [[False] * ldim] * npatches)

    geo = Geometry(domain, ddm=ddm, pdim=ldim)

    # The number of cells and the periodicity are obtained from the decomposition
    assert geo.ncells   == {name: tuple(ncells) for name in domain.interior_names}
    assert geo.periodic == {name: (False,) * ldim for name in domain.interior_names}

    assert geo.mappings == {name: None for name in domain.interior_names}

#==============================================================================
def test_geometry_from_file_multipatch() -> None:

    mesh_dir = os.path.dirname(mesh_mod.__file__)
    filename = os.path.join(mesh_dir, 'multipatch', 'magnet.h5')
    geo = Geometry.from_file(filename)
    names = geo.domain.interior_names

    # Patch information is keyed by the interior names, in the same order
    assert list(geo.ncells)   == names
    assert list(geo.periodic) == names
    assert list(geo.mappings) == names
    assert all(isinstance(n, tuple) and len(n) == 2 for n in geo.ncells.values())
    assert all(p == (False, False) for p in geo.periodic.values())

    # Each spline mapping is defined on the decomposition of its patch
    for F, patch_ddm in zip(geo.mappings.values(), geo.ddm.domains):
        assert F.space.domain_decomposition is patch_ddm

#==============================================================================
def make_domain(npatches: int, *, ldim: int = 2, ornt: int = 1) -> Domain:
    """Create a domain made of one or two unit lines or squares, with generic mappings."""
    Patch = PATCHES[ldim]
    if npatches == 1:
        return Mapping('G', dim=ldim)(Patch('P'))

    A = Mapping('GA', dim=ldim)(Patch('PA'))
    B = Mapping('GB', dim=ldim)(Patch('PB'))
    # 1D interfaces have no orientation
    return Domain.join([A, B], [((0, 0, 1), (1, 0, -1), ornt if ldim > 1 else None)], 'Omega')

def export_geometry(domain: Domain, filename: str) -> None:
    """Export a spline geometry on the domain, with patches side by side along x."""
    names = domain.interior_names
    ldim = domain.dim
    ncells = [2, 3][:ldim]
    if len(domain) == 1:
        ddm = DomainDecomposition(ncells, [False] * ldim)
        patch_ddms = [ddm]
    else:
        ddm = MultiPatchDomainDecomposition([ncells] * len(names),
                                            [[False] * ldim] * len(names))
        patch_ddms = ddm.domains

    # Spline mappings must be defined on the decomposition of their patch
    mappings = {name: make_identity_mapping(patch_ddm, degree=[2] * ldim, shift=i)
                for i, (name, patch_ddm) in enumerate(zip(names, patch_ddms))}

    Geometry(domain, ddm=ddm, pdim=ldim, mappings=mappings).export(filename)

#==============================================================================
@pytest.mark.parametrize('ldim', [1, 2])
@pytest.mark.parametrize('npatches', [1, 2])
def test_geometry_from_file_with_domain(npatches: int, ldim: int, tmp_path) -> None:

    domain = make_domain(npatches, ldim=ldim)
    filename = str(tmp_path / 'geo.h5')
    export_geometry(domain, filename)

    geo = Geometry.from_file(filename, domain=domain)

    # The given domain is kept, and the spline mappings are attached to it
    assert geo.domain is domain
    patches = [domain.interior] if npatches == 1 else domain.interior.args
    for patch, F in zip(patches, geo.mappings.values()):
        assert patch.mapping.get_callable_mapping() is F

#==============================================================================
@pytest.mark.parametrize(('npatches', 'make_wrong_domain', 'message'), [
    (1, lambda: IdentityMapping('G', dim=2)(Square('P')),      'non-analytical'),
    (1, lambda: Square('P'),                                     'non-analytical'),
    (1, lambda: Mapping('G', dim=3)(Cube('P')),                 'dimension'),
    (1, lambda: Mapping('H', dim=2)(Square('P')),               'Patch names'),
    (1, lambda: Mapping('G', dim=2)(Square('P', bounds1=(0, 2))), 'Parametric bounds'),
    (2, lambda: make_domain(2, ornt=-1),                         'Interfaces'),
], ids=['analytical', 'no-mapping', 'dimension', 'names', 'bounds', 'orientation'])
def test_geometry_from_file_with_wrong_domain(npatches: int, make_wrong_domain, message: str, tmp_path) -> None:

    filename = str(tmp_path / 'geo.h5')
    export_geometry(make_domain(npatches), filename)

    with pytest.raises(ValueError, match=message):
        Geometry.from_file(filename, domain=make_wrong_domain())

#==============================================================================
@pytest.mark.parametrize( 'ncells', [[8,8], [12,12], [14,14]] )
@pytest.mark.parametrize( 'degree', [[2,2], [3,2], [2,3], [3,3], [4,4]] )
def test_export_nurbs_to_hdf5(ncells, degree, tmp_path):

    # create pipe geometry
    from igakit.cad import circle, ruled, bilinear, join
    C0      = circle(center=(-1,0),angle=(-np.pi/3,0))
    C1      = circle(radius=2,center=(-1,0),angle=(-np.pi/3,0))
    annulus = ruled(C0,C1).transpose()
    square  = bilinear(np.array([[[0,0],[0,3]],[[1,0],[1,3]]]) )
    pipe    = join(annulus, square, axis=1)

    # refine the nurbs object
    new_pipe = refine_nurbs(pipe, ncells=ncells, degree=degree)

    filename = str(tmp_path / "pipe.h5")
    export_nurbs_to_hdf5(filename, new_pipe)

   # read the geometry
    geo = Geometry.from_file(filename)
    domain = geo.domain

    min_coords = domain.logical_domain.min_coords
    max_coords = domain.logical_domain.max_coords

    assert abs(min_coords[0] - pipe.breaks(0)[0])<1e-15
    assert abs(min_coords[1] - pipe.breaks(1)[0])<1e-15

    assert abs(max_coords[0] - pipe.breaks(0)[-1])<1e-15
    assert abs(max_coords[1] - pipe.breaks(1)[-1])<1e-15

    mapping = geo.mappings[domain.interior.name]

    assert isinstance(mapping, NurbsMapping)

    space  = mapping.space
    knots  = space.knots
    degree = space.degree

    assert all(np.allclose(pk,k, 1e-15, 1e-15) for pk,k in zip(new_pipe.knots, knots))
    assert degree == list(new_pipe.degree)

    assert np.allclose(new_pipe.weights.flatten(), mapping._weights_field.coeffs.toarray(), 1e-15, 1e-15)

    eta1 = refine_array_1d(new_pipe.breaks(0), 10)
    eta2 = refine_array_1d(new_pipe.breaks(1), 10)

    pcoords1 = np.array([[new_pipe(e1,e2) for e2 in eta2] for e1 in eta1])
    pcoords2 = np.array([[mapping(e1,e2) for e2 in eta2] for e1 in eta1])

    assert np.allclose(pcoords1[..., :domain.dim], pcoords2, 1e-15, 1e-15)

#==============================================================================
@pytest.mark.parametrize( 'ncells', [[8,8], [12,12], [14,14]] )
@pytest.mark.parametrize( 'degree', [[2,2], [3,2], [2,3], [3,3], [4,4]] )
def test_import_geopdes_to_nurbs(ncells, degree, tmp_path):

    filename = os.path.join(base_dir, "geo_Lshaped_C1.txt")
    L_shaped = import_geopdes_to_nurbs(filename)

    # refine the nurbs object
    L_shaped = refine_nurbs(L_shaped, ncells=ncells, degree=degree)

    filename = str(tmp_path / "L_shaped.h5")
    export_nurbs_to_hdf5(filename, L_shaped)

   # read the geometry
    geo = Geometry.from_file(filename)
    domain = geo.domain

    min_coords = domain.logical_domain.min_coords
    max_coords = domain.logical_domain.max_coords

    assert abs(min_coords[0] - L_shaped.breaks(0)[0])<1e-15
    assert abs(min_coords[1] - L_shaped.breaks(1)[0])<1e-15

    assert abs(max_coords[0] - L_shaped.breaks(0)[-1])<1e-15
    assert abs(max_coords[1] - L_shaped.breaks(1)[-1])<1e-15

    mapping = geo.mappings[domain.interior.name]

    space  = mapping.space
    knots  = space.knots
    degree = space.degree

    assert all(np.allclose(pk,k, 1e-15, 1e-15) for pk,k in zip(L_shaped.knots, knots))
    assert degree == list(L_shaped.degree)

    if isinstance(mapping, NurbsMapping):
        assert np.allclose(L_shaped.weights.flatten(), mapping._weights_field.coeffs.toarray(), 1e-15, 1e-15)

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
