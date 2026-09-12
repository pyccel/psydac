#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os

import pytest
import numpy as np
from mpi4py import MPI

from sympde.topology import Domain, Line, Square, Cube, SymbolicMapping

from psydac.cad.geometry             import Geometry, export_nurbs_to_hdf5, refine_nurbs
from psydac.cad.geometry             import import_geopdes_to_nurbs
from psydac.cad.cad                  import elevate, refine
from psydac.cad.gallery              import quart_circle, circle
from psydac.mapping.discrete         import SplineMapping, NurbsMapping
from psydac.mapping.discrete_gallery import discrete_mapping
from psydac.fem.splines              import SplineSpace
from psydac.fem.tensor               import TensorFemSpace
from psydac.utilities.utils          import refine_array_1d
from psydac.ddm.cart                 import DomainDecomposition


base_dir = os.path.dirname(os.path.realpath(__file__))
#==============================================================================
@pytest.mark.xdist_group('h5py')
def test_geometry_2d_1():

    ncells = [1,1]
    degree = [2,2]
    # create an identity mapping
    mapping = discrete_mapping('identity', ncells=ncells, degree=degree)

    # create a topological domain
    F      = SymbolicMapping('F', dim=2)
    domain = F(Square(name='Omega'))

    # associate the mapping to the topological domain
    mappings = {domain.name: mapping}

    # Define ncells as a dict
    ncells = {domain.name:ncells}

    # create a geometry from a topological domain and the dict of mappings
    geo = Geometry(domain=domain, pdim=2, ncells=ncells, mappings=mappings)

    # export the geometry
    geo.export('geo.h5')

    # read it again
    geo_0 = Geometry.from_file('geo.h5')

    # export it again
    geo_0.export('geo_0.h5')

    # create a geometry from a discrete mapping
    geo_1 = Geometry.from_discrete_mapping(mapping)

    # export it
    geo_1.export('geo_1.h5')

#==============================================================================
@pytest.mark.xdist_group('h5py')
def test_geometry_2d_2():

    # create a nurbs mapping
    degrees, knots, points, weights = quart_circle( rmin=0.5, rmax=1.0, center=None )

    # Create tensor spline space, distributed
    spaces = [SplineSpace( knots=k, degree=p ) for k,p in zip(knots, degrees)]

    ncells   = [len(space.breaks)-1 for space in spaces]
    domain_decomposition = DomainDecomposition(ncells=ncells, periods=[False]*2, comm=None)

    space = TensorFemSpace( domain_decomposition, *spaces )

    mapping = NurbsMapping.from_control_points_weights( space, points, weights )

    mapping = elevate( mapping, axis=0, times=1 )
    mapping = refine( mapping, axis=0, values=[0.3, 0.6, 0.8] )

    # create a topological domain
    F      = SymbolicMapping('F', dim=2)
    domain = F(Square(name='Omega'))

    # associate the mapping to the topological domain
    mappings = {domain.name: mapping}

    # Define ncells as a dict
    ncells = {domain.name:[len(space.breaks)-1 for space in mapping.space.spaces]}

    periodic = {domain.name:[space.periodic for space in mapping.space.spaces]}

    # create a geometry from a topological domain and the dict of mappings
    geo = Geometry(domain=domain, pdim=2, ncells=ncells, periodic=periodic, mappings=mappings)

    # export the geometry
    geo.export('quart_circle.h5')

    # read it again
    geo_0 = Geometry.from_file('quart_circle.h5')

    # export it again
    geo_0.export('quart_circle_0.h5')

    # create a geometry from a discrete mapping
    geo_1 = Geometry.from_discrete_mapping(mapping)

    # export it
    geo_1.export('quart_circle_1.h5')

#==============================================================================
# TODO to be removed
@pytest.mark.xdist_group('h5py')
def test_geometry_2d_3():

    # create a nurbs mapping
    degrees, knots, points, weights = quart_circle( rmin=0.5, rmax=1.0, center=None )

    # Create tensor spline space, distributed
    spaces = [SplineSpace( knots=k, degree=p ) for k,p in zip(knots, degrees)]
    ncells   = [len(space.breaks)-1 for space in spaces]
    domain_decomposition = DomainDecomposition(ncells=ncells, periods=[False]*2, comm=None)

    space = TensorFemSpace( domain_decomposition, *spaces )

    mapping = NurbsMapping.from_control_points_weights( space, points, weights )

    mapping = elevate( mapping, axis=1, times=1 )

    n = 8
    t = np.linspace(0, 1, n+1)[1:-1]

    # TODO allow for 1d numpy array
    t = list(t)

    for axis in [0, 1]:
        mapping = refine( mapping, axis=axis, values=t )

    # create a geometry from a discrete mapping
    geo = Geometry.from_discrete_mapping(mapping)

    # export it
    geo.export('quart_circle.h5')

#==============================================================================
# TODO to be removed
@pytest.mark.xdist_group('h5py')
def test_geometry_2d_4():

    # create a nurbs mapping
    radius = np.sqrt(2)/2.
    degrees, knots, points, weights = circle( radius=radius, center=None )

    # Create tensor spline space, distributed
    spaces = [SplineSpace( knots=k, degree=p ) for k,p in zip(knots, degrees)]
    ncells   = [len(space.breaks)-1 for space in spaces]
    domain_decomposition = DomainDecomposition(ncells=ncells, periods=[False]*2, comm=None)

    space = TensorFemSpace( domain_decomposition, *spaces )

    mapping = NurbsMapping.from_control_points_weights( space, points, weights )

    n = 8
#    n = 32
    t = np.linspace(0, 1, n+1)[1:-1]

    # TODO allow for 1d numpy array
    t = list(t)

    for axis in [0, 1]:
        mapping = refine( mapping, axis=axis, values=t )

    # create a geometry from a discrete mapping
    geo = Geometry.from_discrete_mapping(mapping)

    # export it
    geo.export('circle.h5')

#==============================================================================
@pytest.mark.mpi
def test_geometry_with_mpi_dims_mask():

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
    F = SymbolicMapping('F', dim=3)
    domain = F(Cube(name='Omega'))

    # associate the mapping to the topological domain
    mappings = {domain.name: mapping}

    # Define d_ncells as a dict
    d_ncells = {domain.name: ncells}

    # Create a geometry from a topological domain and the dict of mappings
    # Here we allow for any distribution of the domain: mpi_dims_mask is not passed
    geo = Geometry(domain=domain, pdim=3, ncells=d_ncells, mappings=mappings, comm=comm)
    geo.export('geo_mpi_dims.h5')

    # Read geometry file in parallel, but using mpi_dims_mask
    geo_from_file = Geometry.from_file(filename='geo_mpi_dims.h5', comm=comm, mpi_dims_mask=mpi_dims_mask)

    # Verify that the domain is distributed as expected
    assert geo_from_file.ddm.starts == expected_starts
    assert geo_from_file.ddm.ends   == expected_ends

    # Safely remove the file
    comm.Barrier()
    if rank == 0:
        os.remove('geo_mpi_dims.h5')

# ==============================================================================
@pytest.mark.mpi
def test_from_discrete_mapping():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    mpi_dims_mask = [False, False, True]  # We will verify that this has an effect
    ncells = [4, 8, 2 * size]  # Each process should have two cells along x3
    degree = [3, 3, 3]

    expected_starts = (0, 0, 2 * rank)
    expected_ends   = (3, 7, 2 * rank + 1)

    # Create a mapping
    mapping = discrete_mapping('identity', ncells=ncells, degree=degree)

    # Create geometry from the mapping using mpi_dims_mask
    geo_from_mapping = Geometry.from_discrete_mapping(mapping, comm=comm, mpi_dims_mask=mpi_dims_mask)

    # Verify that the domain is distributed as expected
    assert geo_from_mapping.ddm.starts == expected_starts
    assert geo_from_mapping.ddm.ends   == expected_ends

# ==============================================================================
def test_spline_mapping_to_defined_mapping_and_geometry_domain_log():
    # WP07b / WP07b-2: SplineMapping.to_defined_mapping wraps the spline in a
    # DiscreteMapping (symbolic carrier, is_analytical=False); Geometry.
    # from_discrete_mapping is built on it, and the logical (parametric) domain
    # tracks -- or is validated against -- the spline's own parametric box.
    from sympde.topology.mapping import DiscreteMapping

    ncells, degree = [4, 4], [2, 2]
    spl = discrete_mapping('identity', ncells=ncells, degree=degree)   # box [0, 1]^2

    M = spl.to_defined_mapping('M')
    assert isinstance(M, DiscreteMapping)
    assert M.get_callable_mapping() is spl
    assert M.is_analytical is False
    assert (M.ldim, M.pdim) == (2, 2)

    # default logical domain: spans the spline's parametric box ([0, 1]^2 here)
    geo0 = Geometry.from_discrete_mapping(spl, name='g0')
    assert isinstance(geo0.domain.mapping, DiscreteMapping)
    assert geo0.domain.mapping.get_callable_mapping() is spl
    assert geo0.domain.mapping.is_analytical is False
    assert geo0.domain.logical_domain.min_coords == (0., 0.)
    assert geo0.domain.logical_domain.max_coords == (1., 1.)

    # a non-unit parametric box: the default logical domain tracks it, and is
    # NOT silently [0, 1]^2.
    qa = discrete_mapping('quarter_annulus', ncells=ncells, degree=degree)  # ((1, 4), (0, pi/2))
    geo_qa = Geometry.from_discrete_mapping(qa, name='gqa')
    assert np.allclose(geo_qa.domain.logical_domain.min_coords, (1., 0.))
    assert np.allclose(geo_qa.domain.logical_domain.max_coords, (4., np.pi / 2))

    # an explicit domain_log that matches the spline box is accepted
    L_ok = Square('L', bounds1=(1., 4.), bounds2=(0., np.pi / 2))
    geo1 = Geometry.from_discrete_mapping(qa, name='g1', domain_log=L_ok)
    assert np.allclose(geo1.domain.logical_domain.min_coords, (1., 0.))
    assert np.allclose(geo1.domain.logical_domain.max_coords, (4., np.pi / 2))
    assert geo1.domain.mapping.get_callable_mapping() is qa

    # an explicit domain_log whose extent disagrees with the spline is rejected
    L_bad = Square('L', bounds1=(0., 2.), bounds2=(0., 1.))
    with pytest.raises(ValueError):
        Geometry.from_discrete_mapping(qa, name='g2', domain_log=L_bad)

    # wrong dimensionality is rejected
    with pytest.raises(ValueError):
        Geometry.from_discrete_mapping(
            qa, name='g3',
            domain_log=Cube('C', bounds1=(1., 4.), bounds2=(0., np.pi / 2), bounds3=(0., 1.)))

# ==============================================================================
def _two_patch_spline_annulus(degree=(2, 2), ncells=(6, 6)):
    """Two 90-deg annular patches, each a SplineMapping approx of a PolarMapping,
    joined into a DiscreteMapping-carried multipatch Domain."""
    from sympde.topology import PolarMapping

    A = Square('A', bounds1=(0.5, 1.0), bounds2=(0.0,      np.pi / 2))
    B = Square('B', bounds1=(0.5, 1.0), bounds2=(np.pi / 2, np.pi   ))

    def approx(pm, sq):
        grids = [np.linspace(sq.min_coords[d], sq.max_coords[d], ncells[d] + 1)
                 for d in range(2)]
        V = TensorFemSpace(DomainDecomposition(list(ncells), [False, False]),
                           *[SplineSpace(degree[d], grid=grids[d], periodic=False)
                             for d in range(2)])
        return SplineMapping.from_mapping(V, pm.get_callable_mapping())

    spl_A = approx(PolarMapping('MA', 2, c1=0., c2=0., rmin=0., rmax=1.), A)
    spl_B = approx(PolarMapping('MB', 2, c1=0., c2=0., rmin=0., rmax=1.), B)
    M_A = spl_A.to_defined_mapping('MA')
    M_B = spl_B.to_defined_mapping('MB')
    Omega = Domain.join([M_A(A), M_B(B)], [((0, 1, 1), (1, 1, -1), 1)], 'ann2')
    return Omega, spl_A, spl_B

# ==============================================================================
def _detached_spline_domain(A, name='M', ncells=(4, 4), degree=(2, 2)):
    """(Omega, spl): Omega = M(A) where M is a DiscreteMapping wrapping spline
    `spl`, with M's attached callable then cleared -- simulates a "detached
    carrier" for the callable-less-DiscreteMapping error paths. `spl` is
    returned too so a caller can build another (non-detached) carrier from the
    same spline without rebuilding it."""
    from sympde.topology import PolarMapping

    grids = [np.linspace(A.min_coords[d], A.max_coords[d], ncells[d] + 1) for d in range(2)]
    V = TensorFemSpace(DomainDecomposition(list(ncells), [False, False]),
                       *[SplineSpace(degree[d], grid=grids[d], periodic=False) for d in range(2)])
    spl = SplineMapping.from_mapping(V, PolarMapping(name, 2, c1=0., c2=0., rmin=0., rmax=1.)
                                    .get_callable_mapping())
    M = spl.to_defined_mapping(name)
    Omega = M(A)
    Omega.interior.mapping._callable_map = None       # simulate a detached carrier
    return Omega, spl

# ==============================================================================
def test_from_discrete_domain_2patch():
    # WP07c-1: Geometry.from_discrete_domain on a 2-patch domain whose patches
    # are spline DiscreteMappings builds the coefficient-space interface
    # connectivity that assembling an interface term needs.
    from sympde.topology.mapping import DiscreteMapping
    from psydac.cad.geometry import is_spline_discrete_domain

    Omega, spl_A, spl_B = _two_patch_spline_annulus()
    assert is_spline_discrete_domain(Omega) is True

    geo = Geometry.from_discrete_domain(Omega)
    assert len(geo) == 2
    interiors = list(Omega.interior.args)
    for itr in interiors:
        assert isinstance(itr.mapping, DiscreteMapping)
        assert itr.mapping.is_analytical is False
        sp = geo.mappings[itr.name].space
        # each patch has the coefficient-space interface on its joined axis/ext,
        # on both the base space and the connectivity-refined space
        assert len(sp.interfaces) == 1
        (axis, ext), = sp.interfaces.keys()
        for key in sp._refined_space:
            assert (axis, ext) in sp.get_refined_space(key).interfaces
        # the spline control-point coeffs carry cross-interface data
        assert (axis, ext) in geo.mappings[itr.name].fields[0].coeffs._interface_data

# ==============================================================================
def test_discretize_domain_dispatches_to_from_discrete_domain():
    # WP07c-1: discretize(Omega) with no filename/ncells dispatches to
    # from_discrete_domain when the domain carries spline DiscreteMappings, and
    # still raises ValueError otherwise.
    from psydac.api.discretization import discretize

    Omega, _, _ = _two_patch_spline_annulus()
    geo_a = discretize(Omega)
    geo_b = Geometry.from_discrete_domain(Omega)
    assert isinstance(geo_a, Geometry)
    assert len(geo_a) == len(geo_b) == 2
    assert set(geo_a.mappings) == set(geo_b.mappings)
    assert geo_a.ncells == geo_b.ncells

    plain = Square('P', bounds1=(0., 1.), bounds2=(0., 1.))
    with pytest.raises(ValueError):
        discretize(plain)

# ==============================================================================
def test_is_spline_discrete_domain_is_public():
    # WP07c-1a F5: is_spline_discrete_domain is part of the module's public API.
    import psydac.cad.geometry as geo_mod
    assert 'is_spline_discrete_domain' in geo_mod.__all__
    ns = {}
    exec('from psydac.cad.geometry import *', ns)
    assert 'is_spline_discrete_domain' in ns

# ==============================================================================
def test_from_discrete_domain_callable_less_mapping_raises_typeerror():
    # WP07c-1a F4: a DiscreteMapping with no attached callable must surface as
    # the documented TypeError, not the raw ValueError from get_callable_mapping().
    A = Square('A', bounds1=(0.5, 1.0), bounds2=(0.0, np.pi / 2))
    Omega, _ = _detached_spline_domain(A)

    with pytest.raises(TypeError):
        Geometry.from_discrete_domain(Omega)

# ==============================================================================
def test_patch_spline_classification():
    # WP07d-1 F3/F5: _patch_spline is the single classification shared by
    # is_spline_discrete_domain and Geometry.from_discrete_domain.
    from sympde.topology import PolarMapping
    from psydac.cad.geometry import _patch_spline, is_spline_discrete_domain

    A = Square('A', bounds1=(0.5, 1.0), bounds2=(0.0, np.pi / 2))

    # 1. no mapping at all (bare topological patch)
    assert A.interior.mapping is None
    assert _patch_spline(A.interior) is None
    assert is_spline_discrete_domain(A) is False

    # 2. an analytic (non-spline) mapping
    F = PolarMapping('F', dim=2, c1=0., c2=0., rmin=0.5, rmax=1.0)
    Omega_analytic = F(A)
    assert _patch_spline(Omega_analytic.interior) is None
    assert is_spline_discrete_domain(Omega_analytic) is False

    # 3. a spline DiscreteMapping with no attached callable
    Omega_detached, spl = _detached_spline_domain(A)
    assert _patch_spline(Omega_detached.interior) is None
    assert is_spline_discrete_domain(Omega_detached) is False

    # 4. a real spline DiscreteMapping
    M2 = spl.to_defined_mapping('M2')
    Omega_spline = M2(A)
    assert _patch_spline(Omega_spline.interior) is spl
    assert is_spline_discrete_domain(Omega_spline) is True

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
    F = SymbolicMapping('F', dim=3)
    domain = F(Cube(name='Omega'))

    # Create geometry from topological domain using mpi_dims_mask
    geo_from_domain = Geometry.from_topological_domain(domain, ncells, comm=comm, mpi_dims_mask=mpi_dims_mask)

    # Verify that the domain is distributed as expected
    assert geo_from_domain.ddm.starts == expected_starts
    assert geo_from_domain.ddm.ends   == expected_ends

#==============================================================================
@pytest.mark.parametrize( 'ncells', [[8,8], [12,12], [14,14]] )
@pytest.mark.parametrize( 'degree', [[2,2], [3,2], [2,3], [3,3], [4,4]] )
@pytest.mark.xdist_group('h5py')
def test_export_nurbs_to_hdf5(ncells, degree):

    # create pipe geometry
    from igakit.cad import circle, ruled, bilinear, join
    C0      = circle(center=(-1,0),angle=(-np.pi/3,0))
    C1      = circle(radius=2,center=(-1,0),angle=(-np.pi/3,0))
    annulus = ruled(C0,C1).transpose()
    square  = bilinear(np.array([[[0,0],[0,3]],[[1,0],[1,3]]]) )
    pipe    = join(annulus, square, axis=1)

    # refine the nurbs object
    new_pipe = refine_nurbs(pipe, ncells=ncells, degree=degree)

    filename = "pipe.h5"
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

    mapping = geo.mappings[domain.logical_domain.name]

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
@pytest.mark.xdist_group('h5py')
def test_import_geopdes_to_nurbs(ncells, degree):

    filename = os.path.join(base_dir, "geo_Lshaped_C1.txt")
    L_shaped = import_geopdes_to_nurbs(filename)

    # refine the nurbs object
    L_shaped = refine_nurbs(L_shaped, ncells=ncells, degree=degree)

    filename = "L_shaped.h5"
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

    mapping = geo.mappings[domain.logical_domain.name]

    space  = mapping.space
    knots  = space.knots
    degree = space.degree

    assert all(np.allclose(pk,k, 1e-15, 1e-15) for pk,k in zip(L_shaped.knots, knots))
    assert degree == list(L_shaped.degree)

    if isinstance(mapping, NurbsMapping):
        assert np.allclose(L_shaped.weights.flatten(), mapping._weights_field.coeffs.toarray(), 1e-15, 1e-15)

#==============================================================================
@pytest.mark.xfail
def test_geometry_1():

    line   = Geometry.as_line(ncells=[10])
    square = Geometry.as_square(ncells=[10, 10])
    cube   = Geometry.as_cube(ncells=[10, 10, 10])

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    import os
    from sympy.core import cache
    cache.clear_cache()

    # Remove HDF5 files generated by Geometry.export()
    filenames = [
        'geo.h5',
        'geo_0.h5',
        'geo_1.h5',
        'quart_circle.h5',
        'quart_circle_0.h5',
        'quart_circle_1.h5',
        'circle.h5',
        'pipe.h5',
        'L_shaped.h5',
    ]
    for fname in filenames:
        if os.path.exists(fname):
            os.remove(fname)

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
