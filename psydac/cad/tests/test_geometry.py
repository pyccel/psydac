#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
import pickle
import tempfile
import warnings

import pytest
import numpy as np
import h5py
import yaml
from mpi4py import MPI

from sympde.topology import Domain, Line, Square, Cube, SymbolicMapping

from psydac.cad.geometry             import Geometry, export_nurbs_to_hdf5, refine_nurbs
from psydac.cad.geometry             import import_geopdes_to_nurbs
from psydac.cad.cad                  import elevate, refine
from psydac.cad.gallery              import quart_circle, circle
from psydac.mapping.discrete         import SplineCallableMapping, NurbsCallableMapping
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

    mapping = NurbsCallableMapping.from_control_points_weights( space, points, weights )

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

    mapping = NurbsCallableMapping.from_control_points_weights( space, points, weights )

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

    mapping = NurbsCallableMapping.from_control_points_weights( space, points, weights )

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
    # WP07b / WP07b-2: SplineCallableMapping.to_defined_mapping wraps the spline in a
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
    """Two 90-deg annular patches, each a SplineCallableMapping approx of a PolarMapping,
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
        return SplineCallableMapping.from_mapping(pm, V)

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
    spl = SplineCallableMapping.from_mapping(
        PolarMapping(name, 2, c1=0., c2=0., rmin=0., rmax=1.), V)
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
@pytest.mark.xdist_group('h5py')
def test_from_file_uses_discrete_mapping():
    # WP10: Geometry.from_file (Geometry.read) builds a domain whose per-patch
    # mapping is a spline-backed DiscreteMapping -- consistent with
    # Geometry.from_discrete_domain (WP07c-1) -- instead of a bare
    # SymbolicMapping + set_callable_mapping.
    from sympde.topology.mapping import DiscreteMapping

    # single patch: export/read round trip, as the other from_file tests do
    mapping = discrete_mapping('identity', ncells=[4, 4], degree=[2, 2])
    geo0 = Geometry.from_discrete_mapping(mapping)
    geo0.export('geo_wp10_single.h5')
    geo0_from_file = Geometry.from_file('geo_wp10_single.h5')
    assert isinstance(geo0_from_file.domain.mapping, DiscreteMapping)
    assert geo0_from_file.domain.mapping.is_analytical is False
    assert geo0_from_file.domain.mapping.get_callable_mapping() is not None
    # the DiscreteMapping carrier must not rename the domain: ncells/periodic
    # are keyed by Domain.from_file's interior names (see Geometry.read()),
    # so a mismatch here would silently desync them from self._domain.
    assert geo0_from_file.domain.name == Domain.from_file('geo_wp10_single.h5').name

    # two patches, from a committed multipatch fixture -- exercises the
    # Domain.join connectivity-rebuild path
    filename = os.path.join(base_dir, '..', 'mesh', 'multipatch', 'square.h5')
    geo_mp = Geometry.from_file(filename)
    interiors = list(geo_mp.domain.interior.args)
    assert len(interiors) == 2
    assert geo_mp.domain.interior_names == Domain.from_file(filename).interior_names
    for itr in interiors:
        assert isinstance(itr.mapping, DiscreteMapping)
        assert itr.mapping.is_analytical is False
        # `mappings` is keyed by interior name (WP15) in every constructor,
        # including read() -- see Geometry.read()
        assert itr.mapping.get_callable_mapping() is geo_mp.mappings[itr.name]
    # the interface connectivity survived the rebuild
    assert geo_mp.domain.interfaces is not None

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

    mapping = geo.mappings[domain.name]

    assert isinstance(mapping, NurbsCallableMapping)

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

    mapping = geo.mappings[domain.name]

    space  = mapping.space
    knots  = space.knots
    degree = space.degree

    assert all(np.allclose(pk,k, 1e-15, 1e-15) for pk,k in zip(L_shaped.knots, knots))
    assert degree == list(L_shaped.degree)

    if isinstance(mapping, NurbsCallableMapping):
        assert np.allclose(L_shaped.weights.flatten(), mapping._weights_field.coeffs.toarray(), 1e-15, 1e-15)

#==============================================================================
@pytest.mark.xdist_group('h5py')
def test_geometry_type_tag_is_frozen():
    """Regression net for WP11: the HDF5 'type' tag must stay the literal
    legacy class name, decoupled from `type(mapping).__name__`, so that a
    future rename of `SplineCallableMapping`/`NurbsCallableMapping` cannot change the
    on-disk geometry-file format.
    """
    # Spline (non-rational) patch
    mapping = discrete_mapping('identity', ncells=[4, 4], degree=[2, 2])
    geo = Geometry.from_discrete_mapping(mapping)
    geo.export('geo_wp11_tag.h5')

    with h5py.File('geo_wp11_tag.h5', mode='r') as h5:
        yml = yaml.safe_load(h5['geometry.yml'][()])
    assert yml['patches'][0]['type'] == 'SplineMapping'  # frozen legacy tag, not the class name

    geo_read = Geometry.from_file('geo_wp11_tag.h5')
    assert any(m is not None for m in geo_read.mappings.values())

    # Nurbs patch
    degrees, knots, points, weights = quart_circle(rmin=0.5, rmax=1.0, center=None)
    spaces = [SplineSpace(knots=k, degree=p) for k, p in zip(knots, degrees)]
    ncells = [len(space.breaks) - 1 for space in spaces]
    domain_decomposition = DomainDecomposition(ncells=ncells, periods=[False] * 2, comm=None)
    space = TensorFemSpace(domain_decomposition, *spaces)
    nurbs_mapping = NurbsCallableMapping.from_control_points_weights(space, points, weights)

    geo_nurbs = Geometry.from_discrete_mapping(nurbs_mapping)
    geo_nurbs.export('geo_wp11_tag_nurbs.h5')

    with h5py.File('geo_wp11_tag_nurbs.h5', mode='r') as h5:
        yml = yaml.safe_load(h5['geometry.yml'][()])
    assert yml['patches'][0]['type'] == 'NurbsMapping'  # frozen legacy tag, not the class name

    # Committed fixture must still be readable with the frozen tag
    fixture = os.path.join(base_dir, '..', 'mesh', 'collela_2d.h5')
    geo_fixture = Geometry.from_file(fixture)
    assert any(m is not None for m in geo_fixture.mappings.values())

#==============================================================================
# WP15-0: pin current Geometry per-patch dict-key / export-byte-identity
# behaviour before WP15-1 re-keys everything by interior name. These tests
# must keep passing, unmodified, after WP15-1 (except one deliberately
# flipped pin -- see WP15-1 step 10).
#==============================================================================
_SINGLE_PATCH_FIXTURES = [
    'bent_pipe', 'circle', 'collela_2d', 'collela_3d',
    'identity_2d', 'identity_3d', 'pipe', 'quarter_annulus',
]
_MULTIPATCH_FIXTURES = ['magnet', 'square', 'square_repeated_knots']
_XFAIL_MULTIPATCH_FIXTURES = [
    'plate_with_hole_mp', 'plate_with_hole_mp_6', 'plate_with_hole_mp_7',
]

def _fixture_path(name):
    if name in _MULTIPATCH_FIXTURES or name in _XFAIL_MULTIPATCH_FIXTURES:
        return os.path.join(base_dir, '..', 'mesh', 'multipatch', name + '.h5')
    return os.path.join(base_dir, '..', 'mesh', name + '.h5')

_GEOMETRY_FIXTURE_PARAMS = (
    [pytest.param(name, id=name) for name in _SINGLE_PATCH_FIXTURES + _MULTIPATCH_FIXTURES]
    + [pytest.param(name, marks=pytest.mark.xfail(raises=ValueError, strict=True), id=name)
       for name in _XFAIL_MULTIPATCH_FIXTURES]
)

@pytest.mark.xdist_group('h5py')
@pytest.mark.parametrize('fixture', _GEOMETRY_FIXTURE_PARAMS)
def test_geometry_fixture_export_is_byte_identical(fixture, tmp_path):
    filename = _fixture_path(fixture)
    geo = Geometry.from_file(filename)
    out = str(tmp_path / 'out.h5')
    geo.export(out)

    with h5py.File(filename, mode='r') as h5_orig, h5py.File(out, mode='r') as h5_new:
        assert h5_orig['geometry.yml'][()] == h5_new['geometry.yml'][()]
        assert h5_orig['topology.yml'][()] == h5_new['topology.yml'][()]

        yml = yaml.safe_load(h5_orig['geometry.yml'][()])
        pdim = yml['pdim']
        for patch in yml['patches']:
            mapping_id = patch['mapping_id']
            g_orig = h5_orig[mapping_id]
            g_new  = h5_new[mapping_id]
            np.testing.assert_array_equal(g_orig['points'][..., :pdim], g_new['points'][..., :pdim])
            for d in range(yml['ldim']):
                key = 'knots_{}'.format(d)
                np.testing.assert_array_equal(g_orig[key][:], g_new[key][:])
            assert list(g_orig.attrs['degree'])   == list(g_new.attrs['degree'])
            assert list(g_orig.attrs['periodic']) == list(g_new.attrs['periodic'])
            assert ('weights' in g_orig) == ('weights' in g_new)

    geo_reread = Geometry.from_file(out)
    assert geo_reread.domain.name          == geo.domain.name
    assert geo_reread.domain.interior_names == geo.domain.interior_names
    for name in geo.domain.interior_names:
        assert geo_reread.ncells[name] == geo.ncells[name]

#==============================================================================
def test_geometry_export_names_in_memory_constructors():
    # from_discrete_mapping: single patch, key = interior name "mapping(Omega)"
    mapping = discrete_mapping('identity', ncells=[4, 4], degree=[2, 2])
    geo = Geometry.from_discrete_mapping(mapping)
    with tempfile.TemporaryDirectory() as d:
        f0 = os.path.join(d, 'g0.h5')
        geo.export(f0)
        with h5py.File(f0, mode='r') as h5:
            assert h5['geometry.yml'][()] == (
                b'ldim: 2\npdim: 2\npatches:\n'
                b'- name: mapping(Omega)\n  mapping_id: mapping_0\n  type: SplineMapping\n'
            )

        geo_r = Geometry.from_file(f0)
        f1 = os.path.join(d, 'g1.h5')
        geo_r.export(f1)
        with h5py.File(f0, mode='r') as h5_0, h5py.File(f1, mode='r') as h5_1:
            assert h5_0['geometry.yml'][()]  == h5_1['geometry.yml'][()]
            assert h5_0['topology.yml'][()]  == h5_1['topology.yml'][()]
        # no double wrap (WP10 bug 2's pin)
        assert geo_r.domain.interior_names == ['mapping(Omega)']

    # from_discrete_domain: two patches, keys = interior names "MA(A)", "MB(B)"
    Omega, spl_A, spl_B = _two_patch_spline_annulus()
    geo_mp = Geometry.from_discrete_domain(Omega)
    with tempfile.TemporaryDirectory() as d:
        f0 = os.path.join(d, 'g0.h5')
        geo_mp.export(f0)
        with h5py.File(f0, mode='r') as h5:
            assert h5['geometry.yml'][()] == (
                b'ldim: 2\npdim: 2\npatches:\n'
                b'- name: MA(A)\n  mapping_id: mapping_0\n  type: SplineMapping\n'
                b'- name: MB(B)\n  mapping_id: mapping_1\n  type: SplineMapping\n'
            )

        geo_mp_r = Geometry.from_file(f0)
        assert geo_mp_r.domain.interior_names == ['MA(A)', 'MB(B)']
        assert [itr.logical_domain.name for itr in geo_mp_r.domain.interior.args] == ['A', 'B']
        assert geo_mp_r.domain.interfaces is not None

#==============================================================================
@pytest.mark.xdist_group('h5py')
def test_geometry_mappings_order_matches_interiors():
    # from_file: geo.mappings.values() is positionally ordered like interiors,
    # regardless of the (legacy patch-name) dict keys.
    for fixture in ('square', 'magnet'):
        filename = os.path.join(base_dir, '..', 'mesh', 'multipatch', fixture + '.h5')
        geo = Geometry.from_file(filename)
        interiors = list(geo.domain.interior.args)
        values = list(geo.mappings.values())
        assert len(values) == len(interiors)
        for i, itr in enumerate(interiors):
            assert values[i] is itr.mapping.get_callable_mapping()

    # from_discrete_domain: splines are rebuilt on interface-aware spaces, so
    # compare control points rather than identity.
    Omega, spl_A, spl_B = _two_patch_spline_annulus()
    geo_dd = Geometry.from_discrete_domain(Omega)
    values = list(geo_dd.mappings.values())
    assert np.allclose(values[0].control_points[...], spl_A.control_points[...])
    assert np.allclose(values[1].control_points[...], spl_B.control_points[...])

    # from_discrete_mapping: single patch.
    mapping = discrete_mapping('identity', ncells=[4, 4], degree=[2, 2])
    geo_dm = Geometry.from_discrete_mapping(mapping)
    values = list(geo_dm.mappings.values())
    assert len(values) == 1
    assert values[0] is geo_dm.domain.mapping.get_callable_mapping()

#==============================================================================
@pytest.mark.xdist_group('h5py')
def test_geometry_legacy_patch_key_access():
    # WP15-1: `mappings`/`periodic` are now keyed canonically by interior
    # name; the legacy on-disk patch name / integer index still resolve,
    # but each access warns (this is the one pin deliberately flipped by
    # WP15-1 step 10 -- see test_geometry_legacy_keys_deprecated for the
    # full behaviour of the alias mechanism).
    filename = os.path.join(base_dir, '..', 'mesh', 'multipatch', 'square.h5')
    geo = Geometry.from_file(filename)

    with pytest.warns(DeprecationWarning, match='patch_0'):
        assert geo.mappings['patch_0'] is list(geo.mappings.values())[0]

    assert isinstance(geo.periodic, dict)
    with pytest.warns(DeprecationWarning, match='1'):
        assert geo.periodic[1] == [False, False]

#==============================================================================
@pytest.mark.xdist_group('h5py')
def test_geometry_dict_keys_are_interior_names():
    # WP15-1: `mappings`/`ncells`/`periodic` are keyed by `domain.
    # interior_names`, in interior order, regardless of constructor.
    geometries = []

    # case 0: Geometry(domain=F(Square('Omega')), ...)
    mapping = discrete_mapping('identity', ncells=[2, 2], degree=[2, 2])
    F = SymbolicMapping('F', dim=2)
    domain0 = F(Square(name='Omega'))
    geometries.append(Geometry(domain=domain0, pdim=2,
                               ncells={domain0.name: [2, 2]},
                               mappings={domain0.name: mapping}))

    # from_topological_domain: single and 2 patches
    geometries.append(Geometry.from_topological_domain(F(Square('Sq')), [4, 4]))
    F1 = SymbolicMapping('F1', dim=2)
    F2 = SymbolicMapping('F2', dim=2)
    A_top = F1(Square('A_top'))
    B_top = F2(Square('B_top', bounds1=(1., 2.), bounds2=(0., 1.)))
    Omega_top = Domain.join([A_top, B_top], [((0, 0, 1), (1, 0, -1), 1)], 'top2')
    geometries.append(Geometry.from_topological_domain(Omega_top, [4, 4]))

    # from_discrete_mapping
    geo_dm = Geometry.from_discrete_mapping(mapping)
    geometries.append(geo_dm)

    # from_discrete_domain: single (reuse geo_dm's DiscreteMapping-carried
    # domain) and 2 patches
    geometries.append(Geometry.from_discrete_domain(geo_dm.domain))
    Omega_dd, _, _ = _two_patch_spline_annulus()
    geometries.append(Geometry.from_discrete_domain(Omega_dd))

    # from_file
    for fixture in ('collela_2d', 'bent_pipe'):
        geometries.append(Geometry.from_file(os.path.join(base_dir, '..', 'mesh', fixture + '.h5')))
    for fixture in ('square', 'magnet'):
        geometries.append(Geometry.from_file(os.path.join(base_dir, '..', 'mesh', 'multipatch', fixture + '.h5')))
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'g.h5')
        geo_dm.export(f)
        geometries.append(Geometry.from_file(f))

    # from_file with an explicit (serial) comm
    geometries.append(Geometry.from_file(
        os.path.join(base_dir, '..', 'mesh', 'multipatch', 'square.h5'), comm=MPI.COMM_SELF))

    for geo in geometries:
        names = geo.domain.interior_names
        for d in (geo.mappings, geo.ncells, geo.periodic):
            assert isinstance(d, dict)
            assert list(d) == names

#==============================================================================
@pytest.mark.xdist_group('h5py')
def test_geometry_legacy_keys_deprecated():
    # WP15-1: legacy keys on `mappings`/`periodic` still resolve, with a
    # DeprecationWarning, and without corrupting the dict's canonical view.
    filename = os.path.join(base_dir, '..', 'mesh', 'multipatch', 'square.h5')
    geo = Geometry.from_file(filename)
    mappings = geo.mappings
    canonical = list(mappings)[0]

    with pytest.warns(DeprecationWarning, match='patch_0'):
        assert mappings['patch_0'] is mappings[canonical]
    with pytest.warns(DeprecationWarning, match='patch_0'):
        assert mappings.get('patch_0') is mappings[canonical]
    with pytest.warns(DeprecationWarning, match='patch_0'):
        assert 'patch_0' in mappings

    periodic_canonical = list(geo.periodic)[1]
    with pytest.warns(DeprecationWarning, match='1'):
        assert geo.periodic[1] == geo.periodic[periodic_canonical]

    # no legacy key leaks into iteration / len / equality
    assert 'patch_0' not in list(mappings)
    assert len(mappings) == len(geo.domain.interior_names)
    assert mappings == dict(mappings.items())

    # an unknown key raises KeyError, with no warning at all
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(KeyError):
            mappings['does_not_exist']

    # pickle round trip, including the aliases -- use `periodic` (plain
    # lists of bool), since `mappings`' values (spline FEM objects) are not
    # picklable for reasons unrelated to `_PatchKeyedDict`.
    periodic2 = pickle.loads(pickle.dumps(geo.periodic))
    assert dict(periodic2) == dict(geo.periodic)
    assert periodic2.aliases == geo.periodic.aliases
    with pytest.warns(DeprecationWarning, match='1'):
        assert periodic2[1] == periodic2[periodic_canonical]

#==============================================================================
@pytest.mark.xdist_group('h5py')
def test_geometry_internal_paths_do_not_use_legacy_keys():
    # WP15-1: every in-tree Geometry consumer migrated to canonical keys in
    # the same diff, so psydac's own internal calls never trip the
    # DeprecationWarning -- on either on-disk naming convention.
    from sympde.topology import ScalarFunctionSpace
    from psydac.api.discretization import discretize

    filenames = [os.path.join(base_dir, '..', 'mesh', 'multipatch', 'square.h5')]
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'g.h5')
        mapping = discrete_mapping('identity', ncells=[4, 4], degree=[2, 2])
        Geometry.from_discrete_mapping(mapping).export(f)
        filenames.append(f)

        for filename in filenames:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter('always')
                discretize(Domain.from_file(filename), filename=filename)
                geo = Geometry.from_file(filename)
                V = ScalarFunctionSpace('V', geo.domain)
                discretize(V, geo, degree=[2, 2])

            legacy_warnings = [w for w in record
                                if issubclass(w.category, DeprecationWarning)
                                and 'Geometry legacy key' in str(w.message)]
            assert not legacy_warnings

#==============================================================================
def _eleven_patch_geometry_file(directory):
    """
    Write an 11-patch geometry file whose on-disk patch order and interior
    order deliberately disagree, and return its path.

    sympde's `Union` sorts interiors lexicographically, so `patch_10` sorts
    between `patch_1` and `patch_2` while the file lists patches in numeric
    order. Ten patches or fewer would not expose this.

    Every patch is made individually identifiable in two independent ways, so
    that a mis-pairing cannot hide behind uniform data:

    - geometrically -- patch `i` is the unit square translated by `i` along
      x, so evaluating it says which patch it really is (this catches a
      mis-bound *mapping*);
    - by discretization -- patch `i` carries `i + 1` elements along x, so its
      `ncells` is `[i + 1, 1]` (this catches a mis-bound *ncells*/*periodic*,
      which uniform `[1, 1]` patches would not).
    """
    from igakit.cad import bilinear
    from psydac.cad.multipatch import export_multipatch_nurbs_to_hdf5

    nurbs = []
    for i in range(11):
        nrb = bilinear(np.array([[[i, 0.], [i, 1.]], [[i + 1., 0.], [i + 1., 1.]]]))
        if i:
            nrb.refine(0, [j / (i + 1) for j in range(1, i + 1)])
        nurbs.append(nrb)
    filename = os.path.join(directory, 'eleven_patches.h5')
    export_multipatch_nurbs_to_hdf5(filename, nurbs, {})
    return filename

@pytest.mark.xdist_group('h5py')
def test_geometry_read_pairs_patches_by_name_not_position():
    # Regression: read() used to pair the i-th on-disk patch with the i-th
    # interior positionally. From 11 patches on the two orders diverge, which
    # bound 9 of 11 splines (and their ncells/periodic/on-disk names) to the
    # wrong patch. Pairing is by name now.
    with tempfile.TemporaryDirectory() as d:
        filename = _eleven_patch_geometry_file(d)
        geo = Geometry.from_file(filename)

        # the orders really do disagree, else this test proves nothing
        yml_names = [p['name'] for p in
                     yaml.safe_load(h5py.File(filename, 'r')['geometry.yml'][()])['patches']]
        assert yml_names != [itr.logical_domain.name for itr in geo.domain.interior.args]

        for key, mapping in geo.mappings.items():
            i = int(key.split('patch_')[1].rstrip(')'))
            assert np.isclose(mapping(0.5, 0.5)[0], i + 0.5), \
                f'{key} is bound to the wrong patch'

        # ncells/periodic follow the same pairing. This is the half that a
        # uniform fixture cannot test: pre-WP15 `ncells`/`periodic` were
        # already keyed by interiors[i].name while taking their values from
        # on-disk patch i, so from 11 patches on they were mis-bound too.
        # (checked in its own loop, so that this -- the substantive pin --
        # is what fails on a regression, rather than the `periodic` lookup
        # below tripping first on unrelated grounds.)
        for itr in geo.domain.interior.args:
            i = int(itr.name.split('patch_')[1].rstrip(')'))
            assert geo.ncells[itr.name] == [i + 1, 1], \
                f'{itr.name} carries another patch\'s ncells'

        for itr in geo.domain.interior.args:
            assert geo.periodic[itr.name] == [False, False]

        # ... and all three dicts are in canonical interior order, not
        # on-disk order, so zipping their .values() pairwise stays correct.
        names = geo.domain.interior_names
        assert list(geo.mappings) == names
        assert list(geo.ncells)   == names
        assert list(geo.periodic) == names

def test_patch_keyed_dict_contains_agrees_with_getitem():
    # An alias may point at a canonical key the dict has no entry for:
    # read() builds `mappings`' aliases from every on-disk patch name, but
    # only spline/NURBS patches get a mapping. `k in d` must not claim a key
    # that `d[k]` would raise on.
    from psydac.cad.geometry import _PatchKeyedDict

    d = _PatchKeyedDict({'Omega': 1}, aliases={'patch_0': 'Omega',
                                               'patch_9': 'missing'})
    with pytest.warns(DeprecationWarning):
        assert 'patch_0' in d
    with pytest.warns(DeprecationWarning):
        assert d['patch_0'] == 1

    with pytest.warns(DeprecationWarning):
        present = 'patch_9' in d
    assert not present
    with pytest.warns(DeprecationWarning):
        with pytest.raises(KeyError):
            d['patch_9']

    # canonical and unknown keys are unaffected and never warn
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert 'Omega' in d
        assert 'nope' not in d

#==============================================================================
@pytest.mark.xdist_group('h5py')
def test_geometry_eleven_patch_export_is_byte_identical():
    # export() must reproduce the on-disk patch order, not the (different)
    # interior order that `mappings` is keyed in.
    with tempfile.TemporaryDirectory() as d:
        filename = _eleven_patch_geometry_file(d)
        geo = Geometry.from_file(filename)
        out = os.path.join(d, 'out.h5')
        geo.export(out)
        with h5py.File(filename, 'r') as h5_orig, h5py.File(out, 'r') as h5_new:
            assert h5_orig['geometry.yml'][()] == h5_new['geometry.yml'][()]
            assert h5_orig['topology.yml'][()] == h5_new['topology.yml'][()]

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
        'geo_wp10_single.h5',
        'geo_wp11_tag.h5',
        'geo_wp11_tag_nurbs.h5',
    ]
    for fname in filenames:
        if os.path.exists(fname):
            os.remove(fname)

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
