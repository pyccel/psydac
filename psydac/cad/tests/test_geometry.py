#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os

import pytest
import numpy as np
from mpi4py import MPI

from sympde.topology import Domain, Line, Square, Cube, Mapping

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
def test_geometry_2d_1():

    ncells = [1,1]
    degree = [2,2]
    # create an identity mapping
    mapping = discrete_mapping('identity', ncells=ncells, degree=degree)

    # create a topological domain
    F      = Mapping('F', dim=2)
    domain = F(Square(name='Omega'))

    # associate the mapping to the topological domain
    mappings = {domain.name: mapping}

    # Define ncells as a dict
    ncells = {domain.name:ncells}

    # create a geometry from a topological domain and the dict of mappings
    geo = Geometry(domain=domain, ncells=ncells, mappings=mappings)

    # export the geometry
    geo.export('geo.h5')

    # read it again
    geo_0 = Geometry(filename='geo.h5')

    # export it again
    geo_0.export('geo_0.h5')

    # create a geometry from a discrete mapping
    geo_1 = Geometry.from_discrete_mapping(mapping)

    # export it
    geo_1.export('geo_1.h5')

#==============================================================================
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
    F      = Mapping('F', dim=2)
    domain = F(Square(name='Omega'))

    # associate the mapping to the topological domain
    mappings = {domain.name: mapping}

    # Define ncells as a dict
    ncells = {domain.name:[len(space.breaks)-1 for space in mapping.space.spaces]}

    periodic = {domain.name:[space.periodic for space in mapping.space.spaces]}

    # create a geometry from a topological domain and the dict of mappings
    geo = Geometry(domain=domain, ncells=ncells, periodic=periodic, mappings=mappings)

    # export the geometry
    geo.export('quart_circle.h5')

    # read it again
    geo_0 = Geometry(filename='quart_circle.h5')

    # export it again
    geo_0.export('quart_circle_0.h5')

    # create a geometry from a discrete mapping
    geo_1 = Geometry.from_discrete_mapping(mapping)

    # export it
    geo_1.export('quart_circle_1.h5')

#==============================================================================
# TODO to be removed
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
    F = Mapping('F', dim=3)
    domain = F(Cube(name='Omega'))

    # associate the mapping to the topological domain
    mappings = {domain.name: mapping}

    # Define d_ncells as a dict
    d_ncells = {domain.name: ncells}

    # Create a geometry from a topological domain and the dict of mappings
    # Here we allow for any distribution of the domain: mpi_dims_mask is not passed
    geo = Geometry(domain=domain, ncells=d_ncells, mappings=mappings, comm=comm)
    geo.export('geo_mpi_dims.h5')

    # Read geometry file in parallel, but using mpi_dims_mask
    geo_from_file = Geometry(filename='geo_mpi_dims.h5', comm=comm, mpi_dims_mask=mpi_dims_mask)

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
    mpi_dims_mask = [False, False, True]  # We swill verify that this has an effect
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

#==============================================================================
@pytest.mark.parametrize( 'ncells', [[8,8], [12,12], [14,14]] )
@pytest.mark.parametrize( 'degree', [[2,2], [3,2], [2,3], [3,3], [4,4]] )
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
    geo = Geometry(filename=filename)
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
def test_import_geopdes_to_nurbs(ncells, degree):


    filename = os.path.join(base_dir, "geo_Lshaped_C1.txt")
    L_shaped = import_geopdes_to_nurbs(filename)

    # refine the nurbs object
    L_shaped = refine_nurbs(L_shaped, ncells=ncells, degree=degree)

    filename = "L_shaped.h5"
    export_nurbs_to_hdf5(filename, L_shaped)

   # read the geometry
    geo = Geometry(filename=filename)
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
