# coding: utf-8
#

import numpy as np

from sympde.topology import Domain, Line, Square, Cube

from spl.cad.geometry             import Geometry
from spl.cad.cad                  import elevate, refine
from spl.cad.gallery              import quart_circle, circle
from spl.mapping.discrete         import SplineMapping, NurbsMapping
from spl.mapping.discrete_gallery import discrete_mapping
from spl.fem.splines              import SplineSpace
from spl.fem.tensor               import TensorFemSpace

#==============================================================================
def test_geometry_2d_1():

    # create an identity mapping
    mapping = discrete_mapping('identity', ncells=[1,1], degree=[2,2])

    # create a topological domain
    domain = Square(name='Omega')

    # associate the mapping to the topological domain
    mappings = {'Omega': mapping}

    # create a geometry from a topological domain and the dict of mappings
    geo = Geometry(domain=domain, mappings=mappings)

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
    space = TensorFemSpace( *spaces, comm=None )

    mapping = NurbsMapping.from_control_points_weights( space, points, weights )

    mapping = elevate( mapping, axis=0, times=1 )
    mapping = refine( mapping, axis=0, values=[0.3, 0.6, 0.8] )

    # create a topological domain
    domain = Square(name='Omega')

    # associate the mapping to the topological domain
    mappings = {'Omega': mapping}

    # create a geometry from a topological domain and the dict of mappings
    geo = Geometry(domain=domain, mappings=mappings)

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
    space = TensorFemSpace( *spaces, comm=None )

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
    space = TensorFemSpace( *spaces, comm=None )

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
def test_geometry_1():

    line   = Geometry.as_line(ncells=[10])
    square = Geometry.as_square(ncells=[10, 10])
    cube   = Geometry.as_cube(ncells=[10, 10, 10])

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
