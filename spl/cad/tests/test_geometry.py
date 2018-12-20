# coding: utf-8
#

from sympde.topology import Domain, Line, Square, Cube

from spl.cad.geometry             import Geometry
from spl.mapping.discrete_gallery import discrete_mapping

#==============================================================================
def test_geometry_2d():

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
    newgeo = Geometry(filename='geo.h5')

    # export it again
    newgeo.export('newgeo.h5')

    # create a geometry from a discrete mapping
    geo_1 = Geometry.from_discrete_mapping(mapping)

    # export it
    geo_1.export('geo_1.h5')


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

