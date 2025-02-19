import numpy as np
import pytest

from psydac.api.discretization  import discretize
from sympde.topology            import ScalarFunctionSpace
from sympde.topology            import Square
from sympde.topology            import CollelaMapping2D
from psydac.mapping.discrete    import SplineMapping
from psydac.feec.pushforward    import Pushforward

def test_basic_call():

    logical_domain = Square('Omega')

    ncells = [2**4, 2**4]
    degree = [2, 2]

    # Mapping and physical domain
    mapping = CollelaMapping2D('M', a=1, b=1, eps=.2)
    domain  = mapping(logical_domain)

    hat_V0 = ScalarFunctionSpace('hat V0', logical_domain)
    domain_h = discretize(logical_domain, ncells = ncells, periodic = [False, True])
    hat_V0_h = discretize(hat_V0, domain_h, degree = degree)
    
    grid_x1 = hat_V0_h.breaks[0]
    grid_x2 = hat_V0_h.breaks[1]

    F = SplineMapping.from_mapping(hat_V0_h, mapping)    
    Pushforward(grid=(grid_x1, grid_x2), mapping=F, grid_type=0)

    F = mapping   
    Pushforward(grid=(grid_x1, grid_x2), mapping=F, grid_type=0)
