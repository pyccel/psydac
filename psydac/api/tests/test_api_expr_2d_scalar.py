# -*- coding: UTF-8 -*-

from mpi4py import MPI
from sympy import pi
import numpy as np

from sympde.calculus import dot, div
from sympde.topology import VectorFunctionSpace
from sympde.topology import element_of
from sympde.topology import Square, IdentityMapping

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

#==============================================================================
def run_poisson_2d_dir(ncells, degree, comm=None):

    # ... abstract model
    domain  = Square()
    mapping = IdentityMapping('M',2)
    domain  = mapping(domain)

    V = VectorFunctionSpace('V', domain)

    F = element_of(V, name='F')
    
    x,y = domain.coordinates
    
    expr = x*y+ pi*div(F) + dot(F, F)

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...
    
    # ... dsicretize the bilinear form
    exprh = discretize(expr, domain_h, Vh)
    
    Fh = FemField(Vh)
   
    x = np.linspace(0., 1., 101)
    y = np.linspace(0., 1., 101)
    
    z = exprh(x, y, F=Fh)


###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_expr_2d_1():
    run_poisson_2d_dir(ncells=[2**3,2**3], degree=[2,2])

