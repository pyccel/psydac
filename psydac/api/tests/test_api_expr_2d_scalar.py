# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy.utilities.lambdify import implemented_function

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of

from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from gelato.expr import GltExpr

from psydac.fem.basic   import FemField
from psydac.fem.vector  import VectorFemField
from psydac.api.discretization import discretize

import numpy as np
from scipy.linalg import eig as eig_solver
from mpi4py import MPI
import pytest


#==============================================================================
def run_poisson_2d_dir(ncells, degree, comm=None):

    # ... abstract model
    domain = Square()

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
    
    Fh = VectorFemField(Vh) 
   
    x = np.linspace(0., 1., 101)
    y = np.linspace(0., 1., 101)
    
    z = exprh(x, y, F=Fh)


###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_glt_poisson_2d_dir_1():
    run_poisson_2d_dir(ncells=[2**3,2**3], degree=[2,2])
