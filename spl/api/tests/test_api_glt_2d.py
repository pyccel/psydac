# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy.utilities.lambdify import implemented_function

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import ScalarField, VectorField
from sympde.topology import ProductSpace
from sympde.topology import ScalarTestFunction
from sympde.topology import VectorTestFunction
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from gelato.expr import GltExpr

from spl.fem.basic   import FemField
from spl.api.discretization import discretize

from numpy import linspace, zeros, allclose
from mpi4py import MPI
import pytest


#==============================================================================
def run_poisson_2d_dir(ncells, degree, comm=None):

    # ... abstract model
    domain = Square()

    V = FunctionSpace('V', domain)

    F = ScalarField(V, name='F')

    v = ScalarTestFunction(V, name='v')
    u = ScalarTestFunction(V, name='u')

    a = BilinearForm((v,u), dot(grad(v), grad(u)))

    glt_a = GltExpr(a)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    ah = discretize(a, domain_h, [Vh, Vh])
    # ...

    # ... dsicretize the glt symbol
#    from spl.api.settings import SPL_BACKEND_GPYCCEL as backend
#    glt_ah = discretize(glt_a, domain_h, [Vh, Vh], backend=backend)
    glt_ah = discretize(glt_a, domain_h, [Vh, Vh])
    x = glt_ah.evaluate([0.5], [0.2])
    assert(x[0,0] ==  0.2697044243683245)
    # ...


###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_glt_poisson_2d_dir_1():

    run_poisson_2d_dir(ncells=[2**3,2**3], degree=[2,2])


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()

test_api_glt_poisson_2d_dir_1()
