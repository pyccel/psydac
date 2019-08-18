# -*- coding: UTF-8 -*-

from sympy import Tuple, Matrix
from sympy import pi, cos, sin

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
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from gelato.expr import GltExpr

from psydac.fem.vector  import VectorFemField
from psydac.api.discretization import discretize

import numpy as np
from scipy.linalg import eig as eig_solver
from mpi4py import MPI
import pytest

#==============================================================================
def run_vector_poisson_2d_dir(ncells, degree):

    # ... abstract model
    domain = Square()

    V = VectorFunctionSpace('V', domain)

    x,y = domain.coordinates

    F = element_of(V, name='F')

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)
    
    a = BilinearForm((v,u), int_0(inner(grad(v), grad(u))))

    glt_a = GltExpr(a)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the bilinear form
    ah = discretize(a, domain_h, [Vh, Vh])
    # ...

    # ... dsicretize the glt symbol
    glt_ah = discretize(glt_a, domain_h, [Vh, Vh])
    # ...

    # ...
    eigh = glt_ah.eig()
    eigh = eigh.ravel()
    eigh.sort()
    # ...

    # ... use eigenvalue solver
    M = ah.assemble().tosparse().todense()
    w, v = eig_solver(M)
    eig = w.real
    eig.sort()
    # ...

    # ...
    error = np.linalg.norm(eig-eigh) / Vh.nbasis
    # ...

    return error

#==============================================================================
def test_api_glt_vector_poisson_2d_dir_1():

    error = run_vector_poisson_2d_dir(ncells=[2**3,2**3], degree=[2,2])
    assert(np.allclose([error], [0.021028350465240004]))



#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
