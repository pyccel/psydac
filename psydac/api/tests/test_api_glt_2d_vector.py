# -*- coding: UTF-8 -*-

import pytest
import numpy as np
from scipy.linalg import eig as eig_solver

from sympde.calculus import grad, inner
from sympde.topology import VectorFunctionSpace
from sympde.topology import elements_of
from sympde.topology import Square
from sympde.expr     import BilinearForm, integral

from gelato.expr import GltExpr

from psydac.api.discretization import discretize

#==============================================================================
def run_vector_poisson_2d_dir(ncells, degree):

    # ... abstract model
    domain = Square()

    V = VectorFunctionSpace('V', domain)

    u, v = elements_of(V, names='u, v')

    int_0 = lambda expr: integral(domain , expr)

    a = BilinearForm((u, v), int_0(inner(grad(u), grad(v))))

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
    M = ah.assemble().toarray()
    w, v = eig_solver(M)
    eig = w.real
    eig.sort()
    # ...

    # ...
    error = np.linalg.norm(eig-eigh) / Vh.nbasis
    # ...

    return error

#==============================================================================
@pytest.mark.xfail
def test_api_glt_vector_poisson_2d_dir_1():

    error = run_vector_poisson_2d_dir(ncells=[2**3,2**3], degree=[2,2])
    assert(np.allclose([error], [0.021028350465240004]))



#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
