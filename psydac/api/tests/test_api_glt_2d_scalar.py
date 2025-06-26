# -*- coding: UTF-8 -*-

import pytest
import numpy as np
from sympy import pi, sin
from scipy.linalg import eig as eig_solver

from sympde.core import Constant
from sympde.calculus import grad, inner
from sympde.topology import dx1, dx2
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import Square
from sympde.expr import BilinearForm, integral

from gelato.expr import GltExpr

from psydac.fem.basic   import FemField
from psydac.api.discretization import discretize

#==============================================================================
def run_poisson_2d_dir(ncells, degree, comm=None):

    # ... abstract model
    domain = Square()

    V = ScalarFunctionSpace('V', domain)

    u, v = elements_of(V, names='u, v')

    int_0 = lambda expr: integral(domain , expr)

    a = BilinearForm((u, v), int_0(inner(grad(u), grad(v))))

    glt_a = GltExpr(a)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
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
def run_field_2d_dir(ncells, degree, comm=None):

    # ... abstract model
    domain = Square()

    V = ScalarFunctionSpace('V', domain)

    F, u, v = elements_of(V, name='F, u, v')

    int_0 = lambda expr: integral(domain , expr)

    a  = BilinearForm((u, v), int_0(inner(grad(u), grad(v)) + F * u * v))
    ae = BilinearForm((u, v), int_0(inner(grad(u), grad(v)) + u * v))

    glt_a  = GltExpr(a)
    glt_ae = GltExpr(ae)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    ah  = discretize(a, domain_h, [Vh, Vh])
    aeh = discretize(ae, domain_h, [Vh, Vh])
    # ...

    # ...
    x = Vh.coeff_space.zeros()
    x[:] = 1.

    phi = FemField( Vh, x )
    # ...

    # ... discretize the glt symbol
    glt_ah  = discretize(glt_a, domain_h, [Vh, Vh])
    glt_aeh = discretize(glt_ae, domain_h, [Vh, Vh])
    # ...

    # ...
    eigh_a = glt_ah.eig(F=phi)
    eigh_a = eigh_a.ravel()
    eigh_a.sort()

    eigh_ae = glt_aeh.eig()
    eigh_ae = eigh_ae.ravel()
    eigh_ae.sort()
    # ...

    # ...
    error = np.linalg.norm(eigh_a-eigh_ae) / Vh.nbasis
    # ...

    return error

#==============================================================================
def run_variable_coeff_2d_dir(ncells, degree, comm=None):

    # ... abstract model
    domain = Square()
    x,y = domain.coordinates

    V = ScalarFunctionSpace('V', domain)

    u, v = elements_of(V, name='u, v')

    c = Constant('c', real=True)

    int_0 = lambda expr: integral(domain , expr)

    expr = (1 + c*sin(pi*(x+y)))*dx1(u)*dx2(v) + (1 + c*sin(pi*(x-y)))*dx1(u)*dx2(v)
    a = BilinearForm((u, v), int_0(expr))
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
    glt_ah = discretize(glt_a, domain_h, [Vh, Vh])
    # ...

    # ...
    eigh = glt_ah.eig(c=0.2)
    eigh = eigh.ravel()
    eigh.sort()
    # ...

    # ... use eigenvalue solver
    M = ah.assemble(c=0.2).toarray()
    w, v = eig_solver(M)
    eig = w.real
    eig.sort()
    # ...

    # ...
    error = np.linalg.norm(eig-eigh) / Vh.nbasis
    # ...

    return error

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
@pytest.mark.xfail
def test_api_glt_poisson_2d_dir_1():

    error = run_poisson_2d_dir(ncells=[2**3,2**3], degree=[2,2])
    assert(np.allclose([error], [0.029738578422276972]))

#==============================================================================
@pytest.mark.xfail
def test_api_glt_field_2d_dir_1():

    error = run_field_2d_dir(ncells=[2**3,2**3], degree=[2,2])
    assert(np.allclose([error], [9.739541824956656e-16]))

#==============================================================================
@pytest.mark.xfail
def test_api_glt_variable_coeff_2d_dir_1():

    error = run_variable_coeff_2d_dir(ncells=[2**3,2**3], degree=[2,2])
    assert(np.allclose([error], [0.015007922966035904]))


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
