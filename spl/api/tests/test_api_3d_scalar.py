# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import Field, VectorField
from sympde.topology import ProductSpace
from sympde.topology import TestFunction
from sympde.topology import VectorTestFunction
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm, Integral
from sympde.expr import Norm
from sympde.expr import Equation, EssentialBC

from spl.fem.basic   import FemField
from spl.api.discretization import discretize

from numpy import linspace, zeros, allclose
from mpi4py import MPI
import pytest

#==============================================================================
def run_poisson_3d_dir(solution, f, ncells, degree, comm=None):

    # ... abstract model
    domain = Cube()

    V = FunctionSpace('V', domain)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(V, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = f*v
    l = LinearForm(v, expr)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh )
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

#==============================================================================
def run_poisson_3d_dirneu(solution, f, boundary, ncells, degree, comm=None):

    assert( isinstance(boundary, (list, tuple)) )

    # ... abstract model
    domain = Cube()

    V = FunctionSpace('V', domain)

    B_neumann = [domain.get_boundary(i) for i in boundary]
    if len(B_neumann) == 1:
        B_neumann = B_neumann[0]

    else:
        B_neumann = Union(*B_neumann)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(V, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = f*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B_neumann)
    l_B_neumann = LinearForm(v, expr)

    expr = l0(v) + l_B_neumann(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    B_dirichlet = domain.boundary.complement(B_neumann)
    bc = EssentialBC(u, 0, B_dirichlet)

    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh )
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error


###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_poisson_3d_dir_1():

    from sympy.abc import x,y,z

    solution = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f        = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

    l2_error, h1_error = run_poisson_3d_dir(solution, f,
                                            ncells=[2**2,2**2,2**2], degree=[2,2,2])

    expected_l2_error =  0.0017546148822053188
    expected_h1_error =  0.048189500102744275

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
#def test_api_poisson_3d_dirneu_1():
#
#    from sympy.abc import x,y,z
#
#    solution = sin(0.5*pi*x)*sin(pi*y)*sin(pi*z)
#    f        = (9./4.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(solution, f, ['Gamma_1'],
#                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])
#
#    expected_l2_error =  0.0014388350122198999
#    expected_h1_error =  0.03929404299152041
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_2():

    from sympy.abc import x,y,z

    solution = sin(0.5*pi*x)*sin(pi*y)*sin(pi*z)
    f        = (9./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(solution, f, ['Gamma_2'],
                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])

    expected_l2_error =  0.0014388350122198999
    expected_h1_error =  0.03929404299152041

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_13():

    from sympy.abc import x,y,z

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)*sin(pi*z)
    f        = (3./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(solution, f, ['Gamma_1', 'Gamma_3'],
                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])

    expected_l2_error =  0.0010275451113309177
    expected_h1_error =  0.027938446826372313

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_24():

    from sympy.abc import x,y,z

    solution = sin(0.5*pi*x)*sin(0.5*pi*y)*sin(pi*z)
    f        = (3./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(solution, f, ['Gamma_2', 'Gamma_4'],
                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])

    expected_l2_error =  0.0010275451113330345
    expected_h1_error =  0.027938446826372445

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_3d_dirneu_123():
#
#    from sympy.abc import x,y,z
#
#    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*sin(pi*z)
#    f        = (21./16.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(solution, f, ['Gamma_1', 'Gamma_2', 'Gamma_3'],
#                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])
#
#    expected_l2_error =  0.0013124098938818625
#    expected_h1_error =  0.035441679549891296
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

##==============================================================================
## TODO DEBUG, not working since merge with devel
#def test_api_poisson_3d_dirneu_1235():
#
#    from sympy.abc import x,y,z
#
#    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*cos(0.5*pi*z)
#    f        = (9./16.)*pi**2*solution
#
#    l2_error, h1_error = run_poisson_3d_dirneu(solution, f, ['Gamma_1', 'Gamma_2', 'Gamma_3', 'Gamma_5'],
#                                               ncells=[2**2,2**2,2**2], degree=[2,2,2])
#
#    expected_l2_error =  0.00019677816055503394
#    expected_h1_error =  0.0058786142515821265
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)

###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
@pytest.mark.parallel
def test_api_poisson_3d_dir_1_parallel():

    from sympy.abc import x,y,z

    solution = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f        = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

    l2_error, h1_error = run_poisson_3d_dir(solution, f,
                                            ncells=[2**2,2**2,2**2], degree=[2,2,2],
                                            comm=MPI.COMM_WORLD)

    expected_l2_error =  0.0017546148822053188
    expected_h1_error =  0.048189500102744275

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
