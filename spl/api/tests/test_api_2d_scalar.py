# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin

from sympde.core import Constant
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import laplace, hessian
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
from sympde.expr import Equation, DirichletBC

from spl.fem.basic   import FemField
from spl.api.discretization import discretize

from numpy import linspace, zeros, allclose


#==============================================================================
def assert_identical_coo(A, B):

    if isinstance(A, (list, tuple)) and isinstance(B, (list, tuple)):
        assert len(A) == len(B)

        for a,b in zip(A, B): assert_identical_coo(a, b)

    elif not(isinstance(A, (list, tuple))) and not(isinstance(B, (list, tuple))):
        A = A.tocoo()
        B = B.tocoo()

        assert(A.shape == B.shape)
        assert(A.nnz == B.nnz)

        assert(allclose(A.row,  B.row))
        assert(allclose(A.col,  B.col))
        assert(allclose(A.data, B.data))

    else:
        raise TypeError('Wrong types for entries')


#==============================================================================
def run_poisson_2d_dir(solution, f, ncells, degree):

    # ... abstract model
    domain = Square()

    V = FunctionSpace('V', domain)

    x,y = domain.coordinates

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

    equation = Equation(a(v,u), l(v), bc=DirichletBC(domain.boundary))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
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
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

#==============================================================================
def run_poisson_2d_dirneu(solution, f, boundary, ncells, degree):

    assert( isinstance(boundary, (list, tuple)) )

    # ... abstract model
    domain = Square()

    V = FunctionSpace('V', domain)

    B_neumann = [domain.get_boundary(i) for i in boundary]
    if len(B_neumann) == 1:
        B_neumann = B_neumann[0]

    else:
        B_neumann = Union(*B_neumann)

    x,y = domain.coordinates

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

    equation = Equation(a(v,u), l(v), bc=DirichletBC(B_dirichlet))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
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
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error

#==============================================================================
def run_laplace_2d_neu(solution, f, ncells, degree):

    # ... abstract model
    domain = Square()

    V = FunctionSpace('V', domain)

    B_neumann = domain.boundary

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(V, name='u')

    expr = dot(grad(v), grad(u)) + v*u
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

    equation = Equation(a(v,u), l(v))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
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
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)
    # ...

    return l2_error, h1_error


#==============================================================================
def test_api_poisson_2d_dir_1():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    l2_error, h1_error = run_poisson_2d_dir(solution, f,
                                            ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.00021808678604760232
    expected_h1_error =  0.013023570720360362

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_1():

    from sympy.abc import x,y

    solution = sin(0.5*pi*(1.-x))*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, ['Gamma_1'],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.00015546057796452772
    expected_h1_error =  0.00926930278452745

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_2():

    from sympy.abc import x,y

    solution = sin(0.5*pi*x)*sin(pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, ['Gamma_2'],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.0001554605779481901
    expected_h1_error =  0.009269302784527256

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_3():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(0.5*pi*(1.-y))
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, ['Gamma_3'],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.0001554605779681901
    expected_h1_error =  0.009269302784528678

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_4():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(0.5*pi*y)
    f        = (5./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f, ['Gamma_4'],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.00015546057796339546
    expected_h1_error =  0.009269302784526841

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_13():

    from sympy.abc import x,y

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)
    f        = (1./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f,
                                               ['Gamma_1', 'Gamma_3'],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  2.6119892736036942e-05
    expected_h1_error =  0.0016032430287934746

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_24():

    from sympy.abc import x,y

    solution = sin(0.5*pi*x)*sin(0.5*pi*y)
    f        = (1./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f,
                                               ['Gamma_2', 'Gamma_4'],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  2.611989253883369e-05
    expected_h1_error =  0.0016032430287973409

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_2d_dirneu_123():

    from sympy.abc import x,y

    solution = cos(pi*x)*cos(0.5*pi*y)
    f        = 5./4.*pi**2*solution

    l2_error, h1_error = run_poisson_2d_dirneu(solution, f,
                                               ['Gamma_1', 'Gamma_2', 'Gamma_3'],
                                               ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.00015494478505412876
    expected_h1_error =  0.009242166414700994

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
def test_api_laplace_2d_neu():

    from sympy.abc import x,y

    solution = cos(pi*x)*cos(pi*y)
    f        = (2.*pi**2 + 1.)*solution

    l2_error, h1_error = run_laplace_2d_neu(solution, f, ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  0.0002172846538950129
    expected_h1_error =  0.012984852988125026

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


##==============================================================================
#def test_api_bilinear_2d_sumform_1():
#
#    # ... abstract model
#    U = FunctionSpace('U', domain)
#    V = FunctionSpace('V', domain)
#
#    v = TestFunction(V, name='v')
#    u = TestFunction(U, name='u')
#
#    alpha = Constant('alpha')
#
#    expr = dot(grad(v), grad(u))
#    a_0 = BilinearForm((v,u), expr, name='a_0')
#
#    expr = alpha*v*u
#    a_1 = BilinearForm((v,u), expr, name='a_1')
#
#    expr = a_0(v,u) + a_1(v,u)
#    a = BilinearForm((v,u), expr, name='a')
#    # ...
#
#    # ... discrete spaces
#    Vh = create_discrete_space()
#    # ...
#
#    # ...
#    ah_0 = discretize(a_0, [Vh, Vh])
#    ah_1 = discretize(a_1, [Vh, Vh])
#
#    M_0 = ah_0.assemble()
#    M_1 = ah_1.assemble(alpha=0.5)
#
#    M_expected = M_0.tocoo() + M_1.tocoo()
#    # ...
#
#    # ...
#    ah = discretize(a, [Vh, Vh])
#    M = ah.assemble(alpha=0.5)
#    # ...
#
#    # ...
#    assert_identical_coo(M.tocoo(), M_expected)
#    # ...
#
#
##==============================================================================
#def test_api_bilinear_2d_sumform_2():
#
#    # ... abstract model
#    B1 = Boundary(r'\Gamma_1', domain)
#    B2 = Boundary(r'\Gamma_2', domain)
#
#    U = FunctionSpace('U', domain)
#    V = FunctionSpace('V', domain)
#
#    v = TestFunction(V, name='v')
#    u = TestFunction(U, name='u')
#
#    alpha = Constant('alpha')
#
#    expr = dot(grad(v), grad(u)) + alpha*v*u
#    a_0 = BilinearForm((v,u), expr, name='a_0')
#
#    expr = v*trace_1(grad(u), B1)
#    a_B1 = BilinearForm((v, u), expr, name='a_B1')
#
#    expr = v*trace_0(u, B2)
#    a_B2 = BilinearForm((v, u), expr, name='a_B2')
#
#    expr = a_0(v,u) + a_B1(v,u) + a_B2(v,u)
#    a = BilinearForm((v,u), expr, name='a')
#    # ...
#
#    # ... discrete spaces
#    Vh = create_discrete_space()
#    # ...
#
#    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
#    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
#
#    # ...
#    ah_0 = discretize(a_0, [Vh, Vh])
#
#    ah_B1 = discretize(a_B1, [Vh, Vh], boundary=B1)
#    ah_B2 = discretize(a_B2, [Vh, Vh], boundary=B2)
#
#    M_0 = ah_0.assemble(alpha=0.5)
#    M_B1 = ah_B1.assemble()
#    M_B2 = ah_B2.assemble()
#
#    M_expected = M_0.tocoo() + M_B1.tocoo() + M_B2.tocoo()
#    # ...
#
#    # ...
#    ah = discretize(a, [Vh, Vh], boundary=[B1, B2])
#    M = ah.assemble(alpha=0.5)
#    # ...
#
#    # ...
#    assert_identical_coo(M.tocoo(), M_expected)
#    # ...


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
