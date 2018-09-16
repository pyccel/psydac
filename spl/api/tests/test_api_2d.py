# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S

from sympde.core import dx, dy, dz
from sympde.core import Mapping
from sympde.core import Constant
from sympde.core import Field
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, Integral
from sympde.core import Norm
from sympde.core import Equation, DirichletBC
from sympde.core import Domain
from sympde.core import Boundary, trace_0, trace_1
from sympde.core import ComplementBoundary
from sympde.gallery import Poisson, Stokes

from spl.fem.basic   import FemField
from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.api.discretization import discretize
from spl.api.boundary_condition import DiscreteBoundary
from spl.api.boundary_condition import DiscreteComplementBoundary
from spl.api.boundary_condition import DiscreteDirichletBC

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose
from utils import assert_identical_coo

DEBUG = False

domain = Domain('\Omega', dim=2)

def create_discrete_space():
    # ... discrete spaces
    # Input data: degree, number of elements
#    p1  = 1 ; p2  = 1
    p1  = 2 ; p2  = 2
#    ne1 = 1 ; ne2 = 1
#    ne1 = 4 ; ne2 = 4
    ne1 = 8 ; ne2 = 8
#    ne1 = 32 ; ne2 = 32

    # Create uniform grid
    grid_1 = linspace( 0., 1., num=ne1+1 )
    grid_2 = linspace( 0., 1., num=ne2+1 )

    # Create 1D finite element spaces and precompute quadrature data
    V1 = SplineSpace( p1, grid=grid_1 ); V1.init_fem()
    V2 = SplineSpace( p2, grid=grid_2 ); V2.init_fem()

    # Create 2D tensor product finite element space
    V = TensorFemSpace( V1, V2 )
    # ...

    return V


def test_api_bilinear_2d_scalar_1():
    print('============ test_api_bilinear_2d_scalar_1 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))

    a = BilinearForm((v,u), expr)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    ah = discretize(a, [Vh, Vh])
    M = ah.assemble()
    # ...

def test_api_bilinear_2d_scalar_2():
    print('============ test_api_bilinear_2d_scalar_2 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    c = Constant('c', real=True, label='mass stabilization')

    expr = dot(grad(v), grad(u)) + c*v*u

    a = BilinearForm((v,u), expr)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    ah = discretize(a, [Vh, Vh])
    M = ah.assemble(c=0.5)
    # ...

def test_api_bilinear_2d_scalar_3():
    print('============ test_api_bilinear_2d_scalar_3 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)

    expr = dot(grad(v), grad(u)) + F*v*u

    a = BilinearForm((v,u), expr)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    ah = discretize(a, [Vh, Vh])

    # Define a field
    phi = FemField( Vh, 'phi' )
    phi._coeffs[:,:] = 1.

    M = ah.assemble(F=phi)
    # ...

def test_api_bilinear_2d_scalar_4():
    print('============ test_api_bilinear_2d_scalar_4 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)
    G = Field('G', space=V)

    expr = dot(grad(G*v), grad(u)) + F*v*u

    a = BilinearForm((v,u), expr)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    ah = discretize(a, [Vh, Vh])

    # Define a field
    phi = FemField( Vh, 'phi' )
    phi._coeffs[:,:] = 1.

    psi = FemField( Vh, 'psi' )
    psi._coeffs[:,:] = 1.

    M = ah.assemble(F=phi, G=psi)
    # ...

def test_api_bilinear_2d_scalar_5():
    print('============ test_api_bilinear_2d_scalar_5 =============')

    # ... abstract model
    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    alpha = Constant('alpha')

    expr = dot(grad(v), grad(u)) + alpha*v*u
    a_0 = BilinearForm((v,u), expr, name='a_0')

    expr = v*trace_1(grad(u), B1)
    a_B1 = BilinearForm((v, u), expr, name='a_B1')

    expr = v*trace_0(u, B2)
    a_B2 = BilinearForm((v, u), expr, name='a_B2')

    expr = a_0(v,u) + a_B1(v,u) + a_B2(v,u)
    a = BilinearForm((v,u), expr, name='a')
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    # ...
    ah_0 = discretize(a_0, [Vh, Vh])

    ah_B1 = discretize(a_B1, [Vh, Vh], boundary=B1)
    ah_B2 = discretize(a_B2, [Vh, Vh], boundary=B2)

    M_0 = ah_0.assemble(alpha=0.5)
    M_B1 = ah_B1.assemble()
    M_B2 = ah_B2.assemble()

    M_expected = M_0.tocoo() + M_B1.tocoo() + M_B2.tocoo()
    # ...

    # ...
    ah = discretize(a, [Vh, Vh], boundary=[B1, B2])
    M = ah.assemble(alpha=0.5)
    # ...

    # ...
    assert_identical_coo(M.tocoo(), M_expected)
    # ...


def test_api_bilinear_2d_scalar_1_mapping():
    print('============ test_api_bilinear_2d_scalar_1_mapping =============')

    # ... abstract model
    M = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))

    a = BilinearForm((v,u), expr, mapping=M)
    # ...

    # ...
    from caid.cad_geometry import square
    geo = square(n=[3,3], p=[2,2])
    mapping = SplineMapping.from_caid( geo )
    V = mapping.space ; V.init_fem()
    # ...

    # ...
    ah = discretize(a, [V, V], mapping)
    M = ah.assemble()
#    # ...

def test_api_bilinear_2d_block_1():
    print('============ test_api_bilinear_2d_block_1 =============')

    # ... abstract model
    U = FunctionSpace('U', domain, is_block=True, shape=2)
    V = FunctionSpace('V', domain, is_block=True, shape=2)

    v = VectorTestFunction(V, name='v')
    u = VectorTestFunction(U, name='u')

    expr = div(v) * div(u) + rot(v) * rot(u)

    a = BilinearForm((v,u), expr)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    ah = discretize(a, [Vh, Vh])
    M = ah.assemble()
    # ...

def test_api_linear_2d_scalar_1():
    print('============ test_api_linear_2d_scalar_1 =============')

    # ... abstract model
    V = FunctionSpace('V', domain)

    v = TestFunction(V, name='v')

    x,y = V.coordinates

    expr = cos(2*pi*x)*cos(4*pi*y)*v

    a = LinearForm(v, expr)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    ah = discretize(a, Vh)
    rhs = ah.assemble()
    # ...

def test_api_linear_2d_scalar_2():
    print('============ test_api_linear_2d_scalar_2 =============')

    # ... abstract model
    V = FunctionSpace('V', domain)

    v = TestFunction(V, name='v')

    expr = v + dx(v) + dy(v)

    a = LinearForm(v, expr)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    ah = discretize(a, Vh)
    rhs = ah.assemble()
    # ...

def test_api_function_2d_scalar_1():
    print('============ test_api_function_2d_scalar_1 =============')

    # ... abstract model
    V = FunctionSpace('V', domain)
    x,y = V.coordinates

    # TODO bug: when expr = 1, there are no free_symbols
    expr = S.One

    a = Integral(expr, domain, coordinates=[x,y])
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    ah = discretize(a, Vh)
    integral = ah.assemble()
    assert(allclose(integral, 1))
    # ...

def test_api_equation_2d_1():
    print('============ test_api_equation_2d_1 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)

    x,y = V.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = 2*pi**2*sin(pi*x)*sin(pi*y)*v
    l = LinearForm(v, expr)

    error = F-sin(pi*x)*sin(pi*y)
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(i) for i in [B1, B2, B3, B4]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(i) for i in [B1, B2, B3, B4]]
    equation_h = discretize(equation, [Vh, Vh], bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh)
    h1norm_h = discretize(h1norm, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    error = l2norm_h.assemble(F=phi)
    print('> L2 norm      = ', error)

    error = h1norm_h.assemble(F=phi)
    print('> H1 seminorm  = ', error)
    # ...

def test_api_equation_2d_2():
    print('============ test_api_equation_2d_2 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x,y = V.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    solution = sin(0.5*pi*x)*sin(pi*y)

    expr = (5./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = l0(v) + l_B2(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(-B2)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    bc = [DiscreteDirichletBC(-B2)]
    equation_h = discretize(equation, [Vh, Vh], boundary=B2, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh)
    h1norm_h = discretize(h1norm, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    error = l2norm_h.assemble(F=phi)
    print('> L2 norm      = ', error)

    error = h1norm_h.assemble(F=phi)
    print('> H1 seminorm  = ', error)
    # ...

def test_api_equation_2d_3():
    print('============ test_api_equation_2d_3 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B2
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x,y = V.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    solution = sin(0.5*pi*x)*sin(0.5*pi*y)

    expr = (1./2.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(-(B1+B2))]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=1, ext= 1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    bc = [DiscreteDirichletBC(-(B1+B2))]
    equation_h = discretize(equation, [Vh, Vh], boundary=[B1,B2], bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh)
    h1norm_h = discretize(h1norm, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    error = l2norm_h.assemble(F=phi)
    print('> L2 norm      = ', error)

    error = h1norm_h.assemble(F=phi)
    print('> H1 seminorm  = ', error)
    # ...



def test_api_model_2d_poisson():
    print('============ test_api_model_2d_poisson =============')

    # ... abstract model
    model = Poisson(domain)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    model_h = discretize(model, [Vh, Vh])
    ah = model_h.forms['a']
    M = ah.assemble()
    # ...

def test_api_model_2d_stokes():
    print('============ test_api_model_2d_stokes =============')

    # ... abstract model
    model = Stokes(domain)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    model_h = discretize(model, [Vh, Vh])
    ah = model_h.forms['a']
    bh = model_h.forms['b']
    M1 = ah.assemble()
    M2 = bh.assemble()

#    # we can assemble the full model either by calling directly the discrete
#    # bilinear form
#    Ah = model_h.forms['A']
#    M = Ah.assemble()
#
#    # or through the equation attribut, which is independent from the model
#    lhs_h = model_h.equation.lhs
#    M = lhs_h.assemble()
#    # ...

###############################################
if __name__ == '__main__':

    # ...
#    test_api_bilinear_2d_scalar_1()
#    test_api_bilinear_2d_scalar_2()
#    test_api_bilinear_2d_scalar_3()
#    test_api_bilinear_2d_scalar_4()
#    test_api_bilinear_2d_scalar_5()
#    test_api_bilinear_2d_block_1()

#    test_api_bilinear_2d_scalar_1_mapping()
#
#    test_api_linear_2d_scalar_1()
#    test_api_linear_2d_scalar_2()
#
#    test_api_function_2d_scalar_1()
#    # ...

#    # ...
#    test_api_model_2d_poisson()
#    test_api_model_2d_stokes()
#    # ...

    test_api_equation_2d_1()
    test_api_equation_2d_2()
    test_api_equation_2d_3()
