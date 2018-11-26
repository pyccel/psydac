# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S
from sympy import Tuple
from sympy import Matrix

from sympde.core import dx, dy, dz
from sympde.core import Mapping
from sympde.core import Constant
from sympde.core import Field
from sympde.core import VectorField
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace, VectorFunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, Integral
from sympde.core import Norm
from sympde.core import Equation, DirichletBC
from sympde.core import Domain
from sympde.core import Boundary, trace_0, trace_1
from sympde.core import ComplementBoundary
from sympde.gallery import Poisson, Stokes

from spl.fem.context import fem_context
from spl.fem.basic   import FemField
from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.fem.vector  import ProductFemSpace, VectorFemField
from spl.api.discretization import discretize
from spl.api.boundary_condition import DiscreteBoundary
from spl.api.boundary_condition import DiscreteComplementBoundary
from spl.api.boundary_condition import DiscreteDirichletBC

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose
from utils import assert_identical_coo

DEBUG = False

domain = Domain('\Omega', dim=2)

def create_discrete_space(p=(2,2), ne=(2,2)):
    # ... discrete spaces
    # Input data: degree, number of elements
    p1,p2 = p
    ne1,ne2 = ne

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


def test_api_bilinear_2d_sumform():
    print('============ test_api_bilinear_2d_sumform =============')

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



def test_api_poisson_2d_dir_1():
    print('============ test_api_poisson_2d_dir_1 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = 2*pi**2*sin(pi*x)*sin(pi*y)*v
    l = LinearForm(v, expr)

    error = F -sin(pi*x)*sin(pi*y)
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

def test_api_poisson_2d_dirneu_1():
    print('============ test_api_poisson_2d_dirneu_1 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x,y = domain.coordinates

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

def test_api_poisson_2d_dirneu_2():
    print('============ test_api_poisson_2d_dirneu_2 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B2
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x,y = domain.coordinates

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


def test_api_poisson_2d_dir_1_mapping():
    print('============ test_api_poisson_2d_dir_1_mapping =============')

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    expr = 2*pi**2*sin(pi*x)*sin(pi*y)*v
    l = LinearForm(v, expr, mapping=mapping)

    error = F-sin(pi*x)*sin(pi*y)
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(i) for i in [B1, B2, B3, B4]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context('square.h5')
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(i) for i in [B1, B2, B3, B4]]
    equation_h = discretize(equation, [Vh, Vh], mapping, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
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

def test_api_poisson_2d_dirneu_1_mapping():
    print('============ test_api_poisson_2d_dirneu_1_mapping ============')

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.5*pi*x)*sin(pi*y)

    expr = (5./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B2(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B2)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context('square.h5')
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    bc = [DiscreteDirichletBC(-B2)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B2, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
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

def test_api_laplace_2d_dir_1():
    print('============ test_api_laplace_2d_dir_1 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)

    x,y = domain.coordinates

    F = Field('F', V)
    alpha = Constant('alpha')

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u)) + alpha*v*u
    a = BilinearForm((v,u), expr)

    expr = (2*pi**2 + alpha)*sin(pi*x)*sin(pi*y)*v
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
    x = equation_h.solve(alpha=0.2)
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

def test_api_vector_laplace_2d_dir_1():
    print('============ test_api_vector_laplace_2d_dir_1 =============')

    # ... abstract model
    U = VectorFunctionSpace('U', domain)
    V = VectorFunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)

    x,y = domain.coordinates

    F = VectorField(V, name='F')

    v = VectorTestFunction(V, name='v')
    u = VectorTestFunction(U, name='u')

    expr = inner(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    f1 = 2*pi**2*sin(pi*x)*sin(pi*y)
    f2 = 2*pi**2*sin(pi*x)*sin(pi*y)
    f = Tuple(f1, f2)
    expr = dot(f, v)
    l = LinearForm(v, expr)

    f = Tuple(sin(pi*x)*sin(pi*y), sin(pi*x)*sin(pi*y))
    error = Matrix([F[0]-f[0], F[1]-f[1]])
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(i) for i in [B1, B2, B3, B4]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space(ne=(2**4, 2**4))
    Vh = ProductFemSpace(Vh, Vh)
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
    L = equation_h.linear_system
    # ...

    # ...
    phi = VectorFemField( Vh, 'phi' )
    phi.coeffs[0][:,:] = x[0][:,:]
    phi.coeffs[1][:,:] = x[1][:,:]
    # ...

    # ... compute norms
    error = l2norm_h.assemble(F=phi)
    print('> L2 norm      = ', error)

    # TODO not working yet => check formulae
    error = h1norm_h.assemble(F=phi)
    print('> H1 seminorm  = ', error)
    # ...

def test_api_vector_l2_projection_2d_dir_1():
    print('============ test_api_vector_l2_projection_2d_dir_1 =============')

    # ... abstract model
    U = VectorFunctionSpace('U', domain)
    V = VectorFunctionSpace('V', domain)

    x,y = domain.coordinates

    F = VectorField(V, name='F')

    v = VectorTestFunction(V, name='v')
    u = VectorTestFunction(U, name='u')

    expr = dot(v, u)
    a = BilinearForm((v,u), expr)

    f1 = sin(pi*x)*sin(pi*y)
    f2 = sin(pi*x)*sin(pi*y)
    f = Tuple(f1, f2)
    expr = dot(f, v)
    l = LinearForm(v, expr)

    f = Tuple(sin(pi*x)*sin(pi*y), sin(pi*x)*sin(pi*y))
    error = Matrix([F[0]-f[0], F[1]-f[1]])
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    equation = Equation(a(v,u), l(v))
    # ...

    # ... discrete spaces
    Vh = create_discrete_space(ne=(2**4, 2**4))
    Vh = ProductFemSpace(Vh, Vh)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh)
    h1norm_h = discretize(h1norm, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    L = equation_h.linear_system
    # ...

    # ...
    phi = VectorFemField( Vh, 'phi' )
    phi.coeffs[0][:,:] = x[0][:,:]
    phi.coeffs[1][:,:] = x[1][:,:]
    # ...

    # ... compute norms
    error = l2norm_h.assemble(F=phi)
    print('> L2 norm      = ', error)

    # TODO not working yet => check formulae
    error = h1norm_h.assemble(F=phi)
    print('> H1 seminorm  = ', error)
    # ...

###############################################
if __name__ == '__main__':

#    # ...
#    test_api_bilinear_2d_sumform()
#    test_api_poisson_2d_dirneu_1()
#    test_api_poisson_2d_dirneu_2()
#    # ...
#
    # ...
    test_api_poisson_2d_dir_1()
    test_api_laplace_2d_dir_1()
    test_api_vector_l2_projection_2d_dir_1()
    test_api_vector_laplace_2d_dir_1()
    test_api_poisson_2d_dir_1_mapping()
    # ...

    # ...
##    # TODO this test works when runned alone, but not after the other tests!!!
##    # is it a problem of sympy namespace?
##    test_api_poisson_2d_dirneu_1_mapping()
##    # ...
