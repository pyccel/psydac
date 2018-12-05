# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import Tuple
from sympy import Matrix

from sympde.core import dx, dy, dz
from sympde.core import Constant
from sympde.core import Field
from sympde.core import VectorField
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace, VectorFunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm
from sympde.core import Norm
from sympde.core import Equation, DirichletBC
from sympde.core import Domain
from sympde.core import Boundary, trace_0, trace_1
from sympde.core import ComplementBoundary

from spl.fem.context import fem_context
from spl.fem.basic   import FemField
from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.fem.vector  import ProductFemSpace, VectorFemField
from spl.api.discretization import discretize
from spl.api.boundary_condition import DiscreteBoundary
from spl.api.boundary_condition import DiscreteComplementBoundary
from spl.api.boundary_condition import DiscreteDirichletBC

from numpy import linspace, zeros, allclose

domain = Domain('\Omega', dim=2)

def create_discrete_space(p=(2,2), ne=(2**3,2**3)):
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


def test_api_vector_2d_2():

    # ... abstract model
    U = VectorFunctionSpace('U', domain)
    V = VectorFunctionSpace('V', domain)

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
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    Vh = ProductFemSpace(Vh, Vh)
    # ...

    # ...
    ah = discretize(a, [Vh, Vh])
    lh = discretize(l, Vh)
    # ...

    # ... abstract model
    UU1 = FunctionSpace('UU1', domain)
    VV1 = FunctionSpace('VV1', domain)

    vv1 = TestFunction(VV1, name='vv1')
    uu1 = TestFunction(UU1, name='uu1')

    expr = dot(grad(vv1), grad(uu1))
    a1 = BilinearForm((vv1,uu1), expr, name='a1')

    expr = 2*pi**2*sin(pi*x)*sin(pi*y)*vv1
    l1 = LinearForm(vv1, expr, name='l1')
    # ...

    # ... discrete spaces
    Wh = create_discrete_space()
    # ...

    # ...
    a1h = discretize(a1, [Wh, Wh])
    l1h = discretize(l1, Wh)
    # ...

    # ...
    A = ah.assemble()
    L = lh.assemble()

    A1 = a1h.assemble()
    L1 = l1h.assemble()
    # ...

    # ...
    allclose( L1.toarray(), L[0].toarray() )
    allclose( L1.toarray(), L[1].toarray() )

    n = len(L1.toarray())
    assert( allclose( L1.toarray(), L.toarray()[:n] ) )
    assert( allclose( L1.toarray(), L.toarray()[n:] ) )
    # ...


def test_api_vector_laplace_2d_dir_1():

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

    f1 = sin(pi*x)*sin(pi*y)
    f2 = sin(pi*x)*sin(pi*y)
    f = Tuple(f1, f2)
    error = Matrix([F[0]-f[0], F[1]-f[1]])
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(i) for i in [B1, B2, B3, B4]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
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
    x = equation_h.solve(settings={'solver':'cg', 'tol':1e-13, 'maxiter':1000,
                                   'verbose':False})
    # ...

    # ...
    phi = VectorFemField( Vh, 'phi' )
    phi.coeffs[0][:,:] = x[0][:,:]
    phi.coeffs[1][:,:] = x[1][:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0003484905993571711
    expected_h1_error =  0.01848136368981003

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

def test_api_vector_l2_projection_2d_dir_1():

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
    Vh = create_discrete_space()
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
    # ...

    # ...
    phi = VectorFemField( Vh, 'phi' )
    phi.coeffs[0][:,:] = x[0][:,:]
    phi.coeffs[1][:,:] = x[1][:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0003289714098362605
    expected_h1_error =  0.018695563450364158

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()

