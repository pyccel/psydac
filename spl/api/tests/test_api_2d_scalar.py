# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin

from sympde.core import dx, dy, dz
from sympde.core import Constant
from sympde.core import Field
from sympde.core import grad, dot
from sympde.core import FunctionSpace
from sympde.core import TestFunction
from sympde.core import BilinearForm, LinearForm
from sympde.core import Norm
from sympde.core import Equation, DirichletBC
from sympde.core import Domain
from sympde.core import Boundary, trace_0, trace_1

from spl.fem.context import fem_context
from spl.fem.basic   import FemField
from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.api.discretization import discretize
from spl.api.boundary_condition import DiscreteBoundary
from spl.api.boundary_condition import DiscreteComplementBoundary
from spl.api.boundary_condition import DiscreteDirichletBC

from numpy import linspace, allclose

domain = Domain('\Omega', dim=2)


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



#==============================================================================
def test_api_poisson_2d_dir_1():

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

    error = F - sin(pi*x)*sin(pi*y)
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
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0002464200659843996
    expected_h1_error =  0.013068297590632992

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...



#==============================================================================
def test_api_laplace_2d_dir_1():

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
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0002464068884663996
    expected_h1_error =  0.013068297990908681

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...


#==============================================================================
def test_api_poisson_2d_dirneu_1():

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
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0001755319490060421
    expected_h1_error =  0.009298116787699227

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...


#==============================================================================
def test_api_poisson_2d_dirneu_2():

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
    #

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
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  2.7971100793185878e-05
    expected_h1_error =  0.0016032816329282472

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...


#==============================================================================
def test_api_bilinear_2d_sumform_1():

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    alpha = Constant('alpha')

    expr = dot(grad(v), grad(u))
    a_0 = BilinearForm((v,u), expr, name='a_0')

    expr = alpha*v*u
    a_1 = BilinearForm((v,u), expr, name='a_1')

    expr = a_0(v,u) + a_1(v,u)
    a = BilinearForm((v,u), expr, name='a')
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ...
    ah_0 = discretize(a_0, [Vh, Vh])
    ah_1 = discretize(a_1, [Vh, Vh])

    M_0 = ah_0.assemble()
    M_1 = ah_1.assemble(alpha=0.5)

    M_expected = M_0.tocoo() + M_1.tocoo()
    # ...

    # ...
    ah = discretize(a, [Vh, Vh])
    M = ah.assemble(alpha=0.5)
    # ...

    # ...
    assert_identical_coo(M.tocoo(), M_expected)
    # ...


#==============================================================================
def test_api_bilinear_2d_sumform_2():

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
    # ...

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

#==============================================================================
def test_api_poisson_2d_dirneu_3():

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)

    expr = (1./2.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B3(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(-(B1+B3))]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)

    bc = [DiscreteDirichletBC(-(B1+B3))]
    equation_h = discretize(equation, [Vh, Vh], boundary=[B1,B3], bc=bc)
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
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  2.7971100909686694e-05
    expected_h1_error =  0.0016032816329212534

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_4():

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain) # Dirichlet H. bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    solution = cos(0.25*pi*x)*cos(0.5*pi*y)

    expr = (5./16.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(B4)]
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

    bc = [DiscreteDirichletBC(B4)]
    equation_h = discretize(equation, [Vh, Vh], boundary=[B1,B2,B3], bc=bc)
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
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  2.5366751560417237e-05
    expected_h1_error =  0.001452350212346307

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_neu_1():

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u)) + v*u
    a = BilinearForm((v,u), expr)

    solution = cos(0.25*pi*x)*cos(0.25*pi*y)

    expr = ((1./8.)*pi**2 + 1.)*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v) + l_B4(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    equation = Equation(a(v,u), l(v))
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    equation_h = discretize(equation, [Vh, Vh], boundary=[B1,B2,B3,B4])
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
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  2.7510665198168697e-06
    expected_h1_error =  0.00015490443857562876

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

