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
from sympde.core import Domain
from sympde.gallery import Poisson, Stokes

from spl.fem.basic   import FemField
from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.api.discretization import discretize

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose

domain = Domain('\Omega', dim=2)

def create_discrete_space():
    # ... discrete spaces
    # Input data: degree, number of elements
    p1  = 2 ; p2  = 2
    ne1 = 4 ; ne2 = 4

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
    M = ah.assemble(0.5)
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

    M = ah.assemble(phi)
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

    M = ah.assemble(phi, psi)
    # ...

def test_api_bilinear_2d_scalar_5():
    print('============ test_api_bilinear_2d_scalar_5 =============')

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

    F = Field('F', space=V)

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

def test_api_model_2d_poisson():
    print('============ test_api_model_2d_poisson =============')

    # ... abstract model
    model = Poisson(domain=domain)
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
    model = Stokes(domain=domain)
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

    # we can assemble the full model either by calling directly the discrete
    # bilinear form
    Ah = model_h.forms['A']
    M = Ah.assemble()

    # or through the equation attribut, which is independent from the model
    lhs_h = model_h.equation.lhs
    M = lhs_h.assemble()
    # ...

###############################################
if __name__ == '__main__':

    # ...
    test_api_bilinear_2d_scalar_1()
    test_api_bilinear_2d_scalar_2()
    test_api_bilinear_2d_scalar_3()
    test_api_bilinear_2d_scalar_4()
    test_api_bilinear_2d_scalar_5()
    test_api_bilinear_2d_block_1()

    test_api_linear_2d_scalar_1()
    test_api_linear_2d_scalar_2()

    test_api_function_2d_scalar_1()
    # ...

    # ...
    test_api_model_2d_poisson()
    test_api_model_2d_stokes()
    # ...
