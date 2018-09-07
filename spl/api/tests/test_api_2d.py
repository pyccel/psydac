# -*- coding: UTF-8 -*-

from sympde.core import dx, dy, dz
from sympde.core import Constant
from sympde.core import Field
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, FunctionForm

from spl.fem.basic   import FemField
from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.api.discretization import discretize

from numpy import linspace, zeros

def test_api_2d_1():
    print('============ test_api_2d_1 =============')

    # ... abstract model
    U = FunctionSpace('U', ldim=2)
    V = FunctionSpace('V', ldim=2)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))

    a = BilinearForm((v,u), expr)
    # ...

    # ... discrete spaces
    # Input data: degree, number of elements
    p1  = 3 ; p2  = 3
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

    # ...
    ah = discretize(a, [V, V])
    M = ah.assemble()
    # ...

def test_api_2d_2():
    print('============ test_api_2d_2 =============')

    # ... abstract model
    U = FunctionSpace('U', ldim=2)
    V = FunctionSpace('V', ldim=2)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    c = Constant('c', real=True, label='mass stabilization')

    expr = dot(grad(v), grad(u)) + c*v*u

    a = BilinearForm((v,u), expr)
    # ...

    # ... discrete spaces
    # Input data: degree, number of elements
    p1  = 3 ; p2  = 3
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

    # ...
    ah = discretize(a, [V, V])
    M = ah.assemble(0.5)
    # ...

def test_api_2d_3():
    print('============ test_api_2d_3 =============')

    # ... abstract model
    U = FunctionSpace('U', ldim=2)
    V = FunctionSpace('V', ldim=2)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)

    expr = dot(grad(v), grad(u)) + F*v*u

    a = BilinearForm((v,u), expr)
    # ...

    # ... discrete spaces
    # Input data: degree, number of elements
    p1  = 3 ; p2  = 3
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

    # ...
    ah = discretize(a, [V, V])

    # Define a field
    phi = FemField( V, 'phi' )
    phi._coeffs[:,:] = 1.

    M = ah.assemble(phi)
    # ...

def test_api_2d_4():
    print('============ test_api_2d_4 =============')

    # ... abstract model
    U = FunctionSpace('U', ldim=2)
    V = FunctionSpace('V', ldim=2)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    F = Field('F', space=V)
    G = Field('G', space=V)

    expr = dot(grad(G*v), grad(u)) + F*v*u

    a = BilinearForm((v,u), expr)
    # ...

    # ... discrete spaces
    # Input data: degree, number of elements
    p1  = 3 ; p2  = 3
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

    # ...
    ah = discretize(a, [V, V])

    # Define a field
    phi = FemField( V, 'phi' )
    phi._coeffs[:,:] = 1.

    psi = FemField( V, 'psi' )
    psi._coeffs[:,:] = 1.

    M = ah.assemble(phi, psi)
    # ...


###############################################
if __name__ == '__main__':

    test_api_2d_1()
    test_api_2d_2()
    test_api_2d_3()
    test_api_2d_4()
