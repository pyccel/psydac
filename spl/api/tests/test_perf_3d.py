# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S
from sympy import Tuple

from sympde.core import dx, dy, dz
from sympde.core import Constant
from sympde.core import Field
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace, VectorFunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorTestFunction
from sympde.core import Domain
from sympde.core import BilinearForm, LinearForm, Integral

from spl.fem.basic   import FemField
from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.api.discretization import discretize
from spl.api.settings import SPL_BACKEND_PYTHON, SPL_BACKEND_PYCCEL

from numpy import linspace, zeros

import time

domain = Domain('\Omega', dim=3)

def create_discrete_space(p=(2,2,2), ne=(2,2,2)):
    # ... discrete spaces
    # Input data: degree, number of elements
    p1,p2,p3 = p
    ne1,ne2,ne3 = ne

    # Create uniform grid
    grid_1 = linspace( 0., 1., num=ne1+1 )
    grid_2 = linspace( 0., 1., num=ne2+1 )
    grid_3 = linspace( 0., 1., num=ne3+1 )

    # Create 1D finite element spaces and precompute quadrature data
    V1 = SplineSpace( p1, grid=grid_1 ); V1.init_fem()
    V2 = SplineSpace( p2, grid=grid_2 ); V2.init_fem()
    V3 = SplineSpace( p3, grid=grid_3 ); V3.init_fem()

    # Create 3D tensor product finite element space
    V = TensorFemSpace( V1, V2, V3 )
    # ...

    return V


def test_api_poisson_3d():
    print('============ test_api_poisson_3d =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    x,y,z = domain.coordinates

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)*v
    l = LinearForm(v, expr)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space(ne=(2**3, 2**3, 2**3))
    # ...

    # ...
    ah = discretize(a, [Vh, Vh], backend=SPL_BACKEND_PYCCEL)
    tb = time.time()
    M_f90 = ah.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (matrix) = ', te-tb)

    ah = discretize(a, [Vh, Vh], backend=SPL_BACKEND_PYTHON)
    tb = time.time()
    M_py = ah.assemble()
    te = time.time()
    print('> [python] elapsed time (matrix) = ', te-tb)
    # ...

    # ...
    lh = discretize(l, Vh, backend=SPL_BACKEND_PYCCEL)
    tb = time.time()
    L_f90 = lh.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (rhs) = ', te-tb)

    lh = discretize(l, Vh, backend=SPL_BACKEND_PYTHON)
    tb = time.time()
    L_py = lh.assemble()
    te = time.time()
    print('> [python] elapsed time (rhs) = ', te-tb)
    # ...


###############################################
if __name__ == '__main__':

    test_api_poisson_3d()
