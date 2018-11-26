# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S
from sympy import Tuple

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
from spl.api.settings import SPL_BACKEND_PYTHON, SPL_BACKEND_PYCCEL

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose
from utils import assert_identical_coo

import time
from tabulate import tabulate
from collections import namedtuple

Timing = namedtuple('Timing', ['kind', 'python', 'pyccel'])

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

def print_timing(ls):
    # ...
    table   = []
    headers = ['Assembly time', 'Python', 'Pyccel', 'Speedup']

    for timing in ls:
        speedup = timing.python / timing.pyccel
        line   = [timing.kind, timing.python, timing.pyccel, speedup]
        table.append(line)

    print(tabulate(table, headers=headers, tablefmt='latex'))
    # ...


def test_api_poisson_2d():
    print('============ test_api_poisson_2d =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

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
    # ...

    # ... discrete spaces
#    Vh = create_discrete_space(p=(3,3), ne=(2**8,2**8))
    Vh = create_discrete_space(p=(2,2), ne=(2**3,2**3))
    # ...

    # ...
    ah = discretize(a, [Vh, Vh], backend=SPL_BACKEND_PYCCEL)
    tb = time.time()
    M_f90 = ah.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (matrix) = ', te-tb)
    t_f90 = te-tb

    ah = discretize(a, [Vh, Vh], backend=SPL_BACKEND_PYTHON)
    tb = time.time()
    M_py = ah.assemble()
    te = time.time()
    print('> [python] elapsed time (matrix) = ', te-tb)
    t_py = te-tb

    matrix_timing = Timing('matrix', t_py, t_f90)
    # ...

    # ...
    lh = discretize(l, Vh, backend=SPL_BACKEND_PYCCEL)
    tb = time.time()
    L_f90 = lh.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (rhs) = ', te-tb)
    t_f90 = te-tb

    lh = discretize(l, Vh, backend=SPL_BACKEND_PYTHON)
    tb = time.time()
    L_py = lh.assemble()
    te = time.time()
    print('> [python] elapsed time (rhs) = ', te-tb)
    t_py = te-tb

    rhs_timing = Timing('rhs', t_py, t_f90)
    # ...

    # ... coeff of phi are 0
    phi = FemField( Vh, 'phi' )
    # ...

    # ...
    l2norm_h = discretize(l2norm, Vh, backend=SPL_BACKEND_PYCCEL)
    tb = time.time()
    L_f90 = l2norm_h.assemble(F=phi)
    te = time.time()
    t_f90 = te-tb
    print('> [pyccel] elapsed time (L2 norm) = ', te-tb)

    l2norm_h = discretize(l2norm, Vh, backend=SPL_BACKEND_PYTHON)
    tb = time.time()
    L_py = l2norm_h.assemble(F=phi)
    te = time.time()
    print('> [python] elapsed time (L2 norm) = ', te-tb)
    t_py = te-tb

    l2norm_timing = Timing('l2norm', t_py, t_f90)
    # ...

    # ...
    print_timing([matrix_timing, rhs_timing, l2norm_timing])
    # ...

def test_api_vector_poisson_2d():
    print('============ test_api_vector_poisson_2d =============')

    # ... abstract model
    U = VectorFunctionSpace('U', domain)
    V = VectorFunctionSpace('V', domain)

    x,y = domain.coordinates

    F = VectorField(V, name='F')

    v = VectorTestFunction(V, name='v')
    u = VectorTestFunction(U, name='u')

    expr = inner(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    f = Tuple(2*pi**2*sin(pi*x)*sin(pi*y), 2*pi**2*sin(pi*x)*sin(pi*y))

    expr = dot(f, v)
    l = LinearForm(v, expr)

    # TODO improve
    error = F[0] -sin(pi*x)*sin(pi*y) + F[1] -sin(pi*x)*sin(pi*y)
    l2norm = Norm(error, domain, kind='l2', name='u')
    # ...

    # ... discrete spaces
#    Vh = create_discrete_space(p=(3,3), ne=(2**8,2**8))
    Vh = create_discrete_space(p=(2,2), ne=(2**3,2**3))
    Vh = ProductFemSpace(Vh, Vh)
    # ...

    # ...
    ah = discretize(a, [Vh, Vh], backend=SPL_BACKEND_PYCCEL)
    tb = time.time()
    M_f90 = ah.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (matrix) = ', te-tb)
    t_f90 = te-tb

    ah = discretize(a, [Vh, Vh], backend=SPL_BACKEND_PYTHON)
    tb = time.time()
    M_py = ah.assemble()
    te = time.time()
    print('> [python] elapsed time (matrix) = ', te-tb)
    t_py = te-tb

    matrix_timing = Timing('matrix', t_py, t_f90)
    # ...

    # ...
    lh = discretize(l, Vh, backend=SPL_BACKEND_PYCCEL)
    tb = time.time()
    L_f90 = lh.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (rhs) = ', te-tb)
    t_f90 = te-tb

    lh = discretize(l, Vh, backend=SPL_BACKEND_PYTHON)
    tb = time.time()
    L_py = lh.assemble()
    te = time.time()
    print('> [python] elapsed time (rhs) = ', te-tb)
    t_py = te-tb

    rhs_timing = Timing('rhs', t_py, t_f90)
    # ...

    # ... coeff of phi are 0
    phi = VectorFemField( Vh, 'phi' )
    # ...

    # ...
    l2norm_h = discretize(l2norm, Vh, backend=SPL_BACKEND_PYCCEL)
    tb = time.time()
    L_f90 = l2norm_h.assemble(F=phi)
    te = time.time()
    t_f90 = te-tb
    print('> [pyccel] elapsed time (L2 norm) = ', te-tb)

    l2norm_h = discretize(l2norm, Vh, backend=SPL_BACKEND_PYTHON)
    tb = time.time()
    L_py = l2norm_h.assemble(F=phi)
    te = time.time()
    print('> [python] elapsed time (L2 norm) = ', te-tb)
    t_py = te-tb

    l2norm_timing = Timing('l2norm', t_py, t_f90)
    # ...

    # ...
    print_timing([matrix_timing, rhs_timing, l2norm_timing])
    # ...

def test_api_stokes_2d():
    print('============ test_api_stokes_2d =============')

    # ... abstract model
    V = VectorFunctionSpace('V', domain)
    W = FunctionSpace('W', domain)

    v = VectorTestFunction(V, name='v')
    u = VectorTestFunction(V, name='u')
    p = TestFunction(W, name='p')
    q = TestFunction(W, name='q')

    A = BilinearForm((v,u), inner(grad(v), grad(u)), name='A')
    B = BilinearForm((v,p), div(v)*p, name='B')
    a = BilinearForm(((v,q),(u,p)), A(v,u) - B(v,p) + B(u,q), name='a')
    # ...

    # ... discrete spaces
#    Vh = create_discrete_space(p=(3,3), ne=(2**8,2**8))
    Vh = create_discrete_space(p=(2,2), ne=(2**3,2**3))
    # ...

    # ...
    ah = discretize(a, [Vh, Vh], backend=SPL_BACKEND_PYCCEL)
    tb = time.time()
    M_f90 = ah.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (matrix) = ', te-tb)
    t_f90 = te-tb

    ah = discretize(a, [Vh, Vh], backend=SPL_BACKEND_PYTHON)
    tb = time.time()
    M_py = ah.assemble()
    te = time.time()
    print('> [python] elapsed time (matrix) = ', te-tb)
    t_py = te-tb

    matrix_timing = Timing('matrix', t_py, t_f90)
    # ...

    # ...
    print_timing([matrix_timing])
#    print_timing([matrix_timing, rhs_timing, l2norm_timing])
    # ...


###############################################
if __name__ == '__main__':

    # ... examples without mapping
    test_api_poisson_2d()
    test_api_vector_poisson_2d()
#    test_api_stokes_2d()
    # ...
