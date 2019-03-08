# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S

from sympde.core import dx, dy, dz
from sympde.core import Mapping
from sympde.core import Constant
from sympde.core import ScalarField
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace, VectorFunctionSpace
from sympde.core import ScalarTestFunction
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, Integral
from sympde.core import Norm
from sympde.core import Equation, DirichletBC
from sympde.core import Domain
from sympde.core import Boundary, trace_0, trace_1
from sympde.core import ComplementBoundary
from sympde.gallery import Poisson, Stokes

from psydac.fem.context import fem_context
from psydac.fem.basic   import FemField
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.api.discretization import discretize
from psydac.api.boundary_condition import DiscreteBoundary
from psydac.api.boundary_condition import DiscreteComplementBoundary
from psydac.api.boundary_condition import DiscreteDirichletBC
from psydac.api.settings import PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_PYCCEL

from psydac.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose, ones
from utils import assert_identical_coo

import time
from tabulate import tabulate
from collections import namedtuple

Timing = namedtuple('Timing', ['kind', 'python', 'pyccel'])

DEBUG = False

domain = Domain('\Omega', dim=1)

def create_discrete_space(p=2, ne=2):
    # ... discrete spaces
    # Input data: degree, number of elements

    # Create uniform grid
    grid = linspace( 0., 1., num=ne+1 )

    # Create finite element space and precompute quadrature data
    V = SplineSpace( p, grid=grid )
    V.init_fem()
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


def test_api_poisson_1d():
    print('============ test_api_poisson_1d =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)

    x = domain.coordinates

    F = ScalarField('F', V)

    v = ScalarTestFunction(V, name='v')
    u = ScalarTestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = pi**2*sin(pi*x)*v
    l = LinearForm(v, expr)

    error = F-sin(pi*x)
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')
    # ...

    # ... discrete spaces
    Vh = create_discrete_space(p=3, ne=2**10)
    # ...

    # ...
    ah = discretize(a, [Vh, Vh], backend=PSYDAC_BACKEND_PYCCEL)
    tb = time.time()
    M_f90 = ah.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (matrix) = ', te-tb)
    t_f90 = te-tb

    ah = discretize(a, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
    tb = time.time()
    M_py = ah.assemble()
    te = time.time()
    print('> [python] elapsed time (matrix) = ', te-tb)
    t_py = te-tb

    matrix_timing = Timing('matrix', t_py, t_f90)
    # ...

    # ...
    lh = discretize(l, Vh, backend=PSYDAC_BACKEND_PYCCEL)
    tb = time.time()
    L_f90 = lh.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (rhs) = ', te-tb)
    t_f90 = te-tb

    lh = discretize(l, Vh, backend=PSYDAC_BACKEND_PYTHON)
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
    l2norm_h = discretize(l2norm, Vh, backend=PSYDAC_BACKEND_PYCCEL)
    tb = time.time()
    L_f90 = l2norm_h.assemble(F=phi)
    te = time.time()
    print('> [pyccel] elapsed time (L2 norm) = ', te-tb)
    t_f90 = te-tb

    l2norm_h = discretize(l2norm, Vh, backend=PSYDAC_BACKEND_PYTHON)
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


###############################################
if __name__ == '__main__':

    # ... examples without mapping
    test_api_poisson_1d()
    # ...
