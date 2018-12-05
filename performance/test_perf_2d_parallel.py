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

from mpi4py import MPI

DEBUG = False

domain = Domain('\Omega', dim=2)

# Communicator, size, rank
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

def create_discrete_space(p=(2,2), ne=(2,2), comm=MPI.COMM_WORLD):
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
    V = TensorFemSpace( V1, V2, comm=comm )
    # ...

    return V

def print_timing(ls, backend):
    # ...
    backend_name = backend['name']
    table   = []
    headers = ['Assembly time', backend_name]

    for timing in ls:
        line   = [timing.kind, timing.value]
        table.append(line)

    print(tabulate(table, headers=headers, tablefmt='latex'))
    # ...


def test_perf_poisson_2d_parallel(backend=SPL_BACKEND_PYTHON):

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
    Vh = create_discrete_space(p=(3,3), ne=(2**8,2**8))
#    Vh = create_discrete_space(p=(2,2), ne=(2**3,2**3))
    # ...

    Timing = namedtuple('Timing', ['kind', 'value'])

    # ...
    ah = discretize(a, [Vh, Vh], backend=backend)
    tb = time.time()
    Ah = ah.assemble()
    te = time.time()

    matrix_timing = Timing('matrix', te-tb)
    # ...

    # ...
    lh = discretize(l, Vh, backend=backend)
    tb = time.time()
    Lh = lh.assemble()
    te = time.time()

    rhs_timing = Timing('rhs', te-tb)
    # ...

    # ... coeff of phi are 0
    phi = FemField( Vh, 'phi' )
    # ...

    # ...
    l2norm_h = discretize(l2norm, Vh, backend=backend)
    tb = time.time()
    error = l2norm_h.assemble(F=phi)
    te = time.time()

    l2norm_timing = Timing('l2norm', te-tb)
    # ...

    # ...
    if mpi_rank == 0:
        print_timing([matrix_timing, rhs_timing, l2norm_timing], backend)
    # ...

def test_perf_vector_poisson_2d(backend=SPL_BACKEND_PYTHON):

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
    Vh = create_discrete_space(p=(3,3), ne=(2**8,2**8))
#    Vh = create_discrete_space(p=(2,2), ne=(2**3,2**3))
    Vh = ProductFemSpace(Vh, Vh)
    # ...

    Timing = namedtuple('Timing', ['kind', 'value'])

    # ...
    ah = discretize(a, [Vh, Vh], backend=backend)
    tb = time.time()
    Ah = ah.assemble()
    te = time.time()

    matrix_timing = Timing('matrix', te-tb)
    # ...

    # ...
    lh = discretize(l, Vh, backend=backend)
    tb = time.time()
    Lh = lh.assemble()
    te = time.time()

    rhs_timing = Timing('rhs', te-tb)
    # ...

    # ... coeff of phi are 0
    phi = VectorFemField( Vh, 'phi' )
    # ...

    # ...
    l2norm_h = discretize(l2norm, Vh, backend=backend)
    tb = time.time()
    L_f90 = l2norm_h.assemble(F=phi)
    te = time.time()

    l2norm_timing = Timing('l2norm', te-tb)
    # ...

    # ...
    if mpi_rank == 0:
        print_timing([matrix_timing, rhs_timing, l2norm_timing], backend)
    # ...


###############################################
if __name__ == '__main__':

    # ...
#    test_perf_poisson_2d_parallel(backend=SPL_BACKEND_PYCCEL)
    test_perf_vector_poisson_2d(backend=SPL_BACKEND_PYCCEL)
#    test_perf_vector_poisson_2d(backend=SPL_BACKEND_PYTHON)
    # ...
