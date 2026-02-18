#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympy import pi, cos, sin
from sympy import S

from sympde.core     import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div

from sympde.topology import dx, dy, dz
from sympde.topology import ScalarField
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of
from sympde.topology import Domain
from sympde.topology import Boundary, trace_0, trace_1
from sympde.expr     import BilinearForm, LinearForm
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC
from sympde.topology import Domain, Line, Square, Cube

from psydac.fem.basic   import FemField
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL
from numpy import linspace, zeros, allclose
from utils import assert_identical_coo

import time
from tabulate import tabulate
from collections import namedtuple

from mpi4py import MPI

DEBUG = False

domain = Square()

# Communicator, size, rank
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

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


def test_perf_poisson_2d_parallel(backend=PSYDAC_BACKEND_PYTHON):

    # ... abstract model
    V = ScalarFunctionSpace('V', domain)

    x,y = domain.coordinates

    F = element_of(V,'F')

    v = element_of(V, 'v')
    u = element_of(V, 'u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = 2*pi**2*sin(pi*x)*sin(pi*y)*v
    l = LinearForm(v, expr)

    error = F -sin(pi*x)*sin(pi*y)
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')
    # ...

    domain_h = discretize(domain, ncells=(2**8,2**8))
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=(3,3))
    # ...

    Timing = namedtuple('Timing', ['kind', 'value'])

    # ...
    ah = discretize(a, domain_h, [Vh, Vh], backend=backend)
    tb = time.time()
    Ah = ah.assemble()
    te = time.time()

    matrix_timing = Timing('matrix', te-tb)
    # ...

    # ...
    lh = discretize(l, domain_h, Vh, backend=backend)
    tb = time.time()
    Lh = lh.assemble()
    te = time.time()

    rhs_timing = Timing('rhs', te-tb)
    # ...

    # ... coeff of phi are 0
    phi = FemField( Vh, 'phi' )
    # ...

    # ...
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    tb = time.time()
    error = l2norm_h.assemble(F=phi)
    te = time.time()

    l2norm_timing = Timing('l2norm', te-tb)
    # ...

    # ...
    if mpi_rank == 0:
        print_timing([matrix_timing, rhs_timing, l2norm_timing], backend)
    # ...

def test_perf_vector_poisson_2d(backend=PSYDAC_BACKEND_PYTHON):

    # ... abstract model
    V = VectorFunctionSpace('V', domain)
    
    x,y = domain.coordinates

    F = Element_of_space(V, 'F')

    v = Element_of_space(V, 'v')
    u = Element_of_space(V, 'u')

    expr = inner(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    f = Tuple(2*pi**2*sin(pi*x)*sin(pi*y), 2*pi**2*sin(pi*x)*sin(pi*y))

    expr = dot(f, v)
    l = LinearForm(v, expr)

    # TODO improve
    error = F[0] -sin(pi*x)*sin(pi*y) + F[1] -sin(pi*x)*sin(pi*y)
    l2norm = Norm(error, domain, kind='l2', name='u')
    # ...

    domain_h = discretize(domain, ncells=(2**8,2**8))
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=(3,3))

    # ...

    Timing = namedtuple('Timing', ['kind', 'value'])

    # ...
    ah = discretize(a, ,domain_h, [Vh, Vh], backend=backend)
    tb = time.time()
    Ah = ah.assemble()
    te = time.time()

    matrix_timing = Timing('matrix', te-tb)
    # ...

    # ...
    lh = discretize(l, domain_h, Vh, backend=backend)
    tb = time.time()
    Lh = lh.assemble()
    te = time.time()

    rhs_timing = Timing('rhs', te-tb)
    # ...

    # ... coeff of phi are 0
    phi = VectorFemField( Vh )
    # ...

    # ...
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
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
#    test_perf_poisson_2d_parallel(backend=PSYDAC_BACKEND_GPYCCEL)
    test_perf_vector_poisson_2d(backend=PSYDAC_BACKEND_GPYCCEL)
#    test_perf_vector_poisson_2d(backend=PSYDAC_BACKEND_PYTHON)
    # ...
