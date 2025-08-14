# -*- coding: UTF-8 -*-

#from sympy import pi, cos, sin
#from sympy import S
#
#from sympde.core     import Constant
#from sympde.calculus import grad, dot, inner, cross, rot, curl, div
#
#from sympde.topology import dx, dy, dz
#from sympde.topology import ScalarField
#from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
#from sympde.topology import element_of
#from sympde.topology import Domain
#from sympde.topology import Boundary, trace_0, trace_1
#from sympde.expr     import BilinearForm, LinearForm
#from sympde.expr     import Norm
#from sympde.expr     import find, EssentialBC
#from sympde.topology import Domain, Line, Square, Cube
#
#from psydac.fem.basic   import FemField
#from psydac.fem.splines import SplineSpace
#from psydac.fem.tensor  import TensorFemSpace
#from psydac.api.discretization import discretize
#from psydac.api.settings import PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL

#from numpy import linspace, zeros
#
#import time
#from tabulate import tabulate
#from collections import namedtuple
#
#Timing = namedtuple('Timing', ['kind', 'python', 'pyccel'])
#
#domain = Cube()

#==============================================================================
#def print_timing(ls):
#    # ...
#    table   = []
#    headers = ['Assembly time', 'Python', 'Pyccel', 'Speedup']
#
#    for timing in ls:
#        speedup = timing.python / timing.pyccel
#        line   = [timing.kind, timing.python, timing.pyccel, speedup]
#        table.append(line)
#
#    print(tabulate(table, headers=headers, tablefmt='latex'))
#    # ...
#
#==============================================================================
#def test_api_stokes_3d():
#    print('============ test_api_stokes_3d =============')
#
#    # ... abstract model
#    U = VectorFunctionSpace('V', domain)
#    V = ScalarFunctionSpace('W', domain)
#
#    W = U*V
#
#    v = element_of(U, 'v')
#    u = element_of(U, 'u')
#    p = element_of(V, 'p')
#    q = element_of(V, 'q')
#
#    A = BilinearForm((v,u), inner(grad(v), grad(u)))
#    B = BilinearForm((v,p), div(v)*p)
#    a = BilinearForm(((v,q),(u,p)), A(v,u) - B(v,p) + B(u,q))
#    # ...
#
#    domain_h = discretize(domain, ncells=(2**3, 2**3, 2**3))
#    # ...
#
#    # ... discrete spaces
#    Vh = discretize(W, domain_h, degree=(3, 3, 3))
#    # ...
#
#    # ...
#    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)
#    tb = time.time()
#    M_f90 = ah.assemble()
#    te = time.time()
#    print('> [pyccel] elapsed time (matrix) = ', te-tb)
#    t_f90 = te-tb
#
#    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
#    tb = time.time()
#    M_py = ah.assemble()
#    te = time.time()
#    print('> [python] elapsed time (matrix) = ', te-tb)
#    t_py = te-tb
#
#    matrix_timing = Timing('matrix', t_py, t_f90)
#    # ...
#
#    # ...
#    print_timing([matrix_timing])
##    print_timing([matrix_timing, rhs_timing, l2norm_timing])
#    # ...

#==============================================================================
import pytest
from psydac.api.settings import PSYDAC_BACKENDS
from utilities import print_timings_table

# IMPORT FUNCTIONS TO BE PROFILED
from psydac.api.tests.test_3d_poisson import test_3d_poisson_dir_1
from psydac.api.tests.test_3d_mapping_laplace import test_3d_laplace_neu_collela

test_functions = (
    test_3d_poisson_dir_1,
    test_3d_laplace_neu_collela,
)

#==============================================================================
@pytest.mark.parametrize('func', test_functions)
def test_compare_python_with_pyccel_gcc(func):

    print()
    print(f'Benchmarking: {func.__name__}')

    # Simple Python run
    python_timing = {}
    func(backend=None, timing=python_timing)

    # Use Pyccel with Fortran code generation, serial
    backend = PSYDAC_BACKENDS['pyccel-gcc']
    pyccel_timing = {}
    func(backend=backend, timing=pyccel_timing)

    print_timings_table(python_timing, pyccel_timing)
    print('NOTE: Pyccel = Fortran language, GFortran compiler, no OpenMP')
    print()

#==============================================================================
# SCRIPT USAGE
#==============================================================================
if __name__ == '__main__':

    for func in test_functions:
        test_compare_python_with_pyccel_gcc(func)

#    test_api_stokes_3d()
