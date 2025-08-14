# -*- coding: UTF-8 -*-

#import time
#from collections import namedtuple
#
#from tabulate import tabulate
#from sympy import pi, sin
#
#from sympde.calculus import grad, dot
#from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
#from sympde.topology import element_of
#from sympde.expr     import BilinearForm, LinearForm, integral
#from sympde.expr     import Norm
#from sympde.topology import Square
#
#from psydac.api.discretization import discretize
#from psydac.api.settings       import PSYDAC_BACKEND_PYCCEL, PSYDAC_BACKEND_PYTHON
#from psydac.fem.basic          import FemField

#Timing = namedtuple('Timing', ['kind', 'python', 'pyccel'])
#
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
#def test_api_stokes_2d():
#    print('============ test_api_stokes_2d =============')
#
#    # ... abstract model
#    V = VectorFunctionSpace('V', domain)
#    W = ScalarFunctionSpace('W', domain)
#
#    v = VectorTestFunction(V, name='v')
#    u = VectorTestFunction(V, name='u')
#    p = ScalarTestFunction(W, name='p')
#    q = ScalarTestFunction(W, name='q')
#
#    A = BilinearForm((v,u), inner(grad(v), grad(u)), name='A')
#    B = BilinearForm((v,p), div(v)*p, name='B')
#    a = BilinearForm(((v,q),(u,p)), A(v,u) - B(v,p) + B(u,q), name='a')
#    # ...
#
#    # ... discrete spaces
##    Vh = create_discrete_space(p=(3,3), ne=(2**8,2**8))
#    Vh = create_discrete_space(p=(2,2), ne=(2**3,2**3))
#
#    # TODO improve this?
#    Vh = ProductFemSpace(Vh, Vh, Vh)
#    # ...
#
#    # ...
#    ah = discretize(a, [Vh, Vh], backend=PSYDAC_BACKEND_PYCCEL)
#    tb = time.time()
#    M_f90 = ah.assemble()
#    te = time.time()
#    print('> [pyccel] elapsed time (matrix) = ', te-tb)
#    t_f90 = te-tb
#
#    ah = discretize(a, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
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
from psydac.api.tests.test_2d_poisson import test_poisson_2d_dir0_1234
from psydac.api.tests.test_2d_vector_poisson import test_vector_poisson_2d_dir0

test_functions = (
    test_poisson_2d_dir0_1234,
    test_vector_poisson_2d_dir0,
)

#==============================================================================
@pytest.mark.parametrize('func', test_functions)
def test_compare_python_with_pyccel_gcc(func):

    print()
    print(f'Benchmarking: {func.__name__}')

    # Simple Python run
    python_timing = {}
    func(timing=python_timing)

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

    # ... examples without mapping
    for func in test_functions:
        test_compare_python_with_pyccel_gcc(func)

#    test_api_stokes_2d()
