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

import time
from tabulate import tabulate
from collections import namedtuple

Timing = namedtuple('Timing', ['kind', 'python', 'pyccel'])

DEBUG = False

#def test_api_vector_poisson_2d():
#    print('============ test_api_vector_poisson_2d =============')
#
#    # ... abstract model
#    U = VectorFunctionSpace('U', domain)
#    V = VectorFunctionSpace('V', domain)
#
#    x,y = domain.coordinates
#
#    F = VectorField(V, name='F')
#
#    v = VectorTestFunction(V, name='v')
#    u = VectorTestFunction(U, name='u')
#
#    expr = inner(grad(v), grad(u))
#    a = BilinearForm((v,u), expr)
#
#    f = Tuple(2*pi**2*sin(pi*x)*sin(pi*y), 2*pi**2*sin(pi*x)*sin(pi*y))
#
#    expr = dot(f, v)
#    l = LinearForm(v, expr)
#
#    # TODO improve
#    error = F[0] -sin(pi*x)*sin(pi*y) + F[1] -sin(pi*x)*sin(pi*y)
#    l2norm = Norm(error, domain, kind='l2', name='u')
#    # ...
#
#    # ... discrete spaces
##    Vh = create_discrete_space(p=(3,3), ne=(2**8,2**8))
#    Vh = create_discrete_space(p=(2,2), ne=(2**3,2**3))
#    Vh = ProductFemSpace(Vh, Vh)
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
#    lh = discretize(l, Vh, backend=PSYDAC_BACKEND_PYCCEL)
#    tb = time.time()
#    L_f90 = lh.assemble()
#    te = time.time()
#    print('> [pyccel] elapsed time (rhs) = ', te-tb)
#    t_f90 = te-tb
#
#    lh = discretize(l, Vh, backend=PSYDAC_BACKEND_PYTHON)
#    tb = time.time()
#    L_py = lh.assemble()
#    te = time.time()
#    print('> [python] elapsed time (rhs) = ', te-tb)
#    t_py = te-tb
#
#    rhs_timing = Timing('rhs', t_py, t_f90)
#    # ...
#
#    # ... coeff of phi are 0
#    phi = VectorFemField( Vh, 'phi' )
#    # ...
#
#    # ...
#    l2norm_h = discretize(l2norm, Vh, backend=PSYDAC_BACKEND_PYCCEL)
#    tb = time.time()
#    L_f90 = l2norm_h.assemble(F=phi)
#    te = time.time()
#    t_f90 = te-tb
#    print('> [pyccel] elapsed time (L2 norm) = ', te-tb)
#
#    l2norm_h = discretize(l2norm, Vh, backend=PSYDAC_BACKEND_PYTHON)
#    tb = time.time()
#    L_py = l2norm_h.assemble(F=phi)
#    te = time.time()
#    print('> [python] elapsed time (L2 norm) = ', te-tb)
#    t_py = te-tb
#
#    l2norm_timing = Timing('l2norm', t_py, t_f90)
#    # ...
#
#    # ...
#    print_timing([matrix_timing, rhs_timing, l2norm_timing])
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

#==============================================================================
def run_poisson(domain, solution, f, ncells, degree, backend):

    # ... abstract model
    V = ScalarFunctionSpace('V', domain)

    x,y = domain.coordinates

    F = element_of(V, 'F')

    v = element_of(V, 'v')
    u = element_of(V, 'u')

    a = BilinearForm((v,u), dot(grad(v), grad(u)))
    l = LinearForm(v, f*v)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    # ...

    # dict to store timings
    d = {}

    # ... bilinear form
    ah = discretize(a, domain_h, [Vh, Vh], backend=backend)

    tb = time.time(); M = ah.assemble(); te = time.time()

    d['matrix'] = te-tb
    # ...

    # ... linear form
    lh = discretize(l, domain_h, Vh, backend=backend)

    tb = time.time(); L = lh.assemble(); te = time.time()

    d['rhs'] = te-tb
    # ...

    # ... norm
    # coeff of phi are 0
    phi = FemField( Vh )

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)

    tb = time.time(); err = l2norm_h.assemble(F=phi); te = time.time()

    d['l2norm'] = te-tb
    # ...

    return d

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_perf_poisson_2d(ncells=[2**3,2**3], degree=[2,2]):
    domain = Square()
    x,y = domain.coordinates

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    # using Python
    d_py = run_poisson( domain, solution, f,
                        ncells=ncells, degree=degree,
                        backend=PSYDAC_BACKEND_PYTHON )

    # using Pyccel
    d_f90 = run_poisson( domain, solution, f,
                         ncells=ncells, degree=degree,
                         backend=PSYDAC_BACKEND_GPYCCEL )

    # ... add every new backend here
    d_all = [d_py, d_f90]

    keys = sorted(list(d_py.keys()))
    timings = []
    for key in keys:
        args = [d[key] for d in d_all]
        timing = Timing(key, *args)
        timings += [timing]

    print_timing(timings)
    # ...

###############################################
if __name__ == '__main__':

    # ... examples without mapping
    test_perf_poisson_2d()
#    test_api_vector_poisson_2d()
#    test_api_stokes_2d()
    # ...
