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

from numpy import linspace, zeros

import time
from tabulate import tabulate
from collections import namedtuple

Timing = namedtuple('Timing', ['kind', 'python', 'pyccel'])

domain = Cube()

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


def test_api_poisson_3d():
    print('============ test_api_poisson_3d =============')

    # ... abstract model
    U = ScalarFunctionSpace('U', domain)

    x,y,z = domain.coordinates

    F = element_of(U, 'F')

    v = element_of(U, 'v')
    u = element_of(U, 'u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)*v
    l = LinearForm(v, expr)

    error = F -sin(pi*x)*sin(pi*y)*sin(pi*z)
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')
    # ...
    
    domain_h = discretize(domain, ncells=(2**3, 2**3, 2**3))
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=(3, 3, 3))
    # ...

    # ...
    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)
    tb = time.time()
    M_f90 = ah.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (matrix) = ', te-tb)
    t_f90 = te-tb

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
    tb = time.time()
    M_py = ah.assemble()
    te = time.time()
    print('> [python] elapsed time (matrix) = ', te-tb)
    t_py = te-tb

    matrix_timing = Timing('matrix', t_py, t_f90)
    # ...

    # ...
    lh = discretize(l, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)
    tb = time.time()
    L_f90 = lh.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (rhs) = ', te-tb)
    t_f90 = te-tb

    lh = discretize(l, domain_h, Vh, backend=PSYDAC_BACKEND_PYTHON)
    tb = time.time()
    L_py = lh.assemble()
    te = time.time()
    print('> [python] elapsed time (rhs) = ', te-tb)
    t_py = te-tb

    rhs_timing = Timing('rhs', t_py, t_f90)
    # ...

    # ... coeff of phi are 0
    phi = FemField( Vh )
    # ...

    # ...
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)
    tb = time.time()
    L_f90 = l2norm_h.assemble(F=phi)
    te = time.time()
    print('> [pyccel] elapsed time (L2 norm) = ', te-tb)
    t_f90 = te-tb

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=PSYDAC_BACKEND_PYTHON)
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

def test_api_stokes_3d():
    print('============ test_api_stokes_3d =============')

    # ... abstract model
    U = VectorFunctionSpace('V', domain)
    V = ScalarFunctionSpace('W', domain)

    W = U*V

    v = element_of(U, 'v')
    u = element_of(U, 'u')
    p = element_of(V, 'p')
    q = element_of(V, 'q')

    A = BilinearForm((v,u), inner(grad(v), grad(u)))
    B = BilinearForm((v,p), div(v)*p)
    a = BilinearForm(((v,q),(u,p)), A(v,u) - B(v,p) + B(u,q))
    # ...

    domain_h = discretize(domain, ncells=(2**3, 2**3, 2**3))
    # ...

    # ... discrete spaces
    Vh = discretize(W, domain_h, degree=(3, 3, 3))
    # ...

    # ...
    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)
    tb = time.time()
    M_f90 = ah.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (matrix) = ', te-tb)
    t_f90 = te-tb

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
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

#    test_api_poisson_3d()
    test_api_stokes_3d()

