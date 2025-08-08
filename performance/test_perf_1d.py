# -*- coding: UTF-8 -*-

import time
from collections import namedtuple

from tabulate import tabulate
from sympy import pi, sin

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.topology import Line

from psydac.fem.basic   import FemField
from psydac.api.discretization import discretize
from psydac.api.settings import PSYDAC_BACKENDS

from utilities import print_timings_table

#==============================================================================
def test_api_poisson_1d():
    print('============ test_api_poisson_1d =============')

    python_timings = {}
    pyccel_timings = {}

    # ... abstract model
    domain = Line()
    V = ScalarFunctionSpace('V', domain)
    x = domain.coordinates

    F = element_of(V, 'F')
    v = element_of(V, 'v')
    u = element_of(V, 'u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v, u), integral(domain, expr))

    expr = pi**2*sin(pi*x)*v
    l = LinearForm(v, integral(domain, expr))

    error = F-sin(pi*x)
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')
    # ...

    # ... discrete spaces
    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=[2**8])
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=[3])
    # ...
    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS['pyccel-gcc'])
    tb = time.time()
    M_f90 = ah.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (matrix) = ', te-tb)
    pyccel_timings['matrix'] = te - tb
    
    ah = discretize(a, domain_h, [Vh, Vh])
    tb = time.time()
    M_py = ah.assemble()
    te = time.time()
    print('> [python] elapsed time (matrix) = ', te-tb)
    python_timings['matrix'] = te - tb
    # ...

    # ...
    lh = discretize(l, domain_h, Vh, backend=PSYDAC_BACKENDS['pyccel-gcc'])
    tb = time.time()
    L_f90 = lh.assemble()
    te = time.time()
    print('> [pyccel] elapsed time (rhs) = ', te-tb)
    pyccel_timings['rhs'] = te - tb

    lh = discretize(l, domain_h, Vh, backend=PSYDAC_BACKENDS['python'])
    tb = time.time()
    L_py = lh.assemble()
    te = time.time()
    print('> [python] elapsed time (rhs) = ', te-tb)
    python_timings['rhs'] = te - tb
    # ...

    # ... coeff of phi are 0
    phi = FemField(Vh)
    # ...

    # ...
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=PSYDAC_BACKENDS['pyccel-gcc'])
    tb = time.time()
    L_f90 = l2norm_h.assemble(F=phi)
    te = time.time()
    print('> [pyccel] elapsed time (L2 norm) = ', te-tb)
    pyccel_timings['L2 norm'] = te - tb

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=PSYDAC_BACKENDS['python'])
    tb = time.time()
    L_py = l2norm_h.assemble(F=phi)
    te = time.time()
    print('> [python] elapsed time (L2 norm) = ', te-tb)
    python_timings['L2 norm'] = te - tb
    # ...

    # ...
    # TODO [YG 08.08.2025]: understand why the call to discretize fails
#    h1norm_h = discretize(h1norm, domain_h, Vh, backend=PSYDAC_BACKENDS['pyccel-gcc'])
#    tb = time.time()
#    L_f90 = l2norm_h.assemble(F=phi)
#    te = time.time()
#    print('> [pyccel] elapsed time (H1 norm) = ', te-tb)
#    pyccel_timings['H1 norm'] = te - tb
#
#    h1norm_h = discretize(h1norm, domain_h, Vh, backend=PSYDAC_BACKENDS['python'])
#    tb = time.time()
#    L_py = h1norm_h.assemble(F=phi)
#    te = time.time()
#    print('> [python] elapsed time (L2 norm) = ', te-tb)
#    python_timings['H1 norm'] = te - tb
    # ...

    # ...
    print()
    print_timings_table(python_timings, pyccel_timings)
    print('NOTE: Pyccel = Fortran language, GFortran compiler, no OpenMP')
    print()
    # ...

#==============================================================================
# SCRIPT USAGE
#==============================================================================
if __name__ == '__main__':

    # ... examples without mapping
    test_api_poisson_1d()
    # ...
