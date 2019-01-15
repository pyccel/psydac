from sympy import pi, cos, sin

from sympde.core import Constant
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import Field, VectorField
from sympde.topology import ProductSpace
from sympde.topology import TestFunction
from sympde.topology import VectorTestFunction
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm, Integral
from sympde.expr import Norm
from sympde.expr import Equation, DirichletBC

from spl.fem.basic   import FemField
from spl.api.discretization import discretize
from spl.api.settings import SPL_BACKEND_PYTHON, SPL_BACKEND_PYCCEL,SPL_BACKEND_NUMBA,SPL_BACKEND_PYTHRAN

import time
from tabulate import tabulate
from collections import namedtuple
 
Timing = namedtuple('Timing', ['kind', 'pyccel', 'numba', 'pythran'])

DEBUG = False


def print_timing(ls):
    # ...
    table   = []
    headers = ['Assembly time', 'Pyccel', 'Numba','pythran']

    for timing in ls:
        #speedup = timing.python / timing.pyccel
        line   = [timing.kind, timing.pyccel, timing.numba, timing.pythran]
        table.append(line)

    print(tabulate(table, headers=headers, tablefmt='latex'))
    

#==============================================================================
def run_poisson(domain, solution, f, ncells, degree, backend):

    # ... abstract model
    V = FunctionSpace('V', domain)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(V, name='u')

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
    n = 1 if backend==SPL_BACKEND_PYTHON else 1
    # dict to store timings
    d = {}

    
    # ... bilinear form
    ah = discretize(a, domain_h, [Vh, Vh], backend=backend)

    tb = time.time(); M = ah.assemble(); te = time.time()
    
    times = []
    for i in range(n):
        tb = time.time(); M = ah.assemble(); te = time.time()
        times.append(te-tb)


    d['matrix'] = sum(times)/len(times)
    # ...

    # ... linear form
    lh = discretize(l, domain_h, Vh, backend=backend)

    tb = time.time(); L = lh.assemble(); te = time.time()
    
    times = []
    for i in range(n):
        tb = time.time(); lh = lh.assemble(); te = time.time()
        times.append(te-tb)

    d['rhs'] = sum(times)/len(times)

    # ... norm
    # coeff of phi are 0
    phi = FemField( Vh, 'phi' )

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    
    err = l2norm_h.assemble(F=phi)
    times = []
    for i in range(n):
        tb = time.time(); err = l2norm_h.assemble(F=phi); te = time.time()
        times.append(te-tb)
        
    d['l2norm'] = sum(times)/len(times)
    # ...

    return d

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_perf_poisson_3d(ncells=[2**3,2**3,2**3], degree=[2,2,2]):
    domain = Cube()
    x,y,z = domain.coordinates

    solution = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f        = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    

                         
    # using pythran
    d_pythran = run_poisson( domain, solution, f,
                          ncells=ncells, degree=degree, 
                          backend=SPL_BACKEND_PYTHRAN)  
                          
    # using Python               
   # d_py = run_poisson( domain, solution, f,
   #                     ncells=ncells, degree=degree,
   #                     backend=SPL_BACKEND_PYTHON )


                         
    # using numba
    d_numba = run_poisson( domain, solution, f,
                          ncells=ncells, degree=degree, 
                          backend=SPL_BACKEND_NUMBA )   
                          
      # using Pyccel
    d_f90 = run_poisson( domain, solution, f,
                         ncells=ncells, degree=degree,
                         backend=SPL_BACKEND_PYCCEL )
                          
                          
 


    # ... add every new backend here
    d_all = [d_f90, d_numba, d_pythran]

    keys = sorted(list(d_f90.keys()))
    timings = []
    
    for key in keys:
        args = [d[key] for d in d_all]
        timing = Timing(key, *args)
        timings += [timing]

    print_timing(timings)
    # ...

###############################################
if __name__ == '__main__':

    test_perf_poisson_3d()

