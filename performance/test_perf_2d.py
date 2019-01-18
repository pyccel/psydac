# -*- coding: UTF-8 -*-

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
import os
from tabulate import tabulate
from collections import namedtuple
 
Timing = namedtuple('Timing', ['kind', 'python', 'pyccel']
DEBUG = False

# ... get the mesh directory
try:
    mesh_dir = os.environ['SPL_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..')
    mesh_dir = os.path.join(base_dir, 'mesh')
#==============================================================================
def print_timing(ls):
    # ...
    table   = []
    headers = ['Assembly time', 'Python', 'Pyccel','Speedup']

    for timing in ls:
        speedup = timing.python / timing.pyccel
        line   = [timing.kind, timing.python, timing.pyccel, speedup]
        table.append(line)

    print(tabulate(table, headers=headers, tablefmt='latex'))
    # ...

#==============================================================================
def run_poisson(domain, solution, f, ncells, degree, backend):

    # ... abstract model
    V = FunctionSpace('V', domain)

    x,y = domain.coordinates

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
    n = 1 if backend==SPL_BACKEND_PYTHON else 4
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
        tb = time.time(); lh = ah.assemble(); te = time.time()
        times.append(te-tb)

    d['rhs'] = sum(times)/len(times)
    # ...

  
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
    
def run_poisson_2d_mapping(filename, solution, f, backend, comm=None):
    
    domain = Domain.from_file(filename)
    V = FunctionSpace('V', domain)

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(V, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = f*v
    l = LinearForm(v, expr)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    n = 1 if backend==SPL_BACKEND_PYTHON else 1
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename, comm=comm)
    # ...


    # ... discrete spaces
    Vh = discretize(V, domain_h)
    # ...

    d = {}

    ah = discretize(a, domain_h, [Vh, Vh], backend=backend)

    # ...
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
    # ...

  
    # ... norm
    # coeff of phi are 0
    phi = FemField( Vh, 'phi' )
    # ...


    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    tb = time.time(); err = l2norm_h.assemble(F=phi); te = time.time()
    
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
def test_perf_poisson_2d(ncells=[2**8,2**8], degree=[2,2]):
    domain = Square()
    x,y = domain.coordinates

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)
    
    # using Pyccel
    d_f90 = run_poisson( domain, solution, f,
                         ncells=ncells, degree=degree,
                         backend=SPL_BACKEND_PYCCEL )
                         
    # using pythran
    d_pythran = run_poisson( domain, solution, f,
                          ncells=ncells, degree=degree, 
                          backend=SPL_BACKEND_PYTHRAN)  
                          
    # using Python               
    d_py = run_poisson( domain, solution, f,
                        ncells=ncells, degree=degree,
                        backend=SPL_BACKEND_PYTHON )


                         
    # using numba
    d_numba = run_poisson( domain, solution, f,
                          ncells=ncells, degree=degree, 
                          backend=SPL_BACKEND_NUMBA )   
                          
                          
 


    # ... add every new backend here
    d_all = [d_py, d_f90, d_numba, d_pythran]

    keys = sorted(list(d_py.keys()))
    timings = []
    
    for key in keys:
        args = [d[key] for d in d_all]
        timing = Timing(key, *args)
        timings += [timing]

    print_timing(timings)
    # ...
 
#==============================================================================
def test_perf_poisson_2d_dir_quart_circle():
    filename = os.path.join(mesh_dir, 'quart_circle.h5')
    
    from sympy.abc import x,y

    c = pi / (1. - 0.5**2)
    r2 = 1. - x**2 - y**2
    solution = x*y*sin(c * r2)
    f = 4.*c**2*x*y*(x**2 + y**2)*sin(c * r2) + 12.*c*x*y*cos(c * r2)

                         
    # using pythran
    d_pythran = run_poisson_2d_mapping(filename, solution, f,
                          backend=SPL_BACKEND_PYTHRAN)  
                          
#    # using Python               
    d_py = run_poisson_2d_mapping(filename, solution, f,
                        backend=SPL_BACKEND_PYTHON )


                         
   
                          
                          
 


    # ... add every new backend here
    d_all = [d_py, d_f90]

    keys = sorted(list(d_py.keys()))
    timings = []
    
    for key in keys:
        args = [d[key] for d in d_all]
        timing = Timing(key, *args)
        timings += [timing]

    print_timing(timings)

###############################################
if __name__ == '__main__':

    # ... examples without mapping
    test_perf_poisson_2d()
    #test_perf_poisson_2d_dir_quart_circle()
    # ...
