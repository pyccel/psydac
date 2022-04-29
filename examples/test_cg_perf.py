import pytest      
import time
import matplotlib.pyplot as plt
import numpy as np

from sympy.core.containers import Tuple
from sympy                 import Matrix               
from sympy                 import Function                                
from sympy                 import pi, cos, sin, exp                        
      
from sympde.core     import Constant
from sympde.calculus import grad, dot, inner, rot, div
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus

from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import InteriorDomain, Union
from sympde.topology import Boundary, NormalVector
from sympde.topology import Domain
from sympde.topology import trace_1
from sympde.topology import Cube
from sympde.topology import ElementDomain
from sympde.topology import Area
from sympde.topology import IdentityMapping, PolarMapping, AffineMapping
                         
from sympde.expr.expr          import LinearExpr
from sympde.expr.expr          import LinearForm, BilinearForm
from sympde.expr.expr          import integral              
from sympde.expr.expr          import Functional, Norm                       
from sympde.expr.expr          import linearize                      
from sympde.expr.evaluation    import TerminalExpr
from sympde.expr               import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.fem.basic          import FemField
from psydac.utilities.utils    import refine_array_1d
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.iterative_solvers import cg

from mpi4py import MPI
#==============================================================================

def run_poisson_3d(solution, f, domain, ncells, degree, comm):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V   = ScalarFunctionSpace('V', domain, kind='h1')

    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')

    kappa  = 10**3
    error  = u - solution
    expr   = dot(grad(u),grad(v))
    expr_b = -v*dot(grad(u),nn) -dot(grad(v), nn)*u + kappa*u*v

    a = BilinearForm((u,v),  integral(domain, expr) + integral(domain.boundary, expr_b))
    l = LinearForm(v, integral(domain, f*v))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))

    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)

    equation_h.set_solver('cg', info=True, tol=1e-14)

    A = equation_h.lhs.assemble()
    b = equation_h.rhs.assemble()

    timing   = {}
    t0       = time.time()
    x, info = cg( A, b, tol=1e-13, maxiter=1000 )
    t1       = time.time()

    T = comm.reduce(t1-t0, op=MPI.MAX)
    uh = FemField(Vh, coeffs=x)

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return uh, info, T, l2_error, h1_error


if __name__ == '__main__':

    domain    = Cube()
    x,y,z     = domain.coordinates
    solution  = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f         = 3*pi**2*solution

    ne     = [2**5,2**5,2**5]
    degree = [3,3,3]

    comm=MPI.COMM_WORLD
    u_h, info, T, l2_error, h1_error = run_poisson_3d(solution, f, domain, ncells=ne, degree=degree, comm=comm)

    if comm.rank == 0:
        # ...
        print( '> Grid          :: [{ne1},{ne2},{ne3}]'.format( ne1=ne[0], ne2=ne[1], ne3=ne[2]) )
        print( '> Degree        :: [{p1},{p2},{p3}]'  .format( p1=degree[0], p2=degree[1], p3=degree[2] ) )
        print( '> CG info       :: ',info )
        print( '> CG time       :: ',T )
        print( '> L2 error      :: {:.2e}'.format( l2_error ) )
        print( '> H1 error      :: {:.2e}'.format( h1_error ) )


