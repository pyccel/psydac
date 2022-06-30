import numpy as np

from mpi4py import MPI

from sympde.topology import Square, ScalarFunctionSpace, VectorFunctionSpace
from sympy import degree


from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.api.discretization import discretize
from psydac.fem.basic import FemField
from psydac.utilities.utils import refine_array_1d



# comm = MPI.COMM_WORLD


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
from sympde.calculus import dot, curl, cross
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus

from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import InteriorDomain, Union
from sympde.topology import Boundary, NormalVector
from sympde.topology import Domain
from sympde.topology import trace_1
from sympde.topology import Square
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

from mpi4py import MPI
#==============================================================================

def run_poisson_2d(solution, f, domain, ncells, degree, comm=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V   = ScalarFunctionSpace('V', domain, kind=None)

    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')

    bc   = EssentialBC(u, solution, domain.boundary)

    error  = u - solution

    I = domain.interfaces

    kappa  = 10**3

    expr_I =- 0.5*dot(grad(plus(u)),nn)*minus(v)  + 0.5*dot(grad(minus(v)),nn)*plus(u)  - kappa*plus(u)*minus(v)\
            + 0.5*dot(grad(minus(u)),nn)*plus(v)  - 0.5*dot(grad(plus(v)),nn)*minus(u)  - kappa*plus(v)*minus(u)\
            - 0.5*dot(grad(minus(v)),nn)*minus(u) - 0.5*dot(grad(minus(u)),nn)*minus(v) + kappa*minus(u)*minus(v)\
            + 0.5*dot(grad(plus(v)),nn)*plus(u)   + 0.5*dot(grad(plus(u)),nn)*plus(v)   + kappa*plus(u)*plus(v)

    expr   = dot(grad(u),grad(v))

    a = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I))
    l = LinearForm(v, integral(domain, f*v))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)

    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh])

    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)

    equation_h.set_solver('cg', info=True, tol=1e-14)

    timing   = {}
    t0       = time.time()
    uh, info = equation_h.solve()
    t1       = time.time()
    timing['solution'] = t1-t0

    t0 = time.time()
    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)
    t1       = time.time()
    timing['diagnostics'] = t1-t0

    return uh, info, timing, l2_error, h1_error

#==============================================================================
def run_maxwell_2d(uex, f, alpha, domain, ncells, degree, k=None, kappa=None, comm=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V  = VectorFunctionSpace('V', domain, kind='hcurl')

    u, v, F  = elements_of(V, names='u, v, F')
    nn       = NormalVector('nn')

    error   = Matrix([F[0]-uex[0],F[1]-uex[1]])

    I        = domain.interfaces
    boundary = domain.boundary

    kappa   = 10*ncells[0]
    k       = 1

    jump = lambda w:plus(w)-minus(w)
    avr  = lambda w:(plus(w) + minus(w))/2

    expr1_I  =  cross(nn, jump(v))*curl(avr(u))\
               +k*cross(nn, jump(u))*curl(avr(v))\
               +kappa*cross(nn, jump(u))*cross(nn, jump(v))

    expr1   = curl(u)*curl(v) + alpha*dot(u,v)
    expr1_b = -cross(nn, v) * curl(u) -k*cross(nn, u)*curl(v)  + kappa*cross(nn, u)*cross(nn, v)

    expr2   = dot(f,v)
    expr2_b = -k*cross(nn, uex)*curl(v) + kappa * cross(nn, uex) * cross(nn, v)

    # Bilinear form a: V x V --> R
    a      = BilinearForm((u,v),  integral(domain, expr1) + integral(I, expr1_I) + integral(boundary, expr1_b))

    # Linear form l: V --> R
    l      = LinearForm(v, integral(domain, expr2) + integral(boundary, expr2_b))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))

    l2norm = Norm(error, domain, kind='l2')
    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree, basis='M')

    equation_h = discretize(equation, domain_h, [Vh, Vh])
    l2norm_h   = discretize(l2norm, domain_h, Vh)

    equation_h.set_solver('pcg', pc='jacobi', tol=1e-8)

    uh = equation_h.solve()

    l2_error = l2norm_h.assemble(F=uh)

    return l2_error, uh


def poisson():

    from psydac.api.tests.build_domain             import build_pretzel

    domain    = build_pretzel()
    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    ne     = [2**2,2**2]
    degree = [2,2]

    u_h, info, timing, l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=ne, degree=degree)

    # ...
    print( '> Grid          :: [{ne1},{ne2}]'.format( ne1=ne[0], ne2=ne[1]) )
    print( '> Degree        :: [{p1},{p2}]'  .format( p1=degree[0], p2=degree[1] ) )
    print( '> CG info       :: ',info )
    print( '> L2 error      :: {:.2e}'.format( l2_error ) )
    print( '> H1 error      :: {:.2e}'.format( h1_error ) )
    print( '' )
    print( '> Solution time :: {:.2e}'.format( timing['solution'] ) )
    print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics'] ) )
    
    Om = OutputManager('poisson_multipatch.yml', 'poisson_multipatch.h5', None, 'w')

    Om.add_spaces(V=u_h.space)
    Om.set_static()
    Om.export_fields(u=u_h)

    Om.export_space_info()

    Om.close()

    Pm = PostProcessManager(domain=domain, space_file='poisson_multipatch.yml', fields_file='poisson_multipatch.h5', comm=None)
    N = 3
    Pm.export_to_vtk('poisson_multipatch', grid=None, npts_per_cell=N, fields={'u_man': 'u'},
                     additional_physical_functions={'u_e': lambda X, Y: X ** 2 + Y ** 2})


def maxwell():

    from psydac.api.tests.build_domain import build_pretzel
    from sympy                         import lambdify

    domain    = build_pretzel()
    x,y       = domain.coordinates

    omega = 1.5
    alpha = -omega**2
    Eex   = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f     = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error, Eh = run_maxwell_2d(Eex, f, alpha, domain, ncells=[2**2, 2**2], degree=[2,2])

    Om = OutputManager('maxwell_multipatch.yml', 'maxwell_multipatch.h5')
    Om.add_spaces(V=Eh.space)
    Om.set_static()
    Om.export_fields(u=Eh)
    Om.export_space_info()
    
    Pm = PostProcessManager(domain=domain, space_file='maxwell_multipatch.yml', fields_file='maxwell_multipatch.h5')

    Eex_x   = lambdify(domain.coordinates, Eex[0])
    Eex_y   = lambdify(domain.coordinates, Eex[1])
    N = 3
    Pm.export_to_vtk('maxwell_multipatch', grid=None, npts_per_cell=N, fields={'u': 'u'},
                     additional_physical_functions={'u_ex': lambda X, Y: (Eex_x(X, Y), Eex_y(X, Y))})



if __name__ == '__main__':
    # poisson()
    maxwell()
