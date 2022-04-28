import pytest      
import time
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

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

def union(domains, name):
    assert len(domains)>1
    domain = domains[0]
    for p in domains[1:]:
        domain = domain.join(p, name=name)
    return domain

def set_interfaces(domain, interfaces):
    for I in interfaces:
        domain = domain.join(domain, domain.name, bnd_minus=I[0], bnd_plus=I[1], direction=I[2])
    return domain

def diag_dot(A,x,y):
    for i in range(len(x.blocks)):
        if A[i,i] is None:continue
        A[i,i].dot(x[i], out=y[i])
#==============================================================================

def run_poisson_2d(solution, f, domain, ncells, degree, comm=None, backend=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    timing   = {}
    t0 = time.time()
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

    t1 = time.time()
    timing["Abstract model"] = t1-t0

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    t0 = time.time()
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)

    equation_h.set_solver('cg', info=True, tol=1e-8)
    t1 = time.time()

    timing["Discretization"] = t1-t0

    comm.Barrier()
    t0       = time.time()
    uh, info = equation_h.solve()
    t1       = time.time()
    timing['solution'] = t1-t0

    comm.Barrier()
    t0 = time.time()
    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)
    t1       = time.time()
    timing['diagnostics'] = t1-t0

    A = equation_h.linear_system.lhs
    b = equation_h.linear_system.rhs
    T1,T2,T3,T4,T5 = 0,0,0,0,0
    y = b.copy()*0
    for i in range(100):

        comm.Barrier()

        t0 = time.time()
        b.update_ghost_regions_without_interfaces()
        t1 = time.time()
        T1 += t1-t0

        comm.Barrier()

        t0 = time.time()
        b.start_update_interface_ghost_regions()
        b.end_update_interface_ghost_regions()
        t1 = time.time()
        T2 += t1-t0

        comm.Barrier()

        t0 = time.time()
        b.update_ghost_regions()
        t1 = time.time()
        T3 += t1-t0

        comm.Barrier()

        t0 = time.time()
        A.dot(b,out=y)
        t1 = time.time()
        T4 += t1-t0

        comm.Barrier()

        t0 = time.time()
        diag_dot(A,b,y)
        t1 = time.time()
        T5 += t1-t0

    T1                       = comm.reduce(T1/100, op=MPI.MAX)
    T2                       = comm.reduce(T2/100, op=MPI.MAX)
    T3                       = comm.reduce(T3/100, op=MPI.MAX)
    T4                       = comm.reduce(T4/100, op=MPI.MAX)
    T5                       = comm.reduce(T5/100, op=MPI.MAX)
    timing["Abstract model"] = comm.reduce(timing["Abstract model"], op=MPI.MAX)
    timing["Discretization"] = comm.reduce(timing["Discretization"], op=MPI.MAX)
    timing['solution']       = comm.reduce(timing['solution'], op=MPI.MAX)
    timing['diagnostics']    = comm.reduce(timing['diagnostics'], op=MPI.MAX)
 
    return uh, info, T1,T2,T3,T4,T5,timing, l2_error, h1_error


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Solve Poisson's equation on a 3D mulipatch domain."
    )

    parser.add_argument( '-d',
        type    = int,
        nargs   = 3,
        default = [2,2,2],
        metavar = ('P1','P2','P3'),
        dest    = 'degree',
        help    = 'Spline degree along each dimension'
    )

    parser.add_argument( '-n',
        type    = int,
        nargs   = 3,
        default = [10,10,10],
        metavar = ('N1','N2','N3'),
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension'
    )


    from collections                               import OrderedDict
    from sympy                                     import lambdify
    from psydac.api.tests.build_domain             import build_pretzel
    from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_grid_vals
    from psydac.feec.multipatch.plotting_utilities import get_patch_knots_gridlines, my_small_plot

    A1 = Cube('A1',bounds1=(0, 0.5), bounds2=(0, 0.5), bounds3=(0, 1))
    A2 = Cube('A2',bounds1=(0.5, 1), bounds2=(0, 0.5), bounds3=(0, 1))
    A3 = Cube('A3',bounds1=(1, 1.5), bounds2=(0, 0.5), bounds3=(0, 1))
    A4 = Cube('A4',bounds1=(0, 0.5), bounds2=(0.5, 1), bounds3=(0, 1))
    A5 = Cube('A5',bounds1=(0.5, 1), bounds2=(0.5, 1), bounds3=(0, 1))
    A6 = Cube('A6',bounds1=(1, 1.5), bounds2=(0.5, 1), bounds3=(0, 1))
    A7 = Cube('A7',bounds1=(0, 0.5), bounds2=(1, 1.5), bounds3=(0, 1))
    A8 = Cube('A8',bounds1=(0.5, 1), bounds2=(1, 1.5), bounds3=(0, 1))
    A9 = Cube('A9',bounds1=(1, 1.5), bounds2=(1, 1.5), bounds3=(0, 1))

    interfaces = [
    [A1.get_boundary(axis=0, ext=+1), A2.get_boundary(axis=0, ext=-1),1],
    [A2.get_boundary(axis=0, ext=+1), A3.get_boundary(axis=0, ext=-1),1],
    [A4.get_boundary(axis=0, ext=+1), A5.get_boundary(axis=0, ext=-1),1],
    [A5.get_boundary(axis=0, ext=+1), A6.get_boundary(axis=0, ext=-1),1],
    [A7.get_boundary(axis=0, ext=+1), A8.get_boundary(axis=0, ext=-1),1],
    [A8.get_boundary(axis=0, ext=+1), A9.get_boundary(axis=0, ext=-1),1],
    [A1.get_boundary(axis=1, ext=+1), A4.get_boundary(axis=1, ext=-1),1],
    [A2.get_boundary(axis=1, ext=+1), A5.get_boundary(axis=1, ext=-1),1],
    [A3.get_boundary(axis=1, ext=+1), A6.get_boundary(axis=1, ext=-1),1],
    [A4.get_boundary(axis=1, ext=+1), A7.get_boundary(axis=1, ext=-1),1],
    [A5.get_boundary(axis=1, ext=+1), A8.get_boundary(axis=1, ext=-1),1],
    [A6.get_boundary(axis=1, ext=+1), A9.get_boundary(axis=1, ext=-1),1],
    ]

    domain = union([A1, A2, A3, A4, A5, A6, A7, A8, A9], name = 'domain')
    domain = set_interfaces(domain, interfaces)

    x,y,z    = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f        = 3*pi**2*solution
    args     = parser.parse_args()

    ne     = vars(args)['ncells']
    degree = vars(args)['degree']

    comm = MPI.COMM_WORLD
    u_h, info, T1, T2, T3, T4, T5, timing, l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=ne, degree=degree, comm=comm, backend=PSYDAC_BACKEND_GPYCCEL)

    if comm.rank == 0:
        print("number of mpi procs             :: ", comm.size)
        print("number of patches               :: ", len(domain))
        print("ncells                          :: ", ne)
        print("degree                          :: ", degree)
        print("interiors communication    time :: ", T1)
        print("interface communication    time :: ", T2)
        print("combined  communication    time :: ", T3)
        print("dot product calculation    time :: ", T4)
        print("dot product calculation lb time :: ", T5)
        print("problem        timings          :: ", timing)
        print("> CG info                       :: ", info )
        print("L2 error                        :: ", l2_error)
        print("H1 error                        :: ", h1_error)
