#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import time

import pytest
import numpy as np
import matplotlib.pyplot as plt
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
from sympde.expr.expr         import LinearExpr
from sympde.expr.expr         import LinearForm, BilinearForm
from sympde.expr.expr         import integral              
from sympde.expr.expr         import Functional, Norm                       
from sympde.expr.expr         import linearize                      
from sympde.expr.evaluation   import TerminalExpr
from sympde.expr              import find, EssentialBC

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

#==============================================================================
def run_poisson_3d_multi_patch(solution, f, domain, ncells, degree, comm=None, backend=None):

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
    nquads = [p + 1 for p in degree]

    t0 = time.time()
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh], nquads=nquads,
                            backend=backend, sum_factorization=False)

    l2norm_h = discretize(l2norm, domain_h, Vh, nquads=nquads, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, nquads=nquads, backend=backend)

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
    timing["Abstract model"] = comm.reduce(timing["Abstract model"], op=MPI.MAX)
    timing["Discretization"] = comm.reduce(timing["Discretization"], op=MPI.MAX)
    timing['solution']       = comm.reduce(timing['solution'], op=MPI.MAX)
    timing['diagnostics']    = comm.reduce(timing['diagnostics'], op=MPI.MAX)
 
    return uh, info, timing, l2_error, h1_error


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Solve Poisson's equation on a 3D multipatch domain."
    )

    parser.add_argument( '-d',
        type    = int,
        nargs   = 3,
        default = [2, 2, 2],
        metavar = ('P1', 'P2', 'P3'),
        dest    = 'degree',
        help    = 'Spline degree along each dimension'
    )

    parser.add_argument( '-n',
        type    = int,
        nargs   = 3,
        default = [10, 10, 10],
        metavar = ('N1','N2','N3'),
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension'
    )

    from collections                               import OrderedDict
    from sympy                                     import lambdify
    from psydac.api.tests.build_domain             import build_pretzel
    
    A1 = Cube('A1', bounds1=(0, 0.5), bounds2=(0, 0.5), bounds3=(0, 1))
    A2 = Cube('A2', bounds1=(0.5, 1), bounds2=(0, 0.5), bounds3=(0, 1))
    A3 = Cube('A3', bounds1=(1, 1.5), bounds2=(0, 0.5), bounds3=(0, 1))
    A4 = Cube('A4', bounds1=(0, 0.5), bounds2=(0.5, 1), bounds3=(0, 1))
    A5 = Cube('A5', bounds1=(0.5, 1), bounds2=(0.5, 1), bounds3=(0, 1))
    A6 = Cube('A6', bounds1=(1, 1.5), bounds2=(0.5, 1), bounds3=(0, 1))
    A7 = Cube('A7', bounds1=(0, 0.5), bounds2=(1, 1.5), bounds3=(0, 1))
    A8 = Cube('A8', bounds1=(0.5, 1), bounds2=(1, 1.5), bounds3=(0, 1))
    A9 = Cube('A9', bounds1=(1, 1.5), bounds2=(1, 1.5), bounds3=(0, 1))

    patches = [A1, A2, A3, A4, A5, A6, A7, A8, A9]
    interfaces = [
        ((0, 0, 1), (1, 0, -1), (1, 1, 1)),
        ((1, 0, 1), (2, 0, -1), (1, 1, 1)),
        ((3, 0, 1), (4, 0, -1), (1, 1, 1)),
        ((4, 0, 1), (5, 0, -1), (1, 1, 1)),
        ((6, 0, 1), (7, 0, -1), (1, 1, 1)),
        ((7, 0, 1), (8, 0, -1), (1, 1, 1)),
        ((0, 1, 1), (3, 1, -1), (1, 1, 1)),
        ((1, 1, 1), (4, 1, -1), (1, 1, 1)),
        ((2, 1, 1), (5, 1, -1), (1, 1, 1)),
        ((3, 1, 1), (6, 1, -1), (1, 1, 1)),
        ((4, 1, 1), (7, 1, -1), (1, 1, 1)),
        ((5, 1, 1), (8, 1, -1), (1, 1, 1)),
    ]
    domain  = Domain.join(patches, interfaces, name='domain')

    x,y,z    = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f        = 3*pi**2*solution
    args     = parser.parse_args()

    ne     = vars(args)['ncells']
    degree = vars(args)['degree']

    comm = MPI.COMM_WORLD
    u_h, info, timing, l2_error, h1_error = run_poisson_3d_multi_patch(solution, f, domain, ncells=ne, degree=degree, comm=comm, backend=PSYDAC_BACKEND_GPYCCEL)

    if comm.rank == 0:
        print("number of mpi procs             :: ", comm.size)
        print("number of patches               :: ", len(domain))
        print("ncells                          :: ", ne)
        print("degree                          :: ", degree)
        print("problem        timings          :: ", timing)
        print("> CG info                       :: ", info )
        print("L2 error                        :: ", l2_error)
        print("H1 error                        :: ", h1_error)
