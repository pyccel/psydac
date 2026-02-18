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
from sympde.topology import Square
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
    nquads = [p+1 for p in degree]

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh], nquads=nquads)

    l2norm_h = discretize(l2norm, domain_h, Vh, nquads=nquads)
    h1norm_h = discretize(h1norm, domain_h, Vh, nquads=nquads)

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


if __name__ == '__main__':

    from collections                               import OrderedDict
    from sympy                                     import lambdify
    from psydac.api.tests.build_domain             import build_pretzel
    from psydac.fem.plotting_utilities import get_plotting_grid, get_grid_vals
    from psydac.fem.plotting_utilities import get_patch_knots_gridlines, my_small_plot

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
    N = 20

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])

    mappings_list = list(mappings.values())

    u_ex = lambdify(domain.coordinates, solution)
    f_ex = lambdify(domain.coordinates, f)
    F    = [f.get_callable_mapping() for f in mappings_list]

    u_ex_log = [lambda xi1, xi2,ff=f : u_ex(*ff(xi1,xi2)) for f in F]

    N=20
    etas, xx, yy = get_plotting_grid(mappings, N)
    gridlines_x1, gridlines_x2 = get_patch_knots_gridlines(u_h.space, N, mappings, plotted_patch=1)

    grid_vals_h1 = lambda v: get_grid_vals(v, etas, mappings_list, space_kind='h1')

    u_ref_vals = grid_vals_h1(u_ex_log)
    u_h_vals   = grid_vals_h1(u_h)
    u_err      = [abs(uir - uih) for uir, uih in zip(u_ref_vals, u_h_vals)]

    my_small_plot(
        title=r'Solution of Poisson problem $\Delta \phi = f$',
        vals=[u_ref_vals, u_h_vals, u_err],
        titles=[r'$\phi^{ex}(x,y)$', r'$\phi^h(x,y)$', r'$|(\phi-\phi^h)(x,y)|$'],
        xx=xx, yy=yy,
        gridlines_x1=gridlines_x1,
        gridlines_x2=gridlines_x2,
        surface_plot=True,
        cmap='jet',
    )

