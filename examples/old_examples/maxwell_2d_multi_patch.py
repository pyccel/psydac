#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import time

import pytest
import numpy as np
from mpi4py import MPI
from sympy  import pi, sin, cos, Tuple, Matrix
from scipy.sparse.linalg import spsolve, inv

from sympde.calculus      import grad, dot, curl, cross
from sympde.calculus      import minus, plus
from sympde.topology      import VectorFunctionSpace
from sympde.topology      import elements_of
from sympde.topology      import NormalVector
from sympde.topology      import Square
from sympde.topology      import IdentityMapping, PolarMapping
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.expr     import Norm
from sympde.expr.equation import find, EssentialBC

from psydac.api.discretization       import discretize
from psydac.api.tests.build_domain   import build_pretzel
from psydac.fem.basic                import FemField
from psydac.api.settings             import PSYDAC_BACKEND_GPYCCEL
from psydac.feec.pull_push           import pull_2d_hcurl

#==============================================================================
def run_maxwell_2d(uex, f, alpha, domain, ncells, degree, k=None, kappa=None, comm=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V  = VectorFunctionSpace('V', domain, kind='hcurl')

    u, v, F = elements_of(V, names='u, v, F')
    nn      = NormalVector('nn')

    error = Matrix([F[0]-uex[0], F[1]-uex[1]])

    I        = domain.interfaces
    boundary = domain.boundary

    kappa = 10*ncells[0]
    k     = 1

    jump = lambda w: plus(w) - minus(w)
    avr  = lambda w: 0.5*plus(w) + 0.5*minus(w)

    expr1_I  =  cross(nn, jump(v)) * curl(avr(u))\
               +k * cross(nn, jump(u)) * curl(avr(v))\
               +kappa * cross(nn, jump(u)) * cross(nn, jump(v))

    expr1   = curl(u) * curl(v) + alpha * dot(u, v)
    expr1_b = -cross(nn, v) * curl(u) -k*cross(nn, u)*curl(v)  + kappa*cross(nn, u)*cross(nn, v)

    expr2   = dot(f, v)
    expr2_b = -k * cross(nn, uex) * curl(v) + kappa * cross(nn, uex) * cross(nn, v)

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v),  integral(domain, expr1) + integral(I, expr1_I) + integral(boundary, expr1_b))

    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, expr2) + integral(boundary, expr2_b))

    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v))

    l2norm = Norm(error, domain, kind='l2')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree, basis='M')

    equation_h = discretize(equation, domain_h, [Vh, Vh])
    l2norm_h   = discretize(l2norm, domain_h, Vh)

    equation_h.assemble()
    jacobi_pc = equation_h.linear_system.lhs.diagonal(inverse=True)
    equation_h.set_solver('pcg', pc=jacobi_pc, tol=1e-8, info=True)

    timing   = {}
    t0       = time.time()
    uh, info = equation_h.solve()
    t1       = time.time()
    timing['solution'] = t1-t0

    t0 = time.time()
    l2_error = l2norm_h.assemble(F=uh)
    t1       = time.time()
    timing['diagnostics'] = t1-t0

    return uh, info, timing, l2_error

#==============================================================================
if __name__ == '__main__':

    from collections                               import OrderedDict
    from sympy                                     import lambdify
    from psydac.api.tests.build_domain             import build_pretzel
    from psydac.fem.plotting_utilities import get_plotting_grid, get_grid_vals
    from psydac.fem.plotting_utilities import get_patch_knots_gridlines, my_small_plot

    domain = build_pretzel()
    x,y    = domain.coordinates
    omega  = 1.5
    alpha  = -omega**2
    Eex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f      = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    ne     = [4, 4]
    degree = [2, 2]

    Eh, info, timing, l2_error = run_maxwell_2d(Eex, f, alpha, domain, ncells=ne, degree=degree)

    # ...
    print( '> Grid          :: [{ne1},{ne2}]'.format( ne1=ne[0], ne2=ne[1]) )
    print( '> Degree        :: [{p1},{p2}]'  .format( p1=degree[0], p2=degree[1] ) )
    print( '> CG info       :: ',info )
    print( '> L2 error      :: {:.2e}'.format( l2_error ) )
    print( '' )
    print( '> Solution time :: {:.2e}'.format( timing['solution'] ) )
    print( '> Evaluat. time :: {:.2e}'.format( timing['diagnostics'] ) )
    N = 20

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = [m for m in mappings.values()]
    call_mappings_list = [m.get_callable_mapping() for m in mappings_list]

    Eex_x   = lambdify(domain.coordinates, Eex[0])
    Eex_y   = lambdify(domain.coordinates, Eex[1])
    Eex_log = [pull_2d_hcurl([Eex_x, Eex_y], f) for f in call_mappings_list]

    etas, xx, yy    = get_plotting_grid(mappings, N=20)
    grid_vals_hcurl = lambda v: get_grid_vals(v, etas, mappings_list, space_kind='hcurl')

    Eh_x_vals, Eh_y_vals = grid_vals_hcurl(Eh)
    E_x_vals , E_y_vals  = grid_vals_hcurl(Eex_log)

    E_x_err = [(u1 - u2) for u1, u2 in zip(E_x_vals, Eh_x_vals)]
    E_y_err = [(u1 - u2) for u1, u2 in zip(E_y_vals, Eh_y_vals)]

    my_small_plot(
        title=r'approximation of solution $u$, $x$ component',
        vals=[E_x_vals, Eh_x_vals, E_x_err],
        titles=[r'$u^{ex}_x(x,y)$', r'$u^h_x(x,y)$', r'$|(u^{ex}-u^h)_x(x,y)|$'],
        xx=xx,
        yy=yy,
        gridlines_x1=None,
        gridlines_x2=None,
    )

    my_small_plot(
        title=r'approximation of solution $u$, $y$ component',
        vals=[E_y_vals, Eh_y_vals, E_y_err],
        titles=[r'$u^{ex}_y(x,y)$', r'$u^h_y(x,y)$', r'$|(u^{ex}-u^h)_y(x,y)|$'],
        xx=xx,
        yy=yy,
        gridlines_x1=None,
        gridlines_x2=None,
    )

