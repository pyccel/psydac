# -*- coding: UTF-8 -*-

import os
import pytest
import numpy as np
from mpi4py import MPI
from sympy  import pi, sin, cos, Tuple, Matrix

from sympde.calculus      import grad, dot, curl, cross
from sympde.calculus      import minus, plus
from sympde.topology      import VectorFunctionSpace
from sympde.topology      import elements_of
from sympde.topology      import NormalVector
from sympde.topology      import Square, Domain
from sympde.topology      import IdentityMapping, PolarMapping
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.expr     import Norm
from sympde.expr.equation import find, EssentialBC

from psydac.api.discretization       import discretize
from psydac.api.tests.build_domain   import build_pretzel
from psydac.fem.basic                import FemField
from psydac.linalg.iterative_solvers import *
from psydac.api.settings             import PSYDAC_BACKEND_GPYCCEL
from psydac.feec.pull_push           import pull_2d_hcurl

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

#==============================================================================
def run_maxwell_2d(uex, f, alpha, domain, *, ncells=None, degree=None, filename=None, k=None, kappa=None, comm=None):

    if filename is None:
        assert ncells is not None
        assert degree is not None

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V  = VectorFunctionSpace('V', domain, kind='hcurl')

    u, v, F  = elements_of(V, names='u, v, F')
    nn       = NormalVector('nn')

    error   = Matrix([F[0]-uex[0],F[1]-uex[1]])

    I        = domain.interfaces
    boundary = domain.boundary

    kappa   = 10*ncells[0] if ncells else 100
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

    if filename is None:
        domain_h = discretize(domain, ncells=ncells, comm=comm)
        Vh       = discretize(V, domain_h, degree=degree)
    else:
        domain_h = discretize(domain, filename=filename, comm=comm)
        Vh       = discretize(V, domain_h)

    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)
    l2norm_h   = discretize(l2norm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)

    equation_h.set_solver('pcg', pc='jacobi', tol=1e-8)

    uh = equation_h.solve()

    l2_error = l2norm_h.assemble(F=uh)

    return l2_error, uh

#------------------------------------------------------------------------------
def test_maxwell_2d_2_patch_dirichlet_0():

    bounds1   = (0.5, 1.)
    bounds2_A = (0, np.pi/2)
    bounds2_B = (np.pi/2, np.pi)

    A = Square('A',bounds1=bounds1, bounds2=bounds2_A)
    B = Square('B',bounds1=bounds1, bounds2=bounds2_B)

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y    = domain.coordinates

    omega = 1.5
    alpha = -omega**2
    Eex   = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f     = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error, Eh      = run_maxwell_2d(Eex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2,2])

    expected_l2_error = 0.012077019124862177

    assert abs(l2_error - expected_l2_error) < 1e-7

#------------------------------------------------------------------------------
def test_maxwell_2d_2_patch_dirichlet_1():

    domain = build_pretzel()
    x,y    = domain.coordinates

    omega = 1.5
    alpha = -omega**2
    Eex   = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f     = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error, Eh      = run_maxwell_2d(Eex, f, alpha, domain, ncells=[2**2, 2**2], degree=[2,2])

    expected_l2_error = 1.5941322657006822

    assert abs(l2_error - expected_l2_error) < 1e-7

#------------------------------------------------------------------------------
def test_maxwell_2d_2_patch_dirichlet_2():

    filename = os.path.join(mesh_dir, 'multipatch/square_repeated_knots.h5')
    domain   = Domain.from_file(filename)
    x,y      = domain.coordinates

    omega = 1.5
    alpha = -omega**2
    Eex   = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f     = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error, Eh      = run_maxwell_2d(Eex, f, alpha, domain, filename=filename)

    expected_l2_error = 0.00024103192798735168

    assert abs(l2_error - expected_l2_error) < 1e-7


###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
@pytest.mark.parallel
def test_maxwell_2d_2_patch_dirichlet_parallel_0():

    bounds1   = (0.5, 1.)
    bounds2_A = (0, np.pi/2)
    bounds2_B = (np.pi/2, np.pi)

    A = Square('A',bounds1=bounds1, bounds2=bounds2_A)
    B = Square('B',bounds1=bounds1, bounds2=bounds2_B)

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y    = domain.coordinates

    omega = 1.5
    alpha = -omega**2
    Eex   = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f     = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error, Eh      = run_maxwell_2d(Eex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2,2], comm=MPI.COMM_WORLD)

    expected_l2_error = 0.012077019124862177

    assert abs(l2_error - expected_l2_error) < 1e-7
#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

if __name__ == '__main__':

    from collections                               import OrderedDict
    from sympy                                     import lambdify
    from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_grid_vals
    from psydac.feec.multipatch.plotting_utilities import get_patch_knots_gridlines, my_small_plot

    domain    = build_pretzel()
    x,y       = domain.coordinates

    omega = 1.5
    alpha = -omega**2
    Eex   = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f     = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error, Eh = run_maxwell_2d(Eex, f, alpha, domain, ncells=[2**2, 2**2], degree=[2,2])

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    Eex_x   = lambdify(domain.coordinates, Eex[0])
    Eex_y   = lambdify(domain.coordinates, Eex[1])
    Eex_log = [pull_2d_hcurl([Eex_x,Eex_y], f) for f in mappings_list]

    etas, xx, yy         = get_plotting_grid(mappings, N=20)
    grid_vals_hcurl      = lambda v: get_grid_vals(v, etas, mappings_list, space_kind='hcurl')

    Eh_x_vals, Eh_y_vals = grid_vals_hcurl(Eh)
    E_x_vals, E_y_vals   = grid_vals_hcurl(Eex_log)

    E_x_err              = [(u1 - u2) for u1, u2 in zip(E_x_vals, Eh_x_vals)]
    E_y_err              = [(u1 - u2) for u1, u2 in zip(E_y_vals, Eh_y_vals)]

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
