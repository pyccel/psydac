# -*- coding: UTF-8 -*-

from mpi4py import MPI
import pytest
import numpy as np
from sympy  import pi, sin

from sympde.calculus      import grad, dot
from sympde.calculus      import minus, plus
from sympde.topology      import ScalarFunctionSpace
from sympde.topology      import elements_of
from sympde.topology      import NormalVector
from sympde.topology      import Square
from sympde.topology      import IdentityMapping, PolarMapping, AffineMapping
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.expr     import Norm
from sympde.expr.equation import find, EssentialBC

from psydac.api.discretization     import discretize
from psydac.api.tests.build_domain import build_pretzel

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

   # expr_I =(
   #         - dot(grad(plus(u)),nn)*minus(v)  + dot(grad(minus(v)),nn)*plus(u) - kappa*plus(u)*minus(v)
   #         + dot(grad(minus(u)),nn)*plus(v)  - dot(grad(plus(v)),nn)*minus(u) - kappa*plus(v)*minus(u)
   #         - dot(grad(plus(v)),nn)*plus(u)   + kappa*plus(u)*plus(v)
   #         - dot(grad(minus(v)),nn)*minus(u) + kappa*minus(u)*minus(v))

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

    uh = equation_h.solve()

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error, uh

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_0():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y = domain.coordinates
    solution = x**2 + y**2
    f        = -4

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**2, 2**2], degree=[2,2])

    expected_l2_error = 6.223948817460227e-09
    expected_h1_error = 8.184613465986152e-09

    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_1():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.017626479960982044
    expected_h1_error = 0.245058938982964

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_2():

    mapping_1 = IdentityMapping('M1', 2)
    mapping_2 = PolarMapping   ('M2', 2, c1 = 0., c2 = 0.5, rmin = 0., rmax=1.)
    mapping_3 = AffineMapping  ('M3', 2, c1 = 0., c2 = np.pi, a11 = -1, a22 = -1, a21 = 0, a12 = 0)

    A = Square('A',bounds1=(0.5, 1.), bounds2=(-1., 0.5))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, np.pi))
    C = Square('C',bounds1=(0.5, 1.), bounds2=(np.pi-0.5, np.pi + 1))

    D1     = mapping_1(A)
    D2     = mapping_2(B)
    D3     = mapping_3(C)

    D1D2      = D1.join(D2, name = 'D1D2',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    domain    = D1D2.join(D3, name = 'D1D2D3',
                bnd_minus = D2.get_boundary(axis=1, ext=1),
                bnd_plus  = D3.get_boundary(axis=1, ext=-1))

    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.0019323697521791872
    expected_h1_error = 0.026714232827734725

    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_3():

    domain    = build_pretzel()
    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.009791660112045008
    expected_h1_error = 0.10671109405333927

    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
@pytest.mark.parallel
def test_poisson_2d_2_patch_dirichlet_parallel_0():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2],
                                        comm=MPI.COMM_WORLD)

    expected_l2_error = 0.017626479960982044
    expected_h1_error = 0.245058938982964

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

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

    from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_grid_vals_scalar
    from psydac.feec.multipatch.plotting_utilities import get_patch_knots_gridlines, my_small_plot
    from collections                               import OrderedDict

    domain    = build_pretzel()
    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error, u_h = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])

    mappings_list = list(mappings.values())

    from sympy import lambdify
    u_ex = lambdify(domain.coordinates, solution)
    f_ex = lambdify(domain.coordinates, f)
    F    = [f.get_callable_mapping() for f in mappings_list]

    u_ex_log = [lambda xi1, xi2,ff=f : u_ex(*ff(xi1,xi2)) for f in F]

    N=20
    etas, xx, yy = get_plotting_grid(mappings, N)
    gridlines_x1, gridlines_x2 = get_patch_knots_gridlines(u_h.space, N, mappings, plotted_patch=1)

    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')

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

