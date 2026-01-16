#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path
from collections import OrderedDict

from mpi4py import MPI
import pytest
import numpy as np
from sympy import pi, cos, sin, symbols, conjugate, exp
from sympy import Tuple, Matrix
from sympy import lambdify

from sympde.calculus import grad, dot, cross, curl
from sympde.calculus import minus, plus
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import NormalVector
from sympde.topology import Union
from sympde.topology import Domain, Square
from sympde.topology import IdentityMapping, AffineMapping, PolarMapping
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm, SemiNorm
from sympde.expr     import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

#==============================================================================
def get_boundaries(*args):

    if not args:
        return ()
    else:
        assert all(1 <= a <= 4 for a in args)
        assert len(set(args)) == len(args)

    boundaries = {1: {'axis': 0, 'ext': -1},
                  2: {'axis': 0, 'ext':  1},
                  3: {'axis': 1, 'ext': -1},
                  4: {'axis': 1, 'ext':  1}}

    return tuple(boundaries[i] for i in args)

#==============================================================================
def run_biharmonic_2d_dir(solution, f, dir_zero_boundary, ncells=None, degree=None, backend=None, comm=None, filename=None):

    assert isinstance(dir_zero_boundary, (list, tuple))

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    if filename:
        domain = Domain.from_file(filename)
    else:
        domain = Square()

    B_dirichlet_0 = Union(*[domain.get_boundary(**kw) for kw in dir_zero_boundary])

    V  = ScalarFunctionSpace('V', domain)
    V.codomain_type = 'complex'
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    nn = NormalVector('nn')

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, laplace(u) * laplace(v)))

    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, f * v))

    # Essential boundary conditions
    dn = lambda a: dot(grad(a), nn)
    bc = []
    bc += [EssentialBC(   u , 0, B_dirichlet_0)]
    bc += [EssentialBC(dn(u), 0, B_dirichlet_0)]

    # Variational model
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    # Error norms
    error  = u - solution
    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')
    h2norm = SemiNorm(error, domain, kind='h2')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    if filename:
        domain_h = discretize(domain, filename=filename, comm=comm)
    else:
        domain_h = discretize(domain, ncells=ncells, comm=comm)

    # Discrete spaces
    if filename:
        Vh = discretize(V, domain_h)
    else:
        Vh = discretize(V, domain_h, degree=degree)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    # Discretize error norms
    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)
    h2norm_h = discretize(h2norm, domain_h, Vh, backend=backend)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    equation_h.set_solver('bicg', tol=1e-9)
    uh = equation_h.solve()

    # Compute error norms
    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)
    h2_error = h2norm_h.assemble(u=uh)

    return l2_error, h1_error, h2_error

#==============================================================================
def run_poisson_2d(solution, f, domain, ncells=None, degree=None, filename=None, backend=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V = ScalarFunctionSpace('V', domain, kind=None)
    V.codomain_type='complex'

    u = element_of(V, name='u')
    v = element_of(V, name='v')

    nn = NormalVector('nn')

    bc = EssentialBC(u, 0, domain.boundary)

    error = u - solution

    I = domain.interfaces

    kappa = 10**3

    #expr_I =- dot(grad(plus(u)),nn)*minus(v)  + dot(grad(minus(v)),nn)*plus(u) - kappa*plus(u)*minus(v)\
    #        + dot(grad(minus(u)),nn)*plus(v)  - dot(grad(plus(v)),nn)*minus(u) - kappa*plus(v)*minus(u)\
    #        - dot(grad(plus(v)),nn)*plus(u)   + kappa*plus(u)*plus(v)\
    #        - dot(grad(minus(v)),nn)*minus(u) + kappa*minus(u)*minus(v)

    expr_I =- 0.5*dot(grad( plus(u)), nn) * minus(v) + 0.5*dot(grad(minus(v)), nn) *  plus(u) - kappa * plus(u)*minus(v)\
            + 0.5*dot(grad(minus(u)), nn) *  plus(v) - 0.5*dot(grad( plus(v)), nn) * minus(u) - kappa * plus(v)*minus(u)\
            - 0.5*dot(grad(minus(v)), nn) * minus(u) - 0.5*dot(grad(minus(u)), nn) * minus(v) + kappa *minus(u)*minus(v)\
            + 0.5*dot(grad( plus(v)), nn) *  plus(u) + 0.5*dot(grad( plus(u)), nn) *  plus(v) + kappa * plus(u)* plus(v)

    expr = dot(grad(u), grad(v))

    a = BilinearForm((u, v), integral(domain, expr) + integral(I, expr_I))
    l = LinearForm(v, integral(domain, f*v))

    equation = find(u, forall=v, lhs=1j*a(u,v), rhs=1j*l(v), bc=bc)

    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    if filename is None:
        domain_h = discretize(domain, ncells=ncells)
        Vh       = discretize(V, domain_h, degree=degree)
    else:
        domain_h = discretize(domain, filename=filename)
        Vh       = discretize(V, domain_h)

    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)

    equation_h.set_solver('bicg')
    uh = equation_h.solve()

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error

#==============================================================================
def run_helmholtz_2d(solution, kappa, e_w_0, dx_e_w_0, domain, ncells=None, degree=None, backend=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V   = ScalarFunctionSpace('V', domain, kind=None)
    V.codomain_type='complex'

    u = element_of(V, name='u')
    v = element_of(V, name='v')

    error  = u - solution

    expr   = dot(grad(u),grad(v)) - 2 * kappa ** 2 * u * v
    boundary_expr = - 1j * kappa * u * v
    x_boundary = Union(domain.get_boundary(axis=0, ext=-1), domain.get_boundary(axis=0, ext=1))

    boundary_source_expr = - dx_e_w_0 * v - 1j * kappa * e_w_0 * v    

    a = BilinearForm((u, v), integral(domain, expr) + integral(x_boundary, boundary_expr))
    l = LinearForm(v, integral(domain.get_boundary(axis=0, ext=-1), boundary_source_expr))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))

    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, periodic=[False, True])
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)

    equation_h.set_solver('gmres')
    uh = equation_h.solve()

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error, uh

#==============================================================================
def run_maxwell_2d(uex, f, alpha, domain, *, ncells=None, degree=None, filename=None, comm=None):

    if filename is None:
        assert ncells is not None
        assert degree is not None

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    V  = VectorFunctionSpace('V', domain, kind='hcurl')
    V.codomain_type = 'complex'

    u, v, F = elements_of(V, names='u, v, F')
    nn      = NormalVector('nn')

    error   = Matrix([F[0]-uex[0], F[1]-uex[1]])

    boundary = domain.boundary
    I        = domain.interfaces

    kappa = 10*ncells[0] if ncells else 100
    k     = 1

    jump = lambda w:  plus(w) - minus(w)
    avr  = lambda w: (plus(w) + minus(w))/2

    expr1_I  =  cross(nn, jump(v))*curl(avr(u))\
               +k*cross(nn, jump(u))*curl(avr(v))\
               +kappa*cross(nn, jump(u))*cross(nn, jump(v))

    expr1   = curl(u)*curl(v) + alpha*dot(u,v)
    expr1_b = -cross(nn, v) * curl(u) -k*cross(nn, u) * curl(v)  + kappa * cross(nn, u) * cross(nn, v)

    expr2   = dot(f,v)
    expr2_b = -k*cross(nn, uex)*curl(v) + kappa * cross(nn, uex) * cross(nn, v)

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, expr1) + integral(boundary, expr1_b) + integral(I, expr1_I))

    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, expr2) + integral(boundary, expr2_b))

    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v))

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

    equation_h.set_solver('bicg', tol=1e-8)

    uh = equation_h.solve()
    l2_error = l2norm_h.assemble(F=uh)

    return l2_error, uh

###############################################################################
#            SERIAL TESTS
###############################################################################
def test_complex_biharmonic_2d():
    # This test solve the biharmonic problem with homogeneous dirichlet condition without a mapping
    x, y, z = symbols('x1, x2, x3', real=True)
    solution = (sin(pi * x)**2 * sin(pi * y)**2 + 1j * sin(2*pi * x)**2 * sin(2*pi * y)**2) * exp(pi * 1j * (x**2+y**2))
    f        = laplace(laplace(solution))

    dir_zero_boundary = get_boundaries(1, 2, 3, 4)

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(solution, f,
            dir_zero_boundary, ncells=[2**3, 2**3], degree=[3, 3])

    expected_l2_error = 0.0027365784556742626
    expected_h1_error = 0.07976499145119309
    expected_h2_error = 1.701552032688161

    assert abs(l2_error - expected_l2_error) < 1.e-7
    assert abs(h1_error - expected_h1_error) < 1.e-7
    assert abs(h2_error - expected_h2_error) < 1.e-7


def test_complex_biharmonic_2d_mapping():
    # This test solve the biharmonic problem with homogeneous dirichlet condition with a mapping

    x, y, z = symbols('x, y, z', real=True)
    filename = os.path.join(mesh_dir, 'collela_2d.h5')

    factor=2.5
    solution = factor * (cos(1) + sin(1) * 1j) * (cos(pi*x/2)*cos(pi*y/2))**2
    f        = laplace(laplace(solution))

    dir_zero_boundary    = get_boundaries(1, 2, 3, 4)

    l2_error, h1_error, h2_error = run_biharmonic_2d_dir(solution, f,
            dir_zero_boundary, filename=filename)

    expected_l2_error = 0.10977627980052021
    expected_h1_error = 0.32254511059711766
    expected_h2_error = 1.87205519824758

    assert abs(l2_error/factor - expected_l2_error) < 1.e-7
    assert abs(h1_error/factor - expected_h1_error) < 1.e-7
    assert abs(h2_error/factor - expected_h2_error) < 1.e-7


def test_complex_poisson_2d_multipatch():
    # This test solve the poisson problem with homogeneous dirichlet condition in multipatch case
    A = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, 1))

    domain = Domain.join([A, B], [((0, 0, 1), (1, 0, -1))], 'domain')

    x, y = domain.coordinates

    solution = (cos(1) + sin(1) * 1j) * x*y*(1-y)*(1-x)
    f        = (-2*x*(x - 1) -2*y*(y - 1))*(cos(1) + sin(1) * 1j)

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2, 2])

    expected_l2_error = 2.176726763610992e-09
    expected_h1_error = 2.9725703533101877e-09

    assert abs(l2_error - expected_l2_error) < 1e-7
    assert abs(h1_error - expected_h1_error) < 1e-7


def test_complex_poisson_2d_multipatch_mapping():
    # This test solve the poisson problem with homogeneous dirichlet condition in multipatch with mapping case
    filename = os.path.join(mesh_dir, 'multipatch/square.h5')
    domain   = Domain.from_file(filename)

    x,y = domain.coordinates
    solution = (cos(1) + sin(1) * 1j) * x*y*(1-y)*(1-x)
    f        = (-2*x*(x - 1) -2*y*(y - 1))*(cos(1) + sin(1) * 1j)

    l2_error, h1_error = run_poisson_2d(solution, f, domain, filename=filename, backend=PSYDAC_BACKEND_GPYCCEL)

    expected_l2_error = 2.176726763610992e-09
    expected_h1_error = 2.9725703533101877e-09

    assert abs(l2_error - expected_l2_error) < 1e-7
    assert abs(h1_error - expected_h1_error) < 1e-7


def test_complex_helmholtz_2d(plot_sol=False):
    # This test solves the homogeneous Helmhotz equation with impedance BC. 
    # In particular, we impose an incoming wave from the left and absorbing boundary conditions at the right.
    # Along y, periodic boundary conditions are considered.
    domain = Square('domain', bounds1=(0, 1), bounds2=(0, 1))

    x, y = domain.coordinates
    kappa = 2*pi
    solution = exp(1j * kappa * x) * sin(kappa * y)
    e_w_0 = sin(kappa * y) # value of incoming wave at x=0, forall y
    dx_e_w_0 = 1j*kappa*sin(kappa * y) # derivative wrt. x of incoming wave at x=0, forall y

    l2_error, h1_error, uh = run_helmholtz_2d(solution, kappa, e_w_0, dx_e_w_0, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.01540947560953227
    expected_h1_error = 0.19040207344639598

    print(f'errors: l2 = {l2_error}, h1 = {h1_error}')
    print('expected errors: l2 = {}, h1 = {}'.format(expected_l2_error, expected_h1_error))
    
    if plot_sol:
        from psydac.fem.plotting_utilities import get_plotting_grid, get_grid_vals
        from psydac.fem.plotting_utilities import get_patch_knots_gridlines, my_small_plot
        from psydac.feec.pull_push                     import pull_2d_h1
        
        Id_mapping = IdentityMapping('M', 2)
        # print(f'domain.interior = {domain.interior}')
        # domain_interior = [domain]
        # print(f'domain.logical_domain = {domain.logical_domain}')
        mappings = OrderedDict([(domain, Id_mapping)])
        mappings_list = [m for m in mappings.values()]
        call_mappings_list = [m.get_callable_mapping() for m in mappings_list]

        uh = [uh]  # single-patch cast as multi-patch solution 

        u   = lambdify(domain.coordinates, solution)
        u_log = [pull_2d_h1(u, f) for f in call_mappings_list]

        etas, xx, yy         = get_plotting_grid(mappings, N=20)
        grid_vals_h1         = lambda v: get_grid_vals(v, etas, mappings_list, space_kind='h1')

        uh_vals = grid_vals_h1(uh)
        u_vals  = grid_vals_h1(u_log)

        u_err   = [(u1 - u2) for u1, u2 in zip(u_vals, uh_vals)]
    
        my_small_plot(
            title=r'Approximation of solution $u$',
            vals=[u_vals, uh_vals, u_err],
            titles=[r'$u^{ex}(x,y)$', r'$u^h(x,y)$', r'$|(u^{ex}-u^h)(x,y)|$'],
            xx=xx,
            yy=yy,
            gridlines_x1=None,
            gridlines_x2=None,
        )

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

def test_maxwell_2d_2_patch_dirichlet_2():
    # This test solve the maxwell problem with non-homogeneous dirichlet condition with penalization on the border of the exact solution
    domain = Square('domain', bounds1=(0, 1), bounds2=(0, 1))
    x,y      = domain.coordinates

    omega = 1.5
    alpha = -1j*omega**2
    Eex   = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f     = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error, Eh = run_maxwell_2d(Eex, f, alpha, domain, ncells=[2**2, 2**2], degree=[2, 2])

    expected_l2_error = 0.012726070686020729

    assert abs(l2_error - expected_l2_error) < 1e-7


@pytest.mark.mpi
def test_maxwell_2d_2_patch_dirichlet_parallel_0():
    # This test solve the maxwell problem with non-homogeneous dirichlet condition with penalization on the border of the exact solution

    bounds1   = (0.5, 1.)
    bounds2_A = (0, np.pi/2)
    bounds2_B = (np.pi/2, np.pi)

    A = Square('A', bounds1=bounds1, bounds2=bounds2_A)
    B = Square('B', bounds1=bounds1, bounds2=bounds2_B)

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1 = mapping_1(A)
    D2 = mapping_2(B)

    domain = Domain.join([D1, D2], [((0, 1, 1), (1, 1, -1))], 'domain')

    x, y = domain.coordinates

    omega = 1.5
    alpha = -1j*omega**2
    Eex   = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f     = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error, Eh = run_maxwell_2d(Eex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2, 2], comm=MPI.COMM_WORLD)

    expected_l2_error = 0.012075890902616281

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

    check = 'complex_helmholtz_2d'

    if check == 'complex_helmholtz_2d':
        test_complex_helmholtz_2d(plot_sol=True)

    else:

        from psydac.fem.plotting_utilities import get_plotting_grid, get_grid_vals
        from psydac.fem.plotting_utilities import get_patch_knots_gridlines, my_small_plot
        from psydac.api.tests.build_domain             import build_pretzel
        from psydac.feec.pull_push                     import pull_2d_hcurl
        
        domain = build_pretzel()
        x,y    = domain.coordinates

        omega = 1.5
        alpha = -omega**2
        Eex   = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
        f     = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                    alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

        l2_error, Eh = run_maxwell_2d(Eex, f, alpha, domain, ncells=[2**2, 2**2], degree=[2,2])

        mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
        mappings_list = list(mappings.values())
        call_mappings_list = [m.get_callable_mapping() for m in mappings_list]

        Eex_x   = lambdify(domain.coordinates, Eex[0])
        Eex_y   = lambdify(domain.coordinates, Eex[1])
        Eex_log = [pull_2d_hcurl([Eex_x,Eex_y], f) for f in call_mappings_list]

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
