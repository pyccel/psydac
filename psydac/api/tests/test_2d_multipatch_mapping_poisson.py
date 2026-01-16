#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

import pytest
import numpy as np
from mpi4py import MPI
from sympy  import pi, sin

from sympde.calculus      import grad, dot
from sympde.calculus      import minus, plus
from sympde.topology      import ScalarFunctionSpace
from sympde.topology      import elements_of
from sympde.topology      import NormalVector, Union
from sympde.topology      import Square, Domain
from sympde.topology      import IdentityMapping, PolarMapping, AffineMapping
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.expr     import Norm, SemiNorm
from sympde.expr.equation import find, EssentialBC

from psydac.api.discretization     import discretize
from psydac.api.tests.build_domain import build_pretzel
from psydac.api.settings           import PSYDAC_BACKEND_GPYCCEL
from psydac.fem.plotting_utilities import plot_field_2d as plot_field

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

#==============================================================================
def run_poisson_2d(solution, f, domain, ncells=None, degree=None, filename=None, comm=None, backend=None):

    if filename is None:
        assert ncells is not None
        assert degree is not None

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V   = ScalarFunctionSpace('V', domain, kind=None)

    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')

    bc   = EssentialBC(u, solution, domain.boundary)

    error  = u - solution

    I = domain.interfaces
    boundary = domain.boundary

    kappa  = 10**3

    expr_I =- 0.5*dot(grad(plus(u)),nn)*minus(v)  + 0.5*dot(grad(minus(v)),nn)*plus(u)  - kappa*plus(u)*minus(v)\
            + 0.5*dot(grad(minus(u)),nn)*plus(v)  - 0.5*dot(grad(plus(v)),nn)*minus(u)  - kappa*plus(v)*minus(u)\
            - 0.5*dot(grad(minus(v)),nn)*minus(u) - 0.5*dot(grad(minus(u)),nn)*minus(v) + kappa*minus(u)*minus(v)\
            + 0.5*dot(grad(plus(v)),nn)*plus(u)   + 0.5*dot(grad(plus(u)),nn)*plus(v)   + kappa*plus(u)*plus(v)

    expr_b = -dot(grad(u),nn)*v -dot(grad(v),nn)*u + kappa*u*v
    expr   = dot(grad(u),grad(v))

    a = BilinearForm((u,v),  integral(domain, expr)+ integral(I, expr_I) + integral(boundary, expr_b))
    l = LinearForm(v, integral(domain, f*v)+ integral(boundary, -dot(grad(v),nn)*solution + kappa*solution*v))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))

    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    if filename is None:
        domain_h = discretize(domain, ncells=ncells, comm=comm)
        Vh       = discretize(V, domain_h, degree=degree)
    else:
        domain_h = discretize(domain, filename=filename, comm=comm)
        Vh       = discretize(V, domain_h)

    equation_h = discretize(equation, domain_h, [Vh, Vh], backend=backend)

    l2norm_h = discretize(l2norm, domain_h, Vh, backend=backend)
    h1norm_h = discretize(h1norm, domain_h, Vh, backend=backend)

    uh = equation_h.solve()

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error, uh

#------------------------------------------------------------------------------
def test_poisson_2d_2_patches_dirichlet_0():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    connectivity = [((0,1,1),(1,1,-1))]
    patches = [D1,D2]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y = domain.coordinates
    solution = x**2 + y**2
    f        = -4

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**2, 2**2], degree=[2,2])

    expected_l2_error = 6.223948817460227e-09
    expected_h1_error = 8.184613465986152e-09

    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patches_dirichlet_1():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    connectivity = [((0,1,1),(1,1,-1))]
    patches = [D1,D2]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error = 0.0010196321182729799
    expected_h1_error = 0.03754732062999426

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patches_dirichlet_2():

    mapping_1 = IdentityMapping('M1', 2)
    mapping_2 = PolarMapping   ('M2', 2, c1 = 0., c2 = 0.5, rmin = 0., rmax=1.)
    mapping_3 = AffineMapping  ('M3', 2, c1 = 0., c2 = np.pi, a11 = -1, a22 = -1, a21 = 0, a12 = 0)

    A = Square('A',bounds1=(0.5, 1.), bounds2=(-1., 0.5))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, np.pi))
    C = Square('C',bounds1=(0.5, 1.), bounds2=(np.pi-0.5, np.pi + 1))

    D1     = mapping_1(A)
    D2     = mapping_2(B)
    D3     = mapping_3(C)

    connectivity = [((0,1,1),(1,1,-1)), ((1,1,1),(2,1,-1))]
    patches = [D1, D2, D3]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    plot_fn=f'uh_multipatch_poisson.pdf'
    plot_field(fem_field=uh, Vh=uh.space, domain=domain, title='uh', filename=plot_fn, hide_plot=True)

    expected_l2_error = 0.0019402242901236006
    expected_h1_error = 0.024759527393621895

    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patches_dirichlet_3():

    domain    = build_pretzel()
    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.009824734742537082
    expected_h1_error = 0.10615177751279731

    assert ( abs(l2_error - expected_l2_error) < 1e-7)
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patches_dirichlet_4():

    filename = os.path.join(mesh_dir, 'multipatch/square.h5')
    domain   = Domain.from_file(filename)

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, filename=filename)

    expected_l2_error = 0.0009564642390937873
    expected_h1_error = 0.03537252217007516

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL])
def test_poisson_2d_2_patches_dirichlet_5(backend):

    filename = os.path.join(mesh_dir, 'multipatch/square_repeated_knots.h5')
    domain   = Domain.from_file(filename)

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, filename=filename, backend=backend)

    expected_l2_error = 1.429395234681141e-05
    expected_h1_error = 0.0007612676504978289

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patches_dirichlet_6():

    filename = os.path.join(mesh_dir, 'multipatch/magnet.h5')
    domain   = Domain.from_file(filename)

    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, filename=filename)

    expected_l2_error = 0.0005134739232637597
    expected_h1_error = 0.011045374959672699

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

def test_poisson_2d_4_patch_dirichlet_0():

    A = Square('A',bounds1=(0.2, 0.6), bounds2=(0, np.pi))
    B = Square('B',bounds1=(0.2, 0.6), bounds2=(np.pi, 2*np.pi))
    C = Square('C',bounds1=(0.6, 1.), bounds2=(0, np.pi))
    D = Square('D',bounds1=(0.6, 1.), bounds2=(np.pi, 2*np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_3 = PolarMapping('M3',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_4 = PolarMapping('M4',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)
    D3     = mapping_3(C)
    D4     = mapping_4(D)

    connectivity = [((0,1,1),(1,1,-1)), ((2,1,1),(3,1,-1)), ((0,0,1),(2,0,-1)),((1,0,1),(3,0,-1))]
    patches = [D1, D2, D3, D4]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    interiors = domain.interior.args
    ncells                    = {}
    ncells[interiors[0].name] = [2**2,2**2]
    ncells[interiors[1].name] = [2**3,2**3]
    ncells[interiors[2].name] = [2**3,2**3]
    ncells[interiors[3].name] = [2**2,2**2]

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=ncells, degree=[2,2])

    expected_l2_error = 1.7195248903000171e-09
    expected_h1_error = 5.959066397620133e-08

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

###############################################################################
#            PARALLEL TESTS
###############################################################################

#==============================================================================
@pytest.mark.mpi
def test_poisson_2d_2_patches_dirichlet_parallel_0():

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    connectivity = [((0,1,1),(1,1,-1))]
    patches = [D1, D2]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, ncells=[2**3,2**3], degree=[2,2],
                                        comm=MPI.COMM_WORLD)

    expected_l2_error = 0.0010196321182729799
    expected_h1_error = 0.03754732062999426

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
@pytest.mark.mpi
def test_poisson_2d_4_patches_dirichlet_parallel_0():

    filename = os.path.join(mesh_dir, 'multipatch/magnet.h5')
    domain   = Domain.from_file(filename)

    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, filename=filename, comm=MPI.COMM_WORLD)

    expected_l2_error = 0.0005134739232637597
    expected_h1_error = 0.011045374959672699

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
@pytest.mark.mpi
@pytest.mark.parametrize('backend',  [None, PSYDAC_BACKEND_GPYCCEL])
def test_poisson_2d_2_patches_dirichlet_parallel_1(backend):

    filename = os.path.join(mesh_dir, 'multipatch/square_repeated_knots.h5')
    domain   = Domain.from_file(filename)

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error, uh = run_poisson_2d(solution, f, domain, filename=filename, backend=backend)

    expected_l2_error = 1.429395234681141e-05
    expected_h1_error = 0.0007612676504978289

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

    from psydac.fem.plotting_utilities import get_plotting_grid, get_grid_vals
    from psydac.fem.plotting_utilities import get_patch_knots_gridlines, my_small_plot
    from collections                               import OrderedDict

    domain    = build_pretzel()
    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    l2_error, h1_error, u_h = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])

    mappings_list = list(mappings.values())
    mappings_list = [mapping.get_callable_mapping() for mapping in mappings_list]

    from sympy import lambdify
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

