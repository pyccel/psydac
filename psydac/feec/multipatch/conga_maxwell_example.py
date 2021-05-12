# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from sympy import pi, cos, sin, Matrix, Tuple, Max, exp
from sympy import symbols
from sympy import lambdify

from sympde.calculus import grad, dot, inner, rot, div, curl, cross
from sympde.topology import NormalVector
from sympde.expr import Norm

from sympde.topology import Derham
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.feec.multipatch.api import discretize

from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.feec.pull_push     import push_2d_hcurl, pull_2d_hcurl

from psydac.utilities.utils    import refine_array_1d

from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, ConformingProjection_V1 #ortho_proj_Hcurl
from psydac.feec.multipatch.operators import time_count
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, my_small_plot, my_small_streamplot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

comm = MPI.COMM_WORLD

#==============================================================================
def run_conga_maxwell_2d(uex, f, alpha, domain, ncells, degree, comm=None, return_sol=False):
    """
    - assemble and solve a Maxwell problem on a multipatch domain, using Conga approach
    - this problem is adapted from the single patch test_api_system_3_2d_dir_1
    """
    hom_bc = (uex is None)
    use_scipy = True
    maxwell_tol = 5e-3
    nquads = [d + 1 for d in degree]

    # multipatch de Rham sequence:
    t_stamp = time_count()

    print('discretizing the domain with ncells = '+repr(ncells)+'...' )
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    t_stamp = time_count(t_stamp)
    print('creating de Rham seq...' )
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS['numba'])
    V1h = derham_h.V1
    V2h = derham_h.V2

    DEBUG_f = False
    if DEBUG_f:
        print('assembling rhs DEBUG...' )
        u, v, F  = elements_of(V1h.symbolic_space, names='u, v, F')

        print( type(f) )
        print(f)
        expr   = dot(f,v)
        if hom_bc:
            l = LinearForm(v, integral(domain, expr))
        else:
            expr_b = penalization * cross(uex, nn) * cross(v, nn)
            l = LinearForm(v, integral(domain, expr) + integral(boundary, expr_b))

        lh = discretize(l, domain_h, V1h, backend=PSYDAC_BACKENDS['numba'])
        b  = lh.assemble()

        print("end debug")
        exit()

    t_stamp = time_count(t_stamp)
    print('assembling the mass matrices...' )
    # Mass matrices for broken spaces (block-diagonal)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)
    M2 = BrokenMass(V2h, domain_h, is_scalar=True)

    t_stamp = time_count(t_stamp)
    print('assembling the broken derivatives...' )
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    t_stamp = time_count(t_stamp)
    print('assembling the conf P1 and I1...' )
    cP1 = ConformingProjection_V1(V1h, domain_h, hom_bc=hom_bc)
    I1 = IdLinearOperator(V1h)

    t_stamp = time_count(t_stamp)
    print('getting A operator...' )
    A1 = alpha * M1 + ComposedLinearOperator([I1-cP1, M1, I1-cP1]) + ComposedLinearOperator([cP1, bD1.transpose(), M2, bD1, cP1])

    u, v, F  = elements_of(V1h.symbolic_space, names='u, v, F')

    if not hom_bc:
        # boundary conditions
        # todo: clean the non-homogeneous case
        # u, v, F  = elements_of(V1h.symbolic_space, names='u, v, F')
        nn  = NormalVector('nn')
        penalization = 10**7
        boundary = domain.boundary
        expr_b = penalization * cross(u, nn) * cross(v, nn)
        a_b = BilinearForm((u,v), integral(boundary, expr_b))
        a_b_h = discretize(a_b, domain_h, [V1h, V1h], backend=PSYDAC_BACKENDS['numba'])
        A_b = FemLinearOperator(fem_domain=V1h, fem_codomain=V1h, matrix=a_b_h.assemble())

        A = A1 + A_b
    else:
        A = A1

    t_stamp = time_count(t_stamp)
    print('assembling rhs...' )
    expr   = dot(f,v)
    if hom_bc:
        l = LinearForm(v, integral(domain, expr))
    else:
        expr_b = penalization * cross(uex, nn) * cross(v, nn)
        l = LinearForm(v, integral(domain, expr) + integral(boundary, expr_b))

    lh = discretize(l, domain_h, V1h) #, backend=PSYDAC_BACKENDS['numba'])
    b  = lh.assemble()

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system

    if use_scipy:
        t_stamp = time_count(t_stamp)
        print("getting sparse matrix...")
        A = A.to_sparse_matrix()
        b = b.toarray()     # todo MCP: why not 'to_array', for consistency with array_to_stencil ?

        t_stamp = time_count(t_stamp)
        print("solving with scipy...")
        x        = spsolve(A, b)
        u_coeffs = array_to_stencil(x, V1h.vector_space)

    else:
        t_stamp = time_count(t_stamp)
        print("solving with psydac cg solver...")

        u_coeffs, info = cg( A, b, tol=maxwell_tol, verbose=True )

    uh = FemField(V1h, coeffs=u_coeffs)
    uh = cP1(uh)

    if uex is not None:
        # error
        error       = Matrix([F[0]-uex[0],F[1]-uex[1]])
        l2_norm     = Norm(error, domain, kind='l2')
        l2_norm_h   = discretize(l2_norm, domain_h, V1h, backend=PSYDAC_BACKENDS['numba'])
        l2_error    = l2_norm_h.assemble(F=uh)
    else:
        l2_error = None

    return l2_error, uh

def run_maxwell_2d_time_harmonic(test_case='manufactured_sol', nc=4, deg=2):
    """
    curl-curl problem with 0 order term and source
    """

    if test_case == 'manufactured_sol':

        domain_name = 'square'

        n_patches=2
        domain = build_multipatch_domain(domain_name=domain_name, n_patches=n_patches)
        mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
        mappings_list = list(mappings.values())
        x,y    = domain.coordinates

        omega = 1  # ?
        alpha  = -omega**2
        uex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
        f      = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                         alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))
        uex_x = lambdify(domain.coordinates, uex[0])
        uex_y = lambdify(domain.coordinates, uex[1])
        uex_log = [pull_2d_hcurl([uex_x,uex_y], f) for f in mappings_list]

    elif test_case == 'pretzel_J':

        r_min = 1
        r_max = 2
        domain_name = 'pretzel'
        # domain_name = 'pretzel_annulus'

        domain = build_multipatch_domain(domain_name=domain_name, r_min=r_min, r_max=r_max)
        mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
        mappings_list = list(mappings.values())
        x,y    = domain.coordinates

        omega = 4  # ?
        alpha  = -omega**2
        # 'rotating' (divergence-free) J field:
        #   J = j(r) * (-sin theta, cos theta)  ___?
        # with j(r) = max(dr-(r-r0)**2,0): supported in [r0-dr, r0+dr]
        r0 = 2.1
        dr = 0.1
        y0 = 0.5
        ax = 2.6/r0

        # gives an error:
        # NotImplementedError: Cannot translate to Sympy:
        # Max(0, 0.01 - 4.41*(0.476190476190476*((1.0*x1*sin(x2) + 0.5)**2 + 0.652366863905326*(1.0*x1*cos(x2) + 1)**2)**0.5 - 1)**2)
        # J_x = -(y-y0) * Max(dr**2 - (((x/ax)**2 + (y-y0)**2)**.5-r0)**2, 0)   # /(x**2 + y**2)
        # J_y =  (x/ax) * Max(dr**2 - (((x/ax)**2 + (y-y0)**2)**.5-r0)**2, 0)

        # also gives an error:
        # "numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
        # NameError: name 'sqrt' is not defined"
        # J_x = -(y-y0) * exp( - ((( (x/ax)**2 + (y-y0)**2 )**.5-r0 )/dr)**2 )   # /(x**2 + y**2)
        # J_y =  (x/ax) * exp( - ((((x/ax)**2 + (y-y0)**2)**.5-r0)/dr)**2 )

        J_x = -(y-y0) * exp( - (( (x/ax)**2 + (y-y0)**2 - r0**2 )/dr)**2 )   # /(x**2 + y**2)
        J_y =  (x/ax) * exp( - (( (x/ax)**2 + (y-y0)**2 - r0**2 )/dr)**2 )
        f = Tuple(J_x, J_y)


        uex = None

    else:
        raise NotImplementedError


    vis_J = False
    if vis_J:

        nc = 2**5
        ncells=[nc, nc]
        degree=[2,2]

        # for plotting J:
        lamJ_x   = lambdify(domain.coordinates, J_x)
        lamJ_y   = lambdify(domain.coordinates, J_y)
        J_log = [pull_2d_hcurl([lamJ_x,lamJ_y], M) for M in mappings_list]

        mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
        mappings_list = list(mappings.values())
        x,y    = domain.coordinates
        nquads = [d + 1 for d in degree]

        # for plotting
        etas, xx, yy = get_plotting_grid(mappings, N=40)
        grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
        grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

        # multipatch de Rham sequence:
        derham  = Derham(domain, ["H1", "Hcurl", "L2"])

        domain_h = discretize(domain, ncells=ncells, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree)
        # V1h = derham_h.V1
        # V2h = derham_h.V2

        print("assembling projection operators...")
        P0, P1, P2 = derham_h.projectors(nquads=nquads)

        J = P1(J_log)

        J_x_vals, J_y_vals = grid_vals_hcurl(J)

        my_small_plot(
            title=r'diverging harmonic field and Conga curl',
            vals=[np.abs(J_x_vals), np.abs(J_y_vals)],
            titles=[r'$|J_x|$', r'$|J_y|$'],  # , r'$div_h J$' ],
            surface_plot=True,
            xx=xx, yy=yy,
        )

        my_small_streamplot(
            title=('J'),
            vals_x=J_x_vals,
            vals_y=J_y_vals,
            xx=xx,
            yy=yy,
            amplification=.5, #20,
        )

        exit()


    # f      = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
    #                  alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    ## call solver
    l2_error, uh = run_conga_maxwell_2d(uex, f, alpha, domain, ncells=[nc, nc], degree=[deg,deg], return_sol=True)

    # else:
    #     # Nitsche
    #     l2_error, uh = run_system_3_2d_dir(uex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2,2], return_sol=True)

    if uex:
        print("max2d: ", l2_error)


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # VISUALIZATION
    #   adapted from examples/poisson_2d_multi_patch.py and
    #   and psydac/api/tests/test_api_feec_2d.py
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    N=20
    etas, xx, yy = get_plotting_grid(mappings, N)
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')
    gridlines_x1 = None
    gridlines_x2 = None

    uh_x_vals, uh_y_vals = grid_vals_hcurl(uh)
    if uex:
        u_x_vals, u_y_vals   = grid_vals_hcurl(uex_log)
        u_x_err = [abs(u1 - u2) for u1, u2 in zip(u_x_vals, uh_x_vals)]
        u_y_err = [abs(u1 - u2) for u1, u2 in zip(u_y_vals, uh_y_vals)]
        # u_x_err = abs(u_x_vals - uh_x_vals)
        # u_y_err = abs(u_y_vals - uh_y_vals)

        my_small_plot(
            title=r'approximation of solution $u$, $x$ component',
            vals=[u_x_vals, uh_x_vals, u_x_err],
            titles=[r'$u^{ex}_x(x,y)$', r'$u^h_x(x,y)$', r'$|(u^{ex}-u^h)_x(x,y)|$'],
            xx=xx,
            yy=yy,
            gridlines_x1=gridlines_x1,
            gridlines_x2=gridlines_x2,
        )

        my_small_plot(
            title=r'approximation of solution $u$, $y$ component',
            vals=[u_y_vals, uh_y_vals, u_y_err],
            titles=[r'$u^{ex}_y(x,y)$', r'$u^h_y(x,y)$', r'$|(u^{ex}-u^h)_y(x,y)|$'],
            xx=xx,
            yy=yy,
            gridlines_x1=gridlines_x1,
            gridlines_x2=gridlines_x2,
        )
    else:

        my_small_plot(
            title=r'approximate solution $u$',
            vals=[uh_x_vals, uh_y_vals],
            titles=[r'$u^h_x(x,y)$', r'$u^h_y(x,y)$'],
            xx=xx,
            yy=yy,
            gridlines_x1=gridlines_x1,
            gridlines_x2=gridlines_x2,
        )

    my_small_streamplot(
        title=('solution'),
        vals_x=uh_x_vals,
        vals_y=uh_y_vals,
        xx=xx,
        yy=yy,
        amplification=2, #20,
    )


    # todo:
    # - compute pretzel eigenvalues
    # - use sparse matrices (faster ?)
    # - see whether solution is orthogonal to harmonic forms ? (why should it be ?)
    # - add orthogonality constraint ?
    # - check 0 divergence property of Jh and Eh ?
    #


if __name__ == '__main__':

    nc = 2**6
    deg = 2

    run_maxwell_2d_time_harmonic(test_case='pretzel_J', nc=nc, deg=deg)



