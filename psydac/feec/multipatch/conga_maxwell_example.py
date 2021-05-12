# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import os
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import save_npz, load_npz

from sympy import pi, cos, sin, Matrix, Tuple, Max, exp
from sympy import symbols
from sympy import lambdify

from sympde.expr     import TerminalExpr
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
def run_conga_maxwell_2d(uex, f, alpha, domain, ncells, degree, gamma_jump=1, save_dir=None, load_dir=None, comm=None, return_sol=False):
    """
    - assemble and solve a Maxwell problem on a multipatch domain, using Conga approach
    - this problem is adapted from the single patch test_api_system_3_2d_dir_1
    """
    print("Running Maxwell source problem solver.")
    if load_dir:
        print(" -- will load matrices from " + load_dir)

    hom_bc = (uex is None)
    use_scipy = True
    maxwell_tol = 5e-3
    nquads = [d + 1 for d in degree]

    t_stamp = time_count()
    print('preparing data for plotting...' )
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    # x,y    = domain.coordinates
    nquads = [d + 1 for d in degree]
    etas, xx, yy = get_plotting_grid(mappings, N=40)
    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

    # multipatch de Rham sequence:
    t_stamp = time_count(t_stamp)
    print('discretizing the domain with ncells = '+repr(ncells)+'...' )
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    t_stamp = time_count(t_stamp)
    print('creating de Rham seq...' )
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree) #, backend=PSYDAC_BACKENDS['numba'])
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    t_stamp = time_count(t_stamp)
    if load_dir:
        print("loading sparse matrices...")
        M0_m = load_npz(load_dir+'M0_m.npz')
        M1_m = load_npz(load_dir+'M1_m.npz')
        M2_m = load_npz(load_dir+'M2_m.npz')
        M0_minv = load_npz(load_dir+'M0_minv.npz')
        cP0_m = load_npz(load_dir+'cP0_m.npz')
        cP1_m = load_npz(load_dir+'cP1_m.npz')
        D0_m = load_npz(load_dir+'D0_m.npz')
        D1_m = load_npz(load_dir+'D1_m.npz')
        I1_m = load_npz(load_dir+'I1_m.npz')
        if save_dir:
            print("(warning: save_dir argument is discarded)")
    else:
        print('assembling the mass matrices...' )
        # Mass matrices for broken spaces (block-diagonal)
        M0 = BrokenMass(V0h, domain_h, is_scalar=True)
        M1 = BrokenMass(V1h, domain_h, is_scalar=False)
        M2 = BrokenMass(V2h, domain_h, is_scalar=True)

        t_stamp = time_count(t_stamp)
        print('assembling the broken derivatives...' )
        bD0, bD1 = derham_h.broken_derivatives_as_operators
        t_stamp = time_count(t_stamp)
        print('assembling conf P0, P1 and I1...' )
        cP0 = ConformingProjection_V0(V0h, domain_h, hom_bc=hom_bc)
        cP1 = ConformingProjection_V1(V1h, domain_h, hom_bc=hom_bc)
        I1 = IdLinearOperator(V1h)
        D0 = ComposedLinearOperator([bD0,cP0])
        D1 = ComposedLinearOperator([bD1,cP1])

        t_stamp = time_count(t_stamp)
        print("converting in sparse matrices...")
        M0_m = M0.to_sparse_matrix()
        M1_m = M1.to_sparse_matrix()
        M2_m = M2.to_sparse_matrix()
        cP0_m = cP0.to_sparse_matrix()
        cP1_m = cP1.to_sparse_matrix()
        D0_m = D0.to_sparse_matrix()  # also possible as matrix product bD0 * cP0
        D1_m = D1.to_sparse_matrix()
        I1_m = I1.to_sparse_matrix()

        M0_minv = inv(M0_m.tocsc())  # todo: for large problems, assemble patch-wise M0_inv, as Hodge operator

    t_stamp = time_count(t_stamp)
    print('building A operator...' )
    jump_penal_m = I1_m-cP1_m
    A1_m = ( alpha * M1_m
        + gamma_jump * jump_penal_m.transpose() * M1_m * jump_penal_m
        + D1_m.transpose() * M2_m * D1_m
        )


    # as psydac operator:
    # A1 = (
    #   alpha * M1 + gamma_jump * ComposedLinearOperator([I1-cP1, M1, I1-cP1])
    #   + ComposedLinearOperator([cP1, bD1.transpose(), M2, bD1, cP1])

    # matrix of the weak div operator V1h -> V0h
    div_m = - M0_minv * D0_m.transpose() * M1_m
    def div_norm(u_c):
        du_c = div_m.dot(u_c)
        return np.dot(du_c,M0_m.dot(du_c))

    u, v, F  = elements_of(V1h.symbolic_space, names='u, v, F')

    if not hom_bc:
        raise NotImplementedError
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
        A_m = A1_m

    t_stamp = time_count(t_stamp)
    print('playing with sympy...' )
    # from sympy.vector import directional_derivative
    # from sympde.calculus import Grad, Rot, Curl, Div

    x,y    = domain.coordinates
    # J_x = -y
    # J_y =  x
    #
    # f = Tuple(J_x, J_y)
    # r0 = 2.1
    # dr = 0.1
    # y0 = 0.5
    # ax = 2.6/r0
    # J_x = -(y-y0) * exp( - (( (x/ax)**2 + (y-y0)**2 - r0**2 )/dr)**2 )
    # J_y =  (x/ax) * exp( - (( (x/ax)**2 + (y-y0)**2 - r0**2 )/dr)**2 )
    # J_x = -(y-y0) #* exp( - (( (x/ax)**2 + (y-y0)**2 - r0**2 )/dr)**2 )   # /(x**2 + y**2)
    # J_y =  (x/ax) #* exp( - (( (x/ax)**2 + (y-y0)**2 - r0**2 )/dr)**2 )
    #
    # J_x = x
    # J_y = y

    # J_x = exp( - y**2 )
    # J_y = exp( - x**2 )
    # J_x = - y
    # J_y = x
    # f = Tuple(J_x, J_y)

    # print(type(f))
    # print(f)
    # df = -2 #
    # df = div(f)
    # print(type(df))
    # print(df)

    J_x = x  # - y
    J_y = y
    f = Tuple(J_x, J_y)
    df = div(f)
    phi  = element_of(V0h.symbolic_space, name='phi')
    df_l = LinearForm(phi, integral(domain, df*phi))
    print("TerminalExpr -->")
    print(TerminalExpr(df_l, domain))
    print("<-- ")
    # hm, there seems to be an error in the discretization of the linear form. When projecting a constant, I
    # print("df_l = ", df_l)
    df_lh = discretize(df_l, domain_h, V0h)
    b  = df_lh.assemble()
    b_c = b.toarray()
    dfh_c = M0_minv.dot(b_c)
    dfh = FemField(V0h, coeffs=array_to_stencil(dfh_c, V0h.vector_space))
    dfh_vals = grid_vals_h1(dfh)
    my_small_plot(
        title=r'L2 proj of div f',
        vals=[dfh_vals, np.abs(dfh_vals)],
        titles=[r'$div fh$', r'$|div fh|$'],  # , r'$div_h J$' ],
        surface_plot=True,
        xx=xx, yy=yy,
    )

    exit()


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

    b_c = b.toarray()
    # representation of discrete source:
    fh_c = spsolve(M1_m, b_c)
    print("|| div fh || = ", div_norm(fh_c))
    fh = FemField(V1h, coeffs=array_to_stencil(fh_c, V1h.vector_space))

    plot_fh = True
    if plot_fh:
        fh_x_vals, fh_y_vals = grid_vals_hcurl(fh)

        my_small_plot(
            title=r'discrete source term for Maxwell curl-curl problem',
            vals=[np.abs(fh_x_vals), np.abs(fh_y_vals)],
            titles=[r'$|fh_x|$', r'$|fh_y|$'],  # , r'$div_h J$' ],
            surface_plot=False,
            xx=xx, yy=yy,
        )

    # correct with P1^T
    fh_c = spsolve(M1_m, cP1_m.transpose().dot(b_c))
    print("|| div fh || = ", div_norm(fh_c))
    fh = FemField(V1h, coeffs=array_to_stencil(fh_c, V1h.vector_space))

    if plot_fh:
        fh_x_vals, fh_y_vals = grid_vals_hcurl(fh)

        my_small_plot(
            title=r'discrete corrected source term for Maxwell curl-curl problem',
            vals=[np.abs(fh_x_vals), np.abs(fh_y_vals)],
            titles=[r'$|fh_x|$', r'$|fh_y|$'],  # , r'$div_h J$' ],
            surface_plot=False,
            xx=xx, yy=yy,
        )

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system

    if use_scipy:
        t_stamp = time_count(t_stamp)
        print("getting sparse matrix...")
        # A = A.to_sparse_matrix()
        b = b.toarray()     # todo MCP: why not 'to_array', for consistency with array_to_stencil ?

        t_stamp = time_count(t_stamp)
        print("solving with scipy...")
        Eh_c = spsolve(A_m, b)
        E_coeffs = array_to_stencil(Eh_c, V1h.vector_space)


    else:
        assert not load_dir
        t_stamp = time_count(t_stamp)
        print("solving with psydac cg solver...")

        E_coeffs, info = cg( A, b, tol=maxwell_tol, verbose=True )

    # Eh = FemField(V1h, coeffs=E_coeffs)
    # Eh = cP1(Eh)
    Eh = FemField(V1h, coeffs=array_to_stencil(cP1_m.dot(Eh_c), V1h.vector_space))

    print("|| div Eh || = ", div_norm(Eh_c))
    # divEh_norm = np.dot(divEh_c,M0_m.dot(divEh_c))
    # print("divEh_norm = ", divEh_norm)
    Eh_norm = np.dot(Eh_c,M1_m.dot(Eh_c))
    print("Eh_norm = ", Eh_norm)
    # divEh_c = div_m.dot(Eh_c)
    # divEh = FemField(V0h, coeffs=array_to_stencil(divEh_c, V0h.vector_space))  # to plot maybe

    if uex is not None:
        # error
        error       = Matrix([F[0]-uex[0],F[1]-uex[1]])
        l2_norm     = Norm(error, domain, kind='l2')
        l2_norm_h   = discretize(l2_norm, domain_h, V1h, backend=PSYDAC_BACKENDS['numba'])
        l2_error    = l2_norm_h.assemble(F=Eh)
    else:
        l2_error = None

    return l2_error, Eh









def run_maxwell_2d_time_harmonic(test_case='manufactured_sol', domain_name='square', n_patches=None, nc=4, deg=2, load_dir=None, save_dir=None):
    """
    curl-curl problem with 0 order term and source
    """
    h = 1/nc
    deg = 2
    # jump penalization factor from Buffa, Perugia and Warburton  >> need to study
    gamma_jump = 10*(deg+1)**2/h

    if domain_name == 'square' and n_patches is None:
        n_patches = 2

    if test_case == 'manufactured_sol':

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

    elif test_case == 'circling_J':

        if domain_name == 'pretzel':
            r_min = 1
            r_max = 2
        else:
            r_min = r_max = None

        domain = build_multipatch_domain(domain_name=domain_name, r_min=r_min, r_max=r_max, n_patches=n_patches)
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

        # J_x = -y/(x**2 + y**2)
        # J_y =  x/(x**2 + y**2)

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

    ## call solver
    l2_error, uh = run_conga_maxwell_2d(
        uex, f, alpha, domain, gamma_jump=gamma_jump,
        ncells=[nc, nc], degree=[deg,deg],
        save_dir=save_dir, load_dir=load_dir, return_sol=True
    )

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


    test_case='circling_J'
    n_patches = None

    if test_case=='circling_J':
        domain_name = 'pretzel'
        # domain_name = 'square'
        # n_patches = 6
        # domain_name = 'annulus'
        # n_patches = 4

        nc = 2**4
        deg = 2

    elif test_case == 'manufactured_sol':
        domain_name = 'square'
        n_patches = 6
        nc = 2**4
        deg = 2

    else:
        raise NotImplementedError

    if n_patches:
        np_suffix = '_'+repr(n_patches)
    else:
        np_suffix = ''

    load_dir = './tmp_matrices/'+domain_name+np_suffix+'_nc'+repr(nc)+'_deg'+repr(deg)+'/'
    if load_dir and not os.path.exists(load_dir):
        print("discarding load_dir, since I cannot find it")
        load_dir = None

    save_dir = None  # todo: allow to save matrices

    run_maxwell_2d_time_harmonic(
        test_case=test_case,
        domain_name=domain_name, n_patches=n_patches,
        load_dir=load_dir, save_dir=save_dir, nc=nc, deg=deg
    )



