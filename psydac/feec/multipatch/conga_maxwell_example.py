# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import os
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import save_npz, load_npz

from tempfile import TemporaryFile

from sympy import pi, cos, sin, Matrix, Tuple, Max, exp
from sympy import symbols
from sympy import lambdify

from sympde.expr     import TerminalExpr
from sympde.calculus import grad, dot, inner, rot, div, curl, cross
from sympde.calculus import minus, plus
from sympde.topology import NormalVector
from sympde.expr     import Norm

from sympde.topology import Derham
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping
from sympde.topology import VectorFunctionSpace

from sympde.expr.equation import find, EssentialBC

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.feec.multipatch.api import discretize

from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.feec.pull_push     import push_2d_hcurl, pull_2d_hcurl

from psydac.utilities.utils    import refine_array_1d

from psydac.linalg.iterative_solvers import pcg


from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, ConformingProjection_V1, ConformingProjection_V0 #ortho_proj_Hcurl
from psydac.feec.multipatch.operators import time_count
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector, get_grid_quad_weights
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, my_small_plot, my_small_streamplot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

from psydac.feec.multipatch.conga_maxwell_eigenproblem_example import get_fem_name, get_load_dir

comm = MPI.COMM_WORLD

# small helper function (useful ?)
def tmp_plot_source(J_x,J_y, domain):

    nc = 2**5
    ncells=[nc, nc]
    degree=[2,2]

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    # x,y    = domain.coordinates
    lamJ_x   = lambdify(domain.coordinates, J_x)
    lamJ_y   = lambdify(domain.coordinates, J_y)
    J_log = [pull_2d_hcurl([lamJ_x,lamJ_y], M) for M in mappings_list]

    etas, xx, yy = get_plotting_grid(mappings, N=20)

    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    # V1h = derham_h.V1
    # V2h = derham_h.V2

    print("assembling projection operators...")
    nquads = [2*d + 1 for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    J = P1(J_log)

    J_x_vals, J_y_vals = grid_vals_hcurl(J)

    my_small_plot(
        title=r'diverging harmonic field and Conga curl',
        vals=[np.abs(J_x_vals), np.abs(J_y_vals)],
        titles=[r'$|J_x|$', r'$|J_y|$'],  # , r'$div_h J$' ],
        cmap='hsv',
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


#==============================================================================
def run_conga_maxwell_2d(E_ex, f, alpha, domain, ncells, degree, gamma_jump=1, save_dir=None, load_dir=None, comm=None,
                         E_ref_x_vals=None, E_ref_y_vals=None, E_ref_filename='', plot_dir='', fem_name=None, plot_source=False, plot_sol=False,
                         plot_source_div=False, N_diag=20, N_vis=20, return_sol=False):
    """
    - assemble and solve a Maxwell problem on a multipatch domain, using Conga approach
    - this problem is adapted from the single patch test_api_system_3_2d_dir_1
    """
    print("Running Maxwell source problem solver.")
    if load_dir:
        print(" -- will load matrices from " + load_dir)
    elif save_dir:
        print(" -- will save matrices in " + save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    hom_bc = (E_ex is None)
    use_scipy = True
    maxwell_tol = 5e-3
    nquads = [d + 1 for d in degree]

    t_stamp = time_count()
    print('preparing data for plotting...' )
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    # x,y    = domain.coordinates
    nquads = [d + 1 for d in degree]
    etas, xx, yy = get_plotting_grid(mappings, N=N_vis)
    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')
    gridlines_x1 = None  # for plotting a patch grid
    gridlines_x2 = None

    ## DEBUG
    DEBUG_weight = False
    if DEBUG_weight:
        for k in range(3):
            eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
            N0 = eta_1.shape[0]
            N1 = eta_1.shape[1]
            # L0 = M.max_coords[0]-M.min_coords[0]
            # L1 = M.max_coords[1]-M.min_coords[1]
            L0 = etas[k][0][-1]-etas[k][0][0]
            L1 = etas[k][1][-1]-etas[k][1][0]

            print(k, "N0, N1 = ", N0, N1, "L0, L1 = ", L0, L1)

        exit()


    # multipatch de Rham sequence:
    t_stamp = time_count(t_stamp)
    print('discretizing the domain with ncells = '+repr(ncells)+'...' )
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    t_stamp = time_count(t_stamp)
    print('discretizing the de Rham seq with degree = '+repr(degree)+'...' )
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree) #, backend=PSYDAC_BACKENDS['numba'])
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    t_stamp = time_count(t_stamp)
    if load_dir:
        # todo: improve this with small interface: load matrix if present, otherwise assemble it and save
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

        if save_dir:
            t_stamp = time_count(t_stamp)
            print("saving sparse matrices to file...")
            save_npz(save_dir+'M0_m.npz', M0_m)
            save_npz(save_dir+'M1_m.npz', M1_m)
            save_npz(save_dir+'M2_m.npz', M2_m)
            save_npz(save_dir+'M0_minv.npz', M0_minv)
            save_npz(save_dir+'cP0_m.npz', cP0_m)
            save_npz(save_dir+'cP1_m.npz', cP1_m)
            save_npz(save_dir+'D0_m.npz', D0_m)
            save_npz(save_dir+'D1_m.npz', D1_m)
            save_npz(save_dir+'I1_m.npz', I1_m)

    t_stamp = time_count(t_stamp)
    print('building A operator...' )
    jump_penal_m = I1_m-cP1_m

    # matrix of the weak div operator V1h -> V0h
    # div_m = - M0_minv * D0_m.transpose() * M1_m
    div_aux_m = D0_m.transpose() * M1_m
    div_m = - M0_minv * div_aux_m

    # A_m += (
    #         D1_m.transpose() * M2_m * D1_m
    #         + alpha * jump_penal_m.transpose() * M1_m * jump_penal_m
    # )

    # note: with
    #   alpha * cP1_m.transpose() * M1_m * cP1_m
    # for the 0-order term, the solution is conforming with any value of gamma_jump.
    # this may be helful if a strong penalization of the jumps leads to high condition numbers
    # in any case, different penalization options should be compared at numerical level (accuracy, condition numbers...)
    A1_m = ( alpha * M1_m
        + gamma_jump * jump_penal_m.transpose() * M1_m * jump_penal_m
        + D1_m.transpose() * M2_m * D1_m
        )

    L_option = 0   # beware: RHS needs to be changed for full HL operator
    if L_option == 1:
        A1_m += div_aux_m.transpose() * M0_minv * div_aux_m
    elif L_option == 2:
        A1_m += (div_aux_m * cP1_m).transpose() * M0_minv * div_aux_m * cP1_m


    # as psydac operator:
    # A1 = (
    #   alpha * M1 + gamma_jump * ComposedLinearOperator([I1-cP1, M1, I1-cP1])
    #   + ComposedLinearOperator([cP1, bD1.transpose(), M2, bD1, cP1])

    def div_norm(u_c):
        du_c = div_m.dot(u_c)
        return np.dot(du_c,M0_m.dot(du_c))**0.5

    u, v, F  = elements_of(V1h.symbolic_space, names='u, v, F')

    if not hom_bc:
        #raise NotImplementedError
        # boundary conditions
        # todo: clean the non-homogeneous case
        # u, v, F  = elements_of(V1h.symbolic_space, names='u, v, F')
        nn  = NormalVector('nn')
        penalization = 10**7
        boundary = domain.boundary
        expr_b = penalization * cross(u, nn) * cross(v, nn)
        a_b = BilinearForm((u,v), integral(boundary, expr_b))
        a_b_h = discretize(a_b, domain_h, [V1h, V1h])

        A_m = A1_m + a_b_h.assemble().tosparse().tocsr()
    else:
        A_m = A1_m

    if plot_source_div:
        # plotting the projected divergence of f
        div_f = div(f)
        phi  = element_of(V0h.symbolic_space, name='phi')
        df_l = LinearForm(phi, integral(domain, div_f*phi))
        df_lh = discretize(df_l, domain_h, V0h)
        b  = df_lh.assemble()
        b_c = b.toarray()
        dfh_c = M0_minv.dot(b_c)
        dfh = FemField(V0h, coeffs=array_to_stencil(dfh_c, V0h.vector_space))
        dfh_vals = grid_vals_h1(dfh)
        my_small_plot(
            title=r'L2 proj of div f:',
            vals=[dfh_vals],
            titles=[r'div f_h$'],  # , r'$div_h J$' ],
            cmap='hsv',
            surface_plot=False,
            xx=xx, yy=yy,
        )

    t_stamp = time_count(t_stamp)
    print('assembling rhs...' )
    expr   = dot(f,v)
    if hom_bc:
        l = LinearForm(v, integral(domain, expr))
    else:
        expr_b = penalization * cross(E_ex, nn) * cross(v, nn)
        l = LinearForm(v, integral(domain, expr) + integral(boundary, expr_b))

    lh = discretize(l, domain_h, V1h) #, backend=PSYDAC_BACKENDS['numba'])
    b  = lh.assemble()
    b_c = b.toarray()

    # b_c = correct source ?
    print("-- correct source -- ")
    b_c = cP1_m.transpose().dot(b_c)

    if plot_source:
        # representation of discrete source:
        fh_c = spsolve(M1_m.tocsc(), b_c)
        fh_norm = np.dot(fh_c,M1_m.dot(fh_c))**0.5
        print("|| fh || = ", fh_norm)
        print("|| div fh ||/|| fh || = ", div_norm(fh_c)/fh_norm)

        if fem_name:
            fig_name=plot_dir+'Jh.png'  # +'_'+fem_name+'.png'
            fig_name_vf=plot_dir+'Jh_vf.png'   # +'_vf_'+fem_name+'.png'
        else:
            fig_name=None
            fig_name_vf=None

        fh = FemField(V1h, coeffs=array_to_stencil(fh_c, V1h.vector_space))

        fh_x_vals, fh_y_vals = grid_vals_hcurl(fh)
        plot_full_fh=False
        if plot_full_fh:
            div_fh = FemField(V0h, coeffs=array_to_stencil(div_m.dot(fh_c), V0h.vector_space))
            div_fh_vals = grid_vals_h1(div_fh)
            my_small_plot(
                title=r'discrete source term for Maxwell curl-curl problem',
                vals=[np.abs(fh_x_vals), np.abs(fh_y_vals), np.abs(div_fh_vals)],
                titles=[r'$|fh_x|$', r'$|fh_y|$', r'$|div_h fh|$'],  # , r'$div_h J$' ],
                cmap='hsv',
                surface_plot=False,
                xx=xx, yy=yy,
            )
        else:
            abs_fh_vals = [np.sqrt(abs(fx)**2 + abs(fy)**2) for fx, fy in zip(fh_x_vals, fh_y_vals)]
            my_small_plot(
                title=r'source term $J_h$',
                vals=[abs_fh_vals],
                titles=[r'$|J_h|$'],  # , r'$div_h J$' ],
                surface_plot=False,
                xx=xx, yy=yy,
                cmap='plasma',
                dpi=400,
                save_fig=fig_name,
            )

        my_small_streamplot(
            title='source J',
            vals_x=fh_x_vals,
            vals_y=fh_y_vals,
            xx=xx, yy=yy,
            amplification=.05,
            save_fig=fig_name_vf,
        )

        # show source corrected with P1^T  -- this doesn't seem to change much, a bit strange -- need to check
        plot_corrected_f = False
        if plot_corrected_f:
            fh_c = spsolve(M1_m.tocsc(), cP1_m.transpose().dot(b_c))
            print("|| fh || = ", np.dot(fh_c,M1_m.dot(fh_c))**0.5)
            print("|| div fh || = ", div_norm(fh_c))
            fh = FemField(V1h, coeffs=array_to_stencil(fh_c, V1h.vector_space))

            fh_x_vals, fh_y_vals = grid_vals_hcurl(fh)
            if plot_full_fh:
                div_fh = FemField(V0h, coeffs=array_to_stencil(div_m.dot(fh_c), V0h.vector_space))
                div_fh_vals = grid_vals_h1(div_fh)
                my_small_plot(
                    title=r'discrete CORRECTED source term for Maxwell curl-curl problem',
                    vals=[np.abs(fh_x_vals), np.abs(fh_y_vals), np.abs(div_fh_vals)],
                    titles=[r'$|fh_x|$', r'$|fh_y|$', r'$|div_h fh|$'],  # , r'$div_h J$' ],
                    cmap='hsv',
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
        # b = b.toarray()     # why not 'to_array', for consistency with array_to_stencil ?

        t_stamp = time_count(t_stamp)
        print("solving with scipy...")
        Eh_c = spsolve(A_m.tocsc(), b_c)
        E_coeffs = array_to_stencil(Eh_c, V1h.vector_space)

    else:
        assert not load_dir
        t_stamp = time_count(t_stamp)
        print("solving with psydac cg solver...")

        E_coeffs, info = cg( A, b, tol=maxwell_tol, verbose=True )

    # projected solution
    PEh_c = cP1_m.dot(Eh_c)
    jumpEh_c = Eh_c - PEh_c

    Eh = FemField(V1h, coeffs=array_to_stencil(PEh_c, V1h.vector_space))
    Eh_norm = np.dot(Eh_c,M1_m.dot(Eh_c))**0.5
    jumpEh_norm = np.dot(jumpEh_c,M1_m.dot(jumpEh_c))**0.5
    print("|| Eh || = ", Eh_norm)
    print("|| div Eh || / || Eh || = ", div_norm(Eh_c)/Eh_norm)
    print("|| (I-P) Eh || / || Eh || = ", jumpEh_norm/Eh_norm)

    jumps_b = jump_penal_m.dot(b_c)
    b_norm = np.dot(b_c,b_c)**0.5
    jumps_b_norm = np.dot(jumps_b,jumps_b)**0.5
    print("|| (I-P) b || / || b || = ", jumps_b_norm/b_norm)


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # VISUALIZATION and error measure
    #   adapted from examples/poisson_2d_multi_patch.py and
    #   and psydac/api/tests/test_api_feec_2d.py
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # etas, xx, yy = get_plotting_grid(mappings, N_diag, )
    # grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

    Eh_x_vals, Eh_y_vals = grid_vals_hcurl(Eh)
    if E_ex:
        E_ex_x = lambdify(domain.coordinates, E_ex[0])
        E_ex_y = lambdify(domain.coordinates, E_ex[1])
        E_ex_log = [pull_2d_hcurl([E_ex_x,E_ex_y], f) for f in mappings_list]

        E_x_vals, E_y_vals   = grid_vals_hcurl(E_ex_log)
        E_x_err = [abs(u1 - u2) for u1, u2 in zip(E_x_vals, Eh_x_vals)]
        E_y_err = [abs(u1 - u2) for u1, u2 in zip(E_y_vals, Eh_y_vals)]

        my_small_plot(
            title=r'approximation of solution $u$, $x$ component',
            vals=[E_x_vals, Eh_x_vals, E_x_err],
            titles=[r'$u^{ex}_x(x,y)$', r'$u^h_x(x,y)$', r'$|(u^{ex}-u^h)_x(x,y)|$'],
            xx=xx,
            yy=yy,
            gridlines_x1=gridlines_x1,
            gridlines_x2=gridlines_x2,
        )

        my_small_plot(
            title=r'approximation of solution $u$, $y$ component',
            vals=[E_y_vals, Eh_y_vals, E_y_err],
            titles=[r'$u^{ex}_y(x,y)$', r'$u^h_y(x,y)$', r'$|(u^{ex}-u^h)_y(x,y)|$'],
            xx=xx,
            yy=yy,
            gridlines_x1=gridlines_x1,
            gridlines_x2=gridlines_x2,
        )
    else:

        if fem_name:
            fig_name=plot_dir+'Eh.png'  # +'_'+fem_name+'.png'
            fig_name_vf=plot_dir+'Eh_vf.png'   # +'_vf_'+fem_name+'.png'
        else:
            fig_name=None
            fig_name_vf=None

        Eh_abs_vals = [np.sqrt(abs(ex)**2 + abs(ey)**2) for ex, ey in zip(Eh_x_vals, Eh_y_vals)]
        my_small_plot(
            title=r'discrete field $E_h$', # for $\omega = $'+repr(omega),
            vals=[Eh_abs_vals], #[Eh_x_vals, Eh_y_vals, Eh_abs_vals],
            titles=[r'$|E^h|$'], #[r'$E^h_x$', r'$E^h_y$', r'$|E^h|$'],
            xx=xx,
            yy=yy,
            surface_plot=False,
            gridlines_x1=gridlines_x1,
            gridlines_x2=gridlines_x2,
            save_fig=fig_name,
            cmap='hsv',
            dpi = 400,
        )

    my_small_streamplot(
        title=r'discrete field $E_h$',  # for $\omega = $'+repr(omega),
        vals_x=Eh_x_vals,
        vals_y=Eh_y_vals,
        skip=1,
        xx=xx,
        yy=yy,
        amplification=1,
        save_fig=fig_name_vf,
        dpi = 200,
    )

    # measure L2 error
    t_stamp = time_count(t_stamp)
    print("computing L2 error:")
    if E_ex is not None:
        print("computing L2 error with explicit (exact) solution...")
        # error
        error       = Matrix([F[0]-E_ex[0],F[1]-E_ex[1]])
        l2_norm     = Norm(error, domain, kind='l2')
        l2_norm_h   = discretize(l2_norm, domain_h, V1h, backend=PSYDAC_BACKENDS['numba'])
        l2_error     = l2_norm_h.assemble(F=Eh)

    else:
        # approx l2 error measure on diag grid (cell-centered)
        etas_cdiag, xx_cdiag, yy_cdiag, patch_logvols = get_plotting_grid(mappings, N=N_diag, centered_nodes=True, return_patch_logvols=True)
        # grid_vals_h1_cdiag = lambda v: get_grid_vals_scalar(v, etas_cdiag, mappings_list, space_kind='h1')
        grid_vals_hcurl_cdiag = lambda v: get_grid_vals_vector(v, etas_cdiag, mappings_list, space_kind='hcurl')
        Eh_x_vals_cdiag, Eh_y_vals_cdiag = grid_vals_hcurl_cdiag(Eh)
        if E_ref_x_vals is not None:
            print("computing approx l2 error with reference discrete solution on diag grid...")
            assert E_ref_y_vals is not None
            quad_weights = get_grid_quad_weights(etas_cdiag, patch_logvols, mappings_list)
            if fem_name:
                fig_name=plot_dir+'Eh_errors.png'
            else:
                fig_name=None
            Eh_errors_cdiag = [np.sqrt( (u1-v1)**2 + (u2-v2)**2 )
                               for u1, v1, u2, v2 in zip(Eh_x_vals_cdiag, E_ref_x_vals, Eh_y_vals_cdiag, E_ref_y_vals)]
            l2_error = (np.sum([J_F * err**2 for err, J_F in zip(Eh_errors_cdiag, quad_weights)]))**0.5
            my_small_plot(
                title=r'error $|E_h-E^{\rm ref}_h|$', # for $\omega = $'+repr(omega),
                vals=[Eh_errors_cdiag], #[Eh_x_vals, Eh_y_vals, Eh_abs_vals],
                titles=[r'$|E^h|$'], #[r'$E^h_x$', r'$E^h_y$', r'$|E^h|$'],
                xx=xx_cdiag,
                yy=yy_cdiag,
                surface_plot=False,
                gridlines_x1=gridlines_x1,
                gridlines_x2=gridlines_x2,
                save_fig=fig_name,
                cmap='cividis',
                dpi = 400,
            )
        else:
            print("no ref solution to compare with!")
            l2_error = None
            if not os.path.isfile(E_ref_filename):
                print("saving solution values in new file (for future needs)"+E_ref_filename)
                with open(E_ref_filename, 'wb') as f:
                    np.savez(f, x_vals=Eh_x_vals_cdiag, y_vals=Eh_y_vals_cdiag)

    t_stamp = time_count(t_stamp)
    print("done -- summary: ")
    print("using jump penalization factor gamma = ", gamma_jump )
    print('nb of spline cells per patch: ' + repr(ncells))
    print('degree: ' + repr(degree))
    nb_dofs = len(Eh_c)
    print(' -- nb of DOFS (V1h): ' + repr(nb_dofs))
    if l2_error is None :
        print("Sorry, no error to show :( ")
    else:
        print("Measured L2 error: ", l2_error)

    if return_sol:
        return nb_dofs, l2_error, Eh
    else:
        if l2_error is None:
            print("Warning: I have no error and I'm not returning the solution !! ")
        return nb_dofs, l2_error


def run_nitsche_maxwell_2d(E_ex, f, alpha, domain, ncells, degree):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V  = VectorFunctionSpace('V', domain, kind='hcurl')

    u, v, F  = elements_of(V, names='u, v, F')
    nn  = NormalVector('nn')

    error   = Matrix([F[0]-E_ex[0],F[1]-E_ex[1]])

    kappa        = 10**10
    penalization = 10**10

    I        = domain.interfaces
    boundary = domain.boundary

    # Bilinear form a: V x V --> R
    eps     = -1
#    expr_I  = -(0.5*curl(plus(u))*cross(minus(v),nn)       - 0.5*curl(minus(u))*cross(plus(v),nn))\
#             + eps*(0.5*curl(plus(v))*cross(minus(u),nn)    - 0.5*curl(minus(v))*cross(plus(u),nn))\
#             + -kappa*cross(plus(u),nn) *cross(minus(v),nn) - kappa*cross(plus(v),nn) * cross(minus(u),nn)\
#             + -(0.5*curl(minus(u))*cross(minus(v),nn)      + 0.5*curl(plus(u))*cross(plus(v),nn))\
#             + eps*(0.5*curl(minus(v))*cross(minus(u),nn)   + 0.5*curl(plus(v))*cross(plus(u),nn))\
#             + kappa*cross(minus(u),nn)*cross(minus(v),nn)  + kappa*cross(plus(u),nn) *cross(plus(v),nn)

    expr_I  =-kappa*cross(plus(u),nn) *cross(minus(v),nn) - kappa*cross(plus(v),nn) * cross(minus(u),nn)\
            + kappa*cross(minus(u),nn)*cross(minus(v),nn) + kappa*cross(plus(u),nn) *cross(plus(v),nn)

    expr   = curl(u)*curl(v) + alpha*dot(u,v)
    expr_b = penalization * cross(u, nn) * cross(v, nn)

    a = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I) + integral(boundary, expr_b))

    expr   = dot(f,v)
    expr_b = penalization * cross(E_ex, nn) * cross(v, nn)

    l = LinearForm(v, integral(domain, expr) + integral(boundary, expr_b))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))

    l2norm = Norm(error, domain, kind='l2')
    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh])
    l2norm_h   = discretize(l2norm, domain_h, Vh)

    equation_h.assemble()
    
    A = equation_h.linear_system.lhs
    b = equation_h.linear_system.rhs
    
    x, info = pcg(A, b, pc='jacobi', tol=1e-8)

    Eh = FemField( Vh, x )

    l2_error = l2norm_h.assemble(F=Eh)
    ndofs    = Vh.nbasis

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    E_ex_x = lambdify(domain.coordinates, E_ex[0])
    E_ex_y = lambdify(domain.coordinates, E_ex[1])
    E_ex_log = [pull_2d_hcurl([E_ex_x,E_ex_y], f) for f in mappings_list]

    etas, xx, yy = get_plotting_grid(mappings, N=20)
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')
    Eh_x_vals, Eh_y_vals = grid_vals_hcurl(Eh)

    E_x_vals, E_y_vals   = grid_vals_hcurl(E_ex_log)
    E_x_err = [abs(u1 - u2) for u1, u2 in zip(E_x_vals, Eh_x_vals)]
    E_y_err = [abs(u1 - u2) for u1, u2 in zip(E_y_vals, Eh_y_vals)]

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

    return ndofs, l2_error, Eh

def run_maxwell_2d_time_harmonic(nc=None, deg=None, test_case='ring_J',domain_name='pretzel', nitsche_method=False):
    """
    curl-curl problem with 0 order term and source
    """


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # test_case selection with domain
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # test_case='ring_J'
    n_patches = None

    plot_source = True
    plot_sol = True

    if domain_name == 'square':
        n_patches = 6

    # domain_name = 'pretzel'
    # domain_name = 'annulus'; n_patches = 4
    #nc = 16; deg = 3
    # nc = 20; deg = 3
    # nc = 20; deg = 8

    # ref solution (discrete):
    # nc_ref = 8; deg_ref = 3
    nc_ref = 20; deg_ref = 8
    # nc_ref =nc; deg_ref = deg

    N_diag = 100 # diagnostics resolution, per patch
    E_ref_fn = 'E_ref_'+test_case+'_N'+repr(N_diag)+'.npz'
    E_ref_filename = get_load_dir(domain_name=domain_name,n_patches=n_patches,nc=nc_ref,deg=deg_ref,data='solutions')+E_ref_fn
    E_ex = None  # exact solution void by default
    E_ref_x_vals = None
    E_ref_y_vals = None
    if os.path.isfile(E_ref_filename):
        print("getting ref solution values from file "+E_ref_filename)
        with open(E_ref_filename, 'rb') as f:
            E_ref_vals = np.load(f)
            # check form of ref values
            # assert 'x_vals' in E_ref_vals; assert 'y_vals' in E_ref_vals
            E_ref_x_vals = E_ref_vals['x_vals']
            E_ref_y_vals = E_ref_vals['y_vals']
            assert isinstance(E_ref_x_vals, (list, np.ndarray)) and isinstance(E_ref_y_vals, (list, np.ndarray))

    else:
        print("-- ref solution file '"+E_ref_filename+"' does not exist, ...")
        solutions_dir = get_load_dir(domain_name=domain_name,n_patches=n_patches,nc=nc,deg=deg,data='solutions')
        E_ref_filename = solutions_dir+E_ref_fn
        print("... so I will save the present solution instead, in file '"+E_ref_filename+"' --")
        if not os.path.exists(solutions_dir):
            os.makedirs(solutions_dir)

    fem_name = get_fem_name(domain_name=domain_name,n_patches=n_patches,nc=nc,deg=deg)
    save_dir = load_dir = get_load_dir(domain_name=domain_name,n_patches=n_patches,nc=nc,deg=deg,data='matrices')
    if load_dir and not os.path.exists(load_dir+'M0_m.npz'):
        print("discarding load_dir, since I cannot find it (or empty)")
        load_dir = None

    plot_dir = './plots/'+fem_name+'/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    domain = build_multipatch_domain(domain_name=domain_name, n_patches=n_patches)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    x,y    = domain.coordinates


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # source definition
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if test_case == 'manu_sol':
        # use a manufactured solution, with ad-hoc (inhomogeneous) bc

        omega = 1  # ?
        alpha  = -omega**2
        E_ex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
        f      = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                         alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))
        E_ex_x = lambdify(domain.coordinates, E_ex[0])
        E_ex_y = lambdify(domain.coordinates, E_ex[1])
        E_ex_log = [pull_2d_hcurl([E_ex_x,E_ex_y], f) for f in mappings_list]

    elif test_case == 'ring_J':

        # 'rotating' (divergence-free) J field:
        #   J = j(r) * (-sin theta, cos theta)

        if domain_name=='square':
            r0 = np.pi/4
            dr = 0.1
            x0 = np.pi/2
            y0 = np.pi/2
            omega = 43/2
            alpha  = -omega**2  # not a square eigenvalue
            J_factor = 100
        else:
            # for pretzel

            omega = 8  # ?
            alpha  = -omega**2

            source_option = 2

            if source_option==1:
                # big circle:
                r0 = 2.4
                dr = 0.05
                x0 = 0
                y0 = 0.5
                J_factor = 10

            elif source_option==2:
                # small circle in corner:
                r0 = 1
                dr = 0.2
                x0 = 1.5
                y0 = 1.5
                J_factor = 10

            elif source_option==3:
                # small circle in corner, seems less interesting
                r0 = 0.0
                dr = 0.05
                x0 = 0.9
                y0 = 0.9
                J_factor = 10
            else:
                raise NotImplementedError

        # note: some other currents give sympde or numba errors, see below [1]
        J_x = -J_factor * (y-y0) * exp( - .5*(( (x-x0)**2 + (y-y0)**2 - r0**2 )/dr)**2 )   # /(x**2 + y**2)
        J_y =  J_factor * (x-x0) * exp( - .5*(( (x-x0)**2 + (y-y0)**2 - r0**2 )/dr)**2 )

        f = Tuple(J_x, J_y)

        vis_J = False
        if vis_J:
            tmp_plot_source(J_x,J_y, domain)

    else:
        raise NotImplementedError

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # calling solver
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # jump penalization factor from Buffa, Perugia and Warburton  >> need to study
    h = 1/nc
    gamma_jump = 10*(deg+1)**2/h

    if nitsche_method == True:
            ndofs, l2_error, Eh = run_nitsche_maxwell_2d(E_ex, f, alpha, domain, ncells=[nc, nc], degree=[deg,deg])
    else:
        ndofs, l2_error, Eh = run_conga_maxwell_2d(
            E_ex, f, alpha, domain, gamma_jump=gamma_jump,
            ncells=[nc, nc], degree=[deg,deg],
            N_diag=N_diag, E_ref_x_vals=E_ref_x_vals, E_ref_y_vals=E_ref_y_vals, E_ref_filename=E_ref_filename,
            save_dir=save_dir, load_dir=load_dir, return_sol=True,
            plot_source=plot_source, plot_sol=plot_sol, plot_dir=plot_dir, fem_name=fem_name,
        )

    return ndofs, l2_error
 
if __name__ == '__main__':
    results = []
    test_case='manu_sol'
    domain_name='pretzel'
    nitsche_method = True    
    deg = 2
    for nc in [2**3]: # [4, 8, 12, 16, 20]:
        print(2*'\n'+'-- running time_harmonic maxwell for test '+test_case+' on '+domain_name+' with deg = '+repr(deg)+', nc = '+repr(nc)+' -- '+2*'\n')
        ndofs, l2_error = run_maxwell_2d_time_harmonic(nc=nc, deg=deg, test_case=test_case, domain_name=domain_name, nitsche_method=nitsche_method)
        results.append([nc, ndofs, l2_error])

    print(2*'\n'+'-- run completed -- '+2*'\n')
    print('results (ncells / ndofs / l2_errors):')
    for nc, ndofs, err in results:
        print(repr(nc)+' '+repr(ndofs)+' '+repr(err)+' ')
    print(2*'\n'+' -- '+2*'\n')



# [1]: errors given by other currents:
#
# J_x = -(y-y0) * Max(dr**2 - (((x/ax)**2 + (y-y0)**2)**.5-r0)**2, 0)   # /(x**2 + y**2)
# J_y =  (x/ax) * Max(dr**2 - (((x/ax)**2 + (y-y0)**2)**.5-r0)**2, 0)
# gives the error:
# NotImplementedError: Cannot translate to Sympy:
# Max(0, 0.01 - 4.41*(0.476190476190476*((1.0*x1*sin(x2) + 0.5)**2 + 0.652366863905326*(1.0*x1*cos(x2) + 1)**2)**0.5 - 1)**2)
#
# J_x = -(y-y0) * exp( - ((( (x/ax)**2 + (y-y0)**2 )**.5-r0 )/dr)**2 )   # /(x**2 + y**2)
# J_y =  (x/ax) * exp( - ((((x/ax)**2 + (y-y0)**2)**.5-r0)/dr)**2 )
# gives the error:
# "numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
# NameError: name 'sqrt' is not defined"
