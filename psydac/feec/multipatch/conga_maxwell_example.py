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
from psydac.feec.multipatch.operators import BrokenMass, ConformingProjection_V1, ConformingProjection_V0 #ortho_proj_Hcurl
from psydac.feec.multipatch.operators import time_count
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector
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
    nquads = [2*d + 1 for d in degree]
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

#==============================================================================
def run_conga_maxwell_2d(E_ex, f, alpha, domain, ncells, degree, gamma_jump=1, save_dir=None, load_dir=None, comm=None,
                         plot_source=False, plot_sol=False, plot_source_div=False, return_sol=False):
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
    etas, xx, yy = get_plotting_grid(mappings, N=40)
    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

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

    if plot_source_div:
        # actually plotting the projected divergence of f
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

    if plot_source:
        # representation of discrete source:
        b_c = b.toarray()
        fh_c = spsolve(M1_m, b_c)
        fh_norm = np.dot(fh_c,M1_m.dot(fh_c))
        print("|| fh || = ", fh_norm)
        print("|| div fh ||/|| fh || = ", div_norm(fh_c)/fh_norm)

        div_fh = FemField(V0h, coeffs=array_to_stencil(div_m.dot(fh_c), V0h.vector_space))
        fh = FemField(V1h, coeffs=array_to_stencil(fh_c, V1h.vector_space))

        div_fh_vals = grid_vals_h1(div_fh)
        fh_x_vals, fh_y_vals = grid_vals_hcurl(fh)
        my_small_plot(
            title=r'discrete source term for Maxwell curl-curl problem',
            vals=[np.abs(fh_x_vals), np.abs(fh_y_vals), np.abs(div_fh_vals)],
            titles=[r'$|fh_x|$', r'$|fh_y|$', r'$|div_h fh|$'],  # , r'$div_h J$' ],
            surface_plot=False,
            xx=xx, yy=yy,
        )

        my_small_streamplot(
            title='source J',
            vals_x=fh_x_vals,
            vals_y=fh_y_vals,
            xx=xx, yy=yy,
            amplification=.05
        )

        # show source corrected with P1^T  -- this doesn't seem to change much, a bit strange -- need to check
        plot_corrected_f = False
        if plot_corrected_f:
            fh_c = spsolve(M1_m, cP1_m.transpose().dot(b_c))
            print("|| fh || = ", np.dot(fh_c,M1_m.dot(fh_c)))
            print("|| div fh || = ", div_norm(fh_c))
            div_fh = FemField(V0h, coeffs=array_to_stencil(div_m.dot(fh_c), V0h.vector_space))
            fh = FemField(V1h, coeffs=array_to_stencil(fh_c, V1h.vector_space))

            div_fh_vals = grid_vals_h1(div_fh)
            fh_x_vals, fh_y_vals = grid_vals_hcurl(fh)

            my_small_plot(
                title=r'discrete CORRECTED source term for Maxwell curl-curl problem',
                vals=[np.abs(fh_x_vals), np.abs(fh_y_vals), np.abs(div_fh_vals)],
                titles=[r'$|fh_x|$', r'$|fh_y|$', r'$|div_h fh|$'],  # , r'$div_h J$' ],
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
        b = b.toarray()     # why not 'to_array', for consistency with array_to_stencil ?

        t_stamp = time_count(t_stamp)
        print("solving with scipy...")
        Eh_c = spsolve(A_m, b)
        E_coeffs = array_to_stencil(Eh_c, V1h.vector_space)

    else:
        assert not load_dir
        t_stamp = time_count(t_stamp)
        print("solving with psydac cg solver...")

        E_coeffs, info = cg( A, b, tol=maxwell_tol, verbose=True )

    # projected solution
    Eh = FemField(V1h, coeffs=array_to_stencil(cP1_m.dot(Eh_c), V1h.vector_space))

    Eh_norm = np.dot(Eh_c,M1_m.dot(Eh_c))
    print("|| Eh || = ", Eh_norm)
    print("|| div Eh || / || Eh || = ", div_norm(Eh_c)/Eh_norm)

    if E_ex is not None:
        # error
        error       = Matrix([F[0]-E_ex[0],F[1]-E_ex[1]])
        l2_norm     = Norm(error, domain, kind='l2')
        l2_norm_h   = discretize(l2_norm, domain_h, V1h, backend=PSYDAC_BACKENDS['numba'])
        l2_error    = l2_norm_h.assemble(F=Eh)
    else:
        l2_error = None

    if return_sol:
        return l2_error, Eh
    else:
        if l2_error is None:
            print("Warning: I have no error and I'm not returning the solution !! ")
        return l2_error






def run_maxwell_2d_time_harmonic():
    """
    curl-curl problem with 0 order term and source
    """

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # test_case selection with domain
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    test_case='circling_J'
    n_patches = None

    plot_source = True
    plot_sol = True

    if test_case=='circling_J':
        domain_name = 'pretzel'
        # domain_name = 'square'; n_patches = 6
        # domain_name = 'annulus'; n_patches = 4
        nc = 2**6; deg = 3
        # nc = 2**5; deg = 3

        # domain_name = 'pretzel_debug'
        # nc = 2

    elif test_case == 'manufactured_sol':
        domain_name = 'square'; n_patches = 6
        nc = 2**4
        deg = 2

    else:
        raise NotImplementedError

    fem_name = get_fem_name(domain_name=domain_name,n_patches=n_patches,nc=nc,deg=deg) #domain_name+np_suffix+'_nc'+repr(nc)+'_deg'+repr(deg)
    save_dir = load_dir = get_load_dir(domain_name=domain_name,n_patches=n_patches,nc=nc,deg=deg)  # './tmp_matrices/'+fem_name+'/'
    if load_dir and not os.path.exists(load_dir):
        print("discarding load_dir, since I cannot find it")
        load_dir = None

    domain = build_multipatch_domain(domain_name=domain_name, n_patches=n_patches)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    x,y    = domain.coordinates


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # source definition
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if test_case == 'manufactured_sol':

        omega = 1  # ?
        alpha  = -omega**2
        E_ex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
        f      = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                         alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))
        E_ex_x = lambdify(domain.coordinates, E_ex[0])
        E_ex_y = lambdify(domain.coordinates, E_ex[1])
        E_ex_log = [pull_2d_hcurl([E_ex_x,E_ex_y], f) for f in mappings_list]

    elif test_case == 'circling_J':

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

        E_ex = None

    else:
        raise NotImplementedError

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # calling solver
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # jump penalization factor from Buffa, Perugia and Warburton  >> need to study
    h = 1/nc
    gamma_jump = 10*(deg+1)**2/h

    l2_error, uh = run_conga_maxwell_2d(
        E_ex, f, alpha, domain, gamma_jump=gamma_jump,
        ncells=[nc, nc], degree=[deg,deg],
        save_dir=save_dir, load_dir=load_dir, return_sol=True,
        plot_source=plot_source, plot_sol=plot_sol,
    )

    # else:
    #     # Nitsche
    #     l2_error, uh = run_system_3_2d_dir(E_ex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2,2], return_sol=True)

    if E_ex:
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

    Eh_x_vals, Eh_y_vals = grid_vals_hcurl(uh)
    if E_ex:
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
        Eh_abs_vals = [np.sqrt(abs(ex)**2 + abs(ey)**2) for ex, ey in zip(Eh_x_vals, Eh_y_vals)]
        my_small_plot(
            title=r'discrete field $E_h$ for $\omega = $'+repr(omega),
            vals=[Eh_x_vals, Eh_y_vals, Eh_abs_vals],
            titles=[r'$E^h_x$', r'$E^h_y$', r'$|E^h|$'],
            xx=xx,
            yy=yy,
            surface_plot=True,
            gridlines_x1=gridlines_x1,
            gridlines_x2=gridlines_x2,
            save_fig='Eh_'+fem_name+'.png',
        )

    my_small_streamplot(
        title=('solution E'),
        vals_x=Eh_x_vals,
        vals_y=Eh_y_vals,
        skip=1,
        xx=xx,
        yy=yy,
        amplification=1,
        save_fig='Eh_vf_'+fem_name+'.png',
    )


if __name__ == '__main__':

    run_maxwell_2d_time_harmonic()





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
