from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict

from sympy import lambdify

from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from sympde.calculus import dot
from sympde.topology import element_of
from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral
from sympde.topology import Derham

from psydac.api.settings import PSYDAC_BACKENDS

from psydac.feec.pull_push import pull_2d_h1, pull_2d_hcurl, pull_2d_l2

from psydac.feec.multipatch.api import discretize
from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator
from psydac.feec.multipatch.operators import time_count, HodgeOperator
from psydac.feec.multipatch.plotting_utilities import plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases import get_source_and_sol_for_magnetostatic_pbm
from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d import get_eigenvalues

def solve_magnetostatic_pbm(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_proj='P_L2_wcurl_J',
        source_type='dipole_J', bc_type='metallic',
        gamma0_h=10., gamma1_h=10.,
        dim_harmonic_space=0,
        project_solution=False,
        plot_source=False, plot_dir=None, hide_plots=True,
        m_load_dir="",
):
    """
    solver for a magnetostatic problem

          div B = 0
         curl B = j

    written in the form of a mixed problem: find p in H1, u in H(curl), such that

          G^* u = f_scal     on \Omega
      G p + A u = f_vect     on \Omega

    with operators

      G:   p -> grad p
      G^*: u -> -div u
      A:   u -> curl curl u

    and sources

      f_scal = 0
      f_vect = curl j

    -- then the solution u = (Bx, By) satisfies the original magnetostatic equation, see e.g.
        Beirão da Veiga, Brezzi, Dassi, Marini and Russo, Virtual Element approx of 2D magnetostatic pbms, CMAME 327 (2017)

    Here the operators G and A are discretized with

      Gh: V0h -> V1h  and  Ah: V1h -> V1h

    in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \Omega,

      V0h  --grad->  V1h  -—curl-> V2h

    and boundary conditions to be specified (see the multi-patch paper for details).

    Harmonic constraint: if dim_harmonic_space > 0, a constraint is added, of the form

        u in H^\perp

    where H = ker(L) is the kernel of the Hodge-Laplace operator L = curl curl u  - grad div

    Note: if source_proj == 'P_L2_wcurl_J' then a scalar J is given and we define the V1h part of the discrete source as
    l(v) := <curl_h v, J>

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma0_h: jump penalization parameter in V0h
    :param gamma1_h: jump penalization parameter in V1h
    :param source_proj: approximation operator for the source, possible values are 'P_geom' or 'P_L2'
    :param source_type: must be implemented as a test-case
    :param bc_type: 'metallic' or 'pseudo-vacuum' -- see details in multi-patch paper
    :param m_load_dir: directory for matrix storage
    """

    ncells = [nc, nc]
    degree = [deg,deg]

    # if backend_language is None:
    #     if domain_name in ['pretzel', 'pretzel_f'] and nc > 8:
    #         backend_language='numba'
    #     else:
    #         backend_language='python'
    # print('[note: using '+backend_language+ ' backends in discretize functions]')
    assert bc_type in ['metallic', 'pseudo-vacuum']

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_mixed_source_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' source_proj = {}'.format(source_proj))
    print(' bc_type = {}'.format(bc_type))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    print('building symbolic and discrete domain...')
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    domain_h = discretize(domain, ncells=ncells)

    print('building symbolic and discrete derham sequences...')
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))

    print('building the discrete operators:')
    print('commuting projection operators...')
    nquads = [4*(d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    # these physical projection operators should probably be in the interface...
    def P0_phys(f_phys):
        f = lambdify(domain.coordinates, f_phys)
        f_log = [pull_2d_h1(f, m) for m in mappings_list]
        return P0(f_log)

    def P1_phys(f_phys):
        f_x = lambdify(domain.coordinates, f_phys[0])
        f_y = lambdify(domain.coordinates, f_phys[1])
        f_log = [pull_2d_hcurl([f_x, f_y], m) for m in mappings_list]
        return P1(f_log)

    def P2_phys(f_phys):
        f = lambdify(domain.coordinates, f_phys)
        f_log = [pull_2d_l2(f, m) for m in mappings_list]
        return P2(f_log)

    I0_m = IdLinearOperator(V0h).to_sparse_matrix()
    I1_m = IdLinearOperator(V1h).to_sparse_matrix()

    print('Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=0)
    H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=1)
    H2 = HodgeOperator(V2h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=2)

    dH0_m = H0.get_dual_Hodge_sparse_matrix()  # = mass matrix of V0
    H0_m  = H0.to_sparse_matrix()              # = inverse mass matrix of V0
    dH1_m = H1.get_dual_Hodge_sparse_matrix()  # = mass matrix of V1
    H1_m  = H1.to_sparse_matrix()              # = inverse mass matrix of V1
    dH2_m = H2.get_dual_Hodge_sparse_matrix()  # = mass matrix of V2
    H2_m  = H2.to_sparse_matrix()              # = inverse mass matrix of V2

    M0_m = dH0_m
    M1_m = dH1_m  # usual notation

    hom_bc = (bc_type == 'pseudo-vacuum')  #  /!\  here u = B is in H(curl), not E  /!\
    print('with hom_bc = {}'.format(hom_bc))

    print('conforming projection operators...')
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    cP0 = derham_h.conforming_projection(space='V0', hom_bc=hom_bc, backend_language=backend_language, load_dir=m_load_dir)
    cP1 = derham_h.conforming_projection(space='V1', hom_bc=hom_bc, backend_language=backend_language, load_dir=m_load_dir)
    cP0_m = cP0.to_sparse_matrix()
    cP1_m = cP1.to_sparse_matrix()

    print('broken differential operators...')
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Conga (projection-based) operator matrices
    print('grad matrix...')
    G_m = bD0_m @ cP0_m
    tG_m = dH1_m @ G_m  # grad: V0h -> tV1h

    print('curl-curl stiffness matrix...')
    C_m = bD1_m @ cP1_m
    CC_m = C_m.transpose() @ dH2_m @ C_m

    # jump penalization and stabilization operators:
    JP0_m = I0_m - cP0_m
    S0_m = JP0_m.transpose() @ dH0_m @ JP0_m

    JP1_m = I1_m - cP1_m
    S1_m = JP1_m.transpose() @ dH1_m @ JP1_m

    if not hom_bc:
        # very small regularization to avoid constant p=1 in the kernel
        reg_S0_m = 1e-16 * M0_m + gamma0_h * S0_m
    else:
        reg_S0_m = gamma0_h * S0_m

    hf_cs = []
    if dim_harmonic_space > 0:

        print('computing the harmonic fields...')
        gamma_Lh = 10  # penalization value should not change the kernel

        GD_m = - tG_m @ H0_m @ G_m.transpose() @ dH1_m   # todo: check with paper
        L_m = CC_m - GD_m + gamma_Lh * S1_m
        eigenvalues, eigenvectors = get_eigenvalues(dim_harmonic_space+1, 1e-6, L_m, dH1_m)

        for i in range(dim_harmonic_space):
            lambda_i =  eigenvalues[i]
            print(".. storing eigenmode #{}, with eigenvalue = {}".format(i, lambda_i))
            # check:
            if abs(lambda_i) > 1e-8:
                print(" ****** WARNING! this eigenvalue should be 0!   ****** ")
            hf_cs.append(eigenvectors[:,i])

        # matrix of the coefs of the harmonic fields (Lambda^H_i) in the basis (Lambda_i), in the form:
        #   hf_m = (c^H_{i,j})_{i < dim_harmonic_space, j < dim_V1}  such that  Lambda^H_i = sum_j c^H_{i,j} Lambda^1_j
        hf_m = bmat(hf_cs).transpose()
        MH_m = M1_m @ hf_m

        # check:
        lambda_i = eigenvalues[dim_harmonic_space]  # should be the first positive eigenvalue of L_h
        if abs(lambda_i) < 1e-4:
            print(" ****** Warning -- something is probably wrong: ")
            print(" ******            eigenmode #{} should have positive eigenvalue: {}".format(dim_harmonic_space, lambda_i))

        print('computing the full operator matrix with harmonic constraint...')
        A_m = bmat([[ reg_S0_m,        tG_m.transpose(),  None ],
                    [     tG_m,  CC_m + gamma1_h * S1_m,  MH_m ],
                    [     None,        MH_m.transpose(),  None ]])

    else:
        print('computing the full operator matrix without harmonic constraint...')

        A_m = bmat([[ reg_S0_m,        tG_m.transpose() ],
                    [     tG_m,  CC_m + gamma1_h * S1_m ]])

    # get exact source, bc's, ref solution...
    # (not all the returned functions are useful here)
    print('getting the source and ref solution...')
    N_diag = 200
    method = 'conga'
    f_scal, f_vect, j_scal, uh_ref = get_source_and_sol_for_magnetostatic_pbm(source_type=source_type, domain=domain, domain_name=domain_name)

    # compute approximate source:
    #   ff_h = (f0_h, f1_h) = (P0_h f_scal, P1_h f_vect)  with projection operators specified by source_proj
    #   and dual-basis coefficients in column array  bb_c = (b0_c, b1_c)
    # note: f1_h may also be defined through the special option 'P_L2_wcurl_J' for magnetostatic problems
    f0_c = f1_c = j2_c = None
    assert source_proj in ['P_geom', 'P_L2', 'P_L2_wcurl_J']

    if f_scal is None:
        tilde_f0_c = np.zeros(V0h.nbasis)
    else:
        print('approximating the V0 source with '+source_proj)
        if source_proj == 'P_geom':
            f0_h = P0_phys(f_scal)
            f0_c = f0_h.coeffs.toarray()
            tilde_f0_c = dH0_m.dot(f0_c)
        else:
            # L2 proj
            tilde_f0_c = derham_h.get_dual_dofs(space='V0', f=f_scal, backend_language=backend_language, return_format='numpy_array')

    if source_proj == 'P_L2_wcurl_J':
        if j_scal is None:
            tilde_j2_c = np.zeros(V2h.nbasis)
            tilde_f1_c = np.zeros(V1h.nbasis)
        else:
            print('approximating the V1 source as a weak curl of j_scal')
            tilde_j2_c = derham_h.get_dual_dofs(space='V2', f=j_scal, backend_language=backend_language, return_format='numpy_array')
            tilde_f1_c = C_m.transpose().dot(tilde_j2_c)
    elif f_vect is None:
        tilde_f1_c  = np.zeros(V1h.nbasis)
    else:
        print('approximating the V1 source with '+source_proj)
        if source_proj == 'P_geom':
            f1_h = P1_phys(f_vect)
            f1_c = f1_h.coeffs.toarray()
            tilde_f1_c = dH1_m.dot(f1_c)
        else:
            assert source_proj == 'P_L2'
            tilde_f1_c = derham_h.get_dual_dofs(space='V1', f=f_vect, backend_language=backend_language, return_format='numpy_array')

    if plot_source:
        if f0_c is None:
            f0_c = H0_m.dot(tilde_f0_c)
        plot_field(numpy_coeffs=f0_c, plot_type='components', Vh=V0h, space_kind='h1', domain=domain, title='f0_h with P = '+source_proj,
                   filename=plot_dir+'f0h_'+source_proj+'.png', hide_plot=hide_plots)
        if f1_c is None:
            f1_c = H1_m.dot(tilde_f1_c)
        plot_field(numpy_coeffs=f1_c, plot_type='vector_field', Vh=V1h, space_kind='hcurl', domain=domain, title='f1_h with P = '+source_proj,
                   filename=plot_dir+'f1h_'+source_proj+'.png', hide_plot=hide_plots)
        if source_proj == 'P_L2_wcurl_J':
            if j2_c is None:
                j2_c = H2_m.dot(tilde_j2_c)
            plot_field(numpy_coeffs=j2_c, plot_type='components', Vh=V2h, space_kind='l2', domain=domain, title='P_L2 jh in V2h',
                       filename=plot_dir+'j2h.png', hide_plot=hide_plots)

    print("building block RHS")
    if dim_harmonic_space > 0:
        tilde_h_c = np.zeros(dim_harmonic_space)  # harmonic part of the rhs
        b_c = np.block([tilde_f0_c, tilde_f1_c, tilde_h_c])
    else:
        b_c = np.block([tilde_f0_c, tilde_f1_c])

    # direct solve with scipy spsolve ------------------------------
    print('solving source problem with scipy.spsolve...')
    sol_c = spsolve(A_m.asformat('csr'), b_c)
    #   ------------------------------------------------------------
    ph_c = sol_c[:V0h.nbasis]
    uh_c = sol_c[V0h.nbasis:V0h.nbasis+V1h.nbasis]
    hh_c = np.zeros(V1h.nbasis)
    if dim_harmonic_space > 0:
        # compute the harmonic part (h) of the solution
        hh_hbcoefs = sol_c[V0h.nbasis+V1h.nbasis:]  # coefs of the harmonic part, in the basis of the harmonic fields
        assert len(hh_hbcoefs) == dim_harmonic_space
        for i in range(dim_harmonic_space):
            hi_c = hf_cs[i]  # coefs the of the i-th harmonic field, in the B/M spline basis of V1h
            hh_c += hh_hbcoefs[i]*hi_c

    if project_solution:
        print('projecting the homogeneous solution on the conforming problem space...')
        uh_c = cP1_m.dot(uh_c)
        u_name = r'$P^1_h B_h$'
        ph_c = cP0_m.dot(ph_c)
        p_name = r'$P^0_h p_h$'
    else:
        u_name = r'$B_h$'
        p_name = r'$p_h$'

    print('getting and plotting the FEM solution from numpy coefs array...')
    params_str = 'gamma0_h={}_gamma1_h={}'.format(gamma0_h, gamma1_h)
    title = r'solution {} (amplitude)'.format(p_name)
    plot_field(numpy_coeffs=ph_c, Vh=V0h, space_kind='h1', plot_type='amplitude',
               domain=domain, title=title, filename=plot_dir+params_str+'_ph.png', hide_plot=hide_plots)
    title = r'solution $h_h$ (amplitude)'
    plot_field(numpy_coeffs=hh_c, Vh=V1h, space_kind='hcurl', plot_type='amplitude',
               domain=domain, title=title, filename=plot_dir+params_str+'_hh.png', hide_plot=hide_plots)
    title = r'solution {} (amplitude)'.format(u_name)
    plot_field(numpy_coeffs=uh_c, Vh=V1h, space_kind='hcurl', plot_type='amplitude',
               domain=domain, title=title, filename=plot_dir+params_str+'_uh.png', hide_plot=hide_plots)
    title = r'solution {} (vector field)'.format(u_name)
    plot_field(numpy_coeffs=uh_c, Vh=V1h, space_kind='hcurl', plot_type='vector_field',
               domain=domain, title=title, filename=plot_dir+params_str+'_uh_vf.png', hide_plot=hide_plots)
    title = r'solution {} (components)'.format(u_name)
    plot_field(numpy_coeffs=uh_c, Vh=V1h, space_kind='hcurl', plot_type='components',
               domain=domain, title=title, filename=plot_dir+params_str+'_uh_xy.png', hide_plot=hide_plots)

if __name__ == '__main__':

    t_stamp_full = time_count()

    bc_type = 'metallic'
    # bc_type = 'pseudo-vacuum'
    source_type = 'dipole_J'

    source_proj = 'P_L2_wcurl_J'

    domain_name = 'pretzel_f'
    dim_harmonic_space = 3

    # nc = 20
    # deg = 4
    nc = 8
    deg = 2

    # domain_name = 'curved_L_shape'
    # dim_harmonic_space = 0

    # nc = 2
    # deg = 2

    run_dir = '{}_{}_bc={}_nc={}_deg={}/'.format(domain_name, source_type, bc_type, nc, deg)
    m_load_dir = 'matrices_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
    solve_magnetostatic_pbm(
        nc=nc, deg=deg,
        domain_name=domain_name,
        source_type=source_type,
        source_proj=source_proj,
        bc_type=bc_type,
        backend_language='numba',
        dim_harmonic_space=dim_harmonic_space,
        plot_source=True,
        plot_dir='./plots/magnetostatic_runs/'+run_dir,
        hide_plots=True,
        m_load_dir=m_load_dir
    )

    time_count(t_stamp_full, msg='full program')
