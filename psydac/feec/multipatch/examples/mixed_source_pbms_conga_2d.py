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

from psydac.feec.pull_push import pull_2d_h1, pull_2d_hcurl

from psydac.feec.multipatch.api import discretize
from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator
from psydac.feec.multipatch.operators import time_count, HodgeOperator
from psydac.feec.multipatch.plotting_utilities import plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases import get_source_and_solution
from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d import get_eigenvalues
comm = MPI.COMM_WORLD

def solve_mixed_source_pbm(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_proj='P_geom', source_type='manu_J',
        gamma0_h=10., gamma1_h=10.,
        dim_harmonic_space=0,
        plot_source=False, plot_dir=None, hide_plots=True,
        m_load_dir="",
):
    """
    solver for the mixed problem: find p in H1_0, u in H_0(curl), such that

          B^* u = f_scal     on \Omega
      B p + A u = f_vect     on \Omega

    with operators

      B:   H1_0 -> L2,                  p -> grad p
      B^*: H(div) -> L2,                u -> -div u
      A:   H_0(curl) \cup H(div) -> L2, u -> curl curl u

    are discretized as

      Bh: V0h -> V1h  and  Ah: V1h -> V1h

    in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \Omega,

      V0h  --grad->  V1h  -—curl-> V2h

    Moreover: if dim_harmonic_space > 0, a harmonic constraint is added, of the form
        u in H^\perp
    where H = ker(L) is the kernel of the Hodge-Laplace operator L = curl curl u  - grad div

    Example:
        for  f_scal = 0  and  f_vect = curl(j)  this corresponds to a magnetostatic problem (u = (Bx, By))
        with scalar current j = J_z and divergence-free + harmonic gauge, see e.g.
        Beirão da Veiga, Brezzi, Dassi, Marini and Russo, Virtual Element approx of 2D magnetostatic pbms, CMAME 327 (2017)

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma_h: jump penalization parameter
    :param source_proj: approximation operator for the source, possible values are 'P_geom' or 'P_L2'
    :param source_type: must be implemented in get_source_and_solution()
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
    if not os.path.exists(m_load_dir):
        os.makedirs(m_load_dir)

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_mixed_source_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' source_proj = {}'.format(source_proj))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    print('building symbolic and discrete domain...')
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    domain_h = discretize(domain, ncells=ncells, comm=comm)

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

    I0_m = IdLinearOperator(V0h).to_sparse_matrix()
    I1_m = IdLinearOperator(V1h).to_sparse_matrix()

    print('Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language, storage_fn=[m_load_dir+"H0_m.npz", m_load_dir+"dH0_m.npz"])
    H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language, storage_fn=[m_load_dir+"H1_m.npz", m_load_dir+"dH1_m.npz"])
    H2 = HodgeOperator(V2h, domain_h, backend_language=backend_language, storage_fn=[m_load_dir+"H2_m.npz", m_load_dir+"dH2_m.npz"])

    dH0_m = H0.get_dual_Hodge_sparse_matrix()  # = mass matrix of V0
    H0_m  = H0.to_sparse_matrix()              # = inverse mass matrix of V0
    dH1_m = H1.get_dual_Hodge_sparse_matrix()  # = mass matrix of V1
    H1_m  = H1.to_sparse_matrix()              # = inverse mass matrix of V1
    dH2_m = H2.get_dual_Hodge_sparse_matrix()  # = mass matrix of V2

    M1_m = dH1_m  # convenient 'alias'

    print('conforming projection operators...')
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    cP0 = derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language, storage_fn=m_load_dir+"cP0_hom_m.npz")
    cP1 = derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language, storage_fn=m_load_dir+"cP1_hom_m.npz")
    cP0_m = cP0.to_sparse_matrix()
    cP1_m = cP1.to_sparse_matrix()

    print('broken differential operators...')
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Conga (projection-based) operator matrices

    print('(filtered) grad matrix...')
    G_m = bD0_m @ cP0_m
    fG_m = cP1_m.transpose() @ dH1_m @ G_m

    print('curl-curl stiffness matrix...')
    C_m = bD1_m @ cP1_m
    CC_m = C_m.transpose() @ dH2_m @ C_m

    # jump penalization and stabilization operators:
    JP0_m = I0_m - cP0_m
    S0_m = JP0_m.transpose() @ dH0_m @ JP0_m

    JP1_m = I1_m - cP1_m
    S1_m = JP1_m.transpose() @ dH1_m @ JP1_m

    hf_cs = []
    if dim_harmonic_space > 0:

        print('computing the harmonic fields...')
        gamma_Lh = 10  # penalization value should not change the kernel

        GD_m = - fG_m @ H0_m @ G_m.transpose() @ dH1_m   # todo: check with papers...
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

        # check:
        lambda_i = eigenvalues[dim_harmonic_space]  # should be the first positive eigenvalue of L_h
        if abs(lambda_i) < 1e-4:
            print(" ****** Warning -- something is probably wrong: ")
            print(" ******            eigenmode #{} should have positive eigenvalue: {}".format(dim_harmonic_space, lambda_i))

        print('computing the full operator matrix with harmonic constraint...')

        MH_m = M1_m @ hf_m
        # todo: try with a filtered version (as in paper)
        #       fMH_m = cP1_m.transpose() @ MH_m

        # DCA_m = bmat([[CC_m + gamma1_h * S1_m, sG_m, HC_m.transpose()], [sG_m.transpose(), gamma_h * JP0_m, None], [HC_m, None, None]])
        # A_m = bmat([[ CC_m + gamma1_h * S1_m, fG_m, HC_m],
        #             [ fG_m.transpose(), gamma0_h * S0_m]])

        # A_m: block square matrix of size  dim_V0h + dim_V1h + dim_harmonic_space
        A_m = bmat([[ gamma0_h * S0_m,       fG_m.transpose(), None ],
                    [            fG_m, CC_m + gamma1_h * S1_m, MH_m ],
                    [            None,       MH_m.transpose(), None ]])

    else:
        print('computing the full operator matrix without harmonic constraint...')

        A_m = bmat([[ gamma0_h * S0_m,       fG_m.transpose()],
                    [            fG_m, CC_m + gamma1_h * S1_m]])

    # get exact source, bc's, ref solution...
    # (not all the returned functions are useful here)
    print('getting the source and ref solution...')
    N_diag = 200
    method = 'conga'
    f_scal, f_vect, u_bc, ph_ref, uh_ref, p_ex, u_ex, phi, grad_phi = get_source_and_solution(
        source_type=source_type, domain=domain, domain_name=domain_name,
        refsol_params=[N_diag, method, source_proj],
    )

    # compute approximate source:
    #   ff_h = (f0_h, f1_h) = (P0_h f_scal, P1_h f_vect)  with projection operators specified by source_proj
    #   and dual-basis coefficients in column array  bb_c = (b0_c, b1_c)
    b1_c = b0_c = f1_c = f0_c = None
    if source_proj == 'P_geom':
        print('projecting the source with commuting projections...')
        f1_x = lambdify(domain.coordinates, f_vect[0])
        f1_y = lambdify(domain.coordinates, f_vect[1])
        f1_log = [pull_2d_hcurl([f1_x, f1_y], m) for m in mappings_list]
        f1_h = P1(f1_log)
        f1_c = f1_h.coeffs.toarray()
        b1_c = dH1_m.dot(f1_c)

        print('projecting the source with commuting projections...')
        f0 = lambdify(domain.coordinates, f_scal)
        f0_log = [pull_2d_h1(f0, m) for m in mappings_list]
        f0_h = P0(f0_log)
        f0_c = f0_h.coeffs.toarray()
        b0_c = dH0_m.dot(f0_c)

    elif source_proj == 'P_L2':
        print('projecting the source with L2 projections...')
        v  = element_of(V0h.symbolic_space, name='v')
        l = LinearForm(v, integral(domain, f_scal * v))
        lh = discretize(l, domain_h, V0h, backend=PSYDAC_BACKENDS[backend_language])
        b0  = lh.assemble()
        b0_c = b0.toarray()
        v  = element_of(V1h.symbolic_space, name='v')
        l = LinearForm(v, integral(domain, dot(f_vect,v)))
        lh = discretize(l, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])
        b1  = lh.assemble()
        b1_c = b1.toarray()
        if plot_source:
            f0_c = H0_m.dot(b0_c)
            f1_c = H1_m.dot(b1_c)
    else:
        raise ValueError(source_proj)

    if plot_source:
        plot_field(numpy_coeffs=f0_c, Vh=V0h, space_kind='h1', domain=domain, title='f0_h with P = '+source_proj,
                   filename=plot_dir+'/f0h_'+source_proj+'.png', hide_plot=hide_plots)
        plot_field(numpy_coeffs=f1_c, Vh=V1h, space_kind='hcurl', domain=domain, title='f1_h with P = '+source_proj,
                   filename=plot_dir+'/f1h_'+source_proj+'.png', hide_plot=hide_plots)

    print("building block RHS")
    if dim_harmonic_space > 0:
        bh_c = np.zeros(dim_harmonic_space)  # harmonic part of the rhs
        b_c = np.block([b0_c, b1_c, bh_c])
    else:
        b_c = np.block([b0_c, b1_c])

    # direct solve with scipy spsolve
    print('solving source problem with scipy.spsolve...')
    sol_c = spsolve(A_m, b_c)

    ph_c = sol_c[:V0h.nbasis]
    uh_c = sol_c[V0h.nbasis:V0h.nbasis+V1h.nbasis]
    if dim_harmonic_space > 0:
        # compute the harmonic part (h) of the solution
        hh_c = np.zeros(V1h.nbasis)
        hh_hbcoefs = sol_c[V0h.nbasis+V1h.nbasis:]  # coefs of the harmonic part, in the basis of the harmonic fields
        assert len(hh_hbcoefs) == dim_harmonic_space
        for i in range(dim_harmonic_space):
            hi_c = hf_cs[i]  # coefs the of the i-th harmonic field, in the B/M spline basis of V1h
            hh_c += hh_hbcoefs[i]*hi_c

    # project the homogeneous solution on the conforming problem space
    print('projecting the homogeneous solution on the conforming problem space...')
    uh_c = cP1_m.dot(uh_c)
    ph_c = cP0_m.dot(ph_c)

    print('getting and plotting the FEM solution from numpy coefs array...')
    params_str = 'gamma0_h={}_gamma1_h={}'.format(gamma0_h, gamma1_h)
    title = r'solution $P0 p_h$ (amplitude)'
    plot_field(numpy_coeffs=ph_c, Vh=V0h, space_kind='h1', domain=domain, title=title, filename=plot_dir+params_str+'_ph.png', hide_plot=hide_plots)
    title = r'solution $P1 u_h$ (amplitude)'
    plot_field(numpy_coeffs=uh_c, Vh=V1h, space_kind='hcurl', plot_type='amplitude',
               domain=domain, title=title, filename=plot_dir+params_str+'_uh.png', hide_plot=hide_plots)
    title = r'solution $P1 u_h$ (vector field)'
    plot_field(numpy_coeffs=uh_c, Vh=V1h, space_kind='hcurl', plot_type='vector_field',
               domain=domain, title=title, filename=plot_dir+params_str+'_uh_vf.png', hide_plot=hide_plots)
    title = r'solution $P1 u_h$ (components)'
    plot_field(numpy_coeffs=uh_c, Vh=V1h, space_kind='hcurl', plot_type='components',
               domain=domain, title=title, filename=plot_dir+params_str+'_uh_xy.png', hide_plot=hide_plots)

if __name__ == '__main__':

    t_stamp_full = time_count()

    quick_run = True
    # quick_run = False

    source_type = 'curl_dipole_J'
    # source_type = 'manu_J'

    if quick_run:
        domain_name = 'curved_L_shape'
        nc = 4
        deg = 2
    else:
        nc = 8
        deg = 4

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

    run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
    m_load_dir = 'matrices_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
    solve_mixed_source_pbm(
        nc=nc, deg=deg,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='numba',
        dim_harmonic_space=dim_harmonic_space,
        plot_source=True,
        plot_dir='./plots/tests_source_feb_13/'+run_dir,
        hide_plots=True,
        m_load_dir=m_load_dir
    )

    time_count(t_stamp_full, msg='full program')