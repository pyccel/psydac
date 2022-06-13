from mpi4py import MPI

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from scipy.sparse import coo_matrix, bmat
from scipy.sparse.linalg import inv as sp_inv

from scipy.sparse.linalg import spilu, lgmres
from scipy.sparse.linalg import LinearOperator, eigsh, minres
from scipy.linalg        import norm

from sympde.topology     import Derham

from psydac.feec.multipatch.api                         import discretize
from psydac.api.settings                                import PSYDAC_BACKENDS
from psydac.feec.multipatch.fem_linear_operators        import IdLinearOperator
from psydac.feec.multipatch.operators                   import HodgeOperator
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.plotting_utilities          import plot_field
from psydac.feec.multipatch.utilities                   import time_count

from sympde.topology      import Square    
from sympde.topology      import IdentityMapping, PolarMapping
from psydac.fem.vector import ProductFemSpace

from non_conf_example_coarse_confP import knots_to_insert, construct_projection_operator


def hcurl_solve_eigen_pbm(nc=4, deg=4, domain_name='pretzel_f', backend_language='python', mu=1, nu=1, gamma_h=10,
                          generalized_pbm=False, sigma=None, ref_sigmas=[], nb_eigs_solve=4, nb_eigs_plot=4, skip_eigs_threshold=None,
                          plot_dir=None, hide_plots=True, m_load_dir="",):
    """
    solver for the eigenvalue problem: find lambda in R and u in H0(curl), such that

      A u   = lambda * u    on \Omega

    with an operator

      A u := mu * curl curl u  -  nu * grad div u

    discretized as  Ah: V1h -> V1h  with a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \Omega,

      V0h  --grad->  V1h  -—curl-> V2h

    Examples:

      - curl-curl eigenvalue problem with
          mu  = 1
          nu  = 0

      - Hodge-Laplacian eigenvalue problem with
          mu  = 1
          nu  = 1

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma_h: jump penalization parameter
    """

    diags = {}
    ncells = [nc, nc]
    degree = [deg,deg]
    if sigma is None:
        raise ValueError('please specify a value for sigma')

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting hcurl_solve_eigen_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    print('building symbolic and discrete domain...')
    
    if domain_name in ['2patch_nc', '2patch_nc_mapped', '2patch_conf', '2patch_conf_mapped']:

        if domain_name in ['2patch_nc_mapped', '2patch_conf_mapped']:
            A = Square('A',bounds1=(0.5, 1), bounds2=(0,       np.pi/2))
            B = Square('B',bounds1=(0.5, 1), bounds2=(np.pi/2, np.pi)  )
            M1 = PolarMapping('M1',2, c1= 0, c2= 0, rmin = 0., rmax=1.)
            M2 = PolarMapping('M2',2, c1= 0, c2= 0, rmin = 0., rmax=1.)
        else:
            A = Square('A',bounds1=(0, np.pi/2), bounds2=(0, np.pi))
            B = Square('B',bounds1=(np.pi/2, np.pi), bounds2=(0, np.pi))
            M1 = IdentityMapping('M1', dim=2)
            M2 = IdentityMapping('M2', dim=2)
        A = M1(A)
        B = M2(B)

        domain = A.join(B, name = 'domain',
                    bnd_minus = A.get_boundary(axis=0, ext=1),
                    bnd_plus  = B.get_boundary(axis=0, ext=-1),
                    direction=1)

    else:
        domain = build_multipatch_domain(domain_name=domain_name)

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    print('building symbolic and discrete derham sequences...')
    
    t_stamp = time_count()
    print(' .. derham sequence...')
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    if domain_name in ['2patch_nc', '2patch_nc_mapped']:
        ncells_c = {
            'M1(A)':[nc, nc],
            'M2(B)':[nc, nc],
        }
        ncells_f = {
            'M1(A)':[2*nc, 2*nc],
            'M2(B)':[2*nc, 2*nc],
        }
        ncells_h = {
            'M1(A)':[2*nc, 2*nc],
            'M2(B)':[nc, nc],
        }

        t_stamp = time_count(t_stamp)
        print(' .. discrete domain...')
        domain_h = discretize(domain, ncells=ncells_h)   # Vh space
        domain_hc = discretize(domain, ncells=ncells_c)  # coarse Vh space
        domain_hf = discretize(domain, ncells=ncells_f)  # fine Vh space

        t_stamp = time_count(t_stamp)
        print(' .. discrete derham sequence...')
        derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])
        derham_hc = discretize(derham, domain_hc, degree=degree, backend=PSYDAC_BACKENDS[backend_language])
        derham_hf = discretize(derham, domain_hf, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    else:
        domain_h = discretize(domain, ncells=ncells)
        derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2    
    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))
    diags['ndofs_V0'] = V0h.nbasis
    diags['ndofs_V1'] = V1h.nbasis
    diags['ndofs_V2'] = V2h.nbasis

    t_stamp = time_count(t_stamp)
    print('building the discrete operators:')
    print('commuting projection operators...')
    nquads = [4*(d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    I1 = IdLinearOperator(V1h)
    I1_m = I1.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print('Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=0)
    H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=1)
    H2 = HodgeOperator(V2h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=2)

    H0_m  = H0.to_sparse_matrix()           # = mass matrix of V0
    dH0_m = H0.get_dual_sparse_matrix()     # = inverse mass matrix of V0
    H1_m  = H1.to_sparse_matrix()           # = mass matrix of V1
    dH1_m = H1.get_dual_sparse_matrix()     # = inverse mass matrix of V1
    H2_m  = H2.to_sparse_matrix()           # = mass matrix of V2

    t_stamp = time_count(t_stamp)
    print('conforming projection operators...')
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)

    if domain_name == '2patch_nc':

        V1h_c = derham_hc.V1
        V1h_f = derham_hf.V1

        cP1_c = derham_hc.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language)
        cP1_f = derham_hf.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language)

        c2f_patch00 = construct_projection_operator(domain=V1h_c.spaces[0].spaces[0], codomain=V1h_f.spaces[0].spaces[0])
        c2f_patch01 = construct_projection_operator(domain=V1h_c.spaces[0].spaces[1], codomain=V1h_f.spaces[0].spaces[1])

        # print(c2f_patch00.shape)
        c2f_patch0 = bmat([
            [c2f_patch00, None],
            [None, c2f_patch01]
        ])

        cf2_t = c2f_patch0.transpose()
        product = cf2_t @ c2f_patch0 
        print(cf2_t.shape)
        print(product.shape)
        inv_prod = sp_inv(product.tocsc())
        f2c_patch0 = inv_prod @ cf2_t

        E0 = c2f_patch0
        E0_star = f2c_patch0

        # numpy:
        cP1_c_00 = cP1_c.matrix[0,0].tosparse()
        cP1_c_10 = cP1_c.matrix[1,0].tosparse()
        cP1_c_01 = cP1_c.matrix[0,1].tosparse()
        cP1_c_11 = cP1_c.matrix[1,1].tosparse()

        cP1_f_00 = cP1_f.matrix[0,0].tosparse()
        cP1_f_10 = cP1_f.matrix[1,0].tosparse()
        cP1_f_01 = cP1_f.matrix[0,1].tosparse()
        cP1_f_11 = cP1_f.matrix[1,1].tosparse()

        print(c2f_patch0.shape)
        print(cP1_c_00.shape)
        print(V1h_f.nbasis)

        cP1_m = bmat([
            [c2f_patch0 @ cP1_c_00 @ f2c_patch0, c2f_patch0 @ cP1_c_01],
            [             cP1_c_10 @ f2c_patch0,              cP1_c_11]
        ])
        cP0_m = None

    else:
        cP0 = derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
        cP1 = derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
        cP0_m = cP0.to_sparse_matrix()
        cP1_m = cP1.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print('broken differential operators...')
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print('converting some matrices to csr format...')
    if cP0_m is not None:
        cP0_m = cP0_m.tocsr()
    H1_m = H1_m.tocsr()
    H2_m = H2_m.tocsr()
    cP1_m = cP1_m.tocsr()
    bD1_m = bD1_m.tocsr()    

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    print('computing the full operator matrix...')
    A_m = np.zeros_like(H1_m) 

    # Conga (projection-based) stiffness matrices
    if mu != 0:
        # curl curl:
        t_stamp = time_count(t_stamp)
        print('mu = {}'.format(mu))
        print('curl-curl stiffness matrix...')
        
        pre_CC_m = bD1_m.transpose() @ H2_m @ bD1_m
        CC_m = cP1_m.transpose() @ pre_CC_m @ cP1_m  # Conga stiffness matrix
        A_m += mu * CC_m 

    # grad div:
    if nu != 0:
        dH0_m = dH0_m.tocsr()
        bD0_m = bD0_m.tocsr()    

        t_stamp = time_count(t_stamp)
        print('nu = {}'.format(nu))
        print('grad-div stiffness matrix...')
        pre_GD_m = - H1_m @ bD0_m @ cP0_m @ dH0_m @ cP0_m.transpose() @ bD0_m.transpose() @ H1_m
        GD_m = cP1_m.transpose() @ pre_GD_m @ cP1_m  # Conga stiffness matrix
        A_m -= nu * GD_m

    # jump stabilization in V1h:
    if gamma_h != 0 or generalized_pbm:
        t_stamp = time_count(t_stamp)
        print('jump stabilization matrix...')
        jump_stab_m = I1_m - cP1_m
        JS_m = jump_stab_m.transpose() @ H1_m @ jump_stab_m
        
    if gamma_h != 0:
        print('gamma_h = {}'.format(gamma_h))
        print('adding jump stabilization to operator matrix...')
        A_m += gamma_h * JS_m

    if generalized_pbm:
        print('adding jump stabilization to RHS of generalized eigenproblem...')
        B_m = cP1_m.transpose() @ H1_m @ cP1_m + JS_m
    else:
        B_m = H1_m

    t_stamp = time_count(t_stamp)
    print('solving matrix eigenproblem...')
    all_eigenvalues, all_eigenvectors_transp = get_eigenvalues(nb_eigs_solve, sigma, A_m, B_m)

    t_stamp = time_count(t_stamp)
    print('sorting out eigenvalues...')
    zero_eigenvalues = []
    if skip_eigs_threshold is not None:
        eigenvalues = []
        eigenvectors = []
        for val, vect in zip(all_eigenvalues, all_eigenvectors_transp.T):
            if abs(val) < skip_eigs_threshold: 
                zero_eigenvalues.append(val)
                # we skip the eigenvector
            else:
                eigenvalues.append(val)
                eigenvectors.append(vect)
    else:
        eigenvalues = all_eigenvalues
        eigenvectors = all_eigenvectors_transp.T

    for k, val in enumerate(eigenvalues):
        diags['eigenvalue_{}'.format(k)] = val #eigenvalues[k]
    
    for k, val in enumerate(zero_eigenvalues):
        diags['skipped eigenvalue_{}'.format(k)] = val

    t_stamp = time_count(t_stamp)
    print('plotting the eigenmodes...')        
    nb_eigs = len(eigenvalues)
    for i in range(min(nb_eigs_plot, nb_eigs)):

        print('looking at emode i = {}... '.format(i))
        lambda_i  = eigenvalues[i]
        # emode_i = np.real(eigenvectors[:,i])
        emode_i = np.real(eigenvectors[i])
        norm_emode_i = np.dot(emode_i,H1_m.dot(emode_i))
        print('norm of computed eigenmode: ', norm_emode_i)
        # plot the broken eigenmode:
        eh_c = emode_i/norm_emode_i  # numpy coeffs of the normalized eigenmode
        params_str = 'gamma_h={}_gen={}'.format(gamma_h, generalized_pbm)
        plot_field(numpy_coeffs=eh_c, Vh=V1h, space_kind='hcurl', domain=domain, title='e_{}, lambda_{}={}'.format(i,i,lambda_i),
                    filename=plot_dir+'/'+params_str+'_e_{}.png'.format(i), hide_plot=hide_plots)

        plot_field(numpy_coeffs=eh_c, Vh=V1h, space_kind='hcurl', domain=domain, title='e_{}, lambda_{}={}'.format(i,i,lambda_i), 
                    filename=plot_dir+'/'+params_str+'_e_{}_comps.png'.format(i),
                    plot_type='components', hide_plot=hide_plots)

        # also plot the projected eigenmode:
        Peh_c = cP1_m.dot(eh_c)
        plot_field(numpy_coeffs=Peh_c, Vh=V1h, space_kind='hcurl', domain=domain, title='P e_{}, lambda_{}={}'.format(i,i,lambda_i),
                    filename=plot_dir+'/'+params_str+'_Pe_{}.png'.format(i), hide_plot=hide_plots)
        # same, as vector field:
        plot_field(numpy_coeffs=Peh_c, Vh=V1h, space_kind='hcurl', domain=domain, title='P e_{}, lambda_{}={}'.format(i,i,lambda_i), 
                    filename=plot_dir+'/'+params_str+'_Pe_{}_vf.png'.format(i),
                    plot_type='vector_field', hide_plot=hide_plots)



        plot_checks = False
        if plot_checks:
            # check: plot jump
            plot_field(numpy_coeffs=eh_c-Peh_c, Vh=V1h, space_kind='hcurl', domain=domain, title='(I-P) e_{}, lambda_{}={}'.format(i,i,lambda_i),
                        filename=plot_dir+'/'+params_str+'_Jump_e_{}.png'.format(i), hide_plot=hide_plots)

            # check: curl e_i
            Ceh_c = bD1_m @ cP1_m.dot(eh_c)
            plot_field(numpy_coeffs=Ceh_c, Vh=V2h, space_kind='l2', domain=domain, title='C e_{}, lambda_{}={}'.format(i,i,lambda_i),
                        filename=plot_dir+'/'+params_str+'_Ce_{}.png'.format(i), hide_plot=hide_plots)

            # check: curl curl e_i
            CCeh_c = dH1_m @ cP1_m.transpose() @ bD1_m.transpose() @ H2_m @ Ceh_c
            plot_field(numpy_coeffs=CCeh_c, Vh=V1h, space_kind='hcurl', domain=domain, title='CC e_{}'.format(i),
                        filename=plot_dir+'/'+params_str+'_CCe_{}.png'.format(i), hide_plot=hide_plots)

            # check: filtered lambda_i * e_i (should be = curl curl e_i)
            fl_eh_c = lambda_i * dH1_m @ cP1_m.transpose() @ H1_m @ Peh_c
            plot_field(numpy_coeffs=fl_eh_c, Vh=V1h, space_kind='hcurl', domain=domain, title='filtered lambda_i * e_{}'.format(i),
                        filename=plot_dir+'/'+params_str+'_fl_e_{}.png'.format(i), hide_plot=hide_plots)
    t_stamp = time_count(t_stamp)

    if ref_sigmas is not None:
        errors = []
        n_errs = min(len(ref_sigmas), len(eigenvalues))
        for k in range(n_errs):
            diags['error_{}'.format(k)] = abs(eigenvalues[k]-ref_sigmas[k])

    return diags


def get_eigenvalues(nb_eigs, sigma, A_m, M_m):
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- ')
    print('computing {0} eigenvalues (and eigenvectors) close to sigma={1} with scipy.sparse.eigsh...'.format(nb_eigs, sigma) )
    mode = 'normal'
    which = 'LM'
    # from eigsh docstring:
    #   ncv = number of Lanczos vectors generated ncv must be greater than k and smaller than n;
    #   it is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))
    ncv = 4*nb_eigs
    print('A_m.shape = ', A_m.shape)
    try_lgmres = True
    max_shape_splu = 24000   # OK for nc=20, deg=6 on pretzel_f
    if A_m.shape[0] < max_shape_splu:
        print('(via sparse LU decomposition)')
        OPinv = None
        tol_eigsh = 0
    else:

        OP_m = A_m - sigma*M_m
        tol_eigsh = 1e-7
        if try_lgmres:
            print('(via SPILU-preconditioned LGMRES iterative solver for A_m - sigma*M1_m)')
            OP_spilu = spilu(OP_m, fill_factor=15, drop_tol=5e-5)
            preconditioner = LinearOperator(OP_m.shape, lambda x: OP_spilu.solve(x) )
            tol = tol_eigsh
            OPinv = LinearOperator(
                matvec=lambda v: lgmres(OP_m, v, x0=None, tol=tol, atol=tol, M=preconditioner,
                                    callback=lambda x: print('cg -- residual = ', norm(OP_m.dot(x)-v))
                                    )[0],
                shape=M_m.shape,
                dtype=M_m.dtype
            )

        else:
            # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html:
            # the user can supply the matrix or operator OPinv, which gives x = OPinv @ b = [A - sigma * M]^-1 @ b.
            # > here, minres: MINimum RESidual iteration to solve Ax=b
            # suggested in https://github.com/scipy/scipy/issues/4170
            print('(with minres iterative solver for A_m - sigma*M1_m)')
            OPinv = LinearOperator(matvec=lambda v: minres(OP_m, v, tol=1e-10)[0], shape=M_m.shape, dtype=M_m.dtype)

    eigenvalues, eigenvectors = eigsh(A_m, k=nb_eigs, M=M_m, sigma=sigma, mode=mode, which=which, ncv=ncv, tol=tol_eigsh, OPinv=OPinv)

    print("done: eigenvalues found: " + repr(eigenvalues))
    return eigenvalues, eigenvectors

# if __name__ == '__main__':

#     t_stamp_full = time_count()

#     # quick_run = True
#     # # quick_run = False

#     # if quick_run:
#     #     domain_name = 'curved_L_shape'
#     #     nc = 4
#     #     deg = 2
#     # else:
#     #     nc = 8
#     #     deg = 4

#     domain_name = 'pretzel_f'
#     # domain_name = 'curved_L_shape'
#     nc = 8
#     deg = 3

#     run_dir = get_run_dir(domain_name, source_type, nc, deg)
#     plot_dir = get_plot_dir(case_dir, run_dir)

#     m_load_dir = get_mat_dir(domain_name, nc, deg)

#     hcurl_solve_eigen_pbm(
#         nc=nc, deg=deg,
#         nu=0,
#         mu=1, #1,
#         sigma=.1,
#         nb_eigs=6,
#         nb_eigs_plot=6,
#         domain_name=domain_name,
#         backend_language='pyccel-gcc',
#         plot_dir='./plots/tests_source_february/'+run_dir,
#         hide_plots=True,
#         m_load_dir=m_load_dir,
#     )

#     time_count(t_stamp_full, msg='full program')
