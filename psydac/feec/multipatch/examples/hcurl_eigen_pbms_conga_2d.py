from mpi4py import MPI

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from scipy.sparse.linalg import spilu, lgmres
from scipy.sparse.linalg import LinearOperator, eigsh, minres

from scipy.linalg        import norm

from sympde.topology import Derham

from psydac.feec.multipatch.api import discretize

from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator
from psydac.feec.multipatch.operators import time_count, HodgeOperator
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.plotting_utilities import plot_field

def hcurl_solve_eigen_pbm(nc=4, deg=4, domain_name='pretzel_f', backend_language='python', mu=1, nu=1, gamma_h=10,
                          sigma=None, nb_eigs=4, nb_eigs_plot=4,
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

    I1 = IdLinearOperator(V1h)
    I1_m = I1.to_sparse_matrix()

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

    print('conforming projection operators...')
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    cP0 = derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
    cP1 = derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
    cP0_m = cP0.to_sparse_matrix()
    cP1_m = cP1.to_sparse_matrix()

    print('broken differential operators...')
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Conga (projection-based) stiffness matrices
    # curl curl:
    print('curl-curl stiffness matrix...')
    pre_CC_m = bD1_m.transpose() @ dH2_m @ bD1_m
    CC_m = cP1_m.transpose() @ pre_CC_m @ cP1_m  # Conga stiffness matrix

    # grad div:
    print('grad-div stiffness matrix...')
    pre_GD_m = - dH1_m @ bD0_m @ cP0_m @ H0_m @ cP0_m.transpose() @ bD0_m.transpose() @ dH1_m
    GD_m = cP1_m.transpose() @ pre_GD_m @ cP1_m  # Conga stiffness matrix

    # jump penalization in V1h:
    jump_penal_m = I1_m - cP1_m
    JP_m = jump_penal_m.transpose() * dH1_m * jump_penal_m

    print('computing the full operator matrix...')
    print('mu = {}'.format(mu))
    print('nu = {}'.format(nu))
    A_m = mu * CC_m - nu * GD_m + gamma_h * JP_m

    eigenvalues, eigenvectors = get_eigenvalues(nb_eigs, sigma, A_m, dH1_m)

    # plot first eigenvalues

    for i in range(min(nb_eigs_plot, nb_eigs)):

        print('looking at emode i = {}... '.format(i))
        lambda_i  = eigenvalues[i]
        emode_i = np.real(eigenvectors[:,i])
        norm_emode_i = np.dot(emode_i,dH1_m.dot(emode_i))
        print('norm of computed eigenmode: ', norm_emode_i)
        eh_c = emode_i/norm_emode_i  # numpy coeffs of the normalized eigenmode
        plot_field(numpy_coeffs=eh_c, Vh=V1h, space_kind='hcurl', domain=domain, title='mode e_{}, lambda_{}={}'.format(i,i,lambda_i),
                   filename=plot_dir+'e_{}.png'.format(i), hide_plot=hide_plots)

    return eigenvalues, eigenvectors


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
    max_shape_splu = 17000
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

if __name__ == '__main__':

    t_stamp_full = time_count()

    quick_run = True
    # quick_run = False

    if quick_run:
        domain_name = 'curved_L_shape'
        nc = 4
        deg = 2
    else:
        nc = 8
        deg = 4

    domain_name = 'pretzel_f'
    # domain_name = 'curved_L_shape'
    nc = 20
    deg = 4

    m_load_dir = 'matrices_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
    run_dir = 'eigenpbm_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
    hcurl_solve_eigen_pbm(
        nc=nc, deg=deg,
        nu=0,
        mu=1, #1,
        sigma=1,
        domain_name=domain_name,
        backend_language='numba',
        plot_dir='./plots/tests_source_february/'+run_dir,
        hide_plots=True,
        m_load_dir=m_load_dir,
    )

    time_count(t_stamp_full, msg='full program')
