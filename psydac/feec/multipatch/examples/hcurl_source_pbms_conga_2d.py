# coding: utf-8

from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict

from sympy import lambdify, Matrix

from scipy.sparse.linalg import spsolve

from sympde.calculus import dot
from sympde.topology import element_of
from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral, Norm
from sympde.topology import Derham

from psydac.api.settings import PSYDAC_BACKENDS
from psydac.feec.pull_push import pull_2d_hcurl

from psydac.feec.multipatch.api import discretize
from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator
from psydac.feec.multipatch.operators import HodgeOperator
from psydac.feec.multipatch.plotting_utilities import plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases import get_source_and_solution_OBSOLETE
from psydac.feec.multipatch.utilities import time_count
from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic import FemField

from psydac.feec.multipatch.non_matching_operators import construct_h1_conforming_projection, construct_hcurl_conforming_projection


def solve_hcurl_source_pbm(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_proj='P_geom', source_type='manu_J',
        eta=-10., mu=1., nu=1., gamma_h=10.,
        plot_source=False, plot_dir=None, hide_plots=True,
        m_load_dir=None,
):
    """
    solver for the problem: find u in H(curl), such that

      A u = f             on \\Omega
      n x u = n x u_bc    on \\partial \\Omega

    where the operator

      A u := eta * u  +  mu * curl curl u  -  nu * grad div u

    is discretized as  Ah: V1h -> V1h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h

    Examples:

      - time-harmonic maxwell equation with
          eta = -omega**2
          mu  = 1
          nu  = 0

      - Hodge-Laplacian operator L = A with
          eta = 0
          mu  = 1
          nu  = 1

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma_h: jump penalization parameter
    :param source_proj: approximation operator for the source, possible values are 'P_geom' or 'P_L2'
    :param source_type: must be implemented in get_source_and_solution()
    :param m_load_dir: directory for matrix storage
    """

    ncells = [nc, nc]
    degree = [deg, deg]

    # if backend_language is None:
    #     backend_language='python'
    # print('[note: using '+backend_language+ ' backends in discretize functions]')
    if m_load_dir is not None:
        if not os.path.exists(m_load_dir):
            os.makedirs(m_load_dir)

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_hcurl_source_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' source_proj = {}'.format(source_proj))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    t_stamp = time_count()
    print('building symbolic domain sequence...')
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping)
                           for P in domain.interior])
    mappings_list = list(mappings.values())

    t_stamp = time_count(t_stamp)
    print('building derham sequence...')
    derham = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print('building discrete domain...')
    domain_h = discretize(domain, ncells=ncells)

    t_stamp = time_count(t_stamp)
    print('building discrete derham sequence...')
    derham_h = discretize(derham, domain_h, degree=degree)

    t_stamp = time_count(t_stamp)
    print('building commuting projection operators...')
    nquads = [4 * (d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    # multi-patch (broken) spaces
    t_stamp = time_count(t_stamp)
    print('calling the multi-patch spaces...')
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))

    t_stamp = time_count(t_stamp)
    print('building the Id operator and matrix...')
    I1 = IdLinearOperator(V1h)
    I1_m = I1.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print('instanciating the Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    # other option: define as Hodge Operators:
    H0 = HodgeOperator(
        V0h,
        domain_h,
        backend_language=backend_language,
        load_dir=m_load_dir,
        load_space_index=0)
    H1 = HodgeOperator(
        V1h,
        domain_h,
        backend_language=backend_language,
        load_dir=m_load_dir,
        load_space_index=1)
    H2 = HodgeOperator(
        V2h,
        domain_h,
        backend_language=backend_language,
        load_dir=m_load_dir,
        load_space_index=2)

    t_stamp = time_count(t_stamp)
    print('building the primal Hodge matrix H0_m = M0_m ...')
    H0_m = H0.to_sparse_matrix()    # = mass matrix of V0

    t_stamp = time_count(t_stamp)
    print('building the dual Hodge matrix dH0_m = inv_M0_m ...')
    dH0_m = H0.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V0

    t_stamp = time_count(t_stamp)
    print('building the primal Hodge matrix H1_m = M1_m ...')
    H1_m = H1.to_sparse_matrix()  # = mass matrix of V1

    t_stamp = time_count(t_stamp)
    print('building the dual Hodge matrix dH1_m = inv_M1_m ...')
    dH1_m = H1.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V1

    # print("dH1_m @ H1_m == I1_m: {}".format(np.allclose((dH1_m @
    # H1_m).todense(), I1_m.todense())) )   # CHECK: OK

    t_stamp = time_count(t_stamp)
    print('building the primal Hodge matrix H2_m = M2_m ...')
    H2_m = H2.to_sparse_matrix()  # = mass matrix of V2

    t_stamp = time_count(t_stamp)
    print('building the conforming Projection operators and matrices...')
    # conforming Projections (should take into account the boundary conditions
    # of the continuous deRham sequence)
    cP0_m = construct_h1_conforming_projection(V0h, hom_bc=True)
    cP1_m = construct_hcurl_conforming_projection(V1h, hom_bc=True)

    t_stamp = time_count(t_stamp)
    print('building the broken differential operators and matrices...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()

    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    def lift_u_bc(u_bc):
        if u_bc is not None:
            print('lifting the boundary condition in V1h...')
            # note: for simplicity we apply the full P1 on u_bc, but we only
            # need to set the boundary dofs
            u_bc_x = lambdify(domain.coordinates, u_bc[0])
            u_bc_y = lambdify(domain.coordinates, u_bc[1])
            u_bc_log = [pull_2d_hcurl(
                [u_bc_x, u_bc_y], m.get_callable_mapping()) for m in mappings_list]
            # it's a bit weird to apply P1 on the list of (pulled back) logical
            # fields -- why not just apply it on u_bc ?
            uh_bc = P1(u_bc_log)
            ubc_c = uh_bc.coeffs.toarray()
            # removing internal dofs (otherwise ubc_c may already be a very
            # good approximation of uh_c ...)
            ubc_c = ubc_c - cP1_m.dot(ubc_c)
        else:
            ubc_c = None
        return ubc_c

    # Conga (projection-based) stiffness matrices
    # curl curl:
    t_stamp = time_count(t_stamp)
    print('computing the curl-curl stiffness matrix...')
    print(bD1_m.shape, H2_m.shape)
    pre_CC_m = bD1_m.transpose() @ H2_m @ bD1_m
    # CC_m = cP1_m.transpose() @ pre_CC_m @ cP1_m  # Conga stiffness matrix

    # grad div:
    t_stamp = time_count(t_stamp)
    print('computing the grad-div stiffness matrix...')
    pre_GD_m = - H1_m @ bD0_m @ cP0_m @ dH0_m @ cP0_m.transpose() @ bD0_m.transpose() @ H1_m
    # GD_m = cP1_m.transpose() @ pre_GD_m @ cP1_m  # Conga stiffness matrix

    # jump penalization:
    t_stamp = time_count(t_stamp)
    print('computing the jump penalization matrix...')
    jump_penal_m = I1_m - cP1_m
    JP_m = jump_penal_m.transpose() * H1_m * jump_penal_m

    t_stamp = time_count(t_stamp)
    print('computing the full operator matrix...')
    print('eta = {}'.format(eta))
    print('mu = {}'.format(mu))
    print('nu = {}'.format(nu))
    # useful for the boundary condition (if present)
    pre_A_m = cP1_m.transpose() @ (eta * H1_m + mu * pre_CC_m - nu * pre_GD_m)
    A_m = pre_A_m @ cP1_m + gamma_h * JP_m

    # get exact source, bc's, ref solution...
    # (not all the returned functions are useful here)
    t_stamp = time_count(t_stamp)
    print('getting the source and ref solution...')
    N_diag = 200
    method = 'conga'
    f_scal, f_vect, u_bc, p_ex, u_ex, phi, grad_phi = get_source_and_solution_OBSOLETE(
        source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,
    )

    # compute approximate source f_h
    t_stamp = time_count(t_stamp)
    b_c = f_c = None
    if source_proj == 'P_geom':
        # f_h = P1-geometric (commuting) projection of f_vect
        print('projecting the source with commuting projection...')
        f_x = lambdify(domain.coordinates, f_vect[0])
        f_y = lambdify(domain.coordinates, f_vect[1])
        f_log = [pull_2d_hcurl([f_x, f_y], m.get_callable_mapping())
                 for m in mappings_list]
        f_h = P1(f_log)
        f_c = f_h.coeffs.toarray()
        b_c = H1_m.dot(f_c)

    elif source_proj == 'P_L2':
        # f_h = L2 projection of f_vect
        print('projecting the source with L2 projection...')
        v = element_of(V1h.symbolic_space, name='v')
        expr = dot(f_vect, v)
        l = LinearForm(v, integral(domain, expr))
        lh = discretize(l, domain_h, V1h)
        b = lh.assemble()
        b_c = b.toarray()
        if plot_source:
            f_c = dH1_m.dot(b_c)
    else:
        raise ValueError(source_proj)

    if plot_source:
        plot_field(
            numpy_coeffs=f_c,
            Vh=V1h,
            space_kind='hcurl',
            domain=domain,
            title='f_h with P = ' +
            source_proj,
            filename=plot_dir +
            '/fh_' +
            source_proj +
            '.png',
            hide_plot=hide_plots)

    ubc_c = lift_u_bc(u_bc)

    if ubc_c is not None:
        # modified source for the homogeneous pbm
        t_stamp = time_count(t_stamp)
        print('modifying the source with lifted bc solution...')
        b_c = b_c - pre_A_m.dot(ubc_c)

    # direct solve with scipy spsolve
    t_stamp = time_count(t_stamp)
    print('solving source problem with scipy.spsolve...')
    uh_c = spsolve(A_m, b_c)

    # project the homogeneous solution on the conforming problem space
    t_stamp = time_count(t_stamp)
    print('projecting the homogeneous solution on the conforming problem space...')
    uh_c = cP1_m.dot(uh_c)

    if ubc_c is not None:
        # adding the lifted boundary condition
        t_stamp = time_count(t_stamp)
        print('adding the lifted boundary condition...')
        uh_c += ubc_c

    t_stamp = time_count(t_stamp)
    print('getting and plotting the FEM solution from numpy coefs array...')
    title = r'solution $u_h$ (amplitude) for $\eta = $' + repr(eta)
    params_str = 'eta={}_mu={}_nu={}_gamma_h={}'.format(eta, mu, nu, gamma_h)

    if plot_dir:
        plot_field(
            numpy_coeffs=uh_c,
            Vh=V1h,
            space_kind='hcurl',
            domain=domain,
            title=title,
            filename=plot_dir +
            params_str +
            '_uh.png',
            hide_plot=hide_plots)

    time_count(t_stamp)

    if u_ex:
        u = element_of(V1h.symbolic_space, name='u')
        l2norm = Norm(
            Matrix([u[0] - u_ex[0], u[1] - u_ex[1]]), domain, kind='l2')
        l2norm_h = discretize(l2norm, domain_h, V1h)
        uh_c = array_to_psydac(uh_c, V1h.vector_space)
        l2_error = l2norm_h.assemble(u=FemField(V1h, coeffs=uh_c))
        return l2_error


if __name__ == '__main__':

    t_stamp_full = time_count()

    quick_run = True
    # quick_run = False

    omega = np.sqrt(170)  # source
    roundoff = 1e4
    eta = int(-omega**2 * roundoff) / roundoff

    source_type = 'manu_maxwell'
    # source_type = 'manu_J'

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
    deg = 2

    # nc = 2
    # deg = 2

    run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
    m_load_dir = 'matrices_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
    solve_hcurl_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1,  # 1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_source=True,
        plot_dir='./plots/tests_source_feb_13/' + run_dir,
        hide_plots=True,
        m_load_dir=m_load_dir
    )

    time_count(t_stamp_full, msg='full program')
