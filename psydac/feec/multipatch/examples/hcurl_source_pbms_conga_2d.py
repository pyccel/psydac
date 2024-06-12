"""
    solver for the problem: find u in H(curl), such that

      A u = f             on \\Omega
      n x u = n x u_bc    on \\partial \\Omega

    where the operator

      A u := eta * u  +  mu * curl curl u  -  nu * grad div u

    is discretized as  Ah: V1h -> V1h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h
"""

import os
from mpi4py import MPI
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
from psydac.feec.multipatch.examples.ppc_test_cases import get_source_and_solution_hcurl
from psydac.feec.multipatch.utils_conga_2d import DiagGrid, P0_phys, P1_phys, P2_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities import time_count
from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic import FemField
from psydac.feec.multipatch.non_matching_operators import construct_h1_conforming_projection, construct_hcurl_conforming_projection
from psydac.api.postprocessing import OutputManager, PostProcessManager


def solve_hcurl_source_pbm(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_proj='P_geom', source_type='manu_J',
        eta=-10., mu=1., nu=1., gamma_h=10.,
        project_sol=False, plot_dir=None, 
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
    :param source_proj: approximation operator (in V1h) for the source, possible values are
         - 'tilde_Pi':  dual commuting projection, an L2 projection filtered by the adjoint conforming projection)
    :param source_type: must be implemented in get_source_and_solution()
    :param m_load_dir: directory for matrix storage
    """
    diags = {}

    degree = [deg, deg]

    if m_load_dir is not None:
        if not os.path.exists(m_load_dir):
            os.makedirs(m_load_dir)

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_hcurl_source_pbm function with: ')
    print(' ncells = {}'.format(nc))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' source_proj = {}'.format(source_proj))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    print()
    print(' -- building discrete spaces and operators  --')

    t_stamp = time_count()
    print(' .. multi-patch domain...')
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping)
                           for P in domain.interior])
    mappings_list = list(mappings.values())

    if isinstance(ncells, int):
        ncells = [nc, nc]
    else:
        ncells = {patch.name: [nc[i], nc[i]]
                    for (i, patch) in enumerate(domain.interior)}

    # for diagnosttics
    diag_grid = DiagGrid(mappings=mappings, N_diag=100)

    t_stamp = time_count(t_stamp)
    print(' .. derham sequence...')
    derham = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. discrete domain...')
    domain_h = discretize(domain, ncells=ncells)

    t_stamp = time_count(t_stamp)
    print(' .. discrete derham sequence...')
    derham_h = discretize(derham, domain_h, degree=degree)

    t_stamp = time_count(t_stamp)
    print(' .. commuting projection operators...')
    nquads = [4 * (d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    t_stamp = time_count(t_stamp)
    print(' .. multi-patch spaces...')
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
    print(' .. Id operator and matrix...')
    I1 = IdLinearOperator(V1h)
    I1_m = I1.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. Hodge operators...')
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
    print(' .. Hodge matrix H0_m = M0_m ...')
    H0_m = H0.to_sparse_matrix()
    t_stamp = time_count(t_stamp)
    print(' .. dual Hodge matrix dH0_m = inv_M0_m ...')
    dH0_m = H0.get_dual_Hodge_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H1_m = M1_m ...')
    H1_m = H1.to_sparse_matrix()
    t_stamp = time_count(t_stamp)
    print(' .. dual Hodge matrix dH1_m = inv_M1_m ...')
    dH1_m = H1.get_dual_Hodge_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H2_m = M2_m ...')
    H2_m = H2.to_sparse_matrix()
    dH2_m = H2.get_dual_Hodge_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. conforming Projection operators...')
    # conforming Projections (should take into account the boundary conditions
    # of the continuous deRham sequence)
    cP0_m = construct_h1_conforming_projection(V0h, hom_bc=True)
    cP1_m = construct_hcurl_conforming_projection(V1h, hom_bc=True)

    t_stamp = time_count(t_stamp)
    print(' .. broken differential operators...')
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
            uh_bc = P1_phys(u_bc, P1, domain, mappings_list)
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
    print(' .. curl-curl stiffness matrix...')
    print(bD1_m.shape, H2_m.shape)
    pre_CC_m = bD1_m.transpose() @ H2_m @ bD1_m
    # CC_m = cP1_m.transpose() @ pre_CC_m @ cP1_m  # Conga stiffness matrix

    # grad div:
    t_stamp = time_count(t_stamp)
    print(' .. grad-div stiffness matrix...')
    pre_GD_m = - H1_m @ bD0_m @ cP0_m @ dH0_m @ cP0_m.transpose() @ bD0_m.transpose() @ H1_m
    # GD_m = cP1_m.transpose() @ pre_GD_m @ cP1_m  # Conga stiffness matrix

    # jump stabilization:
    t_stamp = time_count(t_stamp)
    print(' .. jump stabilization matrix...')
    jump_penal_m = I1_m - cP1_m
    JP_m = jump_penal_m.transpose() @ H1_m @ jump_penal_m

    t_stamp = time_count(t_stamp)
    print(' .. full operator matrix...')
    print('eta = {}'.format(eta))
    print('mu = {}'.format(mu))
    print('nu = {}'.format(nu))
    print('STABILIZATION: gamma_h = {}'.format(gamma_h))
    # useful for the boundary condition (if present)
    pre_A_m = cP1_m.transpose() @ (eta * H1_m + mu * pre_CC_m - nu * pre_GD_m)
    A_m = pre_A_m @ cP1_m + gamma_h * JP_m

    t_stamp = time_count(t_stamp)
    print()
    print(' -- getting source --')

    f_vect, u_bc, u_ex, curl_u_ex, div_u_ex = get_source_and_solution_hcurl(
            source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,)

    # compute approximate source f_h
    t_stamp = time_count(t_stamp)

    # f_h = L2 projection of f_vect, with filtering if tilde_Pi
    print(' .. projecting the source with ' +
        source_proj +' projection...')

    tilde_f_c = derham_h.get_dual_dofs(
        space='V1',
        f=f_vect,
        backend_language=backend_language,
        return_format='numpy_array')
    if source_proj == 'tilde_Pi':
        print(' .. filtering the discrete source with P0.T ...')
        tilde_f_c = cP1_m.transpose() @ tilde_f_c


    ubc_c = lift_u_bc(u_bc)
    if ubc_c is not None:
        # modified source for the homogeneous pbm
        t_stamp = time_count(t_stamp)
        print(' .. modifying the source with lifted bc solution...')
        tilde_f_c = tilde_f_c - pre_A_m.dot(ubc_c)

    # direct solve with scipy spsolve
    t_stamp = time_count(t_stamp)
    print()
    print(' -- solving source problem with scipy.spsolve...')
    uh_c = spsolve(A_m, tilde_f_c)

    # project the homogeneous solution on the conforming problem space
    if project_sol:
        t_stamp = time_count(t_stamp)
        print(' .. projecting the homogeneous solution on the conforming problem space...')
        uh_c = cP1_m.dot(uh_c)
    else:
        print(' .. NOT projecting the homogeneous solution on the conforming problem space')

    if ubc_c is not None:
        # adding the lifted boundary condition
        t_stamp = time_count(t_stamp)
        print(' .. adding the lifted boundary condition...')
        uh_c += ubc_c

    uh = FemField(V1h, coeffs=array_to_psydac(uh_c, V1h.vector_space))
    #need cp1 here?
    f_c = dH1_m.dot(tilde_f_c)
    jh = FemField(V1h, coeffs=array_to_psydac(f_c, V1h.vector_space))

    t_stamp = time_count(t_stamp)

    print(' -- plots and diagnostics  --')
    if plot_dir:
        OM = OutputManager(plot_dir + '/spaces.yml', plot_dir + '/fields.h5')
        OM.add_spaces(V1h=V1h)
        OM.set_static()
        OM.export_fields(vh=uh)
        OM.export_fields(jh=jh)
        OM.export_space_info()
        OM.close()

        PM = PostProcessManager(
            domain=domain,
            space_file=plot_dir +
            '/spaces.yml',
            fields_file=plot_dir +
            '/fields.h5')
        PM.export_to_vtk(
            plot_dir + "/sol",
            grid=None,
            npts_per_cell=[6] * 2,
            snapshots='all',
            fields='vh')
        PM.export_to_vtk(
            plot_dir + "/source",
            grid=None,
            npts_per_cell=[6] * 2,
            snapshots='all',
            fields='jh')

        PM.close()

    time_count(t_stamp)

    if u_ex:
        u_ex_c = P1_phys(u_ex, P1, domain, mappings_list).coeffs.toarray()
        err = u_ex_c - uh_c
        l2_error = np.sqrt(np.dot(err, H1_m.dot(err)))/np.sqrt(np.dot(u_ex_c,H1_m.dot(u_ex_c)))
        print(l2_error)
        #return l2_error
        diags['err'] = l2_error

    return diags
