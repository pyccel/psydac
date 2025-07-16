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
import numpy as np

from sympde.topology import Derham


from psydac.api.discretization import discretize
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases import get_source_and_solution_hcurl
from psydac.feec.multipatch.utils_conga_2d import P1_phys
from psydac.feec.multipatch.utilities import time_count
# from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic import FemField
from psydac.api.postprocessing import OutputManager, PostProcessManager

from psydac.linalg.basic       import IdentityOperator
from psydac.fem.projectors import get_dual_dofs
from psydac.linalg.solvers     import inverse


def solve_hcurl_source_pbm(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_proj='tilde_Pi', source_type='manu_J',
        eta=-10., mu=1., nu=1., gamma_h=10.,
        project_sol=True, plot_dir=None, 
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
    # mappings = OrderedDict([(P.logical_domain, P.mapping)
    #                        for P in domain.interior])
    # mappings_list = list(mappings.values())

    if isinstance(nc, int):
        ncells = [nc, nc]
    else:
        ncells = {patch.name: [nc[i], nc[i]]
                    for (i, patch) in enumerate(domain.interior)}


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
    nquads = [10 * (d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    t_stamp = time_count(t_stamp)
    print(' .. multi-patch spaces...')
    V0h, V1h, V2h = derham_h.spaces
    mappings = derham_h.callable_mapping

    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))
    diags['ndofs_V0'] = V0h.nbasis
    diags['ndofs_V1'] = V1h.nbasis
    diags['ndofs_V2'] = V2h.nbasis

    t_stamp = time_count(t_stamp)
    print(' .. Id operator and matrix...')
    I1 = IdentityOperator(V1h.coeff_space)

    t_stamp = time_count(t_stamp)
    print(' .. Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    # other option: define as Hodge Operators:
    H0, H1, H2 = derham_h.Hodge_operators(kind='linop', backend_language=backend_language)
    dH0, dH1, dH2 = derham_h.Hodge_operators(kind='linop', dual=True, backend_language=backend_language)


    t_stamp = time_count(t_stamp)
    print(' .. conforming Projection operators...')
    # conforming Projections (should take into account the boundary conditions
    # of the continuous deRham sequence)
    cP0, cP1, cP2 = derham_h.conforming_projectors(kind='linop', hom_bc = True)


    t_stamp = time_count(t_stamp)
    print(' .. broken differential operators...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.derivatives(kind='linop')

    # Conga (projection-based) stiffness matrices
    # curl curl:
    t_stamp = time_count(t_stamp)
    print(' .. curl-curl stiffness matrix...')
    pre_CC = bD1.T @ H2 @ bD1

    # grad div:
    t_stamp = time_count(t_stamp)
    print(' .. grad-div stiffness matrix...')
    pre_GD = - H1 @ bD0 @ cP0 @ dH0 @ cP0.T @ bD0.T @ H1

    # jump stabilization:
    t_stamp = time_count(t_stamp)
    print(' .. jump stabilization matrix...')
    JS = (I1 - cP1).T @ H1 @ (I1 - cP1)


    t_stamp = time_count(t_stamp)
    print(' .. full operator matrix...')
    print('eta = {}'.format(eta))
    print('mu = {}'.format(mu))
    print('nu = {}'.format(nu))
    print('STABILIZATION: gamma_h = {}'.format(gamma_h))
    # useful for the boundary condition (if present)
    pre_A = eta * cP1.T @  H1 
    if mu != 0: 
        pre_A += mu * cP1.T @ pre_CC
    if nu != 0: 
        pre_A -= nu * cP1.T @ pre_GD

    A = pre_A @ cP1 + gamma_h * JS

    t_stamp = time_count(t_stamp)
    print()
    print(' -- getting source --')

    f_vect, u_bc, u_ex, curl_u_ex, div_u_ex = get_source_and_solution_hcurl(source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,)

    # compute approximate source f_h
    t_stamp = time_count(t_stamp)

    # f_h = L2 projection of f_vect, with filtering if tilde_Pi
    print(' .. projecting the source with ' + source_proj +' projection...')

    tilde_f = get_dual_dofs(Vh=V1h, f=f_vect, domain_h=domain_h, backend_language=backend_language)

    if source_proj == 'tilde_Pi':
        print(' .. filtering the discrete source with P1.T ...')
        tilde_f = cP1.T @ tilde_f

    def lift_u_bc(u_bc):
        if u_bc is not None:
            ubc = P1_phys(u_bc, P1, domain).coeffs
            ubc = ubc - cP1.dot(ubc)

        else:
            ubc = None
            
        return ubc


    ubc = lift_u_bc(u_bc)

    if ubc is not None:
        # modified source for the homogeneous pbm
        t_stamp = time_count(t_stamp)
        print(' .. modifying the source with lifted bc solution...')
        tilde_f = tilde_f - pre_A.dot(ubc)

    # direct solve with scipy spsolve
    t_stamp = time_count(t_stamp)
    print('solving source problem with conjugate gradient...')
    solver = inverse(A, solver='cg', tol=1e-8)
    u = solver.solve(tilde_f)

    # project the homogeneous solution on the conforming problem space
    t_stamp = time_count(t_stamp)
    if project_sol:
        print(' .. projecting the homogeneous solution on the conforming problem space...')
        u = cP1.dot(u)

    if ubc is not None:
        # adding the lifted boundary condition
        t_stamp = time_count(t_stamp)
        print(' .. adding the lifted boundary condition...')
        u += ubc

    uh = FemField(V1h, coeffs=u)
    #need cp1 here?
    f = dH1.dot(tilde_f)
    jh = FemField(V1h, coeffs=f)

    t_stamp = time_count(t_stamp)

    print(' -- plots and diagnostics  --')
    if plot_dir:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

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
        u_ex_p = P1_phys(u_ex, P1, domain).coeffs
    
        err = u_ex_p - u
        print(err.inner(H1.dot(err)))
        l2_error = np.sqrt( err.inner(H1.dot(err))) / np.sqrt(u_ex_p.inner(H1.dot(u_ex_p)))
        print(l2_error)
        diags['err'] = l2_error

    return diags
