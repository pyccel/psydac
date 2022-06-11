# coding: utf-8

from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict

from sympy import lambdify
from scipy.sparse.linalg import spsolve

from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral, Norm
from sympde.topology  import Derham
from sympde.topology import element_of


from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.feec.multipatch.api import discretize
from psydac.feec.pull_push      import pull_2d_h1

from psydac.feec.multipatch.fem_linear_operators        import IdLinearOperator
from psydac.feec.multipatch.operators                   import HodgeOperator
from psydac.feec.multipatch.plotting_utilities          import plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution_h1
from psydac.feec.multipatch.utils_conga_2d              import DiagGrid, P0_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities                   import time_count

from psydac.linalg.utilities import array_to_stencil
from psydac.fem.basic        import FemField

def solve_h1_source_pbm(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_proj='P_L2', source_type='manu_poisson',
        eta=-10., mu=1., gamma_h=10.,
        project_sol=False,
        plot_source=False, plot_dir=None, hide_plots=True, skip_titles=False,
        m_load_dir="", sol_filename="", sol_ref_filename="",
        ref_nc=None, ref_deg=None,
):
    """
    solver for the problem: find u in H^1, such that

      A u = f             on \Omega
        u = u_bc          on \partial \Omega

    where the operator

      A u := eta * u  -  mu * div grad u

    is discretized as  Ah: V0h -> V0h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \Omega,

      V0h  --grad->  V1h  -—curl-> V2h

    Examples:

      - Helmholtz equation with
          eta = -omega**2
          mu  = 1

      - Poisson equation with Laplace operator L = A,
          eta = 0
          mu  = 1

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma_h: jump penalization (stabilization) parameter
    :param source_proj: approximation operator for the source, possible values are
         - 'P_geom':    primal commuting projection based on geometric dofs
         - 'P_L2':      L2 projection on the broken space
         - 'tilde_Pi':  dual commuting projection, an L2 projection filtered by the adjoint conforming projection)
    :param source_type: must be implemented in get_source_and_solution()
    """
    diags = {}
    ncells = [nc, nc]
    degree = [deg,deg]

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_h1_source_pbm function with: ')
    print(' ncells = {}'.format(ncells))
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
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    t_stamp = time_count(t_stamp)
    print(' .. derham sequence...')
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. discrete domain...')
    domain_h = discretize(domain, ncells=ncells)

    t_stamp = time_count(t_stamp)
    print(' .. discrete derham sequence...')
    derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    t_stamp = time_count(t_stamp)
    print(' .. commuting projection operators...')
    nquads = [4*(d + 1) for d in degree]
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

    I0 = IdLinearOperator(V0h)
    I0_m = I0.to_sparse_matrix()

    print(' .. Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=0)
    H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=1)

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H0_m = M0_m ...')
    H0_m  = H0.to_sparse_matrix()        # = mass matrix of V0
    t_stamp = time_count(t_stamp)
    print(' .. dual Hodge matrix dH0_m = inv_M0_m ...')
    dH0_m = H0.get_dual_sparse_matrix()  # = inverse mass matrix of V0

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H1_m = M1_m ...')
    H1_m  = H1.to_sparse_matrix()  # = mass matrix of V1

    t_stamp = time_count(t_stamp)
    print(' .. conforming Projection operators...')
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    cP0 = derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)    
    cP0_m = cP0.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. broken differential operators...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    # bD1_m = bD1.to_sparse_matrix()

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    def lift_u_bc(u_bc):
        if u_bc is not None:
            print('lifting the boundary condition in V0h...  [warning: Not Tested Yet!]')
            # note: for simplicity we apply the full P0 on u_bc, but we only need to set the boundary dofs
            uh_bc = P0_phys(u_bc, P0, domain, mappings_list)
            ubc_c = uh_bc.coeffs.toarray()
            # removing internal dofs (otherwise ubc_c may already be a very good approximation of uh_c ...)
            ubc_c = ubc_c - cP0_m.dot(ubc_c)
        else:
            ubc_c = None
        return ubc_c

    print(' .. div grad operator...')
    # Conga (projection-based) stiffness matrices:
    # div grad:
    pre_DG_m = - bD0_m.transpose() @ H1_m @ bD0_m

    # jump penalization:
    print(' .. jump stabilization matrix...')
    jump_penal_m = I0_m - cP0_m
    JP0_m = jump_penal_m.transpose() * H0_m * jump_penal_m

    print(' .. full operator matrix...')
    print('eta = {}'.format(eta))
    print('mu = {}'.format(mu))
    pre_A_m = cP0_m.transpose() @ ( eta * H0_m - mu * pre_DG_m )  # useful for the boundary condition (if present)
    A_m = pre_A_m @ cP0_m + gamma_h * JP0_m

    t_stamp = time_count(t_stamp)
    print()
    print(' -- getting source --')
    f_scal, u_bc, u_ex = get_source_and_solution_h1(
        source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,
    )
    # compute approximate source f_h
    # b_c = f_c = None
    tilde_f_c = f_c = None
    if source_proj == 'P_geom':
        print(' .. projecting the source with commuting projection P0...')
        # f = lambdify(domain.coordinates, f_scal)
        # f_log = [pull_2d_h1(f, m) for m in mappings_list]
        # f_h = P0(f_log)
        f_h = P0_phys(f_scal, P0, domain, mappings_list)        
        f_c = f_h.coeffs.toarray()
        tilde_f_c = H0_m.dot(f_c)

    elif source_proj in ['P_L2', 'tilde_Pi']:
        print(' .. projecting the source with '+source_proj+' projection...')
        tilde_f_c = derham_h.get_dual_dofs(space='V0', f=f_scal, backend_language=backend_language, return_format='numpy_array')
        if source_proj == 'tilde_Pi':
            print(' .. filtering the discrete source with P0.T ...')
            tilde_f_c = cP0_m.transpose() @ tilde_f_c
            
    else:
        raise ValueError(source_proj)

    if plot_source:
        if f_c is None:
            f_c = dH0_m.dot(tilde_f_c)
        title = 'f_h with P = ' + source_proj
        if skip_titles:
            title = ''
        plot_field(numpy_coeffs=f_c, Vh=V0h, space_kind='h1', domain=domain, title=title, 
            plot_type='components',
            filename=plot_dir+'/fh_'+source_proj+'.pdf', hide_plot=hide_plots)

    ubc_c = lift_u_bc(u_bc)
    if ubc_c is not None:
        # modified source for the homogeneous pbm
        t_stamp = time_count(t_stamp)
        print('modifying the source with lifted bc solution...')
        tilde_f_c = tilde_f_c - pre_A_m.dot(ubc_c)

    print()
    print(' -- ref solution: writing values on diag grid  --')
    diag_grid = DiagGrid(mappings=mappings, N_diag=100)
    if u_ex is not None:
        print(' .. u_ex is known:')
        print('    setting uh_ref = P_geom(u_ex)')
        uh_ref = P0_phys(u_ex, P0, domain, mappings_list)
        diag_grid.write_sol_ref_values(uh_ref, space='V0')
    else:
        print(' .. u_ex is unknown:')
        print('    importing uh_ref in ref_V1h from file {}...'.format(sol_ref_filename))
        diag_grid.create_ref_fem_spaces(domain=domain, ref_nc=ref_nc, ref_deg=ref_deg)
        diag_grid.import_ref_sol_from_coeffs(sol_ref_filename, space='V0')
        diag_grid.write_sol_ref_values(space='V0')


    # direct solve with scipy spsolve
    t_stamp = time_count(t_stamp)
    print()
    print(' -- solving source problem with scipy.spsolve...')
    uh_c = spsolve(A_m, tilde_f_c)

    # project the homogeneous solution on the conforming problem space
    if project_sol:
        print(' .. projecting the homogeneous solution on the conforming problem space...')
        uh_c = cP0_m.dot(uh_c)

    if ubc_c is not None:
        # adding the lifted boundary condition
        print(' .. adding the lifted boundary condition...')
        uh_c += ubc_c

    uh = FemField(V0h, coeffs=array_to_stencil(uh_c, V0h.vector_space))
    t_stamp = time_count(t_stamp)

    print()
    print(' -- plots and diagnostics  --')
    if plot_dir:
        print(' .. plotting the FEM solution...')
        title = r'solution $\phi_h$' # (amplitude)'
        params_str = 'eta={}_mu={}_gamma_h={}'.format(eta, mu, gamma_h)
        if skip_titles:
            title = ''
        plot_field(numpy_coeffs=uh_c, Vh=V0h, space_kind='h1', plot_type='components',
            domain=domain, title=title, filename=plot_dir+'/'+params_str+'_phi_h.pdf', hide_plot=hide_plots)
    if sol_filename:
        print(' .. saving solution coeffs to file {}'.format(sol_filename))
        np.save(sol_filename, uh_c)

    # diagnostics: errors
    err_diags = diag_grid.get_diags_for(v=uh, space='V0')
    for key, value in err_diags.items():
        diags[key] = value
    
    if u_ex is not None:
        check_diags = get_Vh_diags_for(v=uh, v_ref=uh_ref, M_m=H0_m, msg='error between Ph(u_ex) and u_h')
        diags['norm_Pu_ex'] = check_diags['sol_ref_norm']
        diags['rel_l2_error_in_Vh'] = check_diags['rel_l2_error']

    return diags


if __name__ == '__main__':
    # quick run, to test 

    t_stamp_full = time_count()

    omega = np.sqrt(170) # source
    roundoff = 1e4
    eta = int(-omega**2 * roundoff)/roundoff
    source_type = 'manu_poisson'

    domain_name = 'curved_L_shape'
    nc = 4
    deg = 2

    run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
    solve_h1_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        mu=1, #1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_source=True,
        plot_dir='./plots/h1_tests_source_february/'+run_dir,
        hide_plots=True,
    )

    time_count(t_stamp_full, msg='full program')
