# coding: utf-8

from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict

from sympy import lambdify, Matrix

from scipy.sparse.linalg import spsolve

from sympde.calculus  import dot
from sympde.topology  import element_of
from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral, Norm
from sympde.topology  import Derham

from psydac.api.settings   import PSYDAC_BACKENDS
from psydac.feec.pull_push import pull_2d_hcurl

from psydac.feec.multipatch.api                         import discretize
from psydac.feec.multipatch.fem_linear_operators        import IdLinearOperator
from psydac.feec.multipatch.operators                   import HodgeOperator
from psydac.feec.multipatch.plotting_utilities          import plot_field #, write_field_to_diag_grid, 
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution_hcurl
from psydac.feec.multipatch.utils_conga_2d              import DiagGrid, P0_phys, P1_phys, P2_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities                   import time_count #, export_sol, import_sol
from psydac.linalg.utilities                            import array_to_psydac
from psydac.fem.basic                                   import FemField
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution_OBSOLETE

from psydac.feec.multipatch.non_matching_operators import construct_scalar_conforming_projection, construct_vector_conforming_projection
from psydac.api.postprocessing import OutputManager, PostProcessManager

def solve_hcurl_source_pbm_nc(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_proj='P_geom', source_type='manu_J',
        eta=-10., mu=1., nu=1., gamma_h=10.,     
        project_sol=False,
        plot_source=False, plot_dir=None, hide_plots=True, skip_plot_titles=False,
        cb_min_sol=None, cb_max_sol=None,
        m_load_dir=None, sol_filename="", sol_ref_filename="",
        ref_nc=None, ref_deg=None, test=False
):
    """
    solver for the problem: find u in H(curl), such that

      A u = f             on \Omega
      n x u = n x u_bc    on \partial \Omega

    where the operator

      A u := eta * u  +  mu * curl curl u  -  nu * grad div u

    is discretized as  Ah: V1h -> V1h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \Omega,

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
         - 'P_geom':    primal commuting projection based on geometric dofs
         - 'P_L2':      L2 projection on the broken space
         - 'tilde_Pi':  dual commuting projection, an L2 projection filtered by the adjoint conforming projection)
    :param source_type: must be implemented in get_source_and_solution()
    :param m_load_dir: directory for matrix storage
    """
    diags = {}

    ncells = nc
    degree = [deg,deg]

    # if backend_language is None:
    #     if domain_name in ['pretzel', 'pretzel_f'] and nc > 8:
    #         backend_language='numba'
    #     else:
    #         backend_language='python'
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

    print()
    print(' -- building discrete spaces and operators  --')

    t_stamp = time_count()
    print(' .. multi-patch domain...')
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    ncells_h = {patch.name: [ncells[i], ncells[i]] for (i,patch) in enumerate(domain.interior)}

    #corners in pretzel [2, 2, 2*,2*, 2, 1, 1, 1, 1, 1, 0, 0, 1, 2*, 2*, 2, 0, 0, 0  ]
    #ncells = np.array([8, 8, 16, 16, 8, 4, 4, 4, 4, 4, 2, 2, 4, 16, 16, 8, 2, 2, 2])
    #ncells = np.array([4 for _ in range(18)])

    # for diagnosttics
    diag_grid = DiagGrid(mappings=mappings, N_diag=100)

    t_stamp = time_count(t_stamp)
    print(' .. derham sequence...')
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. discrete domain...')
    domain_h = discretize(domain, ncells=ncells_h)

    t_stamp = time_count(t_stamp)
    print(' .. discrete derham sequence...')
    derham_h = discretize(derham, domain_h, degree=degree)

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

    t_stamp = time_count(t_stamp)
    print(' .. Id operator and matrix...')
    I1 = IdLinearOperator(V1h)
    I1_m = I1.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    # other option: define as Hodge Operators:
    H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=0)
    H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=1)
    H2 = HodgeOperator(V2h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=2)

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H0_m = M0_m ...')
    H0_m  = H0.to_sparse_matrix()              
    t_stamp = time_count(t_stamp)
    print(' .. dual Hodge matrix dH0_m = inv_M0_m ...')
    dH0_m = H0.get_dual_Hodge_sparse_matrix()  

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H1_m = M1_m ...')
    H1_m  = H1.to_sparse_matrix()              
    t_stamp = time_count(t_stamp)
    print(' .. dual Hodge matrix dH1_m = inv_M1_m ...')
    dH1_m = H1.get_dual_Hodge_sparse_matrix()  

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H2_m = M2_m ...')
    H2_m = H2.to_sparse_matrix()              
    dH2_m = H2.get_dual_Hodge_sparse_matrix()              

    t_stamp = time_count(t_stamp)
    print(' .. conforming Projection operators...')
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    #cP0 = derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
    #cP1 = derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
    #cP0_m = cP0.to_sparse_matrix()
    #cP1_m = cP1.to_sparse_matrix()

    # Try the NC one
    cP1_m = construct_vector_conforming_projection(V1h, hom_bc=[True, True])
    cP0_m = construct_scalar_conforming_projection(V0h, hom_bc=[True, True])

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
            # note: for simplicity we apply the full P1 on u_bc, but we only need to set the boundary dofs
            uh_bc = P1_phys(u_bc, P1, domain, mappings_list)
            ubc_c = uh_bc.coeffs.toarray()
            # removing internal dofs (otherwise ubc_c may already be a very good approximation of uh_c ...)
            ubc_c = ubc_c - cP1_m.dot(ubc_c)
        else:
            ubc_c = None
        return ubc_c

    # Conga (projection-based) stiffness matrices
    # curl curl:
    t_stamp = time_count(t_stamp)
    print(' .. curl-curl stiffness matrix...')
    print(bD1_m.shape, H2_m.shape )
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
    JP_m = jump_penal_m.transpose() * H1_m * jump_penal_m

    t_stamp = time_count(t_stamp)
    print(' .. full operator matrix...')
    print('eta = {}'.format(eta))
    print('mu = {}'.format(mu))
    print('nu = {}'.format(nu))
    print('STABILIZATION: gamma_h = {}'.format(gamma_h))
    pre_A_m = cP1_m.transpose() @ ( eta * H1_m + mu * pre_CC_m - nu * pre_GD_m )  # useful for the boundary condition (if present)
    A_m = pre_A_m @ cP1_m + gamma_h * JP_m

    t_stamp = time_count(t_stamp)
    print()
    print(' -- getting source --')
    if source_type == 'manu_maxwell':
        f_scal, f_vect, u_bc, p_ex, u_ex, phi, grad_phi = get_source_and_solution_OBSOLETE(
        source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,
        )
    else:
        f_vect, u_bc, u_ex, curl_u_ex, div_u_ex = get_source_and_solution_hcurl(
        source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,
        )
    # compute approximate source f_h
    t_stamp = time_count(t_stamp)
    tilde_f_c = f_c = None
    if source_proj == 'P_geom':
        # f_h = P1-geometric (commuting) projection of f_vect
        print(' .. projecting the source with primal (geometric) commuting projection...')
        f_h = P1_phys(f_vect, P1, domain, mappings_list)
        f_c = f_h.coeffs.toarray()
        tilde_f_c = H1_m.dot(f_c)

    elif source_proj in ['P_L2', 'tilde_Pi']:
        # f_h = L2 projection of f_vect, with filtering if tilde_Pi
        print(' .. projecting the source with '+source_proj+' projection...')
        tilde_f_c = derham_h.get_dual_dofs(space='V1', f=f_vect, backend_language=backend_language, return_format='numpy_array')
        if source_proj == 'tilde_Pi':
            print(' .. filtering the discrete source with P0.T ...')
            tilde_f_c = cP1_m.transpose() @ tilde_f_c
    else:
        raise ValueError(source_proj)



    if plot_source:
        if True:
            title = ''
            title_vf = ''
        else:
            title = 'f_h with P = '+source_proj
            title_vf = 'f_h with P = '+source_proj
        if f_c is None:
            f_c = dH1_m.dot(tilde_f_c)
        plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, 
            title=title, filename=plot_dir+'/fh_'+source_proj+'.pdf', hide_plot=hide_plots)
        plot_field(numpy_coeffs=f_c, Vh=V1h, plot_type='vector_field', space_kind='hcurl', domain=domain, 
            title=title_vf, filename=plot_dir+'/fh_'+source_proj+'_vf.pdf', hide_plot=hide_plots)

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
    f_c = dH1_m.dot(tilde_f_c)
    jh = FemField(V1h, coeffs=array_to_psydac(f_c, V1h.vector_space))

    t_stamp = time_count(t_stamp)

    print()
    print(' -- plots and diagnostics  --')
    if plot_dir:
        print(' .. plotting the FEM solution...')
        if skip_plot_titles:
            title = ''
            title_vf = ''
        else:
            title = r'solution $u_h$ (amplitude) for $\eta = $'+repr(eta)
            title_vf = r'solution $u_h$ for $\eta = $'+repr(eta)
        params_str = 'eta={}_mu={}_nu={}_gamma_h={}_Pf={}'.format(eta, mu, nu, gamma_h, source_proj)
        plot_field(numpy_coeffs=uh_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
            filename=plot_dir+'/'+params_str+'_uh.pdf', 
            plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)
        plot_field(numpy_coeffs=uh_c, Vh=V1h, space_kind='hcurl', domain=domain, title=title_vf, 
            filename=plot_dir+'/'+params_str+'_uh_vf.pdf', 
            plot_type='vector_field', hide_plot=hide_plots)
    
        OM = OutputManager(plot_dir+'/spaces.yml', plot_dir+'/fields.h5')
        OM.add_spaces(V1h=V1h)
        OM.set_static()
        OM.export_fields(vh = uh)
        OM.export_fields(jh = jh)
        OM.export_space_info()
        OM.close()

        PM = PostProcessManager(domain=domain, space_file=plot_dir+'/spaces.yml', fields_file=plot_dir+'/fields.h5' )
        PM.export_to_vtk(plot_dir+"/sol",grid=None, npts_per_cell=[6]*2,snapshots='all', fields='vh' )
        PM.export_to_vtk(plot_dir+"/source",grid=None, npts_per_cell=[6]*2,snapshots='all', fields='jh' )

        PM.close()

    time_count(t_stamp)

    if test:
        u         = element_of(V1h.symbolic_space, name='u')
        l2norm    = Norm(Matrix([u[0] - u_ex[0],u[1] - u_ex[1]]), domain, kind='l2')
        l2norm_h  = discretize(l2norm, domain_h, V1h)
        uh_c      = array_to_psydac(uh_c, V1h.vector_space)
        l2_error  = l2norm_h.assemble(u=FemField(V1h, coeffs=uh_c))
        print(l2_error)
        return l2_error


    return diags