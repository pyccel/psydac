# coding: utf-8

from pytest import param
from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from sympy import lambdify, Matrix

from scipy.sparse.linalg import spsolve
from scipy import special

from sympde.calculus  import dot
from sympde.topology  import element_of
from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral, Norm
from sympde.topology  import Derham

from psydac.api.settings   import PSYDAC_BACKENDS
from psydac.feec.pull_push import pull_2d_hcurl

from psydac.feec.multipatch.api                         import discretize
from psydac.feec.multipatch.fem_linear_operators        import IdLinearOperator
from psydac.feec.multipatch.operators                   import HodgeOperator, get_K0_and_K0_inv, get_K1_and_K1_inv
from psydac.feec.multipatch.plotting_utilities          import plot_field #, write_field_to_diag_grid, 
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution_hcurl, get_div_free_pulse, get_curl_free_pulse, get_Delta_phi_pulse
from psydac.feec.multipatch.utils_conga_2d              import DiagGrid, P0_phys, P1_phys, P2_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities                   import time_count #, export_sol, import_sol
from psydac.linalg.utilities                            import array_to_stencil
from psydac.fem.basic                                   import FemField

def solve_td_maxwell_pbm(
        nc=4, deg=4, Nt_pp=None, cfl=.8, nb_t_periods=20, omega=20, source_is_harmonic=True,
        domain_name='pretzel_f', backend_language=None, source_proj='P_geom', source_type='manu_J',
        conf_proj='BSP', gamma_h=10.,     
        project_sol=False, filter_source=True,
        E0_type='zero', E0_proj='P_L2', 
        plot_source=False, plot_dir=None, plot_divE=False, hide_plots=True, plot_time_ranges=None, diag_dtau=None,
        cb_min_sol=None, cb_max_sol=None,
        m_load_dir="", th_sol_filename="", sol_ref_filename="",
        ref_nc=None, ref_deg=None,
):
    """
    solver for the TD Maxwell problem: find E(t) in H(curl), B in L2, such that

      dt E - curl B = -J             on \Omega
      dt B + curl E = 0              on \Omega
      n x E = n x E_bc      on \partial \Omega

    with Ampere discretized weakly and Faraday discretized strongly, in a broken-FEEC approach on a 2D multipatch domain \Omega,

      V0h  --grad->  V1h  -—curl-> V2h
                     (Eh)          (Bh)

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma_h: jump penalization parameter
    :param source_proj: approximation operator for the source, possible values are 'P_geom' or 'P_L2'
    :param source_type: must be implemented in get_source_and_solution()
    :param m_load_dir: directory for matrix storage
    """
    diags = {}

    ncells = [nc, nc]
    degree = [deg,deg]

    period_time = 2*np.pi/omega
    final_time = nb_t_periods * period_time

    if plot_time_ranges is None:
        plot_time_ranges = [[0, final_time], 1]

    if diag_dtau is None:
        diag_dtau = nb_t_periods//10

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
    print('Starting solve_td_maxwell_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' E0_type = {}'.format(E0_type))
    print(' E0_proj = {}'.format(E0_proj))
    print(' source_type = {}'.format(source_type))
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

    # for diagnosttics
    diag_grid = DiagGrid(mappings=mappings, N_diag=100)

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
    dH0_m = H0.get_dual_sparse_matrix()  

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H1_m = M1_m ...')
    H1_m  = H1.to_sparse_matrix()              
    t_stamp = time_count(t_stamp)
    print(' .. dual Hodge matrix dH1_m = inv_M1_m ...')
    dH1_m = H1.get_dual_sparse_matrix()  

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix dH2_m = M2_m ...')
    H2_m = H2.to_sparse_matrix()              

    t_stamp = time_count(t_stamp)
    print(' .. conforming Projection operators...')
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    cP0 = derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
    cP1 = derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
    cP0_m = cP0.to_sparse_matrix()
    cP1_m = cP1.to_sparse_matrix()
    if conf_proj == 'GSP':
        print(' [* GSP-conga: using Geometric Spline conf Projections ]')
        K0, K0_inv = get_K0_and_K0_inv(V0h, uniform_patches=True)
        cP0_m = K0_inv @ cP0_m @ K0
        K1, K1_inv = get_K1_and_K1_inv(V1h, uniform_patches=True)
        cP1_m = K1_inv @ cP1_m @ K1
    elif conf_proj == 'BSP':
        print(' [* BSP-conga: using B-Spline conf Projections ]')
    else:
        raise ValueError(conf_proj)

    t_stamp = time_count(t_stamp)
    print(' .. broken differential operators...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()

    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Conga (projection-based) matrices
    t_stamp = time_count(t_stamp)    
    print(' .. matrix of the primal curl (in primal bases)...')
    C_m = bD1_m @ cP1_m
    print(' .. matrix of the dual curl (also in primal bases)...')
    dC_m = dH1_m @ C_m.transpose() @ H2_m
    print(' .. matrix of the dual div (still in primal bases)...')
    div_m = dH0_m @ cP0_m.transpose() @ bD0_m.transpose() @ H1_m

    # jump stabilization (may not be needed)
    t_stamp = time_count(t_stamp)
    print(' .. jump stabilization matrix...')
    jump_penal_m = I1_m - cP1_m
    JP_m = jump_penal_m.transpose() * H1_m * jump_penal_m

    # t_stamp = time_count(t_stamp)
    # print(' .. full operator matrix...')
    # print('STABILIZATION: gamma_h = {}'.format(gamma_h))
    # pre_A_m = cP1_m.transpose() @ ( eta * H1_m + mu * pre_CC_m - nu * pre_GD_m )  # useful for the boundary condition (if present)
    # A_m = pre_A_m @ cP1_m + gamma_h * JP_m

    if Nt_pp is None:
        assert 0 < cfl <= 1
        Nt_pp, dt = compute_stable_dt(cfl, period_time, C_m, dC_m, V1h.nbasis)
    else:
        dt = period_time/Nt_pp
    Nt = Nt_pp * nb_t_periods

    def is_plotting_time(nt):
        answer = (nt+1==Nt)
        for tr, tp in plot_time_ranges:
            if tp is None:
                if source_is_harmonic:
                    tp = max(Nt_pp//10,1)
                elif source_type == 'Il_pulse':
                    tp = max(Nt_pp//4,1)
                else:
                    tp = max(Nt_pp,1)
            if answer:
                break
            answer = (tr[0]*period_time <= nt*dt <= tr[1]*period_time and (nt+1)%tp == 0)
        return answer

    debug = False

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # source

    t_stamp = time_count(t_stamp)
    print()
    print(' -- getting source --')
    f0_c = None
    f0_harmonic_c = None
    if source_type == 'zero':
        pass

    elif source_type == 'pulse':

        f0 = get_div_free_pulse(x_0=1.0, y_0=1.0, domain=domain)

    elif source_type == 'cf_pulse':

        f0 = get_curl_free_pulse(x_0=1.0, y_0=1.0, domain=domain)

    elif source_type == 'Il_pulse':  #Issautier-like pulse
        # source will be 
        #   J = curl A + cos(om*t) * grad phi
        # so that 
        #   dt rho = - div J = - cos(om*t) Delta phi
        # for instance, with rho(t=0) = 0 this  gives
        #   rho = - sin(om*t)/om * Delta phi
        # and Gauss' law reads
        #  div E = rho = - sin(om*t)/om * Delta phi
        f0 = get_div_free_pulse(x_0=1.0, y_0=1.0, domain=domain)  # this is curl A
        f0_harmonic = get_curl_free_pulse(x_0=1.0, y_0=1.0, domain=domain) # this is grad phi
        assert not source_is_harmonic

        rho0 = get_Delta_phi_pulse(x_0=1.0, y_0=1.0, domain=domain) # this is Delta phi
        tilde_rho0_c = derham_h.get_dual_dofs(space='V0', f=rho0, backend_language=backend_language, return_format='numpy_array')
        tilde_rho0_c = cP0_m.transpose() @ tilde_rho0_c
        rho0_c = dH0_m.dot(tilde_rho0_c)

    else:

        f0, u_bc, u_ex, curl_u_ex, div_u_ex = get_source_and_solution_hcurl(
            source_type=source_type, domain=domain, domain_name=domain_name,
        )
        assert u_bc is None  # only homogeneous BC's for now

    # f0_c = np.zeros(V1h.nbasis)

    def source_enveloppe(tau):        
        return 1

    if source_is_harmonic:
        f0_harmonic = f0
        f0 = None
        if E0_type == 'th_sol':
            # use source enveloppe for smooth transition from 0 to 1
            def source_enveloppe(tau):        
                return (special.erf((tau/25)-2)-special.erf(-2))/2

    t_stamp = time_count(t_stamp)
    tilde_f0_c = f0_c = None
    tilde_f0_harmonic_c = f0_harmonic_c = None
    if source_proj == 'P_geom':
        print(' .. projecting the source with commuting projection...')
        if f0 is not None:
            f0_h = P1_phys(f0, P1, domain, mappings_list)
            f0_c = f0_h.coeffs.toarray()
            tilde_f0_c = H1_m.dot(f0_c)  
        if f0_harmonic is not None:
            f0_harmonic_h = P1_phys(f0_harmonic, P1, domain, mappings_list)
            f0_harmonic_c = f0_harmonic_h.coeffs.toarray()
            tilde_f0_harmonic_c = H1_m.dot(f0_harmonic_c) 

    elif source_proj == 'P_L2':
        # helper: save/load coefs
        if f0 is not None:
            if source_type == 'Il_pulse':
                source_name = 'Il_pulse_f0'
            else:
                source_name = source_type
            sdd_filename = m_load_dir+'/'+source_name+'_dual_dofs.npy'
            if os.path.exists(sdd_filename):
                print(' .. loading source dual dofs from file {}'.format(sdd_filename))
                tilde_f0_c = np.load(sdd_filename)
            else:
                print(' .. projecting the source f0 with L2 projection...')
                tilde_f0_c = derham_h.get_dual_dofs(space='V1', f=f0, backend_language=backend_language, return_format='numpy_array')
                print(' .. saving source dual dofs to file {}'.format(sdd_filename))
                np.save(sdd_filename, tilde_f0_c)
        if f0_harmonic is not None:
            if source_type == 'Il_pulse':
                source_name = 'Il_pulse_f0_harmonic'
            else:
                source_name = source_type
            sdd_filename = m_load_dir+'/'+source_name+'_dual_dofs.npy'
            if os.path.exists(sdd_filename):
                print(' .. loading source dual dofs from file {}'.format(sdd_filename))
                tilde_f0_harmonic_c = np.load(sdd_filename)
            else:
                print(' .. projecting the source f0_harmonic with L2 projection...')
                tilde_f0_harmonic_c = derham_h.get_dual_dofs(space='V1', f=f0_harmonic, backend_language=backend_language, return_format='numpy_array')
                print(' .. saving source dual dofs to file {}'.format(sdd_filename))
                np.save(sdd_filename, tilde_f0_harmonic_c)

    else:
        raise ValueError(source_proj)

    t_stamp = time_count(t_stamp)
    if filter_source:
        print(' .. filtering the source...')
        if tilde_f0_c is not None:
            tilde_f0_c = cP1_m.transpose() @ tilde_f0_c            
        if tilde_f0_harmonic_c is not None:
            tilde_f0_harmonic_c = cP1_m.transpose() @ tilde_f0_harmonic_c

    if tilde_f0_c is not None:
        f0_c = dH1_m.dot(tilde_f0_c)

        if debug:
            title = 'f0 part of source'
            params_str = 'omega={}_gamma_h={}_Pf={}'.format(omega, gamma_h, source_proj)
            plot_field(numpy_coeffs=f0_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_f0.pdf', 
                plot_type='components', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)
            plot_field(numpy_coeffs=f0_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_f0_vf.pdf', 
                plot_type='vector_field', cb_min=None, cb_max=None, hide_plot=hide_plots)
            divf0_c = div_m @ f0_c
            title = 'div f0'
            plot_field(numpy_coeffs=divf0_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_divf0.pdf', 
                plot_type='components', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)


    if tilde_f0_harmonic_c is not None:
        f0_harmonic_c = dH1_m.dot(tilde_f0_harmonic_c)            
        
        if debug:
            title = 'f0_harmonic part of source'
            params_str = 'omega={}_gamma_h={}_Pf={}'.format(omega, gamma_h, source_proj)
            plot_field(numpy_coeffs=f0_harmonic_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_f0_harmonic.pdf', 
                plot_type='components', cb_min=None, cb_max=None, hide_plot=hide_plots)
            plot_field(numpy_coeffs=f0_harmonic_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_f0_harmonic_vf.pdf', 
                plot_type='vector_field', cb_min=None, cb_max=None, hide_plot=hide_plots)
            divf0_c = div_m @ f0_harmonic_c
            title = 'div f0_harmonic'
            plot_field(numpy_coeffs=divf0_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_divf0_harmonic.pdf', 
                plot_type='components', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

    # else:
    #     raise NotImplementedError

    if f0_c is None:
        f0_c = np.zeros(V1h.nbasis)


    # if plot_source and plot_dir:
    #     plot_field(numpy_coeffs=f0_c, Vh=V1h, space_kind='hcurl', domain=domain, title='f0_h with P = '+source_proj, filename=plot_dir+'/f0h_'+source_proj+'.png', hide_plot=hide_plots)
        # plot_field(numpy_coeffs=f0_c, Vh=V1h, plot_type='vector_field', space_kind='hcurl', domain=domain, title='f0_h with P = '+source_proj, filename=plot_dir+'/f0h_'+source_proj+'_vf.png', hide_plot=hide_plots)
    
    t_stamp = time_count(t_stamp)
    
    def plot_J_source_nPlusHalf(f_c, nt):
            print(' .. plotting the source...')
            title = r'source $J^{n+1/2}_h$ (amplitude)'+' for $\omega = {}$, $n = {}$'.format(omega,nt)
            params_str = 'omega={}_gamma_h={}_Pf={}'.format(omega, gamma_h, source_proj)
            plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_Jh_nt={}.pdf'.format(nt), 
                plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)
            title = r'source $J^{n+1/2}_h$'+' for $\omega = {}$, $n = {}$'.format(omega,nt)
            plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, title=title, 
                filename=plot_dir+'/'+params_str+'_Jh_vf_nt={}.pdf'.format(nt), 
                plot_type='vector_field', vf_skip=1, hide_plot=hide_plots)

    def plot_E_field(E_c, nt, project_sol=False, plot_omega_normalized_sol=False, plot_divE=False):

        # only E for now
        if plot_dir:

            # project the homogeneous solution on the conforming problem space
            if project_sol:
                # t_stamp = time_count(t_stamp)
                print(' .. projecting the homogeneous solution on the conforming problem space...')
                Ep_c = cP1_m.dot(E_c)
            else:
                Ep_c = E_c
                print(' .. NOT projecting the homogeneous solution on the conforming problem space')
            if plot_omega_normalized_sol:
                print(' .. plotting the E/omega field...')
                u_c = (1/omega)*Ep_c
                title = r'$u_h = E_h/\omega$ (amplitude) for $\omega = {:5.4f}$, $t = {:5.4f}$'.format(omega, dt*nt)
                params_str = 'omega={:5.4f}_gamma_h={}_Pf={}_Nt_pp={}'.format(omega, gamma_h, source_proj, Nt_pp)
            else:
                print(' .. plotting the E field...')                
                if E0_type == 'pulse':
                    title = r'$t = {:5.4f}$'.format(dt*nt)
                else:
                    title = r'$E_h$ (amplitude) at $t = {:5.4f}$'.format(dt*nt)
                u_c = Ep_c
                params_str = 'gamma_h={}_Nt_pp={}'.format(gamma_h, Nt_pp)
            
            plot_field(numpy_coeffs=u_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_Eh_nt={}.pdf'.format(nt),
                plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

            if plot_divE:
                params_str = 'gamma_h={}_Nt_pp={}'.format(gamma_h, Nt_pp)
                if source_type == 'Il_pulse':
                    plot_type = 'components'
                    rho_c = rho0_c * np.sin(omega*dt*nt)/omega
                    rho_norm2 = np.dot(rho_c, H0_m.dot(rho_c))
                    title = r'$\rho_h$ at $t = {:5.4f}, norm = {}$'.format(dt*nt, np.sqrt(rho_norm2))
                    plot_field(numpy_coeffs=rho_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title, 
                    filename=plot_dir+'/'+params_str+'_rho_nt={}.pdf'.format(nt),
                    plot_type=plot_type, cb_min=None, cb_max=None, hide_plot=hide_plots)
                else:
                    plot_type = 'amplitude'

                divE_c = div_m @ Ep_c
                divE_norm2 = np.dot(divE_c, H0_m.dot(divE_c))
                if project_sol:
                    title = r'div $P^1_h E_h$ at $t = {:5.4f}, norm = {}$'.format(dt*nt, np.sqrt(divE_norm2))
                else:
                    title = r'div $E_h$ at $t = {:5.4f}, norm = {}$'.format(dt*nt, np.sqrt(divE_norm2))
                plot_field(numpy_coeffs=divE_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title, 
                    filename=plot_dir+'/'+params_str+'_divEh_nt={}.pdf'.format(nt),
                    plot_type=plot_type, cb_min=None, cb_max=None, hide_plot=hide_plots)

        else:
            print(' -- WARNING: unknown plot_dir !!')

    def plot_B_field(B_c, nt):

        if plot_dir:

            print(' .. plotting B field...')
            params_str = 'gamma_h={}_Nt_pp={}'.format(gamma_h, Nt_pp)

            title = r'$B_h$ (amplitude) for $t = {:5.4f}$'.format(dt*nt)
            plot_field(numpy_coeffs=B_c, Vh=V2h, space_kind='l2', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_Bh_nt={}.pdf'.format(nt),
                plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

        else:
            print(' -- WARNING: unknown plot_dir !!')

    def plot_time_diags(time_diag, E_norm2_diag, B_norm2_diag, divE_norm2_diag, nt_start, nt_end, GaussErr_norm2_diag=None, GaussErrP_norm2_diag=None, fharm_norm2_diag=None, skip_titles=True):
        nt_start = max(nt_start, 0)
        nt_end = min(nt_end, Nt)
        tau_start = nt_start/Nt_pp
        tau_end = nt_end/Nt_pp

        td = time_diag[nt_start:nt_end+1]
        if source_is_harmonic:
            td[:] = td[:]/period_time
            t_label = 't/tau'
        else: 
            t_label = 't'

        # norm || E ||
        fig, ax = plt.subplots()
        ax.plot(td, np.sqrt(E_norm2_diag[nt_start:nt_end+1]), '-', ms=7, mfc='None', mec='k') #, label='||E||', zorder=10)
        if skip_titles:
            title = ''
        else:
            title = r'$||E_h(t)||$'
        ax.set_xlabel(t_label, fontsize=16)
        ax.set_title(title, fontsize=18)
        fig.tight_layout()
        diag_fn = plot_dir+'/diag_E_norm_gamma={}_Nt_pp={}_tau_range=[{},{}].pdf'.format(gamma_h, Nt_pp, tau_start, tau_end)
        print("saving plot for '"+title+"' in figure '"+diag_fn)
        fig.savefig(diag_fn)

        # energy
        fig, ax = plt.subplots()
        ax.plot(td, .5*(E_norm2_diag[nt_start:nt_end+1]+B_norm2_diag[nt_start:nt_end+1]), '-', ms=7, mfc='None', mec='k') #, label='||E||', zorder=10)
        if skip_titles:
            title = ''
        else:
            title = r'$\frac{1}{2} (||E_h(t)||^2+||B_h(t)||^2)$ vs $t/\tau$' 
        if E0_type == 'pulse':
            ax.set_ylim([0, 5])
        ax.set_xlabel(t_label, fontsize=16)                    
        ax.set_title(title, fontsize=18)
        fig.tight_layout()
        diag_fn = plot_dir+'/diag_energy_gamma={}_Nt_pp={}_tau_range=[{},{}].pdf'.format(gamma_h, Nt_pp, tau_start, tau_end)
        print("saving plot for '"+title+"' in figure '"+diag_fn)
        fig.savefig(diag_fn)

        # norm || div E ||
        fig, ax = plt.subplots()
        # print(' -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ') 
        # print('diag_divE:') 
        # print(' -- ') 
        # print(td)
        # print(' -- ')
        # print(divE_norm2_diag[nt_start:nt_end+1])
        # print(np.sqrt(divE_norm2_diag[nt_start:nt_end+1]))
        # print(' -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ') 
        
        ax.plot(td, np.sqrt(divE_norm2_diag[nt_start:nt_end+1]), '-', ms=7, mfc='None', mec='k') #, label='||E||', zorder=10)
        if project_sol:
            diag_fn = plot_dir+'/diag_divPE_gamma={}_Nt_pp={}_tau_range=[{},{}].pdf'.format(gamma_h, Nt_pp, tau_start, tau_end)
            title = r'$||div_h P^1_h E_h(t)||$ vs $t/\tau$' 
        else:
            diag_fn = plot_dir+'/diag_divE_gamma={}_Nt_pp={}_tau_range=[{},{}].pdf'.format(gamma_h, Nt_pp, tau_start, tau_end)
            title = r'$||div_h E_h(t)||$ vs $t/\tau$' 
        if skip_titles:
            title = ''
        ax.set_xlabel(t_label, fontsize=16)  
        ax.set_title(title, fontsize=18)
        fig.tight_layout()
        print("saving plot for '"+title+"' in figure '"+diag_fn)
        fig.savefig(diag_fn)
    
        if fharm_norm2_diag is not None:
            fig, ax = plt.subplots()            
            ax.plot(td, np.sqrt(fharm_norm2_diag[nt_start:nt_end+1]), '-', ms=7, mfc='None', mec='k') #, label='||E||', zorder=10)
            diag_fn = plot_dir+'/diag_fharm_gamma={}_Nt_pp={}_tau_range=[{},{}].pdf'.format(gamma_h, Nt_pp, tau_start, tau_end)
            title = r'$||f_{harm}(t)||$ vs $t/\tau$' 
            if skip_titles:
                title = ''
            ax.set_xlabel(t_label, fontsize=16)  
            ax.set_title(title, fontsize=18)
            fig.tight_layout()
            print("saving plot for '"+title+"' in figure '"+diag_fn)
            fig.savefig(diag_fn)            

        if GaussErr_norm2_diag is not None:
            fig, ax = plt.subplots()            
            ax.plot(td, np.sqrt(GaussErr_norm2_diag[nt_start:nt_end+1]), '-', ms=7, mfc='None', mec='k') #, label='||E||', zorder=10)
            diag_fn = plot_dir+'/diag_GaussErr_gamma={}_Nt_pp={}_tau_range=[{},{}].pdf'.format(gamma_h, Nt_pp, tau_start, tau_end)
            title = r'$||(\rho_h - div_h E_h)(t)||$ vs $t/\tau$' 
            if skip_titles:
                title = ''
            ax.set_xlabel(t_label, fontsize=16)  
            ax.set_title(title, fontsize=18)
            fig.tight_layout()
            print("saving plot for '"+title+"' in figure '"+diag_fn)
            fig.savefig(diag_fn)     

        if GaussErrP_norm2_diag is not None:
            fig, ax = plt.subplots()            
            ax.plot(td, np.sqrt(GaussErrP_norm2_diag[nt_start:nt_end+1]), '-', ms=7, mfc='None', mec='k') #, label='||E||', zorder=10)
            diag_fn = plot_dir+'/diag_GaussErrP_gamma={}_Nt_pp={}_tau_range=[{},{}].pdf'.format(gamma_h, Nt_pp, tau_start, tau_end)
            title = r'$||(\rho_h - div_h P_h E_h)(t)||$ vs $t/\tau$' 
            if skip_titles:
                title = ''
            ax.set_xlabel(t_label, fontsize=16)  
            ax.set_title(title, fontsize=18)
            fig.tight_layout()
            print("saving plot for '"+title+"' in figure '"+diag_fn)
            fig.savefig(diag_fn)     

    E_norm2_diag = np.zeros(Nt+1)
    B_norm2_diag = np.zeros(Nt+1)
    divE_norm2_diag = np.zeros(Nt+1)
    time_diag = np.zeros(Nt+1)

    if source_type == 'Il_pulse':
        GaussErr_norm2_diag = np.zeros(Nt+1)
        GaussErrP_norm2_diag = np.zeros(Nt+1)
        fharm_norm2_diag = np.zeros(Nt+1)
    else:
        GaussErr_norm2_diag = None
        GaussErrP_norm2_diag = None
        fharm_norm2_diag = None

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # initial solution

    print(' .. init..')

    # initial B sol
    B_c = np.zeros(V2h.nbasis)
    
    # initial E sol
    if E0_type == 'th_sol':

        if os.path.exists(th_sol_filename):
            print(' .. loading time-harmonic solution from file {}'.format(th_sol_filename))
            E_c = omega * np.load(th_sol_filename)
            assert len(E_c) == V1h.nbasis
        else:
            print(' .. Error: time-harmonic solution file given {}, but not found'.format(th_sol_filename))
            raise ValueError(th_sol_filename)
            
    elif E0_type == 'zero':
        E_c = np.zeros(V1h.nbasis)

    elif E0_type == 'pulse':       

        E0 = get_div_free_pulse(x_0=1.0, y_0=1.0, domain=domain)
        
        if E0_proj == 'P_geom':
            print(' .. projecting E0 with commuting projection...')
            E0_h = P1_phys(E0, P1, domain, mappings_list)
            E_c = E0_h.coeffs.toarray()
        
        elif E0_proj == 'P_L2':
            # helper: save/load coefs
            E0dd_filename = m_load_dir+'/E0_pulse_dual_dofs.npy'
            if os.path.exists(E0dd_filename):
                print(' .. loading E0 dual dofs from file {}'.format(E0dd_filename))
                tilde_E0_c = np.load(E0dd_filename)
            else:
                print(' .. projecting E0 with L2 projection...')
                tilde_E0_c = derham_h.get_dual_dofs(space='V1', f=E0, backend_language=backend_language, return_format='numpy_array')
                print(' .. saving E0 dual dofs to file {}'.format(E0dd_filename))
                np.save(E0dd_filename, tilde_E0_c)
            E_c = dH1_m.dot(tilde_E0_c)

    else:
        raise ValueError(E0_type)        


    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # time loop

    E_norm2_diag[0] = np.dot(E_c,H1_m.dot(E_c))
    B_norm2_diag[0] = np.dot(B_c,H2_m.dot(B_c))
    # norm_amps_diag[0] = amps_diag[0]/(np.cos(omega*dt*0)+1e-10)
    if project_sol:
        Ep_c = cP1_m.dot(E_c)
    else:
        Ep_c = E_c
    divE_c = div_m @ Ep_c
    divE_norm2_diag[0] = np.dot(divE_c, H0_m.dot(divE_c))
    
    if source_type == 'Il_pulse':
        rho_c = rho0_c * np.sin(omega*dt*0)/omega
        GaussErr = rho_c - div_m @ E_c
        GaussErrP = rho0_c - div_m @ (cP1_m.dot(E_c))
        GaussErr_norm2_diag[0] = np.dot(GaussErr, H0_m.dot(GaussErr))
        GaussErrP_norm2_diag[0] = np.dot(GaussErrP, H0_m.dot(GaussErrP))


    plot_E_field(E_c, nt=0, project_sol=project_sol, plot_divE=plot_divE)
    plot_B_field(B_c, nt=0)
    
    f_c = np.copy(f0_c)
    for nt in range(Nt):
        print(' .. nt+1 = {}/{}'.format(nt+1, Nt))

        # 1/2 faraday: Bn -> Bn+1/2
        B_c[:] -= (dt/2) * C_m @ E_c

        # ampere: En -> En+1
        if f0_harmonic_c is not None:
            f_harmonic_c = f0_harmonic_c * (np.sin(omega*(nt+1)*dt)-np.sin(omega*(nt)*dt))/(dt*omega) # * source_enveloppe(omega*(nt+1/2)*dt)
            fharm_norm2_diag[nt+1] = np.dot(f_harmonic_c,H1_m.dot(f_harmonic_c))
            f_c[:] = f0_c + f_harmonic_c

        if nt == 0:
            plot_J_source_nPlusHalf(f_c, nt=0)
            fharm_norm2_diag[0] = fharm_norm2_diag[1]

        E_c[:] += dt * (dC_m @ B_c - f_c)
        if abs(gamma_h) > 1e-10:
            E_c[:] -= dt * gamma_h * JP_m @ E_c

        # 1/2 faraday: Bn+1/2 -> Bn+1
        B_c[:] -= (dt/2) * C_m @ E_c

        # diags: E norm
        E_norm2_diag[nt+1] = np.dot(E_c,H1_m.dot(E_c))
        B_norm2_diag[nt+1] = np.dot(B_c,H2_m.dot(B_c))
        # nad = amps_diag[nt+1]/(np.cos(omega*dt*(nt+1))+1e-10)
        # if abs(nad) > 100:
        #     nad = 0
        # norm_amps_diag[nt+1] = nad
        time_diag[nt+1] = (nt+1)*dt
        
        if debug:
            divCB_c = div_m @ dC_m @ B_c
            divCB_norm2 = np.dot(divCB_c, H0_m.dot(divCB_c))
            print('-- [{}]: dt*|| div CB || = {}'.format(nt+1, dt*np.sqrt(divCB_norm2)))

            divf_c = div_m @ f_c
            divf_norm2 = np.dot(divf_c, H0_m.dot(divf_c))
            print('-- [{}]: dt*|| div f || = {}'.format(nt+1, dt*np.sqrt(divf_norm2)))

            divE_c = div_m @ E_c
            divE_norm2 = np.dot(divE_c, H0_m.dot(divE_c))
            print('-- [{}]: || div E || = {}'.format(nt+1, np.sqrt(divE_norm2)))

        # diags: div        
        if project_sol:
            Ep_c = cP1_m.dot(E_c)
        else:
            Ep_c = E_c
        divE_c = div_m @ Ep_c
        divE_norm2 = np.dot(divE_c, H0_m.dot(divE_c))
        # print('in diag[{}]: divE_norm = {}'.format(nt+1, np.sqrt(divE_norm2)))
        divE_norm2_diag[nt+1] = divE_norm2

        if source_type == 'Il_pulse':
            rho_c = rho0_c * np.sin(omega*dt*(nt+1))/omega
            GaussErr = rho_c - div_m @ E_c
            GaussErrP = rho0_c - div_m @ (cP1_m.dot(E_c))
            GaussErr_norm2_diag[nt+1] = np.dot(GaussErr, H0_m.dot(GaussErr))
            GaussErrP_norm2_diag[nt+1] = np.dot(GaussErrP, H0_m.dot(GaussErrP))

        if is_plotting_time(nt):
            plot_E_field(E_c, nt=nt+1, project_sol=project_sol, plot_divE=plot_divE)
            plot_B_field(B_c, nt=nt+1)
            plot_J_source_nPlusHalf(f_c, nt=nt)

        if (nt+1)%(diag_dtau*Nt_pp) == 0:
            tau_here = nt+1
            
            plot_time_diags(time_diag, E_norm2_diag, B_norm2_diag, divE_norm2_diag, nt_start=(nt+1)-diag_dtau*Nt_pp, nt_end=(nt+1), 
            fharm_norm2_diag=fharm_norm2_diag, GaussErr_norm2_diag=GaussErr_norm2_diag, GaussErrP_norm2_diag=GaussErrP_norm2_diag)   

    plot_time_diags(time_diag, E_norm2_diag, B_norm2_diag, divE_norm2_diag, nt_start=0, nt_end=Nt, 
    fharm_norm2_diag=fharm_norm2_diag, GaussErr_norm2_diag=GaussErr_norm2_diag, GaussErrP_norm2_diag=GaussErrP_norm2_diag)

    # Eh = FemField(V1h, coeffs=array_to_stencil(E_c, V1h.vector_space))
    # t_stamp = time_count(t_stamp)

    # if sol_filename:
    #     raise NotImplementedError
        # print(' .. saving final solution coeffs to file {}'.format(sol_filename))
        # np.save(sol_filename, E_c)
    
    # time_count(t_stamp)

    # print()
    # print(' -- plots and diagnostics  --')
    
    # # diagnostics: errors
    # err_diags = diag_grid.get_diags_for(v=uh, space='V1')
    # for key, value in err_diags.items():
    #     diags[key] = value

    # if u_ex is not None:
    #     check_diags = get_Vh_diags_for(v=uh, v_ref=uh_ref, M_m=H1_m, msg='error between Ph(u_ex) and u_h')
    #     diags['norm_Pu_ex'] = check_diags['sol_ref_norm']
    #     diags['rel_l2_error_in_Vh'] = check_diags['rel_l2_error']

    # if curl_u_ex is not None:
    #     print(' .. diag on curl_u:')
    #     curl_uh_c = bD1_m @ cP1_m @ uh_c
    #     title = r'curl $u_h$ (amplitude) for $\eta = $'+repr(eta)
    #     params_str = 'eta={}_mu={}_nu={}_gamma_h={}_Pf={}'.format(eta, mu, nu, gamma_h, source_proj)
    #     plot_field(numpy_coeffs=curl_uh_c, Vh=V2h, space_kind='l2', domain=domain, surface_plot=False, title=title, filename=plot_dir+'/'+params_str+'_curl_uh.png', 
    #         plot_type='amplitude', cb_min=None, cb_max=None, hide_plot=hide_plots)

    #     curl_uh = FemField(V2h, coeffs=array_to_stencil(curl_uh_c, V2h.vector_space))
    #     curl_diags = diag_grid.get_diags_for(v=curl_uh, space='V2')
    #     diags['curl_error (to be checked)'] = curl_diags['rel_l2_error']

        
    #     title = r'div_h $u_h$ (amplitude) for $\eta = $'+repr(eta)
    #     params_str = 'eta={}_mu={}_nu={}_gamma_h={}_Pf={}'.format(eta, mu, nu, gamma_h, source_proj)
    #     plot_field(numpy_coeffs=div_uh_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title, filename=plot_dir+'/'+params_str+'_div_uh.png', 
    #         plot_type='amplitude', cb_min=None, cb_max=None, hide_plot=hide_plots)

    #     div_uh = FemField(V0h, coeffs=array_to_stencil(div_uh_c, V0h.vector_space))
    #     div_diags = diag_grid.get_diags_for(v=div_uh, space='V0')
    #     diags['div_error (to be checked)'] = div_diags['rel_l2_error']

    return diags


def compute_stable_dt(cfl, period_time, C_m, dC_m, V1_dim):

    print (" .. compute_stable_dt by estimating the operator norm of ")
    print (" ..     dC_m @ C_m: V1h -> V1h ")
    print (" ..     with dim(V1h) = {}      ...".format(V1_dim))

    def vect_norm_2 (vv):
        return np.sqrt(np.dot(vv,vv))
    t_stamp = time_count()
    vv = np.random.random(V1_dim)
    norm_vv = vect_norm_2(vv)    
    max_ncfl = 500
    ncfl = 0
    spectral_rho = 1
    conv = False
    CC_m = dC_m @ C_m
    while not( conv or ncfl > max_ncfl ):

        vv[:] = (1./norm_vv)*vv
        ncfl += 1
        vv[:] = CC_m.dot(vv)
        
        norm_vv = vect_norm_2(vv)
        old_spectral_rho = spectral_rho
        spectral_rho = vect_norm_2(vv) # approximation
        conv = abs((spectral_rho - old_spectral_rho)/spectral_rho) < 0.001
        print ("    ... spectral radius iteration: spectral_rho( dC_m @ C_m ) ~= {}".format(spectral_rho))
    t_stamp = time_count(t_stamp)
    
    norm_op = np.sqrt(spectral_rho)
    c_dt_max = 2./norm_op    
    
    light_c = 1
    Nt_pp = int(np.ceil(period_time/(cfl*c_dt_max/light_c)))
    assert Nt_pp >= 1 
    dt = period_time / Nt_pp
    
    assert light_c*dt <= cfl * c_dt_max
    
    print("  Time step dt computed for Maxwell solver:")
    print("     Since cfl = " + repr(cfl)+",   we set dt = "+repr(dt)+"  --  and Nt_pp = "+repr(Nt_pp))
    print("     -- note that c*Dt = "+repr(light_c * dt)+", and c_dt_max = "+repr(c_dt_max)+" thus c * dt / c_dt_max = "+repr(light_c*dt/c_dt_max))
    print("     -- and spectral_radius((c*dt)**2* dC_m @ C_m ) = ",  (light_c * dt * norm_op)**2, " (should be < 4).")

    return Nt_pp, dt

if __name__ == '__main__':
    # quick run, to test 

    raise NotImplementedError


    t_stamp_full = time_count()

    omega = np.sqrt(170) # source
    roundoff = 1e4
    eta = int(-omega**2 * roundoff)/roundoff

    source_type = 'manu_maxwell'
    # source_type = 'manu_J'

    domain_name = 'curved_L_shape'
    nc = 4
    deg = 2

    run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
    m_load_dir = 'matrices_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
    solve_hcurl_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1, #1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_source=True,
        plot_dir='./plots/tests_source_feb_13/'+run_dir,
        hide_plots=True,
        m_load_dir=m_load_dir
    )

    time_count(t_stamp_full, msg='full program')
