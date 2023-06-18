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
from psydac.feec.multipatch.plotting_utilities_2          import plot_field #, write_field_to_diag_grid, 
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases     import get_div_free_pulse
from psydac.feec.multipatch.utils_conga_2d              import DiagGrid, P_phys_hdiv, get_Vh_diags_for
from psydac.feec.multipatch.utilities                   import time_count #, export_sol, import_sol
from psydac.feec.multipatch.bilinear_form_scipy         import construct_pairing_matrix
from psydac.feec.multipatch.conf_projections_scipy      import Conf_proj_0, Conf_proj_1

def solve_td_maxwell_pbm(
        nc=4, deg=4, Nt_pp=None, cfl=.8, nb_t_periods=20, omega=20, source_is_harmonic=True,
        domain_name='pretzel_f', backend_language=None, source_proj='P_geom', source_type='manu_J',
        project_sol=False, filter_source=True, quad_param=1,
        D0_type='zero', D0_proj='P_L2', 
        plot_source=False, plot_dir=None, plot_divD=False, hide_plots=True, plot_time_ranges=None, diag_dtau=None,
        skip_plot_titles=False,
        cb_min_sol=None, cb_max_sol=None,
        m_load_dir="",
):
    """
    solver for the TD Maxwell problem: find E(t) in H(curl), B in L2, such that

      dt D - curl H = -J             on \Omega
      dt B + curl E = 0              on \Omega
      n x E = 0                      on \partial \Omega

    with SSC scheme on a 2D multipatch domain \Omega
    involving two strong sequences, a  and a dual 

      primal: p_V0h  --grad->  p_V1h  -—curl-> p_V2h        (with homogeneous bc's)
                                (Eh)            (Bh)

                                (Dh)            (Hh)
      dual:   d_V2h  <--div--  d_V1h  <—curl-- d_V0h        (no bc's)

    the semi-discrete level the equations read

        Ampere: 
            Hh = p_HH2 @ Bh
            dt Dh - d_CC @ Hh = - Jh         
        with
            p_HH2 = hodge:   p_V2h -> d_V0h
            d_CC  = curl:    d_V0h -> d_V1h

        Faraday:    
            Eh = d_HH1 @ Dh
            dt Bh + p_curl @ Eh = 0              
        with
            d_HH1 = hodge:   d_V1h -> p_V1h
            p_CC  = curl:    p_V1h -> p_V2h

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param source_proj: approximation operator for the source (see later)
    :param source_type: must be implemented in get_source_and_solution()
    :param m_load_dir: directory for matrix storage
    """
    diags = {}

    ncells = [nc, nc]
    degree = [deg,deg]

    period_time = 2*np.pi/omega
    final_time = nb_t_periods * period_time

    if plot_time_ranges is None:
        plot_time_ranges = [[0, final_time], 2]

    if diag_dtau is None:
        diag_dtau = nb_t_periods//10

    if m_load_dir is not None:
        pm_load_dir = m_load_dir+"primal"
        dm_load_dir = m_load_dir+"dual"
        for load_dir in [pm_load_dir, dm_load_dir]:        
            if not os.path.exists(load_dir):
                os.makedirs(load_dir)

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_td_maxwell_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' D0_type = {}'.format(D0_type))
    print(' D0_proj = {}'.format(D0_proj))
    print(' source_type = {}'.format(source_type))
    print(' source_proj = {}'.format(source_proj))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    debug = False

    print()
    print(' -- building discrete spaces and operators  --')

    t_stamp = time_count()
    print(' .. multi-patch domain...')
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = [m.get_callable_mapping() for m in mappings.values()]

    # for diagnostics
    diag_grid = DiagGrid(mappings=mappings, N_diag=100)

    t_stamp = time_count(t_stamp)
    print('building symbolic derham sequences...')
    p_derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    d_derham  = Derham(domain, ["H1", "Hdiv", "L2"])

    t_stamp = time_count(t_stamp)
    print('building discrete derham sequences...')
    dual_degree = [d-1 for d in degree]
    domain_h = discretize(domain, ncells=ncells)
    p_derham_h = discretize(p_derham, domain_h, degree=degree)
    d_derham_h = discretize(d_derham, domain_h, degree=dual_degree)
    
    p_V0h = p_derham_h.V0
    p_V1h = p_derham_h.V1
    p_V2h = p_derham_h.V2

    d_V0h = d_derham_h.V0
    d_V1h = d_derham_h.V1
    d_V2h = d_derham_h.V2

    t_stamp = time_count(t_stamp)
    print('building the primal Hodge operator p_H2: p_V2h -> d_V0h ...')
    
    ## NOTE: with a strong-strong diagram we should not call these "Hodge" operators !! 
    
    d_HOp0   = HodgeOperator(d_V0h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=0)
    d_MM0     = d_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
    d_MM0_inv = d_HOp0.to_sparse_matrix()                # inverse mass matrix
    
    print('aa ')

    # d_KK1     = construct_pairing_matrix(p_V1h,d_V1h).tocsr()  # matrix in scipy format  # REMOVE

    p_KK2     = construct_pairing_matrix(d_V0h,p_V2h).tocsr()  # matrix in scipy format
    d_PP0 = Conf_proj_0(d_V0h, nquads = [4*(d + 1) for d in dual_degree])
    # d_PP0_pm = d_derham_h.conforming_projection(space='V0', hom_bc=False, backend_language=backend_language, load_dir=dm_load_dir)
    # d_PP0    = d_PP0_pm.to_sparse_matrix()

    # print('bb ')
    d_PP1 = Conf_proj_1(d_V1h, nquads = [4*(d + 1) for d in dual_degree])
    # d_PP1_pm = d_derham_h.conforming_projection(space='V1', hom_bc=False, backend_language=backend_language, load_dir=dm_load_dir)
    # d_PP1    = d_PP1_pm.to_sparse_matrix()
    
    print('cc ')
    p_HH2     = d_MM0_inv @ d_PP0.transpose() @ p_KK2

    t_stamp = time_count(t_stamp)
    print('building the dual Hodge operator d_H1: d_V1h -> p_V1h ...')
        
    p_HOp1   = HodgeOperator(p_V1h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=1)
    p_MM1     = p_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
    p_MM1_inv = p_HOp1.to_sparse_matrix()                # inverse mass matrix

    d_KK1     = construct_pairing_matrix(p_V1h,d_V1h).tocsr()  # matrix in scipy format

    # TODO: write "C1" conforming proj with hom bc's
    p_PP0 = Conf_proj_0(p_V0h, nquads = [4*(d + 1) for d in degree])
    # p_PP0_pm = p_derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language, load_dir=pm_load_dir)
    # p_PP0    = p_PP0_pm.to_sparse_matrix()

    # TODO: write "C1-curl" conforming proj, with hom bc's
    p_PP1 = Conf_proj_1(p_V1h, nquads = [4*(d + 1) for d in degree])
    # p_PP1_pm = p_derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language, load_dir=pm_load_dir)
    # p_PP1    = p_PP1_pm.to_sparse_matrix()

    d_HH1 = p_MM1_inv @ p_PP1.transpose() @ d_KK1

    t_stamp = time_count(t_stamp)
    print(' .. broken and Conga differential operators...')
    p_bD0, p_bD1 = p_derham_h.broken_derivatives_as_operators
    d_bD0, d_bD1 = d_derham_h.broken_derivatives_as_operators
    
    p_bG = p_bD0.to_sparse_matrix() # broken grad (primal)
    p_GG = p_bG @ p_PP0             # Conga grad (primal)
    p_bC = p_bD1.to_sparse_matrix() # broken curl (primal: scalar-valued)
    p_CC = p_bC @ p_PP1             # Conga curl (primal)
    
    d_bC = d_bD0.to_sparse_matrix() # broken curl (dual: vector-valued)
    d_CC = d_bC @ d_PP0             # Conga curl (dual)    
    d_bD = d_bD1.to_sparse_matrix() # broken div
    d_DD = d_bD @ d_PP1             # Conga div (dual)    

    t_stamp = time_count(t_stamp)
    print(' .. Ampere and Faraday evolution (curl . Hodge) operators...')
    Amp_Op = d_CC @ p_HH2
    Far_Op = p_CC @ d_HH1

    t_stamp = time_count(t_stamp)
    print(' .. other operators for diagnostics...')
    p_HOp2   = HodgeOperator(p_V2h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=2)
    p_MM2    = p_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix

    d_HOp2   = HodgeOperator(d_V2h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=2)
    d_MM2    = d_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix
    # print(m_load_dir)
    # print(d_DD.shape)
    # print(d_MM2.shape)
    # exit()

    # print('dim(V0h) = {}'.format(V0h.nbasis))
    # print('dim(V1h) = {}'.format(V1h.nbasis))
    # print('dim(V2h) = {}'.format(V2h.nbasis))
    # diags['ndofs_V0'] = V0h.nbasis
    # diags['ndofs_V1'] = V1h.nbasis
    # diags['ndofs_V2'] = V2h.nbasis

    # t_stamp = time_count(t_stamp)
    # print(' .. Id operator and matrix...')
    # I1 = IdLinearOperator(V1h)
    # I1_m = I1.to_sparse_matrix()

    # t_stamp = time_count(t_stamp)
    # print(' .. Hodge operators...')
    # # multi-patch (broken) linear operators / matrices
    # # other option: define as Hodge Operators:
    # H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=0)
    # H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=1)
    # H2 = HodgeOperator(V2h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=2)

    # t_stamp = time_count(t_stamp)
    # print(' .. Hodge matrix H0_m = M0_m ...')
    # H0_m  = H0.to_sparse_matrix()              
    # t_stamp = time_count(t_stamp)
    # print(' .. dual Hodge matrix dH0_m = inv_M0_m ...')
    # dH0_m = H0.get_dual_sparse_matrix()  

    # t_stamp = time_count(t_stamp)
    # print(' .. Hodge matrix H1_m = M1_m ...')
    # H1_m  = H1.to_sparse_matrix()              
    # t_stamp = time_count(t_stamp)
    # print(' .. dual Hodge matrix dH1_m = inv_M1_m ...')
    # dH1_m = H1.get_dual_sparse_matrix()  

    # t_stamp = time_count(t_stamp)
    # print(' .. Hodge matrix dH2_m = M2_m ...')
    # H2_m = H2.to_sparse_matrix()              

    # t_stamp = time_count(t_stamp)
    # print(' .. conforming Projection operators...')
    # # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    # cP0 = derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
    # cP1 = derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
    # cP0_m = cP0.to_sparse_matrix()
    # cP1_m = cP1.to_sparse_matrix()

    p_geomP0, p_geomP1, p_geomP2 = p_derham_h.projectors()
    d_geomP0, d_geomP1, d_geomP2 = d_derham_h.projectors()

    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Conga (projection-based) matrices
    # t_stamp = time_count(t_stamp)    
    # print(' .. matrix of the primal curl (in primal bases)...')
    # C_m = bD1_m @ cP1_m
    # print(' .. matrix of the dual curl (also in primal bases)...')
    # dC_m = dH1_m @ C_m.transpose() @ H2_m
    # print(' .. matrix of the dual div (still in primal bases)...')
    # div_m = dH0_m @ cP0_m.transpose() @ bD0_m.transpose() @ H1_m

    # jump stabilization (may not be needed)
    # t_stamp = time_count(t_stamp)
    # print(' .. jump stabilization matrix...')
    # jump_penal_m = I1_m - cP1_m
    # JP_m = jump_penal_m.transpose() * H1_m * jump_penal_m

    # t_stamp = time_count(t_stamp)
    # print(' .. full operator matrix...')
    # print('STABILIZATION: gamma_h = {}'.format(gamma_h))
    # pre_A_m = cP1_m.transpose() @ ( eta * H1_m + mu * pre_CC_m - nu * pre_GD_m )  # useful for the boundary condition (if present)
    # A_m = pre_A_m @ cP1_m + gamma_h * JP_m

    if Nt_pp is None:
        if not( 0 < cfl <= 1):
            print(' ******  ****** ******  ****** ******  ****** ')
            print('         WARNING !!!  cfl = {}  '.format(cfl))
            print(' ******  ****** ******  ****** ******  ****** ')
        print(Amp_Op.shape)
        print(Far_Op.shape)
        print(p_V2h.nbasis)
        Nt_pp, dt, norm_curlh = compute_stable_dt(cfl, period_time, Amp_Op, Far_Op, p_V2h.nbasis)
    else:
        dt = period_time/Nt_pp
        norm_curlh = None
    Nt = Nt_pp * nb_t_periods

    def is_plotting_time(nt):
        answer = (nt==0) or (nt==Nt)
        for tau_range, nt_plots_pp in plot_time_ranges:
            if answer:
                break
            tp = max(Nt_pp//nt_plots_pp,1)
            answer = (tau_range[0]*period_time <= nt*dt <= tau_range[1]*period_time and (nt)%tp == 0)
        return answer

    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' total nb of time steps: Nt = {}, final time: T = {:5.4f}'.format(Nt, final_time))
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' plotting times: the solution will be plotted for...')
    for nt in range(Nt+1):
        if is_plotting_time(nt):
            print(' * nt = {}, t = {:5.4f}'.format(nt, dt*nt))
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # source

    t_stamp = time_count(t_stamp)
    print()
    print(' -- getting source --')
    if source_type == 'zero':
        f0_c = np.zeros(d_V1h.nbasis)

    else:
        raise ValueError(source_type)
        
    t_stamp = time_count(t_stamp)
    
    def plot_D_field(D_c, nt, project_sol=False, plot_divD=False):

        """
        plot E in p_V1h
        """
        if plot_dir:

            plot_omega_normalized_sol = source_is_harmonic
            # project the homogeneous solution on the conforming problem space
            if project_sol:
                raise NotImplementedError
                # t_stamp = time_count(t_stamp)
                print(' .. projecting the homogeneous solution on the conforming problem space...')
                Ep_c = p_PP1_m.dot(E_c)
            else:
                Dp_c = D_c
            print(' .. plotting the D field...')                
            title = r'$D_h$ (amplitude) at $t = {:5.4f}$'.format(dt*nt)
            
            params_str = 'Nt_pp={}'.format(Nt_pp)
            plot_field(numpy_coeffs=Dp_c, Vh=d_V1h, space_kind='hdiv', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_Dh_nt={}.pdf'.format(nt),
                plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

            if plot_divD:
                params_str = 'Nt_pp={}'.format(Nt_pp)
                plot_type = 'amplitude'

                divD_c = d_DD @ Dp_c
                divD_norm2 = np.dot(divD_c, d_MM2.dot(divD_c))
                title = r'div $D_h$ at $t = {:5.4f}, norm = {}$'.format(dt*nt, np.sqrt(divD_norm2))
                plot_field(numpy_coeffs=divD_c, Vh=d_V2h, space_kind='l2', domain=domain, surface_plot=False, title=title, 
                    filename=plot_dir+'/'+params_str+'_divDh_nt={}.pdf'.format(nt),
                    plot_type=plot_type, cb_min=None, cb_max=None, hide_plot=hide_plots)
                
        else:
            print(' -- WARNING: unknown plot_dir !!')

    def plot_B_field(B_c, nt):

        if plot_dir:

            print(' .. plotting H field...')
            params_str = 'Nt_pp={}'.format(Nt_pp)

            title = r'$B_h$ (amplitude) for $t = {:5.4f}$'.format(dt*nt)
            plot_field(numpy_coeffs=B_c, Vh=p_V2h, space_kind='l2', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_Bh_nt={}.pdf'.format(nt),
                plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

        else:
            print(' -- WARNING: unknown plot_dir !!')

    def plot_time_diags(time_diag, E_norm2_diag, H_norm2_diag, divD_norm2_diag, nt_start, nt_end, 
        GaussErr_norm2_diag=None, GaussErrP_norm2_diag=None, 
        PE_norm2_diag=None, I_PE_norm2_diag=None, J_norm2_diag=None, skip_titles=True):
        nt_start = max(nt_start, 0)
        nt_end = min(nt_end, Nt)
        tau_start = nt_start/Nt_pp
        tau_end = nt_end/Nt_pp

        if source_is_harmonic:
            td = time_diag[nt_start:nt_end+1]/period_time
            t_label = r'$t/\tau$'
        else: 
            td = time_diag[nt_start:nt_end+1]
            t_label = r'$t$'

        # norm || E ||
        fig, ax = plt.subplots()
        ax.plot(td, np.sqrt(E_norm2_diag[nt_start:nt_end+1]), '-', ms=7, mfc='None', mec='k') #, label='||E||', zorder=10)
        if skip_titles:
            title = ''
        else:
            title = r'$||E_h(t)||$ vs '+t_label
        ax.set_xlabel(t_label, fontsize=16)
        ax.set_title(title, fontsize=18)
        fig.tight_layout()
        diag_fn = plot_dir+'/diag_E_norm_Nt_pp={}_tau_range=[{},{}].pdf'.format(Nt_pp, tau_start, tau_end)
        print("saving plot for '"+title+"' in figure '"+diag_fn)
        fig.savefig(diag_fn)

        # energy
        fig, ax = plt.subplots()
        E_energ = .5*E_norm2_diag[nt_start:nt_end+1]
        B_energ = .5*H_norm2_diag[nt_start:nt_end+1]
        ax.plot(td, E_energ, '-', ms=7, mfc='None', c='k', label=r'$\frac{1}{2}||E||^2$') #, zorder=10)
        ax.plot(td, B_energ, '-', ms=7, mfc='None', c='g', label=r'$\frac{1}{2}||B||^2$') #, zorder=10)
        ax.plot(td, E_energ+B_energ, '-', ms=7, mfc='None', c='b', label=r'$\frac{1}{2}(||E||^2+||B||^2)$') #, zorder=10)
        ax.legend(loc='best')
        if skip_titles:  
            title = ''
        else:
            title = r'energy vs '+t_label
        if D0_type == 'pulse':
            ax.set_ylim([0, 5])
        ax.set_xlabel(t_label, fontsize=16)                    
        ax.set_title(title, fontsize=18)
        fig.tight_layout()
        diag_fn = plot_dir+'/diag_energy_Nt_pp={}_tau_range=[{},{}].pdf'.format(Nt_pp, tau_start, tau_end)
        print("saving plot for '"+title+"' in figure '"+diag_fn)
        fig.savefig(diag_fn)

        # norm || div E ||
        fig, ax = plt.subplots()
        ax.plot(td, np.sqrt(divD_norm2_diag[nt_start:nt_end+1]), '-', ms=7, mfc='None', mec='k') #, label='||E||', zorder=10)
        diag_fn = plot_dir+'/diag_divD_Nt_pp={}_tau_range=[{},{}].pdf'.format(Nt_pp, tau_start, tau_end)
        title = r'$||div_h E_h(t)||$ vs '+t_label 
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
            diag_fn = plot_dir+'/diag_GaussErr_Nt_pp={}_tau_range=[{},{}].pdf'.format(Nt_pp, tau_start, tau_end)
            title = r'$||(\rho_h - div_h E_h)(t)||$ vs '+t_label
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
            diag_fn = plot_dir+'/diag_GaussErrP_Nt_pp={}_tau_range=[{},{}].pdf'.format(Nt_pp, tau_start, tau_end)
            title = r'$||(\rho_h - div_h P_h E_h)(t)||$ vs '+t_label
            if skip_titles:
                title = ''
            ax.set_xlabel(t_label, fontsize=16)  
            ax.set_title(title, fontsize=18)
            fig.tight_layout()
            print("saving plot for '"+title+"' in figure '"+diag_fn)
            fig.savefig(diag_fn)     
        
    # diags arrays
    E_energ_diag = np.zeros(Nt+1)
    H_energ_diag = np.zeros(Nt+1)
    divD_norm2_diag = np.zeros(Nt+1)
    time_diag = np.zeros(Nt+1)

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # initial solution

    print(' .. initial solution ..')

    # initial B sol
    B_c = np.zeros(p_V2h.nbasis)
    
    # initial D sol
    if D0_type == 'zero':
        D_c = np.zeros(d_V1h.nbasis)

    elif D0_type == 'pulse':       

        D0 = get_div_free_pulse(x_0=1.0, y_0=1.0, domain=domain)
        
        if D0_proj == 'P_geom':
            print(' .. projecting E0 with commuting projection...')
            D0_h = P_phys_hdiv(D0, d_geomP1, domain, mappings_list)
            D_c = D0_h.coeffs.toarray()
        
        elif D0_proj == 'P_L2':
            
            raise NotImplementedError
            # helper: save/load coefs
            E0dd_filename = m_load_dir+'/E0_pulse_dual_dofs_qp{}.npy'.format(quad_param)
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
        raise ValueError(D0_type)        


    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # time loop
    def compute_diags(D_c, B_c, J_c, nt):
        time_diag[nt] = (nt)*dt
        E_c = d_HH1 @ D_c
        H_c = p_HH2 @ B_c
        E_energ_diag[nt] = np.dot(E_c,p_MM1.dot(E_c))
        H_energ_diag[nt] = np.dot(H_c,d_MM0.dot(H_c))
        
        divD_c = d_DD @ D_c
        divD_norm2_diag[nt] = np.dot(divD_c, d_MM2.dot(divD_c))

    plot_D_field(D_c, nt=0, plot_divD=plot_divD)
    plot_B_field(B_c, nt=0)
    
    f_c = np.copy(f0_c)
    for nt in range(Nt):
        print(' .. nt+1 = {}/{}'.format(nt+1, Nt))

        # 1/2 faraday: Bn -> Bn+1/2
        B_c[:] -= (dt/2) * Far_Op @ D_c

        # ampere: En -> En+1        
        if nt == 0:
            compute_diags(D_c, B_c, f_c, nt=0)

        D_c[:] += dt * (Amp_Op @ B_c - f_c)

        # 1/2 faraday: Bn+1/2 -> Bn+1
        B_c[:] -= (dt/2) * Far_Op @ D_c

        # diags: 
        compute_diags(D_c, B_c, f_c, nt=nt+1)
        
        if is_plotting_time(nt+1):
            plot_D_field(D_c, nt=nt+1, project_sol=project_sol, plot_divD=plot_divD)
            plot_B_field(B_c, nt=nt+1)

        if (nt+1)%(diag_dtau*Nt_pp) == 0:
            tau_here = nt+1
            
            plot_time_diags(
                time_diag, 
                E_energ_diag, 
                H_energ_diag, 
                divD_norm2_diag, 
                nt_start=(nt+1)-diag_dtau*Nt_pp, 
                nt_end=(nt+1), 
            )   

    plot_time_diags(
        time_diag, 
        E_energ_diag, 
        H_energ_diag, 
        divD_norm2_diag, 
        nt_start=0, 
        nt_end=Nt, 
    )

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

    return Nt_pp, dt, norm_op

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