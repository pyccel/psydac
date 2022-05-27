# coding: utf-8

from multiprocessing.dummy import Value
from pytest import param
from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from sympy import lambdify, Matrix
from sympy import pi, cos, sin, Tuple, exp

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
from psydac.feec.multipatch.plotting_utilities          import plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution_hcurl, get_div_free_pulse, get_curl_free_pulse, get_phi_pulse
from psydac.feec.multipatch.utils_conga_2d              import DiagGrid, P0_phys, P1_phys, P2_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities                   import time_count
from psydac.feec.multipatch.utilities                   import get_run_dir, get_plot_dir, get_mat_dir
from psydac.fem.basic                                   import FemField

def test_P1(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, rho_source_type=None, source_type=None, source_proj=None,
        conf_proj='GSP', filter_source=True,
        plot_dir=None, hide_plots=True,
        cb_min_sol=None, cb_max_sol=None,
        m_load_dir="",
):
    """
    project the given source and plot
    """
    diags = {}

    ncells = [nc, nc]
    degree = [deg,deg]

    if m_load_dir is not None:
        if not os.path.exists(m_load_dir):
            os.makedirs(m_load_dir)

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_td_maxwell_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
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

    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    def plot_J_source(f_c):
            print(' .. plotting the source...')
            title = r'$J_h$, source = {}, proj = {}, nc = {}, deg = {}'.format(source_type, source_proj, nc, deg)
            params_str = 'J={}_PJ={}'.format(source_type, source_proj)
            plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_Jh.png', cmap='hot',
                plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

            plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_Jh_xy.png', cmap='hot',
                plot_type='components', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

            plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, title=title, 
                filename=plot_dir+'/'+params_str+'_Jh_vf.png',
                plot_type='vector_field', vf_skip=1, hide_plot=hide_plots)

    def plot_rho_source(rho_c):
            rho_c[:] += 1e-10 * np.random.random(size=V0h.nbasis)
            print(' .. plotting the rho source...')
            title = r'$\rho_h$, source = {}, proj = {}, nc = {}, deg = {}'.format(rho_source_type, source_proj, nc, deg)
            params_str = 'rho={}_PJ={}'.format(rho_source_type, source_proj)
            plot_field(numpy_coeffs=rho_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_rho.png', cmap='hot',
                plot_type='components', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

    t_stamp = time_count(t_stamp)


    # -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- 
    # projecting a source in V1

    if source_type:
        if source_type == 'pulse':

            f0 = get_div_free_pulse(x_0=1.0, y_0=1.0, domain=domain)

        elif source_type == 'cf_pulse':

            f0 = get_curl_free_pulse(x_0=1.0, y_0=1.0, domain=domain)

        elif source_type == '1x':

            f0 = Tuple(1, 0)

        elif source_type == '1y':

            f0 = Tuple(0, 1)

        else:

            f0, u_bc, u_ex, curl_u_ex, div_u_ex = get_source_and_solution_hcurl(
                source_type=source_type, domain=domain, domain_name=domain_name,
            )
            assert u_bc is None  # only homogeneous BC's for now

        # f0_c = np.zeros(V1h.nbasis)

        t_stamp = time_count(t_stamp)
        tilde_f0_c = f0_c = None
        if source_proj == 'P_geom':
            print(' .. projecting the source with commuting projection...')
            f0_h = P1_phys(f0, P1, domain, mappings_list)
            f0_c = f0_h.coeffs.toarray()
            tilde_f0_c = H1_m.dot(f0_c)  # not needed here I think

        elif source_proj == 'P_L2':
            # helper: save/load coefs
            sdd_filename = m_load_dir+'/'+source_type+'_dual_dofs.npy'
            if os.path.exists(sdd_filename):
                print(' .. loading source dual dofs from file {}'.format(sdd_filename))
                tilde_f0_c = np.load(sdd_filename)
            else:
                print(' .. projecting the source with L2 projection...')
                tilde_f0_c = derham_h.get_dual_dofs(space='V1', f=f0, backend_language=backend_language, return_format='numpy_array')
                print(' .. saving source dual dofs to file {}'.format(sdd_filename))
                np.save(sdd_filename, tilde_f0_c)

        else:
            raise ValueError(source_proj)

        t_stamp = time_count(t_stamp)
        if filter_source:
            print(' .. filtering the source...')
            tilde_f0_c = cP1_m.transpose() @ tilde_f0_c

        f0_c = dH1_m.dot(tilde_f0_c)

        t_stamp = time_count(t_stamp)
        
        plot_J_source(f0_c)

    else:

        print(' No J source -- ')

    # -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- -------- 
    # projecting a source in V0

    if rho_source_type:

        if rho_source_type == 'scal_pulse':

            rho = get_phi_pulse(x_0=1.0, y_0=1.0, domain=domain)

        elif rho_source_type == '1':

            rho = 1

        else:
            raise ValueError

        t_stamp = time_count(t_stamp)
        tilde_rho_c = rho_c = None
        if source_proj == 'P_geom':
            print(' .. projecting the source with commuting projection...')
            rho_h = P0_phys(rho, P0, domain, mappings_list)
            rho_c = rho_h.coeffs.toarray()
            tilde_rho_c = H0_m.dot(rho_c)  # not needed here I think

        elif source_proj == 'P_L2':
            # helper: save/load coefs
            sdd_filename = m_load_dir+'/'+rho_source_type+'_rho_dual_dofs.npy'
            if os.path.exists(sdd_filename):
                print(' .. loading source dual dofs from file {}'.format(sdd_filename))
                tilde_rho_c = np.load(sdd_filename)
            else:
                print(' .. projecting the source with L2 projection...')
                tilde_rho_c = derham_h.get_dual_dofs(space='V0', f=rho, backend_language=backend_language, return_format='numpy_array')
                print(' .. saving source dual dofs to file {}'.format(sdd_filename))
                np.save(sdd_filename, tilde_rho_c)

        else:
            raise ValueError(source_proj)

        t_stamp = time_count(t_stamp)
        if filter_source:
            print(' .. filtering the source...')
            tilde_rho_c = cP0_m.transpose() @ tilde_rho_c

        rho_c = dH0_m.dot(tilde_rho_c)

        t_stamp = time_count(t_stamp)
        
        plot_rho_source(rho_c)

    else:

        print(' No rho source -- ')
            
    time_count(t_stamp)
    



#  ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
#  ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
#  ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
#  ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 



if __name__ == '__main__':

    rho_source_type = '' # '1' # 'scal_pulse'
    source_type = '1y' # '' # 
    # source_type = 'cf_pulse' # 'pulse' # 'elliptic_J' #
    source_proj = 'P_geom' #'P_L2' # 
    filter_source = False # True # 

    nc_s = [16]
    deg_s = [7] # [2,3,4,5,6]
    
    case_dir = 'test_P1_J=' + source_type + '_' + source_proj
    if filter_source:
        case_dir += '_filter'
    else:
        case_dir += '_nofilter'
    domain_name = 'pretzel_f'

    conf_proj = 'GSP'

    cb_min_sol = None
    cb_max_sol = None 

    for nc in nc_s:
        for deg in deg_s:

            params = {
                'domain_name': domain_name,
                'nc': nc,
                'deg': deg,
                'source_type': source_type,
                'rho_source_type': rho_source_type,
                'source_proj': source_proj,
                'conf_proj': conf_proj,
                'filter_source': filter_source, 
            }
            # backend_language = 'numba'
            backend_language='pyccel-gcc'

            run_dir = get_run_dir(domain_name, nc, deg, source_type=source_type, conf_proj=conf_proj)
            plot_dir = get_plot_dir(case_dir, run_dir)

            # to save and load matrices
            m_load_dir = get_mat_dir(domain_name, nc, deg)

            print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')
            print(' Calling test_P1() with params = {}'.format(params))
            print('\n --- --- --- --- --- --- --- --- --- --- --- --- --- --- \n')

            test_P1(nc=nc, deg=deg, domain_name=domain_name, backend_language=backend_language, 
            source_type=source_type, rho_source_type=rho_source_type, source_proj=source_proj,
            conf_proj=conf_proj, filter_source=filter_source,
            plot_dir=plot_dir, hide_plots=True,
            cb_min_sol=cb_min_sol, cb_max_sol=cb_max_sol,
            m_load_dir=m_load_dir
            )
    
