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
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution_hcurl, get_div_free_pulse, get_curl_free_pulse
from psydac.feec.multipatch.utils_conga_2d              import DiagGrid, P0_phys, P1_phys, P2_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities                   import time_count #, export_sol, import_sol
from psydac.linalg.utilities                            import array_to_stencil
from psydac.fem.basic                                   import FemField

def test_P1(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_type=None, source_proj=None,
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
            title = r'source $J_h$ (amplitude)'
            params_str = 'J={}_PJ={}'.format(source_type, source_proj)
            plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+'_Jh.pdf', cmap='hot',
                plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)
            plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, title=title, 
                filename=plot_dir+'/'+params_str+'_Jh_vf.pdf',
                plot_type='vector_field', vf_skip=1, hide_plot=hide_plots)


    t_stamp = time_count(t_stamp)
    print()
    print(' -- getting source --')
    if source_type == 'zero':

        f0_c = np.zeros(V1h.nbasis)
    
    else:
        
        if source_type == 'pulse':

            f0 = get_div_free_pulse(x_0=1.0, y_0=1.0, domain=domain)

        elif source_type == 'cf_pulse':

            f0 = get_curl_free_pulse(x_0=1.0, y_0=1.0, domain=domain)

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
