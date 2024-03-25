# coding: utf-8

from pytest import param
from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from sympy import lambdify, Matrix, Tuple, sqrt, cos, acos, sin, arg, I 
from sympy import arg, I, sign

from scipy.sparse import save_npz, load_npz, eye as sparse_eye
from scipy.sparse.linalg import spsolve, norm as sp_norm
from scipy.sparse.linalg import inv as spla_inv
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
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain, build_multipatch_rectangle, build_multipatch_annulus
from psydac.feec.multipatch.examples.ppc_test_cases     import get_div_free_pulse, get_polarized_annulus_potential_solution, get_polarized_annulus_potential_source,get_polarized_annulus_potential_solution_old, get_polarized_annulus_potential_source_old
from psydac.feec.multipatch.utils_conga_2d              import DiagGrid, P_phys_l2, P_phys_hdiv, P_phys_hcurl, P_phys_h1, get_Vh_diags_for
from psydac.feec.multipatch.utilities                   import time_count #, export_sol, import_sol
from psydac.feec.multipatch.bilinear_form_scipy         import construct_pairing_matrix, block_diag_inv
# from psydac.feec.multipatch.conf_projections_scipy      import Conf_proj_0, Conf_proj_1, Conf_proj_0_c1, Conf_proj_1_c1
from psydac.feec.multipatch.conf_projections_scipy      import conf_projectors_scipy
from psydac.feec.pull_push   import push_2d_h1, push_2d_hcurl, push_2d_hdiv, push_2d_l2

from sympde.topology      import NormalVector
from sympde.expr.expr     import BilinearForm
from sympde.topology      import elements_of
from sympde import Tuple
from sympy import arg, I

from psydac.api.postprocessing import OutputManager, PostProcessManager

def solve_td_polarized_polar_maxwell_pbm(
        method='ssc',
        nbc=4, deg=4, 
        mom_pres=False, 
        C1_proj_opt=None,
        Nt_pertau=None, cfl=.8, tau=None,
        nb_tau=1, sol_params=None, source_is_harmonic=True,
        domain_name='pretzel_f', backend_language=None, source_proj='P_geom', source_type='manu_J',
        nb_patch_r=1, nb_patch_theta=4, 
        project_sol=False, filter_source=True, quad_param=1,
        solution_type='zero', solution_proj='P_geom', 
        plot_source=False, plot_dir=None, hide_plots=True, plot_time_ranges=None, 
        show_grid=True,
        plot_variables=["E", "D", "B", "Eex", "Dex", "Bex", "divD"], diag_dtau=None,
        skip_plot_titles=False,
        cb_min_sol=None, cb_max_sol=None,
        m_load_dir="",
        dry_run = False
):
    """
    solver for the TD Maxwell problem: find E(t) in H(curl), B in L2, such that

      dt D - curl H = -J             on \Omega
      dt B + curl E = 0              on \Omega
      D = E + kappa * (E - (b \cdot E)b for |b| = 1
      n x E = 0                      on \partial \Omega

    with SSC scheme on a 2D multipatch domain \Omega
    involving two strong sequences, a  and a dual 

      primal: p_V0h  --grad->  p_V1h  -—curl-> p_V2h        (with homogeneous bc's)
                                (Eh)            (Bh)

                                (Dh, Jh)        (Hh)
      dual:   d_V2h  <--div--  d_V1h  <—curl-- d_V0h        (no bc's)

    the semi-discrete level the equations read

        Ampere: 
            Hh = p_HH2 @ Bh
            dt Dh - d_CC @ Hh = - Jh         
        with
            p_HH2 = hodge:   p_V2h -> d_V0h
            d_CC  = curl:    d_V0h -> d_V1h

        Faraday:    
            Eh = d_cP1.transpose() @ d_KK1 @ Dh
            dt Bh + p_curl @ Eh = 0              
        with
            d_KK1 = coupling matrix of (p_V1h, d_V1h):  d_V1h -> p_V1h
            p_CC  = curl:    p_V1h -> p_V2h

    :param nbc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param source_proj: approximation operator for the source (see later)
    :param source_type: must be implemented in get_source_and_solution()
    :param m_load_dir: directory for matrix storage
    """

    if solution_type == 'polarized':
        omega = sol_params['omega']
        kx = sol_params['kx']
        ky = sol_params['ky']
        kappa = sol_params['kappa']
        alpha = sol_params['alpha']
    
    else:
        raise NotImplementedError
    

    diags = {}


    ncells = [nbc, nbc]
    degree = [deg,deg]

    final_time = nb_tau * tau

    print('final_time = ', final_time)
    if plot_time_ranges is None:
        plot_time_ranges = [[0, final_time], 2]

    if diag_dtau is None:
        diag_dtau = nb_tau//10

    if m_load_dir is not None:
        pm_load_dir = m_load_dir+"primal"
        dm_load_dir = m_load_dir+"dual"
        for load_dir in [m_load_dir, pm_load_dir, dm_load_dir]:        
            if not os.path.exists(load_dir):
                os.makedirs(load_dir)

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_td_maxwell_pbm function with: ')
    print(' domain_name = {}'.format(domain_name))
    if domain_name == 'multipatch_rectangle':
        print(' nb_patches = [{},{}]'.format(nb_patch_r, nb_patch_theta))
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' solution_type = {}'.format(solution_type))
    print(' solution_proj = {}'.format(solution_proj))
    print(' source_type = {}'.format(source_type))
    print(' source_proj = {}'.format(source_proj))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    debug = False

    print()
    print(' -- building discrete spaces and operators  --')

    t_stamp = time_count()
    print(' .. multi-patch domain...')
    
    #####
    epsilon = 0.1
    #alpha = 1.5
    r_min = 0.25
    r_max = 1

    #####
    # if domain_name in ['multipatch_rectangle', 'mpr_collela']:
    #     if domain_name == 'multipatch_rectangle':
    #         F_name = 'Identity'
    #     else:
    #         F_name = 'Collela'
        
    #     # domain, domain_h, bnds = build_multipatch_rectangle(
    #     #     nb_patch_r, nb_patch_theta, 
    #     #     x_min=0, x_max=np.pi,
    #     #     y_min=0, y_max=np.pi,
    #     #     perio=[False,False],
    #     #     ncells=ncells,
    #     #     F_name=F_name,
    #     #     )

    #     domain = build_multipatch_domain('annulus_4', r_min, r_max)
    #     domain_h = discretize(domain, ncells=ncells)#, periodic=[False, True])
    # else:
    #     raise NotImplementedError
    #     domain = build_multipatch_domain(domain_name=domain_name)

    domain = build_multipatch_annulus(nb_patch_r, nb_patch_theta, r_min, r_max)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = [m.get_callable_mapping() for m in mappings.values()]


    # for diagnostics
    diag_grid = DiagGrid(mappings=mappings, N_diag=100)

    unif_grid = True

    total_nb_cells_x = nb_patch_r*ncells[0]
    h = np.pi/(total_nb_cells_x)
    
    if unif_grid:
        #standard uniform grid
        grid_type=[np.linspace(-1,1,nc+1) for nc in ncells]

    else:
        #this seems to be quite good
        x = 1-1/ncells[0] #leftmost point - the rest is uniform grid
        grid_type=[np.concatenate(([-1],np.linspace(-x,x,num=nc-1),[1])) for nc in ncells]
    
        # h=(2*x)/(ncells[0]-2)
    

    domain_h = discretize(domain, ncells=ncells)  # todo: remove this ?

    t_stamp = time_count(t_stamp)
    print('building (primal) derham sequence...')
    p_derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    p_derham_h = discretize(p_derham, domain_h, degree=degree, grid_type=grid_type)

    p_V0h = p_derham_h.V0
    p_V1h = p_derham_h.V1
    p_V2h = p_derham_h.V2

    if method == 'ssc':
        t_stamp = time_count(t_stamp)
        print('building dual derham sequence...')
        d_derham  = Derham(domain, ["H1", "Hdiv", "L2"])
        dual_degree = [d-1 for d in degree]
        d_derham_h = discretize(d_derham, domain_h, degree=dual_degree, grid_type=grid_type, pads=degree)

        d_V0h = d_derham_h.V0
        d_V1h = d_derham_h.V1
        d_V2h = d_derham_h.V2

        t_stamp = time_count(t_stamp)       
        print('building the conforming projection matrices in primal spaces ...')
        p_PP0, p_PP1, p_PP2 = conf_projectors_scipy(p_derham_h, reg=1, mom_pres=mom_pres, C1_proj_opt=C1_proj_opt, hom_bc=True)
        t_stamp = time_count(t_stamp)       
        print('building the conforming projection matrices in dual spaces ...')
        d_PP0, d_PP1, d_PP2 = conf_projectors_scipy(d_derham_h, reg=0, mom_pres=mom_pres, C1_proj_opt=C1_proj_opt, hom_bc=False)

    
    elif method in ['swc_C1', 'swc_C0']:
        d_V0h = p_derham_h.V2
        d_V1h = p_derham_h.V1
        d_V2h = p_derham_h.V0

        if method == 'swc_C1':
            reg=1
        else:
            reg=0

        t_stamp = time_count(t_stamp)       
        print('building the conforming projection matrices ...')
        p_PP0, p_PP1, p_PP2 = conf_projectors_scipy(p_derham_h, reg=reg, mom_pres=mom_pres, C1_proj_opt=C1_proj_opt, hom_bc=True)
        d_derham = p_derham
        d_derham_h = p_derham_h
        d_PP0, d_PP1, d_PP2 = conf_projectors_scipy(d_derham_h, reg=reg, mom_pres=mom_pres, C1_proj_opt=C1_proj_opt, hom_bc=True)

    else:
        raise NotImplementedError
    
    t_stamp = time_count(t_stamp)
    print('building the mass matrices ...')
    
    basic_MM = False #True  # do not use with a mapping !!
    if basic_MM:
        p_MM0 = construct_pairing_matrix(p_V0h,p_V0h,domain_h).tocsr()  # matrix in scipy format
        #p_MM0 = p_PP0@p_MM0_b@p_PP0
        print('inverting p_MM0...')
        p_MM0_inv = spla_inv(p_MM0.tocsc())

        p_MM1 = construct_pairing_matrix(p_V1h,p_V1h,domain_h).tocsr()  # matrix in scipy format
        #p_MM1 = p_PP1@p_MM1_b@p_PP1
        print('inverting p_MM1...')
        p_MM1_inv = spla_inv(p_MM1.tocsc())

        p_MM2 = construct_pairing_matrix(p_V2h,p_V2h,domain_h).tocsr()  # matrix in scipy format
        #p_MM2 = p_PP2@p_MM2_b@p_PP2
        print('inverting p_MM2...')
        p_MM2_inv = spla_inv(p_MM2.tocsc())

    else:        
        ## NOTE: with a strong-strong diagram we should not call these "Hodge" operators !! 
        p_HOp0    = HodgeOperator(p_V0h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=0)
        p_MM0     = p_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
        p_MM0_inv = p_HOp0.to_sparse_matrix()                # inverse mass matrix

        p_HOp1   = HodgeOperator(p_V1h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=1)
        p_MM1     = p_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
        p_MM1_inv = p_HOp1.to_sparse_matrix()                # inverse mass matrix

        p_HOp2    = HodgeOperator(p_V2h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=2)
        p_MM2     = p_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix
        p_MM2_inv = p_HOp2.to_sparse_matrix()                # inverse mass matrix

    if method == 'ssc':

        if basic_MM:
            d_MM0 = construct_pairing_matrix(d_V0h,d_V0h,domain_h).tocsr()  # matrix in scipy format
            #d_MM0 = d_PP0@d_MM0_b@d_PP0
            print('inverting d_MM0...')
            d_MM0_inv = spla_inv(d_MM0.tocsc())

            d_MM1 = construct_pairing_matrix(d_V1h,d_V1h,domain_h).tocsr()  # matrix in scipy format
            #d_MM1 = d_PP1@d_MM1_b@d_PP1
            print('inverting d_MM1...')
            d_MM1_inv = spla_inv(d_MM1.tocsc())

            d_MM2 = construct_pairing_matrix(d_V2h,d_V2h,domain_h).tocsr()  # matrix in scipy format
            #d_MM2 = d_PP2@d_MM2_b@d_PP2

        else:   
            d_HOp0   = HodgeOperator(d_V0h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=0)
            d_MM0     = d_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
            d_MM0_inv = d_HOp0.to_sparse_matrix()                # inverse mass matrix

            d_HOp1   = HodgeOperator(d_V1h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=1)
            d_MM1     = d_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
            d_MM1_inv = d_HOp1.to_sparse_matrix()                # inverse mass matrix

            d_HOp2   = HodgeOperator(d_V2h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=2)
            d_MM2    = d_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix

    elif method in ['swc_C1', 'swc_C0']:

        # not sure whether useful...
        d_MM0     = p_MM2_inv
        d_MM0_inv = p_MM2

        d_MM1     = p_MM1_inv
        d_MM1_inv = p_MM1
        
        d_MM2     = p_MM0_inv

    else:
        raise NotImplementedError
    
    t_stamp = time_count(t_stamp)
    print('building the Hodge matrices ...')
    if method == 'ssc':
        p_KK2_storage_fn = m_load_dir+'/p_KK2.npz'
        if os.path.exists(p_KK2_storage_fn):
            # matrix is stored
            print('loading pairing matrix found in '+p_KK2_storage_fn)
            p_KK2 = load_npz(p_KK2_storage_fn)
        else:
            print('pairing matrix not found, computing... ')
            p_KK2 = construct_pairing_matrix(d_V0h,p_V2h,domain_h).tocsr()  # matrix in scipy format
            #p_KK2 = d_PP0@p_KK2_b@p_PP2
            t_stamp = time_count(t_stamp)
            print('storing pairing matrix in '+p_KK2_storage_fn)
            save_npz(p_KK2_storage_fn, p_KK2)

        d_KK1_storage_fn = m_load_dir+'/d_KK1.npz'
        if os.path.exists(d_KK1_storage_fn):
            # matrix is stored
            d_KK1 = load_npz(d_KK1_storage_fn)
        else:
            d_KK1 = construct_pairing_matrix(p_V1h,d_V1h,domain_h).tocsr()  # matrix in scipy format
            #d_KK1 = p_PP1@d_KK1_b@d_PP1
            save_npz(d_KK1_storage_fn, d_KK1)

        p_HH2 = d_MM0_inv @ d_PP0.transpose() @ p_KK2
        d_HH1 = p_MM1_inv @ p_PP1.transpose() @ d_KK1

    elif method in ['swc_C1', 'swc_C0']:

        p_HH2 = d_MM0_inv 
        d_HH1 = p_MM1_inv 
        d_KK1 = sparse_eye(p_V1h.nbasis)
        p_KK2 = sparse_eye(p_V2h.nbasis)
    else:
        raise NotImplementedError

    p_I1 = sparse_eye(p_V1h.nbasis)

    t_stamp = time_count(t_stamp)
    print(' .. differential operators...')
    p_bD0, p_bD1 = p_derham_h.broken_derivatives_as_operators
    p_bG = p_bD0.to_sparse_matrix() # broken grad (primal)
    p_GG = p_bG @ p_PP0             # Conga grad (primal)
    p_bC = p_bD1.to_sparse_matrix() # broken curl (primal: scalar-valued)
    p_CC = p_bC @ p_PP1             # Conga curl (primal)

    if method == 'ssc':
        d_bD0, d_bD1 = d_derham_h.broken_derivatives_as_operators
        d_bC = d_bD0.to_sparse_matrix() # broken curl (dual: vector-valued)
        d_CC = d_bC @ d_PP0             # Conga curl (dual)    
        d_bD = d_bD1.to_sparse_matrix() # broken div
        d_DD = d_bD @ d_PP1             # Conga div (dual)    
        
    elif method in ['swc_C1', 'swc_C0']:
        d_CC = (p_bC @ p_PP1).transpose()
        d_DD = - (p_bG @ p_PP0).transpose()
        
    else:
        raise NotImplementedError

    p_geomP0, p_geomP1, p_geomP2 = p_derham_h.projectors()
    if method == 'ssc':
        d_geomP0, d_geomP1, d_geomP2 = d_derham_h.projectors()
    else:
        d_geomP0, d_geomP1, d_geomP2 = p_derham_h.projectors()

    t_stamp = time_count(t_stamp)
    print(' .. Ampere and Faraday evolution (curl . Hodge) operators...')

    x,y = domain.coordinates
    nb = sqrt(alpha**2 * x**2 + y**2 * 1/alpha**2)
    b = Tuple(-y/alpha * 1/nb, alpha * x * 1/nb)

    u, v = elements_of(p_derham.V1, names='u, v')
    #u, v = elements_of(d_derham.V1, names='u, v')

    
    bE = dot(v, b)
    Eu = dot(u, v)
    ub = dot(u, b)

    mass = BilinearForm((v,u), integral(domain, ((1+kappa)*Eu - kappa * bE * ub)))
    massh = discretize(mass, domain_h, [p_V1h, p_V1h])
    #mass = BilinearForm((u,v), integral(domain, (Eu + kappa * bE * ub)/(1+kappa) ))
    #massh = discretize(mass, domain_h, [d_V1h, d_V1h])

    M = massh.assemble().tosparse().toarray()

    Amp_Op = d_CC @ p_HH2
    if method == 'ssc':
        Coup_Op = np.linalg.inv(M) @ p_PP1.transpose() @ d_KK1
        #Coup_Op = p_MM1_inv @ p_PP1.transpose() @ d_KK1 @ d_MM1_inv @ M

    else:
       # Coup_Op = np.linalg.inv(M) 
        Coup_Op = p_MM1_inv @ M @ d_MM1

    Far_Op = p_CC  

    print(' -- doing some checks: (all matrix norms should be zero)')
    for spn_name, spn in [
        ["sp_norm(d_DD@Amp_Op)", sp_norm(d_DD@Amp_Op)],
        ["sp_norm(p_bC@p_bG)", sp_norm(p_bC@p_bG)],
        ["sp_norm(p_PP0 - p_PP0@p_PP0)",sp_norm(p_PP0 - p_PP0@p_PP0)],
        ["sp_norm(p_PP1 - p_PP1@p_PP1)",sp_norm(p_PP1 - p_PP1@p_PP1)],
        ["sp_norm((p_PP1-p_I1)@p_bG@p_PP0)",sp_norm((p_PP1-p_I1)@p_bG@p_PP0)],
        ["sp_norm(p_CC@p_GG)",sp_norm(p_CC@p_GG)],
        ]:
        print(f'{spn_name} = {spn}')
        if abs(spn) > 1e-10:
            print(20*" WARNING ! ")
            print(f'{spn_name} is too large \n -----------------------------')

    print(' .. ok checks done -- ')
    
    t_stamp = time_count(t_stamp)


    if Nt_pertau is None:
        if not( 0 < cfl <= 1):
            print(' ******  ****** ******  ****** ******  ****** ')
            print('         WARNING !!!  cfl = {}  '.format(cfl))
            print(' ******  ****** ******  ****** ******  ****** ')
        print(Amp_Op.shape)
        print(Far_Op.shape)
        print(p_V2h.nbasis)
        Nt_pertau, dt, norm_curlh = compute_stable_dt(cfl, tau, Amp_Op, Far_Op@Coup_Op, p_V2h.nbasis)        
        #dt = dt/2
        #Nt_pertau = 2 * Nt_pertau
        print(" *** with C1_proj_opt = ", C1_proj_opt)
        print("h    = ", h)
        print("h*total_nb_cells_x = ", h*total_nb_cells_x)
        print("dt   = ", dt)
        print("dt/h = ", dt/h)
        print("norm_curlh = ", norm_curlh)
        print("h*norm_curlh = ", h*norm_curlh)
        final_time = tau * nb_tau
        print('final_time = ', final_time)
        print('Nt = ', Nt_pertau * nb_tau)
        diags["h*norm_curlh"] = h*norm_curlh
    else:
        diags["h*norm_curlh"] = 0
        dt = tau/Nt_pertau
        norm_curlh = None

    diags["Nt_pertau"]    = Nt_pertau
    Nt = Nt_pertau * nb_tau

    def is_plotting_time(nt):
        answer = (nt==0) or (nt==Nt)
        for tau_range, nt_plots_pp in plot_time_ranges:
            if answer:
                break
            tp = max(Nt_pertau//nt_plots_pp,1)
            answer = (tau_range[0]*tau <= nt*dt <= tau_range[1]*tau and (nt)%tp == 0)
        return answer
    
    plot_divD = ("divD" in plot_variables)

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
    elif source_type == 'polarized':
        f0_c = np.zeros(d_V1h.nbasis)
    else:
        raise ValueError(source_type)
        
    t_stamp = time_count(t_stamp)
    
    def plot_D_field(D_c, E_c, J_c, nt, project_sol=False, plot_divD=False, label=''):
        if plot_dir:

            if method in ['swc_C1', 'swc_C0']:
                Dp_c = p_MM1_inv @ D_c # get coefs in primal basis for plotting 
                Vh = p_V1h
                kind = 'hcurl'

                title = r'$Proj D_h$ (amplitude) at $t = {:5.4f}$'.format(dt*nt)
                params_str = 'Nt_pertau={}'.format(Nt_pertau)
                D_test = p_MM1_inv @ (p_I1 - p_PP1.transpose() ) @ p_MM1 @ Dp_c
                plot_field(numpy_coeffs=D_test, Vh=Vh, space_kind=kind, domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+label+'_Dh_proj_nt={}.pdf'.format(nt),
                plot_type='amplitude', show_grid=show_grid, cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

            
            else:
                assert method == 'ssc'
                Dp_c = D_c  # keep coefs in dual space
                Vh = d_V1h
                kind = 'hdiv'

            # project the homogeneous solution on the conforming problem space
            if project_sol:
                raise NotImplementedError
                # t_stamp = time_count(t_stamp)
                print(' .. projecting the homogeneous solution on the conforming problem space...')
                Ep_c = p_PP1_m.dot(E_c)

            print(' .. plotting the '+label+' D field...')                
            title = r'$D_h$ (amplitude) at $t = {:5.4f}$'.format(dt*nt)
            params_str = 'Nt_pertau={}'.format(Nt_pertau)
            plot_field(numpy_coeffs=Dp_c, Vh=Vh, space_kind=kind, domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+label+'_Dh_nt={}.pdf'.format(nt),
                plot_type='amplitude', show_grid=show_grid, cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

            # params_str = 'Nt_pertau={}'.format(Nt_pertau)
            # plot_field(numpy_coeffs=J_c, Vh=Vh, space_kind='hdiv', domain=domain, surface_plot=False, title=title, 
            #     filename=plot_dir+'/'+params_str+label+'_Jh_nt={}.pdf'.format(nt),
            #     plot_type='amplitude', show_grid=show_grid, cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

            title = r'$E_h$ (amplitude) at $t = {:5.4f}$'.format(dt*nt)
            params_str = 'Nt_pertau={}'.format(Nt_pertau)
            plot_field(numpy_coeffs=E_c, Vh=p_V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+label+'_Eh_nt={}.pdf'.format(nt),
                plot_type='amplitude', show_grid=show_grid, cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)
            

            if plot_divD:
                params_str = 'Nt_pertau={}'.format(Nt_pertau)
                plot_type = 'amplitude'

                divD_c = d_DD @ D_c  # divD coefs in dual basis of pV0 (swc) or basis of dV2 (ssc)

                if method in ['swc_C1', 'swc_C0']:
                    Vh_aux = p_V0h                    
                    divDp_c = p_MM0_inv @ divD_c # get coefs in primal basis for plotting 
                    # here, divD_c = coefs in p_V0h
                    # divD_norm2 = np.dot(divD_c, divDp_c) # = np.dot(divDp_c, p_MM0.dot(divDp_c))
                    divD_norm2 = np.dot(divDp_c, p_MM0.dot(divDp_c))
                    kind = 'h1'
                else:
                    assert method == 'ssc'
                    Vh_aux = d_V2h  # plot directly in dual space                    
                    divDp_c = divD_c
                    divD_norm2 = np.dot(divD_c, d_MM2.dot(divD_c))
                    kind = 'l2'

                title = r'div $D_h$ at $t = {:5.4f}, norm = {}$'.format(dt*nt, np.sqrt(divD_norm2))
                plot_field(numpy_coeffs=divDp_c, Vh=Vh_aux, space_kind=kind, domain=domain, surface_plot=False, title=title, 
                    filename=plot_dir+'/'+params_str+label+'_divDh_nt={}.pdf'.format(nt),
                    plot_type=plot_type, cb_min=None, cb_max=None, hide_plot=hide_plots)
                
        else:
            print(' -- WARNING: unknown plot_dir !!')

    def plot_B_field(B_c, H_c, nt, label=''):

        if plot_dir:

            print(' .. plotting B field...')
            params_str = 'Nt_pertau={}'.format(Nt_pertau)

            title = r'$B_h$ (amplitude) for $t = {:5.4f}$'.format(dt*nt)
            plot_field(numpy_coeffs=B_c, Vh=p_V2h, space_kind='l2', domain=domain, surface_plot=False, title=title, 
                filename=plot_dir+'/'+params_str+label+'_Bh_nt={}.pdf'.format(nt),
                plot_type='amplitude', show_grid=show_grid, cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)
            
           # plot_field(numpy_coeffs=p_HH2@B_c - H_c, Vh=d_V0h, space_kind='h1', domain=domain, surface_plot=False, title=title, 
           #     filename=plot_dir+'/'+params_str+label+'_Hh_nt={}.pdf'.format(nt),
           #     plot_type='amplitude', show_grid=show_grid, cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

        else:
            print(' -- WARNING: unknown plot_dir !!')

    def plot_time_diags(time_diag, E_norm2_diag, H_norm2_diag, J_norm2_diag, divD_norm2_diag, nt_start, nt_end, 
        GaussErr_norm2_diag=None, GaussErrP_norm2_diag=None, 
        PE_norm2_diag=None, I_PE_norm2_diag=None, skip_titles=True):
        nt_start = max(nt_start, 0)
        nt_end = min(nt_end, Nt)
        tau_start = nt_start/Nt_pertau
        tau_end = nt_end/Nt_pertau

        if source_is_harmonic:
            td = time_diag[nt_start:nt_end+1]/tau
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
        diag_fn = plot_dir+'/diag_E_norm_Nt_pertau={}_tau_range=[{},{}].pdf'.format(Nt_pertau, tau_start, tau_end)
        print("saving plot for '"+title+"' in figure '"+diag_fn)
        fig.savefig(diag_fn)

        # energy
        fig, ax = plt.subplots()
        E_energ = .5*E_norm2_diag[nt_start:nt_end+1]
        B_energ = .5*H_norm2_diag[nt_start:nt_end+1]
        #J_energ = .5*J_norm2_diag[nt_start:nt_end+1]
        ax.plot(td, E_energ, '-', ms=7, mfc='None', c='k', label=r'$\frac{1}{2}||E||^2$') #, zorder=10)
        ax.plot(td, B_energ, '-', ms=7, mfc='None', c='g', label=r'$\frac{1}{2}||B||^2$') #, zorder=10)
        #ax.plot(td, J_energ, '-', ms=7, mfc='None', c='g', label=r'$\frac{1}{2}||J||^2$') #, zorder=10)
        ax.plot(td, E_energ+B_energ, '-', ms=7, mfc='None', c='b', label=r'$\frac{1}{2}(||E||^2+||B||^2)$') #, zorder=10)

        ax.legend(loc='best')
        if skip_titles:  
            title = ''
        else:
            title = r'energy vs '+t_label
        if solution_type == 'pulse':
            ax.set_ylim([0, 7])
        
        ax.set_xlabel(t_label, fontsize=16)                    
        ax.set_title(title, fontsize=18)
        fig.tight_layout()
        diag_fn = plot_dir+'/diag_energy_Nt_pertau={}_tau_range=[{},{}].pdf'.format(Nt_pertau, tau_start, tau_end)
        print("saving plot for '"+title+"' in figure '"+diag_fn)
        fig.savefig(diag_fn)

        # norm || div E ||
        fig, ax = plt.subplots()
        ax.plot(td, np.sqrt(divD_norm2_diag[nt_start:nt_end+1]), '-', ms=7, mfc='None', mec='k') #, label='||E||', zorder=10)
        diag_fn = plot_dir+'/diag_divD_Nt_pertau={}_tau_range=[{},{}].pdf'.format(Nt_pertau, tau_start, tau_end)
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
            diag_fn = plot_dir+'/diag_GaussErr_Nt_pertau={}_tau_range=[{},{}].pdf'.format(Nt_pertau, tau_start, tau_end)
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
            diag_fn = plot_dir+'/diag_GaussErrP_Nt_pertau={}_tau_range=[{},{}].pdf'.format(Nt_pertau, tau_start, tau_end)
            title = r'$||(\rho_h - div_h P_h E_h)(t)||$ vs '+t_label
            if skip_titles:
                title = ''
            ax.set_xlabel(t_label, fontsize=16)  
            ax.set_title(title, fontsize=18)
            fig.tight_layout()
            print("saving plot for '"+title+"' in figure '"+diag_fn)
            fig.savefig(diag_fn)     
        
    
    def project_exact_polarized_solution(t, proj_type='P_geom'):
    
        E_ex, B_ex, D_ex, J_ex = get_polarized_annulus_potential_solution(b, omega, kx, epsilon, kappa, t=t, r_min=r_min, r_max=r_max, domain=domain)

        if proj_type == 'P_geom':
            
            # E (in p_V1h) and D,J (in d_V1h)
            # B (in p_V2h) and H (in d_V0h)

            Eex_h = P_phys_hcurl(E_ex, p_geomP1, domain, mappings_list)
            Eex_c = Eex_h.coeffs.toarray()
            Bex_h = P_phys_l2(B_ex, p_geomP2, domain, mappings_list)
            Bex_c = Bex_h.coeffs.toarray()
            if method == 'ssc':
                Dex_h = P_phys_hdiv(D_ex, d_geomP1, domain, mappings_list)
                Dex_c = Dex_h.coeffs.toarray()
                Jex_h = P_phys_hdiv(J_ex, d_geomP1, domain, mappings_list)
                Jex_c = Jex_h.coeffs.toarray()
                Hex_h = P_phys_h1(B_ex, d_geomP0, domain, mappings_list)
                Hex_c = Hex_h.coeffs.toarray()

            else:
                Dex_h = P_phys_hcurl(D_ex, p_geomP1, domain, mappings_list).coeffs.toarray()
                Dex_c = p_MM1 @ Dex_h
                Jex_h = P_phys_hcurl(J_ex, p_geomP1, domain, mappings_list).coeffs.toarray()
                Jex_c = p_MM1 @ Jex_h
                Hex_h = P_phys_l2(B_ex, p_geomP2, domain, mappings_list).coeffs.toarray()
                Hex_c = p_MM2 @ Bex_c


        elif proj_type == 'P_L2':

            tilde_Eex_c = p_derham_h.get_dual_dofs(space='V1', f=E_ex, backend_language=backend_language, return_format='numpy_array')
            Eex_c = p_MM1_inv @ tilde_Eex_c            
            tilde_B_ex_c = p_derham_h.get_dual_dofs(space='V2', f=B_ex, backend_language=backend_language, return_format='numpy_array')
            Bex_c = p_MM2_inv @ tilde_B_ex_c

            tilde_Dex_c = d_derham_h.get_dual_dofs(space='V1', f=D_ex, backend_language=backend_language, return_format='numpy_array')
            Dex_c = d_MM1_inv @ tilde_Dex_c
            tilde_Jex_c = d_derham_h.get_dual_dofs(space='V1', f=J_ex, backend_language=backend_language, return_format='numpy_array')
            Jex_c = d_MM1_inv @ tilde_Jex_c

            tilde_H_ex_c = d_derham_h.get_dual_dofs(space='V0', f=B_ex, backend_language=backend_language, return_format='numpy_array')
            Hex_c = d_MM0_inv @ tilde_H_ex_c
  
        else: 
            raise NotImplementedError
        
        return Dex_c, Jex_c, Bex_c, Eex_c, Hex_c
        
    # diags arrays
    E_energ_diag = np.zeros(Nt+1)
    H_energ_diag = np.zeros(Nt+1)
    J_energ_diag = np.zeros(Nt+1)

    E_exact_energ_diag = np.zeros(Nt+1)
    H_exact_energ_diag = np.zeros(Nt+1)
    J_exact_energ_diag = np.zeros(Nt+1)

    divD_norm2_diag = np.zeros(Nt+1)
    time_diag = np.zeros(Nt+1)

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # initial solution

    print(' .. initial solution ..')

    if solution_type == 'polarized':
        D_c, J_c, B_c, E_c, H_c = project_exact_polarized_solution(t=0, proj_type='P_geom')

    else:
        raise NotImplementedError    


    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # time loop
    def compute_diags(D_c, E_c, B_c, H_c, J_c, nt):
        time_diag[nt] = (nt)*dt
        H_c = p_HH2 @ B_c 
        E_energ_diag[nt] = np.dot(E_c, d_KK1.dot(D_c))
        H_energ_diag[nt] = np.dot(H_c, p_KK2.dot(B_c))
        J_energ_diag[nt] = np.dot(J_c,d_MM1.dot(J_c))
        divD_c = d_DD @ D_c
        # print(divD_c.shape)
        # print(d_MM2.shape)
        # print("p_V0h.nbasis = ", p_V0h.nbasis)
        # print("p_V1h.nbasis = ", p_V1h.nbasis)
        if method in ['swc_C1', 'swc_C0']:
            divDp_c = p_MM0_inv @ divD_c # get coefs in primal basis
            # here, divD_c = coefs in p_V0h
            divD_norm2_diag[nt] = np.dot(divDp_c, divD_c)
        else:
            assert method == 'ssc'
            divD_norm2_diag[nt] = np.dot(divD_c, d_MM2.dot(divD_c))

    if "D" in plot_variables: plot_D_field(D_c, E_c, J_c, nt=0, plot_divD=plot_divD)
    if "B" in plot_variables: plot_B_field(B_c, H_c, nt=0)
    if solution_type == 'polarized' and ("Dex" in plot_variables or "Bex" in plot_variables):
        Dex_c, Jex_c, Bex_c, Eex_c, Hex_c = project_exact_polarized_solution(t=0, proj_type='P_geom')
        if "Dex" in plot_variables: plot_D_field(Dex_c, Eex_c, Jex_c, nt=0, plot_divD=plot_divD, label='_ex')
        if "Bex" in plot_variables: plot_B_field(Bex_c, Hex_c, nt=0, label='_ex')


    compute_diags(D_c, E_c, B_c, H_c, J_c, nt=0)
    for nt in range(Nt):
        print(' .. nt+1 = {}/{}'.format(nt+1, Nt))

        # 1/2 faraday: Bn -> Bn+1/2 from En
        B_c[:] -= (dt/2) * Far_Op @ E_c

        # 1/2 faraday: Bn -> Bn+1/2 from En
        # dual ampere: Dn -> Dn+1 from proj Bn+1/2
        # coupling terms: En -> En+1 from Dn+1       
        # 1/2 faraday: Bn+1/2 -> Bn+1 from En+1

        # project source f_c = J_c
        if method == 'ssc':
            Jex_h = P_phys_hdiv(get_polarized_annulus_potential_source(b, omega, kx, epsilon, kappa, (nt+1/2)*dt, r_min, r_max, domain), d_geomP1, domain, mappings_list).coeffs.toarray()
        else:
            Jex_h = p_PP1.transpose() @ p_MM1 @ P_phys_hcurl(get_polarized_annulus_potential_source(b, omega, kx, epsilon, kappa, (nt+1/2)*dt, r_min, r_max, domain), p_geomP1, domain, mappings_list).coeffs.toarray()

        J_c[:] = Jex_h

        D_c[:] += dt * (Amp_Op @ B_c - J_c)

        E_c[:] = Coup_Op @ D_c

        # 1/2 faraday: Bn+1/2 -> Bn+1
        B_c[:] -= (dt/2) * Far_Op @ E_c

        # diags: 
        compute_diags(D_c, E_c, B_c, H_c, J_c, nt=nt+1)
        
        if is_plotting_time(nt+1):
            if "D" in plot_variables: plot_D_field(D_c, E_c, J_c, nt=nt+1, project_sol=project_sol, plot_divD=plot_divD)
            if "B" in plot_variables: plot_B_field(B_c, H_c, nt=nt+1)
            if solution_type == 'polarized' and ("Dex" in plot_variables or "Bex" in plot_variables):
                Dex_c, Jex_c, Bex_c, Eex_c, Hex_c  = project_exact_polarized_solution(t=(nt+1)*dt, proj_type='P_geom')
                if "Dex" in plot_variables: plot_D_field(Dex_c, Eex_c, Jex_c, nt=nt+1, plot_divD=plot_divD, label='_ex')
                if "Bex" in plot_variables: plot_B_field(Bex_c, H_c, nt=nt+1, label='_ex')
                if "Dex" in plot_variables: plot_D_field(Dex_c-D_c, Eex_c-E_c, Jex_c, nt=nt+1, plot_divD=plot_divD, label='_err')
                if "Bex" in plot_variables: plot_B_field(Bex_c-B_c, Hex_c - H_c, nt=nt+1, label='_err')
            
        if (nt+1)%(diag_dtau*Nt_pertau) == 0:
            tau_here = nt+1
            
            plot_time_diags(
                time_diag, 
                E_energ_diag, 
                H_energ_diag, 
                J_energ_diag,
                divD_norm2_diag, 
                nt_start=(nt+1)-diag_dtau*Nt_pertau, 
                nt_end=(nt+1), 
            )   

    plot_time_diags(
        time_diag, 
        E_energ_diag, 
        H_energ_diag, 
        J_energ_diag,
        divD_norm2_diag, 
        nt_start=0, 
        nt_end=Nt, 
    )

    if solution_type == 'polarized':
        t_stamp = time_count(t_stamp)

        print(' .. comparing with a projection of the exact polarized solution...')
        Dex_c, Jex_c, Bex_c, Eex_c, Hex_c  = project_exact_polarized_solution(t=final_time, proj_type='P_geom')

        # D error (in d_V1h)
        D_err_c = D_c - Dex_c
        D_L2_error = np.sqrt(np.dot(D_err_c, d_MM1.dot(D_err_c)))

        # E error (in p_V1h)
        # E_err_c = p_PP1 @ d_HH1 @ D_c - Eex_c
        E_err_c = E_c - Eex_c
        E_L2_error = np.sqrt(np.dot(E_err_c, p_MM1.dot(E_err_c)))

        # B error (in p_V2h)
        B_err_c = B_c - Bex_c
        B_L2_error = np.sqrt(np.dot(B_err_c, p_MM2.dot(B_err_c)))

        # H error (in d_V0h)
        # H_err_c = d_PP0 @ p_HH2 @ B_c - Hex_c
        H_err_c = p_HH2 @ B_c - Hex_c
        H_L2_error = np.sqrt(np.dot(H_err_c, d_MM0.dot(H_err_c)))
                
        print("D_error = ", D_L2_error)
        print("E_error = ", E_L2_error) 
        print("B_error = ", B_L2_error) 
        print("H_error = ", H_L2_error)

        t_stamp = time_count(t_stamp)
        diags["D_error"] = D_L2_error
        diags["E_error"] = E_L2_error
        diags["B_error"] = B_L2_error
        diags["H_error"] = H_L2_error
        
    return diags


def compute_stable_dt(cfl, tau, C_m, dC_m, V1_dim):

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
        spectral_rho = norm_vv.copy() # approximation
        conv = abs((spectral_rho - old_spectral_rho)/spectral_rho) < 0.00001
        print ("    ... spectral radius iteration: spectral_rho( dC_m @ C_m ) ~= {}".format(spectral_rho))
    t_stamp = time_count(t_stamp)
    
    norm_op = np.sqrt(spectral_rho)
    c_dt_max = 2./norm_op    
    
    light_c = 1
    Nt_pertau = int(np.ceil(tau/(cfl*c_dt_max/light_c)))
    assert Nt_pertau >= 1 
    dt = tau / Nt_pertau
    
    assert light_c*dt <= cfl * c_dt_max
    
    print("  Time step dt computed for Maxwell solver:")
    print("     Since cfl = " + repr(cfl)+",   we set dt = "+repr(dt)+"  --  and Nt_pertau = "+repr(Nt_pertau))
    print("     -- note that c*Dt = "+repr(light_c * dt)+", and c_dt_max = "+repr(c_dt_max)+" thus c * dt / c_dt_max = "+repr(light_c*dt/c_dt_max))
    print("     -- and spectral_radius((c*dt)**2* dC_m @ C_m ) = ",  (light_c * dt * norm_op)**2, " (should be < 4).")

    return Nt_pertau, dt, norm_op

if __name__ == '__main__':
    # quick run, to test 
    plot_dir = "./02_cond_test"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    ###
    alpha = 1.25
    kappa = 1.5
    epsilon = 0.3
    k_theta = 6
    omega = 4
    nb_tau = 2000 

    r_max = 1
    r_min = 0.25 
    ###

    degree = [3,3]
    ncells  = [12,12]
    nbp_arr = [[2, 4]]

    mom_pres = True 
    C1_proj_opt = None

    #domain = build_multipatch_domain('annulus_3', r_min, r_max)
    domain = build_multipatch_annulus(nbp_arr[0][0], nbp_arr[0][1])
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = [m.get_callable_mapping() for m in mappings.values()]
    grid_type=[np.linspace(-1,1,nc+1) for nc in ncells]

    domain_h = discretize(domain, ncells=ncells)  

    p_derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    p_derham_h = discretize(p_derham, domain_h, degree=degree)

    p_V0h = p_derham_h.V0
    p_V1h = p_derham_h.V1
    p_V2h = p_derham_h.V2


    d_derham  = Derham(domain, ["H1", "Hdiv", "L2"])
    dual_degree = [d-1 for d in degree]
    d_derham_h = discretize(d_derham, domain_h, degree=dual_degree)

    d_V0h = d_derham_h.V0
    d_V1h = d_derham_h.V1
    d_V2h = d_derham_h.V2

    p_geomP0, p_geomP1, p_geomP2 = p_derham_h.projectors()
    d_geomP0, d_geomP1, d_geomP2 = d_derham_h.projectors()

    p_PP0, p_PP1, p_PP2 = conf_projectors_scipy(p_derham_h, reg=1, mom_pres=mom_pres, C1_proj_opt=C1_proj_opt, hom_bc=True)
    d_PP0, d_PP1, d_PP2 = conf_projectors_scipy(d_derham_h, reg=0, mom_pres=mom_pres, C1_proj_opt=C1_proj_opt, hom_bc=False)
    

    p_HOp0    = HodgeOperator(p_V0h, domain_h)
    p_MM0     = p_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
    p_MM0_inv = p_HOp0.to_sparse_matrix()                # inverse mass matrix

    p_HOp1   = HodgeOperator(p_V1h, domain_h)
    p_MM1     = p_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
    p_MM1_inv = p_HOp1.to_sparse_matrix()                # inverse mass matrix

    p_HOp2    = HodgeOperator(p_V2h, domain_h)
    p_MM2     = p_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix
    p_MM2_inv = p_HOp2.to_sparse_matrix()                # inverse mass matrix

    d_HOp0   = HodgeOperator(d_V0h, domain_h)
    d_MM0     = d_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
    d_MM0_inv = d_HOp0.to_sparse_matrix()                # inverse mass matrix

    d_HOp1   = HodgeOperator(d_V1h, domain_h)
    d_MM1     = d_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
    d_MM1_inv = d_HOp1.to_sparse_matrix()                # inverse mass matrix

    d_HOp2   = HodgeOperator(d_V2h, domain_h)
    d_MM2    = d_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix

    
    p_KK2 = construct_pairing_matrix(d_V0h,p_V2h,domain_h).tocsr()  # matrix in scipy format
    #p_KK2 = d_PP0@p_KK2_b@p_PP2

    d_KK1 = construct_pairing_matrix(p_V1h,d_V1h,domain_h).tocsr()  # matrix in scipy format
    #d_KK1 = p_PP1@d_KK1_b@d_PP1
    
    p_KK1 = construct_pairing_matrix(d_V1h,p_V1h,domain_h).tocsr()  # matrix in scipy format
    #p_KK1 = d_PP1@p_KK1_b@p_PP1

    x,y = domain.coordinates

    nb = sqrt(alpha**2 * x**2 + y**2 * 1/alpha**2)
    b = Tuple(-y/alpha * 1/nb, alpha * x * 1/nb)

#     t = 0#3/7 * np.pi/2
#     E_ex, B_ex, D_ex, J_ex = get_polarized_annulus_potential_solution(b, omega, k_theta, epsilon, kappa, t=t, r_min = r_min, r_max = r_max, domain=domain)
#    # from psydac.feec.multipatch.examples.ppc_test_cases import get_polarized_solution
#    # E_ex, B_ex, D_ex, J_ex = get_polarized_solution(1, 1, 1, kappa, t=t, domain=domain)

#     Eex_c = P_phys_hcurl(E_ex, p_geomP1, domain, mappings_list).coeffs.toarray()
#     Bex_c = P_phys_l2(B_ex, p_geomP2, domain, mappings_list).coeffs.toarray()

#     Dex_c = P_phys_hdiv(D_ex, d_geomP1, domain, mappings_list).coeffs.toarray()
#     Jex_c = P_phys_hdiv(J_ex, d_geomP1, domain, mappings_list).coeffs.toarray()
#     Hex_c = P_phys_h1(B_ex, d_geomP0, domain, mappings_list).coeffs.toarray()

#     print("degree {}, number of cells {}, kappa {}, alpha {}, time {}".format(degree, ncells, kappa, alpha, t))
#     print("epsilon approxiation: ")
#     u, v = elements_of(p_derham.V1, names='u, v')
#     bE = dot(v, b)
#     Eu = dot(u, v)
#     ub = dot(u, b)
#     mass = BilinearForm((u,v), integral(domain, ((1+kappa)*Eu - kappa * bE * ub)))
#     massh = discretize(mass, domain_h, [p_V1h, p_V1h])
#     M = massh.assemble().tosparse().toarray()
#     print("condition number M_eps: {}".format(np.linalg.cond(M)))
#     print("condition number p_MM1: {}".format(np.linalg.cond(p_MM1.toarray())))
    
#     Coup_Op_eps = np.linalg.inv(M) @ p_PP1.transpose() @ d_KK1 
#     eps_err_c = Eex_c - Coup_Op_eps @ Dex_c
#     eps_L2_error = np.sqrt(np.dot(eps_err_c, p_MM1.dot(eps_err_c)))
#     print("L2-error : {}".format(eps_L2_error))


#     print("epsilon inverse approxiation: ")
#     u, v = elements_of(d_derham.V1, names='u, v')
#     bE = dot(v, b)
#     Eu = dot(u, v)
#     ub = dot(u, b)
#     mass = BilinearForm((u,v), integral(domain, (Eu + kappa * bE * ub)/(1+kappa) ))
#     massh = discretize(mass, domain_h, [d_V1h, d_V1h])
#     M = massh.assemble().tosparse().toarray()
#     print("condition number M_eps_inv: {}".format(np.linalg.cond(M)))
#     print("condition number d_MM1: {}".format(np.linalg.cond(d_MM1.toarray())))

#     Coup_Op_eps_inv = p_MM1_inv @ p_PP1.transpose() @ d_KK1 @ d_MM1_inv @ M
#     eps_inv_err_c = Eex_c - Coup_Op_eps_inv @ Dex_c
#     eps_inv_L2_error = np.sqrt(np.dot(eps_inv_err_c, p_MM1.dot(eps_inv_err_c)))
#     print("L2-error : {}".format(eps_inv_L2_error))

    # inverse hodges like in the paper by b&v
    # print("alternative approxiation: ")
    # u, v = elements_of(d_derham.V1, names='u, v')
    # bE = dot(v, b)
    # Eu = dot(u, v)
    # ub = dot(u, b)
    # mass = BilinearForm((u,v), integral(domain, (Eu + kappa * bE * ub)/(1+kappa) ))
    # massh = discretize(mass, domain_h, [d_V1h, d_V1h])
    # M = massh.assemble().tosparse().toarray()
    # print("condition number M_eps_inv: {}".format(np.linalg.cond(M)))
    # print("condition number d_MM1: {}".format(np.linalg.cond(d_MM1.toarray())))

    # Coup_Op_eps_inv = np.linalg.inv(M) @ d_PP1.transpose() @ p_KK1 
    # eps_inv_err_c = Dex_c - Coup_Op_eps_inv @ Eex_c
    # eps_inv_L2_error = np.sqrt(np.dot(eps_inv_err_c, d_MM1.dot(eps_inv_err_c)))
    # print("L2-error : {}".format(eps_inv_L2_error))

    # print("2 alternative approxiation: ")
    # u, v = elements_of(p_derham.V1, names='u, v')
    # bE = dot(v, b)
    # Eu = dot(u, v)
    # ub = dot(u, b)
    # mass = BilinearForm((u,v), integral(domain, ((1+kappa)*Eu - kappa * bE * ub)))
    # massh = discretize(mass, domain_h, [p_V1h, p_V1h])
    # M = massh.assemble().tosparse().toarray()
    # print("condition number M_eps: {}".format(np.linalg.cond(M)))
    # print("condition number p_MM1: {}".format(np.linalg.cond(p_MM1.toarray())))
    
    # Coup_Op_eps = d_MM1_inv @ d_PP1.transpose() @ p_KK1 @ p_MM1_inv @ M
    # eps_err_c = Dex_c - Coup_Op_eps @ Eex_c
    # eps_L2_error = np.sqrt(np.dot(eps_err_c, d_MM1.dot(eps_err_c)))
    # print("L2-error : {}".format(eps_L2_error))


    T = 10 *  np.pi
    Nt = 400

    E_eng = np.zeros(Nt+1)
    H_eng = np.zeros(Nt+1)
    J_eng = np.zeros(Nt+1)

    for tt in range(Nt+1):
        t =  T/Nt * tt
        E_ex, B_ex, D_ex, J_ex = get_polarized_annulus_potential_solution(b, omega, k_theta, epsilon, kappa, t=t, r_min = r_min, r_max = r_max, domain=domain)

        Eex_c = P_phys_hcurl(E_ex, p_geomP1, domain, mappings_list).coeffs.toarray()
        Bex_c = P_phys_l2(B_ex, p_geomP2, domain, mappings_list).coeffs.toarray()

        Dex_c = P_phys_hdiv(D_ex, d_geomP1, domain, mappings_list).coeffs.toarray()
        Jex_c = P_phys_hdiv(J_ex, d_geomP1, domain, mappings_list).coeffs.toarray()
        Hex_c = P_phys_h1(B_ex, d_geomP0, domain, mappings_list).coeffs.toarray()


        E_eng[tt] = np.dot(Eex_c, d_KK1.dot(Dex_c))
        H_eng[tt] = np.dot(Hex_c, p_KK2.dot(Bex_c))
        J_eng[tt] = np.dot(Jex_c, Jex_c)
        

    fig, ax = plt.subplots()

    E_eng = .5*E_eng[:]
    B_eng = .5*H_eng[:]
    J_eng = .5*J_eng[:]

    td = [T/Nt * tt for tt in range(Nt+1)]
    ax.plot(td, E_eng, '-', ms=7, mfc='None', c='k', label=r'$\frac{1}{2}||E||^2$') #, zorder=10)
    ax.plot(td, B_eng, '-', ms=7, mfc='None', c='g', label=r'$\frac{1}{2}||B||^2$') #, zorder=10)
    #ax.plot(td, J_eng, '-', ms=7, mfc='None', c='r', label=r'$\frac{1}{2}||J||^2$') #, zorder=10)
    ax.plot(td, E_eng+B_eng, '-', ms=7, mfc='None', c='b', label=r'$\frac{1}{2}(||E||^2+||B||^2)$') #, zorder=10)

    ax.legend(loc='best')

    ax.set_xlabel('time t', fontsize=16)                    
    ax.set_title('energy', fontsize=18)
    fig.tight_layout()
    fig.savefig(plot_dir+'/energy_exact.png')


    ## PLOTTING INITIAL CONDITION
    # p_OM1 = OutputManager(plot_dir+'/p_spaces1.yml', plot_dir+'/p_fields1.h5')
    # p_OM1.add_spaces(p_V1h=p_V1h)
    # p_OM1.export_space_info()
    
    # p_OM2 = OutputManager(plot_dir+'/p_spaces2.yml', plot_dir+'/p_fields2.h5')
    # p_OM2.add_spaces(p_V2h=p_V2h)
    # p_OM2.export_space_info()

    # d_OM1 = OutputManager(plot_dir+'/d_spaces1.yml', plot_dir+'/d_fields1.h5')
    # d_OM1.add_spaces(d_V1h=d_V1h)
    # d_OM1.export_space_info()
    
    # d_OM0 = OutputManager(plot_dir+'/d_spaces0.yml', plot_dir+'/d_fields0.h5')
    # d_OM0.add_spaces(d_V0h=d_V0h)
    # d_OM0.export_space_info()

    # period_time = 2*np.pi/omega
    # tau = 0.01 * period_time

    # nb_tau = 100  # final time: T = nb_tau * tau
    # T = nb_tau * tau
    # Nt = 1
    # dt = T/Nt
    # print("final Time: {}".format(T))
    # for tt in range(Nt+1):
    #     E_ex, B_ex, D_ex, J_ex = get_polarized_annulus_potential_solution(b, omega, k_theta, epsilon, kappa, t=tt * dt, r_min = r_min, r_max = r_max, domain=domain)
        
    #     Eex_h = P_phys_hcurl(E_ex, p_geomP1, domain, mappings_list)
    #     Bex_h = P_phys_l2(B_ex, p_geomP2, domain, mappings_list)

    #     Dex_h = P_phys_hdiv(D_ex, d_geomP1, domain, mappings_list)
    #     Jex_h = P_phys_hdiv(J_ex, d_geomP1, domain, mappings_list)
    #     Hex_h = P_phys_h1(B_ex, d_geomP0, domain, mappings_list)

    #     p_OM1.add_snapshot(t=tt * dt, ts=tt) 
    #     p_OM1.export_fields(Eh=Eex_h)

    #     p_OM2.add_snapshot(t=tt * dt, ts=tt) 
    #     p_OM2.export_fields(Bh=Bex_h)

    #     d_OM1.add_snapshot(t=tt * dt, ts=tt) 
    #     d_OM1.export_fields(Dh=Dex_h)
    #     d_OM1.export_fields(Jh=Jex_h)

    #     d_OM0.add_snapshot(t=tt * dt, ts=tt) 
    #     d_OM0.export_fields(Hh=Hex_h)
    
    # p_OM1.close()
    # p_OM2.close() 
    # d_OM1.close()
    # d_OM0.close() 

    # n_p_c = 3

    # PM = PostProcessManager(domain=domain, space_file=plot_dir+'/p_spaces1.yml', fields_file=plot_dir+'/p_fields1.h5' )
    # PM.export_to_vtk(plot_dir+"/Eh",grid=None, npts_per_cell=n_p_c,snapshots='all', fields = 'Eh' )
    # PM.close()

    # PM = PostProcessManager(domain=domain, space_file=plot_dir+'/p_spaces2.yml', fields_file=plot_dir+'/p_fields2.h5' )
    # PM.export_to_vtk(plot_dir+"/Bh",grid=None, npts_per_cell=n_p_c,snapshots='all', fields = 'Bh' )
    # PM.close()

    # PM = PostProcessManager(domain=domain, space_file=plot_dir+'/d_spaces1.yml', fields_file=plot_dir+'/d_fields1.h5' )
    # PM.export_to_vtk(plot_dir+"/Dh",grid=None, npts_per_cell=n_p_c,snapshots='all', fields = 'Dh' )
    # PM.export_to_vtk(plot_dir+"/Jh",grid=None, npts_per_cell=n_p_c,snapshots='all', fields = 'Jh' )
    # PM.close()

    # PM = PostProcessManager(domain=domain, space_file=plot_dir+'/d_spaces0.yml', fields_file=plot_dir+'/d_fields0.h5' )
    # PM.export_to_vtk(plot_dir+"/Hh",grid=None, npts_per_cell=n_p_c,snapshots='all', fields = 'Hh' )
    # PM.close()
    #################################################


