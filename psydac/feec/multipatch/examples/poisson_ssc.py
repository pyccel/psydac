# coding: utf-8

from pytest import param
from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from sympy import lambdify, Matrix, Tuple, pi, sqrt, cos, exp, acos, sin, arg, I 
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
from psydac.feec.multipatch.examples.ppc_test_cases     import get_div_free_pulse, get_polarized_annulus_potential_solution, get_polarized_annulus_potential_source,get_polarized_annulus_potential_solution_old, get_polarized_annulus_potential_source_old, get_poisson_annulus_solution, get_poisson_solution
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

def solve_poisson_ssc(
        method='ssc',
        nbc=4, deg=4, 
        mom_pres=False, 
        C1_proj_opt=None,
        domain_name='pretzel_f', backend_language=None,
        nb_patch_r=1, nb_patch_theta=4, 
        project_sol=False, filter_source=True, quad_param=1,
        solution_type='zero', solution_proj='P_geom', 
        plot_dir=None, hide_plots=True,
        cb_min_sol=None, cb_max_sol=None,
        m_load_dir="",
):

    

    diags = {}


    ncells = [nbc, nbc]
    degree = [deg,deg]


    if m_load_dir is not None:
        pm_load_dir = m_load_dir+"primal"
        dm_load_dir = m_load_dir+"dual"
        for load_dir in [m_load_dir, pm_load_dir, dm_load_dir]:        
            if not os.path.exists(load_dir):
                os.makedirs(load_dir)


    t_stamp = time_count()
    print(' .. multi-patch domain...')

    if domain_name == "square":
        x_min = 0
        x_max = pi
        
        y_min = 0
        y_max = pi
        domain = build_multipatch_rectangle(nb_patch_x = nb_patch_x, nb_patch_y = nb_patch_y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, perio=[False,False], comm=None, F_name='Identity')
    elif domain == "annulus":
        r_min = 1
        r_max = 2
        domain = build_multipatch_annulus(nb_patch_x, nb_patch_y, x_min, x_max)
        
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = [m.get_callable_mapping() for m in mappings.values()]
    grid_type=[np.linspace(-1,1,nc+1) for nc in ncells]



    domain_h = discretize(domain, ncells=ncells)  # todo: remove this ?

    t_stamp = time_count(t_stamp)
    print('building (primal) derham sequence...')
    p_derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    p_derham_h = discretize(p_derham, domain_h, degree=degree, grid_type=grid_type)

    p_V0h = p_derham_h.V0
    p_V1h = p_derham_h.V1
    p_V2h = p_derham_h.V2

    t_stamp = time_count(t_stamp)
    print('building dual derham sequence...')
    d_derham  = Derham(domain, ["H1", "Hdiv", "L2"])
    dual_degree = [d-1 for d in degree]
    d_derham_h = discretize(d_derham, domain_h, degree=dual_degree, grid_type=grid_type)

    d_V0h = d_derham_h.V0
    d_V1h = d_derham_h.V1
    d_V2h = d_derham_h.V2

    t_stamp = time_count(t_stamp)       
    print('building the conforming projection matrices in primal spaces ...')
    p_PP0, p_PP1, p_PP2 = conf_projectors_scipy(p_derham_h, reg=1, mom_pres=mom_pres, C1_proj_opt=C1_proj_opt, hom_bc=True)
    t_stamp = time_count(t_stamp)       
    print('building the conforming projection matrices in dual spaces ...')
    d_PP0, d_PP1, d_PP2 = conf_projectors_scipy(d_derham_h, reg=0, mom_pres=mom_pres, C1_proj_opt=C1_proj_opt, hom_bc=False)


    
    p_HOp0    = HodgeOperator(p_V0h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=0)
    p_MM0     = p_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
    p_MM0_inv = p_HOp0.to_sparse_matrix()                # inverse mass matrix

    p_HOp1   = HodgeOperator(p_V1h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=1)
    p_MM1     = p_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
    p_MM1_inv = p_HOp1.to_sparse_matrix()                # inverse mass matrix

    p_HOp2    = HodgeOperator(p_V2h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=2)
    p_MM2     = p_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix
    p_MM2_inv = p_HOp2.to_sparse_matrix()                # inverse mass matrix


    d_HOp0   = HodgeOperator(d_V0h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=0)
    d_MM0     = d_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
    d_MM0_inv = d_HOp0.to_sparse_matrix()                # inverse mass matrix

    d_HOp1   = HodgeOperator(d_V1h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=1)
    d_MM1     = d_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
    d_MM1_inv = d_HOp1.to_sparse_matrix()                # inverse mass matrix

    d_HOp2   = HodgeOperator(d_V2h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=2)
    d_MM2    = d_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix
    d_MM2_inv = d_HOp2.to_sparse_matrix()                # inverse mass matrix

    
    t_stamp = time_count(t_stamp)
    print('building the Hodge matrices ...')


    p_KK1_storage_fn = m_load_dir+'/p_KK1.npz'
    if os.path.exists(p_KK1_storage_fn):
        # matrix is stored
        p_KK1 = load_npz(p_KK1_storage_fn)
    else:
        p_KK1 = construct_pairing_matrix(d_V1h,p_V1h).tocsr()  # matrix in scipy format
        save_npz(p_KK1_storage_fn, p_KK1)

    p_KK0_storage_fn = m_load_dir+'/p_KK0.npz'
    if os.path.exists(p_KK0_storage_fn):
        # matrix is stored
        p_KK0 = load_npz(p_KK0_storage_fn)
    else:
        p_KK0 = construct_pairing_matrix(d_V2h,p_V0h).tocsr()  # matrix in scipy format
        save_npz(p_KK0_storage_fn, p_KK0)

    d_KK2_storage_fn = m_load_dir+'/d_KK2.npz'
    if os.path.exists(d_KK2_storage_fn):
        # matrix is stored
        d_KK2 = load_npz(d_KK2_storage_fn)
    else:
        d_KK2 = construct_pairing_matrix(p_V0h,d_V2h).tocsr()  # matrix in scipy format
        save_npz(d_KK2_storage_fn, d_KK2)
    
    p_HH0 = d_MM2_inv @ d_PP2.transpose() @ p_KK0
    d_HH2 = p_MM0_inv @ p_PP0.transpose() @ d_KK2

    p_I1 = sparse_eye(p_V1h.nbasis)

    p_bD0, p_bD1 = p_derham_h.broken_derivatives_as_operators
    p_bG = p_bD0.to_sparse_matrix() # broken grad (primal)
    p_GG = p_bG @ p_PP0             # Conga grad (primal)
    p_bC = p_bD1.to_sparse_matrix() # broken curl (primal: scalar-valued)
    p_CC = p_bC @ p_PP1             # Conga curl (primal)

    d_bD0, d_bD1 = d_derham_h.broken_derivatives_as_operators
    d_bC = d_bD0.to_sparse_matrix() # broken curl (dual: vector-valued)
    d_CC = d_bC @ d_PP0             # Conga curl (dual)    
    d_bD = d_bD1.to_sparse_matrix() # broken div
    d_DD = d_bD @ d_PP1             # Conga div (dual)    
    
    p_geomP0, p_geomP1, p_geomP2 = p_derham_h.projectors()
    d_geomP0, d_geomP1, d_geomP2 = d_derham_h.projectors()

    print(' .. Ampere and Faraday evolution (curl . Hodge) operators...')

    x,y = domain.coordinates
    gamma = 1 + exp(((x-pi/2)**2 + (y-pi/2)**2)/(20**2))

    hodge_direct = True
    if hodge_direct:
        u, v = elements_of(d_derham.V1, names='u, v')
        mass = BilinearForm((u,v), integral(domain, dot(gamma*u,v)))
        massh = discretize(mass, domain_h, [d_V1h, d_V1h])
        M = massh.assemble().tosparse().toarray()
        p_HH1 = np.linalg.inv(M) @ d_PP1.transpose() @ p_KK1

    else:
        u, v = elements_of(p_derham.V1, names='u, v')
        mass = BilinearForm((u,v), integral(domain, dot(u, v)* 1/gamma ))
        massh = discretize(mass, domain_h, [p_V1h, p_V1h])
        M = massh.assemble().tosparse().toarray()
        p_HH1 = d_MM1_inv @ d_PP1.transpose() @ p_KK1 @ p_MM1_inv @ M

    A_m = -d_HH2 @ d_DD @ p_HH1 @ p_GG 

    I0 = IdLinearOperator(p_V0h).to_sparse_matrix()
    JP = I0 - p_PP0 #jump_penal_m
    A_jp = JP.transpose() @ p_MM0_inv @ JP
    gamma_h = 20

    A = p_PP0.transpose() @ A_m @ p_PP0 + gamma_h * A_jp
    
    if domain_name == "square":
        f, phi = get_poisson_solution(x_min, x_max, y_min, y_max, domain)
    elif domain_name =="annulus":
        f, phi = get_poisson_annulus_solution(x_min, x_max, domain)

    f_h = P_phys_l2(f, d_geomP2, domain, mappings_list)
    f_c = d_HH2 @ f_h.coeffs.toarray()
    plot_field(numpy_coeffs= f_c, Vh=p_V0h, space_kind='h1', plot_type='components', domain=domain, surface_plot=False, title='f', hide_plot=True, filename="f_c.png")


    phi_h = P_phys_h1(phi, p_geomP0, domain, mappings_list)
    phi_c = phi_h.coeffs.toarray()
    plot_field(numpy_coeffs=phi_c, Vh=p_V0h, plot_type='components', space_kind='h1', domain=domain, surface_plot=False, title='phi', hide_plot=True, filename="phi_c.png")

    u_c = spsolve(A, f_c)

    plot_field(numpy_coeffs=u_c, Vh=p_V0h, space_kind='h1', plot_type='components', domain=domain, surface_plot=False, title='u', hide_plot=True, filename="u_c.png")


    err = np.sqrt(np.dot(u_c - phi_c, p_MM0 @ (u_c - phi_c)))
    print(err)

    

if __name__ == '__main__':
    domain_name = "square"
    nb_patch_x = 2
    nb_patch_y = 2
    nbp = nb_patch_x * nb_patch_y
    nbc = 10
    deg = 3
    m_load_dir = 'matrices_{}_nbp={}_nc={}_deg={}/'.format(domain_name, nbp, nbc, deg)
    run_dir = './{}_nbp={}_nc={}_deg={}/'.format(domain_name, nbp, nbc, deg)
    plot_dir= run_dir+'plots/'

    solve_poisson_ssc(
        method='ssc',
        nbc=nbc, deg=deg, 
        mom_pres=True, 
        C1_proj_opt=None,
        domain_name=domain_name, backend_language='python',
        nb_patch_r=nb_patch_x, nb_patch_theta=nb_patch_y, 
        project_sol=False, filter_source=True, quad_param=1,
        solution_type='zero', solution_proj='P_geom', 
        plot_dir=None, hide_plots=True,
        cb_min_sol=None, cb_max_sol=None,
        m_load_dir=m_load_dir,
        )