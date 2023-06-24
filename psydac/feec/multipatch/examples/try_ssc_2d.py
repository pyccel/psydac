
# todo: build and test some basic objects fpr SSC diagram, following the "try_2d_ssc_hermite" in effect

import os
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from sympy import pi, cos, sin, Tuple, exp
from sympy import lambdify

from sympde.topology    import Derham
from sympde.topology    import element_of, elements_of

from sympde.calculus  import dot
from sympde.expr.expr import BilinearForm
from sympde.expr.expr import integral

from psydac.api.discretization       import discretize

from psydac.feec.multipatch.api                         import discretize
from psydac.api.settings                                import PSYDAC_BACKENDS
from psydac.feec.multipatch.fem_linear_operators        import IdLinearOperator
from psydac.feec.multipatch.operators                   import HodgeOperator
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.plotting_utilities_2          import plot_field
from psydac.feec.multipatch.utilities                   import time_count #, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
# from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file
from psydac.feec.pull_push import pull_2d_h1, pull_2d_hcurl, pull_2d_hdiv, pull_2d_l2
from psydac.feec.multipatch.utils_conga_2d              import P_phys_l2, P_phys_hdiv, P_phys_hcurl, P_phys_h1

from psydac.feec.multipatch.bilinear_form_scipy import construct_pairing_matrix
from psydac.feec.multipatch.conf_projections_scipy import Conf_proj_0, Conf_proj_1, Conf_proj_0_c1, Conf_proj_1_c1
from sympde.topology      import Square    
from sympde.topology      import IdentityMapping, PolarMapping
from psydac.fem.vector import ProductFemSpace

from scipy.sparse.linalg import spilu, lgmres, norm as sp_norm
from scipy.sparse.linalg import LinearOperator, eigsh, minres
from scipy.sparse          import csr_matrix
from scipy.linalg        import norm


from psydac.feec.multipatch.examples.fs_domains_examples import create_square_domain



def try_ssc_2d(
        ncells=None, 
        p_degree=[3,3], 
        domain_name='refined_square', 
        plot_dir='./plots/', 
        Htest=None,
        m_load_dir=None,
        backend_language='pyccel-gcc'
        ):

    """
    Testing the Strong-Strong Conga (SSC) sequence:
    with two strong broken DeRham sequences (a primal Hermite with hom BC and a dual Lagrange) and pairing matrices
    """

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting hcurl_solve_eigen_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' p_degree = {}'.format(p_degree))
    print(' domain_name = {}'.format(domain_name))
    # print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')
    t_stamp = time_count()
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if m_load_dir is not None:
        pm_load_dir = m_load_dir+"primal"
        dm_load_dir = m_load_dir+"dual"
        for load_dir in [pm_load_dir, dm_load_dir]:        
            if not os.path.exists(load_dir):
                os.makedirs(load_dir)

    print('building symbolic and discrete domain...')

    if domain_name == 'refined_square' or domain_name =='square_L_shape':
        int_x, int_y = [[0, np.pi],[0, np.pi]]
        # need to change ncells I guess... 
        domain = create_square_domain(ncells, int_x, int_y, mapping='identity')
        ncells_h = {patch.name: [ncells[int(patch.name[2])][int(patch.name[4])], ncells[int(patch.name[2])][int(patch.name[4])]] for patch in domain.interior}
    
    else:
        domain = build_multipatch_domain(domain_name=domain_name)
        ncells_h = ncells   # ?

        # ValueError("Domain not defined.")

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    # mappings_list = list(mappings.values())
    mappings_list = [m.get_callable_mapping() for m in mappings.values()]

    t_stamp = time_count(t_stamp)
    print(' .. discrete domain...')
    domain_h = discretize(domain, ncells=ncells_h)

    print('building symbolic and discrete derham sequences...')
    d_degree = [d-1 for d in p_degree]

    t_stamp = time_count()
    print(' .. Primal derham sequence...')
    p_derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. Primal discrete derham sequence...')
    p_derham_h = discretize(p_derham, domain_h, degree=p_degree) #, backend=PSYDAC_BACKENDS[backend_language])
    # primal is with hom bc, but this is in the conf projections

    t_stamp = time_count()
    print(' .. Dual derham sequence...')
    d_derham  = Derham(domain, ["H1", "Hdiv", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. Dual discrete derham sequence...')
    d_derham_h = discretize(d_derham, domain_h, degree=d_degree) #, backend=PSYDAC_BACKENDS[backend_language])
    
    p_V0h = p_derham_h.V0
    p_V1h = p_derham_h.V1
    p_V2h = p_derham_h.V2

    d_V0h = d_derham_h.V0
    d_V1h = d_derham_h.V1
    d_V2h = d_derham_h.V2

    t_stamp = time_count(t_stamp)
    print('Pairing matrices...')

    p_KK1     = construct_pairing_matrix(d_V1h,p_V1h).tocsr()  # matrix in scipy format
    p_KK2     = construct_pairing_matrix(d_V0h,p_V2h).tocsr()  # matrix in scipy format
    d_KK1     = p_KK1.transpose()

    t_stamp = time_count(t_stamp)
    print('Conforming projections...')
    p_PP0     = Conf_proj_0_c1(p_V0h, nquads = [4*(d + 1) for d in p_degree], hom_bc=False)
    p_PP1     = Conf_proj_1_c1(p_V1h, nquads = [4*(d + 1) for d in p_degree], hom_bc=False)

    p_PP1_C0     = Conf_proj_1(p_V1h, nquads = [4*(d + 1) for d in p_degree])  # to compare

    d_PP0     = Conf_proj_0(d_V0h, nquads = [4*(d + 1) for d in d_degree])
    d_PP1     = Conf_proj_1(d_V1h, nquads = [4*(d + 1) for d in d_degree])

    t_stamp = time_count(t_stamp)
    print('Mass matrices...')
    p_HOp1    = HodgeOperator(p_V1h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=1)
    p_MM1     = p_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
    p_MM1_inv = p_HOp1.to_sparse_matrix()                # inverse mass matrix

    p_HOp2    = HodgeOperator(p_V2h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=2)
    p_MM2     = p_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix
    p_MM2_inv = p_HOp2.to_sparse_matrix()                # inverse mass matrix

    d_HOp0    = HodgeOperator(d_V0h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=0)
    d_MM0     = d_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
    d_MM0_inv = d_HOp0.to_sparse_matrix()                # inverse mass matrix

    d_HOp1    = HodgeOperator(d_V1h, domain_h, backend_language=backend_language, load_dir=dm_load_dir, load_space_index=1)
    d_MM1     = d_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
    d_MM1_inv = d_HOp1.to_sparse_matrix()                # inverse mass matrix

    t_stamp = time_count(t_stamp)
    print('Hodge operators...')
    
    p_HH1     = d_MM1_inv @ d_PP1.transpose() @ p_KK1
    p_HH2     = d_MM0_inv @ d_PP0.transpose() @ p_KK2

    d_HH1     = p_MM1_inv @ p_PP1.transpose() @ d_KK1     #  BAD !!
    
    t_stamp = time_count(t_stamp)
    print('diff operators...')

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

        
    
    
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    t_stamp = time_count(t_stamp)
        
    # some target function
    x,y    = domain.coordinates
    # f_symb  = Tuple(sin(pi*y),
    #                 sin(pi*x))
    f_symb  = Tuple(x*x,
                    0*y)

    # f_x = lambdify(domain.coordinates, f_vect[0])
    # f_y = lambdify(domain.coordinates, f_vect[1])
    nb_patches = len(domain)
    G_sol_log = [[lambda xi1, xi2, ii=i : ii+1 for d in [0,1]] for i in range(nb_patches)]

    g_symb = sin(pi*x)*sin(pi*y)
    g = lambdify(domain.coordinates, g_symb)

    p_geomP0, p_geomP1, p_geomP2 = p_derham_h.projectors()
    d_geomP0, d_geomP1, d_geomP2 = d_derham_h.projectors()


    if Htest == 'p_PP1':

        G_pV1_c = p_geomP1(G_sol_log).coeffs.toarray()
        G_pP1_c = p_PP1 @ G_pV1_c

        ref_c = G_pV1_c
        app_c = G_pP1_c
        label_ref   = 'G_pV1'
        label_app   = 'p_PP1 @ G_pV1'
        MM      = p_MM1
        Vh      = p_V1h
        Vh_kind = 'hcurl'
        plot_type = 'components'

    elif Htest == 'p_PP1_C1':

        G_pV1_c = p_geomP1(G_sol_log).coeffs.toarray()
        dP_G_c  = p_bC @ p_PP1 @ G_pV1_c

        # dG_pP2_c  = p_bC @ G_pV1_c  

        ref_c = np.zeros(p_V2h.nbasis)
        app_c = dP_G_c
        label_ref   = 'noref'
        label_app   = 'p_bC @ G_pP1_c'
        MM      = p_MM2
        Vh      = p_V2h
        Vh_kind = 'l2'
        plot_type = 'amplitude'

    elif Htest == 'p_HH2':
        
        g_dV0   = P_phys_h1(g_symb, d_geomP0, domain, mappings_list)
        g_dV0_c = g_dV0.coeffs.toarray()

        g_pV2   = P_phys_l2(g_symb, p_geomP2, domain, mappings_list)
        g_pV2_c = g_pV2.coeffs.toarray()
        gh_c    = p_HH2 @ g_pV2_c

        ref_c   = g_dV0_c
        app_c   = gh_c
        label_ref   = 'g_dV0'
        label_app   = 'p_HH2 @ g_pV2'
        MM      = d_MM0
        Vh      = d_V0h
        Vh_kind = 'h1'
        plot_type = 'components'

    elif Htest == 'd_HH1':

        f_pV1   = P_phys_hcurl(f_symb, p_geomP1, domain, mappings_list)
        f_pV1_c = f_pV1.coeffs.toarray()

        f_dV1   = P_phys_hdiv(f_symb, d_geomP1, domain, mappings_list)
        f_dV1_c = f_dV1.coeffs.toarray()
        fh_c    = d_HH1 @ f_dV1_c

        ref_c   = f_pV1_c
        app_c   = fh_c
        label_ref   = 'f_pV1'
        label_app   = 'd_HH1 @ f_dV1'
        MM      = p_MM1
        Vh      = p_V1h 
        Vh_kind = 'hdiv'
        plot_type = 'components'


        # print(" -----  approx f in d_V1  ---------")
        # d_f_log = [pull_2d_hdiv([f_x, f_y], F) for F in mappings_list]
        # f_h = d_P1(d_f_log)
        # d_f1 = f_h.coeffs.toarray()
        # print(" -----  Hodge: f -> p_V1  ---------")
        # p_f1     = d_H1  @ d_f1
        # p_f1_PL2 = d_PL2 @ d_f1
        # error_p_f1 = p_f1 - p_f1_PL2
        # L2_error = np.sqrt(np.dot(error_p_f1, p_M1 @ error_p_f1)/np.dot(p_f1, p_M1 @ p_f1))

        # test_label = 'test_d_H1'
        # d_f1_label = 't_f1 = t_P1 @ f'
        # p_f1_label = 'f1 = t_H1 @ t_f1'
        

    elif Htest == 'p_HH1':

        raise NotImplementedError
    
        print(" -----  approx f in p_V1  ---------")
        p_f_log = [pull_2d_hcurl([f_x, f_y], F) for F in mappings_list]
        f_h = p_P1(p_f_log)
        p_f1 = f_h.coeffs.toarray()
        print(" -----  Hodge: f -> d_V1  ---------")
        d_f1     = p_HH1  @ p_f1
        d_f1_PL2 = p_PL2 @ p_f1
        error_d_f1 = d_f1 - d_f1_PL2
        L2_error = np.sqrt(np.dot(error_d_f1, d_M1 @ error_d_f1)/np.dot(d_f1, d_M1 @ d_f1))
        test_label = 'test_p_H1'
        p_f1_label = 'f1 = p_P1 @ f'
        d_f1_label = 't_f1 = H1 @ f1'

    else:
        raise NotImplementedError
    

    plot_field(numpy_coeffs=app_c, Vh=Vh, space_kind=Vh_kind, 
               plot_type=plot_type,
               domain=domain, title=label_app, cmap='viridis', 
               filename=plot_dir+'test='+Htest+'_app_.png', 
               hide_plot=False)
    plot_field(numpy_coeffs=ref_c, Vh=Vh, space_kind=Vh_kind, 
               plot_type=plot_type,
               domain=domain, title=label_ref, cmap='viridis', 
               filename=plot_dir+'test='+Htest+'_ref_.png', 
               hide_plot=False)

    err_c = app_c - ref_c 
    L2_error = np.sqrt(np.dot(err_c, MM @ err_c)/np.dot(ref_c, MM @ ref_c))

    return L2_error    

if __name__ == '__main__':

    t_stamp_full = time_count()

    # Htest = "p_HH2"
    Htest = "p_PP1_C1" # "p_PP1" # "d_HH1"
    refined_square = False
    
    if refined_square:
        domain_name = 'refined_square'
        nc = 10
    else:
        domain_name = 'square_9'
        nc = 8

    deg = 3

    run_dir = '{}_nc={}_deg={}/'.format(domain_name, nc, deg)

    m_load_dir = 'matrices_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
    error = try_ssc_2d(
        ncells=[nc,nc], 
        p_degree=[deg,deg],
        domain_name=domain_name, 
        Htest=Htest,
        plot_dir='./plots/'+run_dir,
        m_load_dir=m_load_dir,
        backend_language='python' #'pyccel-gcc'
    )

    print("error: {}".format(error))
    time_count(t_stamp_full, msg='full program')