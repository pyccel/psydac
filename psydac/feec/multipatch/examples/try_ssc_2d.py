
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

# from psydac.api.discretization       import discretize

from psydac.feec.multipatch.api                         import discretize
from psydac.api.settings                                import PSYDAC_BACKENDS
from psydac.feec.multipatch.fem_linear_operators        import IdLinearOperator
from psydac.feec.multipatch.operators                   import HodgeOperator
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain, build_multipatch_rectangle
from psydac.feec.multipatch.plotting_utilities_2          import plot_field
from psydac.feec.multipatch.utilities                   import time_count #, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
# from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file
from psydac.feec.pull_push import pull_2d_h1, pull_2d_hcurl, pull_2d_hdiv, pull_2d_l2
from psydac.feec.multipatch.utils_conga_2d              import P_phys_l2, P_phys_hdiv, P_phys_hcurl, P_phys_h1

from psydac.feec.multipatch.bilinear_form_scipy import construct_pairing_matrix
import psydac.feec.multipatch.conf_projections_scipy as cps
# from psydac.feec.multipatch.conf_projections_scipy import Conf_proj_0, Conf_proj_1, Conf_proj_0_c1, Conf_proj_1_c1
from sympde.topology      import Square    
from sympde.topology      import IdentityMapping, PolarMapping
from psydac.fem.vector import ProductFemSpace

from scipy.sparse import save_npz, load_npz
from scipy.sparse.linalg import spilu, lgmres, norm as sp_norm
from scipy.sparse.linalg import LinearOperator, eigsh, minres
from scipy.sparse          import csr_matrix
from scipy.linalg        import norm

from psydac.feec.multipatch.examples.fs_domains_examples import create_square_domain
from psydac.feec.multipatch.utils_conga_2d              import write_errors_array_deg_nbp

cps.mom_pres =  True # False # 
cps.gamma = 1

def try_ssc_2d(
        ncells=None, 
        p_degree=[3,3],
        nb_patch_x=2,
        nb_patch_y=2, 
        domain_name='refined_square', 
        plot_dir='./plots/', 
        test_case=None,
        m_load_dir=None,
        backend_language='pyccel-gcc',
        make_plots=True,
        hide_plots=False, 
        ):

    """
    Testing the Strong-Strong Conga (SSC) sequence:
    with two strong broken DeRham sequences (a primal Hermite with hom BC and a dual Lagrange) and pairing matrices
    """

    if cps.mom_pres:
        cps_opts = 'mom_pres_g={}/'.format(cps.gamma)
    else:
        cps_opts = 'nomom_pres_g={}/'.format(cps.gamma)
    plot_dir += cps_opts

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting try_ssc_2d function with: ')
    print(' ncells      = {}'.format(ncells))
    print(' p_degree    = {}'.format(p_degree))
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
    
    t_stamp = time_count(t_stamp)
    print('building symbolic and discrete domain...')

    if domain_name == 'multipatch_rectangle':
        domain, domain_h, bnds = build_multipatch_rectangle(
            nb_patch_x, nb_patch_y, 
            x_min=0, x_max=np.pi,
            y_min=0, y_max=np.pi,
            perio=[False,False],
            ncells=ncells,
            )

    elif domain_name == 'refined_square' or domain_name =='square_L_shape':
        int_x, int_y = [[0, np.pi],[0, np.pi]]
        # need to change ncells I guess... 
        domain = create_square_domain(ncells, int_x, int_y, mapping='identity')
        ncells_h = {patch.name: [ncells[int(patch.name[2])][int(patch.name[4])], ncells[int(patch.name[2])][int(patch.name[4])]] for patch in domain.interior}
        domain_h = discretize(domain, ncells=ncells_h)
    else:
        domain = build_multipatch_domain(domain_name=domain_name)
        domain_h = discretize(domain, ncells=ncells)

        # ValueError("Domain not defined.")
    multipatch = nb_patch_x > 1 or nb_patch_y > 1
    if multipatch:
        mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
        # mappings_list = list(mappings.values())
        mappings_list = [m.get_callable_mapping() for m in mappings.values()]
    else:
        mappings = OrderedDict([(domain.interior.logical_domain, domain.interior.mapping)])
        mappings_list = [m.get_callable_mapping() for m in mappings.values()]

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


    if multipatch:
        t_stamp = time_count(t_stamp)
        print('Pairing matrices...')

        p_KK2_storage_fn = m_load_dir+'/p_KK2.npz'
        if os.path.exists(p_KK2_storage_fn):
            # matrix is stored
            print('loading pairing matrix found in '+p_KK2_storage_fn)
            p_KK2 = load_npz(p_KK2_storage_fn)
        else:
            print('pairing matrix not found, computing... ')
            p_KK2 = construct_pairing_matrix(d_V0h,p_V2h).tocsr()  # matrix in scipy format
            t_stamp = time_count(t_stamp)
            print('storing pairing matrix in '+p_KK2_storage_fn)
            save_npz(p_KK2_storage_fn, p_KK2)

        d_KK1_storage_fn = m_load_dir+'/d_KK1.npz'
        if os.path.exists(d_KK1_storage_fn):
            # matrix is stored
            d_KK1 = load_npz(d_KK1_storage_fn)
        else:
            d_KK1 = construct_pairing_matrix(p_V1h,d_V1h).tocsr()  # matrix in scipy format
            save_npz(d_KK1_storage_fn, d_KK1)

        # p_KK1     = construct_pairing_matrix(d_V1h,p_V1h).tocsr()  # matrix in scipy format
        # p_KK2     = construct_pairing_matrix(d_V0h,p_V2h).tocsr()  # matrix in scipy format
        p_KK1 = d_KK1.transpose()

        t_stamp = time_count(t_stamp)
        print('Conforming projections...')
        p_PP0     = cps.Conf_proj_0_c1(p_V0h, nquads = [4*(d + 1) for d in p_degree], hom_bc=False)
        p_PP1     = cps.Conf_proj_1_c1(p_V1h, nquads = [4*(d + 1) for d in p_degree], hom_bc=False)

        p_PP1_C0     = cps.Conf_proj_1(p_V1h, nquads = [4*(d + 1) for d in p_degree])  # to compare

        d_PP0     = cps.Conf_proj_0(d_V0h, nquads = [4*(d + 1) for d in d_degree])
        d_PP1     = cps.Conf_proj_1(d_V1h, nquads = [4*(d + 1) for d in d_degree])

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

        d_HH1     = p_MM1_inv @ p_PP1.transpose() @ d_KK1
        
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

    else:
        assert test_case == 'norm_Lambda0'
        # p_HOp0    = HodgeOperator(p_V0h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=0)
        # p_MM0     = p_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
        
        Vh = p_V0h
        V = Vh.symbolic_space
        u, v = elements_of(V, names='u, v')
        expr   = u*v
        #     expr   = dot(u,v)
        a = BilinearForm((u,v), integral(domain, expr))
        ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[backend_language])
        p_MM0 = ah.assemble().tosparse()  # Mass matrix in stencil > scipy format
        
    
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    t_stamp = time_count(t_stamp)
        
    # some target function
    x,y    = domain.coordinates
    f_smooth  = Tuple(sin(pi*y),
                    sin(pi*x))
    f_aff  = Tuple(y,
                    0*x)

    # f_x = lambdify(domain.coordinates, f_vect[0])
    # f_y = lambdify(domain.coordinates, f_vect[1])
    nb_patches = len(domain)
    G_sol_log = [[lambda xi1, xi2, ii=i : (ii+1)%2 for d in [0,1]] for i in range(nb_patches)]

    g_symb = sin(pi*x)*sin(pi*y)
    g = lambdify(domain.coordinates, g_symb)

    p_geomP0, p_geomP1, p_geomP2 = p_derham_h.projectors()
    d_geomP0, d_geomP1, d_geomP2 = d_derham_h.projectors()

    sol = 'smooth'

    if test_case == 'p_PP1':

        if sol == "discontinuous":
            G_pV1_c = p_geomP1(G_sol_log).coeffs.toarray()
        elif sol == "affine":
            tilde_G_c = p_derham_h.get_dual_dofs(space='V1', f=f_aff, backend_language=backend_language, return_format='numpy_array')
            G_pV1_c = p_MM1_inv @ tilde_G_c
            # = P_phys_hcurl(f_aff, p_geomP1, domain, mappings_list).coeffs.toarray()
        elif sol == "smooth":
            tilde_G_c = p_derham_h.get_dual_dofs(space='V1', f=f_smooth, backend_language=backend_language, return_format='numpy_array')
            G_pV1_c = p_MM1_inv @ tilde_G_c
            # G_pV1_c = P_phys_hcurl(f_smooth, p_geomP1, domain, mappings_list).coeffs.toarray()

        else:
            raise NotImplementedError
        
        G_pP1_c = p_PP1 @ G_pV1_c

        ref_c = G_pV1_c
        app_c = G_pP1_c
        label_ref   = 'G_pV1'
        label_app   = 'p_PP1 @ G_pV1'
        MM      = p_MM1
        Vh      = p_V1h
        Vh_kind = 'hcurl'
        plot_type = 'components'

    elif test_case == 'd_PP1':

        if sol == "discontinuous":
            G_dV1_c = d_geomP1(G_sol_log).coeffs.toarray()
        elif sol == "affine":
            tilde_G_c = d_derham_h.get_dual_dofs(space='V1', f=f_aff, backend_language=backend_language, return_format='numpy_array')
            G_dV1_c = d_MM1_inv @ tilde_G_c
            # = P_phys_hcurl(f_aff, p_geomP1, domain, mappings_list).coeffs.toarray()
        elif sol == "smooth":
            tilde_G_c = d_derham_h.get_dual_dofs(space='V1', f=f_smooth, backend_language=backend_language, return_format='numpy_array')
            G_dV1_c = d_MM1_inv @ tilde_G_c
            # G_dV1_c = P_phys_hcurl(f_smooth, p_geomP1, domain, mappings_list).coeffs.toarray()

        else:
            raise NotImplementedError
        
        G_dP1_c = d_PP1 @ G_dV1_c

        ref_c = G_dV1_c
        app_c = G_dP1_c
        label_ref   = 'G_dV1'
        label_app   = 'd_PP1 @ G_dV1'
        MM      = d_MM1
        Vh      = d_V1h
        Vh_kind = 'hdiv'
        plot_type = 'components'        

    elif test_case == 'p_PP1_C1':

        G_pV1_c = p_geomP1(G_sol_log).coeffs.toarray()
        dP_G_c  = p_bC @ p_PP1 @ G_pV1_c

        # dG_pP2_c  = p_bC @ G_pV1_c  

        ref_c = np.zeros(p_V2h.nbasis)
        app_c = dP_G_c
        label_ref   = 'noref'
        label_app   = 'p_bC @ p_PP1 @ G_pV1_c'
        MM      = p_MM2
        Vh      = p_V2h
        Vh_kind = 'l2'
        plot_type = 'amplitude'

    elif test_case == 'p_HH2':
        
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

    elif test_case == 'd_HH1':

        if sol == "discontinuous":
            G_pV1_c = p_geomP1(G_sol_log).coeffs.toarray()
            G_dV1_c = d_geomP1(G_sol_log).coeffs.toarray()
        elif sol == "affine":
            p_tilde_G_c = p_derham_h.get_dual_dofs(space='V1', f=f_aff, backend_language=backend_language, return_format='numpy_array')
            G_pV1_c = p_MM1_inv @ p_tilde_G_c

            d_tilde_G_c = d_derham_h.get_dual_dofs(space='V1', f=f_aff, backend_language=backend_language, return_format='numpy_array')
            G_dV1_c = d_MM1_inv @ d_tilde_G_c
            # = P_phys_hcurl(f_aff, p_geomP1, domain, mappings_list).coeffs.toarray()
        
        elif sol == "smooth":
            p_tilde_G_c = p_derham_h.get_dual_dofs(space='V1', f=f_smooth, backend_language=backend_language, return_format='numpy_array')
            G_pV1_c = p_MM1_inv @ p_tilde_G_c

            d_tilde_G_c = d_derham_h.get_dual_dofs(space='V1', f=f_smooth, backend_language=backend_language, return_format='numpy_array')
            G_dV1_c = d_MM1_inv @ d_tilde_G_c

        else:
            raise NotImplementedError

        # f_pV1   = P_phys_hcurl(f_symb, p_geomP1, domain, mappings_list)
        # f_pV1_c = f_pV1.coeffs.toarray()

        # f_dV1   = P_phys_hdiv(f_symb, d_geomP1, domain, mappings_list)
        # f_dV1_c = f_dV1.coeffs.toarray()

        ref_c   = G_pV1_c
        app_c   = d_HH1 @ G_dV1_c

        label_ref   = 'f_pV1'
        label_app   = 'd_HH1 @ f_dV1'
        MM      = p_MM1
        Vh      = p_V1h 
        Vh_kind = 'hcurl'
        plot_type = 'components'

    elif test_case == "pP1_L2":

        if sol == "discontinuous":
            G_pV1_c = p_geomP1(G_sol_log).coeffs.toarray()
            G_dV1_c = d_geomP1(G_sol_log).coeffs.toarray()
        elif sol == "affine":
            p_tilde_G_c = p_derham_h.get_dual_dofs(space='V1', f=f_aff, backend_language=backend_language, return_format='numpy_array')
            G_pV1_c = p_MM1_inv @ p_tilde_G_c

            d_tilde_G_c = d_derham_h.get_dual_dofs(space='V1', f=f_aff, backend_language=backend_language, return_format='numpy_array')
            G_dV1_c = d_MM1_inv @ d_tilde_G_c
            # = P_phys_hcurl(f_aff, p_geomP1, domain, mappings_list).coeffs.toarray()
        
        elif sol == "smooth":
            p_tilde_G_c = p_derham_h.get_dual_dofs(space='V1', f=f_smooth, backend_language=backend_language, return_format='numpy_array')
            G_pV1_c = p_MM1_inv @ p_tilde_G_c

            d_tilde_G_c = d_derham_h.get_dual_dofs(space='V1', f=f_smooth, backend_language=backend_language, return_format='numpy_array')
            G_dV1_c = d_MM1_inv @ d_tilde_G_c

        else:
            raise NotImplementedError

        # f_pV1   = P_phys_hcurl(f_symb, p_geomP1, domain, mappings_list)
        # f_pV1_c = f_pV1.coeffs.toarray()

        # f_dV1   = P_phys_hdiv(f_symb, d_geomP1, domain, mappings_list)
        # f_dV1_c = f_dV1.coeffs.toarray()

        # d_HH1 = p_MM1_inv @ p_PP1.transpose() @ d_KK1
        # app_c   = d_HH1 @ G_dV1_c
        app_c   = p_PP1 @ p_MM1_inv @ d_KK1 @ G_dV1_c

        ref_c   = G_pV1_c
        
        label_ref   = 'f_pV1'
        label_app   = 'd_P1 @ P_L2 @ f_dV1'        
        MM      = p_MM1
        Vh      = p_V1h 
        Vh_kind = 'hcurl'
        plot_type = 'components'

    elif test_case == 'p_HH1':

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
    
    elif test_case == 'norm_Lambda0':
        g0 = exp(-(x**2+y**2)*1e20) # 1 at 0 and 0 elsewhere 
        print(type(g0))
        g0h = P_phys_h1(g0, p_geomP0, domain, mappings_list)
        g0_c = g0h.coeffs.toarray()
        
        ref_c = np.zeros(p_V0h.nbasis)
        app_c   = g0_c

        label_ref   = 'noref'
        label_app   = 'Lambda0_0'
        MM      = p_MM0
        Vh      = p_V0h
        Vh_kind = 'h1'
        plot_type = 'components'

    else:
        raise NotImplementedError
    
    if make_plots:
        # plot_field(numpy_coeffs=app_c, Vh=Vh, space_kind=Vh_kind, 
        #         plot_type=plot_type,
        #         domain=domain, title=label_app, cmap='viridis', 
        #         filename=plot_dir+'test='+test_case+'_app_.png', 
        #         hide_plot=hide_plots, 
        #         N_vis=20,
        #         #    eta_crop=[[0.2,0.3], [0,10]],
        #         surface_plot=True)
        
        # plot_field(numpy_coeffs=ref_c, Vh=Vh, space_kind=Vh_kind, 
        #         plot_type=plot_type,
        #         domain=domain, title=label_ref, cmap='viridis', 
        #         filename=plot_dir+'test='+test_case+'_ref_.png', 
        #         hide_plot=hide_plots, 
        #         N_vis=20,
        #         #    eta_crop=[[0.2,0.3], [0,10]],
        #         surface_plot=True)

        label_err  = label_ref + ' - ' + label_app + ' with: ' + cps_opts
        plot_field(numpy_coeffs=app_c-ref_c, Vh=Vh, space_kind=Vh_kind, 
                plot_type=plot_type,
                domain=domain, title=label_err, cmap='viridis', 
                filename=plot_dir+'test='+test_case+'_err_.png', 
                hide_plot=hide_plots,
                cb_min=-0.00015,
                cb_max=0.00015,
                N_vis=20,
                #    eta_crop=[[0.2,0.3], [0,10]],
                surface_plot=True)

    else:
        print( "-- -- -- skipping plots -- -- -- ")
    
    err_c = app_c - ref_c 
    L2_error = np.sqrt(np.dot(err_c, MM @ err_c)/np.dot(ref_c, MM @ ref_c))

    return L2_error    

if __name__ == '__main__':

    t_stamp_full = time_count()

    # test_case = "norm_Lambda0"
    # test_case = "p_PP1" # "d_HH1" # "p_PP1" # "p_PP1_C1" # "d_HH1" # "p_PP1_C1" # 
    # test_case = "pP1_L2" 
    test_case = "p_HH2"
    refined_square = False
    
    make_plots = True #False
    hide_plots = True
    

    nbp_s = [2] #[2,4] #,6,8]
    deg_s = [3]
    
    # nbc_s = [10]
    nbc_s = [2,4,6,8,16,32]
    
    errors = [[[ None for nbc in nbc_s] for nbp in nbp_s] for deg in deg_s]

    for i_deg, deg in enumerate(deg_s): 
        for i_nbp, nbp in enumerate(nbp_s): 
            for i_nbc, nbc in enumerate(nbc_s): 
                nb_patch_x = nbp
                nb_patch_y = nbp     
                if refined_square:
                    domain_name = 'refined_square'
                    raise NotImplementedError("CHECK FIRST")
                    m_load_dir = 'matrices_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
                else:
                    domain_name = 'multipatch_rectangle' #'square_9'
                    m_load_dir = 'matrices_{}_nbp={}_nc={}_deg={}/'.format(domain_name, nbp, nbc, deg)

                run_dir = '{}_nbp={}_nc={}_deg={}/'.format(domain_name, nbp, nbc, deg)

                error = try_ssc_2d(
                    ncells=[nbc,nbc], 
                    nb_patch_x=nb_patch_x,
                    nb_patch_y=nb_patch_y,
                    p_degree=[deg,deg],
                    domain_name=domain_name, 
                    test_case=test_case,
                    plot_dir='./plots/'+run_dir,
                    m_load_dir=m_load_dir,
                    backend_language='python', #'pyccel-gcc'
                    hide_plots=hide_plots,
                    make_plots=make_plots,
                )
                print("-------------------------------------------------")
                print("for deg = {}, nb_patches = {}**2".format(deg,nbp))
                print("error: {}".format(error))
                print("-------------------------------------------------\n")
        
                errors[i_deg][i_nbp][i_nbc] = error
    
    if len(nbc_s) == 1:
        write_errors_array_deg_nbp(errors, deg_s, nbp_s, nbc, error_dir='./errors', name=test_case)
    else:
        print("WARNING: not writing any error file")        

    # if len(nbc_s) == 1:
    #     write_errors_array_deg_nbp(errors, deg_s, nbp_s, nbc, error_dir='./errors', name=test_case)

    time_count(t_stamp_full, msg='full program')