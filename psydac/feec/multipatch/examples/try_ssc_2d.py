
# todo: build and test some basic objects fpr SSC diagram, following the "try_2d_ssc_hermite" in effect

import os
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from sympy import pi, cos, sin, Tuple, exp
from sympy import lambdify

from sympde.topology    import Derham, Domain
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
import psydac.feec.multipatch.conf_projections_scipy2 as cps2
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
from psydac.feec.multipatch.utils_conga_2d              import write_diags_deg_nbp, check_file

total_tests = 0
failed_tests = 0

def test_err(app_c, ref_c, label_error, MM=None, tol=1e-12):
    global total_tests, failed_tests

    total_tests += 1
    err_c = app_c - ref_c 
    if MM is None:
        # error is a matrix
        error = sp_norm(err_c)
    else:
        # error is a vector of coefs in a space with mass matrix MM
        error = np.sqrt(np.dot(err_c, MM @ err_c))
    if abs(error) > tol:
        msg = f'[TEST FAILED !! (tol = {tol})]'
        failed_tests += 1
    else:
        msg = f'test: ok (passed with tol = {tol})'
    print(f'.. {msg}  --  {label_error} = {error}')
    
def get_polynomial_function(degree, hom_bc_axes, domain):
    x, y = domain.coordinates            
    if hom_bc_axes[0]:                
        assert degree[0] > 1
        g0_x = x * (x-np.pi) * (x-1.554)**(degree[0]-2)
    else:
        if degree[0] > 1:
            g0_x = (x-0.543)**2 * (x-1.554)**(degree[0]-2)
        else:
            g0_x = (x-1.554)**degree[0]

    if hom_bc_axes[1]:                
        assert degree[1] > 1
        g0_y = y * (y-np.pi) * (y-0.324)**(degree[1]-2)
    else:
        if degree[1] > 1:
            g0_y = (y-1.675)**2 * (y-0.324)**(degree[1]-2)

        else:
            g0_y = (y-0.324)**degree[1]

    return g0_x * g0_y

def try_ssc_2d(
        ncells=None, 
        p_degree=[3,3],
        hom_bc=False,
        mom_pres=False,
        reg=0,
        nb_patch_x=2,
        nb_patch_y=2, 
        domain_name='refined_square', 
        plot_dir='./plots/', 
        test_case=None,
        m_load_dir=None,
        backend_language='pyccel-gcc',
        make_plots=True,
        hide_plots=False,
        cps_opts='',
        ):

    """
    Testing the Strong-Strong Conga (SSC) sequence:
    with two strong broken DeRham sequences (a primal Hermite with hom BC and a dual Lagrange) and pairing matrices
    """

    plot_dir += cps_opts+'/'

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

    if domain_name == 'square':
        OmegaLog = Square('domain', bounds1=(0, np.pi), bounds2=(0, np.pi))
        mapping_1 = IdentityMapping('M1',2)
        domain     = mapping_1(OmegaLog)
        domain_h = discretize(domain, ncells=ncells)

    elif domain_name == 'multipatch_rectangle':
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

    # assert isinstance(domain, Domain)
    # print("len(domain.interfaces) = ", len(domain.interfaces))
    # # print(domain.connectivity)
    # print("len(domain.interior) = ", len(domain.interior))
    # # exit()

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
    nquads = [(d + 1) for d in p_degree] # [4*(d + 1) for d in p_degree]
    print(' .. Primal discrete derham sequence...')
    p_derham_h = discretize(p_derham, domain_h, degree=p_degree, nquads=nquads) #, backend=PSYDAC_BACKENDS[backend_language])
    # primal is with hom bc, but this is in the conf projections

    t_stamp = time_count()
    print(' .. Dual derham sequence...')
    d_derham  = Derham(domain, ["H1", "Hdiv", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. Dual discrete derham sequence...')
    d_derham_h = discretize(d_derham, domain_h, degree=d_degree, nquads=nquads) #, backend=PSYDAC_BACKENDS[backend_language])
    
    p_V0h = p_derham_h.V0
    p_V1h = p_derham_h.V1
    p_V2h = p_derham_h.V2

    d_V0h = d_derham_h.V0
    d_V1h = d_derham_h.V1
    d_V2h = d_derham_h.V2

    if test_case == 'norm_Lambda0':
        
        print("p_MM0 already assembled")
        # p_HOp0    = HodgeOperator(p_V0h, domain_h, backend_language=backend_language, load_dir=pm_load_dir, load_space_index=0)
        # p_MM0     = p_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
        
        # Vh = p_V0h
        # V = Vh.symbolic_space
        # u, v = elements_of(V, names='u, v')
        # expr   = u*v
        #     expr   = dot(u,v)
        # a = BilinearForm((u,v), integral(domain, expr))
        # ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[backend_language])
        # p_MM0 = ah.assemble().tosparse()  # Mass matrix in stencil > scipy format
        
    else:

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

        if mom_pres:
            # nb of interior functions per patch (and per dimension): nb_interior = n_cells + p - 2 *(1+reg)
            # we can only preserve moments of degree p if p +1 <= nb_interior
            max_p_moments = [ncells[d] + p_degree[d] - 2 *(1+reg) - 1 for d in range(2)]
            p_moments_V0 = max([max(-1, min(p_degree[d], max_p_moments[d])) for d in range(2)])
            p_moments_V1 = max([max(-1, min(p_degree[d]-1, max_p_moments[d])) for d in range(2)])
            p_moments_V2 = p_moments_V1

            print(f'with max_p_moments = {max_p_moments}, using:')
            print(f' p_moments_V0 = {p_moments_V0}')
            print(f' p_moments_V1 = {p_moments_V1}')
            print(f' p_moments_V2 = {p_moments_V2}')
        else:
            p_moments_V0 = p_moments_V1 = p_moments_V2 = -1

        # p_PP0     = cps.Conf_proj_0_c1(p_V0h, nquads=nquads, hom_bc=hom_bc)
        # p_PP1     = cps.Conf_proj_1_c1(p_V1h, nquads=nquads, hom_bc=hom_bc)

        p_PP0     = cps.Conf_proj_0_c01(p_V0h, reg=reg, p_moments=p_moments_V0, nquads=nquads, hom_bc=hom_bc)
        # p_PP0_std     = cps.Conf_proj_0(p_V0h, nquads = [4*(d + 1) for d in d_degree])
        # print(f'sp_norm(p_PP0_std-p_PP0) = {sp_norm(p_PP0_std-p_PP0)}')
        p_PP1     = cps.Conf_proj_1_c01(p_V1h, reg=reg, p_moments=p_moments_V1, nquads=nquads, hom_bc=hom_bc)
        p_PP2     = cps.Conf_proj_0_c01(p_V2h, reg=reg-1, p_moments=p_moments_V2, nquads=nquads)

        # print('OAIJ')
        # exit()
        # import scipy
        # scipy.set_printoptions(linewidth=300)
        # p_PP0.maxprint = 300
        
        # P0.maxprint = np.inf
        # print(p_PP0) #.toarray())
        
        # from matrepr import mprint
        # exit()

        # p_PP1_C0     = cps.Conf_proj_1(p_V1h, nquads=nquads)  # to compare

        d_PP0     = cps.Conf_proj_0(d_V0h, nquads = [4*(d + 1) for d in d_degree])
        d_PP1     = cps.Conf_proj_1(d_V1h, nquads = [4*(d + 1) for d in d_degree])

        t_stamp = time_count(t_stamp)
        print('Mass matrices...')
        ### TODO: they are not "Hodge Operators" for the primal/dual sequences.... 
        p_HOp0    = HodgeOperator(p_V0h, domain_h, backend_language=backend_language, nquads=nquads, load_dir=pm_load_dir, load_space_index=0)
        p_MM0     = p_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
        p_MM0_inv = p_HOp0.to_sparse_matrix()                # inverse mass matrix

        p_HOp1    = HodgeOperator(p_V1h, domain_h, backend_language=backend_language, nquads=nquads, load_dir=pm_load_dir, load_space_index=1)
        p_MM1     = p_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
        p_MM1_inv = p_HOp1.to_sparse_matrix()                # inverse mass matrix

        p_HOp2    = HodgeOperator(p_V2h, domain_h, backend_language=backend_language, nquads=nquads, load_dir=pm_load_dir, load_space_index=2)
        p_MM2     = p_HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix
        p_MM2_inv = p_HOp2.to_sparse_matrix()                # inverse mass matrix

        d_HOp0    = HodgeOperator(d_V0h, domain_h, backend_language=backend_language, nquads=nquads, load_dir=dm_load_dir, load_space_index=0)
        d_MM0     = d_HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
        d_MM0_inv = d_HOp0.to_sparse_matrix()                # inverse mass matrix

        d_HOp1    = HodgeOperator(d_V1h, domain_h, backend_language=backend_language, nquads=nquads, load_dir=dm_load_dir, load_space_index=1)
        d_MM1     = d_HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
        d_MM1_inv = d_HOp1.to_sparse_matrix()                # inverse mass matrix


        t_stamp = time_count(t_stamp)
        print('Hodge operators (primal/dual sequences)...')
        
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
    G_sol_log = [[lambda xi1, xi2, ii=i : (ii+1) for d in [0,1]] for i in range(nb_patches)]
    
    f0_log = [lambda xi1, xi2, ii=i : (ii+1) for i in range(nb_patches)]

    g_symb = sin(pi*x)*sin(pi*y)
    g = lambdify(domain.coordinates, g_symb)

    p_geomP0, p_geomP1, p_geomP2 = p_derham_h.projectors()
    d_geomP0, d_geomP1, d_geomP2 = d_derham_h.projectors()

    if test_case == "unit_tests":

        print('-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ')
        print(f"do some tests, with: ")
        print(f"    mom_pres = {mom_pres}")
        print(f"    proj_op = {cps.proj_op}")
        print(f"    hom_bc = {hom_bc}")
        print(f"    reg = {reg}")
        
        test_err(p_PP0, p_PP0@p_PP0, label_error='|| p_PP0 - p_PP0@p_PP0 ||')
        test_err(p_PP1, p_PP1@p_PP1, label_error='|| p_PP1 - p_PP1@p_PP1 ||')
        test_err(p_bG@p_PP0, p_PP1@p_bG@p_PP0, label_error='|| p_bG@p_PP0 - p_PP1@p_bG@p_PP0 ||')
        test_err(p_bC@p_PP1, p_PP2@p_bC@p_PP1, label_error='|| p_bC@p_PP1 - p_PP2@p_bC@p_PP1 ||')

        # unit test: different projections of a polynomial should be exact: 
        
        # tests on p_PP0:

        g0 = get_polynomial_function(degree=p_degree, hom_bc_axes=[hom_bc,hom_bc], domain=domain)        
        g0h = P_phys_h1(g0, p_geomP0, domain, mappings_list)
        g0_c = g0h.coeffs.toarray()  
        
        tilde_g0_c = p_derham_h.get_dual_dofs(space='V0', f=g0, backend_language=backend_language, return_format='numpy_array', nquads=nquads)
        g0_L2_c = p_MM0_inv @ tilde_g0_c

        test_err(app_c = g0_L2_c,         ref_c = g0_c, MM = p_MM0, label_error="|| (P0_geom - P0_L2) polynomial ||_L2")
        test_err(app_c = p_PP0 @ g0_L2_c, ref_c = g0_c, MM = p_MM0, label_error="|| (P0_geom - confP0 @ P0_L2) polynomial ||_L2")
        
        if mom_pres:
            # testing that polynomial moments are preserved: the following projection should be exact:
            #   conf_P0* : L2 -> V0 defined by <conf_P0* g, phi> := <g, conf_P0 phi> for all phi in V0            
            g0 = get_polynomial_function(degree=[p_moments_V0,p_moments_V0], hom_bc_axes=[False, False], domain=domain)
            g0h = P_phys_h1(g0, p_geomP0, domain, mappings_list)
            g0_c = g0h.coeffs.toarray()    

            tilde_g0_c = p_derham_h.get_dual_dofs(space='V0', f=g0, backend_language=backend_language, return_format='numpy_array', nquads=nquads)
            g0_star_c = p_MM0_inv @ p_PP0.transpose() @ tilde_g0_c
            test_err(app_c = g0_star_c, ref_c = g0_c, MM = p_MM0, label_error="|| (P0_geom - P0_star) polynomial ||_L2")

        # tests on p_PP1:

        G1 = Tuple(
            get_polynomial_function(degree=[p_degree[0]-1,p_degree[1]], hom_bc_axes=[False,hom_bc], domain=domain),
            get_polynomial_function(degree=[p_degree[0],p_degree[1]-1], hom_bc_axes=[hom_bc,False], domain=domain)
        )

        G1h = P_phys_hcurl(G1, p_geomP1, domain, mappings_list)
        G1_c = G1h.coeffs.toarray()  
        
        tilde_G1_c = p_derham_h.get_dual_dofs(space='V1', f=G1, backend_language=backend_language, return_format='numpy_array', nquads=nquads)
        G1_L2_c = p_MM1_inv @ tilde_G1_c

        test_err(app_c = G1_L2_c,         ref_c = G1_c, MM = p_MM1, label_error="|| (P1_geom - P1_L2) polynomial ||_L2")
        test_err(app_c = p_PP1 @ G1_L2_c, ref_c = G1_c, MM = p_MM1, label_error="|| (P1_geom - confP1 @ P1_L2) polynomial ||_L2")

        if mom_pres:
            # testing that polynomial moments are preserved: the following projection should be exact:            
            #   conf_P1* : L2 -> V1  defined by  <conf_P1* G, Phi> := <G, conf_P1 Phi> for all Phi in V1            
            G1 = Tuple(
                get_polynomial_function(degree=[p_moments_V1,p_moments_V1], hom_bc_axes=[False,False], domain=domain),
                get_polynomial_function(degree=[p_moments_V1,p_moments_V1], hom_bc_axes=[False,False], domain=domain)
            )

            G1h = P_phys_hcurl(G1, p_geomP1, domain, mappings_list)
            G1_c = G1h.coeffs.toarray()  

            tilde_G1_c = p_derham_h.get_dual_dofs(space='V1', f=G1, backend_language=backend_language, return_format='numpy_array', nquads=nquads)
            G1_star_c = p_MM1_inv @ p_PP1.transpose() @ tilde_G1_c
            test_err(app_c = G1_star_c, ref_c = G1_c, MM = p_MM1, label_error="|| (P1_geom - P1_star) polynomial ||_L2")

        # tests on p_PP2 (non trivial for reg = 1):
        
        g2 = get_polynomial_function(degree=[p_degree[d]-1 for d in range(2)], hom_bc_axes=[False,False], domain=domain)        
        g2h = P_phys_l2(g2, p_geomP2, domain, mappings_list)
        g2_c = g2h.coeffs.toarray()  
        
        tilde_g2_c = p_derham_h.get_dual_dofs(space='V2', f=g2, backend_language=backend_language, return_format='numpy_array', nquads=nquads)
        g2_L2_c = p_MM2_inv @ tilde_g2_c

        test_err(app_c = g2_L2_c,         ref_c = g2_c, MM = p_MM2, label_error="|| (P2_geom - P2_L2) polynomial ||_L2")
        test_err(app_c = p_PP2 @ g2_L2_c, ref_c = g2_c, MM = p_MM2, label_error="|| (P2_geom - confP2 @ P2_L2) polynomial ||_L2")

        if mom_pres:
            # testing that polynomial moments are preserved, as above
            g2 = get_polynomial_function(degree=[p_moments_V2,p_moments_V2], hom_bc_axes=[False, False], domain=domain)
            g2h = P_phys_l2(g2, p_geomP2, domain, mappings_list)
            g0_c = g2h.coeffs.toarray()    

            tilde_g2_c = p_derham_h.get_dual_dofs(space='V2', f=g2, backend_language=backend_language, return_format='numpy_array', nquads=nquads)
            g2_star_c = p_MM2_inv @ p_PP2.transpose() @ tilde_g2_c
            test_err(app_c = g2_star_c, ref_c = g2_c, MM = p_MM2, label_error="|| (P2_geom - P2_star) polynomial ||_L2")


        print()
        print(f'batch of tests done')
        print('-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ')

        return    

    sol = 'discontinuous'
    print(f"running with test_case = {test_case} and sol = {sol} ...")

    if test_case == 'p_PP0':
        # no pull-back needed...
        f0h = p_geomP0(f0_log)  # P_phys_h1(f0, p_geomP0, domain, mappings_list)
        f0_c = f0h.coeffs.toarray()
        
        ref_c = f0_c
        label_ref   = 'f'
        
        app_c = p_PP0 @ f0_c
        label_app   = 'P0f'

        MM      = p_MM0
        Vh      = p_V0h
        Vh_kind = 'h1'
        plot_type = 'components'
    
    elif test_case == 'PGP0':
        # no pull-back needed...
        f0h = p_geomP0(f0_log)  # P_phys_h1(f0, p_geomP0, domain, mappings_list)
        f0_c = f0h.coeffs.toarray()
        
        ref_c = p_bG @ p_PP0 @ f0_c
        label_ref   = 'GPf'
        
        app_c = p_PP1 @ p_bG @ p_PP0 @ f0_c
        label_app   = 'PGPf'

        MM      = p_MM1
        Vh      = p_V1h
        Vh_kind = 'hcurl'
        plot_type = 'components'


    elif test_case == 'p_PP1':

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

    elif test_case == "pP0_order":
        # unit test: different projections of a polynomial should be exact: 
        degree = p_degree #[p_degree[d] - 1 for d in range(2)]
        if hom_bc:
            assert degree[0] > 1 and degree[1] > 1
            g0 = (  x * (x-np.pi) * (x-1.554)**(degree[0]-2)
                  * y * (y-np.pi) * (y-0.324)**(degree[1]-2)
             )
        else:
            g0 = (  (x-0.543)**2 * (x-1.554)**(degree[0]-2)
                  * (y-1.675)**2 * (y-0.324)**(degree[1]-2)
             )

        # applying P0_geom (should be exact)
        g0h = P_phys_h1(g0, p_geomP0, domain, mappings_list)
        g0_c = g0h.coeffs.toarray()
        
        # applying conf_P0 @ P0_L2 
        tilde_g0_c = p_derham_h.get_dual_dofs(space='V0', f=g0, backend_language=backend_language, return_format='numpy_array', nquads=nquads)
        g0_L2_c = p_MM0_inv @ tilde_g0_c

        app_c = p_PP0 @ g0_L2_c
        ref_c = g0_c

        label_ref   = 'P_geom g'
        label_app   = 'P_conf P_L2 g'
        MM      = p_MM0
        Vh      = p_V0h
        Vh_kind = 'h1'
        plot_type = 'components'

    elif test_case == "pP0_moment_order":
        # testing that moments against a polynomial are exact:
        # projection P* : L2 -> V0, 
        # <conf_P* g, phi> := <g, conf_P phi> for all phi in V0
        # should be exact
        
        degree = p_degree #[p_degree[d] - 1 for d in range(2)]
        if hom_bc:
            assert degree[0] > 1 and degree[1] > 1
            g0 = (  x * (x-np.pi) * (x-1.554)**(degree[0]-2)
                  * y * (y-np.pi) * (y-0.324)**(degree[1]-2)
             )
        else:
            g0 = (  (x-0.543)**2 * (x-1.554)**(degree[0]-2)
                  * (y-1.675)**2 * (y-0.324)**(degree[1]-2)
             )

        # applying P0_geom (should be exact)
        g0h = P_phys_h1(g0, p_geomP0, domain, mappings_list)
        g0_c = g0h.coeffs.toarray()
        
        # applying P0_L2 
        tilde_g0_c = p_derham_h.get_dual_dofs(space='V0', f=g0, backend_language=backend_language, return_format='numpy_array', nquads=nquads)
        g0_star_c = p_MM0_inv @ p_PP0.transpose() @ tilde_g0_c


        ref_c = g0_c
        app_c = g0_star_c

        label_ref   = 'P_geom g0'
        label_app   = 'P_star g0'
        MM      = p_MM0
        Vh      = p_V0h
        Vh_kind = 'h1'
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
        # plot_type = 'amplitude'
        show_grid = False
        plot_field(numpy_coeffs=app_c, Vh=Vh, space_kind=Vh_kind, 
                plot_type=plot_type,
                domain=domain, title=label_app, cmap='viridis', 
                filename=plot_dir+'test='+test_case+'_approx.png', 
                hide_plot=hide_plots,
                show_grid=show_grid, 
                N_vis=20,
                #    eta_crop=[[0.2,0.3], [0,10]],
                surface_plot=True)
        
        plot_field(numpy_coeffs=ref_c, Vh=Vh, space_kind=Vh_kind, 
                plot_type=plot_type,
                domain=domain, title=label_ref, cmap='viridis', 
                filename=plot_dir+'test='+test_case+'_ref.png', 
                hide_plot=hide_plots, 
                show_grid=show_grid,
                N_vis=20,
                #    eta_crop=[[0.2,0.3], [0,10]],
                surface_plot=True)

        label_err  = label_ref + ' - ' + label_app + ' with: ' + cps_opts
        plot_field(numpy_coeffs=app_c-ref_c, Vh=Vh, space_kind=Vh_kind, 
                plot_type=plot_type,
                domain=domain, title=label_err, cmap='viridis', 
                filename=plot_dir+'test='+test_case+'_err.png', 
                hide_plot=hide_plots,
                show_grid=show_grid,
                cb_min=-0.00015,
                cb_max=0.00015,
                N_vis=20,
                #    eta_crop=[[0.2,0.3], [0,10]],
                surface_plot=True)

    else:
        print( "-- -- -- skipping plots -- -- -- ")
    
    err_c = app_c - ref_c 
    L2_norm  = np.sqrt(max(np.dot(ref_c, MM @ ref_c), np.dot(app_c, MM @ app_c)))
    L2_error = np.sqrt(np.dot(err_c, MM @ err_c))
    L2_relerror = L2_error/L2_norm

    return L2_error, L2_relerror    

if __name__ == '__main__':

    # global total_tests, failed_tests
    t_stamp_full = time_count()

    # test_case = "norm_Lambda0"
    test_case = "p_PP0"
    # test_case = "PGP0"
    # test_case = "p_PP1" # "d_HH1" # "p_PP1" # "p_PP1_C1" # "d_HH1" # "p_PP1_C1" # 
    # test_case = "d_HH1"
    test_case = "pP0_order"
    test_case = "pP0_moment_order"

    test_case = "unit_tests"

    # test_case = "pP1_L2" 
    # test_case = "p_HH2"
    refined_square = False
    
    make_plots = True # False # 
    hide_plots = True # False # 
    

    deg_s = [3] #, 4]
    
    # nb of cells (per patch and dim)
    nbc_s = [4]
    # nbc_s = [2,4,6,8,16,32]

    # nb of patches (per dim)
    # nbp_s = [1] 
    nbp_s = [2] #,8]

    errors = [[[ None for nbc in nbc_s] for nbp in nbp_s] for deg in deg_s]
    error_dir = './errors'      

    cps.proj_op = 1 # use as argument to conf projections
    test_parameters_list = []
    if test_case == "unit_tests":
        for mom_pres in [False, True]:
            for reg in [0, 1]:
                for hom_bc in [False, True]:
                    test_parameters_list.append({
                        'mom_pres' : mom_pres,
                        'reg' : reg,
                        'hom_bc' : hom_bc
                        })

    else:
        test_parameters_list.append({
            'mom_pres' : False,
            'reg' : 1,
            'hom_bc' : True
            })
        assert len(test_parameters_list) == 1

    for params in test_parameters_list:
        mom_pres=params['mom_pres']
        hom_bc=params['hom_bc']
        reg=params['reg']
        
        if mom_pres:
            cps_opts = 'po{}_wimop'.format(cps.proj_op)
            cps_options = 'proj_op={}, with mom preservation'.format(cps.proj_op)
        else:
            cps_opts = 'po{}_nomop'.format(cps.proj_op)
            cps_options = 'proj_op={}, no mom preservation'.format(cps.proj_op)

        check_file(
            error_dir=error_dir,
            name=test_case+'_'+cps_opts, 
            )

        for i_deg, deg in enumerate(deg_s): 
            for i_nbp, nbp in enumerate(nbp_s): 
                for i_nbc, nbc in enumerate(nbc_s): 
                    nb_patch_x = nbp
                    nb_patch_y = nbp     
                    if nbp == 1:
                        domain_name = 'square'
                    elif refined_square:
                        domain_name = 'refined_square'
                        raise NotImplementedError("CHECK FIRST")
                    else:
                        # domain_name = 'square_9'
                        domain_name = 'multipatch_rectangle' #'square_9'

                    m_load_dir = 'matrices_{}_nbp={}_nc={}_deg={}/'.format(domain_name, nbp, nbc, deg)
                    run_dir = '{}_nbp={}_nc={}_deg={}/'.format(domain_name, nbp, nbc, deg)

                    res = try_ssc_2d(
                        ncells=[nbc,nbc], 
                        nb_patch_x=nb_patch_x,
                        nb_patch_y=nb_patch_y,
                        hom_bc=hom_bc,
                        mom_pres=mom_pres,
                        reg=reg,
                        p_degree=[deg,deg],
                        domain_name=domain_name, 
                        test_case='unit_tests',
                        plot_dir='./plots/'+run_dir,
                        m_load_dir=m_load_dir,
                        backend_language='python', #'pyccel-gcc'
                        hide_plots=hide_plots,
                        make_plots=make_plots,
                        cps_opts=cps_opts,
                    )

                    if test_case != 'unit_tests':

                        error, relerror = res

                        print("-------------------------------------------------")
                        print("for deg = {}, nb_patches = {}**2".format(deg,nbp))
                        print("error         : {}".format(error))
                        print("relative error: {}".format(relerror))
                        print("-------------------------------------------------\n")
                
                        errors[i_deg][i_nbp][i_nbc] = error
    
    if test_case == 'unit_tests':
        print("-------------------------------------------------")
        print(f" total nb of failed tests: {failed_tests} / {total_tests}       ")
        print("-------------------------------------------------\n")

    else:    

        if len(nbc_s) == 1:
            diag_filename = error_dir+f'/errors_{cps_opts}.txt'
            write_diags_deg_nbp(errors, deg_s, nbp_s, nbc_s[0], 
                                filename=diag_filename, 
                                name=test_case+'_'+cps_opts,
                                title=test_case + ' ' + cps_options)
        
        else:        
            for i_deg, deg in enumerate(deg_s): 
                for i_nbp, nbp in enumerate(nbp_s): 
                    print("-------------------------------------------------\n")
                    print("errors as nb of cells increase: ")
                    for i_nbc, nbc in enumerate(nbc_s): 
                        h = np.pi / nbc
                        err = errors[i_deg][i_nbp][i_nbc]                    
                        print("error for deg = {}, nbp = {}, nbc = {}:  {}".format(
                            deg, nbp, nbc, err))
                        print("ratio error / h :  {}".format(err / h))

            print("-------------------------------------------------\n")
            print("WARNING: not writing any error file")        
            print("-------------------------------------------------\n")


    time_count(t_stamp_full, msg='full program')