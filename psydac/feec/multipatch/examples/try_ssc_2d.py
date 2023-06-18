
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
from psydac.feec.multipatch.plotting_utilities          import plot_field
from psydac.feec.multipatch.utilities                   import time_count #, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
# from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file
from psydac.feec.pull_push import pull_2d_h1, pull_2d_hcurl, pull_2d_hdiv, pull_2d_l2

from psydac.feec.multipatch.bilinear_form_scipy import construct_pairing_matrix
from psydac.feec.multipatch.conf_projections_scipy import Conf_proj_1
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
        prml_degree=[3,3], 
        domain_name='refined_square', 
        plot_dir='./plots/', 
        test_prml_H1=True, # False # 
        std_P1=False,
        backend_language='pyccel-gcc'
        ):

    """
    Testing the Strong-Strong Conga (SSC) sequence:
    with two strong broken DeRham sequences (a primal Hermite with hom BC and a dual Lagrange) and pairing matrices
    """

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting hcurl_solve_eigen_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' prml_degree = {}'.format(prml_degree))
    print(' domain_name = {}'.format(domain_name))
    # print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')
    t_stamp = time_count()
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

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
    dual_degree = [d-1 for d in prml_degree]

    t_stamp = time_count()
    print(' .. Primal derham sequence...')
    prml_derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. Primal discrete derham sequence...')
    prml_derham_h = discretize(prml_derham, domain_h, degree=prml_degree) #, backend=PSYDAC_BACKENDS[backend_language])
    # primal is with hom bc, but this is in the conf projections

    t_stamp = time_count()
    print(' .. Dual derham sequence...')
    dual_derham  = Derham(domain, ["H1", "Hdiv", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. Dual discrete derham sequence...')
    dual_derham_h = discretize(dual_derham, domain_h, degree=dual_degree) #, backend=PSYDAC_BACKENDS[backend_language])
    
    prml_V0h = prml_derham_h.V0
    prml_V1h = prml_derham_h.V1
    prml_V2h = prml_derham_h.V2

    dual_V0h = dual_derham_h.V0
    dual_V1h = dual_derham_h.V1
    dual_V2h = dual_derham_h.V2

    t_stamp = time_count(t_stamp)
    print('Mass matrices...')
    m_load_dir = None

    ## NOTE: with a strong-strong diagram we should not call these "Hodge" operators !! 
    prml_H1 = HodgeOperator(prml_V1h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=1)
    dual_H1 = HodgeOperator(dual_V1h, domain_h, backend_language=backend_language, load_dir=m_load_dir, load_space_index=1)
    
    if test_prml_H1:
        dual_M1 = dual_H1.get_dual_Hodge_sparse_matrix()  # =         mass matrix of dual_V1
        dual_M1_inv = dual_H1.to_sparse_matrix()  # = inverse mass matrix of dual_V1
        prml_M1_inv = None

    else:
        prml_M1 = prml_H1.get_dual_Hodge_sparse_matrix()  # =         mass matrix of prml_V1
        prml_M1_inv = prml_H1.to_sparse_matrix()              # = inverse mass matrix of prml_V1
        dual_M1_inv = None
    
    t_stamp = time_count(t_stamp)
    print('Pairing matrices...')
    # compute prml_K1 = (<dual_Lambda^1_i, prml_Lambda^1_j>)
    # 

    assemble_bf = False
    if assemble_bf:
        prml_V1 = prml_V1h.symbolic_space
        dual_V1 = dual_V1h.symbolic_space

        u1 = element_of(dual_V1, name='u1')
        v1 = element_of(prml_V1, name='v1')

        a = BilinearForm((u1,v1), integral(domain, dot(u1,v1)))
        ah = discretize(a, domain_h, [dual_V1h, prml_V1h], backend=PSYDAC_BACKENDS[backend_language])

        prml_K1 = ah.assemble().tosparse()  # matrix in scipy format
        dual_K1 = prml_K1.transpose()

    else:
        prml_K1 = construct_pairing_matrix(dual_V1h,prml_V1h).tocsr()  # matrix in scipy format
        dual_K1 = prml_K1.transpose()

    
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    t_stamp = time_count(t_stamp)
    print('Compute conforming projection matrix')
    
    if test_prml_H1:
        # prml_H1: prml_V1 -> dual_V1
        print('Compute conforming projection matrix: dual_P1 ...')
        if std_P1:
            dual_cP1_matrix = dual_derham_h.conforming_projection(space='V1', hom_bc=False, backend_language=backend_language, load_dir=m_load_dir)
            dual_cP1 = dual_cP1_matrix.to_sparse_matrix()
        else:
            dual_cP1 = Conf_proj_1(dual_V1h,nquads = [4*(d + 1) for d in dual_degree])
        print('WARNING: dual_P1 should be corrected to map in the C1 spaces. And to preserve polynomial moments')
        prml_H1 = dual_M1_inv @ dual_cP1.transpose() @ prml_K1
        prml_PL2 = dual_M1_inv @ prml_K1  # L2 projection prml_V1 -> dual_V1 
    else:
        # dual_H1: dual_V1 -> prml_V1
        print('Compute conforming projection matrix: prml_P1 ...')
        prml_cP1_matrix = prml_derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language, load_dir=m_load_dir)
        prml_cP1 = prml_cP1_matrix.to_sparse_matrix()
        print('WARNING: prml_P1 should be corrected to map in the C1 spaces. And to preserve polynomial moments')
        dual_H1 = prml_M1_inv @ prml_cP1.transpose() @ dual_K1
        dual_PL2 = prml_M1_inv @ dual_K1  # L2 projection dual_V1 -> prml_V1

    # some target function
    x,y    = domain.coordinates
    f_vect  = Tuple(sin(pi*y),
                    sin(pi*x))

    prml_P0, prml_P1, prml_P2 = prml_derham_h.projectors()
    dual_P0, dual_P1, dual_P2 = dual_derham_h.projectors()

    f_x = lambdify(domain.coordinates, f_vect[0])
    f_y = lambdify(domain.coordinates, f_vect[1])

    if test_prml_H1:
        print(" -----  approx f in prml_V1  ---------")
        prml_f_log = [pull_2d_hcurl([f_x, f_y], F) for F in mappings_list]
        f_h = prml_P1(prml_f_log)
        prml_f1 = f_h.coeffs.toarray()
        print(" -----  Hodge: f -> dual_V1  ---------")
        dual_f1     = prml_H1  @ prml_f1
        dual_f1_PL2 = prml_PL2 @ prml_f1
        error_dual_f1 = dual_f1 - dual_f1_PL2
        L2_error = np.sqrt(np.dot(error_dual_f1, dual_M1 @ error_dual_f1)/np.dot(dual_f1, dual_M1 @ dual_f1))
        test_label = 'test_prml_H1'
        prml_f1_label = 'f1 = prml_P1 @ f'
        dual_f1_label = 't_f1 = H1 @ f1'
    else:
        print(" -----  approx f in dual_V1  ---------")
        dual_f_log = [pull_2d_hdiv([f_x, f_y], F) for F in mappings_list]
        f_h = dual_P1(dual_f_log)
        dual_f1 = f_h.coeffs.toarray()
        print(" -----  Hodge: f -> prml_V1  ---------")
        prml_f1     = dual_H1  @ dual_f1
        prml_f1_PL2 = dual_PL2 @ dual_f1
        error_prml_f1 = prml_f1 - prml_f1_PL2
        L2_error = np.sqrt(np.dot(error_prml_f1, prml_M1 @ error_prml_f1)/np.dot(prml_f1, prml_M1 @ prml_f1))

        test_label = 'test_dual_H1'
        dual_f1_label = 't_f1 = t_P1 @ f'
        prml_f1_label = 'f1 = t_H1 @ t_f1'

    plot_field(numpy_coeffs=prml_f1, Vh=prml_V1h, space_kind='hcurl', domain=domain, title=prml_f1_label, cmap='viridis', filename=plot_dir+test_label+'_prml_f1.png', hide_plot=False)
    plot_field(numpy_coeffs=dual_f1, Vh=dual_V1h, space_kind='hdiv',  domain=domain, title=dual_f1_label, cmap='viridis', filename=plot_dir+test_label+'_dual_f1.png', hide_plot=False)

    return L2_error    

if __name__ == '__main__':

    t_stamp_full = time_count()

    test_prml_H1 = False # True # 
    std_P1 = False # use moment conserving P1 
    refined_square = False
    
    if refined_square:
        domain_name = 'refined_square'
        nc = 10
    else:
        domain_name = 'square_9'
        nc = 2

    deg = 3

    run_dir = '{}_nc={}_deg={}/'.format(domain_name, nc, deg)

    # m_load_dir = 'matrices_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
    error = try_ssc_2d(
        ncells=[nc,nc], 
        prml_degree=[deg,deg],
        domain_name=domain_name, 
        plot_dir='./plots/'+run_dir,
        test_prml_H1=test_prml_H1,
        std_P1=std_P1,
        backend_language='python' #'pyccel-gcc'
    )

    print("error: {}".format(error))
    time_count(t_stamp_full, msg='full program')