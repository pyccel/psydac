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

def run(degree, ncells, nbp, primal = False, alpha = 1.25,kappa = 1.5, epsilon = 0.3, k_theta = 6,omega = 4):
    mom_pres = True 
    C1_proj_opt = None
    r_min = 0.25 
    r_max = 3.25


    domain = build_multipatch_annulus(nbp[0], nbp[1], r_min, r_max)
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

    
    #p_KK2 = construct_pairing_matrix(d_V0h,p_V2h).tocsr()  # matrix in scipy format
    
    d_KK1 = construct_pairing_matrix(p_V1h,d_V1h).tocsr()  # matrix in scipy format
    p_KK1 = construct_pairing_matrix(d_V1h,p_V1h).tocsr()  # matrix in scipy format


    x,y = domain.coordinates

    nb = sqrt(alpha**2 * x**2 + y**2 * 1/alpha**2)
    b = Tuple(-y/alpha * 1/nb, alpha * x * 1/nb)
    t = 1.23456789 #3/7 * np.pi/2
    E_ex, B_ex, D_ex, J_ex = get_polarized_annulus_potential_solution(b, omega, k_theta, epsilon, kappa, t=t, r_min = r_min, r_max = r_max, domain=domain)

    Eex_c = P_phys_hcurl(E_ex, p_geomP1, domain, mappings_list).coeffs.toarray()
    Bex_c = P_phys_l2(B_ex, p_geomP2, domain, mappings_list).coeffs.toarray()

    Dex_c = P_phys_hdiv(D_ex, d_geomP1, domain, mappings_list).coeffs.toarray()
    Jex_c = P_phys_hdiv(J_ex, d_geomP1, domain, mappings_list).coeffs.toarray()
    Hex_c = P_phys_h1(B_ex, d_geomP0, domain, mappings_list).coeffs.toarray()

    if primal: 
        u, v = elements_of(p_derham.V1, names='u, v')
        bE = dot(v, b)
        Eu = dot(u, v)
        ub = dot(u, b)
        mass = BilinearForm((u,v), integral(domain, ((1+kappa)*Eu - kappa * bE * ub)))
        massh = discretize(mass, domain_h, [p_V1h, p_V1h])
        M = massh.assemble().tosparse().toarray()
        #print("condition number M_eps: {}".format(np.linalg.cond(M)))
       # print("condition number p_MM1: {}".format(np.linalg.cond(p_MM1.toarray())))
        
        Coup_Op = np.linalg.inv(M) @ p_PP1.transpose() @ d_KK1

    else:
        u, v = elements_of(d_derham.V1, names='u, v')
        bE = dot(v, b)
        Eu = dot(u, v)
        ub = dot(u, b)
        mass = BilinearForm((u,v), integral(domain, (Eu + kappa * bE * ub)/(1+kappa) ))
        massh = discretize(mass, domain_h, [d_V1h, d_V1h])
        M = massh.assemble().tosparse().toarray()
        #print("condition number M_eps_inv: {}".format(np.linalg.cond(M)))
        #print("condition number d_MM1: {}".format(np.linalg.cond(d_MM1.toarray())))

        Coup_Op = p_MM1_inv @ p_PP1.transpose() @ d_KK1 @ d_MM1_inv @ M

    err_c = Eex_c - Coup_Op @ Dex_c
    L2_error = np.sqrt(np.dot(err_c, p_MM1.dot(err_c)))

    return L2_error
if __name__ == '__main__':

# quick run, to test 
    plot_dir = "./02_cond_test_new"
    diag_file = plot_dir+'/diag.txt'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(diag_file):
        open(diag_file, 'w')

    ###
    alpha = 1.25
    kappa = 1.5
    epsilon = 0.2
    k_theta = 3
    omega = 2
    primal = True # Approx of the coupling operator
    deg = [3, 4, 5]
    patches = [[1,2], [2, 4], [4,8]]
    ncells  = [12,12]

    errors = np.zeros((len(deg), len(patches)))
    #dof = [np.prod(ncells)*np.prod(k) for k in patches]
    dof = [ [(ncells[0]+d) * k[0] for k in patches] for d in deg]
    

    for (i, d) in enumerate(deg):
        degree = [d, d]
        with open(diag_file, 'a') as a_writer:
            a_writer.write(' \n new run \n \n')
            a_writer.write(' degree: {} \n'.format(degree))
            a_writer.write(' patches: {} \n'.format(patches))

        for (j, nbp) in enumerate(patches): 

            error = run(degree, ncells, nbp, primal, alpha,kappa, epsilon, k_theta,omega)
            print(error)

            errors[i][j] = error
            with open(diag_file, 'a') as a_writer:
                a_writer.write('{}\n'.format(error))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    for (i, d) in enumerate(deg):
        plt.plot(dof[i], errors[i], label="degree {}".format(d))

    for d in [2,3,4,5]:
        plt.plot(dof[0], (1/5)**(d) * dof[0][0]**d * np.power( np.divide(1, dof[0]),d), '--', label="order {}".format(d))

    plt.legend()
    plt.show()
