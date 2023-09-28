import numpy as np
import pytest

from collections import OrderedDict

from scipy.sparse.linalg                import norm as sp_norm
from sympy                              import Tuple
from sympde.topology                    import Derham
from psydac.feec.multipatch.api         import discretize
from psydac.feec.multipatch.operators   import HodgeOperator
from psydac.feec.multipatch.conf_projections_scipy      import conf_projectors_scipy
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_rectangle
from psydac.feec.multipatch.utils_conga_2d              import P_phys_l2, P_phys_hdiv, P_phys_hcurl, P_phys_h1


def get_polynomial_function(degree, hom_bc_axes, domain):
    x, y = domain.coordinates            
    if hom_bc_axes[0]:                
        assert degree[0] > 1
        g0_x = x * (x-np.pi) * (x-1.554)**(degree[0]-2)
    else:
        # if degree[0] > 1:
        #     g0_x = (x-0.543)**2 * (x-1.554)**(degree[0]-2)
        # else:
        g0_x = (x-1.554)**degree[0]

    if hom_bc_axes[1]:                
        assert degree[1] > 1
        g0_y = y * (y-np.pi) * (y-0.324)**(degree[1]-2)
    else:
        # if degree[1] > 1:
        #     g0_y = (y-1.675)**2 * (y-0.324)**(degree[1]-2)

        # else:
        g0_y = (y-0.324)**degree[1]

    return g0_x * g0_y

#==============================================================================
# @pytest.mark.parametrize('V1_type', ["Hcurl","Hdiv"])
# @pytest.mark.parametrize('degree', [2,2]) #[3,3], [4,4])
# @pytest.mark.parametrize('ncells', [3,4])
# @pytest.mark.parametrize('nb_patches', [2,2])
# @pytest.mark.parametrize('reg', [0,1])
# @pytest.mark.parametrize('hom_bc', [False, True])
# @pytest.mark.parametrize('mom_pres', [False, True])

@pytest.mark.parametrize('V1_type', ["Hcurl", "Hdiv"])
@pytest.mark.parametrize('degree', [[2,2], [3,3], [4,4]])
@pytest.mark.parametrize('ncells', [[3,4], [5,5]])
@pytest.mark.parametrize('nb_patches', [[2,2]])
@pytest.mark.parametrize('reg', [0,1])
@pytest.mark.parametrize('hom_bc', [False, True])
@pytest.mark.parametrize('mom_pres', [False, True])

def test_conf_projectors_2d(
        V1_type,
        degree,
        ncells, 
        nb_patches,
        reg,
        hom_bc,
        mom_pres,        
    ):

    domain, domain_h, bnds = build_multipatch_rectangle(
        nb_patches[0], nb_patches[1], 
        x_min=0, x_max=np.pi,
        y_min=0, y_max=np.pi,
        perio=[False,False],
        ncells=ncells,
        )

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = [m.get_callable_mapping() for m in mappings.values()]
    p_derham  = Derham(domain, ["H1", V1_type, "L2"])

    nquads = [(d + 1) for d in degree]
    p_derham_h = discretize(p_derham, domain_h, degree=degree, nquads=nquads)
    p_V0h = p_derham_h.V0
    p_V1h = p_derham_h.V1
    p_V2h = p_derham_h.V2

    # full moment preservation only possible if enough interior functions in a patch (<=> enough cells)
    full_mom_pres = mom_pres and (ncells[0] >= 3 + 2*reg) and (ncells[1] >= 3 + 2*reg)
    # NOTE: if mom_pres but not full_mom_pres we could test reduced order moment preservation...

    # geometric projections (operators)
    p_geomP0, p_geomP1, p_geomP2 = p_derham_h.projectors()

    # conforming projections (scipy matrices)
    cP0, cP1, cP2 = conf_projectors_scipy(p_derham_h, reg=reg, mom_pres=mom_pres, nquads=nquads, hom_bc=hom_bc)

    HOp0   = HodgeOperator(p_V0h, domain_h, nquads=nquads)
    M0     = HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
    M0_inv = HOp0.to_sparse_matrix()                # inverse mass matrix

    HOp1   = HodgeOperator(p_V1h, domain_h, nquads=nquads)
    M1     = HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
    M1_inv = HOp1.to_sparse_matrix()                # inverse mass matrix

    HOp2   = HodgeOperator(p_V2h, domain_h, nquads=nquads)
    M2     = HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix
    M2_inv = HOp2.to_sparse_matrix()                # inverse mass matrix

    bD0, bD1 = p_derham_h.broken_derivatives_as_operators
    
    bD0 = bD0.to_sparse_matrix() # broken grad
    bD1 = bD1.to_sparse_matrix() # broken curl or div
    D0  = bD0 @ cP0               # Conga grad
    D1  = bD1 @ cP1               # Conga curl or div

    np.allclose(sp_norm(cP0 - cP0@cP0), 0, 1e-12, 1e-12) # cP0 is a projection
    np.allclose(sp_norm(cP1 - cP1@cP1), 0, 1e-12, 1e-12) # cP1 is a projection
    np.allclose(sp_norm(cP2 - cP2@cP2), 0, 1e-12, 1e-12) # cP2 is a projection
    np.allclose(sp_norm( D0 - cP1@D0),  0, 1e-12, 1e-12) # D0 maps in the conforming V1 space (where cP1 coincides with Id)
    np.allclose(sp_norm( D1 - cP2@D1),  0, 1e-12, 1e-12) # D1 maps in the conforming V2 space (where cP2 coincides with Id)

    # comparing projections of polynomials which should be exact
    
    # tests on cP0:
    g0 = get_polynomial_function(degree=degree, hom_bc_axes=[hom_bc,hom_bc], domain=domain)        
    g0h = P_phys_h1(g0, p_geomP0, domain, mappings_list)
    g0_c = g0h.coeffs.toarray()  
    
    tilde_g0_c = p_derham_h.get_dual_dofs(space='V0', f=g0, return_format='numpy_array', nquads=nquads)
    g0_L2_c = M0_inv @ tilde_g0_c

    np.allclose(g0_c,     g0_L2_c, 1e-12, 1e-12) # (P0_geom - P0_L2) polynomial = 0
    np.allclose(g0_c, cP0@g0_L2_c, 1e-12, 1e-12) # (P0_geom - confP0 @ P0_L2) polynomial= 0
    
    if full_mom_pres:
        # testing that polynomial moments are preserved: 
        #   the following projection should be exact for polynomials of proper degree (no bc)
        #   conf_P0* : L2 -> V0 defined by <conf_P0* g, phi> := <g, conf_P0 phi> for all phi in V0            
        g0 = get_polynomial_function(degree=degree, hom_bc_axes=[False, False], domain=domain)
        g0h = P_phys_h1(g0, p_geomP0, domain, mappings_list)
        g0_c = g0h.coeffs.toarray()    

        tilde_g0_c = p_derham_h.get_dual_dofs(space='V0', f=g0, return_format='numpy_array', nquads=nquads)
        g0_star_c = M0_inv @ cP0.transpose() @ tilde_g0_c
        np.allclose(g0_c, g0_star_c, 1e-12, 1e-12)  #  (P10_geom - P0_star) polynomial = 0 

    # tests on cP1:

    G1 = Tuple(
        get_polynomial_function(degree=[degree[0]-1,degree[1]],   hom_bc_axes=[False,hom_bc], domain=domain),
        get_polynomial_function(degree=[degree[0],  degree[1]-1], hom_bc_axes=[hom_bc,False], domain=domain)
    )

    if V1_type == "Hcurl":
        G1h = P_phys_hcurl(G1, p_geomP1, domain, mappings_list)
    elif V1_type == "Hdiv":
        G1h = P_phys_hdiv(G1, p_geomP1, domain, mappings_list)
    G1_c = G1h.coeffs.toarray()  
    tilde_G1_c = p_derham_h.get_dual_dofs(space='V1', f=G1, return_format='numpy_array', nquads=nquads)
    G1_L2_c = M1_inv @ tilde_G1_c

    np.allclose(G1_c,       G1_L2_c, 1e-12, 1e-12)  # (P1_geom - P1_L2) polynomial = 0
    np.allclose(G1_c, cP1 @ G1_L2_c, 1e-12, 1e-12)  # (P1_geom - confP1 @ P1_L2) polynomial= 0

    if full_mom_pres:
        # as above
        G1 = Tuple(
            get_polynomial_function(degree=[degree[0]-1,degree[1]],   hom_bc_axes=[False,False], domain=domain),
            get_polynomial_function(degree=[degree[0],  degree[1]-1], hom_bc_axes=[False,False], domain=domain)
        )

        G1h = P_phys_hcurl(G1, p_geomP1, domain, mappings_list)
        G1_c = G1h.coeffs.toarray()  

        tilde_G1_c = p_derham_h.get_dual_dofs(space='V1', f=G1, return_format='numpy_array', nquads=nquads)
        G1_star_c = M1_inv @ cP1.transpose() @ tilde_G1_c
        np.allclose(G1_c, G1_star_c, 1e-12, 1e-12) # (P1_geom - P1_star) polynomial = 0 
    
    # tests on cP2 (non trivial for reg = 1):
    g2 = get_polynomial_function(degree=[degree[0]-1,degree[1]-1], hom_bc_axes=[False,False], domain=domain)        
    g2h = P_phys_l2(g2, p_geomP2, domain, mappings_list)
    g2_c = g2h.coeffs.toarray()  
    
    tilde_g2_c = p_derham_h.get_dual_dofs(space='V2', f=g2, return_format='numpy_array', nquads=nquads)
    g2_L2_c = M2_inv @ tilde_g2_c

    np.allclose(g2_c,       g2_L2_c, 1e-12, 1e-12) # (P2_geom - P2_L2) polynomial = 0
    np.allclose(g2_c, cP2 @ g2_L2_c, 1e-12, 1e-12) # (P2_geom - confP2 @ P2_L2) polynomial = 0

    if full_mom_pres:                
        # as above, here with same degree and bc as 
        # tilde_g2_c = p_derham_h.get_dual_dofs(space='V2', f=g2, return_format='numpy_array', nquads=nquads)
        g2_star_c = M2_inv @ cP2.transpose() @ tilde_g2_c
        np.allclose(g2_c, g2_star_c, 1e-12, 1e-12) # (P2_geom - P2_star) polynomial = 0

