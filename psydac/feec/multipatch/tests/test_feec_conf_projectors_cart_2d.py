import numpy as np
import pytest

from collections import OrderedDict
from sympde.topology import Derham, Square
from sympde.topology import IdentityMapping
from sympde.topology import Boundary, Interface, Union
from scipy.sparse.linalg                import norm as sp_norm
from sympy                              import Tuple
from sympde.topology                    import Derham
from psydac.feec.multipatch.api         import discretize
from psydac.feec.multipatch.operators   import HodgeOperator
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain, create_domain

from psydac.feec.multipatch.non_matching_operators      import construct_scalar_conforming_projection, construct_vector_conforming_projection

from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_rectangle, build_multipatch_domain
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
        g0_x = (x-0.25)#**degree[0]

    if hom_bc_axes[1]:                
        assert degree[1] > 1
        g0_y = y * (y-np.pi) * (y-0.324)**(degree[1]-2)
    else:
        # if degree[1] > 1:
        #     g0_y = (y-1.675)**2 * (y-0.324)**(degree[1]-2)

        # else:
        g0_y = (y-0.75)#**degree[1]

    return g0_x * g0_y

#==============================================================================
@pytest.mark.parametrize('V1_type', ["Hcurl"])
@pytest.mark.parametrize('degree', [[3,3]])
@pytest.mark.parametrize('nc', [4])
@pytest.mark.parametrize('reg', [[0,0]])
@pytest.mark.parametrize('hom_bc', [[False, False]])
@pytest.mark.parametrize('mom_pres', [[-1, -1]])
@pytest.mark.parametrize('domain_name', ["4patch_nc"])
@pytest.mark.parametrize('nonconforming', [True])


def test_conf_projectors_2d(
        V1_type,
        degree,
        nc, 
        reg,
        hom_bc,
        mom_pres,
        domain_name,  
        nonconforming      
    ):

    nquads=None
    print(' .. multi-patch domain...')


    if domain_name == '2patch_nc_mapped':

        A = Square('A', bounds1=(0.5, 1), bounds2=(0,       np.pi/2))
        B = Square('B', bounds1=(0.5, 1), bounds2=(np.pi/2, np.pi))
        M1 = PolarMapping('M1', 2, c1=0, c2=0, rmin=0., rmax=1.)
        M2 = PolarMapping('M2', 2, c1=0, c2=0, rmin=0., rmax=1.)
        A = M1(A)
        B = M2(B)

        domain = create_domain([A, B], [[A.get_boundary(axis=1, ext=1), B.get_boundary(axis=1, ext=-1), 1]], name='domain')

    elif domain_name == '2patch_nc':

        A = Square('A', bounds1=(0, 0.5), bounds2=(0, 1))
        B = Square('B', bounds1=(0.5, 1.), bounds2=(0, 1))
        M1 = IdentityMapping('M1', dim=2)
        M2 = IdentityMapping('M2', dim=2)
        A = M1(A)
        B = M2(B)

        domain = create_domain([A, B], [[A.get_boundary(axis=0, ext=1), B.get_boundary(axis=0, ext=-1), 1]], name='domain')

    elif domain_name == '4patch_nc':

        A = Square('A', bounds1=(0, 0.5), bounds2=(0, 0.5))
        B = Square('B', bounds1=(0.5, 1.), bounds2=(0, 0.5))
        C = Square('C', bounds1=(0, 0.5), bounds2=(0.5, 1))
        D = Square('D', bounds1=(0.5, 1.), bounds2=(0.5, 1))
        M1 = IdentityMapping('M1', dim=2)
        M2 = IdentityMapping('M2', dim=2)
        M3 = IdentityMapping('M3', dim=2)
        M4 = IdentityMapping('M4', dim=2)
        A = M1(A)
        B = M2(B)
        C = M3(C)
        D = M4(D)

        domain = create_domain([A, B, C, D], [[A.get_boundary(axis=0, ext=1), B.get_boundary(axis=0, ext=-1), 1], 
                                            [A.get_boundary(axis=1, ext=1), C.get_boundary(axis=1, ext=-1), 1],
                                            [C.get_boundary(axis=0, ext=1), D.get_boundary(axis=0, ext=-1), 1],
                                            [B.get_boundary(axis=1, ext=1), D.get_boundary(axis=1, ext=-1), 1] ], name='domain')
    else:
        domain = build_multipatch_domain(domain_name=domain_name)

    n_patches = len(domain)

    def levelof(k):
        # some random refinement level (1 or 2 here)
        return 1+((2*k) % 3) % 2

    if nonconforming:
        if len(domain) == 1:
            ncells_h = {
                'M1(A)': [nc, nc],
            }

        elif len(domain) == 2:
            ncells_h = {
                'M1(A)': [nc, nc],
                'M2(B)': [2*nc, 2*nc],
            }
        elif len(domain) == 4:
            ncells_h = {
                'M1(A)': [nc, nc],
                'M2(B)': [2*nc, 2*nc],
                'M3(C)': [2*nc, 2*nc],
                'M4(D)': [4*nc, 4*nc],
            }
        else:
            ncells_h = {}
            for k, D in enumerate(domain.interior):
                print(k, D.name)
                ncells_h[D.name] = [2**k *nc, 2**k * nc]
    else:
        ncells_h = {}
        for k, D in enumerate(domain.interior):
            ncells_h[D.name] = [nc, nc]

    print('ncells_h = ', ncells_h)
    backend_language = 'python'

    print(' .. derham sequence...')
    derham = Derham(domain, ["H1", "Hcurl", "L2"])

    print(ncells_h)

    domain_h = discretize(domain, ncells=ncells_h)   # Vh space
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = [m.get_callable_mapping() for m in mappings.values()]
    p_derham  = Derham(domain, ["H1", V1_type, "L2"])

    nquads = [(d + 1) for d in degree]
    p_derham_h = discretize(p_derham, domain_h, degree=degree, nquads=nquads)
    p_V0h = p_derham_h.V0
    p_V1h = p_derham_h.V1
    p_V2h = p_derham_h.V2

    # full moment preservation only possible if enough interior functions in a patch (<=> enough cells)
    full_mom_pres = (mom_pres[0] >= degree[0] and mom_pres[1] >= degree[1]) and (nc >= 3 + 2*reg[0]) and (nc >= 3 + 2*reg[1])
    # NOTE: if mom_pres but not full_mom_pres we could test reduced order moment preservation...

    # geometric projections (operators)
    p_geomP0, p_geomP1, p_geomP2 = p_derham_h.projectors()

    # conforming projections (scipy matrices)
    cP0 = construct_scalar_conforming_projection(V0h, reg, mom_pres, nquads, hom_bc)
    cP1 = construct_vector_conforming_projection(V1h, reg, mom_pres, nquads, hom_bc)
    cP2 = construct_scalar_conforming_projection(V2h, [reg[0]- 1, reg[1]-1], mom_pres, nquads, hom_bc)

    HOp0   = HodgeOperator(p_V0h, domain_h)
    M0     = HOp0.get_dual_Hodge_sparse_matrix()    # mass matrix
    M0_inv = HOp0.to_sparse_matrix()                # inverse mass matrix

    HOp1   = HodgeOperator(p_V1h, domain_h)
    M1     = HOp1.get_dual_Hodge_sparse_matrix()    # mass matrix
    M1_inv = HOp1.to_sparse_matrix()                # inverse mass matrix

    HOp2   = HodgeOperator(p_V2h, domain_h)
    M2     = HOp2.get_dual_Hodge_sparse_matrix()    # mass matrix
    M2_inv = HOp2.to_sparse_matrix()                # inverse mass matrix

    bD0, bD1 = p_derham_h.broken_derivatives_as_operators
    
    bD0 = bD0.to_sparse_matrix() # broken grad
    bD1 = bD1.to_sparse_matrix() # broken curl or div
    D0  = bD0 @ cP0               # Conga grad
    D1  = bD1 @ cP1               # Conga curl or div

    np.allclose(sp_norm(cP0 - cP0@cP0), 0, 1e-12, 1e-12) # cP0 is a projection
    print(sp_norm(cP0 - cP0@cP0))
    np.allclose(sp_norm(cP1 - cP1@cP1), 0, 1e-12, 1e-12) # cP1 is a projection
    print(sp_norm(cP1 - cP1@cP1))
    np.allclose(sp_norm(cP2 - cP2@cP2), 0, 1e-12, 1e-12) # cP2 is a projection
    print(sp_norm(cP2 - cP2@cP2))

    np.allclose(sp_norm( D0 - cP1@D0),  0, 1e-12, 1e-12) # D0 maps in the conforming V1 space (where cP1 coincides with Id)
    print(sp_norm( D0 - cP1@D0))
    np.allclose(sp_norm( D1 - cP2@D1),  0, 1e-12, 1e-12) # D1 maps in the conforming V2 space (where cP2 coincides with Id)
    print(sp_norm( D1 - cP2@D1))

    # comparing projections of polynomials which should be exact
    
    # tests on cP0:
    g0 = get_polynomial_function(degree=degree, hom_bc_axes=[hom_bc,hom_bc], domain=domain)        
    g0h = P_phys_h1(g0, p_geomP0, domain, mappings_list)
    g0_c = g0h.coeffs.toarray()  
    
    tilde_g0_c = p_derham_h.get_dual_dofs(space='V0', f=g0, return_format='numpy_array')
    g0_L2_c = M0_inv @ tilde_g0_c

    np.allclose(g0_c,     g0_L2_c, 1e-12, 1e-12) # (P0_geom - P0_L2) polynomial = 0
    np.allclose(g0_c, cP0@g0_L2_c, 1e-12, 1e-12) # (P0_geom - confP0 @ P0_L2) polynomial= 0
    print(np.linalg.norm(g0_c-    g0_L2_c))
    print(np.linalg.norm(g0_c-    cP0@g0_L2_c))
    if full_mom_pres:
        # testing that polynomial moments are preserved: 
        #   the following projection should be exact for polynomials of proper degree (no bc)
        #   conf_P0* : L2 -> V0 defined by <conf_P0* g, phi> := <g, conf_P0 phi> for all phi in V0            
        g0 = get_polynomial_function(degree=degree, hom_bc_axes=[False, False], domain=domain)
        g0h = P_phys_h1(g0, p_geomP0, domain, mappings_list)
        g0_c = g0h.coeffs.toarray()    

        tilde_g0_c = p_derham_h.get_dual_dofs(space='V0', f=g0, return_format='numpy_array')
        g0_star_c = M0_inv @ cP0.transpose() @ tilde_g0_c
        np.allclose(g0_c, g0_star_c, 1e-12, 1e-12)  #  (P10_geom - P0_star) polynomial = 0 
        print(np.linalg.norm(g0_c- g0_star_c))
        
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
    tilde_G1_c = p_derham_h.get_dual_dofs(space='V1', f=G1, return_format='numpy_array')
    G1_L2_c = M1_inv @ tilde_G1_c

    np.allclose(G1_c,       G1_L2_c, 1e-12, 1e-12)  
    print(np.linalg.norm(G1_c-      G1_L2_c))# (P1_geom - P1_L2) polynomial = 0
    np.allclose(G1_c, cP1 @ G1_L2_c, 1e-12, 1e-12)  # (P1_geom - confP1 @ P1_L2) polynomial= 0
    print(np.linalg.norm(G1_c-      cP1 @ G1_L2_c))


    if full_mom_pres:
        # as above
        G1 = Tuple(
            get_polynomial_function(degree=[degree[0]-1,degree[1]],   hom_bc_axes=[False,False], domain=domain),
            get_polynomial_function(degree=[degree[0],  degree[1]-1], hom_bc_axes=[False,False], domain=domain)
        )

        G1h = P_phys_hcurl(G1, p_geomP1, domain, mappings_list)
        G1_c = G1h.coeffs.toarray()  

        tilde_G1_c = p_derham_h.get_dual_dofs(space='V1', f=G1, return_format='numpy_array')
        G1_star_c = M1_inv @ cP1.transpose() @ tilde_G1_c
        np.allclose(G1_c, G1_star_c, 1e-12, 1e-12) # (P1_geom - P1_star) polynomial = 0 
        print(np.linalg.norm(G1_c- G1_star_c))

    # tests on cP2 (non trivial for reg = 1):
    g2 = get_polynomial_function(degree=[degree[0]-1,degree[1]-1], hom_bc_axes=[False,False], domain=domain)        
    g2h = P_phys_l2(g2, p_geomP2, domain, mappings_list)
    g2_c = g2h.coeffs.toarray()  

    tilde_g2_c = p_derham_h.get_dual_dofs(space='V2', f=g2, return_format='numpy_array')
    g2_L2_c = M2_inv @ tilde_g2_c

    np.allclose(g2_c,       g2_L2_c, 1e-12, 1e-12) # (P2_geom - P2_L2) polynomial = 0
    np.allclose(g2_c, cP2 @ g2_L2_c, 1e-12, 1e-12) # (P2_geom - confP2 @ P2_L2) polynomial = 0

    if full_mom_pres:                
        # as above, here with same degree and bc as 
        # tilde_g2_c = p_derham_h.get_dual_dofs(space='V2', f=g2, return_format='numpy_array', nquads=nquads)
        g2_star_c = M2_inv @ cP2.transpose() @ tilde_g2_c
        np.allclose(g2_c, g2_star_c, 1e-12, 1e-12) # (P2_geom - P2_star) polynomial = 0

# if __name__ == '__main__':
#     V1_type = "Hcurl"
#     nc = 7
#     deg = 3
#     nonconforming = False

#     degree = [deg, deg]
#     reg=[0,0]
#     mom_pres=[4,4]
#     hom_bc = [False, False]
    
#     # domain_name = 'square_6'
#     # domain_name = 'curved_L_shape'
#     # domain_name = '2patch_nc_mapped'
#     domain_name = '2patch_nc'

#     test_conf_projectors_2d(
#         V1_type,
#         degree,
#         nc, 
#         reg,
#         hom_bc,
#         mom_pres,   
#         domain_name,
#         nonconforming    
#     )