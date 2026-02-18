#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
from collections import OrderedDict

import numpy as np
from sympy import Tuple, lambdify
from scipy.sparse.linalg import norm as sp_norm
from scipy.sparse.linalg import inv

from sympde.topology.domain import Domain
from sympde.topology import Derham, Square, IdentityMapping

from psydac.api.discretization import discretize

from psydac.fem.projectors import get_dual_dofs


def get_polynomial_function(degree, hom_bc_axes, domain):
    """Return a polynomial function of given degree and homogeneus boundary conditions on the domain."""

    x, y = domain.coordinates
    if hom_bc_axes[0]:
        assert degree[0] > 1
        g0_x = x * (x - 1) * (x - 1.554)**(degree[0] - 2)
    else:
        # if degree[0] > 1:
        #     g0_x = (x-0.543)**2 * (x-1.554)**(degree[0]-2)
        # else:
        g0_x = (x - 0.25)**degree[0]

    if hom_bc_axes[1]:
        assert degree[1] > 1
        g0_y = y * (y - 1) * (y - 0.324)**(degree[1] - 2)
    else:
        # if degree[1] > 1:
        #     g0_y = (y-1.675)**2 * (y-0.324)**(degree[1]-2)

        # else:
        g0_y = (y - 0.75)**degree[1]

    expr = g0_x * g0_y
    callable_function = lambdify(domain.coordinates, expr)

    return expr, callable_function


# ==============================================================================
@pytest.mark.parametrize('V1_type', ["Hcurl"])
@pytest.mark.parametrize('degree', [[3, 3]])
@pytest.mark.parametrize('nc', [5])
@pytest.mark.parametrize('reg', [0])
@pytest.mark.parametrize('hom_bc', [False, True])
@pytest.mark.parametrize('domain_name', ["1patch", "4patch_nc", "2patch_nc"])
@pytest.mark.parametrize("nonconforming, full_mom_pres",
                         [(True, True), (False, True)])

def test_conf_projectors_2d(
    V1_type,
    degree,
    nc,
    reg,
    hom_bc,
    full_mom_pres,
    domain_name,
    nonconforming
):
    if domain_name == '1patch':
        log_domain = Square('Omega', bounds1=(0, 1), bounds2=(0, 1))
        mapping = IdentityMapping('M1', dim=2)
        domain = mapping(log_domain)

    elif domain_name == '2patch_nc':

        A = Square('A', bounds1=(0, 0.5), bounds2=(0, 1))
        B = Square('B', bounds1=(0.5, 1.), bounds2=(0, 1))
        M1 = IdentityMapping('M1', dim=2)
        M2 = IdentityMapping('M2', dim=2)
        A = M1(A)
        B = M2(B)

        domain = Domain.join(patches=[A, B],
                             connectivity=[((0, 0, 1), (1, 0, -1), 1)],
                             name='domain')

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

        domain = Domain.join(patches=[A, B, C, D],
                             connectivity=[((0, 0, 1), (1, 0, -1), 1),
                                           ((2, 0, 1), (3, 0, -1), 1),
                                           ((0, 1, 1), (2, 1, -1), 1),
                                           ((1, 1, 1), (3, 1, -1), 1)],
                             name='domain')
    if domain_name == '1patch': 
        ncells_h = {domain.name: [nc, nc]}

    elif nonconforming:
        if len(domain) == 2:
            ncells_h = {
                'M1(A)': [nc, nc],
                'M2(B)': [2 * nc, 2 * nc],
            }
        elif len(domain) == 4:
            ncells_h = {
                'M1(A)': [nc, nc],
                'M2(B)': [nc, nc],
                'M3(C)': [2 * nc, 2 * nc],
                'M4(D)': [4 * nc, 4 * nc],
            }

    else:
        ncells_h = {}
        for k, D in enumerate(domain.interior):
            ncells_h[D.name] = [nc, nc]


    domain_h = discretize(domain, ncells=ncells_h)   # Vh space
    derham = Derham(domain, ["H1", V1_type, "L2"])

    nquads = [(d + 1) for d in degree]
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h, V1h, V2h = derham_h.spaces

    if full_mom_pres:
        # full moment preservation only possible if enough interior functions in a
        # patch (<=> enough cells)
        if nc >= degree[0] + 1:
            mom_pres = degree[0]
        else:
            raise ValueError(f'nc = {nc} too small for full moment preservation')
    else:
        mom_pres = -1
        # NOTE: if mom_pres but not full_mom_pres we could test reduced order
        # moment preservation...

    # geometric projections (operators)
    geomP0, geomP1, geomP2 = derham_h.projectors(nquads=nquads)

    # conforming projections (scipy matrices)
    cP0, cP1, cP2 = derham_h.conforming_projectors(p_moments=mom_pres, hom_bc=hom_bc)
    cP0, cP1, cP2 = (m.tosparse() for m in (cP0, cP1, cP2))
    
    M0, M1, M2 = derham_h.hodge_operators()
    M0, M1, M2 = (m.tosparse().tocsc() for m in (M0, M1, M2))

    M0_inv, M1_inv, M2_inv = (inv(m) for m in (M0, M1, M2))

    bD0, bD1 = derham_h.derivatives()
    bD0, bD1 = (m.tosparse() for m in (bD0, bD1))

    D0 = bD0 @ cP0              # Conga grad
    D1 = bD1 @ cP1              # Conga curl or div

    assert np.allclose(sp_norm(cP0 - cP0 @ cP0), 0, 1e-12, 1e-12)  # cP0 is a projection
    assert np.allclose(sp_norm(cP1 - cP1 @ cP1), 0, 1e-12, 1e-12)  # cP1 is a projection
    assert np.allclose(sp_norm(cP2 - cP2 @ cP2), 0, 1e-12, 1e-12)  # cP2 is a projection

    # D0 maps in the conforming V1 space (where cP1 coincides with Id)
    assert np.allclose(sp_norm(D0 - cP1 @ D0), 0, 1e-12, 1e-12)
    # D1 maps in the conforming V2 space (where cP2 coincides with Id)
    assert np.allclose(sp_norm(D1 - cP2 @ D1), 0, 1e-12, 1e-12)

    # comparing projections of polynomials which should be exact

    # tests on cP0:
    g0, g0_fun = get_polynomial_function(degree=degree, hom_bc_axes=[hom_bc, hom_bc], domain=domain)
    g0h = geomP0(g0_fun)
    g0_c = g0h.coeffs.toarray()

    tilde_g0_c = get_dual_dofs(Vh=V0h, f=g0, domain_h = domain_h, return_format='numpy_array')
    g0_L2_c = M0_inv @ tilde_g0_c

    # (P0_geom - P0_L2) polynomial = 0
    assert np.allclose(g0_c, g0_L2_c, 1e-12, 1e-12)
    # (P0_geom - confP0 @ P0_L2) polynomial= 0
    assert np.allclose(g0_c, cP0 @ g0_L2_c, 1e-12, 1e-12)

    if full_mom_pres:
        # testing that polynomial moments are preserved:
        #  the following projection should be exact for polynomials of proper degree (no bc)
        # conf_P0* : L2 -> V0 defined by <conf_P0* g, phi> := <g, conf_P0 phi>
        # for all phi in V0
        g0, g0_fun = get_polynomial_function(degree=degree, hom_bc_axes=[False, False], domain=domain)
        g0h = geomP0(g0_fun)
        g0_c = g0h.coeffs.toarray()

        tilde_g0_c = get_dual_dofs(Vh=V0h, f=g0, domain_h = domain_h, return_format='numpy_array')

        g0_star_c = M0_inv @ cP0.transpose() @ tilde_g0_c
        # (P10_geom - P0_star) polynomial = 0
        assert np.allclose(g0_c, g0_star_c, 1e-12, 1e-12)

    # tests on cP1:
    G1_x, G1_x_fun = get_polynomial_function(degree=[degree[0] - 1,degree[1]], hom_bc_axes=[False, hom_bc], domain=domain)
    G1_y, G1_y_fun = get_polynomial_function(degree=[degree[0], degree[1] - 1], hom_bc_axes=[hom_bc, False], domain=domain)
    
    G1 = Tuple(G1_x, G1_y)
    G1_fun = [G1_x_fun, G1_y_fun]

    G1h = geomP1(G1_fun)
    G1_c = G1h.coeffs.toarray()
    tilde_G1_c = get_dual_dofs(Vh=V1h, f=G1, domain_h=domain_h, return_format='numpy_array')

    G1_L2_c = M1_inv @ tilde_G1_c

    assert np.allclose(G1_c, G1_L2_c, 1e-12, 1e-12)
    # (P1_geom - confP1 @ P1_L2) polynomial= 0
    assert np.allclose(G1_c, cP1 @ G1_L2_c, 1e-12, 1e-12)

    if full_mom_pres:
        # as above
        G1_x, G1_x_fun = get_polynomial_function(degree=[degree[0] - 1,degree[1]], hom_bc_axes=[False, False], domain=domain)
        G1_y, G1_y_fun = get_polynomial_function(degree=[degree[0], degree[1] - 1], hom_bc_axes=[False, False], domain=domain)
        
        G1 = Tuple(G1_x, G1_y)
        G1_fun = [G1_x_fun, G1_y_fun]

        G1h = geomP1(G1_fun)
        G1_c = G1h.coeffs.toarray()
        
        G1h = geomP1(G1_fun)
        G1_c = G1h.coeffs.toarray()

        tilde_G1_c = get_dual_dofs(Vh=V1h, f=G1, domain_h=domain_h, return_format='numpy_array')
        G1_star_c = M1_inv @ cP1.transpose() @ tilde_G1_c
        # (P1_geom - P1_star) polynomial = 0
        assert np.allclose(G1_c, G1_star_c, 1e-12, 1e-12)

    # tests on cP2 (non trivial for reg = 1):
    g2, g2_fun = get_polynomial_function(degree=[degree[0] - 1, degree[1] - 1], hom_bc_axes=[False, False], domain=domain)
    g2h = geomP2(g2_fun)
    g2_c = g2h.coeffs.toarray()

    tilde_g2_c = get_dual_dofs(Vh=V2h, f=g2, domain_h=domain_h, return_format='numpy_array')
    g2_L2_c = M2_inv @ tilde_g2_c

    # (P2_geom - P2_L2) polynomial = 0
    assert np.allclose(g2_c, g2_L2_c, 1e-12, 1e-12)
    # (P2_geom - confP2 @ P2_L2) polynomial = 0
    assert np.allclose(g2_c, cP2 @ g2_L2_c, 1e-12, 1e-12)

    if full_mom_pres:
        # as above, here with same degree and bc as
        g2_star_c = M2_inv @ cP2.transpose() @ tilde_g2_c
        # (P2_geom - P2_star) polynomial = 0
        assert np.allclose(g2_c, g2_star_c, 1e-12, 1e-12)
