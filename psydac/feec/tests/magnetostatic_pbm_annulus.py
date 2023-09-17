from dataclasses import dataclass
import logging
import numpy as np
import scipy
from scipy.sparse import bmat, csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs, spsolve
from scipy.sparse.linalg import inv
from scipy.sparse._lil import lil_matrix
from scipy.sparse._coo import coo_matrix
from scipy.sparse._csc import csc_matrix

from typing import Tuple

from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.fem  import DiscreteBilinearForm, DiscreteLinearForm
from psydac.cad.geometry     import Geometry
from psydac.feec.global_projectors import projection_matrix_H1_homogeneous_bc, projection_matrix_Hdiv_homogeneous_bc 
from psydac.fem.basic              import FemField


import sympy
from sympde.calculus      import dot
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.topology  import Derham
import sympde.topology as top


# from scipy.sparse._lil import lil_matrix

@dataclass
class _Matrices:
    """
    Matrices needed for solving the problem
    """
    M0 : coo_matrix = None
    M1 : coo_matrix = None
    M2 : coo_matrix = None
    P0h : lil_matrix = None
    P1h : lil_matrix = None
    D0 : coo_matrix = None
    D1 : coo_matrix = None
    D2 : coo_matrix = None
    D0_h : csr_matrix = None
    D1_h : csr_matrix = None
    S0_h_tilde : csc_matrix = None
    S1_h_tilde : csc_matrix = None

# def solve_magnetostatic_pbm_annulus(f : sympy.Tuple, 
#                                     psi_h : FemField, 
#                                     rhs_curve_integral: float,
#                                     derham_h: DiscreteDerham,
#                                     derham: Derham,
#                                     annulus_h: Geometry):
#     """
#     Solves the Hodge Laplacian problem with curve integral constraint on an annulus

#     It solves the Hodge Laplacian problem with right hand side f 
#     with the additional constraint corresponding to a curve integral (see ...)

#     Parameters
#     ----------
#     f : sympy.Tuple
#         Right hand side
#     psi_h : FemField
#         Interpolated integration function
#     rhs_curve_integral : float
#         Sum of inner product of psi with J plus curve integral
#     derham_h : DiscreteDerham#
#         Discretized de Rham sequence
#     derham : 
#         Symbolic de Rham sequence
#     annulus_h : Geometry
#         Discretized annulus domain
#     """
#     logger_magnetostatic = logging.getLogger(name='solve_magnetostatic_pbm_annulus')

#     # Compute the discrete differential operators, mass matrices and projection 
#     # matrices
#     matrices = _Matrices()
#     D0_block_operator, D1_block_operator = derham_h.derivatives_as_matrices
#     matrices.D0 = D0_block_operator.tosparse()
#     matrices.D1 = D1_block_operator.tosparse()
#     assert isinstance(matrices.D0, coo_matrix)
#     assert isinstance(matrices.D1, coo_matrix)
#     matrices.M0, matrices.M1, matrices.M2 = assemble_mass_matrices(derham, derham_h, annulus_h)
#     assert isinstance(matrices.M0, coo_matrix)
#     assert isinstance(matrices.M1, coo_matrix)
#     assert isinstance(matrices.M2 , coo_matrix)
#     matrices.P0h = projection_matrix_H1_homogeneous_bc(derham_h.V0)
#     matrices.P1h = projection_matrix_Hdiv_homogeneous_bc(derham_h.V1)
#     assert isinstance(matrices.P0h, lil_matrix)
#     assert isinstance(matrices.P1h, lil_matrix)

#     # Compute penalty matrix
#     alpha = 10
#     I1 = scipy.sparse.eye(derham_h.V1.nbasis, format='lil')
#     matrices.S1_h_tilde = (I1 - matrices.P1h).transpose() @ matrices.M1 @ (I1 - matrices.P1h)
#     matrices.D0_h = matrices.D0 @ matrices.P0h
#     matrices.D1_h = matrices.D1 @ matrices.P1h
#     assert isinstance(matrices.D0_h, csr_matrix), f"type(D0_h_mat):{type(matrices.D0_h)}"
#     assert isinstance(matrices.D1_h, csr_matrix)
#     assert isinstance(matrices.S1_h_tilde, csc_matrix), f"type(S1_h_tilde_mat):{type(matrices.S1_h_tilde)}"

#     harmonic_block = _assemble_harmonic_block(matrices=matrices, alpha=alpha)

#     # Compute part of right hand side
#     annulus = annulus_h.domain
#     u, v = top.elements_of(derham.V1, names='u v')
#     l_f = LinearForm(v, integral(annulus, dot(f,v)))
#     l_f_h = discretize(l_f, annulus_h, space=derham_h.V1)
#     assert isinstance(l_f_h, DiscreteLinearForm)
#     f_tilde_h_block : BlockVector = l_f_h.assemble()
#     f_tilde_h = f_tilde_h_block.toarray()
#     assert isinstance(f_tilde_h, np.ndarray)
#     f_h = matrices.P1h.transpose() @ f_tilde_h
#     assert isinstance(f_h, np.ndarray)

#     # Compute curl of the integration function psi
#     psi_h_coeffs = psi_h.coeffs.toarray()
#     assert isinstance(psi_h_coeffs, np.ndarray)
#     curl_psi_h_coeffs = matrices.D0 @ psi_h_coeffs
#     curl_psi_h_mat = csr_matrix(curl_psi_h_coeffs)
#     curl_psi_h_mat = curl_psi_h_mat.transpose()
#     logger_magnetostatic.debug('curl_psi_h_mat.shape:%s\n', curl_psi_h_mat.shape)

#     # Assemble the final system and solve
#     curve_integral_rhs_arr = np.array([rhs_curve_integral])
#     DD_tilde_mat = matrices.D1_h.transpose() @ matrices.M2  @ matrices.D1_h
#     A_mat = bmat([[matrices.M0 , -matrices.D0_h.transpose() @ matrices.M1, None],
#                   [matrices.M1 @ matrices.D0_h,  DD_tilde_mat  + alpha * matrices.S1_h_tilde, harmonic_block],
#                   [None, curl_psi_h_mat.transpose() @ matrices.M1 , None]])
#     rhs = np.concatenate((np.zeros(derham_h.V0.nbasis), f_h, curve_integral_rhs_arr))
#     A_mat = csr_matrix(A_mat) # Why is this not transforming into a CSR matrix?
#     logger_magnetostatic.debug('type(A_mat):%s\n', type(A_mat))
#     sol = spsolve(A_mat, rhs)
#     return sol[derham_h.V0.nbasis:-1]

def _assemble_harmonic_block(matrices: _Matrices, alpha) -> csc_matrix:
    """
    Returns the block for stiffness matrix corresponding to the inner product
    with a discrete harmonic form
    """
    summand1 = matrices.M1 @ matrices.D0_h @ inv(matrices.M0) @ matrices.D0_h.transpose() @ matrices.M1
    summand2 = matrices.D1_h.transpose() @ matrices.M2 @ matrices.D1_h
    summand3 = alpha * matrices.S1_h_tilde
    # Matrix representation of stabilized Hodge laplacian from primal to dual basis
    L_h = summand1 + summand2 + summand3
    eig_val, eig_vec = eigs(L_h, sigma=0.001)
    harmonic_form_coeffs = eig_vec[:,0]
    harmonic_block = csr_matrix( matrices.P1h.transpose() @ matrices.M1 @ harmonic_form_coeffs)
    harmonic_block = harmonic_block.transpose()
    return harmonic_block

def assemble_mass_matrices(derham: Derham, derham_h: DiscreteDerham, domain_h : Geometry
                           ) -> Tuple[coo_matrix]:
    """
    Assemble the mass matrices of the spaces a discrete de Rham sequence
    of length 2

    Parameters
    ----------
    derham : Derham
        Symbolic de Rham sequence of length 2
    derham_h : DiscreteDerham
        Discrete de Rham sequence of length 2
    domain_h : Geometry
        Discrete domain

    Returns
    -------
    Tuple[coo_matrix]
        Mass matrices in Scipy COO-format
    """
    # Define symbolic L2 inner products
    u, v = top.elements_of(derham.V1, names='u, v')
    rho, nu = top.elements_of(derham.V2, names='rho nu')
    annulus = domain_h.domain
    l2_inner_product_V2 = BilinearForm(arguments=(rho,nu), 
        expr=integral(annulus, rho*nu)
    )
    l2_inner_product_V1 = BilinearForm(arguments=(u,v), 
        expr=integral(annulus, dot(u,v))
    )
    sigma, tau = top.elements_of(derham.V0, names='sigma, tau')
    l2_inner_product_V0 = BilinearForm(arguments=(sigma,tau), 
        expr=integral(annulus, sigma*tau)
    )

    # Discretize the inner products and convert to scipy COO-matrix
    l2_inner_product_V2h = discretize(l2_inner_product_V2, domain_h, 
                                      spaces=[derham_h.V2, derham_h.V2])
    l2_inner_product_V1h = discretize(l2_inner_product_V1, domain_h, 
                                      spaces=[derham_h.V1, derham_h.V1])
    l2_inner_product_V0h = discretize(l2_inner_product_V0, domain_h, 
                                      spaces=[derham_h.V0, derham_h.V0])
    assert isinstance(l2_inner_product_V2h, DiscreteBilinearForm)
    assert isinstance(l2_inner_product_V1h, DiscreteBilinearForm)
    assert isinstance(l2_inner_product_V0h, DiscreteBilinearForm)
    M0 = l2_inner_product_V0h.assemble().tosparse().tocoo()
    M1 = l2_inner_product_V1h.assemble().tosparse().tocoo()
    M2 = l2_inner_product_V2h.assemble().tosparse().tocoo()
    return M0,M1,M2

# def assemble_mass_matrices_general(derham: Derham, derham_h: DiscreteDerham, domain_h : Geometry
#                            ) -> Tuple[coo_matrix]:
#     """
#     Assemble the mass matrices of the spaces a discrete de Rham sequence
#     of length 2 on an annulus

#     Parameters
#     ----------
#     derham : Derham
#         Symbolic de Rham sequence of length 2
#     derham_h : DiscreteDerham
#         Discrete de Rham sequence of length 2
#     ???
#     Returns
#     -------
#     Tuple[coo_matrix]
#         Mass matrices in Scipy COO-format
#     """
#     # Define symbolic L2 inner products
#     u, v = top.elements_of(derham.V1, names='u, v')
#     rho, nu = top.elements_of(derham.V2, names='rho nu')
#     domain = domain_h.domain
#     l2_inner_product_V2 = BilinearForm(arguments=(rho,nu), 
#         expr=integral(domain, rho*nu)
#     )
#     l2_inner_product_V1 = BilinearForm(arguments=(u,v), 
#         expr=integral(domain, dot(u,v))
#     )
#     sigma, tau = top.elements_of(derham.V0, names='sigma, tau')
#     l2_inner_product_V0 = BilinearForm(arguments=(sigma,tau), 
#         expr=integral(domain, sigma*tau)
#     )

#     # Discretize the inner products and convert to scipy COO-matrix
#     l2_inner_product_V2h = discretize(l2_inner_product_V2, domain_h, 
#                                       spaces=[derham_h.V2, derham_h.V2])
#     l2_inner_product_V1h = discretize(l2_inner_product_V1, domain_h, 
#                                       spaces=[derham_h.V1, derham_h.V1])
#     l2_inner_product_V0h = discretize(l2_inner_product_V0, domain_h, 
#                                       spaces=[derham_h.V0, derham_h.V0])
#     assert isinstance(l2_inner_product_V2h, DiscreteBilinearForm)
#     assert isinstance(l2_inner_product_V1h, DiscreteBilinearForm)
#     assert isinstance(l2_inner_product_V0h, DiscreteBilinearForm)
#     M0 = l2_inner_product_V0h.assemble().tosparse()
#     M1 = l2_inner_product_V1h.assemble().tosparse()
#     M2 = l2_inner_product_V2h.assemble().tosparse()
#     return M0,M1,M2


def solve_magnetostatic_pbm_J_direct_annulus(J : sympy.Expr, 
                                    psi_h : FemField, 
                                    rhs_curve_integral: float,
                                    derham_h: DiscreteDerham,
                                    derham: Derham,
                                    annulus_h: Geometry
                                    ) -> np.ndarray:
    """
    Solves the 2D magnetostatic problem with curve integral constraint on an annulus

    Parameters
    ----------
    J: Scalar valued current source
    psi_h: Discretized function for the curve integral constraint
    rhs_curve_integral: Right hands side for the curve integral constraint
    derham_h: Discretized curl-div de Rham sequence
    derham: curl-div de Rham sequence
    annulus_h: Discretized annulus domain

    Returns
    -------
    Coefficients of the approximated magnetic field

    References
    ----------
    See Alexander Hoffmann's masters thesis "The magnetostatic problem on 
    exterior domains" (2023)
    """

    # Compute the discrete differential operators, mass matrices and projection 
    # matrices
    matrices = _Matrices()
    D0_block_operator, D1_block_operator = derham_h.derivatives_as_matrices
    matrices.D0 = D0_block_operator.tosparse().tocoo()
    matrices.D1 = D1_block_operator.tosparse().tocoo()
    matrices.M0, matrices.M1, matrices.M2 = assemble_mass_matrices(derham, derham_h, annulus_h)
    matrices.P0h = projection_matrix_H1_homogeneous_bc(derham_h.V0)
    matrices.P1h = projection_matrix_Hdiv_homogeneous_bc(derham_h.V1)
    assert isinstance(matrices.P0h, lil_matrix)
    assert isinstance(matrices.P1h, lil_matrix)

    # Compute penalty matrix
    alpha = 10
    I0 = scipy.sparse.eye(derham_h.V0.nbasis, format='lil')
    I1 = scipy.sparse.eye(derham_h.V1.nbasis, format='lil')
    matrices.S0_h_tilde = (I0 - matrices.P0h).transpose() @ matrices.M0 @ (I0 - matrices.P0h)
    matrices.S1_h_tilde = (I1 - matrices.P1h).transpose() @ matrices.M1 @ (I1 - matrices.P1h)
    matrices.D0_h = matrices.D0 @ matrices.P0h
    matrices.D1_h = matrices.D1 @ matrices.P1h
    assert isinstance(matrices.D0_h, csr_matrix), f"type(D0_h_mat):{type(matrices.D0_h)}"
    assert isinstance(matrices.D1_h, csr_matrix)
    assert isinstance(matrices.S1_h_tilde, csc_matrix), f"type(S1_h_tilde_mat):{type(matrices.S1_h_tilde)}"

    harmonic_block = _assemble_harmonic_block(matrices=matrices, alpha=alpha)

    # Compute part of right hand side
    annulus = annulus_h.domain
    sigma, tau = top.elements_of(derham.V0, names='sigma, tau')
    l_J = LinearForm(tau, integral(annulus, J*tau))
    l_J_h = discretize(l_J, annulus_h, space=derham_h.V0)
    assert isinstance(l_J_h, DiscreteLinearForm)
    J_tilde_h_array = l_J_h.assemble().toarray()
    assert isinstance(J_tilde_h_array, np.ndarray)
    J_h_array = matrices.P0h.transpose() @ J_tilde_h_array
    assert isinstance(J_h_array, np.ndarray)

    # Compute curl of the integration function psi
    psi_h_coeffs = psi_h.coeffs.toarray()
    assert isinstance(psi_h_coeffs, np.ndarray)
    curl_psi_h_coeffs = matrices.D0 @ psi_h_coeffs
    curl_psi_h_mat = csr_matrix(curl_psi_h_coeffs)
    curl_psi_h_mat = curl_psi_h_mat.transpose()

    # Assemble the final system and solve
    curve_integral_rhs_arr = np.array([rhs_curve_integral])
    DD_tilde_mat = matrices.D1_h.transpose() @ matrices.M2  @ matrices.D1_h
    A_mat = bmat([[matrices.M0 +  alpha * matrices.S0_h_tilde, -matrices.D0_h.transpose() @ matrices.M1, None],
                  [matrices.M1 @ matrices.D0_h,  DD_tilde_mat  + alpha * matrices.S1_h_tilde, harmonic_block],
                  [None, curl_psi_h_mat.transpose() @ matrices.M1 , None]])
    rhs = np.concatenate((-J_h_array, np.zeros(derham_h.V1.nbasis), curve_integral_rhs_arr))
    A_mat = A_mat.tocsr()
    # logger = logging.getLogger(name='solve_magnetostatic_pbm_J_direct_annulus')
    # logger.debug('type(A_mat):%s\n', type(A_mat))
    sol = spsolve(A_mat, rhs)
    return sol[derham_h.V0.nbasis:-1]


def solve_magnetostatic_pbm_J_direct_with_bc(J : sympy.Expr, 
                                    psi_h : FemField, 
                                    rhs_curve_integral: float,
                                    boundary_data: FemField,
                                    derham_h: DiscreteDerham,
                                    derham: Derham,
                                    domain_h: Geometry
                                    ) -> np.ndarray:
    """
    Solves the 2D magnetostatic problem with curve integral constraint on an annulus
    with non-homogeneous boundary conditions

    Parameters
    ----------
    J: Scalar valued current source
    psi_h: Discretized function for the curve integral constraint
    rhs_curve_integral: Right hands side for the curve integral constraint
    derham_h: Discretized curl-div de Rham sequence
    derham: curl-div de Rham sequence
    annulus_h: Discretized annulus domain

    Returns
    -------
    Coefficients of the approximated magnetic field

    References
    ----------
    See Alexander Hoffmann's masters thesis "The magnetostatic problem on 
    exterior domains" (2023)
    """
    # Compute the discrete differential operators, mass matrices and projection 
    # matrices
    matrices = _Matrices()
    D0_block_operator, D1_block_operator = derham_h.derivatives_as_matrices
    matrices.D0 = D0_block_operator.tosparse()
    matrices.D1 = D1_block_operator.tosparse()
    assert isinstance(matrices.D0, coo_matrix)
    assert isinstance(matrices.D1, coo_matrix)
    matrices.M0, matrices.M1, matrices.M2 = assemble_mass_matrices(derham, derham_h, domain_h)
    # NOTE: assemble_mass_matrices not only for annulus
    assert isinstance(matrices.M0, coo_matrix)
    assert isinstance(matrices.M1, coo_matrix)
    assert isinstance(matrices.M2 , coo_matrix)
    matrices.P0h = projection_matrix_H1_homogeneous_bc(derham_h.V0)
    matrices.P1h = projection_matrix_Hdiv_homogeneous_bc(derham_h.V1)
    assert isinstance(matrices.P0h, lil_matrix)
    assert isinstance(matrices.P1h, lil_matrix)

    # Compute penalty matrix
    alpha = 10
    I0 = scipy.sparse.eye(derham_h.V0.nbasis, format='lil')
    I1 = scipy.sparse.eye(derham_h.V1.nbasis, format='lil')
    matrices.S0_h_tilde = (I0 - matrices.P0h).transpose() @ matrices.M0 @ (I0 - matrices.P0h)
    matrices.S1_h_tilde = (I1 - matrices.P1h).transpose() @ matrices.M1 @ (I1 - matrices.P1h)
    matrices.D0_h = matrices.D0 @ matrices.P0h
    matrices.D1_h = matrices.D1 @ matrices.P1h
    assert isinstance(matrices.D0_h, csr_matrix), f"type(D0_h_mat):{type(matrices.D0_h)}"
    assert isinstance(matrices.D1_h, csr_matrix)
    assert isinstance(matrices.S1_h_tilde, csc_matrix), f"type(S1_h_tilde_mat):{type(matrices.S1_h_tilde)}"

    harmonic_block = _assemble_harmonic_block(matrices=matrices, alpha=alpha)

    # Compute part of right hand side
    domain = domain_h.domain
    sigma, tau = top.elements_of(derham.V0, names='sigma, tau')
    l_J = LinearForm(tau, integral(domain, J*tau))
    l_J_h = discretize(l_J, domain_h, space=derham_h.V0)
    assert isinstance(l_J_h, DiscreteLinearForm)
    J_tilde_h_array = l_J_h.assemble().toarray()
    assert isinstance(J_tilde_h_array, np.ndarray)
    J_h_array = matrices.P0h.transpose() @ J_tilde_h_array
    assert isinstance(J_h_array, np.ndarray)

    # Get the boundary data for rhs
    boundary_data_coeffs = boundary_data.coeffs.toarray()
    assert isinstance(boundary_data_coeffs, np.ndarray)
    B_boundary_coeffs = (I1 - matrices.P1h) @ boundary_data_coeffs
    boundary_rhs = (-matrices.D1_h.transpose() @ matrices.M2 @ matrices.D1 @ B_boundary_coeffs)

    # Compute curl of the integration function psi
    psi_h_coeffs = psi_h.coeffs.toarray()
    assert isinstance(psi_h_coeffs, np.ndarray)
    curl_psi_h_coeffs = matrices.D0 @ psi_h_coeffs
    curl_psi_h_mat = csr_matrix(curl_psi_h_coeffs)
    curl_psi_h_mat = curl_psi_h_mat.transpose()

    # Assemble the final system and solve
    curve_integral_rhs_arr = np.array([rhs_curve_integral])
    DD_tilde_mat = matrices.D1_h.transpose() @ matrices.M2  @ matrices.D1_h
    A_mat = bmat([[matrices.P0h.transpose() @ matrices.M0 @ matrices.P0h +  alpha * matrices.S0_h_tilde, -matrices.D0_h.transpose() @ matrices.M1 @ matrices.P1h, None],
                  [matrices.P1h.transpose() @ matrices.M1 @ matrices.D0_h,  DD_tilde_mat  + alpha * matrices.S1_h_tilde, matrices.P1h.transpose() @ harmonic_block],
                  [None, curl_psi_h_mat.transpose() @ matrices.M1 @ matrices.P1h, None]])
    rhs = np.concatenate((-J_h_array + matrices.D0_h.transpose() @ matrices.M1 @ B_boundary_coeffs, 
                          boundary_rhs, 
                          curve_integral_rhs_arr - curl_psi_h_mat.transpose() @ matrices.M1 @ B_boundary_coeffs))
    A_mat = A_mat.tocsr() 
    sol = spsolve(A_mat, rhs)
    B_0 = sol[derham_h.V0.nbasis:-1]
    B = B_0 + B_boundary_coeffs
    return B

# def solve_magnetostatic_pbm_distorted_annulus(J : sympy.Expr, 
#                                     psi_h : FemField, 
#                                     rhs_curve_integral: float,
#                                     boundary_data: FemField,
#                                     derham_h: DiscreteDerham,
#                                     derham: Derham,
#                                     domain_h: Geometry):
#     """
#     """
#     # Compute the discrete differential operators, mass matrices and projection 
#     # matrices
#     matrices = _Matrices()
#     D0_block_operator, D1_block_operator = derham_h.derivatives_as_matrices
#     matrices.D0 = D0_block_operator.tosparse()
#     matrices.D1 = D1_block_operator.tosparse()
#     assert isinstance(matrices.D0, coo_matrix)
#     assert isinstance(matrices.D1, coo_matrix)
#     matrices.M0, matrices.M1, matrices.M2 = assemble_mass_matrices_general(derham, derham_h, domain_h)
#     assert isinstance(matrices.M0, coo_matrix)
#     assert isinstance(matrices.M1, coo_matrix)
#     assert isinstance(matrices.M2 , coo_matrix)
#     matrices.P0h = projection_matrix_H1_homogeneous_bc(derham_h.V0)
#     matrices.P1h = projection_matrix_Hdiv_homogeneous_bc(derham_h.V1)
#     assert isinstance(matrices.P0h, lil_matrix)
#     assert isinstance(matrices.P1h, lil_matrix)

#     # Compute penalty matrix
#     alpha = 10
#     I0 = scipy.sparse.eye(derham_h.V0.nbasis, format='lil')
#     I1 = scipy.sparse.eye(derham_h.V1.nbasis, format='lil')
#     matrices.S0_h_tilde = (I0 - matrices.P0h).transpose() @ matrices.M0 @ (I0 - matrices.P0h)
#     matrices.S1_h_tilde = (I1 - matrices.P1h).transpose() @ matrices.M1 @ (I1 - matrices.P1h)
#     matrices.D0_h = matrices.D0 @ matrices.P0h
#     matrices.D1_h = matrices.D1 @ matrices.P1h
#     assert isinstance(matrices.D0_h, csr_matrix), f"type(D0_h_mat):{type(matrices.D0_h)}"
#     assert isinstance(matrices.D1_h, csr_matrix)
#     assert isinstance(matrices.S0_h_tilde, csc_matrix), f"type(S0_h_tilde_mat):{type(matrices.S0_h_tilde)}"
#     assert isinstance(matrices.S1_h_tilde, csc_matrix), f"type(S1_h_tilde_mat):{type(matrices.S1_h_tilde)}"

#     harmonic_block = _assemble_harmonic_block(matrices=matrices, alpha=alpha)

#     # Compute part of right hand side
#     domain = domain_h.domain
#     sigma, tau = top.elements_of(derham.V0, names='sigma, tau')
#     l_J = LinearForm(tau, integral(domain, J*tau))
#     l_J_h = discretize(l_J, domain_h, space=derham_h.V0)
#     assert isinstance(l_J_h, DiscreteLinearForm)
#     J_tilde_h_array = l_J_h.assemble().toarray()
#     assert isinstance(J_tilde_h_array, np.ndarray)
#     J_h_array = matrices.P0h.transpose() @ J_tilde_h_array
#     assert isinstance(J_h_array, np.ndarray)

#     # Get the boundary data for rhs
#     boundary_data_coeffs = boundary_data.coeffs.toarray()
#     assert isinstance(boundary_data_coeffs, np.ndarray)
#     B_boundary_coeffs = (I1 - matrices.P1h) @ boundary_data_coeffs
#     boundary_rhs = (-matrices.D1_h.transpose() @ matrices.M2 @ matrices.D1 @ B_boundary_coeffs)

#     # Compute curl of the integration function psi
#     psi_h_coeffs = psi_h.coeffs.toarray()
#     assert isinstance(psi_h_coeffs, np.ndarray)
#     curl_psi_h_coeffs = matrices.D0 @ psi_h_coeffs
#     curl_psi_h_mat = csr_matrix(curl_psi_h_coeffs)
#     curl_psi_h_mat = curl_psi_h_mat.transpose()

#     # Assemble the final system and solve
#     curve_integral_rhs_arr = np.array([rhs_curve_integral])
#     DD_tilde_mat = matrices.D1_h.transpose() @ matrices.M2  @ matrices.D1_h
#     A_mat = bmat([[matrices.P0h.transpose() @ matrices.M0 @ matrices.P0h +  alpha * matrices.S0_h_tilde, -matrices.D0_h.transpose() @ matrices.M1 @ matrices.P1h, None],
#                   [matrices.P1h.transpose() @ matrices.M1 @ matrices.D0_h,  DD_tilde_mat  + alpha * matrices.S1_h_tilde, matrices.P1h.transpose() @ harmonic_block],
#                   [None, curl_psi_h_mat.transpose() @ matrices.M1 @ matrices.P1h, None]])
#     rhs = np.concatenate((-J_h_array + matrices.D0_h.transpose() @ matrices.M1 @ B_boundary_coeffs, 
#                           boundary_rhs, 
#                           curve_integral_rhs_arr - curl_psi_h_mat.transpose() @ matrices.M1 @ B_boundary_coeffs))
#     A_mat = csr_matrix(A_mat) # Why is this not transforming into a CSR matrix?
#     sol = spsolve(A_mat, rhs)
#     B_0 = sol[derham_h.V0.nbasis:-1]
#     B = B_0 + B_boundary_coeffs
#     return B

if __name__ == "__main__":
    logging.basicConfig(filename='mydebug.log', level=logging.DEBUG, filemode='w')
    log_main = logging.getLogger(name='main')
    f = sympy.Tuple(1e-10, 1e-10)
    log_main.debug('f:%s %s', type(f), f)
    solve_magnetostatic_pbm_annulus(f=f, J=1e-10)