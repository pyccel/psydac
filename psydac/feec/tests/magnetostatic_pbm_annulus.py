import numpy as np
import pytest
import logging
from dataclasses import dataclass

from psydac.core.bsplines          import make_knots
from psydac.fem.basic              import FemField
from psydac.fem.splines            import SplineSpace
from psydac.fem.tensor             import TensorFemSpace
from psydac.feec.derivatives       import VectorCurl_2D, Divergence_2D
from psydac.feec.global_projectors import Projector_H1, Projector_L2
from psydac.feec.global_projectors import projection_matrix_H1_homogeneous_bc, projection_matrix_Hdiv_homogeneous_bc 
from psydac.ddm.cart               import DomainDecomposition


import numpy as np
import sympy
from typing import Tuple, List

from sympde.topology  import Derham, Square, IdentityMapping, PolarMapping
from sympde.topology.domain import Domain, Union, Connectivity

from psydac.feec.global_projectors import projection_matrix_Hdiv_homogeneous_bc, projection_matrix_H1_homogeneous_bc

from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.fem  import DiscreteBilinearForm, DiscreteLinearForm
from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.cad.geometry     import Geometry
from psydac.fem.basic import FemField
from psydac.fem.vector import VectorFemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.linalg.block import BlockVector
from psydac.linalg.stencil import StencilVector
from psydac.linalg.utilities import array_to_psydac

from scipy.sparse._lil import lil_matrix
from scipy.sparse._coo import coo_matrix
from scipy.sparse._csc import csc_matrix

from sympde.calculus      import grad, dot
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr.equation import find, EssentialBC
import sympde.topology as top
from sympde.utilities.utils import plot_domain

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy

from psydac.cad.geometry          import Geometry
from psydac.core.bsplines         import quadrature_grid
from psydac.fem.basic             import FemField
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace
from psydac.linalg.kron           import KroneckerLinearSolver
from psydac.linalg.block          import BlockDiagonalSolver
from psydac.utilities.quadratures import gauss_legendre

from sympde.topology.domain       import Domain

from scipy.sparse import bmat, csr_matrix
from scipy.sparse._lil import lil_matrix
from scipy.sparse.linalg import eigs, spsolve
from scipy.sparse.linalg import inv

@dataclass
class _Matrices:
    M0 : coo_matrix = None
    M1 : coo_matrix = None
    M2 : coo_matrix = None
    P0h : lil_matrix = None
    P1h : lil_matrix = None
    P2h : lil_matrix = None
    D0 : coo_matrix = None
    D1 : coo_matrix = None
    D2 : coo_matrix = None
    D0_h : csr_matrix = None
    D1_h : csr_matrix = None
    S_h_tilde : csc_matrix = None

def solve_magnetostatic_pbm_annulus(f : sympy.Tuple, 
                                    psi_h : FemField, 
                                    rhs_curve_integral: float,
                                    derham_h: DiscreteDerham,
                                    derham: Derham,
                                    annulus_h: Geometry):
    """
    Solves the Hodge Laplacian problem with curve integral constraint on an annulus

    It solves the Hodge Laplacian problem with right hand side f 
    with the additional constraint corresponding to a curve integral (see ...)

    Parameters
    ----------
    f : sympy.Tuple
        Right hand side
    psi_h : FemField
        Interpolated integration function
    rhs_curve_integral : float
        Sum of inner product of psi with J plus curve integral
    derham_h : DiscreteDerham#
        Discretized de Rham sequence
    derham : 
        Symbolic de Rham sequence
    annulus_h : Geometry
        Discretized annulus domain
    """
    logger_magnetostatic = logging.getLogger(name='solve_magnetostatic_pbm_annulus')

    # Compute the discrete differential operators, mass matrices and projection 
    # matrices
    matrices = _Matrices()
    D0_block_operator, D1_block_operator = derham_h.derivatives_as_matrices
    matrices.D0 = D0_block_operator.tosparse()
    matrices.D1 = D1_block_operator.tosparse()
    assert isinstance(matrices.D0, coo_matrix)
    assert isinstance(matrices.D1, coo_matrix)
    matrices.M0, matrices.M1, matrices.M2 = assemble_mass_matrices(derham, derham_h, annulus_h)
    assert isinstance(matrices.M0, coo_matrix)
    assert isinstance(matrices.M1, coo_matrix)
    assert isinstance(matrices.M2 , coo_matrix)
    matrices.P0h = projection_matrix_H1_homogeneous_bc(derham_h.V0)
    matrices.P1h = projection_matrix_Hdiv_homogeneous_bc(derham_h.V1)
    assert isinstance(matrices.P0h, lil_matrix)
    assert isinstance(matrices.P1h, lil_matrix)

    # Compute penalty matrix
    alpha = 10
    I1 = scipy.sparse.eye(derham_h.V1.nbasis, format='lil')
    matrices.S_h_tilde = (I1 - matrices.P1h).transpose() @ matrices.M1 @ (I1 - matrices.P1h)
    matrices.D0_h = matrices.D0 @ matrices.P0h
    matrices.D1_h = matrices.D1 @ matrices.P1h
    assert isinstance(matrices.D0_h, csr_matrix), f"type(D0_h_mat):{type(matrices.D0_h)}"
    assert isinstance(matrices.D1_h, csr_matrix)
    assert isinstance(matrices.S_h_tilde, csc_matrix), f"type(S_h_tilde_mat):{type(matrices.S_h_tilde)}"

    harmonic_block = _assemble_harmonic_block(matrices=matrices, alpha=alpha)

    # Compute part of right hand side
    annulus = annulus_h.domain
    u, v = top.elements_of(derham.V1, names='u v')
    l_f = LinearForm(v, integral(annulus, dot(f,v)))
    l_f_h = discretize(l_f, annulus_h, space=derham_h.V1)
    assert isinstance(l_f_h, DiscreteLinearForm)
    f_tilde_h_block : BlockVector = l_f_h.assemble()
    f_tilde_h = f_tilde_h_block.toarray()
    assert isinstance(f_tilde_h, np.ndarray)
    f_h = matrices.P1h.transpose() @ f_tilde_h
    assert isinstance(f_h, np.ndarray)

    # Compute curl of the integration function psi
    psi_h_coeffs = psi_h.coeffs.toarray()
    assert isinstance(psi_h_coeffs, np.ndarray)
    curl_psi_h_coeffs = matrices.D0 @ psi_h_coeffs
    curl_psi_h_mat = csr_matrix(curl_psi_h_coeffs)
    curl_psi_h_mat = curl_psi_h_mat.transpose()
    logger_magnetostatic.debug('curl_psi_h_mat.shape:%s\n', curl_psi_h_mat.shape)

    # Assemble the final system and solve
    curve_integral_rhs_arr = np.array([rhs_curve_integral])
    DD_tilde_mat = matrices.D1_h.transpose() @ matrices.M2  @ matrices.D1_h
    A_mat = bmat([[matrices.M0 , -matrices.D0_h.transpose() @ matrices.M1, None],
                  [matrices.M1 @ matrices.D0_h,  DD_tilde_mat  + alpha * matrices.S_h_tilde, harmonic_block],
                  [None, curl_psi_h_mat.transpose() @ matrices.M1 , None]])
    rhs = np.concatenate((np.zeros(derham_h.V0.nbasis), f_h, curve_integral_rhs_arr))
    A_mat = csr_matrix(A_mat) # Why is this not transforming into a CSR matrix?
    logger_magnetostatic.debug('type(A_mat):%s\n', type(A_mat))
    sol = spsolve(A_mat, rhs)
    return sol[derham_h.V0.nbasis:-1]

def _assemble_harmonic_block(matrices: _Matrices, alpha):
    """
    Returns the block for stiffnesss matrix corresponding to the inner product
    with a harmonic form
    """
    summand1 = matrices.D0_h @ inv(matrices.M0) @ matrices.D0_h.transpose() @ matrices.M1
    summand2 = inv(matrices.M1) @ matrices.D1_h.transpose() @ matrices.M2 @ matrices.D1_h
    summand3 = alpha * inv(matrices.M1) @ matrices.S_h_tilde
    # Matrix representation of stabilized Hodge laplacian from primal 
    # to dual basis (29.5. REALLY?)
    L_h = summand1 + summand2 + summand3
    eig_val, eig_vec = eigs(L_h, sigma=0.001)
    harmonic_form_coeffs = eig_vec[:,0]
    harmonic_block = csr_matrix( matrices.P1h.transpose() @ matrices.M1 @ harmonic_form_coeffs)
    harmonic_block = harmonic_block.transpose()
    return harmonic_block

def assemble_mass_matrices(derham: Derham, derham_h: DiscreteDerham, annulus_h : Geometry
                           ) -> Tuple[coo_matrix]:
    """
    Assemble the mass matrices of the spaces a discrete de Rham sequence
    of length 2 on an annulus

    Parameters
    ----------
    derham : Derham
        Symbolic de Rham sequence of length 2
    derham_h : DiscreteDerham
        Discrete de Rham sequence of length 2
    annulus_h : Geometry
        Discrete annulus domain

    Returns
    -------
    Tuple[coo_matrix]
        Mass matrices in Scipy COO-format
    """
    # Define symbolic L2 inner products
    u, v = top.elements_of(derham.V1, names='u, v')
    rho, nu = top.elements_of(derham.V2, names='rho nu')
    annulus = annulus_h.domain
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
    l2_inner_product_V2h = discretize(l2_inner_product_V2, annulus_h, 
                                      spaces=[derham_h.V2, derham_h.V2])
    l2_inner_product_V1h = discretize(l2_inner_product_V1, annulus_h, 
                                      spaces=[derham_h.V1, derham_h.V1])
    l2_inner_product_V0h = discretize(l2_inner_product_V0, annulus_h, 
                                      spaces=[derham_h.V0, derham_h.V0])
    assert isinstance(l2_inner_product_V2h, DiscreteBilinearForm)
    assert isinstance(l2_inner_product_V1h, DiscreteBilinearForm)
    assert isinstance(l2_inner_product_V0h, DiscreteBilinearForm)
    M0 = l2_inner_product_V0h.assemble().tosparse()
    M1 = l2_inner_product_V1h.assemble().tosparse()
    M2 = l2_inner_product_V2h.assemble().tosparse()
    return M0,M1,M2


def solve_magnetostatic_pbm_J_direct_annulus(J : sympy.Expr, 
                                    psi_h : FemField, 
                                    rhs_curve_integral: float,
                                    derham_h: DiscreteDerham,
                                    derham: Derham,
                                    annulus_h: Geometry):
    """
    """
    logger_magnetostatic = logging.getLogger(name='solve_magnetostatic_pbm_J_rhs')

    # Compute the discrete differential operators, mass matrices and projection 
    # matrices
    matrices = _Matrices()
    D0_block_operator, D1_block_operator = derham_h.derivatives_as_matrices
    matrices.D0 = D0_block_operator.tosparse()
    matrices.D1 = D1_block_operator.tosparse()
    assert isinstance(matrices.D0, coo_matrix)
    assert isinstance(matrices.D1, coo_matrix)
    matrices.M0, matrices.M1, matrices.M2 = assemble_mass_matrices(derham, derham_h, annulus_h)
    assert isinstance(matrices.M0, coo_matrix)
    assert isinstance(matrices.M1, coo_matrix)
    assert isinstance(matrices.M2 , coo_matrix)
    matrices.P0h = projection_matrix_H1_homogeneous_bc(derham_h.V0)
    matrices.P1h = projection_matrix_Hdiv_homogeneous_bc(derham_h.V1)
    assert isinstance(matrices.P0h, lil_matrix)
    assert isinstance(matrices.P1h, lil_matrix)

    # Compute penalty matrix
    alpha = 10
    I1 = scipy.sparse.eye(derham_h.V1.nbasis, format='lil')
    matrices.S_h_tilde = (I1 - matrices.P1h).transpose() @ matrices.M1 @ (I1 - matrices.P1h)
    matrices.D0_h = matrices.D0 @ matrices.P0h
    matrices.D1_h = matrices.D1 @ matrices.P1h
    assert isinstance(matrices.D0_h, csr_matrix), f"type(D0_h_mat):{type(matrices.D0_h)}"
    assert isinstance(matrices.D1_h, csr_matrix)
    assert isinstance(matrices.S_h_tilde, csc_matrix), f"type(S_h_tilde_mat):{type(matrices.S_h_tilde)}"

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
    logger_magnetostatic.debug('curl_psi_h_mat.shape:%s\n', curl_psi_h_mat.shape)

    # Assemble the final system and solve
    curve_integral_rhs_arr = np.array([rhs_curve_integral])
    DD_tilde_mat = matrices.D1_h.transpose() @ matrices.M2  @ matrices.D1_h
    A_mat = bmat([[matrices.M0 , -matrices.D0_h.transpose() @ matrices.M1, None],
                  [matrices.M1 @ matrices.D0_h,  DD_tilde_mat  + alpha * matrices.S_h_tilde, harmonic_block],
                  [None, curl_psi_h_mat.transpose() @ matrices.M1 , None]])
    rhs = np.concatenate((-J_h_array, np.zeros(derham_h.V1.nbasis), curve_integral_rhs_arr))
    A_mat = csr_matrix(A_mat) # Why is this not transforming into a CSR matrix?
    logger_magnetostatic.debug('type(A_mat):%s\n', type(A_mat))
    sol = spsolve(A_mat, rhs)
    return sol[derham_h.V0.nbasis:-1]




if __name__ == "__main__":
    logging.basicConfig(filename='mydebug.log', level=logging.DEBUG, filemode='w')
    log_main = logging.getLogger(name='main')
    f = sympy.Tuple(1e-10, 1e-10)
    log_main.debug('f:%s %s', type(f), f)
    solve_magnetostatic_pbm_annulus(f=f, J=1e-10)