import numpy as np
import pytest
import logging

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
from typing import Tuple

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
from psydac.linalg.utilities import array_to_psydac

from scipy.sparse._lil import lil_matrix
from scipy.sparse._coo import coo_matrix

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



def solve_magnetostatic_pbm_annulus(f : sympy.Tuple):
    logger_magnetostatic = logging.getLogger(name='solve_magnetostatic_pbm_annulus')
    logical_domain = Square(name='square', bounds1=(0,1), bounds2=(0,2*np.pi))
    boundary_logical_domain = Union(logical_domain.get_boundary(axis=0, ext=-1),
                                    logical_domain.get_boundary(axis=0, ext=1))
    logical_domain = Domain(name='logical_domain', 
                            interiors=logical_domain.interior,
                            boundaries=boundary_logical_domain,
                            dim=2)
    polar_mapping = PolarMapping(name='polar_mapping', dim=2, c1=0., c2=0., 
                                 rmin=1.0, rmax=2.0)
    annulus = polar_mapping(logical_domain)
    derham = Derham(domain=annulus, sequence=['H1', 'Hdiv', 'L2'])

    ncells = [10,10]
    annulus_h = discretize(annulus, ncells=ncells)
    derham_h = discretize(derham, annulus_h, degree=[2,2])
    assert isinstance(derham_h, DiscreteDerham)
    D0_block_operator, D1_block_operator = derham_h.derivatives_as_matrices
    D0_mat = D0_block_operator.tosparse()
    D1_mat = D1_block_operator.tosparse()
    assert isinstance(D0_mat, coo_matrix)
    assert isinstance(D1_mat, coo_matrix)

    u, v = top.elements_of(derham.V1, names='u, v')
    rho, nu = top.elements_of(derham.V2, names='rho nu')
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
    logger_magnetostatic.debug('type(M0):%s', type(M0))
    logger_magnetostatic.debug('M0.shape:%s', M0.shape)
    logger_magnetostatic.debug('type(M1):%s', type(M1))
    logger_magnetostatic.debug('M1.shape:%s', M1.shape)

    P0_h_mat = projection_matrix_H1_homogeneous_bc(derham_h.V0)
    P1_h_mat = projection_matrix_Hdiv_homogeneous_bc(derham_h.V1)
    alpha = 10
    I1 = scipy.sparse.eye(derham_h.V1.nbasis, format='lil')
    S_h_mat = alpha * (I1 - P1_h_mat).transpose() @ M1 @ (I1 - P1_h_mat)
    D0_h_mat = D0_mat @ P0_h_mat
    D1_h_mat = D1_mat @ P1_h_mat
    
    summand1 = D0_h_mat @ inv(M0) @ D0_h_mat.transpose() @ M1
    summand2 = inv(M1) @ D1_h_mat.transpose() @ M2 @ D1_h_mat
    summand3 = alpha * inv(M1) @ S_h_mat
    # Matrix representation of stabilized Hodge laplacian from primal 
    # to dual basis
    L_h_tilde = summand1 + summand2 + summand3
    eig_val, eig_vec = eigs(L_h_tilde, sigma=0.001)
    logger_magnetostatic.debug('eig_val.shape:%s', eig_val.shape)
    logger_magnetostatic.debug('L_h_tilde.dot(eig_vec[:,0]):%s\n', 
                               L_h_tilde.dot(eig_vec[:,0]))
    harmonic_form_coeffs = eig_vec[:,0]

    l_f = LinearForm(v, integral(annulus, dot(f,v)))
    l_f_h = discretize(l_f, annulus_h, space=derham_h.V1)
    assert isinstance(l_f_h, DiscreteLinearForm)
    f_tilde_h_block : BlockVector = l_f_h.assemble()
    f_tilde_h = f_tilde_h_block.toarray()
    assert isinstance(f_tilde_h, np.ndarray)
    f_h = inv(M1) @ P1_h_mat.transpose() @ f_tilde_h
    assert isinstance(f_h, np.ndarray)

    # psi_h_coeffs = psi_h.coeffs.toarray()
    # assert isinstance(psi_h_coeffs, np.ndarray)
    # curl_psi_h_coeffs = D0_h_mat @ psi_h_coeffs

    harmonic_block = csr_matrix( P1_h_mat.transpose() @ M1 @ harmonic_form_coeffs)
    harmonic_block = harmonic_block.transpose()
    logger_magnetostatic.debug('harmonic_block.shape:%s\n', harmonic_block.shape)

    DD_tilde_mat = (D1_h_mat @ P1_h_mat).transpose() @ M2 @ D1_h_mat @ P1_h_mat
    logger_magnetostatic.debug('type(DD_tilde_mat):%s', type(DD_tilde_mat))
    logger_magnetostatic.debug('type(P0_h_mat):%s', type(P0_h_mat))
    logger_magnetostatic.debug('type(M1):%s', type(M1))
    logger_magnetostatic.debug('type(M1 @ D0_h_mat @ P0_h_mat):%s\n', type(M1 @ D0_h_mat @ P0_h_mat))

    logger_magnetostatic.debug('(M1 @ D0_h_mat @ P0_h_mat).shape:%s', (M1 @ D0_h_mat @ P0_h_mat).shape)
    logger_magnetostatic.debug('(DD_tilde_mat  + alpha * S_h_mat).shape:%s', (DD_tilde_mat  + alpha * S_h_mat).shape)
    logger_magnetostatic.debug('harmonic_block.shape:%s\n', harmonic_block.shape)

    A_mat = bmat([[M0 , (D0_h_mat @ P0_h_mat).transpose() , None],
                  [M1 @ D0_h_mat @ P0_h_mat,  DD_tilde_mat  + alpha * S_h_mat, harmonic_block],
                  [None, harmonic_block.T , None]])
    
    rhs = np.concatenate((np.zeros(derham_h.V0.nbasis), f_h, np.zeros(1)))
    sol = spsolve(A_mat, rhs)
    return sol[derham_h.V0.nbasis:-1]







if __name__ == "__main__":
    logging.basicConfig(filename='mydebug.log', level=logging.DEBUG, filemode='w')
    log_main = logging.getLogger(name='main')
    f = sympy.Tuple(1e-10, 1e-10)
    log_main.debug('f:%s %s', type(f), f)
    solve_magnetostatic_pbm_annulus(f=f, J=1e-10)