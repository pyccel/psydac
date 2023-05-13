import sys, os

from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.fem  import DiscreteSumForm, DiscreteLinearForm
from psydac.cad.geometry     import Geometry
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector      import ProductFemSpace
from psydac.fem.basic import FemField
from psydac.feec.derivatives import VectorCurl_2D, Divergence_2D
from psydac.feec.multipatch.operators import HodgeOperator
from psydac.feec.pushforward import Pushforward
from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.utilities import array_to_psydac

from sympde.calculus      import grad, dot
from sympde.expr import BilinearForm, LinearForm, integral
import sympde.topology as top

from typing import Callable, List

import sympy
import numpy as np

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

from psydac.linalg.block    import BlockVector, BlockLinearOperator
from psydac.fem.tensor      import TensorFemSpace
#from psydac.linalg.basic    import IdentityOperator
from psydac.fem.basic       import FemField


from psydac.fem.tests.get_integration_function import solve_poisson_2d_annulus

@dataclass
class CurveIntegralData:
    c_0 : float
    curve_integral_function: FemField


def solve_magnetostatic_pbm_curve_integral(J : sympy.Expr, 
                                    curve_integral_data : CurveIntegralData,
                                    domain_h : Geometry, 
                                    derham_h : DiscreteDerham) -> FemField:
    domain = domain_h.domain
    V0h : TensorFemSpace = derham_h.V0
    V1h : ProductFemSpace = derham_h.V1
    V2h : TensorFemSpace = derham_h.V2
    assert isinstance(V0h, TensorFemSpace)
    assert isinstance(V1h, ProductFemSpace)
    assert isinstance(V2h, TensorFemSpace)

    N0 = V0h.vector_space.dimension
    N1 = V1h.vector_space.dimension
    N2 = V2h.vector_space.dimension
    ###DEBUG###
    print("N0:", N0)
    print("N1:", N1)
    ###########
    H0 = HodgeOperator(V0h, domain_h, load_space_index=0)
    H1 = HodgeOperator(V1h, domain_h, load_space_index=1)
    H2 = HodgeOperator(V2h, domain_h, load_space_index=2)

    dH0_m = H0.get_dual_Hodge_sparse_matrix().tocsr()  # = mass matrix of V0
    dH1_m = H1.get_dual_Hodge_sparse_matrix().tocsr()   # = mass matrix of V1
    dH2_m = H2.get_dual_Hodge_sparse_matrix().tocsr()   # = mass matrix of V2
    
    # ###DEBUG###
    # print("Does this print?")
    # print("dH1_m:", dH1_m)
    # ###########

    # Create the sparse matrices of discrete differential operators
    vector_curl_h = VectorCurl_2D(V0h, V1h)
    div_h = Divergence_2D(V1h, V2h)
    vector_curl_h_mat = vector_curl_h.matrix.tosparse().tocsr()
    div_h_mat = div_h.matrix.tosparse().tocsr()
    ###DEBUG###
    # print("vector_curl_h_mat.shape:", vector_curl_h_mat.shape)
    # print("vector_curl_h_mat:", vector_curl_h_mat)
    # print("div_h_mat:", div_h_mat)
    # print("div_h_mat.shape:", div_h_mat.shape)
    ###########
    
    # Compute matrix representation of the curve integral
    # NOTE: We assume for now that our domain is an annulus
    boundary = top.Union(*[domain.get_boundary(axis=0, ext=-1), 
                            domain.get_boundary(axis=0, ext=1)])
    psi_h = curve_integral_data.curve_integral_function      
    curl_psi_h = vector_curl_h(psi_h)

    psi_h_vec = psi_h.coeffs.toarray()
    curl_psi_h_vec = curl_psi_h.coeffs.toarray()
    ###DEBUG###
    # print("div_h_mat.dot(curl_psi_h_vec):", div_h_mat.dot(curl_psi_h_vec))
    ###########


    ###DEBUG###
    # print("psi_h_vec:", psi_h_vec)
    # print("curl_psi_h_vec:", curl_psi_h_vec)
    ###########
    # TBD: name
    curve_integral_array = dH1_m.dot(curl_psi_h_vec)
    ###DEBUG###
    # print("np.abs(curve_integral_array).min():", np.abs(curve_integral_array).min() )
    # print("np.abs(curve_integral_array).max():", np.abs(curve_integral_array).max() )
    ###########
    curve_integral_mat = sparse.csr_matrix(curve_integral_array) # This will be 
                        # submatrix of stiffness matrix in the final system
    ###DEBUG###
    # print("curve_integral_mat:", curve_integral_mat)
    # print("np.abs(curve_integral_array).min():", np.abs(curve_integral_array).min())
    # print("np.abs(curve_integral_array).max():", np.abs(curve_integral_array).max())
    ###########

    tCurl_mat = dH1_m @ vector_curl_h_mat
    DD_m = div_h_mat.transpose().tocsr() @ dH2_m @ div_h_mat
    assert isinstance(tCurl_mat, sparse.csr_matrix)
    assert isinstance(DD_m, sparse.csr_matrix)
    ###DEBUG###
    print("tCurl_mat.transpose().dot(curl_psi_h_vec):", tCurl_mat.transpose().dot(curl_psi_h_vec))

    # Compute the matrix representations of the boundary integrals used to 
    # enforce the boundary conditions
    nn = top.NormalVector(label='nn')
    sigma, tau = top.elements_of(derham_h.V0.symbolic_space, names='sigma, tau')
    B, v = top.elements_of(derham_h.V1.symbolic_space, names='B, v')
    boundary_integral_expr0 = sigma*tau
    boundary_integral_expr1 = dot(nn, B) * dot(nn, v)
    a_B_0 = BilinearForm((sigma,tau), 
                       integral(boundary, boundary_integral_expr0))
    ###DEBUG###
    # print("a_B_0", a_B_0)
    ###########
    a_B_0_h = discretize(a_B_0,  domain_h, spaces=[V0h, V0h])
    a_B_1 = BilinearForm((B,v), integral(boundary, boundary_integral_expr1))
    ###DEBUG###
    # print("a_B_1", a_B_1)
    ###########
    a_B_1_h = discretize(a_B_1, domain_h, spaces=[V1h, V1h])
    assert isinstance(a_B_0_h, DiscreteSumForm)
    assert isinstance(a_B_1_h, DiscreteSumForm)
    a_B_0_h_stencil = a_B_0_h.assemble()
    a_B_1_h_operator = a_B_1_h.assemble()
    assert isinstance(a_B_0_h_stencil, StencilMatrix)
    assert isinstance(a_B_1_h_operator, BlockLinearOperator)
    a_B_0_h_mat = a_B_0_h_stencil.tosparse().tocsr()
    a_B_1_h_mat = a_B_1_h_operator.tosparse().tocsr()
    ###DEBUG###
    # print("a_B_0_h_mat", a_B_0_h_mat)
    ###########

    # Penalization parameters to enforce boundary conditions
    penal0 = 1e20
    penal1 = 1e20

    A_m = sparse.bmat([[penal0*a_B_0_h_mat,  tCurl_mat.transpose(),      None                  ],
                       [tCurl_mat,           DD_m + penal1*a_B_1_h_mat,  curve_integral_mat.transpose()    ],
                       [None,                curve_integral_mat,         None                  ]])
    A_m = A_m.tocsr()
    ###DEBUG###
    print("np.linalg.cond( A_m.toarray()):", np.linalg.cond( A_m.toarray()) )
    ###########
    # TBD: name
    l0 = LinearForm(tau, integral(domain, J*tau))
    l0_h = discretize(l0, domain_h, V0h)
    assert isinstance(l0_h, DiscreteLinearForm)
    l0_h_stencil = l0_h.assemble()
    assert isinstance(l0_h_stencil, StencilVector)
    l0_h_vec = l0_h_stencil.toarray()
    curve_integral_rhs = curve_integral_data.c_0 + np.dot(l0_h_vec, psi_h_vec)
    rhs = np.block([l0_h_vec, np.zeros(N1), curve_integral_rhs])

    # Compute the solution coefficients and convert them to a FEM field
    sol_coeffs = spsolve(A_m, rhs)
    assert isinstance(sol_coeffs, np.ndarray)
    sigma_h_vec = sol_coeffs[:N0]
    B_h_vec = sol_coeffs[N0:(N0+N1)]
    ###DEBUG###
    print("sigma_h_vec:", sigma_h_vec)
    print("a_B_1_h_mat.dot(B_h_vec):", a_B_1_h_mat.dot(B_h_vec))
    print("DD_m.dot(B_h_vec):", DD_m.dot(B_h_vec))
    print("div_h_mat.dot(B_h_vec):", div_h_mat.dot(B_h_vec))
    ###########

    assert isinstance(B_h_vec, np.ndarray)
    p_h_scalar = sol_coeffs[-1]
    B_h_block_vec = array_to_psydac(B_h_vec, V1h.vector_space)
    assert isinstance(B_h_block_vec, BlockVector)
    B_h = FemField(V1h, coeffs=B_h_block_vec)
    assert isinstance(B_h, FemField)
    return B_h

    
    

