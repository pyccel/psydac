# -*- coding: UTF-8 -*-

import numpy as np

from psydac.linalg.stencil  import StencilMatrix, StencilVectorSpace
from psydac.linalg.kron     import KroneckerStencilMatrix
from psydac.linalg.block    import BlockVectorSpace, BlockVector, BlockLinearOperator, BlockMatrix
from psydac.fem.vector      import ProductFemSpace
from psydac.fem.tensor      import TensorFemSpace
from psydac.linalg.identity import IdentityLinearOperator, IdentityStencilMatrix as IdentityMatrix
from psydac.fem.basic       import FemField

__all__ = (
    'd_matrix',
    'DiffOperator',
    'Derivative_1D',
    'Gradient_2D',
    'Gradient_3D',
    'ScalarCurl_2D',
    'VectorCurl_2D',
    'Curl_3D',
    'Divergence_2D',
    'Divergence_3D'
)

#====================================================================================================
def block_tostencil(M):
    """
    Convert a BlockMatrix that contains KroneckerStencilMatrix objects
    to a BlockMatrix that contains StencilMatrix objects
    """
    blocks = [list(b) for b in M.blocks]
    for i1,b in enumerate(blocks):
        for i2, mat in enumerate(b):
            if mat is None:
                continue
            blocks[i1][i2] = mat.tostencil()
    return BlockMatrix(M.domain, M.codomain, blocks=blocks)

#====================================================================================================
def d_matrix(n, p, periodic):
    """
    Create a 1D incidence matrix of shape (n, n) in the periodic case, and (n, n-1) otherwise.
    The incidence matrix has values -1 on the main diagonal, and +1 on the diagonal above it.

    Parameters
    ----------
    n : int
        Number of degrees of freedom.

    p : int
        Padding size.

    periodic : bool
        True if matrix is periodic, False otherwise.

    Results
    -------
    M : psydac.linalg.stencil.StencilMatrix
        1D incidence matrix of shape (n, n) in the periodic case, and (n, n-1) otherwise.

    """
    m = n if periodic else n - 1

    V1 = StencilVectorSpace([n], [p], [periodic])
    V2 = StencilVectorSpace([m], [p], [periodic])
    M  = StencilMatrix(V1, V2)

    for i in range(m):
        M._data[p+i, p]   = -1.
        M._data[p+i, p+1] =  1.

    return M

#====================================================================================================
class DiffOperator:

    @property
    def matrix(self):
        return self._matrix

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

#====================================================================================================
class Derivative_1D(DiffOperator):
    """
    1D derivative.

    Parameters
    ----------
    H1 : 1D TensorFemSpace
        Domain of derivative operator.

    L2 : 1D TensorFemSpace
        Codomain of derivative operator.

    """
    def __init__(self, H1, L2):

        assert isinstance(H1, TensorFemSpace); assert H1.ldim == 1
        assert isinstance(L2, TensorFemSpace); assert L2.ldim == 1
        assert H1.periodic[0] == L2.periodic[0]
        assert H1.degree[0] == L2.degree[0] + 1

        # 1D spline space
        N = H1.spaces[0]

        # 1D differentiation matrix (from B-spline of degree p to M-spline of degree p-1)
        Dx = d_matrix(N.nbasis, N.degree, N.periodic)

        self._domain   = H1
        self._codomain = L2
        self._matrix   = KroneckerStencilMatrix(H1.vector_space, L2.vector_space, Dx)
        self._matrix   = self._matrix.tostencil()

    def __call__(self, u):

        assert isinstance(u, FemField)
        assert u.space == self.domain

        coeffs = self.matrix.dot(u.coeffs)
        coeffs.update_ghost_regions()

        return FemField(self.codomain, coeffs=coeffs)

#====================================================================================================
class Gradient_2D(DiffOperator):
    """
    Gradient operator in 2D.

    Parameters
    ----------
    H1 : 2D TensorFemSpace
        Domain of gradient operator.

    Hcurl : 2D ProductFemSpace
        Codomain of gradient operator.

    """
    def __init__(self, H1, Hcurl):

        assert isinstance(   H1,  TensorFemSpace); assert    H1.ldim == 2
        assert isinstance(Hcurl, ProductFemSpace); assert Hcurl.ldim == 2

        assert Hcurl.spaces[0].periodic == H1.periodic
        assert Hcurl.spaces[1].periodic == H1.periodic

        assert tuple(Hcurl.spaces[0].degree) == (H1.degree[0]-1, H1.degree[1]  )
        assert tuple(Hcurl.spaces[1].degree) == (H1.degree[0]  , H1.degree[1]-1)

        # 1D differentiation matrices (from B-spline of degree p to M-spline of degree p-1)
        Dx, Dy = [d_matrix(N.nbasis, N.degree, N.periodic) for N in H1.spaces]

        # 1D identity matrices for B-spline coefficients
        Ix, Iy = [IdentityMatrix(N.vector_space) for N in H1.spaces]

        # Tensor-product spaces of coefficients - domain
        B_B = H1.vector_space

        # Tensor-product spaces of coefficients - codomain
        (M_B, B_M) = Hcurl.vector_space.spaces

        # Build Gradient matrix block by block
        blocks = [[KroneckerStencilMatrix(B_B, M_B, *(Dx, Iy))],
                  [KroneckerStencilMatrix(B_B, B_M, *(Ix, Dy))]]
        matrix = BlockMatrix(BlockVectorSpace(H1.vector_space), Hcurl.vector_space, blocks=blocks)

        # Store data in object
        self._domain   = H1
        self._codomain = Hcurl
        self._matrix   = block_tostencil(matrix)

    def __call__(self, u):

        assert isinstance(u, FemField)
        assert u.space == self.domain

        coeffs = self.matrix.dot(u.coeffs)
        coeffs.update_ghost_regions()

        return FemField(self.codomain, coeffs=coeffs)

#====================================================================================================
class Gradient_3D(DiffOperator):
    """
    Gradient operator in 3D.

    Parameters
    ----------
    H1 : 3D TensorFemSpace
        Domain of gradient operator.

    Hcurl : 3D ProductFemSpace
        Codomain of gradient operator.

    """
    def __init__(self, H1, Hcurl):

        assert isinstance(   H1,  TensorFemSpace); assert    H1.ldim == 3
        assert isinstance(Hcurl, ProductFemSpace); assert Hcurl.ldim == 3

        assert Hcurl.spaces[0].periodic == H1.periodic
        assert Hcurl.spaces[1].periodic == H1.periodic
        assert Hcurl.spaces[2].periodic == H1.periodic

        assert tuple(Hcurl.spaces[0].degree) == (H1.degree[0]-1, H1.degree[1]  , H1.degree[2]  )
        assert tuple(Hcurl.spaces[1].degree) == (H1.degree[0]  , H1.degree[1]-1, H1.degree[2]  )
        assert tuple(Hcurl.spaces[2].degree) == (H1.degree[0]  , H1.degree[1]  , H1.degree[2]-1)

        # 1D differentiation matrices (from B-spline of degree p to M-spline of degree p-1)
        Dx, Dy, Dz = [d_matrix(N.nbasis, N.degree, N.periodic) for N in H1.spaces]

        # 1D identity matrices for B-spline coefficients
        Ix, Iy, Iz = [IdentityMatrix(N.vector_space) for N in H1.spaces]

        # Tensor-product spaces of coefficients - domain
        B_B_B = H1.vector_space

        # Tensor-product spaces of coefficients - codomain
        (M_B_B, B_M_B, B_B_M) = Hcurl.vector_space.spaces

        # Build Gradient matrix block by block
        blocks = [[KroneckerStencilMatrix(B_B_B, M_B_B, *(Dx, Iy, Iz))],
                  [KroneckerStencilMatrix(B_B_B, B_M_B, *(Ix, Dy, Iz))],
                  [KroneckerStencilMatrix(B_B_B, B_B_M, *(Ix, Iy, Dz))]]
        matrix = BlockMatrix(BlockVectorSpace(H1.vector_space), Hcurl.vector_space, blocks=blocks)

        # Store data in object
        self._domain   = H1
        self._codomain = Hcurl
        self._matrix   = block_tostencil(matrix)

    def __call__(self, u):

        assert isinstance(u, FemField)
        assert u.space == self.domain

        coeffs = self.matrix.dot(u.coeffs)
        coeffs.update_ghost_regions()

        return FemField(self.codomain, coeffs=coeffs)

#====================================================================================================
class ScalarCurl_2D(DiffOperator):
    """
    Scalar curl operator in 2D: computes a scalar field from a vector field.

    Parameters
    ----------
    Hcurl : 2D ProductFemSpace
        Domain of 2D scalar curl operator.

    L2 : 2D TensorFemSpace
        Codomain of 2D scalar curl operator.

    """
    def __init__(self, Hcurl, L2):

        assert isinstance(Hcurl, ProductFemSpace); assert Hcurl.ldim == 2
        assert isinstance(   L2,  TensorFemSpace); assert    L2.ldim == 2

        assert Hcurl.spaces[0].periodic == L2.periodic
        assert Hcurl.spaces[1].periodic == L2.periodic

        assert tuple(Hcurl.spaces[0].degree) == (L2.degree[0]  , L2.degree[1]+1)
        assert tuple(Hcurl.spaces[1].degree) == (L2.degree[0]+1, L2.degree[1]  )

        # 1D spline spaces
        N_basis = [Hcurl.spaces[1].spaces[0], Hcurl.spaces[0].spaces[1]]
        D_basis = L2.spaces

        # 1D differentiation matrices (from B-spline of degree p to M-spline of degree p-1)
        Dx, Dy = [d_matrix(N.nbasis, N.degree, N.periodic) for N in N_basis]

        # 1D identity matrices for M-spline coefficients
        # NOTE: We keep the same padding of the parent space N
        Jx, Jy = [IdentityMatrix(D.vector_space) for D in D_basis]

        # Tensor-product spaces of coefficients - domain
        (M_B, B_M) = Hcurl.vector_space.spaces

        # Tensor-product spaces of coefficients - codomain
        M_M = L2.vector_space

        # Build Curl matrix block by block
        f = KroneckerStencilMatrix
        blocks = [[f(M_B, M_M, *(-Jx, Dy)), f(B_M, M_M, *(Dx, Jy))]]
        matrix = BlockMatrix(Hcurl.vector_space, BlockVectorSpace(L2.vector_space), blocks=blocks)

        # Store data in object
        self._domain   = Hcurl
        self._codomain = L2
        self._matrix   = block_tostencil(matrix)
   
    def __call__(self, u):

        assert isinstance(u, FemField)
        assert u.space == self.domain
        
        coeffs = self.matrix.dot(u.coeffs)
        coeffs.update_ghost_regions()

        return FemField(self.codomain, coeffs=coeffs)

#====================================================================================================
class VectorCurl_2D(DiffOperator):
    """
    Vector curl operator in 2D: computes a vector field from a scalar field.
    This is sometimes called the 'rot' operator.

    Parameters
    ----------
    H1 : 2D TensorFemSpace
        Domain of 2D vector curl operator.

    Hdiv : 2D ProductFemSpace
        Codomain of 2D vector curl operator.

    """
    def __init__(self, H1, Hdiv):

        assert isinstance(  H1,  TensorFemSpace); assert   H1.ldim == 2
        assert isinstance(Hdiv, ProductFemSpace); assert Hdiv.ldim == 2

        assert Hdiv.spaces[0].periodic == H1.periodic
        assert Hdiv.spaces[1].periodic == H1.periodic

        assert tuple(Hdiv.spaces[0].degree) == (H1.degree[0]  , H1.degree[1]-1)
        assert tuple(Hdiv.spaces[1].degree) == (H1.degree[0]-1, H1.degree[1]  )

        # 1D differentiation matrices (from B-spline of degree p to M-spline of degree p-1)
        Dx, Dy = [d_matrix(N.nbasis, N.degree, N.periodic) for N in H1.spaces]

        # 1D identity matrices for B-spline coefficients
        Ix, Iy = [IdentityMatrix(N.vector_space) for N in H1.spaces]

        # Tensor-product spaces of coefficients - domain
        B_B = H1.vector_space

        # Tensor-product spaces of coefficients - codomain
        (B_M, M_B) = Hdiv.vector_space.spaces

        # Build Curl matrix block by block
        blocks = [[KroneckerStencilMatrix(B_B, B_M, *( Ix, Dy))],
                  [KroneckerStencilMatrix(B_B, M_B, *(-Dx, Iy))]]
        matrix = BlockMatrix(BlockVectorSpace(H1.vector_space), Hdiv.vector_space, blocks=blocks)

        # Store data in object
        self._domain   = H1
        self._codomain = Hdiv
        self._matrix   = block_tostencil(matrix)

    def __call__(self, u):

        assert isinstance(u, FemField)
        assert u.space == self.domain

        coeffs = self.matrix.dot(u.coeffs)
        coeffs.update_ghost_regions()

        return FemField(self.codomain, coeffs=coeffs)

#====================================================================================================
class Curl_3D(DiffOperator):
    """
    Curl operator in 3D.

    Parameters
    ----------
    Hcurl : 3D ProductFemSpace
        Domain of 3D curl operator.

    Hdiv : 3D TensorFemSpace
        Codomain of 3D curl operator.

    """
    def __init__(self, Hcurl, Hdiv):

        assert isinstance(Hcurl, ProductFemSpace); assert Hcurl.ldim == 3
        assert isinstance( Hdiv, ProductFemSpace); assert  Hdiv.ldim == 3

        assert Hcurl.spaces[0].periodic == Hdiv.spaces[0].periodic
        assert Hcurl.spaces[1].periodic == Hdiv.spaces[1].periodic
        assert Hcurl.spaces[2].periodic == Hdiv.spaces[2].periodic

        # TODO: checking the degree would be nice here

        # 1D spline spaces
        N_basis = [ Hdiv.spaces[i].spaces[i] for i in range(3)]
        D_basis = [Hcurl.spaces[i].spaces[i] for i in range(3)]

        # 1D differentiation matrices (from B-spline of degree p to M-spline of degree p-1)
        Dx, Dy, Dz = [d_matrix(N.nbasis, N.degree, N.periodic) for N in N_basis]

        # 1D identity matrices for B-spline coefficients
        Ix, Iy, Iz = [IdentityMatrix(N.vector_space) for N in N_basis]

        # 1D identity matrices for M-spline coefficients
        # NOTE: We keep the same padding of the parent space N
        Jx, Jy, Jz = [IdentityMatrix(D.vector_space) for D in D_basis]

        # Tensor-product spaces of coefficients - domain
        (M_B_B, B_M_B, B_B_M) = Hcurl.vector_space.spaces

        # Tensor-product spaces of coefficients - codomain
        (B_M_M, M_B_M, M_M_B) = Hdiv.vector_space.spaces

        # ...
        # Build Curl matrix block by block
        f = KroneckerStencilMatrix

        blocks = [[              None             , f(B_M_B, B_M_M, *(Ix, Jy, -Dz)), f(B_B_M, B_M_M, *( Ix, Dy, Jz))],
                  [f(M_B_B, M_B_M, *(Jx,  Iy, Dz)),               None             , f(B_B_M, M_B_M, *(-Dx, Iy, Jz))],
                  [f(M_B_B, M_M_B, *(Jx, -Dy, Iz)), f(B_M_B, M_M_B, *(Dx, Jy,  Iz)),               None             ]]

        matrix = BlockMatrix(Hcurl.vector_space, Hdiv.vector_space, blocks=blocks)
        # ...

        # Store data in object
        self._domain   = Hcurl
        self._codomain = Hdiv
        self._matrix   = block_tostencil(matrix)
   
    def __call__(self, u):

        assert isinstance(u, FemField)
        assert u.space == self.domain
        
        coeffs = self.matrix.dot(u.coeffs)
        coeffs.update_ghost_regions()

        return FemField(self.codomain, coeffs=coeffs)

#====================================================================================================
class Divergence_2D(DiffOperator):
    """
    Divergence operator in 2D.

    Parameters
    ----------
    Hdiv : 2D ProductFemSpace
        Domain of divergence operator.

    L2 : 2D TensorFemSpace
        Codomain of divergence operator.

    """
    def __init__(self, Hdiv, L2):

        assert isinstance(Hdiv, ProductFemSpace); assert Hdiv.ldim == 2
        assert isinstance(  L2,  TensorFemSpace); assert   L2.ldim == 2

        assert Hdiv.spaces[0].periodic == L2.periodic
        assert Hdiv.spaces[1].periodic == L2.periodic

        assert tuple(Hdiv.spaces[0].degree) == (L2.degree[0]+1, L2.degree[1]  )
        assert tuple(Hdiv.spaces[1].degree) == (L2.degree[0]  , L2.degree[1]+1)

        # 1D spline spaces
        N_basis = [Hdiv.spaces[i].spaces[i] for i in range(2)]
        D_basis = L2.spaces

        # 1D differentiation matrices (from B-spline of degree p to M-spline of degree p-1)
        Dx, Dy = [d_matrix(N.nbasis, N.degree, N.periodic) for N in N_basis]

        # 1D identity matrices for M-spline coefficients
        # NOTE: We keep the same padding of the parent space N
        Jx, Jy = [IdentityMatrix(D.vector_space) for D in D_basis]

        # Tensor-product spaces of coefficients - domain
        (B_M, M_B) = Hdiv.vector_space.spaces

        # Tensor-product spaces of coefficients - codomain
        M_M = L2.vector_space

        # Build Divergence matrix block by block
        f = KroneckerStencilMatrix
        blocks = [[f(B_M, M_M, *(Dx, Jy)), f(M_B, M_M, *(Jx, Dy))]]
        matrix = BlockMatrix(Hdiv.vector_space, BlockVectorSpace(L2.vector_space), blocks=blocks) 

        # Store data in object
        self._domain   = Hdiv
        self._codomain = L2
        self._matrix   = block_tostencil(matrix)

    def __call__(self, u):

        assert isinstance(u, FemField)
        assert u.space == self.domain

        coeffs = self.matrix.dot(u.coeffs)
        coeffs.update_ghost_regions()

        return FemField(self.codomain, coeffs=coeffs)

#====================================================================================================
class Divergence_3D(DiffOperator):
    """
    Divergence operator in 3D.

    Parameters
    ----------
    Hdiv : 3D ProductFemSpace
        Domain of divergence operator.

    L2 : 3D TensorFemSpace
        Codomain of divergence operator.

    """
    def __init__(self, Hdiv, L2):

        assert isinstance(Hdiv, ProductFemSpace); assert Hdiv.ldim == 3
        assert isinstance(  L2,  TensorFemSpace); assert   L2.ldim == 3

        assert Hdiv.spaces[0].periodic == L2.periodic
        assert Hdiv.spaces[1].periodic == L2.periodic
        assert Hdiv.spaces[2].periodic == L2.periodic

        assert tuple(Hdiv.spaces[0].degree) == (L2.degree[0]+1, L2.degree[1]  , L2.degree[2]  )
        assert tuple(Hdiv.spaces[1].degree) == (L2.degree[0]  , L2.degree[1]+1, L2.degree[2]  )
        assert tuple(Hdiv.spaces[2].degree) == (L2.degree[0]  , L2.degree[1]  , L2.degree[2]+1)

        # 1D spline spaces
        N_basis = [Hdiv.spaces[i].spaces[i] for i in range(3)]
        D_basis = L2.spaces

        # 1D differentiation matrices (from B-spline of degree p to M-spline of degree p-1)
        Dx, Dy, Dz = [d_matrix(N.nbasis, N.degree, N.periodic) for N in N_basis]

        # 1D identity matrices for M-spline coefficients
        # NOTE: We keep the same padding of the parent space N
        Jx, Jy, Jz = [IdentityMatrix(D.vector_space) for D in D_basis]

        # Tensor-product spaces of coefficients - domain
        (B_M_M, M_B_M, M_M_B) = Hdiv.vector_space.spaces

        # Tensor-product spaces of coefficients - codomain
        M_M_M = L2.vector_space

        # Build Divergence matrix block by block
        f = KroneckerStencilMatrix
        blocks = [[f(B_M_M, M_M_M, *(Dx, Jy, Jz)), f(M_B_M, M_M_M, *(Jx, Dy, Jz)), f(M_M_B, M_M_M, *(Jx, Jy, Dz))]]
        matrix = BlockMatrix(Hdiv.vector_space, BlockVectorSpace(L2.vector_space), blocks=blocks) 

        # Store data in object
        self._domain   = Hdiv
        self._codomain = L2
        self._matrix   = block_tostencil(matrix)

    def __call__(self, u):

        assert isinstance(u, FemField)
        assert u.space == self.domain

        coeffs = self.matrix.dot(u.coeffs)
        coeffs.update_ghost_regions()

        return FemField(self.codomain, coeffs=coeffs)
