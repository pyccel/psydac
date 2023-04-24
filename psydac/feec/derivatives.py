# -*- coding: UTF-8 -*-

import numpy as np
import scipy.sparse as spa

from psydac.linalg.stencil  import StencilVector, StencilMatrix, StencilVectorSpace
from psydac.linalg.kron     import KroneckerStencilMatrix
from psydac.linalg.block    import BlockVector, BlockLinearOperator
from psydac.fem.vector      import ProductFemSpace, VectorFemSpace
from psydac.fem.tensor      import TensorFemSpace
from psydac.linalg.identity import IdentityStencilMatrix, IdentityMatrix
#from psydac.linalg.basic    import IdentityOperator
from psydac.fem.basic       import FemField
from psydac.linalg.basic    import LinearOperator
from psydac.ddm.cart        import DomainDecomposition, CartDecomposition

__all__ = (
    'DirectionalDerivativeOperator',
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
    Convert a BlockLinearOperator that contains KroneckerStencilMatrix objects
    to a BlockLinearOperator that contains StencilMatrix objects
    """
    blocks = [list(b) for b in M.blocks]
    for i1,b in enumerate(blocks):
        for i2, mat in enumerate(b):
            if mat is None:
                continue
            blocks[i1][i2] = mat.tostencil()
    return BlockLinearOperator(M.domain, M.codomain, blocks=blocks)

#====================================================================================================
class DirectionalDerivativeOperator(LinearOperator):
    """
    Represents a matrix-free derivative operator in a specific cardinal direction.
    Can be negated and transposed.

    Parameters
    ----------
    V : StencilVectorSpace
        The domain of the operator. (or codomain if transposed)
    
    W : StencilVectorSpace
        The codomain of the operator. (or domain, if transposed)
        Has to be compatible with the domain, i.e. it has to
        be equal to it, except for the differentiation direction.
    
    diffdir : int
        The differentiation direction.
    
    negative : bool
        If True, this operator is multiplied by -1 after execution.
        (if False, nothing happens)
    
    transposed : bool
        If True, this operator represents the transposed derivative operator.
        Note that then V is the codomain and W is the domain.
    """

    def __init__(self, V, W, diffdir, *, negative=False, transposed=False):
        assert isinstance(V, StencilVectorSpace)
        assert isinstance(W, StencilVectorSpace)
        assert V.ndim == W.ndim
        assert all([vp==wp for vp, wp in zip(V.periods, W.periods)])
        assert V.parallel == W.parallel
        assert V.dtype is W.dtype
        assert diffdir >= 0 and diffdir < V.ndim

        # no need for the pads to conform, but we want them to be there at least
        # (we need them to perform the diff operation in-place)
        assert all([pad > 0 for pad in V.pads])
        assert all([pad > 0 for pad in W.pads])

        # check that the number of points conforms
        assert all([vn==wn+1 if not vwp and diffdir==i else vn==wn
            for i, (vn, wn, vwp) in enumerate(zip(V.npts, W.npts, V.periods))])

        self._spaceV = V
        self._spaceW = W
        self._diffdir = diffdir
        self._negative = negative
        self._transposed = transposed

        if self._transposed:
            self._domain = W
            self._codomain = V
        else:
            self._domain = V
            self._codomain = W
        
        # the local area in the codomain without padding
        self._idslice = tuple([slice(pad, e-s+1+pad) for pad, s, e
            in zip(self._codomain.pads, self._codomain.starts, self._codomain.ends)])

        # prepare the slices (they are of the right size then, we checked this already)
        # identity slice
        idslice = self._idslice
        
        # differentiation slice (moved by one in the direction of differentiation)
        diff_pad = self._codomain.pads[self._diffdir]
        diff_s = self._codomain.starts[self._diffdir]
        diff_e = self._codomain.ends[self._diffdir]

        # the diffslice depends on the transposition
        if self._transposed:
            diff_partslice = slice(diff_pad-1, diff_e-diff_s+1+diff_pad-1)
        else:
            diff_partslice = slice(diff_pad+1, diff_e-diff_s+1+diff_pad+1)
        
        diffslice = tuple([diff_partslice if i==self._diffdir else idslice[i]
                            for i in range(self._domain.ndim)])
        

        # define differentiation lambda based on the parameter negative (or sign)
        if self._negative:
            self._do_diff = lambda v,out: np.subtract(v._data[idslice],
                                v._data[diffslice], out=out._data[idslice])
        else:
            self._do_diff = lambda v,out: np.subtract(v._data[diffslice],
                                v._data[idslice], out=out._data[idslice])

    @property
    def domain(self):
        return self._domain

    # ...
    @property
    def codomain(self):
        return self._codomain

    # ...
    @property
    def dtype( self ):
        return self.domain.dtype

    def __truediv__(self, a):
        """ Divide by scalar. """
        return self * (1.0 / a)

    def __itruediv__(self, a):
        """ Divide by scalar, in place. """
        self *= 1.0 / a
        return self

    # ...
    def dot(self, v, out=None):
        """
        Applies the derivative operator on the given StencilVector.

        This operation will not allocate any temporary memory
        unless used in-place. (i.e. only if `v is out`)

        Parameters
        ----------
        v : StencilVector
            The input StencilVector. Has to be in the domain space.
        
        out : StencilVector | NoneType
            The output StencilVector, or None. If given, it has to be in the codomain space.
        
        Returns
        -------
        out : StencilVector
            Either a new allocation (if `out is None`), or a reference to the parameter `out`.
        """
        assert isinstance(v, StencilVector)

        # setup, space checks
        assert v.space is self._domain

        # Check if the ghost regions are up to date
        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        if out is None:
            out = self._codomain.zeros()
        
        assert isinstance(out, StencilVector)
        assert out.space is self._codomain

        # apply the differentiation and return the result
        self._do_diff(v, out)

        return out
    
    def tokronstencil(self):
        """
        Converts this KroneckerDerivativeOperator into a KroneckerStencilMatrix.

        Returns
        -------
        out : KroneckerStencilMatrix
            The resulting KroneckerStencilMatrix.
        """
        # build derivative stencil matrix (don't care for transposition here)
        # hence, use spaceV and spaceW instead of domain, codomain
        periodic_d = self._spaceV.periods[self._diffdir]
        nc  = self._spaceV.cart.domain_decomposition.ncells[self._diffdir]
        p_d = self._spaceV.pads[self._diffdir]
        n_d = self._spaceV.npts[self._diffdir]
        m_d = self._spaceW.npts[self._diffdir]

        domain_1d = DomainDecomposition([nc], [periodic_d])
        cart1_1d  = CartDecomposition( domain_1d, [n_d], [[0]], [[n_d-1]], [p_d], [1] )
        cart2_1d  = CartDecomposition( domain_1d, [m_d], [[0]], [[m_d-1]], [p_d], [1] )
        V1_d = StencilVectorSpace(cart1_1d)
        V2_d = StencilVectorSpace(cart2_1d)
        M  = StencilMatrix(V1_d, V2_d)

        # handle sign already here for now...
        sign = -1. if self._negative else 1.
        M._data[p_d:p_d+m_d, p_d]   = -1. * sign
        M._data[p_d:p_d+m_d, p_d+1] =  1. * sign
        
        # now transpose, if needed
        if self._transposed:
            M = M.T

        # identity matrices
        def make_id(i):
            nc  = self._spaceV.cart.domain_decomposition.ncells[i]
            n_i = self._domain.npts[i]
            p_i = self._domain.pads[i]
            periodic_i = self._domain.periods[i]
            domain_1d  = DomainDecomposition([nc], [periodic_i])
            cart       = CartDecomposition( domain_1d, [n_i], [[0]], [[n_i-1]], [p_i], [1] )
            #return IdentityOperator(StencilVectorSpace(cart))
            return IdentityStencilMatrix(StencilVectorSpace(cart))

        # combine to Kronecker matrix
        mats = [M if i == self._diffdir else make_id(i) for i in range(self._domain.ndim)]
        return KroneckerStencilMatrix(self._domain, self._codomain, *mats)
    
    def transpose(self, conjugate=False):
        """
        Transposes this operator. Creates and returns a new object.

        Returns
        -------
        out : DirectionalDerivativeOperator
            The transposed operator.
        """
        return DirectionalDerivativeOperator(self._spaceV, self._spaceW,
                self._diffdir, negative=self._negative, transposed=not self._transposed)

    @property
    def T(self):
        """
        Short-hand for transposing this operator. Creates and returns a new object.
        """
        return self.transpose()
    
    def __neg__(self):
        """
        Negates this operator. Creates and returns a new object.
        """
        return DirectionalDerivativeOperator(self._spaceV, self._spaceW,
                self._diffdir, negative=not self._negative, transposed=self._transposed)
    
    def toarray(self, **kwargs):
        """
        Transforms this operator into a dense matrix.

        Returns
        -------
        out : ndarray
            The resulting matrix.
        """
        return self.tosparse(**kwargs).toarray()
    
    def tosparse(self, **kwargs):

        """
        Transforms this operator into a sparse matrix in COO format.
        Includes padding in both domain and codomain which is optional, if the domain is serial,
        but mandatory if the domain is parallel.

        Parameters
        ----------
        with_pads : Bool,optional
            If true, then padding in domain and codomain direction is included. Enabled by default.

        Returns
        -------
        out : COOMatrix
            The resulting matrix.
        """
        # again, we do the transposition later

        with_pads = kwargs.pop('with_pads', False)

        # avoid this case (no pads, but parallel)
        assert not (self.domain.parallel and not with_pads)

        # begin with a 1×1 matrix
        matrix = spa.identity(1, format='coo')
        sign = -1 if self._negative else 1

        # then, iterate over all dimensions
        for d in range(self._spaceV.ndim):
            # domain and codomain sizes...
            domain_local = self._spaceV.ends[d] - self._spaceV.starts[d] + 1
            codomain_local = self._spaceW.ends[d] - self._spaceW.starts[d] + 1

            if with_pads:
                # ... potentially with pads
                domain_local += 2 * self._spaceV.pads[d]
                codomain_local += 2 * self._spaceW.pads[d]

            if self._diffdir == d:
                # if we are at the differentiation direction, construct differentiation matrix
                maindiag = np.ones(domain_local) * (-sign)
                adddiag = np.ones(domain_local) * sign

                # handle special case with not self.domain.parallel and not with_pads and periodic
                if self.domain.periods[d] and not self.domain.parallel and not with_pads:
                    # then: add element to other side of the array
                    adddiagcirc = np.array([sign])
                    offsets = (-codomain_local+1, 0, 1)
                    diags = (adddiagcirc, maindiag, adddiag)
                else:
                    # else, just take main and off diagonal
                    offsets = (0,1)
                    diags = (maindiag, adddiag)
                
                addmatrix = spa.diags(diags, offsets=offsets, shape=(codomain_local, domain_local), format='coo')
            else:
                # avoid using padding, if possible
                addmatrix = spa.identity(domain_local)
            
            # finally, take the Kronecker product
            matrix = spa.kron(matrix, addmatrix)
        
        # now, we may transpose (this should be very cheap)
        if self._transposed:
            matrix = matrix.T

        return matrix

    def copy(self):
        """
        Create an identical copy of this operator. Creates and returns a new object.
        """
        return DirectionalDerivativeOperator(self._spaceV, self._spaceW,
                self._diffdir, negative=self._negative, transposed=self._transposed)
    
    # other methods from the Matrix abstract class
    # (mostly delegating work to the KroneckerStencilMatrix)
    # (i.e. these are not really meant to be used in practice)

    def __mul__(self, a):
        return self.tokronstencil() * a

    def __add__(self, m):
        return self.tokronstencil() + m

    def __sub__(self, m):
        return self.tokronstencil() - m

    # we cannot allow in-place operations
    # (except with an explicit neutral element, if there was one)

    def __imul__(self, a):
        if isinstance(a, IdentityMatrix): return
        if a == 1: return
        raise NotImplementedError("Not supported for this class.")

    def __iadd__(self, m):
        if m == 0: return
        raise NotImplementedError("Not supported for this class.")

    def __isub__(self, m):
        if m == 0: return
        raise NotImplementedError("Not supported for this class.")

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
    
    def __call__(self, u):
        assert isinstance(u, FemField)
        assert u.space == self.domain

        coeffs = self.matrix.dot(u.coeffs)

        return FemField(self.codomain, coeffs=coeffs)

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

        self._domain   = H1
        self._codomain = L2
        self._matrix   = DirectionalDerivativeOperator(H1.vector_space, L2.vector_space, 0)

#====================================================================================================
class Gradient_2D(DiffOperator):
    """
    Gradient operator in 2D.

    Parameters
    ----------
    H1 : 2D TensorFemSpace
        Domain of gradient operator.

    Hcurl : 2D VectorFemSpace
        Codomain of gradient operator.

    """
    def __init__(self, H1, Hcurl):

        assert isinstance(   H1,  TensorFemSpace); assert    H1.ldim == 2
        assert isinstance(Hcurl, VectorFemSpace); assert Hcurl.ldim == 2

        assert Hcurl.spaces[0].periodic == H1.periodic
        assert Hcurl.spaces[1].periodic == H1.periodic

        assert tuple(Hcurl.spaces[0].degree) == (H1.degree[0]-1, H1.degree[1]  )
        assert tuple(Hcurl.spaces[1].degree) == (H1.degree[0]  , H1.degree[1]-1)

        # Tensor-product spaces of coefficients - domain
        B_B = H1.vector_space

        # Tensor-product spaces of coefficients - codomain
        (M_B, B_M) = Hcurl.vector_space.spaces

        # Build Gradient matrix block by block
        blocks = [[DirectionalDerivativeOperator(B_B, M_B, 0)],
                  [DirectionalDerivativeOperator(B_B, B_M, 1)]]
        matrix = BlockLinearOperator(H1.vector_space, Hcurl.vector_space, blocks=blocks)

        # Store data in object
        self._domain   = H1
        self._codomain = Hcurl
        self._matrix   = matrix

#====================================================================================================
class Gradient_3D(DiffOperator):
    """
    Gradient operator in 3D.

    Parameters
    ----------
    H1 : 3D TensorFemSpace
        Domain of gradient operator.

    Hcurl : 3D VectorFemSpace
        Codomain of gradient operator.

    """
    def __init__(self, H1, Hcurl):

        assert isinstance(   H1,  TensorFemSpace); assert    H1.ldim == 3
        assert isinstance(Hcurl, VectorFemSpace); assert Hcurl.ldim == 3

        assert Hcurl.spaces[0].periodic == H1.periodic
        assert Hcurl.spaces[1].periodic == H1.periodic
        assert Hcurl.spaces[2].periodic == H1.periodic

        assert tuple(Hcurl.spaces[0].degree) == (H1.degree[0]-1, H1.degree[1]  , H1.degree[2]  )
        assert tuple(Hcurl.spaces[1].degree) == (H1.degree[0]  , H1.degree[1]-1, H1.degree[2]  )
        assert tuple(Hcurl.spaces[2].degree) == (H1.degree[0]  , H1.degree[1]  , H1.degree[2]-1)

        # Tensor-product spaces of coefficients - domain
        B_B_B = H1.vector_space

        # Tensor-product spaces of coefficients - codomain
        (M_B_B, B_M_B, B_B_M) = Hcurl.vector_space.spaces

        # Build Gradient matrix block by block
        blocks = [[DirectionalDerivativeOperator(B_B_B, M_B_B, 0)],
                  [DirectionalDerivativeOperator(B_B_B, B_M_B, 1)],
                  [DirectionalDerivativeOperator(B_B_B, B_B_M, 2)]]
        matrix = BlockLinearOperator(H1.vector_space, Hcurl.vector_space, blocks=blocks)

        # Store data in object
        self._domain   = H1
        self._codomain = Hcurl
        self._matrix   = matrix

#====================================================================================================
class ScalarCurl_2D(DiffOperator):
    """
    Scalar curl operator in 2D: computes a scalar field from a vector field.

    Parameters
    ----------
    Hcurl : 2D VectorFemSpace
        Domain of 2D scalar curl operator.

    L2 : 2D TensorFemSpace
        Codomain of 2D scalar curl operator.

    """
    def __init__(self, Hcurl, L2):

        assert isinstance(Hcurl, VectorFemSpace); assert Hcurl.ldim == 2
        assert isinstance(   L2,  TensorFemSpace); assert    L2.ldim == 2

        assert Hcurl.spaces[0].periodic == L2.periodic
        assert Hcurl.spaces[1].periodic == L2.periodic

        assert tuple(Hcurl.spaces[0].degree) == (L2.degree[0]  , L2.degree[1]+1)
        assert tuple(Hcurl.spaces[1].degree) == (L2.degree[0]+1, L2.degree[1]  )

        # Tensor-product spaces of coefficients - domain
        (M_B, B_M) = Hcurl.vector_space.spaces

        # Tensor-product spaces of coefficients - codomain
        M_M = L2.vector_space

        # Build Curl matrix block by block
        blocks = [[-DirectionalDerivativeOperator(M_B, M_M, 1),
                  DirectionalDerivativeOperator(B_M, M_M, 0)]]
        matrix = BlockLinearOperator(Hcurl.vector_space, L2.vector_space, blocks=blocks)

        # Store data in object
        self._domain   = Hcurl
        self._codomain = L2
        self._matrix   = matrix

#====================================================================================================
class VectorCurl_2D(DiffOperator):
    """
    Vector curl operator in 2D: computes a vector field from a scalar field.
    This is sometimes called the 'rot' operator.

    Parameters
    ----------
    H1 : 2D TensorFemSpace
        Domain of 2D vector curl operator.

    Hdiv : 2D VectorFemSpace
        Codomain of 2D vector curl operator.

    """
    def __init__(self, H1, Hdiv):

        assert isinstance(  H1,  TensorFemSpace); assert   H1.ldim == 2
        assert isinstance(Hdiv, VectorFemSpace); assert Hdiv.ldim == 2

        assert Hdiv.spaces[0].periodic == H1.periodic
        assert Hdiv.spaces[1].periodic == H1.periodic

        assert tuple(Hdiv.spaces[0].degree) == (H1.degree[0]  , H1.degree[1]-1)
        assert tuple(Hdiv.spaces[1].degree) == (H1.degree[0]-1, H1.degree[1]  )

        # Tensor-product spaces of coefficients - domain
        B_B = H1.vector_space

        # Tensor-product spaces of coefficients - codomain
        (B_M, M_B) = Hdiv.vector_space.spaces

        # Build Curl matrix block by block
        blocks = [[DirectionalDerivativeOperator(B_B, B_M, 1)],
                  [-DirectionalDerivativeOperator(B_B, M_B, 0)]]
        matrix = BlockLinearOperator(H1.vector_space, Hdiv.vector_space, blocks=blocks)

        # Store data in object
        self._domain   = H1
        self._codomain = Hdiv
        self._matrix   = matrix

#====================================================================================================
class Curl_3D(DiffOperator):
    """
    Curl operator in 3D.

    Parameters
    ----------
    Hcurl : 3D VectorFemSpace
        Domain of 3D curl operator.

    Hdiv : 3D VectorFemSpace
        Codomain of 3D curl operator.

    """
    def __init__(self, Hcurl, Hdiv):

        assert isinstance(Hcurl, VectorFemSpace); assert Hcurl.ldim == 3
        assert isinstance( Hdiv, VectorFemSpace); assert  Hdiv.ldim == 3

        assert Hcurl.spaces[0].periodic == Hdiv.spaces[0].periodic
        assert Hcurl.spaces[1].periodic == Hdiv.spaces[1].periodic
        assert Hcurl.spaces[2].periodic == Hdiv.spaces[2].periodic

        Hdiv0, Hdiv1, Hdiv2 = Hdiv.spaces
        assert tuple(Hcurl.spaces[1].degree) == (Hdiv0.degree[0]  , Hdiv0.degree[1]  , Hdiv0.degree[2]+1)
        assert tuple(Hcurl.spaces[2].degree) == (Hdiv0.degree[0]  , Hdiv0.degree[1]+1, Hdiv0.degree[2]  )
        assert tuple(Hcurl.spaces[0].degree) == (Hdiv1.degree[0]  , Hdiv1.degree[1]  , Hdiv1.degree[2]+1)
        assert tuple(Hcurl.spaces[2].degree) == (Hdiv1.degree[0]+1, Hdiv1.degree[1]  , Hdiv1.degree[2]  )
        assert tuple(Hcurl.spaces[0].degree) == (Hdiv2.degree[0]  , Hdiv2.degree[1]+1, Hdiv2.degree[2]  )
        assert tuple(Hcurl.spaces[1].degree) == (Hdiv2.degree[0]+1, Hdiv2.degree[1]  , Hdiv2.degree[2]  )

        # Tensor-product spaces of coefficients - domain
        (M_B_B, B_M_B, B_B_M) = Hcurl.vector_space.spaces

        # Tensor-product spaces of coefficients - codomain
        (B_M_M, M_B_M, M_M_B) = Hdiv.vector_space.spaces

        # ...
        # Build Curl matrix block by block
        D = DirectionalDerivativeOperator
        blocks = [[       None         , -D(B_M_B, B_M_M, 2) ,  D(B_B_M, B_M_M, 1)],
                  [ D(M_B_B, M_B_M, 2) ,        None,          -D(B_B_M, M_B_M, 0)],
                  [-D(M_B_B, M_M_B, 1) ,  D(B_M_B, M_M_B, 0) ,        None        ]]

        matrix = BlockLinearOperator(Hcurl.vector_space, Hdiv.vector_space, blocks=blocks)
        # ...

        # Store data in object
        self._domain   = Hcurl
        self._codomain = Hdiv
        self._matrix   = matrix

#====================================================================================================
class Divergence_2D(DiffOperator):
    """
    Divergence operator in 2D.

    Parameters
    ----------
    Hdiv : 2D VectorFemSpace
        Domain of divergence operator.

    L2 : 2D TensorFemSpace
        Codomain of divergence operator.

    """
    def __init__(self, Hdiv, L2):

        assert isinstance(Hdiv,  VectorFemSpace); assert Hdiv.ldim == 2
        assert isinstance(  L2,  TensorFemSpace); assert   L2.ldim == 2

        assert Hdiv.spaces[0].periodic == L2.periodic
        assert Hdiv.spaces[1].periodic == L2.periodic

        assert tuple(Hdiv.spaces[0].degree) == (L2.degree[0]+1, L2.degree[1]  )
        assert tuple(Hdiv.spaces[1].degree) == (L2.degree[0]  , L2.degree[1]+1)

        # Tensor-product spaces of coefficients - domain
        (B_M, M_B) = Hdiv.vector_space.spaces

        # Tensor-product spaces of coefficients - codomain
        M_M = L2.vector_space

        # Build Divergence matrix block by block
        f = KroneckerStencilMatrix
        blocks = [[DirectionalDerivativeOperator(B_M, M_M, 0), DirectionalDerivativeOperator(M_B, M_M, 1)]]
        matrix = BlockLinearOperator(Hdiv.vector_space, L2.vector_space, blocks=blocks) 

        # Store data in object
        self._domain   = Hdiv
        self._codomain = L2
        self._matrix   = matrix

#====================================================================================================
class Divergence_3D(DiffOperator):
    """
    Divergence operator in 3D.

    Parameters
    ----------
    Hdiv : 3D VectorFemSpace
        Domain of divergence operator.

    L2 : 3D TensorFemSpace
        Codomain of divergence operator.

    """
    def __init__(self, Hdiv, L2):

        assert isinstance(Hdiv,  VectorFemSpace); assert Hdiv.ldim == 3
        assert isinstance(  L2,  TensorFemSpace); assert   L2.ldim == 3

        assert Hdiv.spaces[0].periodic == L2.periodic
        assert Hdiv.spaces[1].periodic == L2.periodic
        assert Hdiv.spaces[2].periodic == L2.periodic

        assert tuple(Hdiv.spaces[0].degree) == (L2.degree[0]+1, L2.degree[1]  , L2.degree[2]  )
        assert tuple(Hdiv.spaces[1].degree) == (L2.degree[0]  , L2.degree[1]+1, L2.degree[2]  )
        assert tuple(Hdiv.spaces[2].degree) == (L2.degree[0]  , L2.degree[1]  , L2.degree[2]+1)

        # Tensor-product spaces of coefficients - domain
        (B_M_M, M_B_M, M_M_B) = Hdiv.vector_space.spaces

        # Tensor-product spaces of coefficients - codomain
        M_M_M = L2.vector_space

        # Build Divergence matrix block by block
        blocks = [[DirectionalDerivativeOperator(B_M_M, M_M_M, 0),
                   DirectionalDerivativeOperator(M_B_M, M_M_M, 1),
                   DirectionalDerivativeOperator(M_M_B, M_M_M, 2)]]
        matrix = BlockLinearOperator(Hdiv.vector_space, L2.vector_space, blocks=blocks) 

        # Store data in object
        self._domain   = Hdiv
        self._codomain = L2
        self._matrix   = matrix
