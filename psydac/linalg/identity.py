"""
# coding: utf-8
#
from psydac.linalg.basic   import LinearOperator, Matrix, Vector, VectorSpace
from psydac.linalg.stencil import StencilMatrix

from numpy        import eye as dense_id
from scipy.sparse import eye as sparse_id

__all__ = ['IdentityLinearOperator', 'IdentityMatrix', 'IdentityStencilMatrix']

class IdentityLinearOperator(LinearOperator):

    def __init__(self, V):
        assert isinstance( V, VectorSpace )
        self._V  = V

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain( self ):
        return self._V

    # ...
    @property
    def codomain( self ):
        return self._V

    # ...
    @property
    def dtype( self ):
        return self.domain.dtype

    def toarray(self):
        raise NotImplementedError('toarray() is not defined for this outdated class.')

    def tosparse(self):
        raise NotImplementedError('tosparse() is not defined for this outdated class.')

    # ...
    def dot( self, v, out=None ):
"""
"""
        Returns the input vector. If out is None or v is the same vector object as out (`v is out`), v is returned (no copy).
        In all other cases, v is copied to out, and out is returned.

        Parameters
        ----------
        v : Vector
            The vector to return.
        
        out : Vector | None
            Output vector. Has to be either none, or a vector from the same space as v. Behavior is described above.
        
        Returns
        -------
        Described above.
"""
"""
        assert isinstance( v, Vector )
        assert v.space is self.domain

        if out is not None and out is not v:
            assert isinstance(out, Vector)
            assert v.space is out.space
            out *= 0.0
            out += v
            return out

        return v

class IdentityMatrix( Matrix, IdentityLinearOperator ):

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    def toarray( self ):
        if hasattr(self.codomain, 'dtype'):
            return dense_id(*self.shape, dtype=self.codomain.dtype)
        else:
            return dense_id(*self.shape)

    def tosparse( self ):
        if hasattr(self.codomain, 'dtype'):
            return sparse_id(*self.shape, dtype=self.codomain.dtype)
        else:
            return sparse_id(*self.shape)
    
    def copy(self):
        return IdentityMatrix(self.domain)

    def __neg__(self):
        raise NotImplementedError()

    def __mul__(self, a):
        raise NotImplementedError()

    def __rmul__(self, a):
        raise NotImplementedError()

    def __add__(self, m):
        raise NotImplementedError()

    def __sub__(self, m):
        raise NotImplementedError()

    def __imul__(self, a):
        raise NotImplementedError()

    def __iadd__(self, m):
        raise NotImplementedError()

    def __isub__(self, m):
        raise NotImplementedError()

class IdentityStencilMatrix( StencilMatrix ):
    def __init__(self, V, pads=None):
        assert pads is None or len(pads) == V.ndim

        super().__init__(V, V, pads=pads)

        idslice = (*((slice(None),) * V.ndim), *self.pads)
        self._data[idslice] = 1.

    #-------------------------------------
    # Deferred methods
    #-------------------------------------

    def dot( self, v, out=None ):
"""
"""
        Returns the input vector. If out is None, or v is the same vector object as out (`v is out`), v is returned (no copy).
        In all other cases, v is copied to out, and out is returned.

        Parameters
        ----------
        v : Vector
            The vector to return.
        
        out : Vector | None
            Output vector. Has to be either none, or a vector from the same space as v. Behavior is described above.
        
        Returns
        -------
        Described above.
"""
"""
        assert isinstance( v, Vector )
        assert v.space is self.domain
        
        if out is not None and out is not v:
            assert isinstance(out, Vector)
            assert v.space is out.space
            out *= 0.0
            out += v
            return out

        return v

"""