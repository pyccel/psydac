# coding: utf-8
#
from psydac.linalg.basic   import LinearOperator, Matrix, Vector, VectorSpace
from psydac.linalg.stencil import StencilMatrix

from numpy        import zeros as dense_null
from scipy.sparse import coo_matrix as sparse_null

__all__ = ['NullLinearOperator', 'NullMatrix', 'NullStencilMatrix']

class NullLinearOperator(LinearOperator):

    def __init__(self, V, W):
        assert isinstance( V, VectorSpace )
        assert isinstance( W, VectorSpace )
        self._V  = V
        self._W  = W

    #-------------------------------------
    # Deferred methods
    #-------------------------------------

    @property
    def domain( self ):
        return self._V

    # ...
    @property
    def codomain( self ):
        return self._W

    # ...
    @property
    def dtype( self ):
        return self.domain.dtype

    def toarray(self):
        raise NotImplementedError('toarray() is not defined for this outdated class.')

    def tosparse(self):
        raise NotImplementedError('tosparse() is not defined for this outdated class.')

    def dot( self, v, out=None ):
        """
        Returns a zero vector. If out is not None, then out is zeroed and returned (otherwise, a new vector is created).

        Parameters
        ----------
        v : Vector
            Ignored. (it is not even checked if v is a Vector at all)
        
        out : Vector | None
            Output vector. Has to be either none, or a vector from any space. Behavior is described above.
        
        Returns
        -------
        Described above.
        """
        # no need to care for v

        if out is not None:
            assert isinstance(out, Vector)
            out *= 0.0
            return out

        return self.codomain.zeros()

class NullMatrix( Matrix, NullLinearOperator ):

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    def toarray( self ):
        if hasattr(self.codomain, 'dtype'):
            return dense_null(*self.shape, dtype=self.codomain.dtype)
        else:
            return dense_null(*self.shape)

    def tosparse( self ):
        if hasattr(self.codomain, 'dtype'):
            return sparse_null(*self.shape, dtype=self.codomain.dtype)
        else:
            return sparse_null(*self.shape)
    
    def copy(self):
        return NullMatrix(self.domain, self.codomain)

    def __neg__(self):
        return self

    def __mul__(self, a):
        return self

    def __rmul__(self, a):
        return self

    def __add__(self, m):
        return m

    def __sub__(self, m):
        return -m

    def __imul__(self, a):
        return self

    def __iadd__(self, m):
        raise NotImplementedError()

    def __isub__(self, m):
        raise NotImplementedError()

class NullStencilMatrix( StencilMatrix ):
    def __init__(self, V, W, pads=None):
        super().__init__(V, W, pads=pads)

    #-------------------------------------
    # Deferred methods
    #-------------------------------------

    def dot( self, v, out=None ):
        """
        Returns a zero vector. If out is not None, then out is zeroed and returned (otherwise, a new vector is created).

        Parameters
        ----------
        v : Vector
            Ignored. (it is not even checked if v is a Vector at all)
        
        out : Vector | None
            Output vector. Has to be either none, or a vector from any space. Behavior is described above.
        
        Returns
        -------
        Described above.
        """
        # no need to care for v

        if out is not None:
            assert isinstance(out, Vector)
            out *= 0.0
            return out
            
        return self.codomain.zeros()
