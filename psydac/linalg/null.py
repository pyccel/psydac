# coding: utf-8
#
from psydac.linalg.basic   import LinearOperator, Matrix, Vector, VectorSpace, NullElement
from psydac.linalg.stencil import StencilMatrix

from numpy        import zeros as dense_null
from scipy.sparse import zeros as sparse_null

__all__ = ['NullLinearOperator', 'NullMatrix', 'NullStencilMatrix']

class NullLinearOperator(LinearOperator, NullElement):

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

    @property
    def codomain( self ):
        return self._W

    def dot( self, v, out=None ):
        # no need to care for v

        if out is not None:
            # find a way to handle out not None
            raise NotImplementedError()

        return self.codomain.zeros()

class NullMatrix( Matrix, IdentityLinearOperator, NullElement ):

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    def toarray( self ):
        return dense_null(*self.shape)

    def tosparse( self ):
        return sparse_null(*self.shape)

class NullStencilMatrix( StencilMatrix, NullElement ):
    def __init__(self, V, W, p=None):
        super().__init__(V, V, W, pads=(p,))

    #-------------------------------------
    # Deferred methods
    #-------------------------------------

    def dot( self, v, out=None ):
        # no need to care for v

        if out is not None:
            # find a way to handle out not None
            raise NotImplementedError()
            
        return self.codomain.zeros()

