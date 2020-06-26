# coding: utf-8
#
from psydac.linalg.basic import LinearOperator, Vector, VectorSpace
from psydac.linalg.basic import Matrix

from numpy        import eye as dense_id
from scipy.sparse import eye as sparse_id

__all__ = ['IdentityLinearOperator', 'IdentityMatrix']

class IdentityLinearOperator(LinearOperator):

    def __init__(self, V):
        assert isinstance( V, VectorSpace )
        self._V  = V

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    def domain( self ):
        return self._V

    @property
    def codomain( self ):
        return self._V

    def dot( self, v, out=None ):
        assert isinstance( v, Vector )
        assert v.space is self.domain
        # find a way to handle out not None
        return v

class IdentityMatrix( Matrix, IdentityLinearOperator ):

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    def toarray( self ):
        return dense_id(*self.shape)

    def tosparse( self ):
        return sparse_id(*self.shape)
    
