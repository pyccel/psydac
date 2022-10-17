import numpy as np

from psydac.linalg2.basic import VectorSpace, Vector, LinearOperator

__all__ = ("NdarrayVectorSpace", "NdarrayVector", "NdarrayLinearOperator",)

class NdarrayVectorSpace( VectorSpace ):
    """
    space of real ndarrays, dtype only allowing float right now
    """
    def __init__( self, dim, dtype=float ):
        assert np.isscalar(dim)
        self._dim = dim
        self._dtype = dtype

    @property
    def dimension( self ):
        return self._dim

    @property
    def dtype( self ):
        return self._dtype

    def zeros( self ):
        return NdarrayVector(space=self)

    def ones( self ):
        return NdarrayVector(space=self, data=np.ones(self._dim, dtype=self._dtype))


class NdarrayVector( Vector ):
    def __init__( self, space, data=None ):
        
        assert isinstance(space, NdarrayVectorSpace)
        self._space = space        

        if data is None:
            self._data = np.zeros(space.dimension, dtype=space.dtype)
        elif isinstance(data, np.ndarray):
            assert data.shape == (space.dimension, ), f"data neither scalar nor of right dimension"
            assert data.dtype == space.dtype, f"data of wrong dtype"                                     
            self._data = data
        elif np.isscalar(data):
            self._data = np.full(shape=space.dimension, fill_value=data, dtype=space.dtype)
        else:
            raise ValueError(data)

    @property
    def data( self ):
        return self._data

    @property
    def space( self ):
        return self._space

    @property
    def dtype( self ):
        return self.space.dtype

    def dot( self, v ):
        assert isinstance(v, NdarrayVector), f"v is not a NdarrayVector"
        assert self.space is v.space, f"v and self dont belong to the same space"
        return np.dot(self._data, v.data)

    def __mul__( self, a ):
        assert np.isscalar(a), f"a is not a scalar"
        return NdarrayVector(space=self._space, data=np.multiply(self._data,a))

    def __rmul__( self, a ):
        return self * a

    def __imul__( self, a ):
        self = self * a
        return self    

    def __add__( self, v ):
        assert isinstance(v, NdarrayVector), f"v is not NdarrayVector"
        assert self.space is v.space, f"v space is not self space"
        return NdarrayVector(space = self._space, data = self._data + v.data)

    def __iadd__( self, v ):
        assert isinstance(v, NdarrayVector), f"v is not NdarrayVector"
        assert self.space is v.space, f"v space is not self space"
        self._data += v.data
        return self

    def __neg__( self ):
        return self * (-1)

    def __sub__(self, v ):
        assert isinstance(v, NdarrayVector)
        assert self.space is v.space
        return NdarrayVector(space = self._space, data = self._data - v.data)

    def __isub__(self, v ):
        assert isinstance(v, NdarrayVector)
        assert self.space is v.space
        self._data -= v.data
        return self


class NdarrayLinearOperator( LinearOperator ):
    def __init__( self, domain=None, codomain=None, matrix=None ):

        assert domain
        assert isinstance(domain,NdarrayVectorSpace)
        self._domain = domain
        if codomain:
            assert isinstance(codomain,NdarrayVectorSpace)
            self._codomain = codomain
        else:
            self._codomain = domain
        if matrix is not None:
            assert np.shape(matrix) == (self._codomain.dimension, self._domain.dimension)
            self._matrix = matrix
            self._dtype = matrix.dtype
        else:
            self._matrix = None

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def matrix( self ):
        return self._matrix

    @property
    def dtype( self ):
        if self._matrix is not None:
            return self._dtype
        else:
            raise NotImplementedError('Class does not provide a dtype method without a matrix')    

    def dot( self, v ):
        assert isinstance(v, NdarrayVector)
        assert v.space == self.domain
        if self._matrix is not None:
            return NdarrayVector(space=self._codomain, data=np.dot(self._matrix,v._data))
        else:
            raise NotImplementedError('Class does not provide a dot() method without a matrix')