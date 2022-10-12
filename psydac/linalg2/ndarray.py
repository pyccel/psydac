from abc   import ABCMeta, abstractmethod

import numpy as np
from numpy import ndarray

from psydac.linalg2.basic import VectorSpace, Vector, LinearOperator

__all__ = ("NdarrayVectorSpace", "NdarrayVector", "NdarrayLinearOperator",)

class NdarrayVectorSpace( VectorSpace ):
    """
    space of real ndarrays, dtype only allowing float right now
    """
    def __init__( self, dim, dtype=float ):
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
    def space( self ):
        return self._space

    @property
    def dtype( self ):
        return self._dtype

    def dot( self, v ):
        assert isinstance(v, NdarrayVector), f"v is not a NdarrayVector"
        assert self.space is v.space, f"v and self dont belong to the same space"
        return np.dot(self._data, v._data)

    def __mul__( self, a ):
        assert np.isscalar(a), f"a is not a scalar"
        return NdarrayVector(space=self._space, data=np.multiply(self._data,a))

    def __rmul__( self, a ):
        assert np.isscalar(a), f"a is not a scalar"
        return NdarrayVector(space=self._space, data=np.multiply(self._data,a))

    def __add__( self, v ):
        assert isinstance(v, NdarrayVector), f"v is not NdarrayVector"
        assert self.space is v.space, f"v space is not self space"
        return NdarrayVector(space=self._space, data=self._data+v._data)


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
            assert np.shape(matrix)[1] == self._domain.dimension
            assert np.shape(matrix)[0] == self._codomain.dimension
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

    def __add__( self, B ):
        from psydac.linalg2.expr import SumLinearOperator
        return SumLinearOperator(self,B)

    def __matmul__( self, B ):
        from psydac.linalg2.expr import ConvLinearOperator
        return ConvLinearOperator(self,B)

    def __mul__( self, c ):
        from psydac.linalg2.expr import ZeroOperator, ScalLinearOperator
        assert np.isscalar(c)
        if c==0:
            return ZeroOperator(domain=self._domain, codomain=self._codomain)
        elif c == 1:
            return self
        else:
            return ScalLinearOperator(c, self)

    def __rmul__( self, c ):
        from psydac.linalg2.expr import ZeroOperator, ScalLinearOperator
        assert np.isscalar(c)
        if c==0:
            return ZeroOperator(domain=self._domain, codomain=self._codomain)
        elif c == 1:
            return self
        else:
            return ScalLinearOperator(c, self)

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def idot( self, v, out ):
        assert isinstance(v, NdarrayVector)
        assert v.space == self._domain
        if out is not None:
            assert isinstance(out, NdarrayVector)
            assert out.space == self._codomain
            out += self.dot(v)
            return out
        else:
            return self.dot(v)