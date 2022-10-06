
from multiprocessing.sharedctypes import Value
import numpy as np
from psydac.linalg2.basic import VectorSpace, Vector, LinearOperator

class NdarrayVectorSpace(VectorSpace):
    """
    space of ndarrays
    """
    def __init__(self, dim, dtype):
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

class NdarrayVector(Vector):
    def __init__(self, space, data=None):
        
        self._space = space        

        if data is None:
            self._data = np.zeros(space.dimension, dtype=space.dtype)
        elif isinstance(data, np.ndarray):
            assert data.shape == (space.dimension, ) and data.dtype == space.dtype
            self._data = data
        elif np.isscalar(data):
            # just a test
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
        assert isinstance(v, NdarrayVector)
        assert self.space is v.space
        return np.dot(self._data, v._data)

class NdarrayLinearOperator(LinearOperator):
    def __init__(self):
        pass

#===============================================================================
if __name__ == "__main__":

    V = NdarrayVectorSpace(dim=5, dtype=float)
    a = NdarrayVector(space=V)
    b = NdarrayVector(space=V, data=1)

    p = a.dot(b)

    if p != 0:
        print(' strange ')

    print( a._data )
    print( b._data )

    c = a+b


