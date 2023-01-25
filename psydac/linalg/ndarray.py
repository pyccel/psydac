import numpy as np

from psydac.linalg.basic import VectorSpace, Vector, LinearOperator, ZeroOperator

__all__ = ("NdarrayVectorSpace", "NdarrayVector", "NdarrayLinearOperator",)

class NdarrayVectorSpace( VectorSpace ):
    """
    space of real ndarrays, dtype only allowing float right now
    """
    def __init__( self, dimension, dtype=float ):
        assert np.isscalar(dimension)
        assert dimension >= 1
        self._dimension = dimension
        self._dtype = dtype

    @property
    def dimension( self ):
        return self._dimension

    @property
    def dtype( self ):
        return self._dtype

    def zeros( self ):
        return NdarrayVector(space=self)


class NdarrayVector( Vector ):
    """
    test
    
    """
    def __init__( self, space, data=None ):
        """
        test3
        
        """
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
        """ Returns a np.ndarray representing self. """
        return self._data

    @property
    def space( self ):
        """ Returns the NdarrayVectorSpace self belongs to. """
        return self._space

    @property
    def dtype( self ):
        """ Returns the dtype attribute of the NdarrayVectorSpace self belongs to. """
        return self.space.dtype

    def copy( self, out=None ):
        """ Returns a copy of self. """
        return NdarrayVector(self._space, np.copy(self._data))

    def toarray( self, **kwargs ):
        return self._data

    def dot( self, v ):
        """ Computes the canonical scalar product between self and an NdarrayVector v belonging to the same NdarrayVectorSpace as self. """
        assert isinstance(v, NdarrayVector), f"v is not a NdarrayVector"
        assert self.space is v.space, f"v and self dont belong to the same space"
        return np.dot(self._data, v.data)

    def __mul__( self, c ):
        """ Returns a new NdarrayVector whose components have been multiplied by c. """
        assert np.isscalar(c), f"c is not a scalar"
        return NdarrayVector(space=self._space, data=np.multiply(self._data,c))

    def __rmul__( self, c ):
        """ As __mul__. """
        return self * c

    def __imul__( self, c ):
        """ Multiplies own components by c, does not create new NdarrayVector object and does not return. """
        self._data = self._data * c
        return self    

    def __add__( self, v ):
        """ Creates new NdarrayVector object whose components are the sum of the components of self and v. """
        assert isinstance(v, NdarrayVector), f"v is not NdarrayVector"
        assert self.space is v.space, f"v space is not self space"
        return NdarrayVector(space = self._space, data = self._data + v.data)

    def __iadd__( self, v ):
        """ Adds v to self, does not create new NdarrayVector object and does not return. """
        assert isinstance(v, NdarrayVector), f"v is not NdarrayVector"
        assert self.space is v.space, f"v space is not self space"
        self._data += v.data
        return self

    def __neg__( self ):
        """ Creates new NdarrayVector object whose components have been negated. """
        return self * (-1)

    def __sub__(self, v ):
        """ Creates new NdarrayVector object whose components are the difference between the components of self and v. """
        assert isinstance(v, NdarrayVector)
        assert self.space is v.space
        return NdarrayVector(space = self._space, data = self._data - v.data)

    def __isub__(self, v ):
        """ Substracts v from self, does not create new NdarrayVector object and does not return. """
        assert isinstance(v, NdarrayVector)
        assert self.space is v.space
        self._data -= v.data
        return self


class NdarrayLinearOperator( LinearOperator ):
    """
    test2
    
    """
    def __new__( cls, domain=None, codomain=None, matrix=None ):

        assert domain
        assert isinstance(domain, NdarrayVectorSpace)
        if matrix is not None:
            if codomain:
                assert isinstance(codomain, NdarrayVectorSpace)
                assert np.shape(matrix) == (codomain.dimension, domain.dimension)
                if (matrix == np.zeros((codomain.dimension, domain.dimension))).all():
                    return ZeroOperator(domain=domain, codomain=codomain)
            else:
                assert np.shape(matrix) == (domain.dimension, domain.dimension)
                if (matrix == np.zeros((domain.dimension, domain.dimension))).all():
                    return ZeroOperator(domain=domain, codomain=domain)
        return super().__new__(cls)

    def __init__( self, domain=None, codomain=None, matrix=None ):

        self._domain = domain
        if codomain:
            assert isinstance(codomain,NdarrayVectorSpace)
            self._codomain = codomain
        else:
            self._codomain = domain
        if matrix is not None:
            self._matrix = matrix
            self._dtype = matrix.dtype
        else:
            self._matrix = None

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    def domain( self ):
        """ Returns the domain of self, an object of class NdarrayVectorSpace. """
        return self._domain

    @property
    def codomain( self ):
        """ Returns the codomain of self, an object of class NdarrayVectorSpace. """
        return self._codomain

    @property
    def matrix( self ):
        """ If given, returns a np.ndarray matrix representing self, else returns None. """
        return self._matrix

    @property
    def dtype( self ):
        """ If a np.ndarray matrix is given, returns its dtype, else raises a NotImplementedError. """
        if self._matrix is not None:
            return self._dtype
        else:
            raise NotImplementedError('Class does not provide a dtype method without a matrix')

    def toarray(self):
        return self._matrix

    def tosparse(self):
        from scipy.sparse import csr_matrix
        return csr_matrix(self._matrix)

    def transpose( self ):
        if self._matrix is not None:
            return NdarrayLinearOperator(domain=self._codomain, codomain=self._domain, matrix=self._matrix.transpose())
        else:
            raise NotImplementedError('NdarrayLinearOperator can`t be transposed if no matrix is given.')

    def dot( self, v, out=None ):
        """ Evaluates self at v if v belongs to domain; creates a new object of the codomain class; makes use of np.dot. Needs update due to new out """
        assert isinstance(v, NdarrayVector)
        assert v.space == self.domain
        if self._matrix is not None:
            if out is not None:
                assert isinstance(out, Vector)
                assert out.space == self._codomain
                out._data = np.dot(self._matrix, v._data)
                return out
            else:
                return NdarrayVector(space=self._codomain, data=np.dot(self._matrix,v._data))
        else:
            raise NotImplementedError('Class does not provide a dot() method without a matrix')

    def __add__( self, B ):
        """
        Returns a new NdarrayLinearOperator object if both addends belong to said class, representing their sum. 
        Else calls parents __add__ method, eventually creating a SumLinearOperator object. 

        """
        assert isinstance(B, LinearOperator)
        if isinstance(B, NdarrayLinearOperator):
            assert self._domain == B.domain
            assert self._codomain == B.codomain
            return NdarrayLinearOperator(domain=self._domain, codomain=self._codomain, matrix=self._matrix+B.matrix)
        else:
            return super().__add__(B)