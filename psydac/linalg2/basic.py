# coding: utf-8
#
# Copyright 2018 Yaman Güçlü, Jalal Lakhlili

from abc   import ABCMeta, abstractmethod
import numpy as np

__all__ = ['VectorSpace', 'Vector', 'LinearOperator', 'SumLinearOperator', 'CompLinearOperator', 'ScalLinearOperator', 'ZeroOperator', 'IdOperator', ]

#===============================================================================
class VectorSpace( metaclass=ABCMeta ):
    """
    Generic vector space V.

    """
    @property
    @abstractmethod
    def dimension( self ):
        """
        The dimension of a vector space V is the cardinality
        (i.e. the number of vectors) of a basis of V over its base field.

        """

    @property
    @abstractmethod
    def dtype( self ):
        """
        The data type of the space elements.
        """

    @abstractmethod
    def zeros( self ):
        """
        Get a copy of the null element of the vector space V.

        Returns
        -------
        null : Vector
            A new vector object with all components equal to zero.

        """

#===============================================================================
class Vector( metaclass=ABCMeta ):
    """
    Element of a (normed) vector space V.

    """
    @property
    def shape( self ):
        return (self.space.dimension, )

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def space( self ):
        pass

    @property
    @abstractmethod
    def dtype( self ):
        pass

    @property
    @abstractmethod
    def data( self ):
        pass

    @abstractmethod
    def dot( self, v ):
        pass

    # @abstractmethod
    # def toarray( self, **kwargs ):
    #     """ Convert to Numpy 1D array. """

    @abstractmethod
    def copy( self ):
        pass

    @abstractmethod
    def __neg__( self ):
        pass

    @abstractmethod
    def __mul__( self, a ):
        pass

    @abstractmethod
    def __rmul__( self, a ):
        pass

    @abstractmethod
    def __add__( self, v ):
        pass

    @abstractmethod
    def __sub__( self, v ):
        pass

    @abstractmethod
    def __imul__( self, a ):
        pass

    @abstractmethod
    def __iadd__( self, v ):
        pass

    @abstractmethod
    def __isub__( self, v ):
        pass

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def __truediv__( self, a ):
        return self * (1.0 / a)

    def __itruediv__( self, a ):
        self *= 1.0 / a
        return self

#===============================================================================
class LinearOperator( metaclass=ABCMeta ):
    """
    Linear operator acting between two (normed) vector spaces V (domain)
    and W (codomain).

    """
    @property
    def shape( self ):
        return (self.codomain.dimension, self.domain.dimension)

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def domain( self ):
        pass

    @property
    @abstractmethod
    def codomain( self ):
        pass

    @property
    @abstractmethod
    def dtype( self ):
        pass

    @abstractmethod
    def dot( self, v, out=None ):
        pass

    def __add__( self, B ):
        return SumLinearOperator(self._domain, self._codomain, self, B)

    def __mul__( self, c ):
        assert np.isscalar(c)
        if c==0:
            return ZeroOperator(self._domain, self._codomain)
        elif c == 1:
            return self
        else:
            return ScalLinearOperator(self._domain, self._codomain, c, self)

    def __rmul__( self, c ):
        return self * c

    def __matmul__( self, B ):
        return CompLinearOperator(B.domain, self._codomain, self, B)

    def __pow__( self, n ):
        return PowLinearOperator(self._domain, self._codomain, self, n)

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def idot( self, v, out ):
        assert isinstance(v, Vector)
        assert isinstance(v.space, self.domain)
        assert isinstance(out, Vector)
        assert isinstance(out.space, self.codomain)
        out += self.dot(v)

#===============================================================================
class SumLinearOperator( LinearOperator ):
    def __new__( cls, domain, codomain, *args ):

        if len(args) == 0:
            return ZeroOperator(domain,codomain)
        else:
            return super().__new__(cls)
    
    def __init__( self, domain, codomain, *args ):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)
        for a in args:
            assert isinstance(a, LinearOperator)
            assert a.domain == domain
            assert a.codomain == codomain

        addends = ()
        for a in args:
            if isinstance(a, SumLinearOperator):
                addends = (*addends, *a.addends)
            else:
                addends = (*addends, a)

        self._domain = domain
        self._codomain = codomain
        self._addends = addends
    
    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def addends( self ):
        return self._addends

    @property
    def dtype( self ):
        return None

    def dot( self, v ):
        assert isinstance(v, Vector)
        assert v.space == self._domain

        out = self._codomain.zeros()
        for a in self._addends:
            out += a.dot(v)
        return out

#===============================================================================
class CompLinearOperator( LinearOperator ):
    def __init__( self, domain, codomain, *args ):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)
        
        for a in args:
            assert isinstance(a, LinearOperator)
        assert args[0].codomain == codomain
        assert args[-1].domain == domain

        for i in range(len(args)-1):
            assert args[i].domain == args[i+1].codomain

        multiplicants = ()
        for a in args:
            if isinstance(a, CompLinearOperator):
                multiplicants = (*multiplicants, *a.multiplicants)
            else:
                multiplicants = (*multiplicants, a)

        self._domain = domain
        self._codomain = codomain
        self._multiplicants = multiplicants

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def multiplicants( self ):
        return self._multiplicants

    @property
    def dtype( self ):
        return None

    def dot( self, v ):
        assert isinstance(v,Vector)
        assert v.space == self._domain
        out = self._multiplicants[-1].dot(v)
        for i in range(1,len(self._multiplicants)):
            out = self._multiplicants[-1-i].dot(out)
        return out

#===============================================================================
class ScalLinearOperator( LinearOperator ):
    def __init__( self, domain, codomain, c, A ):

        assert np.isscalar(c)
        assert isinstance(A, LinearOperator)
        assert domain == A.domain
        assert codomain == A.codomain

        if isinstance(A,ScalLinearOperator):
            scalar = A.scalar*c
            operator = A.operator
        else:
            scalar = c
            operator = A

        self._operator = operator
        self._scalar = scalar
        self._domain = domain
        self._codomain = codomain

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def scalar( self ):
        return self._scalar

    @property
    def operator( self ):
        return self._operator

    @property
    def dtype( self ):
        return None

    def dot( self, v ):
        assert isinstance(v, Vector)
        assert v.space == self._domain        
        return self._operator.dot(v) * self._scalar

#===============================================================================
class PowLinearOperator( LinearOperator ):
    def __new__( cls, domain, codomain, A, n ):

        assert isinstance(n, int)
        assert n >= 0

        assert isinstance(A, LinearOperator)       
        assert A.domain == domain
        assert A.codomain == codomain
        assert domain == codomain

        if n == 0:
            return IdOperator(domain, codomain)
        elif n == 1:
            return A
        else:
            return super().__new__(cls)
    
    def __init__( self, domain, codomain, A, n ):

        if isinstance(A, PowLinearOperator):
            self._operator = A.operator
            self._factorial = A.factorial*n
        else:
            self._operator = A
            self._factorial = n
        self._domain = domain
        self._codomain = codomain

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return None

    @property
    def operator( self ):
        return self._operator

    @property
    def factorial( self ):
        return self._factorial

    def dot( self, v ):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        out = v.copy
        for i in range(self._factorial):
            out = self._operator.dot(out)
        return out

#===============================================================================
class ZeroOperator( LinearOperator ):
    def __init__(self, domain=None, codomain=None ):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)

        self._domain = domain
        self._codomain = codomain

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return None

    def dot( self, v ):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        return self._codomain.zeros()

#===============================================================================
class IdOperator( LinearOperator ):
    def __init__(self, domain=None, codomain=None ):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)
        assert domain == codomain

        self._domain = domain
        self._codomain = domain

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return None

    def dot( self, v ):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        return v