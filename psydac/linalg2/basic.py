# coding: utf-8
#
# Copyright 2018 Yaman Güçlü, Jalal Lakhlili

from abc   import ABCMeta, abstractmethod
from numpy import ndarray

__all__ = ['VectorSpace', 'Vector', 'LinearOperator']

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

    @abstractmethod
    def dot( self, v ):
        pass

    # @abstractmethod
    # def toarray( self, **kwargs ):
    #     """ Convert to Numpy 1D array. """

    # @abstractmethod
    # def copy( self ):
    #     pass

    # @abstractmethod
    # def __neg__( self ):
    #     pass

    @abstractmethod
    def __mul__( self, a ):
        pass

    @abstractmethod
    def __rmul__( self, a ):
        pass

    @abstractmethod
    def __add__( self, v ):
        pass

    # @abstractmethod
    # def __sub__( self, v ):
    #     pass

    # @abstractmethod
    # def __imul__( self, a ):
    #     pass

    # @abstractmethod
    # def __iadd__( self, v ):
    #     pass

    # @abstractmethod
    # def __isub__( self, v ):
    #     pass

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
    # makes more sense in this order?

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

    @abstractmethod
    def __add__( self, B ):
        pass

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def idot( self, v, out ):
        assert isinstance(v, Vector)
        assert isinstance(v.space, self.domain)
        assert isinstance(out, Vector)
        assert isinstance(out.space, self.codomain)
        out += self.dot(v)

