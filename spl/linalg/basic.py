# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

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

    @abstractmethod
    def dot( self, v ):
        pass

    @abstractmethod
    def copy( self ):
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

Vector.register( ndarray )

#===============================================================================
class LinearOperator( metaclass=ABCMeta ):
    """
    Linear operator acting on a (normed) vector space V.

    """
    @property
    def shape( self ):
        return (self.domain.dimension, self.codomain.dimension)

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

    @abstractmethod
    def dot( self, v, out=None ):
        pass

LinearOperator.register( ndarray )

#===============================================================================
del ABCMeta, abstractmethod, ndarray
