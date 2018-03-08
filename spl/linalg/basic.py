# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from abc   import ABCMeta, abstractmethod
from numpy import ndarray

__all__ = ['Vector', 'LinearOperator']

#===============================================================================
class Vector( metaclass=ABCMeta ):
    """
    Generic element of a (normed) vector space V.

    """
    @property
    @abstractmethod
    def shape( self ):
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

Vector.register( ndarray )

#===============================================================================
class LinearOperator( metaclass=ABCMeta ):
    """
    Linear operator acting on a (normed) vector space V.

    """
    @property
    @abstractmethod
    def shape( self ):
        pass

    @abstractmethod
    def dot( self, v, out=None ):
        pass

LinearOperator.register( ndarray )

#===============================================================================
del ABCMeta, abstractmethod, ndarray
