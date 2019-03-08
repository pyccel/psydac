# coding: utf-8
#
# Copyright 2018 Yaman Güçlü, Jalal Lakhlili

from abc   import ABCMeta, abstractmethod
from numpy import ndarray

__all__ = ['VectorSpace', 'Vector', 'LinearOperator', 'LinearSolver', 'Matrix']

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
    Linear operator acting between two (normed) vector spaces V (domain)
    and W (codomain).

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
class Matrix( LinearOperator ):
    """
    Linear operator whose coefficients can be viewed as a 2D matrix.

    """
    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @abstractmethod
    def toarray( self ):
        """ Convert to Numpy 2D array. """

    @abstractmethod
    def tosparse( self ):
        """ Convert to any Scipy sparse matrix format. """

#===============================================================================
class LinearSolver( metaclass=ABCMeta ):
    """
    Solver for square linear system Ax=b, where x and b belong to (normed)
    vector space V.

    """
    @property
    def shape( self ):
        return (self.space.dimension, self.space.dimension)

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def space( self ):
        pass

    @abstractmethod
    def solve( self, rhs, out=None ):
        pass

#===============================================================================
del ABCMeta, abstractmethod, ndarray
