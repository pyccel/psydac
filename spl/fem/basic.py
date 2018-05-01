# coding: utf-8

"""
In order to avoid multiple inheritence, we define the base objects for Finite
Elements as abstract classes that contain a topological member. This member can
be used to specify the used data structure for example.
"""

from abc   import ABCMeta, abstractmethod

from spl.linalg.basic import (VectorSpace, Vector)


#===============================================================================
class FemSpace( metaclass=ABCMeta ):
    """
    Generic Finite Element space V.

    """
    @property
    @abstractmethod
    def vector_space( self ):
        pass

    @property
    @abstractmethod
    def nbasis( self ):
        pass

    @property
    @abstractmethod
    def degree( self ):
        pass

    @property
    @abstractmethod
    def pdim( self ):
        """Parametric dimension."""
        pass

    @property
    @abstractmethod
    def ncells( self ):
        pass


#===============================================================================
class FemField( metaclass=ABCMeta ):
    """
    Element of a finite element space V.

    """
    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def space( self ):
        pass

    @property
    @abstractmethod
    def coeffs( self ):
        pass
