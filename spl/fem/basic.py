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

    A unique basis is associated to a FemSpace, i.e. FemSpace = Span( basis )

    """
    @property
    @abstractmethod
    def ldim( self ):
        """
        Number of dimensions in logical space,
        i.e. number of scalar logical coordinates.

        """

    @property
    @abstractmethod
    def periodic( self ):
        """
        Tuple of booleans: along each logical dimension,
        say if domain is periodic.

        """

    @property
    @abstractmethod
    def mapping( self ):
        """
        Mapping from logical coordinates 'eta' to physical coordinates 'x'.
        If None, we assume identity mapping (hence x=eta).

        """

    @property
    @abstractmethod
    def vector_space( self ):
        """Topologically associated vector space."""

# NOTE: why not giving the number of field components?
    @property
    @abstractmethod
    def is_scalar( self ):
        """Elements of space are scalar fields? [True|False]."""

# NOTE: why does 'nbasis' have different behavior for tensor product spaces?
    @property
    @abstractmethod
    def nbasis( self ):
        """
        Number of linearly independent elements in basis.
        For a tensor product space this is a tuple of integers.

        """

# NOTE: why is 'degree' part of abstract interface?
#       What if one were to use a global basis like Fourier?
    @property
    @abstractmethod
    def degree( self ):
        """Tuple of integers: polynomial degree along each logical dimension."""

# NOTE: why is 'ncells' part of abstract interface?
#       What if one were to use a global basis like Fourier?
    @property
    @abstractmethod
    def ncells( self ):
        """Tuple of integers: number of grid cells along each logical dimension."""

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
        """Finite element space to which this field belongs."""

    @property
    @abstractmethod
    def coeffs( self ):
        """
        Coefficients that uniquely identify this field as a linear combination of
        the elements of the basis of a Finite element space.

        Coefficients are stored into one element of the vector space in
        'self.space.vector_space', which is topologically associated to
        the finite element space.

        """

    @abstractmethod
    def __call__( self, eta ):
        """Evaluate field at location identified by logical coordinates eta."""


    @abstractmethod
    def gradient( self, eta ):
        """Evaluate gradient of field at location identified by logical coordinates eta."""
