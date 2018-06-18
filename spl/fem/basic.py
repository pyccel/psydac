# coding: utf-8

"""
In order to avoid multiple inheritence, we define the base objects for Finite
Elements as abstract classes that contain a topological member. This member can
be used to specify the used data structure for example.
"""

from abc import ABCMeta, abstractmethod

#===============================================================================
# ABSTRACT BASE CLASS: FINITE ELEMENT SPACE
#===============================================================================
class FemSpace( metaclass=ABCMeta ):
    """
    Generic Finite Element space V.

    A unique basis is associated to a FemSpace, i.e. FemSpace = Span( basis )

    """
    #-----------------------------------------
    # Abstract interface: read-only attributes
    #-----------------------------------------
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

    # NOTE: not sure the following is useful
    @property
    @abstractmethod
    def fields( self ):
        """Dictionary containing all FemField objects associated to this space."""

    #---------------------------------------
    # Abstract interface: evaluation methods
    #---------------------------------------
    @abstractmethod
    def eval_field( self, field, *eta ):
        """
        Evaluate field at location(s) eta.

        Parameters
        ----------
        field : FemField
            Field object (element of FemSpace) to be evaluated.

        eta : list of float or numpy.ndarray
            Evaluation point(s) in logical domain.

        Returns
        -------
        value : float or numpy.ndarray
            Field value(s) at location(s) eta.

        """

    @abstractmethod
    def eval_field_gradient( self, field, *eta ):
        """
        Evaluate field gradient at location(s) eta.

        Parameters
        ----------
        field : FemField
            Field object (element of FemSpace) to be evaluated.

        eta : list of float or numpy.ndarray
            Evaluation point(s) in logical domain.

        Returns
        -------
        value : float or numpy.ndarray
            Value(s) of field gradient at location(s) eta.

        """

#---------------------------------------
# OLD STUFF
#---------------------------------------
#
#  # NOTE: why not giving the number of field components?
#      @property
#      @abstractmethod
#      def is_scalar( self ):
#          """Elements of space are scalar fields? [True|False]."""
#
#  # NOTE: why does 'nbasis' have different behavior for tensor product spaces?
#      @property
#      @abstractmethod
#      def nbasis( self ):
#          """
#          Number of linearly independent elements in basis.
#          For a tensor product space this is a tuple of integers.
#  
#          """
#
#  # NOTE: why is 'degree' part of abstract interface?
#  #       What if one were to use a global basis like Fourier?
#      @property
#      @abstractmethod
#      def degree( self ):
#          """Tuple of integers: polynomial degree along each logical dimension."""
#
#  # NOTE: why is 'ncells' part of abstract interface?
#  #       What if one were to use a global basis like Fourier?
#      @property
#      @abstractmethod
#      def ncells( self ):
#          """Tuple of integers: number of grid cells along each logical dimension."""

#===============================================================================
# CONCRETE CLASS: ELEMENT OF A FEM SPACE
#===============================================================================
class FemField:
    """
    Element of a finite element space V.

    Parameters
    ----------
    space : FemSpace
        Finite element space to which this field belongs.

    name : str
        Name of new field to be created.

    """
    def __init__( self, space, name ):

        assert isinstance( space, FemSpace )
        assert isinstance( name, str )

        self._space  = space
        self._name   = name
        self._coeffs = space.vector_space.zeros()

        # NOTE: not sure the following is useful
        # NOTE: since each FemField object contains a reference to its FemSpace,
        #       only weak references should be stored in the FemSpace 'fields'
        #       dictionary to avoid cyclic references
        import weakref
        assert name not in space.fields
        space.fields[name] = weakref.ref( self )

    # ...
    # NOTE: not sure the following is useful
    def __del__( self ):
        #self._space.fields.__delitem__( self._name )
        del self._space.fields[self._name]

    # ...
    @property
    def space( self ):
        """Finite element space to which this field belongs."""
        return self._space

    # ...
    @property
    def name( self ):
        """Name assigned to current field object."""
        return self._name

    # ...
    @property
    def coeffs( self ):
        """
        Coefficients that uniquely identify this field as a linear combination of
        the elements of the basis of a Finite element space.

        Coefficients are stored into one element of the vector space in
        'self.space.vector_space', which is topologically associated to
        the finite element space.

        """
        return self._coeffs

    # ...
    def __call__( self, *eta ):
        """Evaluate field at location identified by logical coordinates eta."""
        return self._space.eval_field( self, *eta )

    # ...
    def gradient( self, *eta ):
        """Evaluate gradient of field at location identified by logical coordinates eta."""
        return self._space.eval_field_gradient( self, *eta )
