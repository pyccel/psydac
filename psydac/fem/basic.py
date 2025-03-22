# coding: utf-8

"""
In order to avoid multiple inheritence, we define the base objects for Finite
Elements as abstract classes that contain a topological member. This member can
be used to specify the used data structure for example.
"""

from abc import ABCMeta, abstractmethod
from psydac.linalg.basic import Vector

__all__ = ('FemSpace', 'FemField')

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
        # question [MCP 03.2025]:
            # rename as coeff_space ?
            # meaning of docstring? see different one in MultipatchFemSpace

    @property
    @abstractmethod
    def is_multipatch( self ):
        """
        Boolean flag that describes whether the space is a multi-patch space.
        :rtype: bool
        """

    @property
    @abstractmethod
    def is_vector_valued( self ):
        """
        Boolean flag that describes whether the space is vector-valued.
        :rtype: bool
        """

    @property
    def is_product( self ):
        """
        Boolean flag that describes whether the space is a product space.
        If True, an element of this space can be decomposed into separate fields.

        """
        return self.is_multipatch or self.is_vector_valued

    @property
    @abstractmethod
    def symbolic_space( self ):
        """Symbolic space."""

    @property
    @abstractmethod
    def patch_spaces(self):
        """
        Return the patch spaces (self if single-patch) as a tuple.
        """

    @property
    @abstractmethod
    def component_spaces(self):
        """
        Return the component spaces (self if scalar-valued) as a tuple.
        """

    @property
    @abstractmethod
    def axis_spaces(self):
        """
        Return the axis spaces (self if univariate) as a tuple.
        """


    #---------------------------------------
    # Abstract interface: evaluation methods
    #---------------------------------------
    @abstractmethod
    def eval_field( self, field, *eta, weights=None):
        """
        Evaluate field at location(s) eta.

        Parameters
        ----------
        field : FemField
            Field object (element of FemSpace) to be evaluated.

        eta : list of float or numpy.ndarray
            Evaluation point(s) in logical domain.

        weights : StencilVector, optional
            Weights of the basis functions, such that weights.space == field.coeffs.space.

        Returns
        -------
        value : float or numpy.ndarray
            Field value(s) at location(s) eta.

        """

    @abstractmethod
    def eval_field_gradient( self, field, *eta , weights=None):
        """
        Evaluate field gradient at location(s) eta.

        Parameters
        ----------
        field : FemField
            Field object (element of FemSpace) to be evaluated.

        eta : list of float or numpy.ndarray
            Evaluation point(s) in logical domain.

        weights : StencilVector, optional
            Weights of the basis functions, such that weights.space == field.coeffs.space.

        Returns
        -------
        value : float or numpy.ndarray
            Value(s) of field gradient at location(s) eta.

        """


    #----------------------
    # Concrete methods
    #----------------------
    def __mul__(self, a):
    #     from psydac.fem.vector import create_product_space

    #     spaces = [*(self.spaces if self.is_product else [self]),
    #               *(   a.spaces if    a.is_product else    [a])]

    #     space = create_product_space(*spaces)
    #     if a.symbolic_space and self.symbolic_space:
    #         space._symbolic_space =  self.symbolic_space*a.symbolic_space
    #     return space
        raise NotImplementedError('if this method __mul__ is used, it should not be implemented like this: TODO')

    # ...
    def __rmul__(self, a):
    #     from psydac.fem.vector import create_product_space

    #     spaces = [*(   a.spaces if    a.is_product else    [a]),
    #               *(self.spaces if self.is_product else [self]),]

    #     space = create_product_space(*spaces)

    #     if a.symbolic_space and self.symbolic_space:
    #         space._symbolic_space =  a.symbolic_space * self.symbolic_space
    #     return space
        raise NotImplementedError('if this method __rmul__ is used, it should not be implemented like this: TODO')



#---------------------------------------
# OLD STUFF
#---------------------------------------

#    @abstractmethod
#    def integral( self, f ):
#        """
#        Compute integral of scalar callable function $f(\eta)$ over logical domain
#        $\Omega$, with Jacobian determinant of mapping $J(\eta)$ as weighting function:
#
#        I = \integral_{\Omega} f(\eta) |J(\eta)| d\eta.
#
#        Parameters
#        ----------
#        f : callable
#            Integrand scalar function $f(\eta)$ over logical domain.
#
#        Returns
#        -------
#        value : float
#            Integral of $f(\eta) J(\eta)$ over logical domain.
#
#        """
#
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
    space : psydac.fem.basic.FemSpace
        Finite element space to which this field belongs.

    coeffs : psydac.linalg.basic.Vector (optional)
        Vector of coefficients in finite element basis
        (by default assume zero vector).

    """
    def __init__( self, space, coeffs=None ):

        assert isinstance( space, FemSpace )

        if coeffs is not None:
            assert isinstance( coeffs, Vector )
            assert space.vector_space is coeffs.space
        else:
            coeffs = space.vector_space.zeros()

        # Case of a vector field, element of a ProductSpace
        if space.is_product:
            fields = tuple(FemField(V, c) for V, c in zip(space.spaces, coeffs))
        else:
            fields = tuple()

        self._space  = space
        self._coeffs = coeffs
        self._fields = fields

    # ...
    @property
    def space( self ):
        """Finite element space to which this field belongs."""
        return self._space

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
    @property
    def fields(self):
        return self._fields

    @property
    def is_vector_valued(self):
        return self.space.is_vector_valued

    @property
    def patch_fields(self):
        """ Return the patch fields (only self if single-patch) as a tuple """
        if self.space.is_multipatch:
            return self.fields
        else:
            return (self,)

    @property
    def component_fields(self):
        """ Return the component fields (only self if scalar-valued) as a tuple """
        if self.space.is_vector_valued:
            return self.fields
        else:
            return (self,)

    # ...
    def __getitem__(self, key):
        return self._fields[key]

    # ...
    def __call__( self, *eta , weights=None):
        """Evaluate weighted field at location identified by logical coordinates eta."""
        return self._space.eval_field( self, *eta , weights=weights)

    # ...
    def gradient( self, *eta , weights=None):
        """Evaluate gradient of weighted field at location identified by logical coordinates eta."""
        return self._space.eval_field_gradient( self, *eta , weights=weights)

    # ...
    def divergence(self, *eta, weights=None):
        """Evaluate divergence of weighted vector field at location identified by logical coordinates eta."""
        return self._space.eval_field_divergence(self, *eta, weights=weights)

    # ...
    def copy(self):
        return FemField(self._space, coeffs = self._coeffs.copy())

    # ...
    def __neg__(self):
        return FemField(self._space, coeffs = -self._coeffs)

    # ...
    def __mul__(self, a):
        return FemField(self._space, coeffs = self._coeffs * a)

    # ...
    def __rmul__(self, a):
        return FemField(self._space, coeffs = a * self._coeffs)

    # ...
    def __add__(self, other):
        assert isinstance(other, FemField)
        assert self._space is other._space
        return FemField(self._space, coeffs = self._coeffs + other._coeffs)

    # ...
    def __sub__(self, other):
        assert isinstance(other, FemField)
        assert self._space is other._space
        return FemField(self._space, coeffs = self._coeffs - other._coeffs)

    # ...
    def __imul__(self, a):
        self._coeffs *= a
        return self

    # ...
    def __iadd__(self, other):
        assert isinstance(other, FemField)
        assert self._space is other._space
        self._coeffs += other._coeffs
        return self

    # ...
    def __isub__(self, other):
        assert isinstance(other, FemField)
        assert self._space is other._space
        self._coeffs -= other._coeffs
        return self
