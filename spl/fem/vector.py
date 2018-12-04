# coding: utf-8

# TODO: - have a block version for VectorSpace when all component spaces are the same

from spl.linalg.stencil import StencilVectorSpace
from spl.fem.basic      import FemSpace, FemField

from numpy import unique, asarray, allclose


#===============================================================================
class VectorFemSpace( FemSpace ):
    """
    FEM space with a vector basis

    """

    def __init__( self, *args, **kwargs ):
        """."""
        self._spaces = tuple(args)

        # ... make sure that all spaces have the same parametric dimension
        ldims = [V.ldim for V in self.spaces]
        assert (len(unique(ldims)) == 1)

        self._ldim = ldims[0]
        # ...

        # ... make sure that all spaces have the same number of cells
        ncells = [V.ncells for V in self.spaces]

        if self.ldim == 1:
            assert( len(unique(ncells)) == 1 )
        else:
            ns = asarray(ncells[0])
            for ms in ncells[1:]:
                assert( allclose(ns, asarray(ms)) )

        self._ncells = ncells[0]
        # ...

        # TODO serial case
        self._vector_space = None

        # TODO parallel case

        # Dictionary with all FemFields objects belonging to this space
        self._fields = {}

    #--------------------------------------------------------------------------
    # Abstract interface: read-only attributes
    #--------------------------------------------------------------------------
    @property
    def ldim( self ):
        """ Parametric dimension.
        """
        return self._ldim

    @property
    def periodic(self):
        return [V.periodic for V in self.spaces]

    @property
    def mapping(self):
        return None

    @property
    def vector_space(self):
        """Returns the topological associated vector space."""
        return self._vector_space

    @property
    def fields( self ):
        """Dictionary containing all FemField objects associated to this space."""
        return self._fields

    #--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field( self, field, *eta ):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == self._ldim

        raise NotImplementedError( "VectorFemSpace not yet operational" )

    # ...
    def eval_field_gradient( self, field, *eta ):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == self._ldim

        raise NotImplementedError( "VectorFemSpace not yet operational" )

    # ...
    def integral( self, f ):

        assert hasattr( f, '__call__' )

        raise NotImplementedError( "VectorFemSpace not yet operational" )

    #--------------------------------------------------------------------------
    # Other properties and methods
    #--------------------------------------------------------------------------
    @property
    def is_scalar(self):
        return len( self.spaces ) == 1

    @property
    def nbasis(self):
        dims = [V.nbasis for V in self.spaces]
        # TODO: check if we should compute the product, or return a tuple
        return sum(dims)

    @property
    def degree(self):
        return [V.degree for V in self.spaces]

    @property
    def ncells(self):
        return self._ncells

    @property
    def spaces( self ):
        return self._spaces

    @property
    def is_block(self):
        """Returns True if all components are identical spaces."""
        # TODO - improve this tests. for the moment, we only check the degree,
        #      - shall we check the bc too?

        degree = [V.degree for V in self.spaces]
        if self.pdim == 1:
            return len(unique(degree)) == 1
        else:
            ns = asarray(degree[0])
            for ms in degree[1:]:
                if not( allclose(ns, asarray(ms)) ): return False
            return True

    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> ldim   :: {ldim}\n'.format(ldim=self.ldim)
        txt += '> total nbasis  :: {dim}\n'.format(dim=self.nbasis)

        dims = ', '.join(str(V.nbasis) for V in self.spaces)
        txt += '> nbasis :: ({dims})\n'.format(dims=dims)
        return txt

# TODO still experimental
#===============================================================================
from spl.linalg.block import ProductSpace
class ProductFemSpace( FemSpace ):
    """
    Product of FEM space
    """

    def __init__( self, *args, **kwargs ):
        """."""
        self._spaces = tuple(args)

        # ... make sure that all spaces have the same parametric dimension
        ldims = [V.ldim for V in self.spaces]
        assert (len(unique(ldims)) == 1)

        self._ldim = ldims[0]
        # ...

        # ... make sure that all spaces have the same number of cells
        ncells = [V.ncells for V in self.spaces]

        if self.ldim == 1:
            assert( len(unique(ncells)) == 1 )
        else:
            ns = asarray(ncells[0])
            for ms in ncells[1:]:
                assert( allclose(ns, asarray(ms)) )

        self._ncells = ncells[0]
        # ...

        # ...
        v_spaces = [V.vector_space for V in self.spaces]
        self._vector_space = ProductSpace(*v_spaces)
        # ...

    #--------------------------------------------------------------------------
    # Abstract interface: read-only attributes
    #--------------------------------------------------------------------------
    @property
    def ldim( self ):
        """ Parametric dimension.
        """
        return self._ldim

    @property
    def periodic(self):
        return [V.periodic for V in self.spaces]

    @property
    def mapping(self):
        return None

    @property
    def vector_space(self):
        """Returns the topological associated vector space."""
        return self._vector_space

    @property
    def fields( self ):
        """Dictionary containing all FemField objects associated to this space."""
        raise NotImplementedError( "ProductFemSpace not yet operational" )

    #--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field( self, field, *eta ):
        raise NotImplementedError( "ProductFemSpace not yet operational" )

    # ...
    def eval_field_gradient( self, field, *eta ):
        raise NotImplementedError( "ProductFemSpace not yet operational" )

    # ...
    def integral( self, f ):
        raise NotImplementedError( "ProductFemSpace not yet operational" )

    #--------------------------------------------------------------------------
    # Other properties and methods
    #--------------------------------------------------------------------------
    @property
    def nbasis(self):
        dims = [V.nbasis for V in self.spaces]
        # TODO: check if we should compute the product, or return a tuple
        return sum(dims)

    @property
    def degree(self):
        return [V.degree for V in self.spaces]

    @property
    def ncells(self):
        return self._ncells

    @property
    def spaces( self ):
        return self._spaces

    @property
    def n_components( self ):
        return len(self.spaces)

    # ...
    def init_fem( self, **kwargs ):
        for V in self.spaces:
            V.init_fem( **kwargs )

    # ...
    def init_collocation( self ):
        for V in self.spaces:
            # TODO: check if OK to access private attribute...
            if not V._collocation_ready:
                V.init_collocation()

# TODO still experimental
#===============================================================================
class VectorFemField:
    """
    Element of a finite element product space V.

    Parameters
    ----------
    space : ProductFemSpace
        Finite element product space to which this field belongs.

    name : str
        Name of new field to be created.

    """
    def __init__( self, space, name ):

        assert isinstance( space, ProductFemSpace )
        assert isinstance( name, str )

        self._space  = space
        self._name   = name
        self._coeffs = space.vector_space.zeros()

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
