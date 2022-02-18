# coding: utf-8

# TODO: - have a block version for VectorSpace when all component spaces are the same
from sympde.topology.space import BasicFunctionSpace

from psydac.linalg.basic   import Vector
from psydac.linalg.stencil import StencilVectorSpace
from psydac.linalg.block   import BlockVectorSpace
from psydac.fem.basic      import FemSpace, FemField

from numpy import unique, asarray, allclose, array, moveaxis, ascontiguousarray, zeros_like, reshape


#===============================================================================
class VectorFemSpace( FemSpace ):
    """
    FEM space with a vector basis

    """

    def __init__( self, *spaces ):
        """."""
        self._spaces = spaces

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

        self._symbolic_space   = None
        self._vector_space     = None

        # TODO serial case
        # TODO parallel case

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
    def is_product(self):
        return True

    @property
    def symbolic_space( self ):
        return self._symbolic_space

    @symbolic_space.setter
    def symbolic_space( self, symbolic_space ):
        assert isinstance(symbolic_space, BasicFunctionSpace)
        self._symbolic_space = symbolic_space

    #--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field( self, field, *eta, weights=None):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == self._ldim

        raise NotImplementedError( "VectorFemSpace not yet operational" )

    # ...
    def eval_fields(self, *fields, refine_factor=1, weights=None):
        result = []
        for i in range(self.ldim):
            fields_i = list(field.fields[i] for field in fields)
            result.append(self._spaces[i].eval_fields(*fields_i, refine_factor=refine_factor, weights=weights))
        result = array(result)

        return ascontiguousarray(moveaxis(result, 0, -2))

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
        # TODO [MCP, 08.03.2021]: check if we should return a tuple
        return sum(dims)

    @property
    def degree(self):
        return [V.degree for V in self.spaces]

    @property
    def multiplicity(self):
        return [V.multiplicity for V in self.spaces]

    @property
    def pads(self):
        return [V.pads for V in self.spaces]

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

    # ...
    def pushforward(self, *fields, mapping, refine_factor=1):
        from psydac.core.kernels import pushforward_2d_l2, pushforward_3d_l2, pushforward_2d_hdiv, pushforward_3d_hdiv,\
                                        pushforward_2d_hcurl, pushforward_3d_hcurl

        kind = self.symbolic_space.kind

        # Shape of out_fields = (n_0,...,n_ldim, ldim, len(fields))
        out_fields = self.eval_fields(*fields, refine_factor=refine_factor)
        pushed_fields = zeros_like(out_fields)

        if kind == 'L2SpaceType()':
            pushed_fields = reshape(pushed_fields, newshape=(*pushed_fields.shape[:-2],
                                                             pushed_fields.shape[-1] * pushed_fields.shape[-2]))

            out_fields= reshape(out_fields, newshape=(*pushed_fields.shape[:-2],
                                                         pushed_fields.shape[-1] * pushed_fields.shape[-2]))
            if self.ldim == 2:
                pushforward_2d_l2(out_fields, mapping.metric_det_grid(refine_factor=refine_factor), pushed_fields)

            if self.ldim == 3:
                pushforward_3d_l2(out_fields, mapping.metric_det_grid(refine_factor=refine_factor), pushed_fields)

            pushed_fields = reshape(pushed_fields, newshape=(*pushed_fields.shape[:-1],
                                                                self.ldim,
                                                                pushed_fields.shape[-1] // self.ldim))

        if kind == 'HdivSpaceType()':
            if self.ldim == 2:
                pushforward_2d_hdiv(out_fields, mapping.jac_mat_grid(refine_factor=refine_factor), pushed_fields)
            if self.ldim == 3:
                pushforward_3d_hdiv(out_fields, mapping.jac_mat_grid(refine_factor=refine_factor), pushed_fields)

        if kind == 'HcurlSpaceType()':
            if self.ldim == 2:
                pushforward_2d_hcurl(out_fields, mapping.inv_jac_mat_grid(refine_factor=refine_factor), pushed_fields)
            if self.ldim == 3:
                pushforward_3d_hcurl(out_fields, mapping.inv_jac_mat_grid(refine_factor=refine_factor), pushed_fields)

        return pushed_fields


    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> ldim   :: {ldim}\n'.format(ldim=self.ldim)
        txt += '> total nbasis  :: {dim}\n'.format(dim=self.nbasis)

        dims = ', '.join(str(V.nbasis) for V in self.spaces)
        txt += '> nbasis :: ({dims})\n'.format(dims=dims)
        return txt

#===============================================================================
class ProductFemSpace( FemSpace ):
    """
    Product of FEM space
    """

    def __new__(cls, *spaces):

        if len(spaces) == 1:
            return spaces[0]
        else:
            return FemSpace.__new__(cls)

    def __init__( self, *spaces):
        """."""

        if len(spaces) == 1:
            return

        self._spaces = spaces

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

        self._vector_space    = BlockVectorSpace(*[V.vector_space for V in self.spaces])
        self._symbolic_space  = None

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
    def is_product(self):
        return True

    @property
    def symbolic_space( self ):
        return self._symbolic_space

    @symbolic_space.setter
    def symbolic_space( self, symbolic_space ):
        assert isinstance(symbolic_space, BasicFunctionSpace)
        self._symbolic_space = symbolic_space

    #--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field( self, field, *eta, weights=None):
        raise NotImplementedError( "ProductFemSpace not yet operational" )

    # ...
    def eval_fields(self, *fields, refine_factor=1, weights=None):
        result = []
        for i in range(self.ldim):
            fields_i = list(field.fields[i] for field in fields)
            result.append(self._spaces[i].eval_fields(*fields_i, refine_factor=refine_factor, weights=weights))
        result = array(result)

        return ascontiguousarray(moveaxis(result, 0, -2))

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
        # TODO [MCP, 08.03.2021]: check if we should return a tuple
        return sum(dims)

    @property
    def degree(self):
        return [V.degree for V in self.spaces]

    @property
    def multiplicity(self):
        return [V.multiplicity for V in self.spaces]

    @property
    def pads(self):
        return [V.pads for V in self.spaces]

    @property
    def ncells(self):
        return self._ncells

    @property
    def spaces( self ):
        return self._spaces

    @property
    def n_components( self ):
        return len(self.spaces)

    # TODO improve
    @property
    def comm( self ):
        return self.spaces[0].comm

    # ...
    def pushforward(self, *fields, mapping=None, refine_factor=1):
        """ Push forward

        Parameters
        ----------
        fields: list of psydac.fem.basic.FemField

        mapping: psydac.mapping.SplineMapping
            Mapping on which to push-forward

        refine_factor: int
            Degree of refinement of the grid

        Returns
        -------
        pushed_fields:
            push-forwarded fields

        """

        from psydac.core.kernels import pushforward_2d_l2, pushforward_3d_l2, pushforward_2d_hdiv, pushforward_3d_hdiv,\
                                        pushforward_2d_hcurl, pushforward_3d_hcurl

        kind = str(self.symbolic_space.kind)

        # Shape of out_fields = (n_0,...,n_ldim, ldim, len(fields))
        out_fields = self.eval_fields(*fields, refine_factor=refine_factor)

        if mapping is None:
            raise TypeError("pushforward() missing 1 required keyword-only argument: 'mapping'")

        if kind == 'H1SpaceType()':
            return out_fields

        pushed_fields = zeros_like(out_fields)

        if kind == 'L2SpaceType()':
            pushed_fields = reshape(pushed_fields, newshape=(*pushed_fields.shape[:-2],
                                                             pushed_fields.shape[-1] * pushed_fields.shape[-2]))

            out_fields= reshape(out_fields, newshape=(*pushed_fields.shape[:-2],
                                                      pushed_fields.shape[-1] * pushed_fields.shape[-2]))
            if self.ldim == 2:
                pushforward_2d_l2(out_fields, mapping.metric_det_grid(refine_factor=refine_factor), pushed_fields)

            if self.ldim == 3:
                pushforward_3d_l2(out_fields, mapping.metric_det_grid(refine_factor=refine_factor), pushed_fields)

            pushed_fields = reshape(pushed_fields, newshape=(*pushed_fields.shape[:-1],
                                                             self.ldim,
                                                             pushed_fields.shape[-1] // self.ldim))

        if kind == 'HdivSpaceType()':
            if self.ldim == 2:
                pushforward_2d_hdiv(out_fields, mapping.jac_mat_grid(refine_factor=refine_factor), pushed_fields)
            if self.ldim == 3:
                pushforward_3d_hdiv(out_fields, mapping.jac_mat_grid(refine_factor=refine_factor), pushed_fields)

        if kind == 'HcurlSpaceType()':
            if self.ldim == 2:
                pushforward_2d_hcurl(out_fields, mapping.inv_jac_mat_grid(refine_factor=refine_factor), pushed_fields)
            if self.ldim == 3:
                pushforward_3d_hcurl(out_fields, mapping.inv_jac_mat_grid(refine_factor=refine_factor), pushed_fields)

        return pushed_fields
