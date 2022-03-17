# coding: utf-8

# TODO: - have a block version for VectorSpace when all component spaces are the same
import numpy as np

from sympde.topology.space import BasicFunctionSpace
from sympde.topology.datatype import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType

from psydac.linalg.basic   import Vector
from psydac.linalg.stencil import StencilVectorSpace
from psydac.linalg.block   import BlockVectorSpace
from psydac.fem.basic      import FemSpace, FemField
from psydac.fem.tensor     import TensorFemSpace

from psydac.core.kernels import (pushforward_2d_hdiv,
                                 pushforward_3d_hdiv,
                                 pushforward_2d_hcurl,
                                 pushforward_3d_hcurl)

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
        assert len(np.unique(ldims)) == 1

        self._ldim = ldims[0]
        # ...

        # ... make sure that all spaces have the same number of cells
        ncells = [V.ncells for V in self.spaces]

        if self.ldim == 1:
            assert len(np.unique(ncells)) == 1
        else:
            ns = np.asarray(ncells[0])
            for ms in ncells[1:]:
                assert np.allclose(ns, np.asarray(ms))

        self._ncells = ncells[0]
        # ...

        self._symbolic_space   = None
        self._vector_space     = BlockVectorSpace(*[V.vector_space for V in self.spaces])
        self._refined_space    = {}

        if isinstance(spaces[0], TensorFemSpace):
            self._refined_space[tuple(self._ncells)] = self
            for key in self.spaces[0]._refined_space:
                if key == tuple(self._ncells):continue
                self._refined_space[key] = VectorFemSpace(*[V._refined_space[key] for V in self.spaces])

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

        assert isinstance(field, FemField)
        assert field.space is self
        assert len(eta) == self._ldim

        return self.eval_fields(eta, field, weights=weights)[0]

    # ...
    def eval_fields(self, grid, *fields, weights=None, npts_per_cell=None):
        """Evaluate one or several fields on the given location(s) grid.

        Parameters
        -----------
        grid : List of ndarray
            Grid on which to evaluate the fields.
            Each array in this list corresponds to one logical coordinate.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate.

        weights : psydac.fem.basic.FemField or None, optional
            Weights field.

        npts_per_cell: int, tuple of int or None, optional
            Number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        List of list of ndarray
            List of the same lengths as `fields`, containing for each field,
            a list of `self.ldim` arrays, on for each logical coordinate.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """
        result = []
        for i in range(self.ldim):
            fields_i = list(field.fields[i] for field in fields)
            result.append(self._spaces[i].eval_fields(grid,
                                                      *fields_i,
                                                      npts_per_cell=npts_per_cell,
                                                      weights=weights.fields[i]))
        return [[result[j][i] for j in range(self.ldim)] for i in range(len(fields))]

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

    # ...
    def pushforward_grid(self, grid, *fields, mapping=None, npts_per_cell=None):
        """ Push forward fields on a given grid and a given mapping

        Parameters
        ----------
        grid : List of ndarray
            Grid on which to evaluate the fields

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate

        mapping: psydac.mapping.SplineMapping
            Mapping on which to push-forward

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        List of ndarray
            push-forwarded fields

        """

        # Check that a mapping is given
        if mapping is None:
            raise ValueError("A mapping is needed to push-forward")

        # Check that the fields belong to our space
        assert all(f.space is self for f in fields)

        # Check the grid argument
        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # --------------------------
        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return [self.pushforward(f, *grid, mapping=mapping) for f in fields]

        # Case 2. 1D array of coordinates and no npts_per_cell is given
        # -> grid is tensor-product, but npts_per_cell is not the same in each cell
        elif grid[0].ndim == 1 and npts_per_cell is None:
            raise NotImplementedError("Having a different number of evaluation"
                                      "points in the cells belonging to the same "
                                      "logical dimension is not supported yet. "
                                      "If you did use valid inputs, you need to provide"
                                      "the number of evaluation points per cell in each direction"
                                      "via the npts_per_cell keyword")

        # Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer
        # -> grid is tensor-product, and each cell has the same number of evaluation points
        elif grid[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * self.ldim

            for i in range(self.ldim):
                ncells_i = len(self.spaces[0].breaks[i]) - 1
                grid[i] = np.reshape(grid[i], newshape=(ncells_i, npts_per_cell[i]))

            pushed_fields = self.pushforward_regular_tensor_grid(grid, *fields, mapping=mapping)
            # return a list of list of C-contiguous arrays, one list for each field
            # with one array for each dimension.
            return [[np.ascontiguousarray(pushed_fields[..., j, i]) for j in range(self._ldim)]
                    for i in range(len(fields))]

        # Case 4. (self.ldim)D arrays of coordinates and no npts_per_cell
        # -> unstructured grid
        elif grid[0].ndim == self.ldim and npts_per_cell is None:
            raise NotImplementedError("Unstructured grids are not supported yet.")

        # Case 5. Nonsensical input
        else:
            raise ValueError("This combination of argument isn't understood. The 4 cases understood are :\n"
                             "Case 1. Scalar coordinates\n"
                             "Case 2. 1D array of coordinates and no npts_per_cell is given\n"
                             "Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer\n"
                             "Case 4. {0}D arrays of coordinates and no npts_per_cell".format(self.ldim))

    # ...
    def pushforward(self, field, *eta, mapping=None, parent_kind=None):
        assert field.space is self
        assert len(eta) == self._ldim

        if parent_kind is None:
            kind = self._symbolic_space.kind
        else:
            kind = parent_kind

        raise NotImplementedError("VectorFemSpace not yet operational")

    # ...
    def pushforward_regular_tensor_grid(self, grid, *fields, mapping=None, parent_kind=None):
        """Push-forwards fields on a regular tensor grid using a given a mapping.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        fields: List of psydac.fem.basic.FemField

        mapping: psydac.mapping.SplineMapping
            Mapping on which to push-forward

        parent_kind : sympde.topology.datatype

        Returns
        -------
        List of list of ndarray
            Push-forwarded fields
        """

        if parent_kind is None:
            kind = self._symbolic_space.kind
        else:
            kind = parent_kind

        if kind is L2SpaceType() or kind is H1SpaceType():

            pushed_fields_int = [self.spaces[i].pushforward_regular_tensor_grid(grid, *[f.fields[i] for f in fields])
                                 for i in range(self._ldim)]
            return [[pushed_fields_int[i][j] for i in range(self._ldim)] for j in range(len(fields))]

        # out_fields is a list self._ldim of arrays of shape grid.shape + (len(fields),)
        out_fields = np.asarray([self.spaces[i].eval_fields_regular_tensor_grid(grid, *[f.fields[i] for f in fields])
                                 for i in range(self._ldim)])

        pushed_fields = np.zeros(shape=out_fields.shape[1:-1] + (self.ldim, len(fields)))

        if kind is HdivSpaceType():
            jacobians = mapping.jac_mat_regular_tensor_grid(grid)
            if self.ldim == 2:
                pushforward_2d_hdiv(out_fields, jacobians, pushed_fields)
            if self.ldim == 3:
                pushforward_3d_hdiv(out_fields, jacobians, pushed_fields)

        elif kind is HcurlSpaceType():
            inv_jacobians = mapping.inv_jac_mat_regular_tensor_grid(grid)
            if self.ldim == 2:
                pushforward_2d_hcurl(out_fields, inv_jacobians, pushed_fields)
            if self.ldim == 3:
                pushforward_3d_hcurl(out_fields, inv_jacobians, pushed_fields)

        else:
            raise ValueError(f"Spaces of kind {kind} are not understood")

        return pushed_fields

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
            return len(np.unique(degree)) == 1
        else:
            ns = np.asarray(degree[0])
            for ms in degree[1:]:
                if not( np.allclose(ns, np.asarray(ms)) ): return False
            return True

    # ...
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
        assert len(np.unique(ldims)) == 1
         # ...

        self._ldim = ldims[0]
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
        assert isinstance(field, FemField)
        assert field.space is self
        assert len(eta) == self._ldim
        raise NotImplementedError()
        # return self.eval_fields(eta, field, weights=weights)[0]

    # ...
    def eval_fields(self, grid, *fields, weights=None, npts_per_cell=None):
        """Evaluate one or several fields on the given location(s) grid.

        Parameters
        -----------
        grid : List of ndarray
            Grid on which to evaluate the fields.
            Each array in this list corresponds to one logical coordinate.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate.

        weights : psydac.fem.basic.FemField or None, optional
            Weights field.

        npts_per_cell: int, tuple of int or None, optional
            Number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        List of list of ndarray
            List of the same lengths as `fields`, containing for each field
            a list of `self.ldim` arrays, one for each logical coordinate.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """
        result = []
        if weights is not None:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)

                result.append(self._spaces[i].eval_fields(grid,
                                                          *fields_i,
                                                          npts_per_cell=npts_per_cell,
                                                          weights=weights.fields[i]))
        else:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)
                result.append(self._spaces[i].eval_fields(grid,
                                                          *fields_i,
                                                          npts_per_cell=npts_per_cell))
        return [[result[j][i] for j in range(self.ldim)] for i in range(len(fields))]

    # ...
    def eval_field_gradient( self, field, *eta ):
        raise NotImplementedError( "ProductFemSpace not yet operational" )

    # ...
    def integral( self, f ):
        raise NotImplementedError( "ProductFemSpace not yet operational" )

    # ...
    def pushforward_fields(self, grid, *fields, mapping=None, npts_per_cell=None):
        """ Push forward fields on a given grid and a given mapping

        Parameters
        ----------
        grid : List of ndarray
            Grid on which to evaluate the fields

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate

        mapping: psydac.mapping.SplineMapping or sympde.topology.callable_mapping.CallableMapping
            Mapping on which to push-forward

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        List of ndarray
            push-forwarded fields
        """

        # Check that a mapping is given
        if mapping is None:
            raise TypeError("pushforward_fields() missing 1 required keyword-only argument: 'mapping'")

        # Check that the fields belong to our space
        assert all(f.space is self for f in fields)

        # Check the grid argument
        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # --------------------------
        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return [self.pushforward_field(f, *grid, mapping=mapping) for f in fields]

        # Case 2. 1D array of coordinates and no npts_per_cell is given
        # -> grid is tensor-product, but npts_per_cell is not the same in each cell
        elif grid[0].ndim == 1 and npts_per_cell is None:
            raise NotImplementedError("Having a different number of evaluation"
                                      "points in the cells belonging to the same "
                                      "logical dimension is not supported yet. "
                                      "If you did use valid inputs, you need to provide"
                                      "the number of evaluation points per cell in each direction"
                                      "via the npts_per_cell keyword")

        # Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer
        # -> grid is tensor-product, and each cell has the same number of evaluation points
        elif grid[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * self.ldim

            for i in range(self.ldim):
                ncells_i = len(self.spaces[0].breaks[i]) - 1
                grid[i] = np.reshape(grid[i], newshape=(ncells_i, npts_per_cell[i]))

            pushed_fields = self.pushforward_fields_regular_tensor_grid(grid, *fields, mapping=mapping)
            # return a list of list of C-contiguous arrays, one list for each field
            # with one array for each dimension.
            return [[np.ascontiguousarray(pushed_fields[..., i, j]) for i in range(self._ldim)]
                    for j in range(len(fields))]

        # Case 4. (self.ldim)D arrays of coordinates and no npts_per_cell
        # -> unstructured grid
        elif grid[0].ndim == self.ldim and npts_per_cell is None:
            raise NotImplementedError("Unstructured grids are not supported yet.")

        # Case 5. Nonsensical input
        else:
            raise ValueError("This combination of argument isn't understood. The 4 cases understood are :\n"
                             "Case 1. Scalar coordinates\n"
                             "Case 2. 1D array of coordinates and no npts_per_cell is given\n"
                             "Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer\n"
                             "Case 4. {0}D arrays of coordinates and no npts_per_cell".format(self.ldim))

    # ...
    def pushforward_field(self, field, *eta, mapping=None, parent_kind=None):
        assert field.space is self
        assert len(eta) == self._ldim

        raise NotImplementedError("ProductFemSpace not yet operational")

    # ...
    def pushforward_fields_regular_tensor_grid(self, grid, *fields, mapping=None, parent_kind=None):
        """Push-forwards fields on a regular tensor grid using a given a mapping.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        fields: tuple of psydac.fem.basic.FemField
            List of fields to evaluate.

        mapping: psydac.mapping.SplineMapping
            Mapping on which to push-forward

        Returns
        -------
        List of list of ndarray
            Push-forwarded fields
        """

        if parent_kind is None:
            kind = self._symbolic_space.kind
        else:
            kind = parent_kind

        if kind is L2SpaceType() or kind is H1SpaceType():
            pushed_fields_int = [self.spaces[i].pushforward_regular_tensor_grid(grid,
                                                                                *[f.fields[i] for f in fields],
                                                                                mapping=mapping,
                                                                                parent_kind=kind)
                                 for i in range(self._ldim)]
            return [[pushed_fields_int[i][j] for i in range(self._ldim)] for j in range(len(fields))]

        # out_fields is a list self._ldim of arrays of shape grid.shape + (len(fields),)
        out_fields = np.asarray([self.spaces[i].eval_fields_regular_tensor_grid(grid, *[f.fields[i] for f in fields])
                                 for i in range(self._ldim)])

        pushed_fields = np.zeros(shape=out_fields.shape[1:-1] + (self.ldim, len(fields)))

        if kind is HdivSpaceType():
            jacobians = mapping.jac_mat_regular_tensor_grid(grid)
            if self.ldim == 2:
                pushforward_2d_hdiv(out_fields, jacobians, pushed_fields)
            if self.ldim == 3:
                pushforward_3d_hdiv(out_fields, jacobians, pushed_fields)

        elif kind is HcurlSpaceType():
            inv_jacobians = mapping.inv_jac_mat_regular_tensor_grid(grid)
            if self.ldim == 2:
                pushforward_2d_hcurl(out_fields, inv_jacobians, pushed_fields)
            if self.ldim == 3:
                pushforward_3d_hcurl(out_fields, inv_jacobians, pushed_fields)

        else:
            raise ValueError(f"Spaces of kind {kind} are not understood")

        return pushed_fields

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
