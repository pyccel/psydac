# coding: utf-8

# TODO: - have a block version for VectorSpace when all component spaces are the same
import numpy as np

from functools import reduce

from sympde.topology.space import BasicFunctionSpace
from sympde.topology.datatype import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType

from psydac.linalg.basic   import Vector
from psydac.linalg.stencil import StencilVectorSpace
from psydac.linalg.block   import BlockVectorSpace
from psydac.fem.basic      import FemSpace, FemField

from psydac.core.field_evaluation_kernels import (pushforward_2d_hdiv,
                                                  pushforward_3d_hdiv,
                                                  pushforward_2d_hcurl,
                                                  pushforward_3d_hcurl)

__all__ = ('VectorFemSpace', 'ProductFemSpace')

#===============================================================================
class VectorFemSpace( FemSpace ):
    """
    FEM space with a vector basis defined on a single patch
    this class is used to represent either spaces of vector-valued fem fields,
    or product spaces involved in systems of equations.
    """

    def __init__( self, *spaces ):

        # all input spaces are flattened into a single list of scalar spaces
        new_spaces = [sp.spaces if isinstance(sp, VectorFemSpace) else [sp] for sp in spaces]
        new_spaces = tuple(sp2 for sp1 in new_spaces for sp2 in sp1)

        self._spaces = new_spaces

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

        self._symbolic_space = None
        if all(s.symbolic_space for s in spaces):
            symbolic_spaces = [s.symbolic_space for s in spaces]
            self._symbolic_space = reduce(lambda x,y:x*y, symbolic_spaces)

        self._vector_space     = BlockVectorSpace(*[V.vector_space for V in self.spaces])
        self._refined_space    = {}

        self.set_refined_space(self._ncells, self)
        for key in self.spaces[0]._refined_space:
            if key == tuple(self._ncells):continue
            self.set_refined_space(key, VectorFemSpace(*[V._refined_space[key] for V in self.spaces]))
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
        """Returns the vector space of the coefficients (mapping invariant)."""
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
    def eval_fields(self, grid, *fields, weights=None, npts_per_cell=None, overlap=0):
        """Evaluates one or several fields on the given location(s) grid.

        Parameters
        ----------
        grid : List of ndarray
            Grid on which to evaluate the fields.
            Each array in this list corresponds to one logical coordinate.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate.

        weights : psydac.fem.basic.FemField or None, optional
            Weights field used to weight the basis functions thus
            turning them into NURBS. The same weights field is used
            for all of fields and they thus have to use the same basis functions.

        npts_per_cell: int, tuple of int or None, optional
            Number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.
            
        Returns
        -------
        List of list of ndarray
            List of the same length as `fields`, containing for each field
            a list of `self.ldim` arrays, i.e. one array for each logical coordinate.

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
                                                          weights=weights.fields[i],
                                                          overlap=overlap))
        else:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)
                result.append(self._spaces[i].eval_fields(grid,
                                                          *fields_i,
                                                          npts_per_cell=npts_per_cell,
                                                          overlap=overlap))
        return [tuple(result[j][i] for j in range(self.ldim)) for i in range(len(fields))]

    # ...
    def eval_fields_regular_tensor_grid(self, grid, *fields, weights=None, overlap=0):
        """Evaluates one or several fields on a regular tensor grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate.

        weights : psydac.fem.basic.FemField or None, optional
            Weights field used to weight the basis functions thus
            turning them into NURBS. The same weights field is used
            for all of fields and they thus have to use the same basis functions.

        overlap : int
            How much to overlap. Only used in the distributed context.
            
        Returns
        -------
        List of list of ndarray
            List of the same length as `fields`, containing for each field
            a list of `self.ldim` arrays, i.e. one array for each logical coordinate.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """
        for f in fields:
            # Necessary if vector coeffs is distributed across processes
            if not f.coeffs.ghost_regions_in_sync:
                f.coeffs.update_ghost_regions()

        result = []
        if isinstance(overlap, int):
            overlap = [overlap] * self.ldim
        if weights is not None:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)

                result.append(self._spaces[i].eval_fields_regular_tensor_grid(grid,
                                                                              *fields_i,
                                                                              weights=weights.fields[i],
                                                                              overlap=overlap[i]))
        else:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)

                result.append(self._spaces[i].eval_fields_regular_tensor_grid(grid,
                                                                              *fields_i,
                                                                              overlap=overlap[i]))

        return result

    # ...
    def eval_fields_irregular_tensor_grid(self, grid, *fields, weights=None, overlap=0):
        """Evaluates one or several fields on an irregular tensor grid i.e.
        a tensor grid where the number of points per cell depends on the cell.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate.

        weights : psydac.fem.basic.FemField or None, optional
            Weights field used to weight the basis functions thus
            turning them into NURBS. The same weights field is used
            for all of fields and they thus have to use the same basis functions.

        overlap : int
            How much to overlap. Only used in the distributed context.
            
        Returns
        -------
        List of list of ndarray
            List of the same length as `fields`, containing for each field
            a list of `self.ldim` arrays, i.e. one array for each logical coordinate.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """
        for f in fields:
            # Necessary if vector coeffs is distributed across processes
            if not f.coeffs.ghost_regions_in_sync:
                f.coeffs.update_ghost_regions()

        result = []
        if isinstance(overlap, int):
            overlap = [overlap] * self.ldim
        if weights is not None:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)

                result.append(self._spaces[i].eval_fields_irregular_tensor_grid(grid,
                                                                                *fields_i,
                                                                                weights=weights.fields[i],
                                                                                overlap=overlap[i]))
        else:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)

                result.append(self._spaces[i].eval_fields_irregular_tensor_grid(grid,
                                                                                *fields_i,
                                                                                overlap=overlap[i]))

        return result

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

    # ...
    def get_refined_space(self, ncells):
        return self._refined_space[tuple(ncells)]

    def set_refined_space(self, ncells, new_space):
        assert all(nc1==nc2 for nc1,nc2 in zip(ncells, new_space.ncells))
        self._refined_space[tuple(ncells)] = new_space

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
    Product of FEM spaces
    this class is used to represent FEM spaces on a multi-patch domain.
    """

    def __new__(cls, *spaces, connectivity=None):

        if len(spaces) == 1:
            return spaces[0]
        else:
            return FemSpace.__new__(cls)

    def __init__( self, *spaces, connectivity=None):
        """
        Parameters
        ----------
        *spaces : 
            single-patch FEM spaces                        
        """

        if len(spaces) == 1:
            return

        self._spaces = spaces

        # ... make sure that all spaces have the same parametric dimension
        ldims = [V.ldim for V in self.spaces]
        assert len(np.unique(ldims)) == 1

        self._ldim = ldims[0]
        # ...

        connectivity          = connectivity if connectivity is not None else {}
        self._vector_space    = BlockVectorSpace(*[V.vector_space for V in self.spaces], connectivity=connectivity)
        self._symbolic_space  = None
        self._connectivity    = connectivity.copy()
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
        """Returns the vector space of the coefficients (mapping invariant)."""
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
    def eval_fields(self, grid, *fields, weights=None, npts_per_cell=None, overlap=0):
        """Evaluates one or several fields on the given location(s) grid.

        Parameters
        ----------
        grid : List of ndarray
            Grid on which to evaluate the fields.
            Each array in this list corresponds to one logical coordinate.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate.

        weights : psydac.fem.basic.FemField or None, optional
            Weights field used to weight the basis functions thus
            turning them into NURBS. The same weights field is used
            for all of fields and they thus have to use the same basis functions.

        npts_per_cell: int, tuple of int or None, optional
            Number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.
            
        Returns
        -------
        List of list of ndarray
            List of the same length as `fields`, containing for each field
            a list of `self.ldim` arrays, i.e. one array for each logical coordinate.

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
                                                          weights=weights.fields[i],
                                                          overlap=overlap))
        else:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)
                result.append(self._spaces[i].eval_fields(grid,
                                                          *fields_i,
                                                          npts_per_cell=npts_per_cell,
                                                          overlap=overlap))
        return [tuple(result[j][i] for j in range(self.ldim)) for i in range(len(fields))]

    # ...
    def eval_fields_regular_tensor_grid(self, grid, *fields, weights=None, overlap=0):
        """Evaluates one or several fields on a regular tensor grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate.

        weights : psydac.fem.basic.FemField or None, optional
            Weights field used to weight the basis functions thus
            turning them into NURBS. The same weights field is used
            for all of fields and they thus have to use the same basis functions.

        overlap : int
            How much to overlap. Only used in the distributed context.
            
        Returns
        -------
        List of list of ndarray
            List of the same length as `fields`, containing for each field
            a list of `self.ldim` arrays, i.e. one array for each logical coordinate.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """
        for f in fields:
            # Necessary if vector coeffs is distributed across processes
            if not f.coeffs.ghost_regions_in_sync:
                f.coeffs.update_ghost_regions()

        result = []
        if isinstance(overlap, int):
            overlap = [overlap] * self.ldim
        if weights is not None:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)

                result.append(self._spaces[i].eval_fields_regular_tensor_grid(grid,
                                                                              *fields_i,
                                                                              weights=weights.fields[i],
                                                                              overlap=overlap[i]))
        else:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)

                result.append(self._spaces[i].eval_fields_regular_tensor_grid(grid,
                                                                              *fields_i,
                                                                              overlap=overlap[i]))
        
        return result

    # ...
    def eval_fields_irregular_tensor_grid(self, grid, *fields, weights=None, overlap=0):
        """Evaluates one or several fields on an irregular tensor grid i.e.
        a tensor grid where the number of points per cell depends on the cell.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        *fields : tuple of psydac.fem.basic.FemField
            Fields to evaluate.

        weights : psydac.fem.basic.FemField or None, optional
            Weights field used to weight the basis functions thus
            turning them into NURBS. The same weights field is used
            for all of fields and they thus have to use the same basis functions.

        overlap : int
            How much to overlap. Only used in the distributed context.
            
        Returns
        -------
        List of list of ndarray
            List of the same length as `fields`, containing for each field
            a list of `self.ldim` arrays, i.e. one array for each logical coordinate.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """
        for f in fields:
            # Necessary if vector coeffs is distributed across processes
            if not f.coeffs.ghost_regions_in_sync:
                f.coeffs.update_ghost_regions()

        result = []
        if isinstance(overlap, int):
            overlap = [overlap] * self.ldim
        if weights is not None:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)

                result.append(self._spaces[i].eval_fields_irregular_tensor_grid(grid,
                                                                                *fields_i,
                                                                                weights=weights.fields[i],
                                                                                overlap=overlap[i]))
        else:
            for i in range(self.ldim):
                fields_i = list(field.fields[i] for field in fields)

                result.append(self._spaces[i].eval_fields_irregular_tensor_grid(grid,
                                                                                *fields_i,
                                                                                overlap=overlap[i]))
        
        return result

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
