#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

# TODO: - have a block version for VectorSpace when all component spaces are the same

from functools import reduce
from typing import Optional

import numpy as np

from sympde.topology.space import BasicFunctionSpace
from sympde.topology.callable_mapping import BasicCallableMapping
# from sympde.topology.datatype import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType

from psydac.linalg.block import BlockVectorSpace
from psydac.fem.basic    import FemSpace, FemField

__all__ = ('VectorFemSpace', 'MultipatchFemSpace')

#===============================================================================
class VectorFemSpace(FemSpace):
    """
    FEM space with a vector basis defined on a single patch.
    This class is used to represent either spaces of vector-valued FEM fields,
    or product spaces involved in systems of equations.

    Parameters
    ----------
    *spaces : FemSpace
        Single-patch FEM spaces, either scalar or vector-valued.
    """
    def __init__(self, *spaces):

        # Check that all input spaces are of the correct type
        assert all(isinstance(V, FemSpace) for V in spaces)

        # We do not accept multipatch spaces yet
        assert not any(V.is_multipatch for V in spaces)

        # All input spaces are flattened into a tuple `new_spaces` of scalar spaces
        new_spaces = [sp.spaces if isinstance(sp, VectorFemSpace) else [sp] for sp in spaces]
        new_spaces = tuple(sp2 for sp1 in new_spaces for sp2 in sp1)

        # Check that we indeed have scalar spaces only
        assert not any(V.is_vector_valued for V in new_spaces)

        # Check that all spaces have the same parametric dimension
        ldims = [V.ldim for V in new_spaces]
        assert len(set(ldims)) == 1

        # Make sure that all spaces have the same periodicity along each axis
        periodic = [V.periodic for V in new_spaces]
        for pp in zip(*periodic):
            assert len(set(pp)) == 1

        # Make sure that all spaces have the same mapping or no mapping at all
        # Mapping must be of type BasicCallableMapping defined in SymPDE
        # [YG, 27.03.2025]: this class was setting its mapping to None
        mappings = [V.mapping for V in new_spaces]
        assert len(set(mappings)) == 1
        assert mappings[0] is None or isinstance(mappings[0], BasicCallableMapping)

        # Make sure that all spaces have the same number of cells along each axis
        # [YG, 27.03.2025]: This is not part of the abstract interface of
        #       FemSpace and it assumes that all spaces are TensorFemSpaces
        ncells = [V.ncells for V in new_spaces]
        for nc in zip(*ncells):
            assert len(set(nc)) == 1

        # Compute the SymPDE symbolic space from the symbolic spaces of the input spaces
        symbolic_spaces = [V.symbolic_space for V in spaces]
        symbolic_space = reduce(lambda x, y: x * y, symbolic_spaces) if all(symbolic_spaces) else None

        # Compute the VectorSpace of the coefficients
        coeff_space = BlockVectorSpace(*[V.coeff_space for V in new_spaces])

        # Store information in private attributes
        self._ldim           : int              = ldims[0]
        self._periodic       : tuple[bool, ...] = periodic[0]
        self._spaces         : tuple[FemSpace]  = new_spaces
        self._coeff_space    : BlockVectorSpace = coeff_space
        self._ncells         : tuple[int, ...]  = ncells[0] # not used in the abstract interface
        self._mapping        : Optional[BasicCallableMapping] = mappings[0]
        self._symbolic_space : Optional[BasicFunctionSpace] = symbolic_space

        # ++++++++++++++ Extra operations for multigrid methods ++++++++++++++
        # Initialize the dictionary that will store the refined VectorFemSpaces
        self._refined_space = {}

        # Compute the refined VectorFemSpaces from the refined spaces of the
        # scalar spaces `new_spaces`. The method `set_refined_space` is used to
        # update the dictionary, and it perfoms additional checks. We also use
        # the property `spaces` which is not part of the abstract interface.
        self.set_refined_space(self.ncells, self)
        for key in self.spaces[0]._refined_space:
            if key != tuple(self.ncells):
                V_fine = VectorFemSpace(*[V._refined_space[key] for V in self.spaces])
                self.set_refined_space(key, V_fine)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #--------------------------------------------------------------------------
    # Abstract interface: read-only attributes
    #--------------------------------------------------------------------------
    @property
    def ldim(self):
        """ Parametric dimension.
        """
        return self._ldim

    @property
    def periodic(self):
        """
        Tuple of booleans: along each logical dimension,
        say if domain is periodic.
        :rtype: tuple[bool]
        """
        return self._periodic

    @property
    def mapping(self):
        # [YG, 27.03.2025]: not clear why there should be no mapping here
        #return None
        return self._mapping

    @property
    def coeff_space(self):
        """
        Vector space of the coefficients (mapping invariant).
        :rtype: psydac.linalg.block.BlockVectorSpace
        """
        return self._coeff_space

    @property
    def symbolic_space( self ):
        return self._symbolic_space

    @symbolic_space.setter
    def symbolic_space( self, symbolic_space ):
        assert isinstance(symbolic_space, BasicFunctionSpace)
        self._symbolic_space = symbolic_space

    @property
    def patch_spaces(self):
        return (self,)

    @property
    def component_spaces(self):
        return self._spaces

    @property
    def axis_spaces(self):
        raise NotImplementedError('Vector Fem space has no list of axis spaces')

    @property
    def is_multipatch(self):
        return False

    @property
    def is_vector_valued(self):
        return True

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
    def nbasis(self):
        dims = [V.nbasis for V in self.spaces]
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
        # [YG, 27.03.2025]: It appears that this method is assuming that the
        # refined space has ldim=2, and that the number of cells is the same
        # along each axis. These two conditions are very strong.
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
class MultipatchFemSpace(FemSpace):
    """
    Product of single-patch FEM spaces.

    Parameters
    ----------
    *spaces : FemSpace
        Single-patch FEM spaces, either scalar or vector-valued.

    connectivity : dict, optional
        Dictionary representing the connectivity between the patches.
    """
    def __init__(self, *spaces, connectivity=None):
        if connectivity is None:
            connectivity = {}

        # [YG, 28.03.2025]: What happens if we have only one space?
        if len(spaces) == 1:
            return

        self._spaces = spaces

        # ... make sure that all spaces have the same parametric dimension
        ldims = [V.ldim for V in self.spaces]
        assert len(np.unique(ldims)) == 1

        self._ldim = ldims[0]
        # ...

        self._coeff_space     = BlockVectorSpace(*[V.coeff_space for V in self.spaces], connectivity=connectivity)
        self._symbolic_space  = None
        self._connectivity    = connectivity.copy()

    #--------------------------------------------------------------------------
    # Abstract interface: read-only attributes
    #--------------------------------------------------------------------------
    @property
    def ldim(self):
        """ Parametric dimension.
        """
        return self._ldim

    @property
    def periodic(self):
        # [YG, 28.03.2025]: this is not consistent with the abstract interface,
        # which requires a tuple of booleans, but the periodicity of a multipatch
        # space is not well defined in general.
        return [V.periodic for V in self.spaces]

    @property
    def mapping(self):
        return None

    @property
    def coeff_space(self):
        """
        Vector space of the coefficients (mapping invariant).
        :rtype: psydac.linalg.basic.BlockVectorSpace
        """
        return self._coeff_space

    @property
    def symbolic_space( self ):
        return self._symbolic_space

    @symbolic_space.setter
    def symbolic_space( self, symbolic_space ):
        assert isinstance(symbolic_space, BasicFunctionSpace)
        self._symbolic_space = symbolic_space

    @property
    def patch_spaces(self):
        return self._spaces

    @property
    def component_spaces(self):
        """
        Return the component spaces (self if scalar-valued) as a tuple.
        """
        if self.is_vector_valued:
            # should we return here the multipatch scalar-valued space?
            raise NotImplementedError('Component spaces not implemented for multipatch spaces')
        else:
            return self._spaces

    @property
    def axis_spaces(self):
        raise NotImplementedError('Multipatch space has no list of axis spaces')

    @property
    def is_multipatch(self):
        return True

    @property
    def is_vector_valued(self):
        return self.patch_spaces[0].is_vector_valued

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
        raise NotImplementedError( "MultipatchFemSpace not yet operational" )

    # ...
    def integral( self, f ):
        raise NotImplementedError( "MultipatchFemSpace not yet operational" )

    #--------------------------------------------------------------------------
    # Other properties and methods
    #--------------------------------------------------------------------------
    @property
    def nbasis(self):
        dims = [V.nbasis for V in self.spaces]
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
