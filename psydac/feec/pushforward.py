#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from sympde.topology.mapping import Mapping
from sympde.topology.callable_mapping import CallableMapping
from sympde.topology.analytical_mapping import IdentityMapping
from sympde.topology.datatype import UndefinedSpaceType, H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType

from psydac.mapping.discrete import SplineMapping
from psydac.fem.basic import FemField
from psydac.fem.vector import MultipatchFemSpace, VectorFemSpace
from psydac.core.bsplines import cell_index
from psydac.core.field_evaluation_kernels import (pushforward_2d_l2, pushforward_3d_l2,
                                                  pushforward_2d_hdiv, pushforward_3d_hdiv,
                                                  pushforward_2d_hcurl, pushforward_3d_hcurl)

# L2 and Hdiv push-forward algorithms use the metric determinant and
# not the jacobian determinant. For this reason, sign descrepancies can
# happen when comparing against algorithms which use the latter.

__all__ = ('Pushforward',)

class Pushforward:
    """
    Class used to help push-forwarding several fields using the
    same mapping. This class does not perform any checks on its arguments.

    Parameters
    ----------
    grid : list of arrays
        Grid on which fields and other quantities will be evaluated.
        If it's a regular tensor grid, then it is expected to be
        a list of 2-D arrays with number of cells as the first dimension.

    mapping : SplineMapping or Mapping or None
        Mapping used to push-forward. None is equivalent to
        the identity mapping.

    npts_per_cell : tuple of int or int, optional
        Number of points per cell

    cell_indexes : list of arrays of int, optional
        Cell indices of the points in grid for each direction.

    grid_type : int, optional
        0 for irregular tensor grid,
        1 for regular tensor grid,
        2 for unstructured grid.

    local_domain : 2-tuple of tuple of int
        Index of the first and last cell owned by the current process
        for each direction given as a tuple of starting index and a tuple
        of ending index. This is most commonly given by the attribute
        ``TensorFemSpace.local_domain``.

    global_ends : tuple of int
        Index of the last cell of the domain for each direction.
        This is simply the tuple of ``ncell - 1`` in each direction.

    grid_local : list of arrays
        Grid that is local to the process. This is necessary when the mapping
        is an Analytical mapping that isn't the identity.
        In serial this is the same as ``grid``.

    Notes
    -----
    L2 and Hdiv push-forward algorithms use the metric determinant and
    not the jacobian determinant. For this reason, sign descrepancies can
    happen when comparing against algorithms which use the latter.
    """
    _eval_functions = {
        0: 'eval_fields_irregular_tensor_grid',
        1: 'eval_fields_regular_tensor_grid',
        2: 'NotImplementedError'
    }

    def __init__(
        self,
        grid,
        mapping=None,
        npts_per_cell=None,
        cell_indexes=None,
        grid_type=None,
        local_domain=None,
        global_ends=None,
        grid_local=None,
        ):

        self.is_identity = mapping is None or isinstance(mapping, IdentityMapping)
        if self.is_identity:
            ldim = len(grid)
        else:
            ldim = mapping.ldim

        self.ldim = ldim
        self.jac_temp = None
        self.inv_jac_temp = None
        self.sqrt_metric_det_temp = None

        self.grid=grid
        self.npts_per_cell = npts_per_cell
        self.cell_indexes = cell_indexes
        self.grid_type=grid_type
        if grid_local is None:
            grid_local=grid

        if isinstance(mapping, Mapping):
            self._mesh_grids = np.meshgrid(*grid_local, indexing='ij', sparse=True)
            if isinstance(mapping.get_callable_mapping(), SplineMapping):
                c_m = mapping.get_callable_mapping()
                self.mapping = c_m
                self.local_domain = c_m.space.local_domain
                self.global_ends = tuple(nc_i - 1 for nc_i in c_m.space.ncells)
            else : 
                assert mapping.is_analytical
                self.mapping = mapping.get_callable_mapping()
                self.local_domain = local_domain
                self.global_ends = global_ends

        elif isinstance(mapping, SplineMapping):
            self.mapping = mapping
            self.local_domain = mapping.space.local_domain
            self.global_ends = tuple(nc_i - 1 for nc_i in mapping.space.ncells)

        else:
            assert self.is_identity
            self.local_domain = local_domain
            self.global_ends = global_ends

        self._eval_func = self._eval_functions[self.grid_type]

    def jacobian(self):
        if isinstance(self.mapping, CallableMapping):
            return np.ascontiguousarray(
                        np.moveaxis(
                            self.mapping.jacobian(*self._mesh_grids), [0, 1], [-2, -1]
                        )
                    )
        elif isinstance(self.mapping, SplineMapping):
            if self.grid_type == 0:
                return self.mapping.jac_mat_irregular_tensor_grid(self.grid)
            elif self.grid_type == 1:
                return self.mapping.jac_mat_regular_tensor_grid(self.grid)

    def jacobian_inv(self):
        if isinstance(self.mapping, CallableMapping):
            return np.ascontiguousarray(
                        np.moveaxis(
                            self.mapping.jacobian_inv(*self._mesh_grids), [0, 1], [-2, -1]
                        )
                    )
        elif isinstance(self.mapping, SplineMapping):
            if self.grid_type == 0:
                return self.mapping.inv_jac_mat_irregular_tensor_grid(self.grid)
            elif self.grid_type == 1:
                return self.mapping.inv_jac_mat_regular_tensor_grid(self.grid)

    def sqrt_metric_det(self):
        if isinstance(self.mapping, CallableMapping):
            return np.ascontiguousarray(
                        np.sqrt(self.mapping.metric_det(*self._mesh_grids))
                    )
        elif isinstance(self.mapping, SplineMapping):
            if self.grid_type == 0:
                return np.abs(self.mapping.jac_det_irregular_tensor_grid(self.grid))
            elif self.grid_type == 1:
                return np.abs(self.mapping.jac_det_regular_tensor_grid(self.grid))


    def __call__(self, fields):
        """
        Push-forward fields

        Parameters
        ----------
        fields : FemField, list of FemFields or dict {str: FemField}
            Fields to push-forward
        """
        # Check for lack of arguments
        if fields is None or fields == {}:
            return []

        # Turn fields arg into a dictionary
        if isinstance(fields, FemField):
            fields = {'field': fields}

        if isinstance(fields, list):
            fields = {f'field_{i}': fields[i] for i in range(len(fields))}

        if not isinstance(fields, dict):
            raise ValueError(f"fields should be a FemField, a list or a dict and not {type(fields)}")

        # Group fields by space
        space_dict = {}
        for field_name, field in fields.items():
            try:
                space_dict[field.space][0].append(field)
                space_dict[field.space][1].append(field_name)
            except KeyError:
                space_dict[field.space] = ([field], [field_name])

        # Compute cell_indexes if needed
        if self.grid_type == 0 and self.cell_indexes is None:
            try:
                breaks = list(space_dict.keys())[0].breaks
            except AttributeError:
                breaks = list(space_dict.keys())[0].spaces[0].breaks
            self.cell_indexes = [cell_index(breaks[i], i_grid=self.grid[i]) for i in range(self.ldim)]

        # Set the local_domain and global_ends if it hasn't been yet
        if self.local_domain is None and self.global_ends is None:
            try:
                self.local_domain = list(space_dict.keys())[0].local_domain
                self.global_ends = tuple(nc_i - 1 for nc_i in list(space_dict.keys())[0].ncells)
            except AttributeError:
                self.local_domain = list(space_dict.keys())[0].spaces[0].local_domain
                self.global_ends = tuple(nc_i - 1 for nc_i in list(space_dict.keys())[0].spaces[0].ncells)

        # Call pushforward dispatcher
        out = []
        for space, (field_list, field_names) in space_dict.items():
            list_pushed_fields = self._dispatch_pushforward(space, *field_list)
            out.extend((field_names[i], list_pushed_fields[i]) for i in range(len(list_pushed_fields)))

        return out

    def _dispatch_pushforward(self, space, *fields):
        """
        Simple function to take care of the kind of the spaces.

        Parameters
        ----------
        space : Femspace

        *fields : list of FemField
        """
        # Check the kind
        try:
            kind = space.symbolic_space.kind
        except AttributeError:
            kind = UndefinedSpaceType()

        # if IdentityMapping do as if everything was H1
        if kind is H1SpaceType() or kind is UndefinedSpaceType() or self.is_identity:
            return self._pushforward_h1(space, *fields)

        elif kind is L2SpaceType():
            return self._pushforward_l2(space, *fields)

        elif kind is HcurlSpaceType():
            return self._pushforward_hcurl(space, *fields)

        elif kind is HdivSpaceType():
            return self._pushforward_hdiv(space, *fields)

    def _pushforward_h1(self, space, *field_list):
        """
        Pushforward of h1 spaces

        Parameters
        ----------
        space : FemSpace

        fields_list : list of FemFields
        """
        overlap, index_trim = self._index_trimming_helper(space)

        fields_eval = getattr(space, self._eval_func)(self.grid, *field_list, overlap=overlap)

        if isinstance(space, (VectorFemSpace, MultipatchFemSpace)):
            return [tuple(np.ascontiguousarray(fields_eval[i][index_trim[i] + (j,)], dtype=field_list[0].coeffs.dtype) for i in range(self.ldim)) for j in range(len(field_list))]
        else:
            return [np.ascontiguousarray(fields_eval[index_trim + (j,)], dtype=field_list[0].coeffs.dtype) for j in range(len(field_list))]

    def _pushforward_l2(self, space, *field_list):
        """
        Pushforward of l2 spaces

        Parameters
        ----------
        space : FemSpace

        fields_list : list of FemFields
        """
        overlap, index_trim = self._index_trimming_helper(space)

        fields_eval = getattr(space, self._eval_func)(self.grid, *field_list, overlap=overlap)

        if self.sqrt_metric_det_temp is None:
            self.sqrt_metric_det_temp = self.sqrt_metric_det()

        if isinstance(space, (VectorFemSpace, MultipatchFemSpace)):

            fields_to_push = [np.ascontiguousarray(fields_eval[i][index_trim[i]]) for i in range(self.ldim)]
            pushed_fields_list = [np.zeros_like(fields_to_push[i], dtype=fields_eval[i].dtype) for i in range(self.ldim)]
            if self.ldim == 2:
                for i in range(2):
                    pushforward_2d_l2(fields_to_push[i], self.sqrt_metric_det_temp, pushed_fields_list[i])
            if self.ldim == 3:
                for i in range(3):
                    pushforward_3d_l2(fields_to_push[i], self.sqrt_metric_det_temp, pushed_fields_list[i])

            return [tuple(np.ascontiguousarray(pushed_fields_list[i][..., j]) for i in range(self.ldim)) for j in range(len(field_list))]

        else:

            fields_to_push = np.ascontiguousarray(fields_eval[index_trim])
            pushed_fields = np.zeros_like(fields_to_push, dtype=fields_eval.dtype)
            if self.ldim == 2:
                pushforward_2d_l2(fields_to_push, self.sqrt_metric_det_temp, pushed_fields)
            if self.ldim == 3:
                pushforward_3d_l2(fields_to_push, self.sqrt_metric_det_temp, pushed_fields)

            return [np.ascontiguousarray(pushed_fields[..., j]) for j in range(len(field_list))]

    def _pushforward_hdiv(self, space, *field_list):
        """
        Pushforward of hdiv spaces

        Parameters
        ----------
        space : FemSpace

        fields_list : list of FemFields
        """
        overlap, index_trim = self._index_trimming_helper(space)

        fields_eval = getattr(space, self._eval_func)(self.grid, *field_list, overlap=overlap)

        if self.jac_temp is None:
            self.jac_temp = self.jacobian()
        if self.sqrt_metric_det_temp is None:
            self.sqrt_metric_det_temp = self.sqrt_metric_det()

        fields_eval = np.ascontiguousarray(np.stack([fields_eval[i][(*index_trim[i], Ellipsis)] for i in range(self.ldim)], axis=0))

        pushed_fields = np.zeros((fields_eval.shape[-1], *fields_eval.shape[:-1]), dtype=fields_eval.dtype)
        if self.ldim == 2:
            pushforward_2d_hdiv(fields_eval, self.jac_temp, self.sqrt_metric_det_temp, pushed_fields)
        if self.ldim == 3:
            pushforward_3d_hdiv(fields_eval, self.jac_temp, self.sqrt_metric_det_temp, pushed_fields)

        return [tuple(np.ascontiguousarray(pushed_fields[j, i]) for i in range(self.ldim)) for j in range(len(field_list))]

    def _pushforward_hcurl(self, space, *field_list):
        """
        Pushforward of hcurl spaces

        Parameters
        ----------
        space : FemSpace

        fields_list : list of FemFields
        """
        overlap, index_trim = self._index_trimming_helper(space)
        fields_eval = getattr(space, self._eval_func)(self.grid, *field_list, overlap=overlap)
        if self.inv_jac_temp is None:
            self.inv_jac_temp = self.jacobian_inv()
        fields_eval = np.ascontiguousarray(np.stack([fields_eval[i][(*index_trim[i], Ellipsis)] for i in range(self.ldim)], axis=0))
        pushed_fields = np.zeros((fields_eval.shape[-1], *fields_eval.shape[:-1]), dtype=fields_eval.dtype)

        if self.ldim == 2:
            pushforward_2d_hcurl(fields_eval, self.inv_jac_temp, pushed_fields)
        if self.ldim == 3:
            pushforward_3d_hcurl(fields_eval, self.inv_jac_temp, pushed_fields)

        return [tuple(np.ascontiguousarray(pushed_fields[j, i]) for i in range(self.ldim)) for j in range(len(field_list))]

    def _compute_index_trimming(self, local_domain):
        """
        Computes the indexing needed to trim the arrays down
        to the local_domain of the mapping for alignement purposes.

        Parameters
        ----------
        local_domain :  tuple of tuples of ints
            Local domain of the misaligned space. It is usually given by
            `TensorFemSpace.local_domain`. It is a 2-tuple (`starts`, `ends`)
            of `self.ldim`-tuples or integers indices. The indices in `start`
            correspond to the first cell that the space owns in a particular
            direction and the indices in ends correspond to the first cell
            that doesn't belong to the space.

        Returns
        -------
        index_trim : tuple of slices
            tuple of ldim slice objects that convert from the misaligned
            domain + 1 cell of overlap to `self.local_domain`.
        """
        global_ends = self.global_ends
        target_starts, target_ends = self.local_domain
        local_starts, local_ends = local_domain

        ldim = self.ldim

        index_trimming = ()

        if self.npts_per_cell is not None:
            difference_starts = tuple(l_s  - t_s  for l_s, t_s in zip(local_starts, target_starts))
            difference_ends = tuple(l_e - t_e for l_e, t_e in zip(local_ends, target_ends))
            for i in range(ldim):
                if local_starts[i] != 0:
                    theoretical_start = (1 - difference_starts[i]) * self.npts_per_cell[i]
                else:
                    theoretical_start = 0
                if theoretical_start < 0:
                    raise ValueError(f"Spaces seem to be incompatible:\n"
                                     f"local domain of space 0 {self.local_domain}\n"
                                     f"local domain of the current space: {local_domain}")

                if local_ends[i] != global_ends[i]:
                    theoretical_end = - (1 + difference_ends[i]) * self.npts_per_cell[i]
                    if theoretical_end == 0: # We use negative index so it is needed to change -0 to None
                        theoretical_end = None

                    elif theoretical_end > 0:
                        raise ValueError(f"Spaces seem to be incompatible:\n"
                                        f"local domain of space 0 {self.local_domain}\n"
                                        f"local domain of the current space: {local_domain}")
                else:
                    theoretical_end = None


                index_trimming += (slice(theoretical_start, theoretical_end, 1),)

        elif self.cell_indexes is not None:
            for i in range(ldim):
                if local_starts[i] != 0:
                    i_start_local_overlap = np.searchsorted(self.cell_indexes[i], local_starts[i] - 1, side='left')
                    i_start_target = np.searchsorted(self.cell_indexes[i], target_starts[i], side='left')
                    theoretical_start = i_start_target - i_start_local_overlap
                else:
                    theoretical_start = 0
                if theoretical_start < 0:

                    raise ValueError(f"Spaces seem to be incompatible:\n"
                                     f"local domain of space 0 {self.local_domain}\n"
                                     f"local domain of the current space: {local_domain}")

                if local_ends[i] != global_ends[i]:
                    i_end_local_overlap = np.searchsorted(self.cell_indexes[i], local_ends[i] + 1, side='right') - 1
                    i_end_target = np.searchsorted(self.cell_indexes[i], target_ends[i], side='right') - 1
                    theoretical_end = - (i_end_local_overlap - i_end_target)
                    if theoretical_end == 0: # We use negative index so it is needed to change -0 to None
                        theoretical_end = None
                    elif theoretical_end > 0:
                        raise ValueError(f"Spaces seem to be incompatible:\n"
                                        f"local domain of space 0 {self.local_domain}\n"
                                        f"local domain of the current space: {local_domain}")
                else:
                    theoretical_end = None


                index_trimming += (slice(theoretical_start, theoretical_end, 1),)

        else:
            raise NotImplementedError("Unstructured grids are not supported yet")
        return index_trimming

    def _index_trimming_helper(self, space):
        """
        Function that computes the overlap and index trimming to avoid repeating code

        Parameters
        ----------
        space : FemSpace
        """
        if isinstance(space, (MultipatchFemSpace, VectorFemSpace)):
            overlap = []
            index_trim = []
            for i in range(len(space.spaces)):
                overlap_i, index_trim_i = self._index_trimming_helper(space.spaces[i])
                overlap.append(overlap_i)
                index_trim.append(index_trim_i)
        else:
            if space.local_domain != self.local_domain:
                overlap = 1
                index_trim = self._compute_index_trimming(space.local_domain)
            else:
                overlap = 0
                index_trim = (slice(0, None, 1),) * self.ldim
        return overlap, index_trim
