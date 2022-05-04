import numpy as np

from sympde.topology.mapping import Mapping
from sympde.topology.analytical_mapping import IdentityMapping
from sympde.topology.datatype import UndefinedSpaceType, H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType

from psydac.mapping.discrete import SplineMapping
from psydac.fem.basic import FemField
from psydac.fem.vector import ProductFemSpace, VectorFemSpace
from psydac.core.bsplines import cell_index
from psydac.core.kernels import (pushforward_2d_l2, pushforward_3d_l2,
                                 pushforward_2d_hdiv, pushforward_3d_hdiv,
                                 pushforward_2d_hcurl, pushforward_3d_hcurl)

class Pushforward:
    """
    Class used to help push-forwarding several fields using the 
    same mapping

    Parameters 
    ----------
    mapping : SplineMapping or Mapping
        Mapping used to push-forward
    
    grid : list of arrays
        Grid on which fields and other quantities will be evaluated
    
    npts_per_cell : tuple of int or int, optional
        Number of points per cell 
    """
    _eval_functions = {
        0: 'eval_fields_irregular_tensor_grid',
        1: 'eval_fields_regular_tensor_grid',
        2: 'NotImplementedError'
    }
    def __init__(self, mapping, grid, npts_per_cell=None, local_domain=None, global_domain=None, grid_local=None):
        # Get ldim
        ldim = mapping.ldim
        self.is_identity = isinstance(mapping, IdentityMapping)
        

        self.ldim = ldim
        self.jac_temp = None
        self.inv_jac_temp = None
        self.jac_det_temp = None

        # Process grid argument
        # Check consistency
        assert len(grid) == ldim
        grid_as_arrays = [np.array(grid[i]) for i in range(ldim)]
        assert all(grid_as_arrays[i].ndim == grid_as_arrays[i + 1].ndim for i in range(ldim - 1))
        
        self.npts_per_cell = None
        self.cell_indexes = None
        # 3 cases
        # 1: irregular tensor grid
        if grid_as_arrays[0].ndim == 1 and npts_per_cell is None:
            self.grid_type = 0
            self.grid = grid_as_arrays

        # 2: regular tensor grid
        elif grid_as_arrays[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * ldim
            assert len(npts_per_cell) == ldim
            for i in range(ldim):
                grid_as_arrays[i] = np.reshape(grid_as_arrays[i], (len(grid_as_arrays[i])//npts_per_cell[i], npts_per_cell[i]))
            self.grid_type = 1
            self.grid = grid_as_arrays
            self.npts_per_cell = npts_per_cell
        
        # 3: irregular grid
        elif grid_as_arrays[0].ndim == ldim:
            self.grid_type = 2
            self.grid = grid_as_arrays

        else:
            raise ValueError("Grid argument is not understood")
        if isinstance(mapping, Mapping):
            meshgrids = np.meshgrid(*grid_local, indexing='ij', sparse=True)

        if isinstance(mapping, Mapping):
            assert mapping.is_analytical
            # No support for non analytical mappings for now
            mapping_call = mapping.get_callable_mapping()
            self.jacobian = lambda : np.ascontiguousarray(
                                                  np.moveaxis(
                                                      mapping_call.jacobian(*meshgrids), [0, 1], [-1, -2]
                                                  )
                                              )
            self.jacobian_inv = lambda : np.ascontiguousarray(
                                                      np.moveaxis(
                                                          mapping_call.jacobian_inv(*meshgrids), [0, 1], [-1, -2]
                                                      )
                                                  )
            self.jacobian_det = lambda : np.ascontiguousarray(
                                                      np.sqrt(mapping_call.metric_det(*meshgrids))
                                                  )
            self.local_domain = local_domain
            self.global_domain = global_domain
        
        elif isinstance(mapping, SplineMapping):
            
            if self.grid_type == 1:
                self.jacobian = lambda : mapping.jac_mat_regular_tensor_grid(grid_as_arrays)
                self.jacobian_inv = lambda : mapping.inv_jac_mat_regular_tensor_grid(grid_as_arrays)
                self.jacobian_det = lambda : mapping.jac_det_regular_tensor_grid(grid_as_arrays)
            elif self.grid_type == 0:
                self.jacobian = lambda : mapping.jac_mat_irregular_tensor_grid(grid_as_arrays)
                self.jacobian_inv = lambda : mapping.inv_jac_mat_irregular_tensor_grid(grid_as_arrays)
                self.jacobian_det = lambda : mapping.jac_det_irregular_tensor_grid(grid_as_arrays)
            else:
                raise NotImplementedError("Unstructured Grids aren't supported yet")
            
            self.local_domain = mapping.space.local_domain
            self.global_domain = ((0,) * ldim, tuple(nc_i - 1 for nc_i in mapping.space.ncells))
        
        self._eval_func = self._eval_functions[self.grid_type]
    
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

        # Turn fields arg into a dictionnary
        if isinstance(fields, FemField):
            fields = {'field': fields}

        if isinstance(fields, list):
            fields = {f'field_{i}': fields[i] for i in range(len(fields))}
    
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

        # Set the local_domain and global_domain if it hasn't been yet
        if self.local_domain is None and self.global_domain is None:
            try:
                self.local_domain = list(space_dict.keys())[0].local_domain
                self.global_domain = ((0,) * self.ldim, tuple(nc_i - 1 for nc_i in list(space_dict.keys())[0].ncells))
            except AttributeError:
                self.local_domain = list(space_dict.keys())[0].spaces[0].local_domain
                self.global_domain = ((0,) * self.ldim, tuple(nc_i - 1 for nc_i in list(space_dict.keys())[0].spaces[0].ncells))
        
        # Call pushforward dispatcher
        out = []
        for space, (field_list, field_names) in space_dict.items():
            list_pushed_fields = self._dispatch_pushforward(space, field_list)
            out.extend((field_names[i], list_pushed_fields[i]) for i in range(len(list_pushed_fields)))
        
        return out
    
    def _dispatch_pushforward(self, space, field_list):
        """
        Simple function to take care of the kind of the spaces.
        """
        # Check the kind
        try:
            kind = space.symbolic_space.kind
        except AttributeError:
            kind = UndefinedSpaceType()

        # if IdentityMapping do as if everything was H1
        if kind is H1SpaceType() or kind is UndefinedSpaceType() or self.is_identity:
            return self._pushforward_h1(space, field_list)
        
        elif kind is L2SpaceType():
            return self._pushforward_l2(space, field_list)
        
        elif kind is HcurlSpaceType():
            return self._pushforward_hcurl(space, field_list)
        
        elif kind is HdivSpaceType():
            return self._pushforward_hdiv(space, field_list)
    
    def _pushforward_h1(self, space, field_list):
        """
        Pushforward of h1 spaces

        Parameters 
        ----------
        space : FemSpace
        
        fields_list : list of FemFields
        """
        overlap, index_trim = self._index_trimming_helper(space)           

        fields_eval = getattr(space, self._eval_func)(self.grid, *field_list, overlap=overlap)
        
        if isinstance(space, (VectorFemSpace, ProductFemSpace)):
            return [tuple(np.ascontiguousarray(fields_eval[i][index_trim[i] + (j,)]) for i in range(self.ldim)) for j in range(len(field_list))]
        else:
            return [np.ascontiguousarray(fields_eval[index_trim + (j,)]) for j in range(len(field_list))]
    
    def _pushforward_l2(self, space, field_list):
        """
        Pushforward of l2 spaces

        Parameters 
        ----------
        space : FemSpace
        
        fields_list : list of FemFields        
        """
        overlap, index_trim = self._index_trimming_helper(space)           

        fields_eval = getattr(space, self._eval_func)(self.grid, *field_list, overlap=overlap)
        
        if self.jac_det_temp is None:
            self.jac_det_temp = self.jacobian_det()
        
        if isinstance(space, (VectorFemSpace, ProductFemSpace)):

            fields_to_push = [np.ascontiguousarray(fields_eval[i][index_trim[i]]) for i in range(self.ldim)]
            pushed_fields_list = [np.zeros_like(fields_to_push[i]) for i in range(self.ldim)]
            if self.ldim == 2:
                for i in range(2):
                    pushforward_2d_l2(fields_to_push[i], self.jac_det_temp, pushed_fields_list[i])
            if self.ldim == 3:
                for i in range(3):
                    pushforward_3d_l2(fields_to_push[i], self.jac_det_temp, pushed_fields_list[i])               

            return [tuple(np.ascontiguousarray(pushed_fields_list[i][..., j]) for i in range(self.ldim)) for j in range(len(field_list))]

        else:

            fields_to_push = np.ascontiguousarray(fields_eval[index_trim])
            pushed_fields = np.zeros_like(fields_to_push)
            if self.ldim == 2:
                pushforward_2d_l2(fields_to_push, self.jac_det_temp, pushed_fields)
            if self.ldim == 3:
                pushforward_3d_l2(fields_to_push, self.jac_det_temp, pushed_fields)
            
            return [np.ascontiguousarray(pushed_fields[..., j]) for j in range(len(field_list))]

    def _pushforward_hdiv(self, space, field_list):
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
        
        fields_eval = np.ascontiguousarray(np.stack([fields_eval[i][(*index_trim[i], Ellipsis)] for i in range(self.ldim)], axis=0))

        pushed_fields = np.zeros((fields_eval.shape[-1], *fields_eval.shape[:-1]))
        if self.ldim == 2:
            pushforward_2d_hdiv(fields_eval, self.jac_temp, pushed_fields)
        if self.ldim == 3:
            pushforward_3d_hdiv(fields_eval, self.jac_temp, pushed_fields)
        
        return [tuple(np.ascontiguousarray(pushed_fields[j, i]) for i in range(self.ldim)) for j in range(len(field_list))]

    def _pushforward_hcurl(self, space, field_list):
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
        pushed_fields = np.zeros((fields_eval.shape[-1], *fields_eval.shape[:-1]))

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
        local_domain :  tuple of tuples
            Local domain of the misaligned space.
        
        Returns
        -------
        index_trim : tuple of slices
            tuple of ldim slice objects that 
        """
        global_starts, global_ends = self.global_domain
        target_starts, target_ends = self.local_domain
        local_starts, local_ends = local_domain

        ldim = self.ldim

        index_trimming = ()

        if self.npts_per_cell is not None:
            difference_starts = tuple(l_s  - t_s  for l_s, t_s in zip(local_starts, target_starts))
            difference_ends = tuple(l_e - t_e for l_e, t_e in zip(local_ends, target_ends))
            for i in range(ldim):
                if local_starts[i] != global_starts[i]:
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
                if local_starts[i] != global_starts[i]:
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
            raise NotImplementedError("Not supported yet")
        return index_trimming

    def _index_trimming_helper(self, space):
        """
        Function that computes the overlap and index trimming to avoid repeating code

        Parameters
        ----------
        space : FemSpace
        """
        if isinstance(space, (ProductFemSpace, VectorFemSpace)):
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