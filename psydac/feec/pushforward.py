import numpy as np

from sympde.topology.mapping import Mapping
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
    def __init__(self, mapping, grid, npts_per_cell=None):
        # Process mapping argument
        ldim = mapping.ldim

        if isinstance(mapping, Mapping):
            assert mapping.is_analytical
            # No support for non analytical mappings for now
            mapping_call = mapping.get_callable_mapping()
            self.jacobian = lambda grid: mapping_call.jacobian(*grid)
            self.jacobian_inv = lambda grid: mapping_call.jacobian_inv(*grid)
            self.jacobian_det = lambda grid: np.sqrt(mapping_call.metric_det(*grid))
            self.local_domain = None
            self.global_domain = None

        elif isinstance(mapping, SplineMapping):
            self.jacobian = mapping.jac_mat_grid
            self.jacobian_inv = mapping.inv_jac_mat_grid
            self.jacobian_det = mapping.jac_det_grid

            self.local_domain = mapping.space.local_domain
            self.global_domain = ((0,) * ldim, tuple(nc_i - 1 for nc_i in mapping.space.ncells))

        
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

        
    def __call__(self, fields):
        """
        Push-forward fields

        Parameters
        ----------
        fields : FemField or list of dict
            Fields to push-forward
        """
        # group fields by space
        space_dict = {}
        if isinstance(fields, FemField):
            fields = {'field': fields}

        if isinstance(fields, list):
            fields = {f'field_{i}': fields[i] for i in range(len(fields))}
    
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
            self.cell_indexes = [cell_index(breaks[i], i_grid=self.grid[0]) for i in range(self.ldim)]
        
        # Set the local_domain and global_domain if it hasn't been yet
        if self.local_domain is None and self.global_domain is None:
            try:
                self.local_domain = list(space_dict.keys())[0].local_domain
                self.global_domain = ((0,) * self.ldim, tuple(nc_i - 1 for nc_i in list(space_dict.keys())[0].ncells))
            except AttributeError:
                self.local_domain = list(space_dict.keys())[0].spaces[0].local_domain
                self.global_domain = ((0,) * self.ldim, tuple(nc_i - 1 for nc_i in list(space_dict.keys())[0].spaces[0].ncells))

        out = []
        for space, (field_list, field_names) in space_dict.items():
            list_pushed_fields = self._dispatch_pushforward(space, field_list)
            out.extend((field_names[i], list_pushed_fields[i]) for i in range(len(list_pushed_fields)))
        return out
    
    def _dispatch_pushforward(self, space, field_list):
        """
        Simple function to take care of the kindof the spaces.
        """
        try:
            kind = space.symbolic_space.kind
        except AttributeError:
            kind = UndefinedSpaceType()
        
        if kind is H1SpaceType() or UndefinedSpaceType():
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
        if isinstance(space, (VectorFemSpace, ProductFemSpace)):
            overlap = [int(space.spaces[i].local_domain != self.local_domain) for i in range(self.ldim)]
            index_trim = []
            for i in range(self.ldim):
                if overlap[i] == 1:
                    index_trim.append(self._compute_index_trimming(local_domain=space.spaces[i].local_domain))
                else:
                    index_trim.append((slice(0, None, 1),) * self.ldim)
        else:
            overlap = int(space.local_domain != self.local_domain)
            if overlap == 1:
                index_trim = self._compute_index_trimming(local_domain=space.local_domain)
            else:
                index_trim = (slice(0, None, 1),) * self.ldim            

        if self.grid_type == 0:
            fields_eval = space.eval_fields_irregular_tensor_grid(self.grid, *field_list, overlap=overlap)
        elif self.grid_type == 1:
            fields_eval = space.eval_fields_regular_tensor_grid(self.grid, *field_list, overlap=overlap)
        
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
        if isinstance(space, (VectorFemSpace, ProductFemSpace)):
            overlap = [int(space.spaces[i].local_domain != self.local_domain) for i in range(self.ldim)]
            index_trim = []
            for i in range(self.ldim):
                if overlap[i] == 1:
                    index_trim.append(self._compute_index_trimming(local_domain=space.spaces[i].local_domain))
                else:
                    index_trim.append((slice(0, None, 1),) * self.ldim)
        else:
            overlap = int(space.local_domain != self.local_domain)
            if overlap == 1:
                index_trim = self._compute_index_trimming(local_domain=space.local_domain)
            else:
                index_trim = (slice(0, None, 1),) * self.ldim              
        
        if self.jac_det_temp is None:
            self.jac_det_temp = self.jacobian_det(self.grid)
        
        if self.grid_type == 0:
            fields_eval = space.eval_fields_irregular_tensor_grid(self.grid, *field_list, overlap=overlap)

        elif self.grid_type == 1:
            fields_eval = space.eval_fields_regular_tensor_grid(self.grid, *field_list, overlap=overlap)

        if isinstance(space, (VectorFemSpace, ProductFemSpace)):
            pushed_fields_list = [np.zeros_like(fields_eval[i][index_trim[i]]) for i in range(self.ldim)]
            if self.ldim == 2:
                for i in range(2):
                    pushforward_2d_l2(fields_eval[i][index_trim[i]], self.jac_det_temp, pushed_fields_list[i])
            if self.ldim == 3:
                for i in range(3):
                    pushforward_3d_l2(fields_eval[i][index_trim[i]], self.jac_det_temp, pushed_fields_list[i])               

            return [tuple(np.ascontiguousarray(pushed_fields_list[i][..., j]) for i in range(self.ldim)) for j in range(len(field_list))]
        else:
            pushed_fields = np.zeros_like(fields_eval[index_trim])
            if self.ldim == 2:
                pushforward_2d_l2(fields_eval[index_trim], self.jac_det_temp, pushed_fields)
            if self.ldim == 3:
                pushforward_3d_l2(fields_eval[index_trim], self.jac_det_temp, pushed_fields)
            
            return [np.ascontiguousarray(pushed_fields[..., j]) for j in range(len(field_list))]

    def _pushforward_hdiv(self, space, field_list):
        """
        Pushforward of hdiv spaces

        Parameters 
        ----------
        space : FemSpace
        
        fields_list : list of FemFields        
        """
        overlap = [int(space.spaces[i].local_domain != self.local_domain) for i in range(self.ldim)]
        for i in range(self.ldim):
            index_trim = []
            if overlap[i] == 1:
                index_trim.append(self._compute_index_trimming(local_domain=space.spaces[i].local_domain))
            else:
                index_trim.append((slice(0, None, 1),) * self.ldim)        
        
        if self.jac_temp is None:
            self.jac_temp = self.jacobian(self.grid)
        
        if self.grid_type == 0:
            fields_eval = space.eval_fields_irregular_tensor_grid(self.grid, *field_list, overlap=overlap)

        elif self.grid_type == 1:
            fields_eval = space.eval_fields_regular_tensor_grid(self.grid, *field_list, overlap=overlap)
        
        fields_eval = np.stack([fields_eval[i][index_trim[i], :] for i in range(self.ldim)], axis=0)

        pushed_fields = np.zeros_like(fields_eval)

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
        overlap = [int(space.spaces[i].local_domain != self.local_domain) for i in range(self.ldim)]
        for i in range(self.ldim):
            index_trim = []
            if overlap[i] == 1:
                index_trim.append(self._compute_index_trimming(local_domain=space.spaces[i].local_domain))
            else:
                index_trim.append((slice(0, None, 1),) * self.ldim)   
        
        if self.inv_jac_temp is None:
            self.inv_jac_temp = self.jacobian(self.grid)
        
        if self.grid_type == 0:
            fields_eval = space.eval_fields_irregular_tensor_grid(self.grid, *field_list, overlap=overlap)

        elif self.grid_type == 1:
            fields_eval = space.eval_fields_regular_tensor_grid(self.grid, *field_list, overlap=overlap)
        
        fields_eval = np.stack([fields_eval[i][index_trim[i], :] for i in range(self.ldim)], axis=0)

        pushed_fields = np.zeros_like(fields_eval)

        if self.ldim == 2:
            pushforward_2d_hcurl(fields_eval, self.jac_temp, pushed_fields)
        if self.ldim == 3:
            pushforward_3d_hcurl(fields_eval, self.jac_temp, pushed_fields)
        
        return [tuple(np.ascontiguousarray(pushed_fields[j, i]) for i in range(self.ldim)) for j in range(len(field_list))]
    
    def _compute_index_trimming(self, local_domain):
        """
        Computes the indexing needed to trim the arrays down 
        to the local_domain of the mapping for alignement purposes.

        Parameters
        ----------
        local_domain :  tuple of tuples
            Domain of the space that needs to be trimmed down.
        """
        global_starts, global_ends = self.global_domain
        target_starts, target_ends = self.local_domain
        local_starts, local_ends = local_domain

        ldim = self.ldim

        index_trimming = ()

        difference_starts = tuple(l_s - t_s  for l_s, t_s in zip(local_starts, target_starts))
        difference_ends = tuple(l_e - t_e for l_e, t_e in zip(local_ends, target_ends))

        if self.npts_per_cell is not None:
            for i in range(ldim):
                if local_starts[i] != global_starts[i]:
                    theoretical_start = (difference_starts[i] - 1) * self.npts_per_cell[i] 
                    if theoretical_start < 0:
                        theoretical_start = 0
                else:
                    theoretical_start = 0

                if local_ends[i] != global_ends[i]:
                    theoretical_end = - (1 + difference_ends[i]) * self.npts_per_cell[i]  
                    if theoretical_end >= 0:
                        theoretical_end = None
                else:
                    theoretical_end = None
                index_trimming += (slice(theoretical_start, theoretical_end, 1),)
        elif self.cell_indexes is not None:
            raise NotImplementedError("WIP")
        else:
            raise NotImplementedError("Not supported yet")
        return index_trimming