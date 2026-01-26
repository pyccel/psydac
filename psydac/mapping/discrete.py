#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from itertools import product
import string
import random

import yaml
import numpy as np
import h5py

from sympde.topology.callable_mapping import BasicCallableMapping

from psydac.fem.basic    import FemField
from psydac.fem.tensor   import TensorFemSpace


__all__ = ('SplineMapping', 'NurbsMapping')

#==============================================================================
class SplineMapping(BasicCallableMapping):

    def __init__(self, *components, name=None):

        # Sanity checks
        assert len(components) >= 1
        assert all(isinstance(c, FemField) for c in components)
        assert all(isinstance(c.space, TensorFemSpace) for c in components)
        assert all(c.space is components[0].space for c in components)

        # Store spline space and one field for each coordinate X_i
        self._space  = components[0].space
        self._fields = components

        # Store number of logical and physical dimensions
        self._ldim = components[0].space.ldim
        self._pdim = len(components)

        # Create helper object for accessing control points with slicing syntax
        # as if they were stored in a single multi-dimensional array C with
        # indices [i1, ..., i_n, d] where (i1, ..., i_n) are indices of logical
        # coordinates, and d is index of physical component of interest.
        self._control_points = SplineMapping.ControlPoints(self)
        self._name           = name

    @property
    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    #--------------------------------------------------------------------------
    # Option [1]: initialize from TensorFemSpace and pre-existing mapping
    #--------------------------------------------------------------------------
    @classmethod
    def from_mapping(cls, tensor_space, mapping):

        assert isinstance(tensor_space, TensorFemSpace)
        assert isinstance(mapping, BasicCallableMapping)
        assert tensor_space.ldim == mapping.ldim

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        fields = [FemField(tensor_space) for d in range(mapping.pdim)]

        V = tensor_space.coeff_space
        values = [V.zeros() for d in range(mapping.pdim)]
        ranges = [range(s, e+1) for s, e in zip(V.starts, V.ends)]
        grids  = [space.greville for space in tensor_space.spaces]

        # Evaluate analytical mapping at Greville points (tensor-product grid)
        # and store vector values in one separate scalar field for each
        # physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        for index in product(*ranges):
            x = [grid[i] for grid, i in zip(grids, index)]
            u = mapping(*x)
            for d, ud in enumerate(u):
                values[d][index] = ud

        # Compute spline coefficients for each coordinate X_i
        for pvals, field in zip(values, fields):
            tensor_space.compute_interpolant(pvals, field)

        # Create SplineMapping object
        return cls(*fields)

    #--------------------------------------------------------------------------
    # Option [2]: initialize from TensorFemSpace and spline control points
    #--------------------------------------------------------------------------
    @classmethod
    def from_control_points(cls, tensor_space, control_points):

        assert isinstance(tensor_space, TensorFemSpace)
        assert isinstance(control_points, (np.ndarray, h5py.Dataset))

        assert control_points.ndim       == tensor_space.ldim + 1
        assert control_points.shape[:-1] == tuple(V.nbasis for V in tensor_space.spaces)
        assert control_points.shape[ -1] >= tensor_space.ldim

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        fields = [FemField(tensor_space) for d in range(control_points.shape[-1])]

        # Get spline coefficients for each coordinate X_i
        starts = tensor_space.coeff_space.starts
        ends   = tensor_space.coeff_space.ends

        idx_to = tuple(slice(s, e+1) for s, e in zip(starts, ends))
        for i,field in enumerate(fields):
            idx_from = (*idx_to, i)
            field.coeffs[idx_to] = control_points[idx_from]
            field.coeffs.update_ghost_regions()

        # Create SplineMapping object
        return cls(*fields)

    #--------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    def __call__(self, *eta):
        return [map_Xd(*eta) for map_Xd in self._fields]

    # ...
    def jacobian(self, *eta):
        return np.array([map_Xd.gradient(*eta) for map_Xd in self._fields])

    # ...
    def jacobian_inv(self, *eta):
        return np.linalg.inv(self.jacobian(*eta))

    # ...
    def metric(self, *eta):
        J = self.jacobian(*eta)
        return np.dot(J.T, J)

    # ...
    def metric_det(self, *eta):
        return np.linalg.det(self.metric(*eta))

    @property
    def ldim(self):
        return self._ldim

    @property
    def pdim(self):
        return self._pdim

    #--------------------------------------------------------------------------
    # Fast evaluation on a grid
    #--------------------------------------------------------------------------
    def build_mesh(self, grid, npts_per_cell=None, overlap=0):
        """Evaluation of the mapping on the given grid.

        Parameters
        ----------
        grid : List of ndarray
            Grid on which to evaluate the fields.
            Each array in this list corresponds to one logical coordinate.

        npts_per_cell: int, tuple of int or None, optional
            Number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.
        
        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        mesh: tuple
            ldim ldim-D arrays. One for each component.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """

        mesh = self.space.eval_fields(grid, *self._fields, npts_per_cell=npts_per_cell, overlap=overlap)
        return mesh

    # ...
    def jac_mat_grid(self, grid, npts_per_cell=None, overlap=0):
        """Evaluates the Jacobian matrix of the mapping at the given location(s) grid.

        Parameters
        ----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        array_like
            Jacobian matrix at the location(s) grid.

        See Also
        --------
        mapping.SplineMapping.inv_jac_mat_grid : Evaluates the inverse
            of the Jacobian matrix of the mapping at the given location(s) grid.
        mapping.SplineMapping.metric_det_grid : Evaluates the metric determinant
            of the mapping at the given location(s) grid.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # --------------------------
        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return self.jac_mat(*grid)

        # Case 2. 1D array of coordinates and no npts_per_cell is given
        # -> grid is tensor-product, but npts_per_cell is not the same in each cell
        elif grid[0].ndim == 1 and npts_per_cell is None:
            jac_mats = self.jac_mat_irregular_tensor_grid(grid, overlap=overlap)
            return jac_mats

        # Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer
        # -> grid is tensor-product, and each cell has the same number of evaluation points
        elif grid[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * self.ldim
            for i in range(self.ldim):
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], (ncells_i, npts_per_cell[i]))
            jac_mats = self.jac_mat_regular_tensor_grid(grid, overlap=overlap)
            return jac_mats

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
    def jac_mat_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian matrix on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the Jacobian matrix at the location corresponding
            to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jacobians_2d, eval_jacobians_3d

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                            + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_3d(*ncells, *degree, *n_eval_points, *global_basis, 
                              *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                              jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_2d(*ncells, *degree, *n_eval_points, *global_basis, 
                              *global_spans, global_arr_x, global_arr_y, jac_mats)

        else:
            raise NotImplementedError("TODO")

        return jac_mats

    # ...
    def jac_mat_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian matrix on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.
            
        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the Jacobian matrix at the location corresponding
            to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jacobians_irregular_2d, eval_jacobians_irregular_3d

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        jac_mats = np.zeros(tuple(local_shape) + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_irregular_3d(*npts, *degree, *cell_indexes, *global_basis, 
                                        *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                        jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_irregular_2d(*npts, *degree, *cell_indexes, *global_basis, 
                                        *global_spans, global_arr_x, global_arr_y, jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return jac_mats

    # ...
    def inv_jac_mat_grid(self, grid, npts_per_cell=None, overlap=0):
        """Evaluates the inverse of the Jacobian matrix of the mapping at the given location(s) grid.

        Parameters
        ----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        array_like
            Inverse of the Jacobian matrix at the location(s) grid.

        See Also
        --------
        mapping.SplineMapping.jac_mat_grid : Evaluates the Jacobian matrix
            of the mapping at the given location(s) `grid`.
        mapping.SplineMapping.metric_det_grid : Evaluates the metric determinant
            of the mapping at the given location(s) `grid`.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # --------------------------
        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return np.linalg.inv(self.jac_mat(*grid))

        # Case 2. 1D array of coordinates and no npts_per_cell is given
        # -> grid is tensor-product, but npts_per_cell is not the same in each cell
        elif grid[0].ndim == 1 and npts_per_cell is None:
            inv_jac_mats = self.inv_jac_mat_irregular_tensor_grid(grid, overlap=overlap)
            return inv_jac_mats

        # Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer
        # -> grid is tensor-product, and each cell has the same number of evaluation points
        elif grid[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * self.ldim
            for i in range(self.ldim):
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], (ncells_i, npts_per_cell[i]))
            inv_jac_mats = self.inv_jac_mat_regular_tensor_grid(grid, overlap=overlap)
            return inv_jac_mats

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
    def inv_jac_mat_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the inverse of the Jacobian matrix on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        inv_jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the Jacobian matrix
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jacobians_inv_2d, eval_jacobians_inv_3d

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        inv_jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                                + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_3d(*ncells, *degree, *n_eval_points, *global_basis, 
                                  *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                  inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_2d(*ncells, *degree, *n_eval_points, *global_basis, 
                                  *global_spans, global_arr_x, global_arr_y, inv_jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return inv_jac_mats

    # ...
    def inv_jac_mat_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the inverse of the Jacobian matrix on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        inv_jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the Jacobian matrix
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import (eval_jacobians_inv_irregular_2d,
                                                          eval_jacobians_inv_irregular_3d)

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        inv_jac_mats = np.zeros(tuple(local_shape) + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_irregular_3d(*npts, *degree, *cell_indexes, *global_basis, 
                                            *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                            inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_irregular_2d(*npts, *degree, *cell_indexes, *global_basis, 
                                            *global_spans, global_arr_x, global_arr_y, inv_jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return inv_jac_mats

    # ...
    def jac_det_grid(self, grid, npts_per_cell=None, overlap=0):
        """Evaluates the Jacobian determinant of the mapping at the given location(s) grid.

        Parameters
        ----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        array_like
            Jacobian determinant at the location(s) grid.

        See Also
        --------
        mapping.SplineMapping.jac_mat_grid : Evaluates the Jacobian matrix
            of the mapping at the given location(s) grid.
        mapping.SplineMapping.inv_jac_mat_grid : Evaluates the inverse
            of the Jacobian matrix of the mapping at the given location(s) grid.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # --------------------------
        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return self.metric(*grid) ** 0.5

        # Case 2. 1D array of coordinates and no npts_per_cell is given
        # -> grid is tensor-product, but npts_per_cell is not the same in each cell
        elif grid[0].ndim == 1 and npts_per_cell is None:
            jac_dets = self.jac_det_irregular_tensor_grid(grid, overlap=overlap)
            return jac_dets

        # Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer
        # -> grid is tensor-product, and each cell has the same number of evaluation points
        elif grid[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * self.ldim
            for i in range(self.ldim):
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], (ncells_i, npts_per_cell[i]))
            jac_dets = self.jac_det_regular_tensor_grid(grid, overlap=overlap)
            return jac_dets

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
    def jac_det_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian determinant on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_dets : ndarray
            ``self.ldim`` D array of shape ``(n_x_1, ..., n_x_ldim)``.
            ``jac_dets[x_1, ..., x_ldim]`` is the Jacobian determinant
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jac_det_3d, eval_jac_det_2d

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, 
                                                                                                    overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        jac_dets = np.zeros(shape=tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim)), dtype=self._fields[0].coeffs.dtype)

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jac_det_3d(*ncells, *degree, *n_eval_points, *global_basis, 
                            *global_spans, global_arr_x, global_arr_y, global_arr_z, jac_dets)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jac_det_2d(*ncells, *degree, *n_eval_points, *global_basis, 
                            *global_spans, global_arr_x, global_arr_y, jac_dets)

        else:
            raise NotImplementedError("TODO")

        return jac_dets

    # ...
    def jac_det_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian determinant on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.
            
        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_dets : ndarray
            ``self.ldim`` D array of shape ``(n_x_1, ..., n_x_ldim)``.
            ``jac_dets[x_1, ..., x_ldim]`` is the Jacobian determinant
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jac_det_irregular_3d, eval_jac_det_irregular_2d

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        jac_dets = np.zeros(local_shape)

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jac_det_irregular_3d(*npts, *degree, *cell_indexes, *global_basis, 
                                      *global_spans, global_arr_x, global_arr_y, global_arr_z, jac_dets)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jac_det_irregular_2d(*npts, *degree, *cell_indexes, *global_basis, 
                                      *global_spans, global_arr_x, global_arr_y, jac_dets)

        else:
            raise NotImplementedError("TODO")

        return jac_dets

    #--------------------------------------------------------------------------
    # Other properties/methods
    #--------------------------------------------------------------------------
    def jacobian_det(self, *eta):
        return np.linalg.det(self.jac_mat(*eta))

    @property
    def space(self):
        return self._space

    @property
    def fields(self):
        return self._fields

    @property
    def control_points(self):
        return self._control_points

    # TODO: move to 'Geometry' class in 'psydac.cad.geometry' module
    def export(self, filename):
        """
        Export tensor-product spline space and mapping to geometry file in HDF5
        format (single-patch only).

        Parameters
        ----------
        filename : str
          Name of HDF5 output file.

        """
        space = self.space
        comm  = space.coeff_space.cart.comm

        # Create dictionary with geometry metadata
        yml = {}
        yml['ldim'] = self.ldim
        yml['pdim'] = self.pdim
        yml['patches'] = [{ [('name' , 'patch_{}'.format( 0 ) ),
                                        ('type' , 'cad_nurbs'            ),
                                        ('color', 'None'                 )] }]
        yml['internal_faces'] = []
        yml['external_faces'] = [[0,i] for i in range( 2*self.ldim )]
        yml['connectivity'  ] = []

        # Dump geometry metadata to string in YAML file format
        geo = yaml.dump(data = yml, sort_keys = False)

        # Create HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = dict( driver='mpio', comm=comm ) if comm.size > 1 else {}
        h5 = h5py.File( filename, mode='w', **kwargs )

        # Write geometry metadata as fixed-length array of ASCII characters
        h5['geometry.yml'] = np.array( geo, dtype='S' )

        # Create group for patch 0
        group = h5.create_group( yml['patches'][0]['name'] )
        group.attrs['shape'      ] = space.coeff_space.npts
        group.attrs['degree'     ] = space.degree
        group.attrs['periodic'   ] = space.periodic
        for d in range( self.pdim ):
            group['knots_{}'.format( d )] = space.spaces[d].knots

        # Collective: create dataset for control points
        shape = [n for n in space.coeff_space.npts] + [self.pdim]
        dtype = space.coeff_space.dtype
        dset  = group.create_dataset( 'points', shape=shape, dtype=dtype )

        # Independent: write control points to dataset
        starts = space.coeff_space.starts
        ends   = space.coeff_space.ends
        index  = [slice(s, e+1) for s, e in zip(starts, ends)] + [slice(None)]
        index  = tuple( index )
        dset[index] = self.control_points[index]

        # Close HDF5 file
        h5.close()

    #==========================================================================
    class ControlPoints:
        """ Convenience object to access control points.

        """
        # TODO: should not allow access to ghost regions

        def __init__(self, mapping):
            assert isinstance(mapping, SplineMapping)
            self._mapping = mapping

        # ...
        @property
        def mapping(self):
            return self._mapping

        # ...
        def __getitem__(self, key):

            m = self._mapping

            if key is Ellipsis:
                key = tuple(slice(None) for i in range(m.ldim + 1))
            elif isinstance(key, tuple):
                assert len(key) == m.ldim + 1
            else:
                raise ValueError(key)

            pnt_idx = key[:-1]
            dim_idx = key[-1]

            if isinstance(dim_idx, slice):
                dim_idx = range(*dim_idx.indices(m.pdim))
                coeffs = np.array([m.fields[d].coeffs[pnt_idx] for d in dim_idx])
                coords = np.moveaxis(coeffs, 0, -1)
            else:
                coords = np.array(m.fields[dim_idx].coeffs[pnt_idx])

            return coords

#==============================================================================
class NurbsMapping(SplineMapping):

    def __init__(self, *components, name=None):

        weights    = components[-1]
        components = components[:-1]

        SplineMapping.__init__(self, *components, name=name)

        self._weights = NurbsMapping.Weights(self)
        self._weights_field = weights

    #--------------------------------------------------------------------------
    # Option [2]: initialize from TensorFemSpace and spline control points
    #--------------------------------------------------------------------------
    @classmethod
    def from_control_points_weights(cls, tensor_space, control_points, weights):

        assert isinstance(tensor_space, TensorFemSpace)
        assert isinstance(control_points, (np.ndarray, h5py.Dataset))
        assert isinstance(weights, (np.ndarray, h5py.Dataset))

        assert control_points.ndim       == tensor_space.ldim + 1
        assert control_points.shape[:-1] == tuple(V.nbasis for V in tensor_space.spaces)
        assert control_points.shape[ -1] >= tensor_space.ldim
        assert weights.shape == tuple(V.nbasis for V in tensor_space.spaces)

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        fields  = [FemField(tensor_space) for d in range(control_points.shape[-1])]
        fields += [FemField(tensor_space)]

        # Get spline coefficients for each coordinate X_i
        # we store w*x where w is the weight and x is the control point
        starts = tensor_space.coeff_space.starts
        ends   = tensor_space.coeff_space.ends
        idx_to = tuple(slice(s, e+1) for s,e in zip(starts, ends))
        for i, field in enumerate(fields[:-1]):
            idx_from = (*idx_to, i)
#            idw_from = tuple(idx_to)
            field.coeffs[idx_to] = control_points[idx_from] #* weights[idw_from]

        # weights
        idx_from = tuple(idx_to)
        fields[-1].coeffs[idx_to] = weights[idx_from]

        # Create SplineMapping object
        return cls(*fields)

    #--------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    def __call__(self, *eta):
        map_W = self._weights_field
        w = map_W(*eta)
        Xd = [map_Xd(*eta , weights=map_W.coeffs) for map_Xd in self._fields]
        return np.asarray(Xd) / w

    # ...
    def jacobian(self, *eta):
        map_W = self._weights_field
        w = map_W(*eta)
        grad_w = np.array(map_W.gradient(*eta))
        v = np.array([map_Xd(*eta, weights=map_W.coeffs)  for map_Xd in self._fields])
        grad_v = np.array([map_Xd.gradient(*eta, weights=map_W.coeffs) for map_Xd in self._fields])
        return grad_v / w - v[:, None] @ grad_w[None, :] / w**2

    #--------------------------------------------------------------------------
    # Fast evaluation on a grid
    #--------------------------------------------------------------------------
    def build_mesh(self, grid, npts_per_cell=None, overlap=0):
        """Evaluation of the mapping on the given grid.

        Parameters
        ----------
        grid : List of ndarray
            Each array in the list should correspond to a logical coordinate.

        npts_per_cell : int, tuple of int or None, optional

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        mesh: tuple
            ldim ldim-D arrays. One for each component.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """
        mesh = self.space.eval_fields(grid, *self._fields, npts_per_cell=npts_per_cell, weights=self._weights_field, overlap=overlap)
        return mesh

    # ...
    def jac_mat_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian matrix on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the Jacobian matrix at the location corresponding
            to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jacobians_2d_weights, eval_jacobians_3d_weights

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                            + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_3d_weights(ncells[0], ncells[1], ncells[2], degree[0], degree[1],
                                      degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2], global_basis[0],
                                      global_basis[1], global_basis[2], global_spans[0], global_spans[1],
                                      global_spans[2], global_arr_x, global_arr_y, global_arr_z, global_arr_weights,
                                      jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_2d_weights(ncells[0], ncells[1], degree[0], degree[1], n_eval_points[0],
                                      n_eval_points[1], global_basis[0], global_basis[1], global_spans[0],
                                      global_spans[1], global_arr_x, global_arr_y, global_arr_weights, jac_mats)

        else:
            raise NotImplementedError("1D case not Implemented")

        return jac_mats

    # ...
    def jac_mat_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian matrix on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the Jacobian matrix at the location corresponding
            to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import (eval_jacobians_irregular_2d_weights,
                                                          eval_jacobians_irregular_3d_weights)

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        jac_mats = np.zeros(tuple(local_shape) + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_irregular_3d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                                *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                                global_arr_weights, jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_irregular_2d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                                *global_spans, global_arr_x, global_arr_y, 
                                                global_arr_weights, jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return jac_mats

    # ...
    def inv_jac_mat_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the inverse of the Jacobian matrix on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        inv_jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the Jacobian matrix a
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """

        from psydac.core.field_evaluation_kernels import eval_jacobians_inv_2d_weights, eval_jacobians_inv_3d_weights

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        inv_jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                                + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_3d_weights(ncells[0], ncells[1], ncells[2], degree[0],
                                          degree[1], degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2],
                                          global_basis[0], global_basis[1], global_basis[2], global_spans[0],
                                          global_spans[1], global_spans[2], global_arr_x, global_arr_y, global_arr_z,
                                          global_arr_weights, inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_2d_weights(ncells[0], ncells[1], degree[0], degree[1],
                                          n_eval_points[0], n_eval_points[1], global_basis[0], global_basis[1],
                                          global_spans[0], global_spans[1], global_arr_x, global_arr_y,
                                          global_arr_weights, inv_jac_mats)

        else:
            raise NotImplementedError("1D case not Implemented")

        return inv_jac_mats

    # ...
    def inv_jac_mat_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the inverse of the Jacobian matrix on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        inv_jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the Jacobian matrix
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import (eval_jacobians_inv_irregular_2d_weights,
                                                          eval_jacobians_inv_irregular_3d_weights)

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        inv_jac_mats = np.zeros(tuple(local_shape) + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_irregular_3d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                                    *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                                    global_arr_weights, inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_irregular_2d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                                    *global_spans, global_arr_x, global_arr_y, 
                                                    global_arr_weights, inv_jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return inv_jac_mats

    # ...
    def jac_det_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian determinant on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_dets : ndarray
            ``self.ldim`` D array of shape ``(n_x_1, ..., n_x_ldim)``.
            ``jac_dets[x_1, ..., x_ldim]`` is the Jacobian determinant
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jac_det_3d_weights, eval_jac_det_2d_weights
        
        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        jac_dets = np.zeros(shape=tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim)))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jac_det_3d_weights(ncells[0], ncells[1], ncells[2], degree[0], degree[1],
                                    degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2], global_basis[0],
                                    global_basis[1], global_basis[2], global_spans[0], global_spans[1],
                                    global_spans[2], global_arr_x, global_arr_y, global_arr_z, global_arr_weights,
                                    jac_dets)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jac_det_2d_weights(ncells[0], ncells[1], degree[0], degree[1], n_eval_points[0],
                                    n_eval_points[1], global_basis[0], global_basis[1], global_spans[0],
                                    global_spans[1], global_arr_x, global_arr_y, global_arr_weights, jac_dets)

        else:
            raise NotImplementedError("1D case not Implemented")

        return jac_dets

    # ...
    def jac_det_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian determinant on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_dets : ndarray
            ``self.ldim`` D array of shape ``(n_x_1, ..., n_x_ldim)``.
            ``jac_dets[x_1, ..., x_ldim]`` is the Jacobian determinant
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import (eval_jac_det_irregular_3d_weights,
                                                          eval_jac_det_irregular_2d_weights)

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        jac_dets = np.zeros(local_shape)

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jac_det_irregular_3d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                              *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                              global_arr_weights, jac_dets)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jac_det_irregular_2d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                              *global_spans, global_arr_x, global_arr_y, 
                                              global_arr_weights, jac_dets)

        else:
            raise NotImplementedError("TODO")

        return jac_dets

    #--------------------------------------------------------------------------
    # Other properties/methods
    #--------------------------------------------------------------------------
    @property
    def weights_field( self ):
        return self._weights_field

    @property
    def weights( self ):
        return self._weights

    # TODO: move to 'Geometry' class in 'psydac.cad.geometry' module
    def export( self, filename ):
        """
        Export tensor-product spline space and mapping to geometry file in HDF5
        format (single-patch only).

        Parameters
        ----------
        filename : str
          Name of HDF5 output file.

        """
        raise NotImplementedError('')

    #==========================================================================
    class Weights:
        """ Convenience object to access weights.

        """
        # TODO: should not allow access to ghost regions

        def __init__( self, mapping ):
            assert isinstance( mapping, NurbsMapping )
            self._mapping = mapping

        # ...
        @property
        def mapping( self ):
            return self._mapping

        # ...
        def __getitem__( self, key ):

            m = self._mapping

            if key is Ellipsis:
                key = tuple( slice( None ) for i in range( m.ldim ) )
            elif isinstance( key, tuple ):
                assert len( key ) == m.ldim
            else:
                raise ValueError( key )

            pnt_idx = key[:]

            return np.array( m._weights_field.coeffs[pnt_idx] )
