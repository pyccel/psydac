# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from itertools import product
import numpy as np
import string
import random

import h5py
import yaml

from sympde.topology.mapping  import Mapping

from psydac.fem.tensor    import TensorFemSpace
from psydac.fem.basic     import FemField

__all__ = ['SplineMapping', 'NurbsMapping']

#==============================================================================
def random_string( n ):
    chars    = string.ascii_uppercase + string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

#==============================================================================
class SplineMapping:

    def __init__( self, *components, name=None ):

        # Sanity checks
        assert len( components ) >= 1
        assert all( isinstance( c, FemField ) for c in components )
        assert all( isinstance( c.space, TensorFemSpace ) for c in components )
        assert all( c.space is components[0].space for c in components )

        # Store spline space and one field for each coordinate X_i
        self._space  = components[0].space
        self._fields = components

        # Store number of logical and physical dimensions
        self._ldim = components[0].space.ldim
        self._pdim = len( components )

        # Create helper object for accessing control points with slicing syntax
        # as if they were stored in a single multi-dimensional array C with
        # indices [i1, ..., i_n, d] where (i1, ..., i_n) are indices of logical
        # coordinates, and d is index of physical component of interest.
        self._control_points = SplineMapping.ControlPoints( self )

        self._name = name

    @property
    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    #--------------------------------------------------------------------------
    # Option [1]: initialize from TensorFemSpace and pre-existing mapping
    #--------------------------------------------------------------------------
    @classmethod
    def from_mapping( cls, tensor_space, mapping ):

        assert isinstance( tensor_space, TensorFemSpace )
        assert isinstance( mapping, Mapping )
        assert tensor_space.ldim == mapping.ldim

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        fields = [FemField( tensor_space ) for d in range( mapping.pdim )]

        V = tensor_space.vector_space
        values = [V.zeros() for d in range( mapping.pdim )]
        ranges = [range(s,e+1) for s,e in zip( V.starts, V.ends )]
        grids  = [space.greville for space in tensor_space.spaces]

        # Evaluate analytical mapping at Greville points (tensor-product grid)
        # and store vector values in one separate scalar field for each
        # physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        callable_mapping = mapping.get_callable_mapping()
        for index in product( *ranges ):
            x = [grid[i] for grid,i in zip( grids, index )]
            u = callable_mapping( *x )
            for d,ud in enumerate( u ):
                values[d][index] = ud

        # Compute spline coefficients for each coordinate X_i
        for pvals, field in zip( values, fields ):
            tensor_space.compute_interpolant( pvals, field )

        # Create SplineMapping object
        return cls( *fields )

    #--------------------------------------------------------------------------
    # Option [2]: initialize from TensorFemSpace and spline control points
    #--------------------------------------------------------------------------
    @classmethod
    def from_control_points( cls, tensor_space, control_points ):

        assert isinstance( tensor_space, TensorFemSpace )
        assert isinstance( control_points, (np.ndarray, h5py.Dataset) )

        assert control_points.ndim       == tensor_space.ldim + 1
        assert control_points.shape[:-1] == tuple( V.nbasis for V in tensor_space.spaces )
        assert control_points.shape[ -1] >= tensor_space.ldim

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        fields = [FemField( tensor_space ) for d in range( control_points.shape[-1] )]

        # Get spline coefficients for each coordinate X_i
        starts = tensor_space.vector_space.starts
        ends   = tensor_space.vector_space.ends
        idx_to = tuple( slice( s, e+1 ) for s,e in zip( starts, ends ) )
        for i,field in enumerate( fields ):
            idx_from = tuple(list(idx_to)+[i])
            field.coeffs[idx_to] = control_points[idx_from]
            field.coeffs.update_ghost_regions()

        # Create SplineMapping object
        return cls( *fields )

    # --------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    def __call__( self, *eta):
        return [map_Xd( *eta) for map_Xd in self._fields]

    def build_mesh(self, grid, npts_per_cell=None):
        """Evaluation of the mapping on the given grid.

        Parameters
        ----------
        grid : List of ndarray
            Grid on which to evaluate the fields.
            Each array in this list corresponds to one logical coordinate.

        npts_per_cell: int, tuple of int or None, optional
            Number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.
        Returns
        -------
        x_mesh: 3D array of floats
            X component of the mesh
        y_mesh: 3D array of floats
            Y component of the mesh
        z_mesh: 3D array of floats
            Z component of the mesh

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """

        mesh = self.space.eval_fields(grid, *self._fields, npts_per_cell=npts_per_cell)
        if self.ldim == 2:
            x_mesh = mesh[0][:, :, None]
            y_mesh = mesh[1][:, :, None]
            z_mesh = np.zeros_like(x_mesh)
        elif self.ldim == 3:
            x_mesh = mesh[0]
            y_mesh = mesh[1]
            z_mesh = mesh[2]
        else:
            raise NotImplementedError("1D case not implemented")

        return x_mesh, y_mesh, z_mesh

    # ...
    def jac_mat( self, *eta):
        return np.array( [map_Xd.gradient( *eta ) for map_Xd in self._fields] )

    # ...
    def metric( self, *eta):
        J = self.jac_mat( *eta )
        return np.dot( J.T, J )

    # ...
    def metric_det( self, *eta):
        return np.linalg.det( self.metric( *eta ) )

    # ...
    def jac_mat_grid(self, grid, npts_per_cell=None):
        """Evaluates the jacobian of the mapping at the given location(s) grid.

        Parameters
        -----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        array_like
            Jacobian at the location(s) grid.

        See Also
        --------
        mapping.SplineMapping.inv_jac_mat_grid : Evaluates the inverse
            of the jacobian of the mapping at the given location(s) grid.
        mapping.SplineMapping.metric_det_grid : Evaluates the determinant
            of the jacobian of the mapping at the given location(s) grid.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return self.jac_mat(*grid)

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
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], newshape=(ncells_i, npts_per_cell[i]))
            jac_mats = self.jac_mat_regular_tensor_grid(grid)
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
    def jac_mat_regular_tensor_grid(self, grid):
        """Evaluates the jacobian on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape (n_x_1, ..., n_x_ldim, ldim, ldim).
            ``jac_mats[x_1, ..., x_ldim]`` is the jacobian at the location corresponding
            to `(x_1, ..., x_ldim)`.
        """

        from psydac.core.kernels import eval_jacobians_2d, eval_jacobians_3d

        ncells = [grid[i].shape[0] for i in range(self.ldim)]
        n_eval_points = [grid[i].shape[-1] for i in range(self.ldim)]

        pads, degree, global_basis, global_spans = self.space.preprocess_regular_tensor_grid(grid, der=1)

        jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                            + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_3d(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0], degree[1],
                              degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2], global_basis[0],
                              global_basis[1], global_basis[2], global_spans[0], global_spans[1], global_spans[2],
                              global_arr_x, global_arr_y, global_arr_z, jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_2d(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1], n_eval_points[0],
                              n_eval_points[1], global_basis[0], global_basis[1], global_spans[0], global_spans[1],
                              global_arr_x, global_arr_y, jac_mats)

        else:
            raise NotImplementedError("TODO")

        return jac_mats

    # ...
    def inv_jac_mat_grid(self, grid, npts_per_cell=None):
        """Evaluates the inverse of the jacobian of the mapping at the given location(s) grid.

        Parameters
        -----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        array_like
            Inverse of the jacobian at the location(s) grid.

        See Also
        --------
        mapping.SplineMapping.jac_mat_grid : Evaluates the jacobian
            of the mapping at the given location(s) `grid`.
        mapping.SplineMapping.metric_det_grid : Evaluates the determinant
            of the jacobian of the mapping at the given location(s) `grid`.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return np.linalg.inv(self.jac_mat(*grid))

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
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], newshape=(ncells_i, npts_per_cell[i]))
            inv_jac_mats = self.inv_jac_mat_regular_tensor_grid(grid)
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
    def inv_jac_mat_regular_tensor_grid(self, grid):
        """Evaluates the inverse of the jacobian on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        Returns
        -------
        inv_jac_mats : ndarray
            ``self.ldim + 2`` D array of shape (n_x_1, ..., n_x_ldim, ldim, ldim).
            ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the jacobian a
            at the location corresponding to `(x_1, ..., x_ldim)`.
        """
        from psydac.core.kernels import eval_jacobians_inv_2d, eval_jacobians_inv_3d

        ncells = [grid[i].shape[0] for i in range(self.ldim)]
        n_eval_points = [grid[i].shape[-1] for i in range(self.ldim)]

        pads, degree, global_basis, global_spans = self.space.preprocess_regular_tensor_grid(grid, der=1)

        inv_jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                                + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_3d(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0], degree[1],
                                  degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2], global_basis[0],
                                  global_basis[1], global_basis[2], global_spans[0], global_spans[1], global_spans[2],
                                  global_arr_x, global_arr_y, global_arr_z, inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_2d(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1], n_eval_points[0],
                                  n_eval_points[1], global_basis[0], global_basis[1], global_spans[0], global_spans[1],
                                  global_arr_x, global_arr_y, inv_jac_mats)

        else:
            raise NotImplementedError("TODO")

        return inv_jac_mats

    # ...
    def metric_det_grid(self, grid, npts_per_cell=None):
        """Evaluates the determinant of the jacobian of the mapping at the given location(s) grid.

        Parameters
        -----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        array_like
            Determinant of the jacobian at the location(s) grid.

        See Also
        --------
        mapping.SplineMapping.jac_mat_grid : Evaluates the jacobian
            of the mapping at the given location(s) grid.
        mapping.SplineMapping.inv_jac_mat_grid : Evaluates the inverse
            of the jacobian of the mapping at the given location(s) grid.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return self.metric(*grid)

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
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], newshape=(ncells_i, npts_per_cell[i]))
            inv_jac_mats = self.metric_det_regular_tensor_grid(grid)
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
    def metric_det_regular_tensor_grid(self, grid):
        """Evaluates the determinant of the jacobian on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        Returns
        -------
        metric_det : ndarray
            ``self.ldim`` D array of shape (n_x_1, ..., n_x_ldim).
            ``jac_mats[x_1, ..., x_ldim]`` is the determinant of the jacobian
            at the location corresponding to `(x_1, ..., x_ldim)`.
        """
        from psydac.core.kernels import eval_det_metric_3d, eval_det_metric_2d

        ncells = [grid[i].shape[0] for i in range(self.ldim)]
        n_eval_points = [grid[i].shape[-1] for i in range(self.ldim)]

        pads, degree, global_basis, global_spans = self.space.preprocess_regular_tensor_grid(grid, der=1)

        metric_det = np.zeros(shape=tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim)))

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_det_metric_3d(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0], degree[1],
                               degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2], global_basis[0],
                               global_basis[1], global_basis[2], global_spans[0], global_spans[1], global_spans[2],
                               global_arr_x, global_arr_y, global_arr_z, metric_det)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_det_metric_2d(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1], n_eval_points[0],
                               n_eval_points[1], global_basis[0], global_basis[1], global_spans[0], global_spans[1],
                               global_arr_x, global_arr_y, metric_det)

        else:
            raise NotImplementedError("TODO")

        return metric_det

    @property
    def ldim( self ):
        return self._ldim

    @property
    def pdim( self ):
        return self._pdim

    #--------------------------------------------------------------------------
    # Other properties/methods
    #--------------------------------------------------------------------------

    @property
    def space( self ):
        return self._space

    @property
    def fields( self ):
        return self._fields

    @property
    def control_points( self ):
        return self._control_points

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
        space = self.space
        comm  = space.vector_space.cart.comm

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
        group.attrs['shape'      ] = space.vector_space.npts
        group.attrs['degree'     ] = space.degree
        group.attrs['periodic'   ] = space.periodic
        for d in range( self.pdim ):
            group['knots_{}'.format( d )] = space.spaces[d].knots

        # Collective: create dataset for control points
        shape = [n for n in space.vector_space.npts] + [self.pdim]
        dtype = space.vector_space.dtype
        dset  = group.create_dataset( 'points', shape=shape, dtype=dtype )

        # Independent: write control points to dataset
        starts = space.vector_space.starts
        ends   = space.vector_space.ends
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

        def __init__( self, mapping ):
            assert isinstance( mapping, SplineMapping )
            self._mapping = mapping

        # ...
        @property
        def mapping( self ):
            return self._mapping

        # ...
        def __getitem__( self, key ):

            m = self._mapping

            if key is Ellipsis:
                key = tuple( slice( None ) for i in range( m.ldim+1 ) )
            elif isinstance( key, tuple ):
                assert len( key ) == m.ldim+1
            else:
                raise ValueError( key )

            pnt_idx = key[:-1]
            dim_idx = key[-1]

            if isinstance( dim_idx, slice ):
                dim_idx = range( *dim_idx.indices( m.pdim ) )
                coeffs = np.array( [m.fields[d].coeffs[pnt_idx] for d in dim_idx] )
                coords = np.moveaxis( coeffs, 0, -1 )
            else:
                coords = np.array( m.fields[dim_idx].coeffs[pnt_idx] )

            return coords

#==============================================================================
class NurbsMapping( SplineMapping ):

    def __init__( self, *components, name=None ):

        weights    = components[-1]
        components = components[:-1]

        SplineMapping.__init__( self, *components, name=name )

        self._weights = NurbsMapping.Weights( self )
        self._weights_field = weights

    #--------------------------------------------------------------------------
    # Option [2]: initialize from TensorFemSpace and spline control points
    #--------------------------------------------------------------------------
    @classmethod
    def from_control_points_weights( cls, tensor_space, control_points, weights ):

        assert isinstance( tensor_space, TensorFemSpace )
        assert isinstance( control_points, (np.ndarray, h5py.Dataset) )
        assert isinstance( weights, (np.ndarray, h5py.Dataset) )

        assert control_points.ndim       == tensor_space.ldim + 1
        assert control_points.shape[:-1] == tuple( V.nbasis for V in tensor_space.spaces )
        assert control_points.shape[ -1] >= tensor_space.ldim
        assert weights.shape == tuple( V.nbasis for V in tensor_space.spaces )

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        fields  = [FemField( tensor_space ) for d in range( control_points.shape[-1] )]
        fields += [FemField( tensor_space )]

        # Get spline coefficients for each coordinate X_i
        # we store w*x where w is the weight and x is the control point
        starts = tensor_space.vector_space.starts
        ends   = tensor_space.vector_space.ends
        idx_to = tuple( slice( s, e+1 ) for s,e in zip( starts, ends ) )
        for i,field in enumerate( fields[:-1] ):
            idx_from = tuple(list(idx_to)+[i])
#            idw_from = tuple(idx_to)
            field.coeffs[idx_to] = control_points[idx_from] #* weights[idw_from]
            field.coeffs.update_ghost_regions()

        # weights
        idx_from = tuple(idx_to)
        fields[-1].coeffs[idx_to] = weights[idx_from]
        fields[-1].coeffs.update_ghost_regions()

        # Create SplineMapping object
        return cls( *fields )

    #--------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    def __call__( self, *eta):
        map_W = self._weights_field
        w = map_W( *eta )
        Xd = [map_Xd( *eta , weights=map_W.coeffs) for map_Xd in self._fields]
        return np.asarray( Xd ) / w

    def build_mesh(self, grid, npts_per_cell=None):
        """Evaluation of the mapping on the given grid.

        Parameters
        ----------
        grid : List of ndarray
            Each array in the list should correspond to a logical coordinate.
        npts_per_cell : int, tuple of int or None, optional

        Returns
        -------
        x_mesh: 3D array of floats
            X component of the mesh
        y_mesh: 3D array of floats
            Y component of the mesh
        z_mesh: 3D array of floats
            Z component of the mesh

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """
        mesh = self.space.eval_fields(grid, *self._fields, npts_per_cell=npts_per_cell, weights=self._weights_field)
        if self.ldim == 2:
            x_mesh = mesh[0][:, :, None]
            y_mesh = mesh[1][:, :, None]
            z_mesh = np.zeros_like(x_mesh)
        elif self.ldim == 3:
            x_mesh = mesh[0]
            y_mesh = mesh[1]
            z_mesh = mesh[2]
        else:
            raise NotImplementedError("1D case not implemented")

        return x_mesh, y_mesh, z_mesh
    
    # ... 
    def jac_mat( self, *eta):
        raise NotImplementedError('TODO')
#        return np.array( [map_Xd.gradient( *eta ) for map_Xd in self._fields] )

    # ...
    def metric( self, *eta):
        raise NotImplementedError('TODO')
#        J = self.jac_mat( *eta )
#        return np.dot( J.T, J )

    # ...
    def metric_det( self, *eta):
        raise NotImplementedError('TODO')
    #   return np.linalg.det( self.metric( *eta ) )

    # ...
    def jac_mat_grid(self, grid, npts_per_cell=None):
        """Evaluates the jacobian of the mapping at the given location(s) grid.

        Parameters
        -----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        array_like
            Jacobian at the location(s) grid.

        See Also
        --------
        mapping.NurbsMapping.inv_jac_mat_grid : Evaluates the inverse
            of the jacobian of the mapping at the given location(s) grid.
        mapping.NurbsMapping.metric_det_grid : Evaluates the determinant
            of the jacobian of the mapping at the given location(s) grid.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return self.jac_mat(*grid)

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
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], newshape=(ncells_i, npts_per_cell[i]))
            jac_mats = self.jac_mat_regular_tensor_grid(grid)
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
    def jac_mat_regular_tensor_grid(self, grid):
        """Evaluates the jacobian on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape (n_x_1, ..., n_x_ldim, ldim, ldim).
            ``jac_mats[x_1, ..., x_ldim]`` is the jacobian at the location corresponding
            to `(x_1, ..., x_ldim)`.
        """
        from psydac.core.kernels import eval_jacobians_2d_weights, eval_jacobians_3d_weights

        ncells = [grid[i].shape[0] for i in range(self.ldim)]
        n_eval_points = [grid[i].shape[-1] for i in range(self.ldim)]

        pads, degree, global_basis, global_spans = self.space.preprocess_regular_tensor_grid(grid, der=1)

        jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                            + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_3d_weights(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0], degree[1],
                                      degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2], global_basis[0],
                                      global_basis[1], global_basis[2], global_spans[0], global_spans[1],
                                      global_spans[2], global_arr_x, global_arr_y, global_arr_z, global_arr_weights,
                                      jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_2d_weights(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1], n_eval_points[0],
                                      n_eval_points[1], global_basis[0], global_basis[1], global_spans[0],
                                      global_spans[1], global_arr_x, global_arr_y, global_arr_weights, jac_mats)

        else:
            raise NotImplementedError("1D case not Implemented")

        return jac_mats

    # ...
    def inv_jac_mat_grid(self, grid, npts_per_cell=None):
        """Evaluates the inverse of the jacobian of the mapping at the given location(s) grid.

        Parameters
        -----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        array_like
            Inverse of the jacobian at the location(s) grid.

        See Also
        --------
        mapping.NurbsMapping.jac_mat_grid : Evaluates the jacobian
            of the mapping at the given location(s) grid.
        mapping.NurbsMapping.metric_det_grid : Evaluates the determinant
            of the jacobian of the mapping at the given location(s) grid.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return np.linalg.inv(self.jac_mat(*grid))

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
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], newshape=(ncells_i, npts_per_cell[i]))
            inv_jac_mats = self.inv_jac_mat_regular_tensor_grid(grid)
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
    def inv_jac_mat_regular_tensor_grid(self, grid):
        """Evaluates the inverse of the jacobian on a regular tensor product grid.

         Parameters
         ----------
         grid : List of ndarray
             List of 2D arrays representing each direction of the grid.
             Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
             number of cells in the domain in the direction xi and nv_xi is the number of
             evaluation points in the same direction.

         Returns
         -------
         inv_jac_mats : ndarray
             ``self.ldim + 2`` D array of shape (n_x_1, ..., n_x_ldim, ldim, ldim).
             ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the jacobian a
             at the location corresponding to `(x_1, ..., x_ldim)`.
         """

        from psydac.core.kernels import eval_jacobians_inv_2d_weights, eval_jacobians_inv_3d_weights

        ncells = [grid[i].shape[0] for i in range(self.ldim)]
        n_eval_points = [grid[i].shape[-1] for i in range(self.ldim)]

        pads, degree, global_basis, global_spans = self.space.preprocess_regular_tensor_grid(grid, der=1)

        inv_jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                                + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_3d_weights(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0],
                                          degree[1], degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2],
                                          global_basis[0], global_basis[1], global_basis[2], global_spans[0],
                                          global_spans[1], global_spans[2], global_arr_x, global_arr_y, global_arr_z,
                                          global_arr_weights, inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_2d_weights(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1],
                                          n_eval_points[0], n_eval_points[1], global_basis[0], global_basis[1],
                                          global_spans[0], global_spans[1], global_arr_x, global_arr_y,
                                          global_arr_weights, inv_jac_mats)

        else:
            raise NotImplementedError("1D case not Implemented")

        return inv_jac_mats

    # ...
    def metric_det_grid(self, grid, npts_per_cell=None):
        """Evaluates the determinant of the jacobian of the mapping at the given location(s) grid.

        Parameters
        -----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        Returns
        -------
        array_like
            Determinant of the jacobian at the location(s) grid.

        See Also
        --------
        mapping.NurbsMapping.jac_mat_grid : Evaluates the jacobian
            of the mapping at the given location(s) grid.
        mapping.NurbsMapping.inv_jac_mat_grid : Evaluates the inverse
            of the jacobian of the mapping at the given location(s) grid.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # Case 1. Scalar coordinates
        if (grid[0].size == 1) or grid[0].ndim == 0:
            return self.metric(*grid)

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
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], newshape=(ncells_i, npts_per_cell[i]))
            inv_jac_mats = self.metric_det_regular_tensor_grid(grid)
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
    def metric_det_regular_tensor_grid(self, grid):
        """Evaluates the determinant of the jacobian on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        Returns
        -------
        metric_det : ndarray
            ``self.ldim`` D array of shape (n_x_1, ..., n_x_ldim).
            ``jac_mats[x_1, ..., x_ldim]`` is the determinant of the jacobian
            at the location corresponding to `(x_1, ..., x_ldim)`.
        """
        from psydac.core.kernels import eval_det_metric_3d_weights, eval_det_metric_2d_weights

        ncells = [grid[i].shape[0] for i in range(self.ldim)]
        n_eval_points = [grid[i].shape[-1] for i in range(self.ldim)]

        pads, degree, global_basis, global_spans = self.space.preprocess_regular_tensor_grid(grid, der=1)

        metric_det = np.zeros(shape=tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim)))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_det_metric_3d_weights(ncells[0], ncells[1], ncells[2], pads[0], pads[1], pads[2], degree[0], degree[1],
                                       degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2], global_basis[0],
                                       global_basis[1], global_basis[2], global_spans[0], global_spans[1],
                                       global_spans[2], global_arr_x, global_arr_y, global_arr_z, global_arr_weights,
                                       metric_det)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_det_metric_2d_weights(ncells[0], ncells[1], pads[0], pads[1], degree[0], degree[1], n_eval_points[0],
                                       n_eval_points[1], global_basis[0], global_basis[1], global_spans[0],
                                       global_spans[1], global_arr_x, global_arr_y, global_arr_weights, metric_det)

        else:
            raise NotImplementedError("1D case not Implemented")

        return metric_det

    #--------------------------------------------------------------------------
    # Other properties/methods
    #--------------------------------------------------------------------------

    @property
    def control_points( self ):
        return self._control_points

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
