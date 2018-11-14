# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from itertools import product
import numpy as np
import string
import random
import h5py

from spl.mapping.basic import Mapping
from spl.fem.tensor    import TensorFemSpace
from spl.fem.basic     import FemField

__all__ = ['SplineMapping']

#==============================================================================
def random_string( n ):
    chars    = string.ascii_uppercase + string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

#==============================================================================
class SplineMapping( Mapping ):

    def __init__( self, *components ):

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

    #--------------------------------------------------------------------------
    # Option [1]: initialize from TensorFemSpace and pre-existing mapping
    #--------------------------------------------------------------------------
    @classmethod
    def from_mapping( cls, tensor_space, mapping ):

        assert isinstance( tensor_space, TensorFemSpace )
        assert isinstance( mapping, Mapping )
        assert tensor_space.ldim == mapping.ldim

        name   = random_string( 8 )
        fields = [FemField( tensor_space, 'mapping_{name}_x{d}'.format( name=name, d=d ) )
                  for d in range( mapping.pdim )]

        V = tensor_space.vector_space
        values = [V.zeros() for d in range( mapping.pdim )]
        ranges = [range(s,e+1) for s,e in zip( V.starts, V.ends )]
        grids  = [space.greville for space in tensor_space.spaces]

        # Evaluate analytical mapping at Greville points (tensor-product grid)
        # and store vector values in one separate scalar field for each
        # physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        for index in product( *ranges ):
            x = [grid[i] for grid,i in zip( grids, index )]
            u = mapping( x )
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
        name   = random_string( 8 )
        fields = [FemField( tensor_space, 'mapping_{name}_x{d}'.format( name=name, d=d ) )
                  for d in range( control_points.shape[2] )]

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

    #--------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    def __call__( self, eta ):
        return [map_Xd( *eta ) for map_Xd in self._fields]

    def jac_mat( self, eta ):
        return np.array( [map_Xd.gradient( *eta ) for map_Xd in self._fields] )

    def metric( self, eta ):
        J = self.jac_mat( eta )
        return np.dot( J.T, J )

    def metric_det( self, eta ):
        return np.linalg.det( self.metric( eta ) )

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
