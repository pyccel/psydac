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
    # Option [3]: initialize from caid.cad_geometry
    #--------------------------------------------------------------------------
    @classmethod
    def from_caid( cls, geo ):
        # TODO should be used only in serial runs
        from caid.cad_geometry import cad_geometry
        from spl.fem.splines import SplineSpace
        from spl.fem.tensor  import TensorFemSpace

        assert isinstance( geo, cad_geometry )
        assert len(geo) == 1

        nrb = geo[0]
        dim = len(nrb.degree)

        spaces = [SplineSpace( p, knots=t ) for p,t in zip(nrb.degree, nrb.knots)]
        if dim == 1:
            V = spaces[0]

        else:
            V = TensorFemSpace( *spaces )

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        name   = random_string( 8 )
        fields = [FemField( V, 'mapping_{name}_x{d}'.format( name=name, d=d ) )
                  for d in range( dim )]

        # Get spline coefficients for each coordinate X_i
        idx_to = tuple( slice( 0, e ) for e in nrb.shape )
        for i,field in enumerate( fields ):
            idx_from = tuple(list(idx_to)+[i])
            field.coeffs[idx_to] = nrb.points[idx_from]

        # Create SplineMapping object
        return cls( *fields )

    @property
    def space(self):
        return self._space

    #--------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    def __call__( self, eta ):
        return [map_Xd( *eta ) for map_Xd in self._fields]

    # TODO: jac_mat
    def jac_mat( self, eta ):
        pass

    # TODO: metric
    def metric( self, eta ):
        pass

    # TODO: metric_det
    def metric_det( self, eta ):
        pass

    @property
    def ldim( self ):
        return self._ldim

    @property
    def pdim( self ):
        return self._pdim

