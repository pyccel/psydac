# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from itertools import product

from spl.mapping.basic import Mapping
from spl.fem.tensor    import TensorFemSpace
from spl.fem.basic     import FemField

__all__ = ['SplineMapping']

#==============================================================================

class SplineMapping( Mapping ):


    def __init__( self, *args, **kwargs ):
        self._init_from_mapping( **kwargs )


    def _init_from_mapping( self, tensor_space, mapping ):

        assert isinstance( tensor_space, TensorFemSpace )
        assert isinstance( mapping, Mapping )
        assert tensor_space.ldim == mapping.ldim

        fields = [FemField( tensor_space, 'x{}'.format(d) )
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

        # Store spline space and one field for each coordinate X_i
        self._space  = tensor_space
        self._fields = fields

        # Store number of logical and physical dimensions
        self._ldim = mapping.ldim
        self._pdim = mapping.pdim

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

