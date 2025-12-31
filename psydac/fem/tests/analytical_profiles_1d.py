#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import math
import numpy as np

from psydac.fem.tests.analytical_profiles_base import AnalyticalProfile
from psydac.fem.tests.utilities                import horner, falling_factorial

__all__ = ('AnalyticalProfile1D_Cos', 'AnalyticalProfile1D_Poly')

#===============================================================================
class AnalyticalProfile1D_Cos( AnalyticalProfile ):

    def __init__( self, n=1, c=0.0 ):
        twopi     = 2.0*math.pi
        self._k   = twopi * n
        self._phi = twopi * c

    @property
    def ndims( self ):
        return 1

    @property
    def domain( self ):
        return (0.0, 1.0)

    @property
    def poly_order( self ):
        return -1

    def eval( self, x, diff=0 ):
        return self._k**diff * np.cos( 0.5*math.pi*diff + self._k*x + self._phi )

    def max_norm( self, diff=0 ):
        return self._k**diff

#===============================================================================
class AnalyticalProfile1D_Sin( AnalyticalProfile ):

    def __init__( self, n=1, c=0.0 ):
        twopi     = 2.0*math.pi
        self._k   = twopi * n
        self._phi = twopi * c

    @property
    def ndims( self ):
        return 1

    @property
    def domain( self ):
        return (0.0, 1.0)

    @property
    def poly_order( self ):
        return -1

    def eval( self, x, diff=0 ):
        return self._k**diff * np.sin( 0.5*math.pi*diff + self._k*x + self._phi )

    def max_norm( self, diff=0 ):
        return self._k**diff
#===============================================================================
class AnalyticalProfile1D_Poly( AnalyticalProfile ):

    def __init__( self, deg ):

        coeffs = np.random.random_sample( 1+deg )  # 0 <= c < 1
        coeffs = 1.0 - coeffs                      # 0 < c <= 1

        self._deg    = deg
        self._coeffs = coeffs

    @property
    def ndims( self ):
        return 1

    @property
    def domain( self ):
        return (-1.0, 1.0)

    @property
    def poly_order( self ):
        return self_deg

    def eval( self, x, diff=0 ):
        d = diff
        coeffs = [c * falling_factorial( i+d, d ) \
                  for i,c in enumerate( self._coeffs[d:] )]
        return horner( x, *coeffs )

    def max_norm( self, diff=0 ):
        xmin, xmax = self.domain

        if xmax < abs(xmin):
            raise NotImplementedError( "General formula not implemented" )

        # For xmax >= |xmin|:
        # max(|f^(d)(x)|) = f^(d)(xmax)
        return self.eval( xmax, diff )
