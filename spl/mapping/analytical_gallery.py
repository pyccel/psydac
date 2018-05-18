# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from spl.mapping.analytical import AnalyticalMapping

__all__ = ['AffineMapping', 'Annulus', 'Sphere']

#==============================================================================
class AffineMapping( AnalyticalMapping ):
    """ Linear transformation from parametric to physical space.
        This is equivalent to rototranslation plus rescaling plus shearing.
    """
    def __init__( self, x0, jac_mat ):

        x0 = np.asarray( x0 )
        jm = np.asarray( jac_mat )

        # Check input data
        assert x0.ndim == 1
        assert jm.ndim == 2
        assert jm.shape[0] == x0.shape[0]
        assert jm.shape[1] >= x0.shape[0]

        eta     = ['eta{}'.format(j) for j in range(jm.shape[1])]
        mapping = ['x0_{} + '.format(i) +
                   ' + '.join( 'J_{0}_{1}*eta{1}'.format( i,j )
                               for j in range(jm.shape[1]) )
                   for i in range(jm.shape[0]) ]

        from collections import OrderedDict
        params = OrderedDict()

        for i,xi in enumerate( x0 ):
            key = 'x0_{}'.format(i)
            params[key] = xi

        for (i,j),mat_ij in np.ndenumerate( jac_mat ):
            key = 'J_{}_{}'.format( i,j )
            params[key] = mat_ij

        super().__init__( eta, mapping, **params )

#==============================================================================
class Annulus( AnalyticalMapping ):

    def __init__( self, rmin=0.0, rmax=1.0, xc1=0.0, xc2=0.0, t0=0.0 ):

        eta     = ['s','t']
        mapping = ['xc1 + (rmin*(1-s)+rmax*s)*cos(t+t0)',
                   'xc2 + (rmin*(1-s)+rmax*s)*sin(t+t0)']

        params = dict( rmin=rmin, rmax=rmax, xc1=xc1, xc2=xc2, t0=t0 )
        super().__init__( eta, mapping, **params )

#==============================================================================
class Sphere( AnalyticalMapping ):

    def __init__( self, R0=1.0, xc1=0.0, xc2=0.0, xc3=0.0 ):

        eta     = ['s','t','p']
        mapping = ['xc1 + R0*s*sin(t)*cos(p)',
                   'xc2 + R0*s*sin(t)*sin(p)',
                   'xc3 + R0*s*cos(t)'       ]

        params = dict( R0=R0, xc1=xc1, xc2=xc2, xc3=xc3 )
        super().__init__( eta, mapping, **params )
