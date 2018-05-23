# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from spl.mapping.analytical import AnalyticalMapping

__all__ = ['Annulus', 'Sphere']

#==============================================================================
class Annulus( AnalyticalMapping ):

    eta_symbols = ['s','t']
    expressions = ['xc1 + (rmin*(1-s)+rmax*s)*cos(t+t0)',
                   'xc2 + (rmin*(1-s)+rmax*s)*sin(t+t0)']

    default_params = dict( rmin=0.0, rmax=1.0, xc1=0.0, xc2=0.0, t0=0.0 )

#==============================================================================
class Sphere( AnalyticalMapping ):

    eta_symbols = ['s','t','p']
    expressions = ['xc1 + R0*s*sin(t)*cos(p)',
                   'xc2 + R0*s*sin(t)*sin(p)',
                   'xc3 + R0*s*cos(t)'       ]

    default_params = dict( R0=1.0, xc1=0.0, xc2=0.0, xc3=0.0 )
