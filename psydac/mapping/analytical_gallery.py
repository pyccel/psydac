# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from psydac.mapping.analytical import AnalyticalMapping

__all__ = ['Annulus', 'Sphere', 'Target', 'Czarny', 'TwistedTarget', 'Torus',
           'Collela']

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

#==============================================================================
class Target( AnalyticalMapping ):

    eta_symbols = ['s','t']
    expressions = ['x0 + (1-k)*s*cos(t) - D*s**2',
                   'y0 + (1+k)*s*sin(t)'         ]

    # With k=0 and D=0 Target geometry reduces to a circle
    default_params = dict( x0=0, y0=0, k=0.3, D=0.2 )

#==============================================================================
class Czarny( AnalyticalMapping ):

    eta_symbols = ['s','t']
    expressions = ['(1 - sqrt( 1 + eps*(eps + 2*s*cos(t)) )) / eps',
                   'y0 + (b / sqrt(1-eps**2/4) * s * sin(t)) /'
                        '(2 - sqrt( 1 + eps*(eps + 2*s*cos(t)) ))']

    default_params = dict( y0=0, b=1.4, eps=0.3 )

#==============================================================================
class TwistedTarget( AnalyticalMapping ):

    eta_symbols = ['s','t']
    expressions = ['x0 + (1-k)*s*cos(t) - D*s**2',
                   'y0 + (1+k)*s*sin(t)'         ,
                   'z0 + c*s**2*sin(2*t)' ]

    # With c=0 and z0=0 surface 3D reduces to Target geometry
    # With k=0 and D=0 Target geometry reduces to a circle
    default_params = dict( x0=0, y0=0, z0=0, k=0.3, D=0.2, c=0.5 )

#==============================================================================
class Torus( AnalyticalMapping ):

    eta_symbols = ['t','p']
    expressions = ['(R0+a*cos(t))*cos(p)',
                   '(R0+a*cos(t))*sin(p)',
                       'a*sin(t)' ]

    default_params = dict( R0=5.0, a=1.0 )

#==============================================================================
class Collela( AnalyticalMapping ):

    eta_symbols = ['s','t']
    expressions = ['2.*(s + eps*sin(2.*pi*k1*s)*sin(2.*pi*k2*t)) - 1.',
                   '2.*(t + eps*sin(2.*pi*k1*s)*sin(2.*pi*k2*t)) - 1.']

    default_params = dict( k1=1.0, k2=1.0, eps=0.1 )

#==============================================================================
class Collela3D( AnalyticalMapping ):

    eta_symbols = ['s','t','r']
#    expressions = ['2.*(s + eps*sin(2.*pi*k1*s)*sin(2.*pi*k2*t)*sin(2.*pi*k3*r)) - 1.',
#                   '2.*(t + eps*sin(2.*pi*k1*s)*sin(2.*pi*k2*t)*sin(2.*pi*k3*r)) - 1.',
#                   '2.*(r + eps*sin(2.*pi*k1*s)*sin(2.*pi*k2*t)*sin(2.*pi*k3*r)) - 1.']

    expressions = ['2.*(s + eps*sin(2.*pi*k1*s)*sin(2.*pi*k2*t)) - 1.',
                   '2.*(t + eps*sin(2.*pi*k1*s)*sin(2.*pi*k2*t)) - 1.',
                   '2.*r  - 1.']

#    expressions = ['2.*s  - 1.',
#                   '2.*t  - 1.',
#                   '2.*r  - 1.']

    default_params = dict( k1=1.0, k2=1.0, k3=1.0, eps=0.1 )
