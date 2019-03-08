# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import numpy as np

#===============================================================================
def refine_array_1d( x, n ):

    xr = []
    for (a,b) in zip(x[:-1],x[1:]):
        xr.extend( np.linspace( a, b, n, endpoint=False ) )
    xr.append( x[-1] )

    return xr
