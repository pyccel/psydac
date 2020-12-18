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

#===============================================================================
def unroll_edges(domain, xgrid):
    """If necessary, "unroll" intervals that cross boundary of periodic domain.
    """

    xA, xB = domain

    assert all(np.diff(xgrid) >= 0)
    assert xA < xB
    assert xA <= xgrid[0]
    assert xgrid[-1] <= xB

    if xgrid[0] == xA and xgrid[-1] == xB:
        return xgrid

    elif xgrid[0] != xA:
        return np.array([xgrid[-1] - (xB-xA), *xgrid])

    elif xgrid[-1] != xB:
        return np.array([*xgrid, xgrid[0] + (xB-xA)])
