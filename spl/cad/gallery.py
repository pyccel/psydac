# coding: utf-8
#
import numpy as np
import string
import random

from spl.fem.splines      import SplineSpace
from spl.fem.tensor       import TensorFemSpace
from spl.mapping.discrete import SplineMapping, NurbsMapping
from spl.cad.utils        import plot_mapping

#==============================================================================
def quart_circle( rmin=0.5, rmax=1.0, center=None ):

    degrees = (2, 1)

    knots = [[0.0 , 0.0 , 0.0 , 1.0 , 1.0 , 1.0],
             [0.0 , 0.0 , 1.0 , 1.0] ]

    points          = np.zeros((3,2,2))
    j = 0
    points[0,j,:]   = [0.0   , -rmin]
    points[1,j,:]   = [-rmin , -rmin]
    points[2,j,:]   = [-rmin , 0.0  ]
    j = 1
    points[0,j,:]   = [0.0   , -rmax]
    points[1,j,:]   = [-rmax , -rmax]
    points[2,j,:]   = [-rmax , 0.0  ]

    if center is not None:
        points[...,0] += center[0]
        points[...,1] += center[1]

    weights         = np.zeros((3,2))
    j = 0
    weights[0,j]   = 1.0
    weights[1,j]   = 0.707106781187
    weights[2,j]   = 1.0
    j = 1
    weights[0,j]   = 1.0
    weights[1,j]   = 0.707106781187
    weights[2,j]   = 1.0

    return degrees, knots, points, weights


#==============================================================================
if __name__ == '__main__':

    degrees, knots, points, weights = quart_circle( rmin=0.5, rmax=1.0, center=None )

    # Create tensor spline space, distributed
    spaces = [SplineSpace( knots=k, degree=p ) for k,p in zip(knots, degrees)]
    space = TensorFemSpace( *spaces, comm=None )

    mapping = NurbsMapping.from_control_points_weights( space, points, weights )
#    mapping = SplineMapping.from_control_points( space, points )
    plot_mapping(mapping, N=100)

