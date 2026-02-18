#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import string
import random
import numpy as np

from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.mapping.discrete import SplineMapping, NurbsMapping
from psydac.cad.utils        import plot_mapping

#==============================================================================
def quart_circle( rmin=0.5, rmax=1.0, center=None ):

    degrees = (2, 1)
    pdim    = 2

    knots = [[0.0 , 0.0 , 0.0 , 1.0 , 1.0 , 1.0],
             [0.0 , 0.0 , 1.0 , 1.0] ]

    points          = np.zeros((3,2,pdim))
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

    for i in range(pdim):
        points[...,i] *= weights[...]

    return degrees, knots, points, weights


#==============================================================================
def annulus(rmin=0.5, rmax=1.0, center=None):

    degrees = (1, 2)
    pdim    = 2

    knots = [[0.0 , 0.0 , 1.0 , 1.0],
             [0.0 , 0.0 , 0.0 , 0.25 , 0.25 , 0.5 , 0.5 , 0.75 , 0.75 , 1.0 , 1.0 , 1.0] ]

    points          = np.zeros((2,9,pdim))
    j = 0
    points[j,0,:]   = np.asarray([0.0   , -rmin])
    points[j,1,:]   = np.asarray([-rmin , -rmin])
    points[j,2,:]   = np.asarray([-rmin , 0.0  ])
    points[j,3,:]   = np.asarray([-rmin , rmin ])
    points[j,4,:]   = np.asarray([0.0   , rmin ])
    points[j,5,:]   = np.asarray([rmin  , rmin ])
    points[j,6,:]   = np.asarray([rmin  , 0.0  ])
    points[j,7,:]   = np.asarray([rmin  , -rmin])
    points[j,8,:]   = np.asarray([0.0   , -rmin])
    j = 1
    points[j,0,:]   = np.asarray([0.0   , -rmax])
    points[j,1,:]   = np.asarray([-rmax , -rmax])
    points[j,2,:]   = np.asarray([-rmax , 0.0  ])
    points[j,3,:]   = np.asarray([-rmax , rmax ])
    points[j,4,:]   = np.asarray([0.0   , rmax ])
    points[j,5,:]   = np.asarray([rmax  , rmax ])
    points[j,6,:]   = np.asarray([rmax  , 0.0  ])
    points[j,7,:]   = np.asarray([rmax  , -rmax])
    points[j,8,:]   = np.asarray([0.0   , -rmax])

    if center is not None:
        points[...,0] += center[0]
        points[...,1] += center[1]

    weights         = np.zeros((2,9))
    j = 0
    weights[j,0]   = 1.0
    weights[j,1]   = 0.707106781187
    weights[j,2]   = 1.0
    weights[j,3]   = 0.707106781187
    weights[j,4]   = 1.0
    weights[j,5]   = 0.707106781187
    weights[j,6]   = 1.0
    weights[j,7]   = 0.707106781187
    weights[j,8]   = 1.0
    j = 1
    weights[j,0]   = 1.0
    weights[j,1]   = 0.707106781187
    weights[j,2]   = 1.0
    weights[j,3]   = 0.707106781187
    weights[j,4]   = 1.0
    weights[j,5]   = 0.707106781187
    weights[j,6]   = 1.0
    weights[j,7]   = 0.707106781187
    weights[j,8]   = 1.0

    for i in range(pdim):
        points[...,i] *= weights[...]

    return degrees, knots, points, weights


#==============================================================================
def circle(radius=1.0, center=None):
    degrees = (2, 2)
    pdim    = 2

    s = 1./np.sqrt(2)
    knots = [[0.0 , 0.0 , 0.0 , 1.0 , 1.0 , 1.0],
             [0.0 , 0.0 , 0.0 , 1.0 , 1.0 , 1.0] ]

    points          = np.zeros((3,3,pdim))
    points[0,0,:]   = np.asarray([-s   , -s  ])
    points[1,0,:]   = np.asarray([-2*s , 0.  ])
    points[2,0,:]   = np.asarray([-s   , s   ])
    points[0,1,:]   = np.asarray([0.   , -2*s])
    points[1,1,:]   = np.asarray([0.   , 0.0 ])
    points[2,1,:]   = np.asarray([0.   , 2*s ])
    points[0,2,:]   = np.asarray([s    , -s  ])
    points[1,2,:]   = np.asarray([2*s  , 0.  ])
    points[2,2,:]   = np.asarray([s    , s   ])
    points         *= radius

    if center is not None:
        points[...,0] += center[0]
        points[...,1] += center[1]

    weights         = np.zeros((3,3))
    weights[0,0]    = 1.
    weights[1,0]    = s
    weights[2,0]    = 1.
    weights[0,1]    = s
    weights[1,1]    = 1.
    weights[2,1]    = s
    weights[0,2]    = 1.
    weights[1,2]    = s
    weights[2,2]    = 1.

    for i in range(pdim):
        points[...,i] *= weights[...]

    return degrees, knots, points, weights

#==============================================================================
if __name__ == '__main__':

#    degrees, knots, points, weights = quart_circle( rmin=0.5, rmax=1.0, center=None )
#    degrees, knots, points, weights = annulus( rmin=0.5, rmax=1.0, center=None )
    degrees, knots, points, weights = circle( radius=1., center=None )

    # Create tensor spline space, distributed
    spaces = [SplineSpace( knots=k, degree=p ) for k,p in zip(knots, degrees)]
    space = TensorFemSpace( *spaces, comm=None )

    mapping = NurbsMapping.from_control_points_weights( space, points, weights )
#    plot_mapping(mapping, N=100)

    from psydac.cad.cad import elevate, refine
    mapping = elevate( mapping, axis=0, times=1 )
    mapping = refine( mapping, axis=0, values=[0.3, 0.6, 0.8] )
    mapping = refine( mapping, axis=1, values=[0.3, 0.6, 0.8] )
    plot_mapping(mapping, N=10)

