# -*- coding: utf-8 -*-
import numpy as np

from datatypes import SplineCurve
from datatypes import SplineSurface
from datatypes import SplineVolume
from datatypes import NurbsCurve
from datatypes import NurbsSurface
from datatypes import NurbsVolume

# ...
def make_line(origin=(0.,0.), end=(1.,0.)):
    knots  = [0., 0., 1., 1.]
    degree = 1
    n      = len(knots) - degree - 1

    P = np.zeros((n, 2))
    P[:, 0] = [origin[0], end[0]]
    P[:, 1] = [origin[1], end[1]]

    return SplineCurve(knots=knots, degree=degree, points=P)

def make_arc(center=(0.,0.), radius=1., angle=90.):
    if angle == 90.:
        knots  = [0., 0., 0., 1., 1., 1.]
        degree = 2
        n      = len(knots) - degree - 1

        P = np.zeros((n, 2))
        P[:, 0] = [1., 1., 0.]
        P[:, 1] = [0., 1., 1.]

        # weights
        s2 = 1./np.sqrt(2)
        W = np.zeros(n)
        W[:] = [1., s2, 1.]

    elif angle == 120.:
        knots  = [0., 0., 0., 1., 1., 1.]
        degree = 2
        n      = len(knots) - degree - 1

        P = np.zeros((n, 2))
        a = np.cos(np.pi/6.)
        P[:, 0] = [ a, 0., -a]
        P[:, 1] = [.5, 2., .5]

        # weights
        W = np.zeros(n)
        W[:] = [1., 1./2., 1.]

    elif angle == 180.:
        knots  = [0., 0., 0., 0., 1., 1., 1., 1.]
        degree = 3
        n      = len(knots) - degree - 1

        P = np.zeros((n, 2))
        P[:, 0] = [1., 1., -1., -1.]
        P[:, 1] = [0., 2.,  2.,  0.]

        # weights
        W = np.zeros(n)
        W[:] = [1., 1./3., 1./3., 1.]

    else:
        raise NotImplementedError('TODO, given {}'.format(angle))

    P *= radius
    P[:,0] += center[0]
    P[:,1] += center[1]

    return NurbsCurve(knots=knots, degree=degree, points=P, weights=W)

def make_square(origin=(0,0), length=1.):
    Tu  = [0., 0., 1., 1.]
    Tv  = [0., 0., 1., 1.]
    pu = 1
    pv = 1
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)

    origin = np.asarray(origin)

    P = np.asarray([[[0.,0.],[0.,1.]],[[1.,0.],[1.,1.]]])
    for i in range(0, 2):
        for j in range(0, 2):
            P[i,j,:] = origin + P[i,j,:]*length

    return SplineSurface(knots=(Tu, Tv), degree=(pu, pv), points=P)

def make_circle(center=(0.,0.), radius=1.):
    Tu  = [0., 0., 0., 1, 1., 1.]
    Tv  = [0., 0., 0., 1, 1., 1.]
    pu = 2
    pv = 2
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)


    s = 1./np.sqrt(2)
    P          = np.zeros((nu,nv,2))
    P[0,0,:]   = np.asarray([-s   , -s   ])
    P[1,0,:]   = np.asarray([-2*s , 0.   ])
    P[2,0,:]   = np.asarray([-s   , s    ])
    P[0,1,:]   = np.asarray([0.   , -2*s ])
    P[1,1,:]   = np.asarray([0.   , 0.0  ])
    P[2,1,:]   = np.asarray([0.   , 2*s  ])
    P[0,2,:]   = np.asarray([s    , -s   ])
    P[1,2,:]   = np.asarray([2*s  , 0.   ])
    P[2,2,:]   = np.asarray([s    , s    ])

    P *= radius
    P[:,:,0] += center[0]
    P[:,:,1] += center[1]

    W       = np.zeros((3,3))
    W[0,0]  = 1.
    W[1,0]  = s
    W[2,0]  = 1.
    W[0,1]  = s
    W[1,1]  = 1.
    W[2,1]  = s
    W[0,2]  = 1.
    W[1,2]  = s
    W[2,2]  = 1.

    return NurbsSurface(knots=(Tu, Tv), degree=(pu, pv), points=P, weights=W)
# ...

def make_half_annulus_cubic(center=(0.,0.), rmax=1., rmin=0.5):
    Tu  = [0., 0., 0., 0., 1., 1., 1., 1.]
    Tv  = [0., 0., 1., 1.]

    pu = 3
    pv = 1
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)

    # ctrl points for radius = 1
    x = np.asarray([1., 1., -1., -1.])
    y = np.asarray([0., 2.,  2.,  0.])

    P = np.zeros((nu,nv,2))
    P[:, 0, 0] = rmax * x
    P[:, 0, 1] = rmax * y

    P[:, 1, 0] = rmin * x
    P[:, 1, 1] = rmin * y

    P[:,:,0] += center[0]
    P[:,:,1] += center[1]

    # weights
    W = np.zeros((nu,nv))
    W[:,0] = [1., 1./3., 1./3., 1.]
    W[:,1] = [1., 1./3., 1./3., 1.]

    return NurbsSurface(knots=(Tu, Tv), degree=(pu, pv), points=P, weights=W)

def make_L_shape_C1(center=None):
    Tu  = [0., 0., 0., 0.5, 1., 1., 1.]
    Tv  = [0., 0., 0., 1., 1., 1.]

    pu = 2
    pv = 2
    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    gridu = np.unique(Tu)
    gridv = np.unique(Tv)

    # ctrl points
    P = np.zeros((nu,nv,2))
    P[:,:,0] = np.asarray([[-1., -0.5, 0.], [-1., -0.707106781186548, 0.], [-1., -0.292893218813452, 0.], [1., 1., 1.]])
    P[:,:,1] = np.asarray([[-1., -1., -1.], [ 1.,  0.292893218813452, 0.], [ 1.,  0.707106781186548, 0.], [1., .5, 0.]])

    if not( center is None ):
        P[:,:,0] += center[0]
        P[:,:,1] += center[1]

    return SplineSurface(knots=(Tu, Tv), degree=(pu, pv), points=P)
