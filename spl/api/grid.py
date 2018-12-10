# coding: utf-8

import numpy as np

from spl.utilities.quadratures  import gauss_legendre
from spl.core.bsplines          import quadrature_grid
from spl.fem.splines            import SplineSpace
from spl.fem.tensor             import TensorFemSpace
from spl.fem.vector             import ProductFemSpace
from spl.api.boundary_condition import DiscreteBoundary

#==============================================================================
def _compute_quadrature_SplineSpace( V, quad_order=None ):
    """
    returns quadrature points and weights over a grid, given a 1d Spline space
    """
    assert( isinstance( V, SplineSpace ) )

    T    = V.knots       # knots sequence
    p    = V.degree      # spline degree
    n    = V.nbasis      # total number of control points
    grid = V.breaks      # breakpoints
    ne   = V.ncells      # number of cells in domain (ne=len(grid)-1)

    # ...
    if quad_order is None:
        quad_order = p

    elif not( isinstance( quad_order, int )):
        raise TypeError('Expecting int value for quad_order')
    # ...

    u, w = gauss_legendre( quad_order )
    u = u[::-1] # reorder quad points
    w = w[::-1] # reorder quad weights
    points, weights = quadrature_grid( grid, u, w )

    return points, weights

#==============================================================================
def _compute_quadrature_TensorFemSpace( V, quad_order=None ):
    """
    returns quadrature points and weights for a tensor space
    """
    # ...
    if quad_order is None:
        quad_order = [None for d in range(V.ldim)]

    elif isinstance( quad_order, int ):
        quad_order = [quad_order for d in range(V.ldim)]

    elif not isinstance( quad_order, (list, tuple) ):
        raise TypeError('Expecting None, int or list/tuple of int')

    else:
        assert( len(quad_order) == V.ldim )
    # ...

    return [_compute_quadrature_SplineSpace( W, quad_order=order )
            for (W, order) in zip(V.spaces, quad_order)]

#==============================================================================
# TODO must take the max of degrees if quad_order is not present and
# spaces.degrees are different
def _compute_quadrature_ProductFemSpace( V, quad_order=None ):
    """
    returns quadrature points and weights for a product space
    """
    # ...
    if quad_order is None:
        quad_order = [None for d in range(V.ldim)]

    elif isinstance( quad_order, int ):
        quad_order = [quad_order for d in range(V.ldim)]

    elif not isinstance( quad_order, (list, tuple) ):
        raise TypeError('Expecting None, int or list/tuple of int')

    else:
        assert( len(quad_order) == V.ldim )
    # ...

    return [compute_quadrature( W, quad_order=quad_order ) for W in V.spaces]

#==============================================================================
def compute_quadrature( V, quad_order=None ):
    """
    """
    _avail_classes = [SplineSpace, TensorFemSpace, ProductFemSpace]

    classes = type(V).__mro__
    classes = set(classes) & set(_avail_classes)
    classes = list(classes)
    if not classes:
        raise TypeError('> wrong argument type {}'.format(type(V)))

    cls = classes[0]

    pattern = '_compute_quadrature_{name}'
    func = pattern.format( name = cls.__name__ )

    func = eval(func)
    pw = func(V, quad_order=quad_order)
    if isinstance( V, SplineSpace ):
        pw = [pw]

    return pw

#==============================================================================
class QuadratureGrid():
    def __init__( self, V, quad_order=None ):
        pw = compute_quadrature( V, quad_order=quad_order )
        points, weights = zip(*pw)

        points  = list(points)
        weights = list(weights)

        self._points  = points
        self._weights = weights

    @property
    def points(self):
        return self._points

    @property
    def weights(self):
        return self._weights

    @property
    def quad_order(self):
        return [w.shape[1] for w in self.weights]

#==============================================================================
class BoundaryQuadratureGrid(QuadratureGrid):
    def __init__( self, V, axis, ext, quad_order=None ):
        if isinstance(V, ProductFemSpace):
            raise NotImplementedError('')

        pw = compute_quadrature( V, quad_order=quad_order )
        points, weights = zip(*pw)

        points  = list(points)
        weights = list(weights)

        # ...
        if V.ldim == 1:
            assert( isinstance( V, SplineSpace ) )

            bounds = {}
            bounds[-1] = V.domain[0]
            bounds[1]  = V.domain[1]

            points[axis] = np.asarray([[bounds[ext]]])
            weights[axis] = np.asarray([[1.]])

        elif V.ldim in [2, 3]:
            assert( isinstance( V, TensorFemSpace ) )

            bounds = {}
            bounds[-1] = V.spaces[axis].domain[0]
            bounds[1]  = V.spaces[axis].domain[1]

            points[axis] = np.asarray([[bounds[ext]]])
            weights[axis] = np.asarray([[1.]])
        # ...

        self._points  = points
        self._weights = weights


################################################
if __name__ == '__main__':
    from numpy import linspace

    # ...
    p=(2,2)
    ne=(2**1,2**1)

    # ... discrete spaces
    # Input data: degree, number of elements
    p1,p2 = p
    ne1,ne2 = ne

    # Create uniform grid
    grid_1 = linspace( 0., 1., num=ne1+1 )
    grid_2 = linspace( 0., 1., num=ne2+1 )

    # Create 1D finite element spaces and precompute quadrature data
    V1 = SplineSpace( p1, grid=grid_1 )
    V2 = SplineSpace( p2, grid=grid_2 )

    # Create 2D tensor product finite element space
    V = TensorFemSpace( V1, V2 )
    # ...

#    # ...
#    print('=== 1D case ===')
#    qd_1d = QuadratureGrid(V1)
#
#    print('> points  = ', qd_1d.points)
#    print('> weights = ', qd_1d.weights)
#    # ...

    # ...
    print('=== 2D case ===')
    qd_2d = QuadratureGrid(V)

    for x,w in zip(qd_2d.points, qd_2d.weights):
        print('> points  = ', x)
        print('> weights = ', w)
    # ...

    # ...
    print('=== 2D case boundary ===')
    qd_2d = BoundaryQuadratureGrid(V, axis=1, ext=-1)

    for x,w in zip(qd_2d.points, qd_2d.weights):
        print('> points  = ', x)
        print('> weights = ', w)
    # ...
