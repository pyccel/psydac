# coding: utf-8

import numpy as np

from spl.utilities.quadratures  import gauss_legendre
from spl.core.bsplines          import quadrature_grid
from spl.fem.splines            import SplineSpace
from spl.fem.tensor             import TensorFemSpace
from spl.fem.vector             import ProductFemSpace
from spl.fem.grid import FemAssemblyGrid

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

    return compute_quadrature( V.spaces[0], quad_order=quad_order )

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

        quad_grid = create_fem_assembly_grid( V, quad_order=quad_order )

        self._fem_grid            = quad_grid
        self._n_elements          = [g.num_elements        for g in quad_grid]
        self._local_element_start = [g.local_element_start for g in quad_grid]
        self._local_element_end   = [g.local_element_end   for g in quad_grid]
        self._points              = [g.points              for g in quad_grid]
        self._weights             = [g.weights             for g in quad_grid]


    @property
    def fem_grid(self):
        return self._fem_grid

    @property
    def n_elements(self):
        return self._n_elements

    @property
    def local_element_start( self ):
        """ Local index of first element owned by process.
        """
        return self._local_element_start

    # ...
    @property
    def local_element_end( self ):
        """ Local index of last element owned by process.
        """
        return self._local_element_end

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
        assert( not( isinstance(V, ProductFemSpace) ) )

        QuadratureGrid.__init__( self, V, quad_order=quad_order )

        points     = self.points
        weights    = self.weights

        # ...
        if V.ldim == 1:
            assert( isinstance( V, SplineSpace ) )

            bounds = {}
            bounds[-1] = V.domain[0]
            bounds[1]  = V.domain[1]

            points[axis]     = np.asarray([[bounds[ext]]])
            weights[axis]    = np.asarray([[1.]])

        elif V.ldim in [2, 3]:
            assert( isinstance( V, TensorFemSpace ) )

            bounds = {}
            bounds[-1] = V.spaces[axis].domain[0]
            bounds[1]  = V.spaces[axis].domain[1]

            points[axis]     = np.asarray([[bounds[ext]]])
            weights[axis]    = np.asarray([[1.]])
        # ...

        self._points     = points
        self._weights    = weights


#==============================================================================
def create_fem_assembly_grid(V, quad_order=None, nderiv=1):
    # TODO we assume all spaces are the same for the moment
    if isinstance(V, ProductFemSpace):
        return create_fem_assembly_grid(V.spaces[0])

    # ...
    if not( quad_order is None ):
        if isinstance( quad_order, int ):
            quad_order = [quad_order for i in range(V.ldim)]

        elif not isinstance( quad_order, (list, tuple) ):
            raise TypeError('Expecting a tuple/list or int')

    else:
        quad_order = [None for i in range(V.ldim)]

    # ...

    # ...
    if not( nderiv is None ):
        if isinstance( nderiv, int ):
            nderiv = [nderiv for i in range(V.ldim)]

        elif not isinstance( nderiv, (list, tuple) ):
            raise TypeError('Expecting a tuple/list or int')

    else:
        nderiv = [1 for i in range(V.ldim)]
    # ...

    if V.ldim == 1:
        grid = FemAssemblyGrid( V, V.vector_space.starts[0], V.vector_space.ends[0],
                                quad_order=quad_order[0], nderiv=nderiv[0] )
        return [ grid ]

    elif V.ldim > 1:

        return [FemAssemblyGrid(W, s, e, quad_order=n, nderiv=d )
                for W,s,e,n,d in zip( V.spaces,
                                      V.vector_space.starts, V.vector_space.ends,
                                      quad_order, nderiv ) ]

    else:
        raise ValueError('Expecting dimension 1, 2 or 3')


#==============================================================================
class BasisValues():
    def __init__( self, V, grid, nderiv ):
        assert( isinstance( grid, QuadratureGrid ) )

        # TODO quad_order in FemAssemblyGrid must be be the order and not the
        # degree
        quad_order = [q-1 for q in grid.quad_order]
        quad_grid = create_fem_assembly_grid( V,
                                              quad_order=quad_order,
                                              nderiv=nderiv )

        self._spans = [g.spans for g in quad_grid]
        self._basis = [g.basis for g in quad_grid]

    @property
    def basis(self):
        return self._basis

    @property
    def spans(self):
        return self._spans

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

    qd_1d = QuadratureGrid(V1)
    qd_2d = QuadratureGrid(V)

    values_1d = BasisValues(V1, qd_1d, nderiv=1)
    print('> basis  = ', values_1d.basis)

    values_2d = BasisValues(V, qd_2d, nderiv=1)
    print('> basis  = ', values_2d.basis)

#    # ...
#    print('=== 1D case ===')
#    qd_1d = QuadratureGrid(V1)
#
#    print('> points  = ', qd_1d.points)
#    print('> weights = ', qd_1d.weights)
#    # ...

#    # ...
#    print('=== 2D case ===')
#    qd_2d = QuadratureGrid(V)
#
#    for x,w in zip(qd_2d.points, qd_2d.weights):
#        print('> points  = ', x)
#        print('> weights = ', w)
#    # ...
#
#    # ...
#    print('=== 2D case boundary ===')
#    qd_2d = BoundaryQuadratureGrid(V, axis=1, ext=-1)
#
#    for x,w in zip(qd_2d.points, qd_2d.weights):
#        print('> points  = ', x)
#        print('> weights = ', w)
#    # ...
