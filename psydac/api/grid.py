# coding: utf-8

import numpy as np

from psydac.utilities.quadratures  import gauss_legendre
from psydac.core.bsplines          import quadrature_grid
from psydac.core.bsplines          import find_span
from psydac.core.bsplines          import basis_funs_all_ders
from psydac.core.bsplines          import basis_ders_on_quad_grid
from psydac.fem.splines            import SplineSpace
from psydac.fem.tensor             import TensorFemSpace
from psydac.fem.vector             import ProductFemSpace
from psydac.fem.grid               import FemAssemblyGrid

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

        if isinstance(V, ProductFemSpace):
            quad_order = np.array([v.degree for v in V.spaces])
            quad_order = tuple(quad_order.max(axis=0))
            V = V.spaces[0] 

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

        points  = self.points
        weights = self.weights

        # ...
        if V.ldim == 1:
            assert( isinstance( V, SplineSpace ) )

            bounds = {}
            bounds[-1] = V.domain[0]
            bounds[1]  = V.domain[1]

            points[axis]  = np.asarray([[bounds[ext]]])
            weights[axis] = np.asarray([[1.]])

        elif V.ldim in [2, 3]:
            assert( isinstance( V, TensorFemSpace ) )

            bounds = {}
            bounds[-1] = V.spaces[axis].domain[0]
            bounds[1]  = V.spaces[axis].domain[1]

            points[axis]  = np.asarray([[bounds[ext]]])
            weights[axis] = np.asarray([[1.]])
        # ...

        self._axis    = axis
        self._ext     = ext
        self._points  = points
        self._weights = weights

    @property
    def axis(self):
        return self._axis

    @property
    def ext(self):
        return self._ext

#==============================================================================
def create_fem_assembly_grid(V, quad_order=None, nderiv=1):
    if isinstance(V, ProductFemSpace):
        quad_order = np.array([v.degree for v in V.spaces])
        quad_order = tuple(quad_order.max(axis=0))
        return [create_fem_assembly_grid(space,quad_order,nderiv) for space in V.spaces]
    # ...
    if not( quad_order is None ):
        if isinstance( quad_order, int ):
            quad_order = [quad_order for i in range(V.ldim)]

        elif not isinstance( quad_order, (list, tuple) ):
            raise TypeError('Expecting a tuple/list or int')

    else:
        quad_order = [None for i in range(V.ldim)]
    # ...
    if not( nderiv is None ):
        if isinstance( nderiv, int ):
            nderiv = [nderiv for i in range(V.ldim)]

        elif not isinstance( nderiv, (list, tuple) ):
            raise TypeError('Expecting a tuple/list or int')

    else:
        nderiv = [1 for i in range(V.ldim)]
    # ...

    return [FemAssemblyGrid(W, s, e, normalize=W.normalize, quad_order=n, nderiv=d )
            for W,s,e,n,d in zip( V.spaces,
                                  V.vector_space.starts, V.vector_space.ends,
                                  quad_order, nderiv ) ]


#==============================================================================
class BasisValues():
    def __init__( self, V, grid, nderiv ):
        assert( isinstance( grid, QuadratureGrid ) )

        # TODO quad_order in FemAssemblyGrid must be be the order and not the
        # degree
        quad_order = [q-1 for q in grid.quad_order]
        global_quad_grid = create_fem_assembly_grid( V,
                                              quad_order=quad_order,
                                              nderiv=nderiv )

        if isinstance(V, ProductFemSpace):
            self._spans = [[g.spans for g in quad_grid] for quad_grid in global_quad_grid]
            self._basis = [[g.basis for g in quad_grid] for quad_grid in global_quad_grid]
        else:
            self._spans = [g.spans for g in global_quad_grid]
            self._basis = [g.basis for g in global_quad_grid]

        # Modify data for boundary grid
        if isinstance(grid, BoundaryQuadratureGrid):
            axis = grid.axis
            if isinstance(V, ProductFemSpace):
                for i in range(len(V.spaces)):
                    sp_space = V.spaces[i].spaces[axis]
                    points = grid.points[axis]
                    boundary_basis = basis_ders_on_quad_grid(sp_space.knots, sp_space.degree, points, nderiv)
                    self._basis[i][axis][0:1, :, :, 0:1] = boundary_basis
            else:
                sp_space = V.spaces[axis]
                points = grid.points[axis]
                boundary_basis = basis_ders_on_quad_grid(sp_space.knots, sp_space.degree, points, nderiv)
                self._basis[axis][0:1, :, :, 0:1] = boundary_basis

    @property
    def basis(self):
        return self._basis

    @property
    def spans(self):
        return self._spans

#==============================================================================
# TODO have a parallel version of this function, as done for fem
def create_collocation_basis( glob_points, space, nderiv=1 ):

    T    = space.knots      # knots sequence
    p    = space.degree     # spline degree
    n    = space.nbasis     # total number of control points
    grid = space.breaks     # breakpoints
    nc   = space.ncells     # number of cells in domain (nc=len(grid)-1)

    #-------------------------------------------
    # GLOBAL GRID
    #-------------------------------------------

    # List of basis function values on each element
    nq = len(glob_points)
    glob_spans = np.zeros( nq, dtype='int' )
#    glob_basis = np.zeros( (p+1,nderiv+1,nq) ) # TODO use this for local basis fct
    glob_basis = np.zeros( (n+p,nderiv+1,nq) ) # n+p for ghosts
    for iq,xq in enumerate(glob_points):
        span = find_span( T, p, xq )
        glob_spans[iq] = span

        ders = basis_funs_all_ders( T, p, xq, span, nderiv )
        glob_basis[span:span+p+1,:,iq] = ders.transpose()

    return glob_points, glob_spans, glob_basis


#==============================================================================
# TODO experimental
class CollocationBasisValues():
    def __init__( self, points, V, nderiv ):

        assert(isinstance(V, TensorFemSpace))

        _points  = []
        _basis = []
        _spans = []

        for i,W in enumerate(V.spaces):
            ps, sp, bs = create_collocation_basis( points[i], W, nderiv=nderiv )
            _points.append(ps)
            _basis.append(bs)
            _spans.append(sp)

        self._spans = _spans
        self._basis = _basis

    @property
    def basis(self):
        return self._basis

    @property
    def spans(self):
        return self._spans
