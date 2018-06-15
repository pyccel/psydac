# coding: utf-8

import numpy as np

from spl.linalg.stencil import StencilVectorSpace
from spl.fem.basic      import FemSpace, FemField
from spl.core.bsplines  import find_span, basis_funs, greville


#===============================================================================
class SplineSpace( FemSpace ):
    """
    a 1D Splines Finite Element space

    Parameters
    ----------
    degree : int
        Polynomial degree.

    knots : array_like
        Coordinates of knots (clamped or extended by periodicity).

    grid: array_like
        Coorinates of the grid. Used to construct the knots sequence, if not given.

    periodic : bool
        True if domain is periodic, False otherwise.
        Default: False

    dirichlet : tuple, list
        True if using homogeneous dirichlet boundary conditions, False
        otherwise. Must be specified for each bound
        Default: (False, False)

    """
    def __init__( self, degree, knots=None, grid=None,
                  periodic=False, dirichlet=(False, False),
                  quad_order=None, nderiv=1 ):

        self._degree    = degree
        self._periodic  = periodic
        self._dirichlet = dirichlet

        if not( knots is None ) and not( grid is None ):
            raise ValueError( 'Cannot provide both grid and knots.' )

        if knots is None:
            # create knots from grid and bc
            from spl.core.interface import make_knots
            knots = make_knots( grid, degree, periodic )

        self._knots  = knots
        self._ncells = len(self.breaks) - 1
        self._nderiv = nderiv

        if quad_order is None:
            self._quad_order = degree + 1
        else:
            self._quad_order = quad_order

        if periodic:
            self._nbasis = self.ncells
        else:
            defect = 0
            if dirichlet[0]: defect += 1
            if dirichlet[1]: defect += 1
            self._nbasis = len(knots) - degree - 1 - defect

        self._vector_space = StencilVectorSpace( [self.nbasis], [self.degree], [periodic] )
        self._fields = {}
        self._initialize()

    #--------------------------------------------------------------------------
    # Abstract interface: read-only attributes
    #--------------------------------------------------------------------------
    @property
    def ldim( self ):
        """ Parametric dimension.
        """
        return 1

    @property
    def periodic( self ):
        """ True if domain is periodic, False otherwise.
        """
        return self._periodic

    @property
    def mapping( self ):
        """ Assume identity mapping for now.
        """
        return None

    @property
    def vector_space( self ):
        """Returns the topological associated vector space."""
        return self._vector_space

    @property
    def fields( self ):
        """Dictionary containing all FemField objects associated to this space."""
        return self._fields

    #--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field( self, field, *eta ):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == 1

        span  =  find_span( self.knots, self.degree, eta[0] )
        basis = basis_funs( self.knots, self.degree, eta[0], span )

        return np.dot( field.coeffs[span-degree:span+1], basis )

    # ...
    def eval_field_gradient( self, field, *eta ):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == 1

        raise NotImplementedError()

    #--------------------------------------------------------------------------
    # Other properties
    #--------------------------------------------------------------------------
    @property
    def is_scalar( self ):
        """ Only scalar field is implemented for now.
        """
        return True

    @property
    def nbasis( self ):
        """
        """
        return self._nbasis

    @property
    def degree( self ):
        """ Degree of B-splines.
        """
        return self._degree

    @property
    def ncells( self ):
        """ Number of cells in domain.
        """
        return self._ncells

    @property
    def dirichlet( self ):
        """ True if using homogeneous dirichlet boundary conditions, False otherwise.
        """
        return self._dirichlet

    @property
    def knots( self ):
        """ Knot sequence.
        """
        return self._knots

    @property
    def breaks( self ):
        """ List of breakpoints.
        """
        if not self.periodic:
            return np.unique(self.knots)
        else:
            p = self._degree
            return self._knots[p:-p]

    @property
    def domain( self ):
        """ Domain boundaries [a,b].
        """
        breaks = self.breaks
        return breaks[0], breaks[-1]

    @property
    def quad_order(self):
        """Returns the quadrature order."""
        return self._quad_order

    @property
    def nderiv(self):
        """Returns number of derivatives."""
        return self._nderiv

    @property
    def basis(self):
        """Returns B-Splines and their derivatives on the quadrature grid."""
        return self._basis

    @property
    def spans(self):
        """Returns the last non-vanishing spline for each element."""
        return self._spans

    @property
    def points(self):
        """Returns the quadrature points over the whole domain."""
        return self._points

    @property
    def weights(self):
        """Returns the quadrature weights over the whole domain."""
        return self._weights

    @property
    def greville( self ):
        """ Coordinates of all Greville points.
        """
        return greville( self._knots, self._degree, self._periodic )

    def _initialize(self):
        """Initializes the Spline space. Here we prepare some data that may be
        useful for assembling finite element matrices"""

        from spl.core.interface import construct_grid_from_knots
        from spl.core.interface import construct_quadrature_grid
        from spl.core.interface import compute_spans
        from spl.core.interface import eval_on_grid_splines_ders
        from spl.utilities.quadratures import gauss_legendre

        T = self.knots
        p = self.degree
        ne = self.ncells
        k = self.quad_order
        d = self.nderiv

        # ... total number of control points
        if self.periodic:
            raise NotImplementedError('periodic bc not yet available')
        else:
            n = len(T) - p - 1
        # ...

        # constructs the grid from the knot vector
        grid = construct_grid_from_knots(p, n, T)

        # compute spans
        spans = compute_spans(p, n, T)

        # gauss-legendre quadrature rule
        u, w = gauss_legendre(p)
        points, weights = construct_quadrature_grid(ne, k, u, w, grid)

        basis = eval_on_grid_splines_ders(p, n, k, d, T, points)

        self._basis = basis
        self._spans = spans
        self._points = points
        self._weights = weights

    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> ldim   :: {ldim}\n'.format( ldim=self.ldim )
        txt += '> nbasis :: {dim} \n'.format( dim=self.nbasis )
        txt += '> degree :: {degree}'.format( degree=self.degree )
        return txt

#===============================================================================

#------
# TODO: remove this class and make FemField a concrete class!
#------
# 
# class Spline( FemField ):
#     """
#     A field spline is an element of the SplineSpace.
# 
#     """
#     def __init__(self, space):
#         self._space = space
#         self._coeffs = StencilVector( space.vector_space )
# 
#     #--------------------------------------------------------------------------
#     # Abstract interface
#     #--------------------------------------------------------------------------
#     @property
#     def space( self ):
#         return self._space
# 
#     @property
#     def coeffs( self ):
#         return self._coeffs
# 
#     def __call__( self, *eta ):
#         return self.space.eval_field( self, *eta )
# 
#     def gradient( self, *eta ):
#         return self.space.eval_field_gradient( self, *eta )
