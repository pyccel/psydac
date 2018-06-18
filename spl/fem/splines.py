# coding: utf-8

import numpy as np

from spl.linalg.stencil import StencilVectorSpace
from spl.fem.basic      import FemSpace, FemField
from spl.core.bsplines  import (find_span, basis_funs, breakpoints, greville,
                               elements_spans, make_knots, quadrature_grid)
from spl.utilities.quadratures import gauss_legendre

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

    quad_order : int
        Order of Gaussian quadrature.
        Default: degree+1

    nderiv : int
        Number of derivatives to be pre-computed at quadrature points.
        Default: 1

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
            knots = make_knots( grid, degree, periodic )

        # TODO: verify that user-provided knots make sense in periodic case

        self._knots  = knots
        self._ncells = len(self.breaks) - 1
        self._nderiv = nderiv

        if quad_order is None:
            self._quad_order = degree + 1
        else:
            self._quad_order = quad_order

        if periodic:
            self._nbasis = len(knots) - 2*degree - 1
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

        return np.dot( field.coeffs[span-self.degree:span+1], basis )

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
        return breakpoints( self._knots, self._degree )

    @property
    def domain( self ):
        """ Domain boundaries [a,b].
        """
        breaks = self.breaks
        return breaks[0], breaks[-1]

    @property
    def nderiv(self):
        """Returns number of derivatives."""
        return self._nderiv

    @property
    def spans(self):
        """Returns the last non-vanishing spline for each element."""
        return self._spans

    @property
    def quad_order(self):
        """Returns the quadrature order."""
        return self._quad_order

    @property
    def quad_points(self):
        """Returns the quadrature points over the whole domain."""
        return self._quad_points

    @property
    def quad_weights(self):
        """Returns the quadrature weights over the whole domain."""
        return self._quad_weights

    @property
    def quad_basis(self):
        """Returns B-Splines and their derivatives on the quadrature grid."""
        return self._quad_basis

    @property
    def greville( self ):
        """ Coordinates of all Greville points.
        """
        return greville( self._knots, self._degree, self._periodic )

    #--------------------------------------------------------------------------
    # Other methods
    #--------------------------------------------------------------------------
    def _initialize(self):
        """Initializes the Spline space. Here we prepare some data that may be
        useful for assembling finite element matrices"""

        from spl.core.interface import eval_on_grid_splines_ders

        T    = self.knots   # knots sequence
        p    = self.degree  # spline degree
        n    = self.nbasis  # total number of control points
        grid = self.breaks  # breakpoints
        ne   = self.ncells  # number of cells in domain (ne=len(grid)-1)
        k    = self.quad_order
        d    = self.nderiv

        # compute spans
        spans = elements_spans( T, p )

        # gauss-legendre quadrature rule
        u, w = gauss_legendre(p)
        points, weights = quadrature_grid( self.breaks, u, w )

        basis = eval_on_grid_splines_ders(p, n, k, d, T, points.T)

        self._spans        = spans
        self._quad_basis   = basis
        self._quad_points  = points
        self._quad_weights = weights

    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> ldim   :: {ldim}\n'.format( ldim=self.ldim )
        txt += '> nbasis :: {dim} \n'.format( dim=self.nbasis )
        txt += '> degree :: {degree}'.format( degree=self.degree )
        return txt
