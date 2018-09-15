# coding: utf-8
# Copyright 2018 Ahmed Ratnani, Yaman Güçlü

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix

from spl.linalg.stencil        import StencilVectorSpace
from spl.linalg.direct_solvers import BandedSolver, SparseSolver
from spl.fem.basic             import FemSpace, FemField
from spl.core.bsplines         import (
        find_span,
        basis_funs,
        collocation_matrix,
        breakpoints,
        greville,
        elements_spans,
        make_knots,
        quadrature_grid,
        basis_ders_on_quad_grid
        )
from spl.utilities.quadratures import gauss_legendre
from spl.utilities.quadratures import quadrature_inter

__all__ = ['SplineSpace']

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
                  periodic=False, dirichlet=(False, False) ):

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

        if periodic:
            self._nbasis = len(knots) - 2*degree - 1
        else:
            defect = 0
            if dirichlet[0]: defect += 1
            if dirichlet[1]: defect += 1
            self._nbasis = len(knots) - degree - 1 - defect

        self._vector_space = StencilVectorSpace( [self.nbasis], [self.degree], [periodic] )
        self._fields = {}

        self._spans        = elements_spans( knots, degree )
        self._quad_order   = None
        self._quad_basis   = None
        self._quad_points  = None
        self._quad_weights = None

        # Store flag: object NOT YET prepared for interpolation
        self._collocation_ready = False

    # ...
    def init_fem( self, quad_order=None, nderiv=1 ):
        """
        Prepare some data that is useful for assembling finite element matrices.

        Parameters
        ----------
        quad_order : int
            Order of Gaussian quadrature (default: spline degree).

        nderiv : int
            Number of derivatives to be precomputed at the Gauss points (default: 1).

        """
        T    = self.knots       # knots sequence
        p    = self.degree      # spline degree
        n    = self.nbasis      # total number of control points
        grid = self.breaks      # breakpoints
        ne   = self.ncells      # number of cells in domain (ne=len(grid)-1)
        k    = quad_order or p  # polynomial order for which the mass matrix is exact

        # gauss-legendre quadrature rule
        if self.periodic:
            u, w = gauss_legendre( k )
            points, weights = quadrature_grid( self.breaks, u, w )

        else:
            # it seems that legendre gives better results than this formula with
            # radau
            # TODO check if it is a precision problem
            points, weights = quadrature_inter( self.breaks, k )

        basis = basis_ders_on_quad_grid( T, p, points, nderiv )

        self._quad_order   = k + 1
        self._quad_basis   = basis
        self._quad_points  = points
        self._quad_weights = weights

    # ...
    def init_collocation( self ):
        """
        Compute the 1D interpolation matrix and factorize it, in preparation
        for the calculation of a spline interpolant given the values at the
        Greville points.

        """
        imat = collocation_matrix(
            self.knots,
            self.degree,
            self.greville,
            self.periodic
        )

        if self.periodic:
            # Convert to CSC format and compute sparse LU decomposition
            self._interpolator = SparseSolver( csc_matrix( imat ) )
        else:
            # Convert to LAPACK banded format (see DGBTRF function)
            dmat = dia_matrix( imat )
            l = abs( dmat.offsets.min() )
            u =      dmat.offsets.max()
            cmat = csr_matrix( dmat )
            bmat = np.zeros( (1+u+2*l, cmat.shape[1]) )
            for i,j in zip( *cmat.nonzero() ):
                bmat[u+l+i-j,j] = cmat[i,j]
            self._interpolator = BandedSolver( u, l, bmat )

        # Store flag
        self._collocation_ready = True

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

    # ...
    # Identity mapping assumed for now
    # TODO: take into account Jacobian determinant of mapping
    def integral( self, f ):

        assert hasattr( f, '__call__' )

        if self.quad_basis is None: self.init_fem()

        nk      = self.ncells
        nq      = self.quad_order
        points  = self.quad_points
        weights = self.quad_weights

        c = 0.0
        for k in range( nk ):
            x =  points[k,:]
            w = weights[k,:]
            for q in range( nq ):
                c += f( x[q] ) * w[q]

        return c

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
    def compute_interpolant( self, values, field ):
        """
        Compute field (i.e. update its spline coefficients) such that it
        interpolates a certain function $f(x)$ at the Greville points.

        Parameters
        ----------
        values : array_like (nbasis,)
            Function values $f(x_i)$ at the 'nbasis' Greville points $x_i$,
            to be interpolated.

        field : FemField
            Input/output argument: spline that has to interpolate the given
            values.

        """
        assert len( values ) == self.nbasis
        assert isinstance( field, FemField )
        assert field.space is self

        if not self._collocation_ready:
            self.init_collocation()

        n = self.nbasis
        c = field.coeffs

        c[0:n] = self._interpolator.solve( values )
        c.update_ghost_regions()

    # ...
    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> ldim   :: {ldim}\n'.format( ldim=self.ldim )
        txt += '> nbasis :: {dim} \n'.format( dim=self.nbasis )
        txt += '> degree :: {degree}'.format( degree=self.degree )
        return txt
