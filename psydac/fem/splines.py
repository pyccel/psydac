# coding: utf-8
# Copyright 2018 Ahmed Ratnani, Yaman Güçlü

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix

from psydac.linalg.stencil        import StencilVectorSpace
from psydac.linalg.direct_solvers import BandedSolver, SparseSolver
from psydac.fem.basic             import FemSpace, FemField
from psydac.core.bsplines         import (
        find_span,
        basis_funs,
        collocation_matrix,
        histopolation_matrix,
        breakpoints,
        greville,
        elements_spans,
        make_knots,
        scaling_vector
        )
from psydac.utilities.quadratures import gauss_legendre

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

    basis : str
        Set to "B" for B-splines (have partition of unity)
        Set to "M" for M-splines (have unit integrals)

    """
    def __init__( self, degree, knots=None, grid=None,
                  periodic=False, dirichlet=(False, False), basis='B' ):

        if basis not in ['B', 'M']:
            raise ValueError(" only options for basis functions are B or M ")

        self._degree    = degree
        self._periodic  = periodic
        self._dirichlet = dirichlet
        self._basis     = basis
        
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

        # Store flag: object NOT YET prepared for interpolation
        self._interpolation_ready = False

        # Coefficients to convert B-splines to M-splines (if needed)
        if self.basis == 'M':
            self._scaling_vector = scaling_vector(
                self.knots,
                self.degree,
                self.periodic)
        else:
            self._scaling_vector = None

        # Greville points
        self._greville = greville(self.knots, self.degree, self.periodic)

        # Knot sequence and Greville points of "extended" space of degree p+1
        # These are needed for performing histopolation
        self._ext_knots = make_knots(self.breaks, self.degree + 1, self.periodic)
        self._ext_greville = greville(self.ext_knots, self.degree + 1, self.periodic)

    # ...
    def collocation_solver(self):
        """
        Compute the 1D collocation matrix and factorize it, in preparation
        for the calculation of a spline interpolant given the values at the
        Greville points.

        """
        cmat = collocation_matrix(
            self.knots,
            self.degree,
            self.greville,
            self.periodic)

        self._collocation_solver = SparseSolver(csc_matrix(cmat))

    # ...
    def init_histopolation(self):
        """
        Compute the 1D histopolation matrix and factorize it, in preparation
        for the calculation of a spline interpolant give the integrals in the
        intervals defined by the extended Greville points.

        """
        # TODO: change signature of histopolation matrix
        hmat = histopolation_matrix(
            self.ext_knots,
            self.degree + 1,
            self.ext_greville,
            self.periodic)

        self._histopolation_solver = SparseSolver(csc_matrix(hmat))

    # ...
    def init_interpolation( self ):
        """
        Prepare collocation for B-splines, or histopolation for M-splines.

        B-splines:
            Compute the 1D collocation matrix and factorize it, in preparation
            for the calculation of a spline interpolant given the values at the
            Greville points.

        M-splines:
            Compute the 1D histopolation matrix and factorize it, in preparation
            for the calculation of a spline interpolant give the integrals in
            the intervals defined by the extended Greville points.

        """
        if self.basis == 'B':
            imat = collocation_matrix(
                self.knots,
                self.degree,
                self.greville,
                self.periodic)
        elif self.basis == 'M':
            imat = histopolation_matrix(
                self.ext_knots,
                self.degree + 1,
                self.ext_greville,
                self.periodic)
        else:
            raise NotImplementedError()

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
        self._interpolation_ready = True

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

    #--------------------------------------------------------------------------
    # Abstract interface: evaluation methods
    #--------------------------------------------------------------------------
    def eval_field( self, field, *eta ):

        assert isinstance( field, FemField )
        assert field.space is self
        assert len( eta ) == 1

        span  =  find_span( self.knots, self.degree, eta[0] )
        basis = basis_funs( self.knots, self.degree, eta[0], span )
        index = slice(span-self.degree, span+1)

        if self.basis == 'M':
            basis *= self._scaling_vector[index]

        return np.dot( field.coeffs[index], basis )

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
    def basis( self ):
        return self._basis

    @property
    def interpolation_grid( self ):
        if self.basis == 'B':
            return self.greville
        elif self.basis == 'M':
            return self.ext_greville
        else:
            raise NotImplementedError()

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
    def ext_knots( self ):
        """ Knot sequence of 'extended' space with degree p+1.
        """
        return self._ext_knots

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
    def greville( self ):
        """ Coordinates of all Greville points.
        """
        return self._greville

    @property
    def ext_greville( self ):
        """ Greville coordinates of 'extended' space with degree p+1.
        """
        return self._ext_greville
        
    @property
    def normalize(self):
        return self._normalize

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

        if not self._interpolation_ready:
            self.init_interpolation()

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
