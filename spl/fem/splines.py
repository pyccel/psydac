# coding: utf-8

from numpy import unique

from spl.linalg.stencil import VectorSpace as StencilVectorSpace
from spl.linalg.stencil import Vector as StencilVector
from spl.fem.basic import FemSpace, FemField


#===============================================================================
class SplineSpace( FemSpace ):
    """
    a 1D Splines Finite Element space

    Parameters
    ----------
    knots : array_like
        Coordinates of knots (clamped or extended by periodicity).

    degree : int
        Polynomial degree.

    periodic : bool
        True if domain is periodic, False otherwise.
        Default: False

    dirichlet : tuple, list
        True if using homogeneous dirichlet boundary conditions, False
        otherwise. Must be specified for each bound
        Default: (False, False)

    """
    def __init__( self, knots, degree, periodic=False, dirichlet=(False, False) ):

        self._knots    = knots
        self._degree   = degree
        self._periodic = periodic
        self._dirichlet = dirichlet
        self._ncells   = None # TODO will be computed later
        self._nbasis   = None # TODO self._ncells if periodic else self._ncells+degree
        if periodic:
            raise NotImplementedError('periodic bc not yet available')
        else:
            defect = 0
            if dirichlet[0]: defect += 1
            if dirichlet[1]: defect += 1
            self._nbasis = len(knots) - degree - 1 - defect

        starts = [0]
        ends = [self.nbasis]
        pads = [degree]
        self._vector_space = StencilVectorSpace(starts, ends, pads)

    @property
    def vector_space(self):
        """Returns the topological associated vector space."""
        return self._vector_space

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
    def nbasis( self ):
        """
        """
        return self._nbasis

    @property
    def periodic( self ):
        """ True if domain is periodic, False otherwise.
        """
        return self._periodic

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
            return unique(self.knots)
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
    def greville( self ):
        """ Coordinates of all Greville points.
        """
        raise NotImplementedError('TODO')

    def __str__(self):
        """Pretty printing"""
        txt  = '\n'
        txt += '> nbasis :: {dim}\n'.format(dim=self.nbasis)
        txt += '> degree :: {degree}'.format(degree=self.degree)
        return txt

#===============================================================================
class Spline( FemField ):
    """
    A field spline is an element of the SplineSpace.

    """
    def __init__(self, space):
        self._space = space
        self._coeffs = StencilVector( space.vector_space )

    @property
    def space( self ):
        return self._space

    @property
    def coeffs( self ):
        return self._coeffs
