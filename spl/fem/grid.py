# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import numpy as np

from spl.core.bsplines         import elements_spans
from spl.core.bsplines         import quadrature_grid
from spl.core.bsplines         import basis_ders_on_quad_grid
from spl.utilities.quadratures import gauss_legendre

__all__ = ['FemAssemblyGrid']

#==============================================================================
class FemAssemblyGrid:
    """
    Class that collects all 1D information local to process that are necessary
    for the correct assembly of l.h.s. matrix and r.h.s. vector in a finite
    element method.

    This works in the case of clamped and periodic splines, for a global or
    distributed domain.

    A 'TensorFemSpace' object will create one object of this class for each
    1D space.

    Parameters
    ----------
    space : SplineSpace
        1D finite element space.

    start : int
        Index of first 1D basis local to process.

    end : int
        Index of last 1D basis local to process.

    quad_order : int
        Polynomial order for which mass matrix is exact (assuming identity map).

    nderiv : int
        Number of basis functions' derivatives to be computed and stored.

    """
    def __init__( self, space, start, end, *, quad_order=None, nderiv=1 ):

        T    = space.knots      # knots sequence
        p    = space.degree     # spline degree
        n    = space.nbasis     # total number of control points
        grid = space.breaks     # breakpoints
        nc   = space.ncells     # number of cells in domain (nc=len(grid)-1)
        k    = quad_order or p  # polynomial order for which the mass matrix is exact

        # Gauss-legendre quadrature rule
        u, w = gauss_legendre( k )
        glob_points, glob_weights = quadrature_grid( grid, u, w )
        glob_basis = basis_ders_on_quad_grid( T, p, glob_points, nderiv )
        glob_spans = elements_spans( T, p )

        # Lists of local quadrature points and weights, basis functions values
        spans   = []
        basis   = []
        points  = []
        weights = []
        ne      = 0

        # a) Periodic case only, left-most process in 1D domain
        if space.periodic:
            for k in range( nc ):
                gk = glob_spans[k]
                if start <= gk-n and gk-n-p <= end:
                    spans  .append( glob_spans[k]-n )
                    basis  .append( glob_basis  [k] )
                    points .append( glob_points [k] )
                    weights.append( glob_weights[k] )
                    ne += 1

        # b) All cases
        for k in range( nc ):
            gk = glob_spans[k]
            if start <= gk and gk-p <= end:
                spans  .append( glob_spans  [k] )
                basis  .append( glob_basis  [k] )
                points .append( glob_points [k] )
                weights.append( glob_weights[k] )
                ne += 1

        # STORE
        self._num_elements = ne
        self._num_quad_pts = len( u )
        self._spans   = np.array( spans   )
        self._basis   = np.array( basis   )
        self._points  = np.array( points  )
        self._weights = np.array( weights )

    # ...
    @property
    def num_elements( self ):
        """ Number of elements over which integration should be performed.
        """
        return self._num_elements

    # ...
    @property
    def num_quad_pts( self ):
        """ Number of quadrature points in each element.
        """
        return self._num_quad_pts

    # ...
    @property
    def spans( self ):
        """ Span index in each element.
        """
        return self._spans

    # ...
    @property
    def basis( self ):
        """ Basis function values (and their derivatives) at each quadrature point.
        """
        return self._basis

    # ...
    @property
    def points( self ):
        """ Location of each quadrature point.
        """
        return self._points

    # ...
    @property
    def weights( self ):
        """ Weight assigned to each quadrature point.
        """
        return self._weights
