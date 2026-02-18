#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from psydac.core.bsplines         import elements_spans
from psydac.core.bsplines         import quadrature_grid
from psydac.core.bsplines         import basis_ders_on_quad_grid
from psydac.core.bsplines         import elevate_knots
from psydac.utilities.quadratures import gauss_legendre
from psydac.fem.splines           import SplineSpace

__all__ = ('FemAssemblyGrid',)

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
        Index of first element local to process.

    end : int
        Index of last element local to process.

    nquads : int
        Number of quadrature points used in the Gauss-Legendre quadrature formula.

    nderiv : int
        Number of basis functions' derivatives to be precomputed at the Gauss
        points (default: 1).

    """
    def __init__(self, space, start, end, *, nquads, nderiv=1):

        assert isinstance(space, SplineSpace)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(nquads, int)
        assert isinstance(nderiv, int)

        # Useful shortcuts
        T      = space.knots   # knots sequence
        degree = space.degree  # spline degree
        n      = space.nbasis  # total number of control points
        grid   = space.breaks  # breakpoints

        # Gauss-legendre quadrature rule
        u, w = gauss_legendre(nquads)

        #-------------------------------------------
        # GLOBAL GRID
        #-------------------------------------------

        # Lists of quadrature coordinates and weights on each element
        global_points, global_weights = quadrature_grid(grid, u, w)

        # List of basis function values on each element
        global_basis = basis_ders_on_quad_grid(T, degree, global_points, nderiv, space.basis)

        # List of spans on each element
        # (Span is global index of last non-vanishing basis function)
        global_spans = elements_spans(T, degree)

        grid    = grid[start : end + 2]
        spans   = global_spans  [start : end + 1].copy()
        basis   = global_basis  [start : end + 1].copy()
        points  = global_points [start : end + 1].copy()
        weights = global_weights[start : end + 1].copy()

        #-------------------------------------------
        # DATA STORAGE IN OBJECT
        #-------------------------------------------

        # Quadrature data on extended distributed domain
        self._num_elements = len(grid) - 1
        self._num_quad_pts = len(u)
        self._spans        = spans
        self._basis        = basis
        self._points       = points
        self._weights      = weights
        self._indices      = tuple(range(start, end + 1))
        self._quad_rule_x  = u
        self._quad_rule_w  = w

        # Local index of start/end elements of domain partitioning
        self._local_element_start = 0
        self._local_element_end   = self._num_elements - 1

    # ...
    @property
    def num_elements(self):
        """ Number of elements over which integration should be performed.
        """
        return self._num_elements

    # ...
    @property
    def num_quad_pts(self):
        """ Number of quadrature points in each element.
        """
        return self._num_quad_pts

    # ...
    @property
    def spans(self):
        """ Span index in each element.
        """
        return self._spans

    # ...
    @property
    def basis(self):
        """ Basis function values (and their derivatives) at each quadrature point.
        """
        return self._basis

    # ...
    @property
    def points(self):
        """ Location of each quadrature point.
        """
        return self._points

    # ...
    @property
    def weights(self):
        """ Weight assigned to each quadrature point.
        """
        return self._weights

    # ...
    @property
    def indices(self):
        """ Global index of each element used in assembly process.
        """
        return self._indices

    # ...
    @property
    def quad_rule_x(self):
        """ Coordinates of quadrature points on canonical interval [-1,1].
        """
        return self._quad_rule_x

    # ...
    @property
    def quad_rule_w(self):
        """ Weights assigned to quadrature points on canonical interval [-1,1].
        """
        return self._quad_rule_w

    # ...
    @property
    def local_element_start(self):
        """ Local index of first element owned by process.
        """
        return self._local_element_start

    # ...
    @property
    def local_element_end(self):
        """ Local index of last element owned by process.
        """
        return self._local_element_end
