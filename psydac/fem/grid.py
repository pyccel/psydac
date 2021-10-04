# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import numpy as np

from psydac.core.bsplines         import elements_spans
from psydac.core.bsplines         import quadrature_grid
from psydac.core.bsplines         import basis_ders_on_quad_grid
from psydac.core.bsplines         import elevate_knots
from psydac.utilities.quadratures import gauss_legendre

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
        Polynomial order for which mass matrix is exact, assuming identity map
        (default: spline degree).

    nderiv : int
        Number of basis functions' derivatives to be precomputed at the Gauss
        points (default: 1).

    parent_start: int
        Index of first 1D parent basis local to process.

    parent_end: int
        Index of last 1D parent basis local to process.
    """
    def __init__( self, space, start, end, *, quad_order=None, nderiv=1, parent_start=None, parent_end=None):

        T            = space.knots           # knots sequence
        degree       = space.degree          # spline degree
        n            = space.nbasis          # total number of control points
        grid         = space.breaks          # breakpoints
        nc           = space.ncells          # number of cells in domain (nc=len(grid)-1)
        k            = quad_order or degree  # polynomial order for which the mass matrix is exact
        pad          = space.pads            # padding
        multiplicity = space.multiplicity    # multiplicity of the knots

        # Gauss-legendre quadrature rule
        u, w = gauss_legendre( k )

        # invert order
        u = u[::-1]
        w = w[::-1]

        #-------------------------------------------
        # GLOBAL GRID
        #-------------------------------------------

        # Lists of quadrature coordinates and weights on each element
        glob_points, glob_weights = quadrature_grid( grid, u, w )

        # List of basis function values on each element
        glob_basis = basis_ders_on_quad_grid( T, degree, glob_points, nderiv, space.basis )

        # List of spans on each element
        # (Span is global index of last non-vanishing basis function)

        glob_spans = elements_spans( T, degree )

        #-------------------------------------------
        # LOCAL GRID, EXTENDED (WITH GHOST REGIONS)
        #-------------------------------------------

        # Lists of local quadrature points and weights, basis functions values
        spans   = []
        basis   = []
        points  = []
        weights = []
        indices = []
        ne      = 0

        # Current start/end represent the parent start/end when the space is a reduction
        # from a parent space, otherwise we use the provided start/end.
 
        if pad==degree:
            current_glob_spans  = glob_spans
            current_start       = start
            current_end         = end
        elif pad-degree == 1:
            multiplicity  = space.parent_multiplicity
            elevated_T    = elevate_knots(T, degree, space.periodic, multiplicity=multiplicity)
            current_start = parent_start or start
            current_end   = parent_end   or end
            current_glob_spans  = elements_spans( elevated_T, pad )
        else:
            raise NotImplementedError('TODO')

        # a) Periodic case only, left-most process in 1D domain
#        if space.periodic:
#            for k in range( nc ):
#                gk = glob_spans[k]
#                if start <= gk-n and gk-n-pad <= end:
#                    spans  .append( glob_spans[k]-n )
#                    basis  .append( glob_basis  [k] )
#                    points .append( glob_points [k] )
#                    weights.append( glob_weights[k] )
#                    indices.append( k )
#                    ne += 1

        m = multiplicity if multiplicity>1 else 0

        # b) All cases
        for k in range( nc ):
            gk = current_glob_spans[k]
            gs = glob_spans  [k]
            if current_start-m <= gk-pad*space.periodic and gk-pad*space.periodic <= current_end:
                if m>0 and pad-degree==1 and start>gs:continue
                spans  .append( glob_spans  [k] )
                basis  .append( glob_basis  [k] )
                points .append( glob_points [k] )
                weights.append( glob_weights[k] )
                indices.append( k )
                ne += 1

        #-------------------------------------------
        # DATA STORAGE IN OBJECT
        #-------------------------------------------
        # Quadrature data on extended distributed domain
        self._num_elements = ne
        self._num_quad_pts = len( u )
        self._spans        = np.array( spans   )
        self._basis        = np.array( basis   )
        self._points       = np.array( points  )
        self._weights      = np.array( weights )
        self._indices      = np.array( indices )
        self._quad_rule_x  = u
        self._quad_rule_w  = w

        #-------------------------------------------
        # LOCAL GRID, PROPER (WITHOUT GHOST REGIONS)
        #-------------------------------------------

        # Local indices of first/last elements in proper domain
        if space.periodic:
            local_element_start = 0
            local_element_end   = ne-1
        else:
            if end+1 >= degree:
                local_element_start = 0
                local_element_end   = ne-1
            else:
                # in this edge case: no local elements for now
                local_element_start = 1
                local_element_end = 0

        # Local index of start/end elements of domain partitioning
        self._local_element_start = local_element_start
        self._local_element_end   = local_element_end
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

    # ...
    @property
    def indices( self ):
        """ Global index of each element used in assembly process.
        """
        return self._indices

    # ...
    @property
    def quad_rule_x( self ):
        """ Coordinates of quadrature points on canonical interval [-1,1].
        """
        return self._quad_rule_x

    # ...
    @property
    def quad_rule_w( self ):
        """ Weights assigned to quadrature points on canonical interval [-1,1].
        """
        return self._quad_rule_w

    # ...
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
