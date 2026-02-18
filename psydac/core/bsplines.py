#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Basic module that provides the means for evaluating the B-Splines basis
functions and their derivatives. In order to simplify automatic Fortran code
generation with Pyccel, no object-oriented features are employed.

References:

   - [1] L. Piegl and W. Tiller. The NURBS Book, 2nd ed., Springer-Verlag Berlin Heidelberg GmbH, 1997.

   - [2] SELALIB, Semi-Lagrangian Library. http://selalib.gforge.inria.fr

"""
import numpy as np

from psydac.core.bsplines_kernels import (find_span_p,
                                          find_spans_p,
                                          basis_funs_p,
                                          basis_funs_array_p,
                                          basis_funs_1st_der_p,
                                          basis_funs_all_ders_p,
                                          collocation_matrix_p,
                                          histopolation_matrix_p,
                                          greville_p,
                                          breakpoints_p,
                                          elements_spans_p,
                                          make_knots_p,
                                          elevate_knots_p,
                                          quadrature_grid_p,
                                          basis_ders_on_quad_grid_p,
                                          basis_integrals_p,
                                          cell_index_p,
                                          basis_ders_on_irregular_grid_p)

__all__ = ('find_span',
           'find_spans',
           'basis_funs',
           'basis_funs_array',
           'basis_funs_1st_der',
           'basis_funs_all_ders',
           'collocation_matrix',
           'histopolation_matrix',
           'breakpoints',
           'greville',
           'elements_spans',
           'make_knots',
           'elevate_knots',
           'quadrature_grid',
           'basis_integrals',
           'basis_ders_on_quad_grid',
           'cell_index',
           'basis_ders_on_irregular_grid')


#==============================================================================
def find_span(knots, degree, x):
    """
    Determine the knot span index at location x, given the B-Splines' knot
    sequence and polynomial degree. See Algorithm A2.1 in [1].

    For a degree p, the knot span index i identifies the indices [i-p:i] of all
    p+1 non-zero basis functions at a given location x.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Location of interest.

    Returns
    -------
    span : int
        Knot span index.
    """
    x = float(x)
    knots = np.ascontiguousarray(knots, dtype=float)
    return find_span_p(knots, degree, x)

#==============================================================================
def find_spans(knots, degree, x, out=None):
    """
    Determine the knot span index at a set of locations x, given the B-Splines' knot
    sequence and polynomial degree. See Algorithm A2.1 in [1].

    For a degree p, the knot span index i identifies the indices [i-p:i] of all
    p+1 non-zero basis functions at a given location x.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : array_like of floats
        Locations of interest.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    spans : array of ints
        Knots span indexes.
    """
    knots = np.ascontiguousarray(knots, dtype=float)
    x = np.ascontiguousarray(x, dtype=float)
    if out is None:
        out = np.zeros_like(x, dtype=int)
    else:
        assert out.shape == x.shape and out.dtype == np.dtype('int')

    find_spans_p(knots, degree, x, out)
    return out

#==============================================================================
def basis_funs(knots, degree, x, span, out=None):
    """
    Compute the non-vanishing B-splines at a unique location.

    Parameters
    ----------
    knots : array_like of floats
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    array
        1D array containing the values of ``degree + 1`` non-zero
        Bsplines at location ``x``.
    """
    knots = np.ascontiguousarray(knots, dtype=float)
    # Get native float
    x = float(x)
    if out is None:
        out = np.zeros(degree + 1, dtype=float)
    else:
        assert out.shape == (degree + 1,) and out.dtype == np.dtype('float')
    basis_funs_p(knots, degree, x, span, out)
    return out

#==============================================================================
def basis_funs_array(knots, degree, span, x, out=None):
    """Compute the non-vanishing B-splines at several locations.

    Parameters
    ----------
    knots : array_like of floats
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : array_like of floats
        Evaluation points.

    span : array_like of int
        Knot span indexes.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    array
        2D array of shape ``(len(x), degree + 1)`` containing the values of ``degree + 1`` non-zero
        Bsplines at each location in ``x``.
    """
    knots = np.ascontiguousarray(knots, dtype=float)
    x = np.ascontiguousarray(x, dtype=float)
    if out is None:
        out = np.zeros(x.shape + (degree + 1,), dtype=float)
    else:
        assert out.shape == x.shape + (degree + 1,) and out.dtype == np.dtype('float')
    basis_funs_array_p(knots, degree, x, span,  out)
    return out

#==============================================================================
def basis_funs_1st_der(knots, degree, x, span, out=None):
    """
    Compute the first derivative of the non-vanishing B-splines at a location.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    array
        1D array of size ``degree + 1`` containing the derivatives of the
        ``degree + 1`` non-vanishing B-Splines at location x.

    Notes
    -----
    See function 's_bsplines_non_uniform__eval_deriv' in Selalib's ([2]) source file
    'src/splines/sll_m_bsplines_non_uniform.F90'.

    References
    ----------
    .. [2] SELALIB, Semi-Lagrangian Library. http://selalib.gforge.inria.fr
    """
    knots = np.ascontiguousarray(knots, dtype=float)
    # Get native float to work on windows
    x = float(x)
    if out is None:
        out = np.zeros(degree + 1, dtype=float)
    else:
        assert out.shape == (degree + 1,) and out.dtype == np.dtype('float')

    basis_funs_1st_der_p(knots, degree, x, span, out)
    return out

#==============================================================================
def basis_funs_all_ders(knots, degree, x, span, n, normalization='B', out=None):
    """
    Evaluate value and n derivatives at x of all basis functions with
    support in interval :math:`[x_{span-1}, x_{span}]`.

    If called with normalization='M', this uses M-splines instead of B-splines.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    x : float
        Evaluation point.

    span : int
        Knot span index.

    n : int
        Max derivative of interest.

    normalization: str
        Set to 'B' to get B-Splines and 'M' to get M-Splines

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    ders : array
        2D array of n+1 (from 0-th to n-th) derivatives at x of all (degree+1)
        non-vanishing basis functions in given span.
        ders[i,j] = (d/dx)^i B_k(x) with k=(span-degree+j),
        for 0 <= i <= n and 0 <= j <= degree+1.
    """
    knots = np.ascontiguousarray(knots, dtype=float)
    # Get native float to work on windows
    x = float(x)
    if out is None:
        out = np.zeros((n + 1, degree + 1), dtype=float)
    else:
        assert out.shape == (n + 1, degree + 1) and out.dtype == np.dtype('float')

    basis_funs_all_ders_p(knots, degree, x, span, n, normalization == 'M', out)
    return out

#==============================================================================
def collocation_matrix(knots, degree, periodic, normalization, xgrid, out=None, multiplicity = 1):
    """Computes the collocation matrix

    If called with normalization='M', this uses M-splines instead of B-splines.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of spline space.

    periodic : bool
        True if domain is periodic, False otherwise.

    normalization : str
        Set to 'B' for B-splines, and 'M' for M-splines.

    xgrid : array_like
        Evaluation points.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.
        
    multiplicity : int
        Multiplicity of the knots in the knot sequence, we assume that the same 
        multiplicity applies to each interior knot.

    Returns
    -------
    colloc_matrix : ndarray of floats
        Array containing the collocation matrix.

    Notes
    -----
    The collocation matrix :math:`C_ij = B_j(x_i)`, contains the
    values of each B-spline basis function :math:`B_j` at all locations :math:`x_i`.
    """
    if xgrid.size == 1:
        return np.ones((1, 1), dtype=float)

    knots = np.ascontiguousarray(knots, dtype=float)
    xgrid = np.ascontiguousarray(xgrid, dtype=float)
    if out is None:
        nb = len(knots) - degree - 1
        if periodic:
            nb -= degree + 1 - multiplicity

        out = np.zeros((xgrid.shape[0], nb), dtype=float)
    else:
        assert out.shape == ((xgrid.shape[0], nb)) and out.dtype == np.dtype('float')

    bool_normalization = normalization == "M"
    multiplicity = int(multiplicity)

    collocation_matrix_p(knots, degree, periodic, bool_normalization, xgrid, out, multiplicity=multiplicity)
    return out

#==============================================================================
def histopolation_matrix(knots, degree, periodic, normalization, xgrid, multiplicity=1, check_boundary=True, out=None):
    """Computes the histopolation matrix.

    If called with normalization='M', this uses M-splines instead of B-splines.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of spline space.

    periodic : bool
        True if domain is periodic, False otherwise.

    normalization : str
        Set to 'B' for B-splines, and 'M' for M-splines.

    xgrid : array_like
        Grid points.
        
    multiplicity : int
        Multiplicity of the knots in the knot sequence, we assume that the same 
        multiplicity applies to each interior knot.

    check_boundary : bool, default=True
        If true and ``periodic``, will check the boundaries of ``xgrid``.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    array
        Histopolation matrix

    Notes
    -----
    The histopolation matrix :math:`H_{ij} = \\int_{x_i}^{x_{i+1}}B_j(x)\\,dx`
    contains the integrals of each B-spline basis function :math:`B_j` between
    two successive grid points.
    """
    # Check that knots are ordered (but allow repeated knots)
    if not np.all(np.diff(knots) >= 0):
        raise ValueError("Cannot accept knot sequence: {}".format(knots))

    # Check that spline degree is non-negative integer
    if not isinstance(degree, (int, np.integer)):
        raise TypeError("Degree {} must be integer, got type {} instead".format(degree, type(degree)))
    if degree < 0:
        raise ValueError("Cannot accept negative degree: {}".format(degree))

    # Check 'periodic' flag
    if not isinstance(periodic, bool):
        raise TypeError("Cannot accept non-boolean 'periodic' parameter: {}".format(periodic))

    # Check 'normalization' option
    if normalization not in ['B', 'M']:
        raise ValueError("Cannot accept 'normalization' parameter: {}".format(normalization))

    # Check that grid points are ordered, and do not allow repetitions
    if not np.all(np.diff(xgrid) > 0):
        raise ValueError("Grid points must be ordered, with no repetitions: {}".format(xgrid))

    knots = np.ascontiguousarray(knots, dtype=float)
    xgrid = np.ascontiguousarray(xgrid, dtype=float)
    elevated_knots = elevate_knots(knots, degree, periodic, multiplicity=multiplicity)

    normalization = normalization == "M"

    if out is None:
        if periodic:
            out = np.zeros((len(xgrid), len(knots) - 2 * degree - 2 + multiplicity), dtype=float)
        else:
            out = np.zeros((len(xgrid) - 1, len(elevated_knots) - (degree + 1) - 1 - 1), dtype=float)
    else:
        if periodic:
            assert out.shape == (len(xgrid), len(knots) - 2 * degree - 2 + multiplicity)
        else:
            assert out.shape == (len(xgrid) - 1, len(elevated_knots) - (degree + 1) - 1 - 1)
        assert out.dtype == np.dtype('float')
    multiplicity = int(multiplicity)
    histopolation_matrix_p(knots, degree, periodic, normalization, xgrid, check_boundary, elevated_knots, out, multiplicity = multiplicity)
    return out

#==============================================================================
def breakpoints(knots, degree, tol=1e-15, out=None):
    """
    Determine breakpoints' coordinates.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    tol: float
        If the distance between two knots is less than tol, we assume 
        that they are repeated knots which correspond to the same break point.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    breaks : numpy.ndarray (1D)
        Abscissas of all breakpoints.
    """
    knots = np.ascontiguousarray(knots, dtype=float)
    if out is None:
        out = np.zeros(len(knots), dtype=float)
    else:
        assert out.shape == knots.shape and out.dtype == np.dtype('float')
    i_final = breakpoints_p(knots, degree, out, tol)
    return out[:i_final]

#==============================================================================
def greville(knots, degree, periodic, out=None, multiplicity=1):
    """
    Compute coordinates of all Greville points.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    periodic : bool
        True if domain is periodic, False otherwise.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.
        
    multiplicity : int
        Multiplicity of the knots in the knot sequence, we assume that the same 
        multiplicity applies to each interior knot.

    Returns
    -------
    greville : numpy.ndarray (1D)
        Abscissas of all Greville points.

    """
    knots = np.ascontiguousarray(knots, dtype=float)
    if out is None:
        n = len(knots) - 2 * degree - 2 + multiplicity if periodic else len(knots) - degree - 1
        out = np.zeros(n)
    multiplicity = int(multiplicity)
    greville_p(knots, degree, periodic, out, multiplicity)
    return out

#===============================================================================
def elements_spans(knots, degree, out=None):
    """
    Compute the index of the last non-vanishing spline on each grid element
    (cell). The length of the returned array is the number of cells.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    spans : numpy.ndarray (1D)
        Index of last non-vanishing spline on each grid element.

    Examples
    --------
    >>> import numpy as np
    >>> from psydac.core.bsplines import make_knots, elements_spans

    >>> p = 3 ; n = 8
    >>> grid  = np.arange( n-p+1 )
    >>> knots = make_knots( breaks=grid, degree=p, periodic=False )
    >>> spans = elements_spans( knots=knots, degree=p )
    >>> spans
    array([3, 4, 5, 6, 7])

    Notes
    -----
    1) Numbering of basis functions starts from 0, not 1;
    2) This function could be written in two lines:

       breaks = breakpoints( knots, degree )
       spans  = np.searchsorted( knots, breaks[:-1], side='right' ) - 1

    """
    knots = np.ascontiguousarray(knots, dtype=float)
    if out is None:
        out = np.zeros(len(knots), dtype=np.int64)
    else:
        assert out.shape == knots.shape and out.dtype == np.dtype('int64')
    i_final = elements_spans_p(knots, degree, out)
    return out[:i_final]

#===============================================================================
def make_knots(breaks, degree, periodic, multiplicity=1, out=None):
    """
    Create spline knots from breakpoints, with appropriate boundary conditions.
    
    If domain is periodic, knot sequence is extended by periodicity to have a 
    total of (n_cells-1)*mult+2p+2 knots (all break points are repeated mult 
    time and we add p+1-mult knots by periodicity at each side).
    
    Otherwise, knot sequence is clamped (i.e. endpoints have multiplicity p+1).

    Parameters
    ----------
    breaks : array_like
        Coordinates of breakpoints (= cell edges); given in increasing order and
        with no duplicates.

    degree : int
        Spline degree (= polynomial degree within each interval).

    periodic : bool
        True if domain is periodic, False otherwise.

    multiplicity: int
        Multiplicity of the knots in the knot sequence, we assume that the same 
        multiplicity applies to each interior knot.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    T : numpy.ndarray (1D)
        Coordinates of spline knots.

    """
    # Type checking
    assert isinstance( degree  , int  )
    assert isinstance( periodic, bool )

    # Consistency checks
    assert len(breaks) > 1
    assert all( np.diff(breaks) > 0 )
    assert degree >= 0
    assert 1 <= multiplicity and multiplicity <= degree + 1
    # Cast potential numpy.int64 into python native int
    multiplicity = int(multiplicity)

    if periodic:
        assert len(breaks) > degree

    breaks = np.ascontiguousarray(breaks, dtype=float)
    if out is None:
        out = np.zeros(multiplicity * len(breaks[1:-1]) + 2 + 2 * degree)
    else:
        assert out.shape == (multiplicity * len(breaks[1:-1]) + 2 + 2 * degree,) \
            and out.dtype == np.dtype('float')
    make_knots_p(breaks, degree, periodic, out, multiplicity)

    return out

#==============================================================================
def elevate_knots(knots, degree, periodic, multiplicity=1, tol=1e-15, out=None):
    """
    Given the knot sequence of a spline space S of degree p, compute the knot
    sequence of a spline space S_0 of degree p+1 such that u' is in S for all
    u in S_0.

    Specifically, on bounded domains the first and last knots are repeated in
    the sequence, and in the periodic case the knot sequence is extended by
    periodicity.

    Parameters
    ----------
    knots : array_like
        Knots sequence of spline space of degree p.

    degree : int
        Spline degree (= polynomial degree within each interval).

    periodic : bool
        True if domain is periodic, False otherwise.

    multiplicity : int
        Multiplicity of the knots in the knot sequence, we assume that the same 
        multiplicity applies to each interior knot.

    tol: float
        If the distance between two knots is less than tol, we assume 
        that they are repeated knots which correspond to the same break point.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    new_knots : ndarray
        Knots sequence of spline space of degree p+1.
    """
    multiplicity = int(multiplicity)
    knots = np.ascontiguousarray(knots, dtype=float)
    if out is None:
        if periodic:
            out = np.zeros(knots.shape[0] + 2, dtype=float)
        else:
            shape = 2*(degree + 2)
            if len(knots) - 2 * (degree + 1) > 0:
                uniques = (np.diff(knots[degree + 1:-degree - 1]) > tol).nonzero()
                shape += multiplicity * (1 + uniques[0].shape[0])
            out = np.zeros(shape, dtype=float)
    else:
        if periodic:
            assert out.shape == (knots.shape[0] + 2,) and out.dtype == np.dtype('float')
        else:
            shape = 2*(degree + 2)
            if len(knots) - 2 * (degree + 1) > 0:
                uniques = (np.diff(knots[degree + 1:-degree - 1]) > tol).nonzero()
                shape += multiplicity * (1 + uniques[0].shape[0])
            assert out.shape == shape and out.dtype == np.dtype('float')

    elevate_knots_p(knots, degree, periodic, out, multiplicity, tol)
    return out

#==============================================================================
def quadrature_grid(breaks, quad_rule_x, quad_rule_w):
    """
    Compute the quadrature points and weights for performing integrals over
    each element (interval) of the 1D domain, given a certain Gaussian
    quadrature rule.

    An n-point Gaussian quadrature rule for the canonical interval :math:`[-1,+1]`
    and trivial weighting function :math:`\\omega(x)=1` is defined by the n abscissas
    :math:`x_i` and n weights :math:`w_i` that satisfy the following identity for
    polynomial functions :math:`f(x)` of degree :math:`2n-1` or less:

    .. math :: \\int_{-1}^{+1} f(x) dx = \\sum_{i=0}^{n-1} w_i f(x_i)

    Parameters
    ----------
    breaks : array_like of floats
        Coordinates of spline breakpoints.

    quad_rule_x : array_like of ints
        Coordinates of quadrature points on canonical interval [-1,1].

    quad_rule_w : array_like of ints
        Weights assigned to quadrature points on canonical interval [-1,1].

    Returns
    -------
    quad_x : 2D numpy.ndarray
        Abscissas of quadrature points on each element (interval) of the 1D
        domain. See notes below.

    quad_w : 2D numpy.ndarray
        Weights assigned to the quadrature points on each element (interval)
        of the 1D domain. See notes below.

    Notes
    -----
    Contents of 2D output arrays 'quad_x' and 'quad_w' are accessed with two
    indices (ie,iq) where:

      - ie is the global element index;
      - iq is the local index of a quadrature point within the element.

    """
    # Check that input arrays have correct size
    assert len(breaks)      >= 2
    assert len(quad_rule_x) == len(quad_rule_w)

    # Check that provided quadrature rule is defined on interval [-1,1]
    assert min(quad_rule_x) >= -1
    assert max(quad_rule_x) <= +1

    breaks = np.ascontiguousarray(breaks, dtype=float)

    quad_rule_x = np.ascontiguousarray( quad_rule_x, dtype=float )
    quad_rule_w = np.ascontiguousarray( quad_rule_w, dtype=float )

    out1 = np.zeros((len(breaks) - 1, len(quad_rule_x)))
    out2 = np.zeros_like(out1)
    
    quadrature_grid_p(breaks, quad_rule_x, quad_rule_w, out1, out2)

    return out1, out2

#==============================================================================
def basis_ders_on_quad_grid(knots, degree, quad_grid, nders, normalization, offset=0, out=None):
    """
    Evaluate B-Splines and their derivatives on the quadrature grid.

    If called with normalization='M', this uses M-splines instead of B-splines.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    quad_grid: ndarray
        2D array of shape (ne, nq). Coordinates of quadrature points of
        each element in 1D domain.
    nders : int
        Maximum derivative of interest.

    normalization : str
        Set to 'B' for B-splines, and 'M' for M-splines.
    
    offset : int, default=0
        Assumes that the quadrature grid starts from cell number offset.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    basis: ndarray
        Values of B-Splines and their derivatives at quadrature points in
        each element of 1D domain. Indices are
        . ie: global element         (0 <= ie <  ne    )
        . il: local basis function   (0 <= il <= degree)
        . id: derivative             (0 <= id <= nders )
        . iq: local quadrature point (0 <= iq <  nq    )

    Examples
    --------
    >>> knots = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1., 1.])
    >>> degree = 2
    >>> bk = breakpoints(knots, degree)
    >>> grid = np.array([np.linspace(bk[i], bk[i+1], 4, endpoint=False) for i in range(len(bk) - 1)])
    >>> basis_ders_on_quad_grid(knots, degree, grid, 0, "B")
    array([[[[0.5, 0.28125, 0.125, 0.03125]],
            [[0.5, 0.6875 , 0.75 , 0.6875 ]],
            [[0. , 0.03125, 0.125, 0.28125]]],
           [[[0.5, 0.28125, 0.125, 0.03125]],
            [[0.5, 0.6875 , 0.75 , 0.6875 ]],
            [[0. , 0.03125, 0.125, 0.28125]]]])
    """
    offset = int(offset)
    ne, nq = quad_grid.shape
    knots = np.ascontiguousarray(knots, dtype=float)
    quad_grid = np.ascontiguousarray(quad_grid, dtype=float)
    if out is None:
        out = np.zeros((ne, degree + 1, nders + 1, nq), dtype=float)
    else:
        assert out.shape == (ne, degree + 1, nders + 1, nq) and out.dtype == np.dtype('float')
    basis_ders_on_quad_grid_p(knots, degree, quad_grid, nders, normalization == 'M', offset, out)
    return out


#==============================================================================
def basis_integrals(knots, degree, out=None):
    """
    Return the integral of each B-spline basis function over the real line:

    .. math:: K[i] = \\int_{-\\infty}^{+\\infty} B_i(x) dx = (T[i+p+1]-T[i]) / (p+1).

    This array can be used to convert B-splines to M-splines, which have unit
    integral over the real line but no partition-of-unity property.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    K : numpy.ndarray
        Array with the integrals of each B-spline basis function.

    Notes
    -----
    For convenience, this function does not distinguish between periodic and
    non-periodic spaces, hence the length of the output array is always equal
    to (len(knots)-degree-1). In the periodic case the last (degree) values in
    the array are redundant, as they are a copy of the first (degree) values.
    """
    knots = np.ascontiguousarray(knots, dtype=float)
    if out is None:
        out = np.zeros(len(knots) - degree - 1, dtype=float)
    else:
        assert out.shape is (len(knots) - degree - 1,) and out.dtype == np.dtype('float')
    basis_integrals_p(knots, degree, out)
    return out

#==============================================================================
def cell_index(breaks, i_grid, tol=1e-15, out=None):
    """
    Computes in which cells a given array of locations belong.

    Locations close to a interior breakpoint will be assumed to be
    present twice in the grid, once of for each cell. Boundary breakpoints are snapped to the interior of the domain.

    Parameters
    ----------
    breaks : array_like
        Coordinates of breakpoints (= cell edges); given in increasing order and
        with no duplicates.

    i_grid : ndarray
        1D array of all of the points on which to evaluate the 
        basis functions. The points do not need to be sorted.

    tol : float, default=1e-15
        If the distance between a given point in ``i_grid`` and 
        a breakpoint is less than ``tol`` then it is considered 
        to be the breakpoint.
    
    out : array, optional
        If given, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.
    
    Returns
    -------
    cell_index: ndarray
        1D array of the same shape as ``i_grid``.
        ``cell_index[i]`` is the index of the cell in which
        ``i_grid[i]`` belong.
    """
    breaks = np.ascontiguousarray(breaks, dtype=float)
    i_grid = np.ascontiguousarray(i_grid, dtype=float)
    if out is None:
        out = np.zeros_like(i_grid, dtype=np.int64)
    else:
        assert out.shape == i_grid.shape and out.dtype == np.dtype('int64')
    status = cell_index_p(breaks, i_grid, tol, out)
    if status == -1:
        raise ValueError("Encountered a point that was outside of the domain")
    elif status == -2:
        raise ValueError("Loop too long in cell search: a point might be slightly outside of the domain")
    return out

#==============================================================================
def basis_ders_on_irregular_grid(knots, degree, i_grid, cell_index, nders, normalization, out=None):
    """
    Evaluate B-Splines and their derivatives on an irregular_grid.

    If called with normalization='M', this uses M-splines instead of B-splines.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    i_grid : ndarray
        1D array of all of the points on which to evaluate the 
        basis functions. The points do not need to be sorted
    
    cell_index : ndarray
        1D array of the same shape as ``i_grid``.
        ``cell_index[i]`` is the index of the cell in which
        ``i_grid[i]`` belong.

    nders : int
        Maximum derivative of interest.

    normalization : str
        Set to 'B' for B-splines, and 'M' for M-splines.

    out : array, optional
        If given, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    out: ndarray
        3D output array containing the values of B-Splines and their derivatives
        at each point in ``i_grid``. Indices are:
        . ie: location               (0 <= ie <  nx    )
        . il: local basis function   (0 <= il <= degree)
        . id: derivative             (0 <= id <= nders )
    """
    knots = np.ascontiguousarray(knots, dtype=float)
    i_grid = np.ascontiguousarray(i_grid, dtype=float)
    if out is None:
        nx = i_grid.shape[0]
        out = np.zeros((nx, degree + 1, nders + 1), dtype=float)
    else:
        assert out.shape == (nx, degree + 1, nders + 1) and out.dtype == np.dtype('float')
    basis_ders_on_irregular_grid_p(knots, degree, i_grid, cell_index, nders, normalization == 'M', out)
    return out

#==============================================================================
def _refinement_matrix_one_stage(t, p, knots):
    """
    Computes the refinement matrix corresponding to the insertion of a given knot.

    For more details see:

      [1] : Les Piegl , Wayne Tiller, The NURBS Book,
            https://doi.org/10.1007/978-3-642-97385-7. (Section 5.2)

    Parameters
    ----------
    t : float
      knot to be inserted.

    p: int
        spline degree.

    knots : array_like
        Knots sequence.

    Returns
    -------
    mat : np.array[:,:]
        h-refinement matrix.

    new_knots : array_like
        the Knots sequence with the inserted knot.
    """

    # ...
    def alpha_function(i, k, t, n, p, knots):
        if i <= k-p:
            alpha = 1.

        elif (k-p < i) and (i <= k):
            alpha = (t-knots[i]) / (knots[i+p] - knots[i])

        else:
            alpha = 0.

        return alpha
    # ...

    n = len(knots) - p - 1

    mat = np.zeros((n+1,n))

    left = find_span( knots, p, t )

    # ...
    j = 0
    alpha = alpha_function(j, left, t, n, p, knots)
    mat[j,j] = alpha

    for j in range(1, n):
        alpha = alpha_function(j, left, t, n, p, knots)
        mat[j,j]   = alpha
        mat[j,j-1] = 1.0 - alpha

    j = n
    alpha = alpha_function(j, left, t, n, p, knots)
    mat[j,j-1] = 1.0 - alpha
    # ...

    # ...
    new_knots = np.zeros(n+1+p+1)

    new_knots[:left+1] = knots[:left+1]
    new_knots[left+1] = t
    new_knots[left+2:] = knots[left+1:]
    # ...

    return mat, new_knots

#==============================================================================
def hrefinement_matrix(ts, p, knots):
    """
    Computes the refinement matrix corresponding to the insertion of a given list of knots.

    For more details see:

      [1] : Les Piegl , Wayne Tiller, The NURBS Book,
            https://doi.org/10.1007/978-3-642-97385-7. (Section 5.2)

    Parameters
    ----------
    ts: np.array
        array containing the knots to be inserted

    p: int
        spline degree.

    knots : array_like
        Knots sequence.

    Returns
    -------
    mat : np.array[:,:]
        h-refinement matrix

    Examples
    --------
    >>> import numpy as np
    >>> from psydac.core.bsplines import make_knots
    >>> from psydac.core.bsplines import hrefinement_matrix
    >>> grid = np.linspace(0.,1.,5)
    >>> degree = 2
    >>> knots = make_knots(grid, degree, periodic=False)
    >>> ts    = np.array([0.1, 0.2, 0.4, 0.5, 0.7, 0.8])
    >>> hrefinement_matrix(ts, p, knots)
    array([[1.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.6 , 0.4 , 0.  , 0.  , 0.  , 0.  ],
           [0.12, 0.72, 0.16, 0.  , 0.  , 0.  ],
           [0.  , 0.6 , 0.4 , 0.  , 0.  , 0.  ],
           [0.  , 0.2 , 0.8 , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.7 , 0.3 , 0.  , 0.  ],
           [0.  , 0.  , 0.5 , 0.5 , 0.  , 0.  ],
           [0.  , 0.  , 0.1 , 0.9 , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.6 , 0.4 , 0.  ],
           [0.  , 0.  , 0.  , 0.4 , 0.6 , 0.  ],
           [0.  , 0.  , 0.  , 0.  , 0.8 , 0.2 ],
           [0.  , 0.  , 0.  , 0.  , 0.  , 1.  ]])
    """

    m = len(ts)
    n = len(knots) - p - 1
    out = np.eye(n)

    for i in range(m):
        t = ts[i]
        mat, knots = _refinement_matrix_one_stage(t, p, knots)
        out = np.matmul(mat, out)

    return out
