# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

"""
Basic module that provides the means for evaluating the B-Splines basis
functions and their derivatives. In order to simplify automatic Fortran code
generation with Pyccel, no object-oriented features are employed.

References
----------
[1] L. Piegl and W. Tiller. The NURBS Book, 2nd ed.,
    Springer-Verlag Berlin Heidelberg GmbH, 1997.

[2] SELALIB, Semi-Lagrangian Library. http://selalib.gforge.inria.fr

"""
import numpy as np

from psydac.core.bsplines_pyccel import (find_span_p,
                                         find_spans_p,
                                         basis_funs_p,
                                         basis_funs_array_p,
                                         basis_funs_1st_der_p,
                                         basis_funs_all_ders_p,
                                         collocation_matrix_p,
                                         elevate_knots_p,
                                         basis_integrals_p)

__all__ = ['find_span',
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
           'basis_ders_on_quad_grid']


#==============================================================================
def find_span(knots: 'float[:]', degree: int, x: float):
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
    return find_span_p(knots, degree, x)

#==============================================================================
def find_spans(knots: 'float[:]', degree: int, x: 'float[:]', out=None):
    """Determine the knot span index at locations in x, given the B-Splines' knot
    sequence and polynomial degree. See Algorithm A2.1 in [1].

    For a degree p, the knot span index i identifies the indices [i-p:i] of all
    p+1 non-zero basis functions at a given location x.

    Parameters
    ----------
    knots: array_like of floats
        Knot sequence.

    degree: int
        Polynomial degree of the BSplines.

    x: array_like of floats
        Locations of interest

    out: array_like of ints or None
        Knot span index for each location in x.

    See Also
    --------
    psydac.core.bsplines.find_span : Determines the knot span at a location.
    """
    if out is None:
        out = np.zeros_like(x)
    find_spans_p(knots, degree, x, out)
    return out

#==============================================================================
def basis_funs(knots: 'float[:]', degree: int, x: float, span: int, out=None):
    """
    Compute the non-vanishing B-splines at location x, given the knot sequence,
    polynomial degree and knot span. See Algorithm A2.2 in [1].

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

    out : array_like of floats
        Values of p+1 non-vanishing B-Splines at location x.

    Notes
    -----
    The original Algorithm A2.2 in The NURBS Book [1] is here slightly improved
    by using 'left' and 'right' temporary arrays that are one element shorter.

    """
    if out is None:
        out = np.zeros(degree + 1)
    basis_funs_p(knots, degree, x, span, out)
    return out

#==============================================================================
def basis_funs_array(knots: 'float[:]', degree: int, span: 'int[:]', x: 'float[:]', out=None):
    """
    Compute the non-vanishing B-splines at locations in x, given the knot sequence,
    polynomial degree and knot span. See Algorithm A2.2 in [1].

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

    out : array_like of floats
        Values of p+1 non-vanishing B-Splines at all locations in x.

    Returns
    -------
    out : array_like of floats
        Values of p+1 non-vanishing B-Splines at all locations in x.
    """
    if out is None:
        out = np.zeros((x.shape, degree + 1))
    basis_funs_array_p(knots, degree, x, span,  out)
    return out

#==============================================================================
def basis_funs_1st_der( knots, degree, x, span, out=None):
    """
    Compute the first derivative of the non-vanishing B-splines at location x,
    given the knot sequence, polynomial degree and knot span.

    See function 's_bsplines_non_uniform__eval_deriv' in Selalib's source file
    'src/splines/sll_m_bsplines_non_uniform.F90'.

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

    out : numpy.ndarray or None

    Returns
    -------
    out : numpy.ndarray
        Derivatives of p+1 non-vanishing B-Splines at location x.

    """
    if out is None:
        out = np.empty(degree + 1)
    basis_funs_1st_der_p(knots, degree, x, span, out)
    return out

#==============================================================================
def basis_funs_all_ders(knots, degree, x, span, n, normalization='B', out=None):
    """
    Evaluate value and n derivatives at x of all basis functions with
    support in interval [x_{span-1}, x_{span}].

    If called with normalization='M', this uses M-splines instead of B-splines.

    ders[i,j] = (d/dx)^i B_k(x) with k=(span-degree+j),
                for 0 <= i <= n and 0 <= j <= degree+1.

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

    Returns
    -------
    out : numpy.ndarray (n+1,degree+1)
        2D array of n+1 (from 0-th to n-th) derivatives at x of all (degree+1)
        non-vanishing basis functions in given span.

    Notes
    -----
    The original Algorithm A2.3 in The NURBS Book [1] is here improved:
        - 'left' and 'right' arrays are 1 element shorter;
        - inverse of knot differences are saved to avoid unnecessary divisions;
        - innermost loops are replaced with vector operations on slices.

    """
    if out is None:
        out = np.zeros((n + 1, degree + 1))
    basis_funs_all_ders_p(knots, degree, x, span, n, normalization == 'M', out)
    return out

#==============================================================================
def collocation_matrix(knots, degree, periodic, normalization, xgrid, out=None):
    """
    Compute the collocation matrix $C_ij = B_j(x_i)$, which contains the
    values of each B-spline basis function $B_j$ at all locations $x_i$.

    If called with normalization='M', this uses M-splines instead of B-splines.

    Parameters
    ----------
    knots : 1D array_like
        Knots sequence.

    degree : int
        Polynomial degree of spline space.

    periodic : bool
        True if domain is periodic, False otherwise.

    normalization : str
        Set to 'B' for B-splines, and 'M' for M-splines.

    xgrid : 1D array_like
        Evaluation points.

    out : 2D numpy.ndarray or None
        Collocation matrix: values of all basis functions on each point in xgrid.

    """
    if out is None:
        nb = len(knots) - degree - 1
        if periodic:
            nb -= degree

        out = np.zeros((xgrid.shape[0], nb))

    bool_normalization = normalization == "M"

    collocation_matrix_p(knots, degree, periodic, bool_normalization, xgrid, out)

    return out

#==============================================================================
def histopolation_matrix(knots, degree, periodic, normalization, xgrid, check_boundary=True, out=None):
    r"""
    Compute the histopolation matrix $H_{ij} = \int_{x_i}^{x_{i+1}} B_j(x) dx$,
    which contains the integrals of each B-spline basis function $B_j$ between
    two successive grid points.

    If called with normalization='M', this uses M-splines instead of B-splines.

    Parameters
    ----------
    knots : 1D array_like
        Knots sequence.

    degree : int
        Polynomial degree of spline space.

    periodic : bool
        True if domain is periodic, False otherwise.

    normalization : str
        Set to 'B' for B-splines, and 'M' for M-splines.

    xgrid : 1D array_like
        Grid points.

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

    elevated_knots = elevate_knots(knots, degree, periodic)

    normalization = normalization == "M"

    if out is None:
        if periodic:
            out = np.zeros((len(xgrid), len(knots) - 2 * degree -1))
        else:
            out = np.zeros((len(xgrid) - 1, len(elevated_knots) - (degree + 1) - 1 - 1))

    #histopolation_matrix_p(knots, degree, periodic, normalization, xgrid, check_boundary, elevated_knots, out)

    return out

#==============================================================================
def breakpoints( knots, degree ,tol=1e-15):
    """
    Determine breakpoints' coordinates.

    Parameters
    ----------
    knots : 1D array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    tol: float
        If the distance between two knots is less than tol, we assume 
        that they are repeated knots which correspond to the same break point.

    Returns
    -------
    breaks : numpy.ndarray (1D)
        Abscissas of all breakpoints.

    """
    knots = np.array(knots)
    diff  = np.append(True, abs(np.diff(knots[degree:-degree]))>tol)
    return knots[degree:-degree][diff]

#==============================================================================
def greville( knots, degree, periodic ):
    """
    Compute coordinates of all Greville points.

    Parameters
    ----------
    knots : 1D array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    periodic : bool
        True if domain is periodic, False otherwise.

    Returns
    -------
    xg : numpy.ndarray (1D)
        Abscissas of all Greville points.

    """
    T = knots
    p = degree
    n = len(T)-2*p-1 if periodic else len(T)-p-1

    # Compute greville abscissas as average of p consecutive knot values
    xg = np.array([sum(T[i:i+p])/p for i in range(1,1+n)])

    # Domain boundaries
    a = T[p]
    b = T[-1-p]

    # If needed apply periodic boundary conditions, then sort array
    if periodic:
        xg = (xg-a) % (b-a) + a
        xg = xg[np.argsort(xg)]

    # Make sure roundoff errors don't push Greville points outside domain
    xg[ 0] = max(xg[ 0], a)
    xg[-1] = min(xg[-1], b)

    return xg

#===============================================================================
def elements_spans( knots, degree ):
    """
    Compute the index of the last non-vanishing spline on each grid element
    (cell). The length of the returned array is the number of cells.

    Parameters
    ----------
    knots : 1D array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

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
    breaks = breakpoints( knots, degree )
    nk     = len(knots)
    ne     = len(breaks)-1
    spans  = np.zeros( ne, dtype=int )

    ie = 0
    for ik in range( degree, nk-degree ):
        if knots[ik+1]-knots[ik]>=1e-15:
            spans[ie] = ik
            ie += 1
        if ie == ne:
            break

    return spans

#===============================================================================
def make_knots( breaks, degree, periodic, multiplicity=1 ):
    """
    Create spline knots from breakpoints, with appropriate boundary conditions.
    Let p be spline degree. If domain is periodic, knot sequence is extended
    by periodicity so that first p basis functions are identical to last p.
    Otherwise, knot sequence is clamped (i.e. endpoints are repeated p times).

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

    Result
    ------
    T : numpy.ndarray (1D)
        Coordinates of spline knots.

    """
    # Type checking
    assert isinstance( degree  , int  )
    assert isinstance( periodic, bool )

    # Consistency checks
    assert len(breaks) > 1
    assert all( np.diff(breaks) > 0 )
    assert degree > 0
    assert 1 <= multiplicity and multiplicity <= degree + 1

    if periodic:
        assert len(breaks) > degree

    p = degree
    T = np.zeros( multiplicity*len(breaks[1:-1])+2+2*p )

    T[p+1:-p-1] = np.repeat(breaks[1:-1], multiplicity)
    T[p]        = breaks[ 0]
    T[-p-1]     = breaks[-1]

    if periodic:
        period = breaks[-1]-breaks[0]
        T[0:p] = [xi-period for xi in breaks[-p-1:-1 ]]
        T[-p:] = [xi+period for xi in breaks[   1:p+1]]
    else:
        T[0:p+1] = breaks[ 0]
        T[-p-1:] = breaks[-1]

    return T

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

    Returns
    -------
    new_knots : 1D numpy.ndarray
        Knots sequence of spline space of degree p+1.

    """

    if out is None:
        if periodic:
            out = np.zeros(knots.shape[0] + 2)
        else:
            shape = 2*(degree + 2)
            if len(knots) - 2 * (degree + 1) > 0:
                uniques = np.asarray(np.diff(knots[degree + 1:-degree - 1]) > tol).nonzero()
                shape += multiplicity * (1 + uniques[0].shape[0])
            out = np.zeros(shape)
    elevate_knots_p(knots, degree, periodic, out, multiplicity, tol)

    return out


#==============================================================================
def quadrature_grid( breaks, quad_rule_x, quad_rule_w ):
    r"""
    Compute the quadrature points and weights for performing integrals over
    each element (interval) of the 1D domain, given a certain Gaussian
    quadrature rule.

    An n-point Gaussian quadrature rule for the canonical interval $[-1,+1]$
    and trivial weighting function $\omega(x)=1$ is defined by the n abscissas
    $x_i$ and n weights $w_i$ that satisfy the following identity for
    polynomial functions $f(x)$ of degree $2n-1$ or less:

    $\int_{-1}^{+1} f(x) dx = \sum_{i=0}^{n-1} w_i f(x_i)$.

    Parameters
    ----------
    breaks : 1D array_like
        Coordinates of spline breakpoints.

    quad_rule_x : 1D array_like
        Coordinates of quadrature points on canonical interval [-1,1].

    quad_rule_w : 1D array_like
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
      . ie is the global element index;
      . iq is the local index of a quadrature point within the element.

    """
    # Check that input arrays have correct size
    assert len(breaks)      >= 2
    assert len(quad_rule_x) == len(quad_rule_w)

    # Check that provided quadrature rule is defined on interval [-1,1]
    assert min(quad_rule_x) >= -1
    assert max(quad_rule_x) <= +1

    quad_rule_x = np.asarray( quad_rule_x )
    quad_rule_w = np.asarray( quad_rule_w )

    ne     = len(breaks)-1
    nq     = len(quad_rule_x)
    quad_x = np.zeros( (ne,nq) )
    quad_w = np.zeros( (ne,nq) )

    # Compute location and weight of quadrature points from basic rule
    for ie,(a,b) in enumerate(zip(breaks[:-1],breaks[1:])):
        c0 = 0.5*(a+b)
        c1 = 0.5*(b-a)
        quad_x[ie,:] = c1*quad_rule_x[:] + c0
        quad_w[ie,:] = c1*quad_rule_w[:]

    return quad_x, quad_w

#==============================================================================
def basis_ders_on_quad_grid(knots, degree, quad_grid, nders, normalization):
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
        each element in 1D domain, which can be given by quadrature_grid()
        or chosen arbitrarily.

    nders : int
        Maximum derivative of interest.

    normalization : str
        Set to 'B' for B-splines, and 'M' for M-splines.

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
    >>> # Valid grid
    >>> grid = np.array([np.linspace(bk[i], bk[i+1], 4, endpoint=False) for i in range(len(bk) - 1)])
    >>> # Also a valid grid (doesn't start on the left boundary of the domain)
    >>> grid_2 = grid[2:, :]
    >>> basis_ders_on_quad_grid(knots, degree, grid, 0, "B")
    array([[[[0.5, 0.28125, 0.125, 0.03125]],
            [[0.5, 0.6875 , 0.75 , 0.6875 ]],
            [[0. , 0.03125, 0.125, 0.28125]]],
           [[[0.5, 0.28125, 0.125, 0.03125]],
            [[0.5, 0.6875 , 0.75 , 0.6875 ]],
            [[0. , 0.03125, 0.125, 0.28125]]]])
    >>> basis_ders_on_quad_grid(knots, degree, grid_2, 0, "B")
    array([[[[0.5, 0.28125, 0.125, 0.03125]],
            [[0.5, 0.6875 , 0.75 , 0.6875 ]],
            [[0. , 0.03125, 0.125, 0.28125]]]])

    """
    ne, nq = quad_grid.shape
    basis = np.zeros((ne, degree+1, nders+1, nq))

    if normalization == 'M':
        scaling = 1. / basis_integrals(knots, degree)
    spans = elements_spans(knots, degree)

    assert ne <= len(spans)
    # Test to see if the grid doesn't start at 0
    offset = 0
    span_exact = find_span(knots, degree, quad_grid[0, 0])
    if span_exact != spans[0]:
        offset = span_exact - spans[0]

    for ie in range(ne):
        xx = quad_grid[ie, :]
        span = spans[offset + ie]
        for iq, xq in enumerate(xx):
            ders = basis_funs_all_ders(knots, degree, xq, span, nders)
            if normalization == 'M':
                ders *= scaling[span - degree:span + 1]
            basis[ie, :, :, iq] = ders.transpose()
    return basis

#==============================================================================
def basis_integrals(knots, degree, out=None):
    r"""
    Return the integral of each B-spline basis function over the real line:

    K[i] := \int_{-\infty}^{+\infty} B[i](x) dx = (T[i+p+1]-T[i]) / (p+1).

    This array can be used to convert B-splines to M-splines, which have unit
    integral over the real line but no partition-of-unity property.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    Returns
    -------
    K : 1D numpy.ndarray
        Array with the integrals of each B-spline basis function.

    Notes
    -----
    For convenience, this function does not distinguish between periodic and
    non-periodic spaces, hence the length of the output array is always equal
    to (len(knots)-degree-1). In the periodic case the last (degree) values in
    the array are redundant, as they are a copy of the first (degree) values.

    """
    if out is None:
        out = np.zeros(len(knots) - degree - 1)
    return basis_integrals_p(knots, degree, out)
