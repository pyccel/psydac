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

__all__ = ['find_span',
           'basis_funs',
           'basis_funs_1st_der',
           'basis_funs_all_ders',
           'collocation_matrix',
           'breakpoints',
           'greville',
           'elements_spans',
           'make_knots',
           'quadrature_grid',
           'basis_ders_on_quad_grid']

#==============================================================================
def find_span( knots, degree, x ):
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
    # Knot index at left/right boundary
    low  = degree
    high = len(knots)-1-degree

    # Check if point is exactly on left/right boundary, or outside domain
    if x <= knots[low ]: return low
    if x >= knots[high]: return high-1

    # Perform binary search
    span = (low+high)//2
    while x < knots[span] or x >= knots[span+1]:
        if x < knots[span]:
           high = span
        else:
           low  = span
        span = (low+high)//2

    return span

#==============================================================================
def basis_funs( knots, degree, x, span ):
    """
    Compute the non-vanishing B-splines at location x, given the knot sequence,
    polynomial degree and knot span. See Algorithm A2.2 in [1].

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

    Results
    -------
    values : numpy.ndarray
        Values of p+1 non-vanishing B-Splines at location x.

    Notes
    -----
    The original Algorithm A2.2 in The NURBS Book [1] is here slightly improved
    by using 'left' and 'right' temporary arrays that are one element shorter.

    """
    left   = np.empty( degree  , dtype=float )
    right  = np.empty( degree  , dtype=float )
    values = np.empty( degree+1, dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

    return values

#==============================================================================
def basis_funs_1st_der( knots, degree, x, span ):
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

    Results
    -------
    ders : numpy.ndarray
        Derivatives of p+1 non-vanishing B-Splines at location x.

    """
    # Compute nonzero basis functions and knot differences for splines
    # up to degree deg-1
    values = basis_funs( knots, degree-1, x, span )

    # Compute derivatives at x using formula based on difference of splines of
    # degree deg-1
    # -------
    # j = 0
    ders  = np.empty( degree+1, dtype=float )
    saved = degree * values[0] / (knots[span+1]-knots[span+1-degree])
    ders[0] = -saved
    # j = 1,...,degree-1
    for j in range(1,degree):
        temp    = saved
        saved   = degree * values[j] / (knots[span+j+1]-knots[span+j+1-degree])
        ders[j] = temp - saved
    # j = degree
    ders[degree] = saved

    return ders

#==============================================================================
def basis_funs_all_ders( knots, degree, x, span, n ):
    """
    Evaluate value and n derivatives at x of all basis functions with
    support in interval [x_{span-1}, x_{span}].

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

    Results
    -------
    ders : numpy.ndarray (n+1,degree+1)
        2D array of n+1 (from 0-th to n-th) derivatives at x of all (degree+1)
        non-vanishing basis functions in given span.

    Notes
    -----
    The original Algorithm A2.3 in The NURBS Book [1] is here improved:
        - 'left' and 'right' arrays are 1 element shorter;
        - inverse of knot differences are saved to avoid unnecessary divisions;
        - innermost loops are replaced with vector operations on slices.

    """
    left  = np.empty( degree )
    right = np.empty( degree )
    ndu   = np.empty( (degree+1, degree+1) )
    a     = np.empty( (       2, degree+1) )
    ders  = np.zeros( (     n+1, degree+1) ) # output array

    # Number of derivatives that need to be effectively computed
    # Derivatives higher than degree are = 0.
    ne = min( n, degree )

    # Compute nonzero basis functions and knot differences for splines
    # up to degree, which are needed to compute derivatives.
    # Store values in 2D temporary array 'ndu' (square matrix).
    ndu[0,0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            # compute inverse of knot differences and save them into lower triangular part of ndu
            ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
            # compute basis functions and save them into upper triangular part of ndu
            temp       = ndu[r,j] * ndu[j+1,r]
            ndu[r,j+1] = saved + right[r] * temp
            saved      = left[j-r] * temp
        ndu[j+1,j+1] = saved

    # Compute derivatives in 2D output array 'ders'
    ders[0,:] = ndu[:,degree]
    for r in range(0,degree+1):
        s1 = 0
        s2 = 1
        a[0,0] = 1.0
        for k in range(1,ne+1):
            d  = 0.0
            rk = r-k
            pk = degree-k
            if r >= k:
               a[s2,0] = a[s1,0] * ndu[pk+1,rk]
               d = a[s2,0] * ndu[rk,pk]
            j1 = 1   if (rk  > -1 ) else -rk
            j2 = k-1 if (r-1 <= pk) else degree-r
            a[s2,j1:j2+1] = (a[s1,j1:j2+1] - a[s1,j1-1:j2]) * ndu[pk+1,rk+j1:rk+j2+1]
            d += np.dot( a[s2,j1:j2+1], ndu[rk+j1:rk+j2+1,pk] )
            if r <= pk:
               a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
               d += a[s2,k] * ndu[r,pk]
            ders[k,r] = d
            j  = s1
            s1 = s2
            s2 = j

    # Multiply derivatives by correct factors
    r = degree
    for k in range(1,ne+1):
        ders[k,:] = ders[k,:] * r
        r = r * (degree-k)

    return ders

#==============================================================================
def collocation_matrix( knots, degree, xgrid, periodic ):
    """
    Compute the collocation matrix $C_ij = B_j(x_i)$, which contains the
    values of each B-spline basis function $B_j$ at all locations $x_i$.

    Parameters
    ----------
    knots : 1D array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    xgrid : 1D array_like
        Evaluation points.

    periodic : bool
        True if domain is periodic, False otherwise.

    Returns
    -------
    mat : 2D numpy.ndarray
        Collocation matrix: values of all basis functions on each point in xgrid.

    """
    # Number of basis functions (in periodic case remove degree repeated elements)
    nb = len(knots)-degree-1
    if periodic:
        nb -= degree

    # Number of evaluation points
    nx = len(xgrid)

    # Collocation matrix as 2D Numpy array (dense storage)
    mat = np.zeros( (nx,nb) )

    # Indexing of basis functions (periodic or not) for a given span
    if periodic:
        js = lambda span: [(span-degree+s) % nb for s in range( degree+1 )]
    else:
        js = lambda span: slice( span-degree, span+1 )

    # Fill in non-zero matrix values
    for i,x in enumerate( xgrid ):
        span  =  find_span( knots, degree, x )
        basis = basis_funs( knots, degree, x, span )
        mat[i,js(span)] = basis

    return mat

#==============================================================================
def breakpoints( knots, degree ):
    """
    Determine breakpoints' coordinates.

    Parameters
    ----------
    knots : 1D array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    Returns
    -------
    breaks : numpy.ndarray (1D)
        Abscissas of all breakpoints.

    """
    return np.unique( knots[degree:-degree] )

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
    s = 1+p//2       if periodic else 1
    n = len(T)-2*p-1 if periodic else len(T)-p-1

    # Compute greville abscissas as average of p consecutive knot values
    xg = np.around( [sum(T[i:i+p])/p for i in range(s,s+n)], decimals=15 )

    # If needed apply periodic boundary conditions
    if periodic:
        a  = T[ p]
        b  = T[-p]
        xg = np.around( (xg-a)%(b-a)+a, decimals=15 )

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
        if knots[ik] != knots[ik+1]:
            spans[ie] = ik
            ie += 1
        if ie == ne:
            break

    return spans

#===============================================================================
def make_knots( breaks, degree, periodic ):
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
    if periodic:
        assert len(breaks) > degree

    p = degree
    T = np.zeros( len(breaks)+2*p )
    T[p:-p] = breaks

    if periodic:
        period = breaks[-1]-breaks[0]
        T[0:p] = [xi-period for xi in breaks[-p-1:-1 ]]
        T[-p:] = [xi+period for xi in breaks[   1:p+1]]
    else:
        T[0:p] = breaks[ 0]
        T[-p:] = breaks[-1]

    return T

#==============================================================================
def quadrature_grid( breaks, quad_rule_x, quad_rule_w ):
    """
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
def basis_ders_on_quad_grid( knots, degree, quad_grid, nders, normalize=False ):
    """
    Evaluate B-Splines and their derivatives on the quadrature grid.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    quad_grid: 2D numpy.ndarray (ne,nq)
        Coordinates of quadrature points of each element in 1D domain,
        given by quadrature_grid() function.

    nders : int
        Maximum derivative of interest.

    Returns
    -------
    basis: 4D numpy.ndarray
        Values of B-Splines and their derivatives at quadrature points in
        each element of 1D domain. Indices are
        . ie: global element         (0 <= ie <  ne    )
        . il: local basis function   (0 <= il <= degree)
        . id: derivative             (0 <= id <= nders )
        . iq: local quadrature point (0 <= iq <  nq    )

    """
    # TODO: add example to docstring
    # TODO: check if it is safe to compute span only once for each element

    ne,nq = quad_grid.shape
    basis = np.zeros( (ne,degree+1,nders+1,nq) )

    for ie in range(ne):
        xx = quad_grid[ie,:]
        for iq,xq in enumerate(xx):
            span = find_span( knots, degree, xq )
            ders = basis_funs_all_ders( knots, degree, xq, span, nders )
            basis[ie,:,:,iq] = ders.transpose()

    if normalize:
        x = scaling_matrix(degree, ne+degree, knots)
        basis *= x[0]

    return basis

#==============================================================================
def scaling_matrix(p, n, T):
    """Returns the scaling array for M-splines.
    It is an array whose elements are (p+1)/(T[i+p+1]-T[i])


    """

    x = np.zeros(n)
    for i in range(0, n):
        x[i] = (p+1)/(T[i+p+1]-T[i])
    return x
