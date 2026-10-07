#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import cunumpy as xp
import pytest

from feectools.utilities.quadratures import gauss_legendre
from feectools.core.bsplines import (find_span,
                                  basis_funs,
                                  basis_funs_1st_der,
                                  basis_funs_all_ders,
                                  collocation_matrix,
                                  histopolation_matrix,
                                  breakpoints,
                                  greville,
                                  elements_spans,
                                  make_knots,
                                  elevate_knots,
                                  quadrature_grid,
                                  basis_integrals,
                                  basis_ders_on_quad_grid)

# The pytest-xdist plugin requires that every worker sees the same parameters
# in the unit tests. As in this module random parameters are used, here we set
# the same random seed for all workers.
xp.random.seed(0)

###############################################################################
# "True" Functions
###############################################################################

def find_span_true( knots, degree, x ):
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
def basis_funs_true( knots, degree, x, span ):
    left   = xp.empty( degree  , dtype=float )
    right  = xp.empty( degree  , dtype=float )
    values = xp.empty( degree+1, dtype=float )

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
def basis_funs_1st_der_true( knots, degree, x, span ):
    # Compute nonzero basis functions and knot differences for splines
    # up to degree deg-1
    values = basis_funs_true( knots, degree-1, x, span )

    # Compute derivatives at x using formula based on difference of splines of
    # degree deg-1
    # -------
    # j = 0
    ders  = xp.empty( degree+1, dtype=float )
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
def basis_funs_all_ders_true(knots, degree, x, span, n, normalization='B'):
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
    left  = xp.empty( degree )
    right = xp.empty( degree )
    ndu   = xp.empty( (degree+1, degree+1) )
    a     = xp.empty( (       2, degree+1) )
    ders  = xp.zeros( (     n+1, degree+1) ) # output array

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
            d += xp.dot( a[s2,j1:j2+1], ndu[rk+j1:rk+j2+1,pk] )
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

    # Normalization to get M-Splines
    if normalization == 'M':
        scaling = xp.asarray([(degree + 1) / (knots[i + degree + 1] - knots[i])
                              for i in range(span - degree, span + 1)])
        ders *= scaling
    return ders

#==============================================================================
def collocation_matrix_true(knots, degree, periodic, normalization, xgrid):
    # Number of basis functions (in periodic case remove degree repeated elements)
    nb = len(knots)-degree-1
    if periodic:
        nb -= degree

    # Number of evaluation points
    nx = len(xgrid)

    # Collocation matrix as 2D Numpy array (dense storage)
    mat = xp.zeros( (nx,nb) )

    # Indexing of basis functions (periodic or not) for a given span
    if periodic:
        js = lambda span: [(span-degree+s) % nb for s in range( degree+1 )]
    else:
        js = lambda span: slice( span-degree, span+1 )

    # Rescaling of B-splines, to get M-splines if needed
    if normalization == 'B':
        normalize = lambda basis, span: basis
    elif normalization == 'M':
        scaling = 1 / basis_integrals_true(knots, degree)
        normalize = lambda basis, span: basis * scaling[span-degree: span+1]

    # Fill in non-zero matrix values
    for i,x in enumerate( xgrid ):
        span  =  find_span_true( knots, degree, x )
        basis = basis_funs_true( knots, degree, x, span )
        values = normalize(basis, span)
        if periodic:
            # NumPy and CuPy differ for indexed assignment with repeated
            # indices (which occurs when nb <= degree).  The production
            # kernel assigns in loop order, so make the reference explicit.
            for j, value in zip(js(span), values):
                mat[i, j] = value
        else:
            mat[i, js(span)] = values

    # Mitigate round-off errors
    mat[abs(mat) < 1e-14] = 0.0

    return mat

#==============================================================================
def histopolation_matrix_true(knots, degree, periodic, normalization, xgrid):
    # Check that knots are ordered (but allow repeated knots)
    if not xp.all(xp.diff(knots) >= 0):
        raise ValueError("Cannot accept knot sequence: {}".format(knots))

    # Check that spline degree is non-negative integer
    if not isinstance(degree, (int, xp.integer)):
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
    if not xp.all(xp.diff(xgrid) > 0):
        raise ValueError("Grid points must be ordered, with no repetitions: {}".format(xgrid))

    # Number of basis functions (in periodic case remove degree repeated elements)
    nb = len(knots)-degree-1
    if periodic:
        nb -= degree

    # Number of evaluation points
    nx = len(xgrid)

    # In periodic case, make sure that evaluation points include domain boundaries
    # TODO: only do this if the user asks for it!
    if periodic:
        xmin  = knots[degree]
        xmax  = knots[-1-degree]
        if xgrid[0] > xmin:
            xgrid = [xmin, *xgrid]
        if xgrid[-1] < xmax:
            xgrid = [*xgrid, xmax]

    # B-splines of degree p+1: basis[i,j] := Bj(xi)
    #
    # NOTES:
    #  . cannot use M-splines in analytical formula for histopolation matrix
    #  . always use non-periodic splines to avoid circulant matrix structure
    C = collocation_matrix_true(
        knots    = elevate_knots_true(knots, degree, periodic),
        degree   = degree + 1,
        periodic = False,
        normalization = 'B',
        xgrid = xgrid
    )

    # Rescaling of M-splines, to get B-splines if needed
    if normalization == 'M':
        normalize = lambda bi, j: bi
    elif normalization == 'B':
        scaling = basis_integrals_true(knots, degree)
        normalize = lambda bi, j: bi * scaling[j]

    # Compute span for each row (index of last non-zero basis function)
    # TODO: would be better to have this ready beforehand
    # TODO: use tolerance instead of comparing against zero
    spans = [int((row != 0).argmax()) + (degree+1) for row in C]

    # Compute histopolation matrix from collocation matrix of higher degree
    m = C.shape[0] - 1
    n = C.shape[1] - 1
    H = xp.zeros((m, n))
    for i in range(m):
        # Indices of first/last non-zero elements in row of collocation matrix
        jstart = spans[i] - (degree+1)
        jend   = min(spans[i+1], n)
        # Compute non-zero values of histopolation matrix
        for j in range(1+jstart, jend+1):
            s = C[i, 0:j].sum() - C[i+1, 0:j].sum()
            H[i, j-1] = normalize(s, j-1)

    # Mitigate round-off errors
    H[abs(H) < 1e-14] = 0.0
    # Non periodic case: stop here
    if not periodic:
        return H

    # Periodic case: wrap around histopolation matrix
    #  1. identify repeated basis functions (sum columns)
    #  2. identify split interval (sum rows)
    Hp = xp.zeros((nx, nb))
    for i in range(m):
        for j in range(n):
            Hp[i % nx, j % nb] += H[i, j]

    return Hp

#==============================================================================
def breakpoints_true( knots, degree ,tol=1e-15):
    knots = xp.array(knots)
    diff  = xp.append(True, abs(xp.diff(knots[degree:-degree]))>tol)
    return knots[degree:-degree][diff]

#==============================================================================
def greville_true( knots, degree, periodic ):
    T = knots
    p = degree
    n = len(T)-2*p-1 if periodic else len(T)-p-1

    # Compute greville abscissas as average of p consecutive knot values
    xg = xp.array([sum(T[i:i+p])/p for i in range(1,1+n)])

    # Domain boundaries
    a = T[p]
    b = T[-1-p]

    # If needed apply periodic boundary conditions, then sort array
    if periodic:
        xg = (xg-a) % (b-a) + a
        xg = xg[xp.argsort(xg)]

    # Make sure roundoff errors don't push Greville points outside domain
    xg[ 0] = max(xg[ 0], a)
    xg[-1] = min(xg[-1], b)

    return xg

#===============================================================================
def elements_spans_true( knots, degree ):
    breaks = breakpoints_true( knots, degree )
    nk     = len(knots)
    ne     = len(breaks)-1
    spans  = xp.zeros( ne, dtype=int )

    ie = 0
    for ik in range( degree, nk-degree ):
        if knots[ik+1]-knots[ik]>=1e-15:
            spans[ie] = ik
            ie += 1
        if ie == ne:
            break

    return spans

#===============================================================================
def make_knots_true( breaks, degree, periodic, multiplicity=1 ):
    # Type checking
    assert isinstance( degree  , int  )
    assert isinstance( periodic, bool )

    # Consistency checks
    assert len(breaks) > 1
    assert all( xp.diff(breaks) > 0 )
    assert degree > 0
    assert 1 <= multiplicity and multiplicity <= degree + 1

    if periodic:
        assert len(breaks) > degree
        
    T = xp.zeros(multiplicity * len(breaks[1:-1]) + 2 + 2 * degree)
    ncells = len(breaks) - 1
    
    for i in range(0, ncells+1):
        T[degree + 1 + (i-1) * multiplicity  :degree + 1 + i * multiplicity ] = breaks[i]
    
    len_out = len(T)
    
    if periodic:
        period = breaks[-1]-breaks[0]

        T[: degree + 1 - multiplicity] = T[len_out - 2 * (degree + 1 )+ multiplicity: len_out - (degree + 1)] - period
        T[len_out - (degree + 1 - multiplicity) :] = T[degree + 1:2*(degree + 1)- multiplicity] + period

    else:
        T[0:degree + 1 - multiplicity] = breaks[0]
        T[len_out - degree - 1 + multiplicity:] = breaks[-1]

    return T

#==============================================================================
def elevate_knots_true(knots, degree, periodic, multiplicity=1, tol=1e-15):
    knots = xp.array(knots)

    if periodic:
        T, p = knots, degree
        period = T[len(knots) -1 - p] - T[p]
        left   = [T[len(knots) -2 - 2 * p + multiplicity-1] - period]
        right  = [T[2 * p + 2 - multiplicity] + period]
    else:
        left  = [knots[0],*knots[:degree+1]]
        right = [knots[-1],*knots[-degree-1:]]

        diff   = xp.append(True, xp.diff(knots[degree+1:-degree-1])>tol)
        if len(knots[degree+1:-degree-1])>0:
            unique = knots[degree+1:-degree-1][diff]
            knots  = xp.repeat(unique, multiplicity)
        else:
            knots = knots[degree+1:-degree-1]

    return xp.array([*left, *knots, *right])

#==============================================================================
def quadrature_grid_true(breaks, quad_rule_x, quad_rule_w):
    # Check that input arrays have correct size
    assert len(breaks)      >= 2
    assert len(quad_rule_x) == len(quad_rule_w)

    # Check that provided quadrature rule is defined on interval [-1, 1]
    assert min(quad_rule_x) >= -1
    assert max(quad_rule_x) <= +1

    quad_rule_x = xp.asarray(quad_rule_x)
    quad_rule_w = xp.asarray(quad_rule_w)

    ne     = len(breaks) - 1
    nq     = len(quad_rule_x)
    quad_x = xp.zeros((ne, nq))
    quad_w = xp.zeros((ne, nq))

    # Compute location and weight of quadrature points from basic rule
    for ie, (a, b) in enumerate(zip(breaks[:-1], breaks[1:])):
        c0 = 0.5 * (a + b)
        c1 = 0.5 * (b - a)
        quad_x[ie, :] = c1 * quad_rule_x[:] + c0
        quad_w[ie, :] = c1 * quad_rule_w[:]

    return quad_x, quad_w

#==============================================================================
def basis_ders_on_quad_grid_true(knots, degree, quad_grid, nders, normalization):
    ne,nq = quad_grid.shape
    basis = xp.zeros((ne, degree+1, nders+1, nq))

    if normalization == 'M':
        scaling = 1. / basis_integrals_true(knots, degree)

    for ie in range(ne):
        xx = quad_grid[ie, :]
        for iq, xq in enumerate(xx):
            span = find_span_true(knots, degree, xq)
            ders = basis_funs_all_ders_true(knots, degree, xq, span, nders)
            if normalization == 'M':
                ders *= scaling[None, span-degree:span+1]
            basis[ie, :, :, iq] = ders.transpose()

    return basis

#==============================================================================
def basis_integrals_true(knots, degree):
    T = knots
    p = degree
    n = len(T)-p-1
    K = xp.array([(T[i+p+1] - T[i]) / (p + 1) for i in range(n)])

    return K


###############################################################################
# Tests
###############################################################################
# Tolerance for testing float equality
RTOL = 1e-11
ATOL = 1e-11


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
@pytest.mark.parametrize('x', (xp.random.random(), xp.random.random(), xp.random.random()))

def test_find_span(knots, degree, x):
    expected = find_span_true(knots, degree, x)
    out = find_span(knots, degree, x)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
@pytest.mark.parametrize('x', (xp.random.random(), xp.random.random(), xp.random.random()))
def test_basis_funs(knots, degree, x):
    span = find_span(knots, degree, x)
    expected = basis_funs_true(knots, degree, x, span)
    out = basis_funs(knots, degree, x, span)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
@pytest.mark.parametrize('x', (xp.random.random(), xp.random.random(), xp.random.random()))
def test_basis_funs_1st_der(knots, degree, x):
    span = find_span(knots, degree, x)
    expected = basis_funs_1st_der_true(knots, degree, x, span)
    out = basis_funs_1st_der(knots, degree, x, span)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
@pytest.mark.parametrize('x', (xp.random.random(), xp.random.random(), xp.random.random()))
@pytest.mark.parametrize('n', (2, 3, 4, 5))
@pytest.mark.parametrize('normalization', ('B', 'M'))
def test_basis_funs_all_ders(knots, degree, x, n, normalization):
    span = find_span(knots, degree, x)
    expected = basis_funs_all_ders_true(knots, degree, x, span, n, normalization)
    out = basis_funs_all_ders(knots, degree, x, span, n, normalization)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
@pytest.mark.parametrize('periodic', (True, False))
@pytest.mark.parametrize('normalization', ('B', 'M'))
@pytest.mark.parametrize('xgrid', (xp.random.random(10), xp.random.random(15)))
def test_collocation_matrix(knots, degree, periodic, normalization, xgrid):
    expected = collocation_matrix_true(knots, degree, periodic, normalization, xgrid)
    out = collocation_matrix(knots, degree, periodic, normalization, xgrid)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('normalization', ('B', 'M'))
@pytest.mark.parametrize('xgrid', (xp.random.random(10), xp.random.random(15)))
def test_histopolation_matrix(knots, degree, periodic, normalization, xgrid):
    xgrid = xp.sort(xp.unique(xgrid))
    expected = histopolation_matrix_true(knots, degree, periodic, normalization, xgrid)
    out = histopolation_matrix(knots, degree, periodic, normalization, xgrid)
    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
def test_breakpoints(knots, degree):
    expected = breakpoints_true(knots, degree)
    out = breakpoints(knots, degree)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)

@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
@pytest.mark.parametrize('periodic', [True, False])
def test_greville(knots, degree, periodic):
    expected = greville_true(knots, degree, periodic)
    out = greville(knots, degree, periodic)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
def test_elements_spans(knots, degree):
    expected = elements_spans_true(knots, degree)
    out = elements_spans(knots, degree)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)

@pytest.mark.parametrize('breaks', (xp.linspace(0, 1, 10, endpoint=False),
                                    xp.sort(xp.random.random(15))))
@pytest.mark.parametrize(('degree', 'multiplicity'), [(2, 1),
                                                      (3, 1), (3, 2),
                                                      (4, 1), (4, 2), (4, 3),
                                                      (5, 1), (5, 2), (5, 3), (5, 4)])
@pytest.mark.parametrize('periodic', (True, False))
def test_make_knots(breaks, degree, periodic, multiplicity):
    expected = make_knots_true(breaks, degree, periodic, multiplicity)
    out = make_knots(breaks, degree, periodic, multiplicity)
    print(out, expected)
    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
@pytest.mark.parametrize('periodic', (True, False))
@pytest.mark.parametrize('multiplicity', (1, 2, 3))
def test_elevate_knots(knots, degree, periodic, multiplicity):
    expected = elevate_knots_true(knots, degree, periodic, multiplicity)
    out = elevate_knots(knots, degree, periodic, multiplicity)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('breaks', (xp.linspace(0, 1, 10, endpoint=False),
                                    xp.sort(xp.random.random(15))))
@pytest.mark.parametrize('nquads', (2, 3, 4, 5))
def test_quadrature_grid(breaks, nquads):
    quad_x, quad_w = gauss_legendre(nquads)
    expected = quadrature_grid_true(breaks, quad_x, quad_w)
    out = quadrature_grid(breaks, quad_x, quad_w)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
@pytest.mark.parametrize('n', (2, 3, 4, 5))
@pytest.mark.parametrize('normalization', ('B', 'M'))
@pytest.mark.parametrize('nquads', (2, 3, 4, 5))
def test_basis_ders_on_quad_grid(knots, degree, n, normalization, nquads):
    quad_rule_x, quad_rule_w = gauss_legendre(nquads)
    breaks = breakpoints_true(knots, degree)
    quad_grid, quad_weights = quadrature_grid_true(breaks, quad_rule_x, quad_rule_w)

    expected = basis_ders_on_quad_grid_true(knots, degree, quad_grid, n, normalization)
    out = basis_ders_on_quad_grid(knots, degree, quad_grid, n, normalization)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(('knots', 'degree'),
                         [(xp.sort(xp.random.random(15)), 2),
                          (xp.sort(xp.random.random(15)), 3),
                          (xp.sort(xp.random.random(15)), 4),
                          (xp.sort(xp.random.random(15)), 5),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 3),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 4),
                          (xp.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]), 5),
                          (xp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 2),
                          (xp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]), 3)])
def test_basis_integrals(knots, degree):
    expected = basis_integrals_true(knots, degree)
    out = basis_integrals(knots, degree)

    assert xp.allclose(expected, out, atol=ATOL, rtol=RTOL)
