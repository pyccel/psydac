# This file holds the pyccelisable versions of the functions in bsplines.py
# This will be changed once pyccel can return arrays and can get out=None arguments
# like Numpy functions.

import numpy as np

#==============================================================================
def find_span_p(knots: 'float[:]', degree: int, x: float):
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
def find_spans_p(knots: 'float[:]', degree: int, x: 'float[:]', out: 'int[:]'):
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

    out: array_like of ints
        Knot span index for each location in x.

    See Also
    --------
    psydac.core.bsplines.find_span : Determines the knot span at a location.
    """
    n = x.shape[0]

    for i in range(n):
        out[i] = find_span_p(knots, degree, x[i])

#==============================================================================
def basis_funs_p(knots: 'float[:]', degree: int, x: float, span: int, out: 'float[:]'):
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
    left = np.zeros(degree, dtype=float)
    right = np.zeros(degree, dtype=float)

    out[0] = 1.0
    for j in range(0, degree):
        left[j]  = x - knots[span - j]
        right[j] = knots[span + 1 + j] - x
        saved    = 0.0
        for r in range(0, j + 1):
            temp   = out[r] / (right[r] + left[j - r])
            out[r] = saved + right[r] * temp
            saved  = left[j - r] * temp
        out[j + 1] = saved

#==============================================================================
def basis_funs_array_p(knots: 'float[:]', degree: int, x: 'float[:]', span: 'int[:]', out: 'float[:,:]'):
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

    """

    n = x.shape[0]
    for i in range(n):
        basis_funs_p(knots, degree, x[i], span[i], out[i, :])

#==============================================================================
def basis_funs_1st_der_p(knots: 'float[:]', degree: int, x: float, span: int, out: 'float[:]'):
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

    out : numpy.ndarray
        Derivatives of p+1 non-vanishing B-Splines at location x.

    """
    # Compute nonzero basis functions and knot differences for splines
    # up to degree deg-1
    values = np.zeros(degree)
    basis_funs_p(knots, degree-1, x, span, values)

    # Compute derivatives at x using formula based on difference of splines of
    # degree deg-1
    # -------
    # j = 0
    saved = degree * values[0] / (knots[span+1]-knots[span+1-degree])
    out[0] = -saved
    # j = 1,...,degree-1
    for j in range(1,degree):
        temp    = saved
        saved   = degree * values[j] / (knots[span+j+1]-knots[span+j+1-degree])
        out[j] = temp - saved
    # j = degree
    out[degree] = saved


def dot_2d(a: 'float[:, :]', b: 'float[:, :]', out: 'float[:, :]'):
    for i in range(a.shape[0]):
        for k in range(b.shape[1]):
            out[i, k] = np.sum(a[i, :] * b[:, k])


def basis_funs_all_ders_p(knots: 'float[:]', degree: int, x: float, span: int, n: int, out: 'float[:,:]'):
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
    left  = np.empty(degree)
    right = np.empty(degree)
    ndu   = np.empty((degree+1, degree+1))
    a     = np.empty((2, degree+1))
    temp_dot_array = np.zeros((1, 1))
    # Number of derivatives that need to be effectively computed
    # Derivatives higher than degree are = 0.
    ne = min(n, degree)

    # Compute nonzero basis functions and knot differences for splines
    # up to degree, which are needed to compute derivatives.
    # Store values in 2D temporary array 'ndu' (square matrix).

    ndu[0, 0] = 1.0
    for j in range(0, degree):
        left[j]  = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0, j+1):
            # compute inverse of knot differences and save them into lower triangular part of ndu
            ndu[j + 1, r] = 1.0 / (right[r] + left[j - r])
            # compute basis functions and save them into upper triangular part of ndu
            temp          = ndu[r, j] * ndu[j + 1, r]
            ndu[r, j + 1] = saved + right[r] * temp
            saved         = left[j - r] * temp
        ndu[j + 1, j + 1] = saved

    # Compute derivatives in 2D output array 'out'
    out[0, :] = ndu[:, degree]

    for r in range(0, degree+1):

        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, ne + 1):
            d  = 0.0
            rk = r-k
            pk = degree-k
            if r >= k:
               a[s2, 0] = a[s1, 0] * ndu[pk + 1, rk]
               d = a[s2, 0] * ndu[rk, pk]

            j1 = 1   if (rk  > -1 ) else -rk
            j2 = k-1 if (r-1 <= pk) else degree-r

            a[s2, j1:j2 + 1] = (a[s1, j1:j2 + 1] - a[s1, j1 - 1:j2]) * ndu[pk + 1, rk + j1:rk + j2 + 1]
            dot_2d(a[s2:s2 + 1, j1:j2 + 1], ndu[rk + j1:rk + j2 + 1, pk: pk + 1], temp_dot_array)
            d += temp_dot_array[0, 0]

            if r <= pk:
               a[s2, k] = - a[s1, k - 1] * ndu[pk + 1, r]
               d += a[s2, k] * ndu[r, pk]

            out[k, r] = d
            j  = s1
            s1 = s2
            s2 = j

    # Multiply derivatives by correct factors
    r = degree
    for k in range(1, ne+1):
        out[k, :] = out[k, :] * r
        r = r * (degree-k)

def basis_integrals_p(knots: 'float[:]', degree: int, out: 'float[:]'):
    r"""
    Return the integral of each B-spline basis function over the real line:

    K[i] := \int_{-\infty}^{+\infty} B[i](x) dx = (T[i+p+1]-T[i]) / (p+1).

    This array can be used to convert B-splines to M-splines, which have unit
    integral over the real line but no partition-of-unity property.

    Parameters
    ----------
    knots : 1D array_like
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
    T = knots
    p = degree
    n = len(T)-p-1
    for i in range(n):
        out[i] = (T[i + p + 1] - T[i])/ (p + 1)


def collocation_matrix_p(knots: 'float[:]', degree: int, periodic: bool, normalization: bool, xgrid: 'float[:]',
                         out: 'float[:,:]'):
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

    normalization : bool
        Set to False for B-splines, and True for M-splines.

    xgrid : 1D array_like
        Evaluation points.

    out : 2D numpy.ndarray
        Collocation matrix: values of all basis functions on each point in xgrid.

    """

    # Number of basis functions (in periodic case remove degree repeated elements)
    nb = len(knots)-degree-1
    if periodic:
        nb -= degree

    # Number of evaluation points
    nx = len(xgrid)

    basis = np.zeros((nx, degree + 1))
    spans = np.zeros(nx, dtype=int)
    find_spans_p(knots, degree, xgrid, spans)
    basis_funs_array_p(knots, degree, xgrid, spans, basis)

    # Fill in non-zero matrix values

    # Rescaling of B-splines, to get M-splines if needed
    if not normalization:
        if periodic:
            for i in range(0, nx):
                for j in range(0, degree + 1):
                    actual_j = (spans[i] - degree + j) % nb
                    out[i, actual_j] = basis[i, j]
        else:
            for i in range(0, nx):
                out[i, spans[i] - degree:spans[i] + 1] = basis[i, :]
    else:
        integrals = np.zeros(knots.shape[0] - degree - 1)
        basis_integrals_p(knots, degree, integrals)
        if periodic:
            for i in range(0, nx):
                for j in range(0, degree + 1):
                    actual_j = (spans[i] - degree + j) % nb
                    out[i, actual_j] = basis[i, j] / integrals[spans[i] - degree + j]

        else:
            scaling = np.ones_like(integrals) / integrals
            for i in range(0, nx):
                local_scaling = scaling[spans[i] - degree:spans[i] + 1]
                out[i, spans[i] - degree:spans[i] + 1] = basis[i, :] * local_scaling[:]

    # Mitigate round-off errors
    for x in range(nx):
        for y in range(nb):
            if abs(out[x, y]) < 1e-14:
                out[x, y] = 0.0


def histopolation_matrix(knots: 'float[:]', degree: int, periodic: bool, normalization: bool, xgrid: 'float[:]',
                         check_boundary: bool, elevated_knots: 'float[:]', out: 'float[:,:]'):

    nb = len(knots) - degree - 1
    if periodic:
        nb -= degree

    # Number of evaluation points
    nx = len(xgrid)

    # In periodic case, make sure that evaluation points include domain boundaries
    if periodic:
        xmin = knots[degree]
        xmax = knots[-1 - degree]
        if xgrid[0] > xmin:
            xgrid = [xmin, *xgrid]
        if xgrid[-1] < xmax:
            xgrid = [*xgrid, xmax]

    # B-splines of degree p+1: basis[i,j] := Bj(xi)
    #
    # NOTES:
    #  . cannot use M-splines in analytical formula for histopolation matrix
    #  . always use non-periodic splines to avoid circulant matrix structure

    colloc = np.zeros((nx, nb))
    collocation_matrix_p(elevated_knots,
                         degree + 1,
                         False,
                         False,
                         xgrid,
                         colloc)

    # if normalization:
    #     normalize = lambda bi, j: bi
    # elif normalization == 'B':
    #     scaling = basis_integrals(knots, degree)
    #     normalize = lambda bi, j: bi * scaling[j]

    spans = np.zeros(colloc.shape[0])
    for i in range(colloc.shape[0]):
        local_span = 0
        for j in range(colloc.shape[0] - 1, 0, -1):
            if abs(colloc[i, j] > 1e-15):
                local_span = j
                break
        spans[i] = local_span + degree + 1

    # Compute histopolation matrix from collocation matrix of higher degree
    m = colloc.shape[0] - 1
    n = colloc.shape[1] - 1
    H = np.zeros((m, n))
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

    if periodic:

        # Periodic case: wrap around histopolation matrix
        #  1. identify repeated basis functions (sum columns)
        #  2. identify split interval (sum rows)
        Hp = np.zeros((nx, nb))
        for i in range(m):
            for j in range(n):
                Hp[i%nx, j%nb] += H[i, j]




def elevate_knots_p(knots: 'float[:]', degree: int, periodic: bool, out: 'float[:]',
                    multiplicity: int = 1,
                    tol: float = 1e-15):
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

    if periodic:
        T, p = knots, degree
        period = T[len(knots) -1 - p] - T[p]
        left   = T[len(knots) -2 - 2 * p] - period
        right  = T[2 * p + 1] + period

        out[0] = left
        out[-1] = right
        out[1: - 1] = knots

    else:
        out[0] = knots[0]
        out[1:degree + 2] = knots[:degree+1]

        n_out = out.shape[0]
        n_knots = len(knots)
        out[n_out - degree - 2] = knots[n_knots - 1]
        out[n_out - degree - 1:n_out] = knots[n_knots - degree - 1:]

        if len(knots[degree + 1:n_knots - degree - 1]) > 0:
            out[degree + 2: degree + 2 + multiplicity] = knots[degree + 1]

            unique_index = 0

            for i in range(degree + 1, len(knots) - degree - 2, 1):
                if knots[i + 1] - knots[i] > tol:
                    out[degree + 2 + multiplicity * (unique_index + 1):
                        degree + 2 + multiplicity * (unique_index + 2)] = knots[i + 1]
                    unique_index += 1

        else:
            out[degree + 2: n_out - degree - 2] = knots[degree + 1:n_knots - degree - 1]
