# This file holds the pyccelisable versions of the functions in bsplines.py
# This will be changed once pyccel can return arrays and can get out=None arguments
# like Numpy functions.

import numpy as np


# =============================================================================
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


# =============================================================================
def find_spans_p(knots: 'float[:]', degree: int, x: 'float[:]', out: 'int[:]'):
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

    x : array
        Location of interest.

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.
    """
    n = x.shape[0]

    for i in range(n):
        out[i] = find_span_p(knots, degree, x[i])


# =============================================================================
def basis_funs_p(knots: 'float[:]', degree: int, x: float, span: int, out: 'float[:]'):
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.
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


# =============================================================================
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.
    """

    n = x.shape[0]
    for i in range(n):
        basis_funs_p(knots, degree, x[i], span[i], out[i, :])


# =============================================================================
def basis_funs_1st_der_p(knots: 'float[:]', degree: int, x: float, span: int, out: 'float[:]'):
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Notes
    -----
    See function 's_bsplines_non_uniform__eval_deriv' in Selalib's ([2]) source file
    'src/splines/sll_m_bsplines_non_uniform.F90'.

    References
    ----------
    .. [2] SELALIB, Semi-Lagrangian Library. http://selalib.gforge.inria.fr
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


# =============================================================================
def dot_2d(a: 'float[:, :]', b: 'float[:, :]', out: 'float[:, :]'):
    for i in range(a.shape[0]):
        for k in range(b.shape[1]):
            out[i, k] = np.sum(a[i, :] * b[:, k])


# =============================================================================
def basis_funs_all_ders_p(knots: 'float[:]', degree: int, x: float, span: int, n: int, normalization: bool,
                          out: 'float[:,:]'):
    """
    Evaluate value and n derivatives at x of all basis functions with
    support in interval :math:`[x_{span-1}, x_{span}]`.

    If called with normalization='M', this uses M-splines instead of B-splines.

    Fills a  2D array with n+1 (from 0-th to n-th) derivatives at x
    of all (degree+1) non-vanishing basis functions in given span.

    .. math::
        ders[i,j] = \\frac{d^i}{dx^i} B_k(x) \\, \\text{with} k=(span-degree+j),
        \\forall (i,j),  0 \\leq i \\leq n \\, 0 \\leq j \\leq \\text{degree}+1.

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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.
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

    if normalization:
        for i in range(degree + 1):
            out[:, i] *= (degree + 1) / (knots[i + span + 1] - knots[i + span - degree])


# =============================================================================
def basis_integrals_p(knots: 'float[:]', degree: int, out: 'float[:]'):
    """
    Return the integral of each B-spline basis function over the real line:

    :math: K[i] := \\int_{-\\infty}^{+\\infty} B[i](x) dx = (T[i+p+1]-T[i]) / (p+1).

    This array can be used to convert B-splines to M-splines, which have unit
    integral over the real line but no partition-of-unity property.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.

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


# =============================================================================
def collocation_matrix_p(knots: 'float[:]', degree: int, periodic: bool, normalization: bool, xgrid: 'float[:]',
                         out: 'float[:,:]'):
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Notes
    -----
    The collocation matrix :math:`C_ij = B_j(x_i)`, contains the
    values of each B-spline basis function :math:`B_j` at all locations :math:`x_i`.
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


# =============================================================================
def histopolation_matrix_p(knots: 'float[:]', degree: int, periodic: bool, normalization: bool, xgrid: 'float[:]',
                           check_boundary: bool, elevated_knots: 'float[:]', out: 'float[:,:]'):
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

    check_boundary : bool, default=True
        If true and ``periodic``, will check the boundaries of ``xgrid``.

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Notes
    -----
    The histopolation matrix :math:`H_{ij} = \\int_{x_i}^{x_{i+1}}B_j(x)\\,dx`
    contains the integrals of each B-spline basis function :math:`B_j` between
    two successive grid points.
    """
    nb = len(knots) - degree - 1
    if periodic:
        nb -= degree

    # Number of evaluation points
    nx = len(xgrid)

    # In periodic case, make sure that evaluation points include domain boundaries

    if periodic:
        xgrid_new = np.zeros(len(xgrid) + 2)
        actual_len = len(xgrid)
        if check_boundary:
            xmin = knots[degree]
            xmax = knots[len(knots) - 1 - degree]

            if xgrid[0] > xmin and xgrid[-1] < xmax:
                xgrid_new[0] = xmin
                xgrid_new[1:-1] = xgrid[:]
                xgrid_new[-1] = xmax
                actual_len += 2

            elif xgrid[0] > xmin:
                xgrid_new[0] = xmin
                xgrid_new[1:-1] = xgrid[:]
                actual_len += 1

            elif xgrid[-1] < xmax:
                xgrid_new[-2] = xmax
                xgrid_new[:-2] = xgrid
                actual_len += 1
            else:
                xgrid_new[:-2] = xgrid

        else:
            xgrid_new[:-2] = xgrid

        # B-splines of degree p+1: basis[i,j] := Bj(xi)

        # NOTES:
        #  . cannot use M-splines in analytical formula for histopolation matrix
        #  . always use non-periodic splines to avoid circulant matrix structure
        nb_elevated = len(elevated_knots) - (degree + 1) - 1
        colloc = np.zeros((actual_len, nb_elevated))
        collocation_matrix_p(elevated_knots,
                             degree + 1,
                             False,
                             False,
                             xgrid_new[:actual_len],
                             colloc)

        m = colloc.shape[0] - 1
        n = colloc.shape[1] - 1

        spans = np.zeros(colloc.shape[0], dtype=int)
        for i in range(colloc.shape[0]):
            local_span = 0
            for j in range(colloc.shape[1]):
                if abs(colloc[i, j]) != 0:
                    local_span = j
                    break
            spans[i] = local_span + degree + 1

        # Compute histopolation matrix from collocation matrix of higher degree
        H = np.zeros((m, n))
        if normalization:
            for i in range(m):
                # Indices of first/last non-zero elements in row of collocation matrix
                jstart = spans[i] - (degree + 1)
                jend = min(spans[i + 1], n)
                # Compute non-zero values of histopolation matrix
                for j in range(1 + jstart, jend + 1):
                    s = np.sum(colloc[i, 0:j]) - np.sum(colloc[i + 1, 0:j])
                    H[i, j - 1] = s

        else:
            integrals = np.zeros(knots.shape[0] - degree - 1)
            basis_integrals_p(knots, degree, integrals)
            for i in range(m):
                # Indices of first/last non-zero elements in row of collocation matrix
                jstart = spans[i] - (degree + 1)
                jend = min(spans[i + 1], n)
                # Compute non-zero values of histopolation matrix
                for j in range(1 + jstart, jend + 1):
                    s = np.sum(colloc[i, 0:j]) - np.sum(colloc[i + 1, 0:j])
                    H[i, j - 1] = s * integrals[j - 1]

        # Mitigate round-off errors
        for i in range(m):
            for j in range(n):
                if abs(H[i, j]) < 1e-14:
                    H[i, j] = 0.0

        # Periodic case: wrap around histopolation matrix
        #  1. identify repeated basis functions (sum columns)
        #  2. identify split interval (sum rows)
        for i in range(m):
            for j in range(n):
                out[i % nx, j % nb] += H[i, j]

    else:

        # B-splines of degree p+1: basis[i,j] := Bj(xi)

        # NOTES:
        #  . cannot use M-splines in analytical formula for histopolation matrix
        #  . always use non-periodic splines to avoid circulant matrix structure
        nb_elevated = len(elevated_knots) - (degree + 1) - 1
        colloc = np.zeros((nx, nb_elevated))
        collocation_matrix_p(elevated_knots,
                             degree + 1,
                             False,
                             False,
                             xgrid,
                             colloc)

        spans = np.zeros(colloc.shape[0], dtype=int)
        for i in range(colloc.shape[0]):
            local_span = 0
            for j in range(colloc.shape[1]):
                if abs(colloc[i, j]) != 0:
                    local_span = j
                    break
            spans[i] = local_span + degree + 1

        m = colloc.shape[0] - 1
        n = colloc.shape[1] - 1
        # Compute histopolation matrix from collocation matrix of higher degree
        if normalization:
            for i in range(m):
                # Indices of first/last non-zero elements in row of collocation matrix
                jstart = spans[i] - (degree + 1)
                jend = min(spans[i + 1], n)
                # Compute non-zero values of histopolation matrix
                for j in range(1 + jstart, jend + 1):
                    s = np.sum(colloc[i, 0:j]) - np.sum(colloc[i + 1, 0:j])
                    out[i, j - 1] = s

        else:
            integrals = np.zeros(knots.shape[0] - degree - 1)
            basis_integrals_p(knots, degree, integrals)
            for i in range(m):
                # Indices of first/last non-zero elements in row of collocation matrix
                jstart = spans[i] - (degree + 1)
                jend = min(spans[i + 1], n)
                # Compute non-zero values of histopolation matrix
                for j in range(1 + jstart, jend + 1):
                    s = np.sum(colloc[i, 0:j]) - np.sum(colloc[i + 1, 0:j])
                    out[i, j - 1] = s * integrals[j - 1]

        # Mitigate round-off errors
        for i in range(m):
            for j in range(n):
                if abs(out[i, j]) < 1e-14:
                    out[i, j] = 0.0

            # Non periodic case: stop here


# =============================================================================
def merge_sort(a: 'float[:]') -> 'float[:]':
    if len(a) != 1 and len(a) != 0:
        n = len(a)

        a1 = np.zeros(n // 2)
        a1[:] = a[:n // 2]
        a2 = np.zeros(n - n // 2)
        a2[:] = a[n // 2:]

        merge_sort(a1)
        merge_sort(a2)

        i_a1 = 0
        i_a2 = 0
        for i in range(n):
            a1_i = a1[i_a1]
            a2_i = a2[i_a2]
            if a1_i < a2_i:
                a[i] = a1_i
                i_a1 += 1
            else:
                a[i] = a2_i
                i_a2 += 1
            if i_a1 == len(a1) or i_a2 == len(a2):
                last_i = i
                break

        for i_1 in range(n // 2 - i_a1):
            a[last_i + 1 + i_1] = a1[i_a1 + i_1]

        for i_2 in range(n - n // 2 - i_a2):
            a[last_i + 1 + i_2] = a2[i_a2 + i_2]


# =============================================================================
def breakpoints_p(knots: 'float[:]', degree: int, out: 'float[:]', tol: float):
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Returns
    -------
    breaks : numpy.ndarray (1D)
        Abscissas of all breakpoints.
    """
    out[0] = knots[degree]
    i_out = 1
    for i in range(degree + 1, len(knots) - degree):
        if abs(knots[i] - knots[i + 1]) > tol:
            out[i_out] = knots[i]
            i_out += 1
    return i_out

# =============================================================================
def greville_p(knots: 'float[:]', degree: int, periodic: bool, out:'float[:]'):
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.
    """
    T = knots
    p = degree
    n = len(T)-2*p-1 if periodic else len(T)-p-1

    # Compute greville abscissas as average of p consecutive knot values
    for i in range(1, 1+n):
        out[i - 1] = sum(T[i:i + p]) / p

    # Domain boundaries
    a = T[p]
    b = T[len(T) - 1 - p]

    # If needed apply periodic boundary conditions, then sort array
    if periodic:
        out[:] = (out[:] - a) % (b-a) + a
        merge_sort(out)

    # Make sure roundoff errors don't push Greville points outside domain
    out[0] = max(out[0], a)
    out[-1] = min(out[-1], b)


# =============================================================================
def elements_spans_p(knots: 'float[:]', degree: int, out: 'int[:]'):
    """
    Compute the index of the last non-vanishing spline on each grid element
    (cell). The length of the returned array is the number of cells.

    Parameters
    ----------
    knots : array_like
        Knots sequence.

    degree : int
        Polynomial degree of B-splines.

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Notes
    -----
    1) Numbering of basis functions starts from 0, not 1;
    2) This function could be written in two lines:

       breaks = breakpoints( knots, degree )
       spans  = np.searchsorted( knots, breaks[:-1], side='right' ) - 1
    """
    temp_array = np.zeros(len(knots))

    actual_len = breakpoints_p( knots, degree, temp_array)

    nk     = len(knots)
    ne     = actual_len - 1

    ie = 0
    for ik in range(degree, nk-degree):
        if knots[ik + 1] - knots[ik] >= 1e-15:
            out[ie] = ik
            ie += 1
        if ie == ne:
            break

    return ne


# =============================================================================
def make_knots_p(breaks: 'float[:]', degree: int, periodic: bool, out: 'float[:]', multiplicity: int = 1):
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.
.
    """
    for i in range(1, len(breaks) - 1):
        out[degree + 1  + (i - 1) * multiplicity:degree + 1 + i * multiplicity] = breaks[i]

    out[degree] = breaks[0]
    out[len(out) - degree - 1] = breaks[-1]

    if periodic:
        period = breaks[-1]-breaks[0]
        for i in range(degree):
            out[i] = breaks[len(breaks) - degree - 1 + i] - period
            out[len(out) - 1 - i] = breaks[degree - i] + period
    else:
        out[0:degree + 1] = breaks[0]
        out[len(out) - degree - 1:] = breaks[-1]


# =============================================================================
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.
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


# =============================================================================
def quadrature_grid_p(breaks: 'float[:]', quad_rule_x: 'float[:]', quad_rule_w: 'float[:]', out: 'float[:,:,:]'):
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Notes
    -----
    Contents of 2D output arrays 'quad_x' and 'quad_w' are accessed with two
    indices (ie,iq) where:
      . ie is the global element index;
      . iq is the local index of a quadrature point within the element.

    """
    ne = len(breaks) - 1
    nq = len(quad_rule_x)

    # Compute location and weight of quadrature points from basic rule
    for ie in range(len(breaks) - 1):
        a = breaks[ie]
        b = breaks[ie + 1]

        c0 = 0.5 * (a + b)
        c1 = 0.5 * (b - a)
        out[ie, :, 0] = c1 * quad_rule_x[:] + c0
        out[ie, :, 1] = c1 * quad_rule_w[:]


# =============================================================================
def basis_ders_on_quad_grid_p(knots: 'float[:]', degree: int, quad_grid: 'float[:,:]', nders: int, normalization: bool,
                            out: 'float[:,:,:,:]'):
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

    out : array
        The result will be inserted into this array.
        It should be of the appropriate shape and dtype.

    Notes
    -----
        Values of B-Splines and their derivatives at quadrature points in
        each element of 1D domain. Indices are
        . ie: global element         (0 <= ie <  ne    )
        . il: local basis function   (0 <= il <= degree)
        . id: derivative             (0 <= id <= nders )
        . iq: local quadrature point (0 <= iq <  nq    )
    """

    ne, nq = quad_grid.shape
    if normalization:
        integrals = np.zeros(knots.shape[0] - degree - 1)
        basis_integrals_p(knots, degree, integrals)

    temp_spans = np.zeros(len(knots), dtype=int)
    actual_index = elements_spans_p(knots, degree, temp_spans)
    spans = temp_spans[:actual_index]

    ders = np.zeros((nders + 1, degree + 1))

    for ie in range(ne):
        xx = quad_grid[ie, :]
        span = spans[ie]
        for iq, xq in enumerate(xx):
            basis_funs_all_ders_p(knots, degree, xq, span, nders, False, ders)
            if normalization:
                ders /= integrals[span - degree:span + 1]
            for i_der in range(nders + 1):
                for i_basis in range(degree + 1):
                    out[ie, i_basis, i_der, iq] = ders[i_der, i_basis]
