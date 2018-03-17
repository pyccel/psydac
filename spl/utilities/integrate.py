# -*- coding: UTF-8 -*-

import numpy as np

def integrate(points, weights, f):
    """Integrates the function f over the quadrature grid
    defined by (points,weights).

    points: np.array
        a multi-dimensional array describing the quadrature points mapped onto
        the grid. it must be constructed using construct_quadrature_grid

    weights: np.array
        a multi-dimensional array describing the quadrature weights (scaled) mapped onto
        the grid. it must be constructed using construct_quadrature_grid

    Examples

    >>> from spl.core.interface import make_open_knots
    >>> from spl.core.interface import construct_grid_from_knots
    >>> from spl.core.interface import construct_quadrature_grid
    >>> from spl.core.interface import compute_greville
    >>> from spl.utilities.quadratures import gauss_legendre

    >>> n_elements = 8
    >>> p = 2                    # spline degree
    >>> n = n_elements + p - 1   # number of control points
    >>> T = make_open_knots(p, n)
    >>> grid = compute_greville(p, n, T)
    >>> u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
    >>> k = len(u)
    >>> ne = len(grid) - 1        # number of elements
    >>> points, weights = construct_quadrature_grid(ne, k, u, w, grid)
    >>> f = lambda u: u*(1.-u)
    >>> f_int = integrate(points, weights, f)
    >>> f_int
    [0.00242954 0.01724976 0.02891156 0.03474247 0.03474247 0.02891156
     0.01724976 0.00242954]
    """

    ne = points.shape[1]
    f_int = np.zeros(ne)
    for ie in range(0, ne):
        X = points[:, ie]
        W = weights[:, ie]
        f_int[ie] = np.sum(w*f(x) for (x,w) in zip(X,W))

    return f_int


class Integral(object):
    """Class for 1d integration. It is presented as a class in order to store
    locally all the needed information for performing the integration.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    T: list, np.array
        knot vector

    kind: str
        one among {'natural', 'greville'}.
        'natural' for standard integration over the grid
        induced by the knot vector.

    k: int
        quadrature order. if not given it will be p+1
    """

    def __init__(self, p, n, T, kind='natural', k=None):
        from spl.core import construct_grid_from_knots
        from spl.core import compute_greville
        from spl.core import construct_quadrature_grid
        from spl.utilities import gauss_legendre

        assert(kind in ['natural', 'greville'])

        if kind == 'natural':
            grid = construct_grid_from_knots(p, n, T)

        if kind == 'greville':
            grid = compute_greville(p, n, T)

        if k is None:
            k = p + 1

        u, w = gauss_legendre(k-1)  # gauss-legendre quadrature rule
        ne = len(grid) - 1          # number of elements
        points, weights = construct_quadrature_grid(ne, k, u, w, grid)

        self._grid = grid
        self._kind = kind
        self._p = p
        self._n = n
        self._T = T
        self._points = points
        self._weights = weights

    def __call__(self, f):
        """Computes the integral of the function f over each element of the grid."""
        return integrate(self._points, self._weights, f)


class Interpolation(object):
    """Class for 1d interpolation. It is presented as a class in order to store
    locally all the needed information for performing the interpolation.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    T: list, np.array
        knot vector

    sites: list, np.array
        sites for interpolation.
        if not given, the greville abscissae will be used
    """

    def __init__(self, p, n, T, sites=None):
        from spl.core import compute_greville

        if sites is None:
            sites = compute_greville(p, n, T)

        self._sites = sites
        self._p = p
        self._n = n
        self._T = T

    def __call__(self, f):
        """evaluates the function over sites."""
        return np.array([f(x) for x in self._sites])


class Contribution(object):

    def __init__(self, p, n, T, sites=None):
        """Returns the 1d rhs for the function f."""
        from spl.core.interface import construct_grid_from_knots
        from spl.core.interface import construct_quadrature_grid
        from spl.core.interface import eval_on_grid_splines_ders
        from spl.core.interface import compute_spans
        from spl.utilities.quadratures import gauss_legendre

        # constructs the grid from the knot vector
        grid = construct_grid_from_knots(p, n, T)

        ne = len(grid) - 1        # number of elements
        spans = compute_spans(p, n, T)

        u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
        k = len(u)
        points, weights = construct_quadrature_grid(ne, k, u, w, grid)

        d = 1                     # number of derivatives
        basis = eval_on_grid_splines_ders(p, n, k, d, T, points)

        self._grid = grid
        self._p = p
        self._n = n
        self._T = T
        self._points = points
        self._weights = weights
        self._spans = spans
        self._basis = basis

    def __call__(self, f):
        """Returns the contribution of the function f over the FEM basis."""
        ne = len(self._grid) - 1        # number of elements
        k  = self._points.shape[0]
        p = self._p
        n = self._n
        spans = self._spans
        basis = self._basis
        points = self._points
        weights = self._weights

        # ...
        rhs = np.zeros(n)
        # ...

        # ... build matrix
        for ie in range(0, ne):
            i_span = spans[ie]
            for il in range(0, p+1):
                i = i_span - p  - 1 + il

                v_rhs = 0.0
                for g in range(0, k):
                    bi_0 = basis[il, 0, g, ie]

                    wvol = weights[g, ie]
                    x    = points[g, ie]

                    v_rhs += bi_0 * f(x) * wvol

                rhs[i] += v_rhs
        # ...

        return rhs
