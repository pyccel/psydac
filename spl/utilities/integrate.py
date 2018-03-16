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
    from spl.core.interface import make_open_knots
    from spl.core.interface import construct_grid_from_knots
    from spl.core.interface import construct_quadrature_grid
    from spl.core.interface import compute_greville
    from spl.utilities.quadratures import gauss_legendre

    ne = points.shape[1]
    f_int = np.zeros(ne)
    for ie in range(0, ne):
        X = points[:, ie]
        W = weights[:, ie]
        f_int[ie] = np.sum(w*f(x) for (x,w) in zip(X,W))

    return f_int
