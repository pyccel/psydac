# -*- coding: UTF-8 -*-

import numpy as np


def integrate_1d(points, weights, fun):
     """Integrates the function f over the quadrature grid
    defined by (points,weights) in 1d.

    points: np.array
        a multi-dimensional array describing the quadrature points mapped onto
        the grid. it must be constructed using construct_quadrature_grid

    weights: np.array
        a multi-dimensional array describing the quadrature weights (scaled) mapped onto
        the grid. it must be constructed using construct_quadrature_grid

    Examples

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import construct_grid_from_knots
    >>> from psydac.core.interface import construct_quadrature_grid
    >>> from psydac.core.interface import compute_greville
    >>> from psydac.utilities.quadratures import gauss_legendre

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
    n = points.shape[0]
    k = points.shape[1]
    """
    f_int = np.zeros(n)
    
    for ie in range(n):
        for g in range(k):
            f_int[ie] += weights[ie, g]*fun(points[ie, g])
        
    return f_int



def integrate_2d(points, weights, fun):

    """Integrates the function f over the quadrature grid
    defined by (points,weights) in 2d.

    points: list, tuple
        list of quadrature points, as they should be passed for `integrate`

    weights: list, tuple
        list of quadrature weights, as they should be passed for `integrate`

    Examples

    """  
    pts_0, pts_1 = points
    wts_0, wts_1 = weights
    
    n0 = pts_0.shape[0]
    n1 = pts_1.shape[0]
    k0 = pts_0.shape[1]
    k1 = pts_1.shape[1]
    
    f_int = np.zeros((n0, n1))
    
    for ie_0 in range(n0):
        for ie_1 in range(n1):
            for g_0 in range(k0):
                for g_1 in range(k1):
                    f_int[ie_0, ie_1] += wts_0[ie_0, g_0]*wts_1[ie_1, g_1]*fun(pts_0[ie_0, g_0], pts_1[ie_1, g_1])
                     
    return f_int



def integrate_3d(points, weights, fun):
    
    pts_0, pts_1, pts_2 = points
    wts_0, wts_1, wts_2 = weights
    
    n0 = pts_0.shape[0]
    n1 = pts_1.shape[0]
    n2 = pts_2.shape[0]
    k0 = pts_0.shape[1]
    k1 = pts_1.shape[1]
    k2 = pts_2.shape[1]
    
    f_int = np.zeros((n0, n1, n2))
    
    for ie_0 in range(n0):
        for ie_1 in range(n1):
            for ie_2 in range(n2):
                for g_0 in range(k0):
                    for g_1 in range(k1):
                        for g_2 in range(k2):
                            f_int[ie_0, ie_1, ie_2] += wts_0[ie_0, g_0]*wts_1[ie_1, g_1]*wts_2[ie_2, g_2]\
                                                       *fun(pts_0[ie_0, g_0], pts_1[ie_1, g_1], pts_2[ie_2, g_2])
                     
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
        from psydac.core.interface        import construct_grid_from_knots
        from psydac.core.interface        import compute_greville
        from psydac.core.interface        import construct_quadrature_grid
        from psydac.utilities.quadratures import gauss_legendre

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
        return integrate_1d(self._points, self._weights, f)


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
        from psydac.core.interface import compute_greville

        if sites is None:
            sites = compute_greville(p, n, T)

        self._sites = sites
        self._p = p
        self._n = n
        self._T = T

    @property
    def sites(self):
        return self._sites

    def __call__(self, f):
        """evaluates the function over sites."""
        return np.array([f(x) for x in self._sites])

