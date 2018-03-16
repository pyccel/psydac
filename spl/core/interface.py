# -*- coding: UTF-8 -*-

import numpy as np

from spl.utilities.quadratures import gauss_legendre

def make_open_knots(p, n):
    """Returns an open knots sequence for n splines and degree p.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    Examples

    >>> from spl.core.interface import make_open_knots

    >>> T = make_open_knots(3, 8)
    >>> T
    array([0. , 0. , 0. , 0. , 0.2, 0.4, 0.6, 0.8, 1. , 1. , 1. , 1. ])

    """
    from spl.core.bsp  import bsp_utils as _core
    T = _core.make_open_knots(p, n)
    return T

def construct_grid_from_knots(p, n, T):
    """Returns the grid associated to a knot vector. It consists of the breaks
    of the knot vector, where every knot has a unique occurence.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    T: list, np.array
        knot vector

    Examples

    >>> from spl.core.interface import make_open_knots
    >>> from spl.core.interface import construct_grid_from_knots

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> grid = construct_grid_from_knots(p, n, T)
    >>> grid
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])

    """
    from spl.core.bsp  import bsp_utils as _core
    grid = _core.construct_grid_from_knots(p, n, T)
    return grid

def construct_quadrature_grid(ne, k, u, w, grid):
    """maps the quadrature points and weights (of order k)
    onto a given grid of ne elements.

    ne: int
        number of elements

    k: int
        quadrature rule order

    u: list, array
        quadrature points on [-1, 1]

    weights: list, array
        quadrature weights

    grid: list, array
        a 1d grid

    Examples

    >>> from spl.core.interface import make_open_knots
    >>> from spl.core.interface import construct_grid_from_knots
    >>> from spl.core.interface import construct_quadrature_grid
    >>> from spl.utilities.quadratures import gauss_legendre

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> grid = construct_grid_from_knots(p, n, T)
    >>> u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
    >>> k = len(u)
    >>> ne = len(grid) - 1        # number of elements
    >>> points, weights = construct_quadrature_grid(ne, k, u, w, grid)

    """
    from spl.core.bsp  import bsp_utils as _core
    points, weights = _core.construct_quadrature_grid(ne, k, u, w, grid)
    return points, weights

def eval_on_grid_splines_ders(p, n, k, d, T, points):
    """Evaluates B-Splines and their derivatives on the quadrature grid.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    k: int
        quadrature rule order

    d: int
       number of derivatives

    T: list, np.array
        knot vector

    points: np.array
        a multi-dimensional array that was constructed using
        construct_quadrature_grid

    Examples

    >>> from spl.core.interface import make_open_knots
    >>> from spl.core.interface import construct_grid_from_knots
    >>> from spl.core.interface import construct_quadrature_grid
    >>> from spl.core.interface import eval_on_grid_splines_ders
    >>> from spl.utilities.quadratures import gauss_legendre

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> grid = construct_grid_from_knots(p, n, T)
    >>> u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
    >>> k = len(u)
    >>> ne = len(grid) - 1        # number of elements
    >>> points, weights = construct_quadrature_grid(ne, k, u, w, grid)
    >>> d = 1                     # number of derivatives
    >>> basis = eval_on_grid_splines_ders(p, n, k, d, T, points)

    """
    from spl.core.bsp  import bsp_utils as _core
    basis = _core.eval_on_grid_splines_ders(p, n, k, d, T, points)
    return basis

def compute_spans(p, n, T):
    """compute the last non-vanishing spline on each element. The returned array
    is of the size of T, which means that you will need, in general, to take
    only the first n-elements entries.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    T: list, np.array
        knot vector

    Examples

    >>> from spl.core.interface import make_open_knots
    >>> from spl.core.interface import compute_spans

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> spans = compute_spans(p, n, T)
    >>> spans
    array([4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0], dtype=int32)
    """
    from spl.core.bsp  import bsp_utils as _core
    spans = _core.compute_spans(p, n, T)
    return spans

def collocation_matrix(p, n, m, knots, u):
    """Returns the collocation matrix representing the evaluation of all
    B-Splines over the sites array u.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    T: list, np.array
        knot vector

    u: list, np.array
        sites over which we evaluate the B-Splines

    Examples

    >>> from spl.core.interface import make_open_knots
    >>> from spl.core.interface import collocation_matrix

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> m = 7 ; u = np.linspace(0., 1., m)
    >>> mat = collocation_matrix(p, n, m, T, u)
    >>> mat.shape
    (7, 8)

    """
    from spl.core.bsp  import bsp_utils as _core
    mat = _core.collocation_matrix(p, n, m, knots, u)
    return mat

def compute_greville(p, n, knots):
    """Returns the Greville abscissae associated to a given knot vector.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    T: list, np.array
        knot vector

    Examples

    >>> from spl.core.interface import make_open_knots
    >>> from spl.core.interface import compute_greville

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> greville = compute_greville(p, n, T)
    >>> greville
    array([0.        , 0.06666667, 0.2       , 0.4       , 0.6       ,
           0.8       , 0.93333333, 1.        ])

    """
    from spl.core.bsp  import bsp_utils as _core
    x = _core.compute_greville(p, n, knots)
    return x


####################################################################################
#if __name__ == '__main__':

