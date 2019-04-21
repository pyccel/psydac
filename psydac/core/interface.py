# -*- coding: UTF-8 -*-

# TODO parameterized tests using pytest

import numpy as np

from psydac.utilities.quadratures import gauss_legendre
from psydac.core.bsp import bsp_utils as _core

__all__ = [
    'make_open_knots',
    'make_periodic_knots',
    'construct_grid_from_knots',
    'construct_quadrature_grid',
    'eval_on_grid_splines_ders',
    'compute_spans',
    'compute_greville',
    'collocation_matrix',
    'collocation_cardinal_splines',
    'histopolation_matrix',
    'mass_matrix',
    'matrix_multi_stages'
]

def make_open_knots(p, n):
    """Returns an open knots sequence for n splines and degree p.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    Examples

    >>> from psydac.core.interface import make_open_knots

    >>> T = make_open_knots(3, 8)
    >>> T
    array([0. , 0. , 0. , 0. , 0.2, 0.4, 0.6, 0.8, 1. , 1. , 1. , 1. ])

    """
    T = _core.make_open_knots(p, n)
    return T

def make_periodic_knots(p, n):
    """Returns a periodic knots sequence for n splines and degree p.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    Examples

    >>> from psydac.core.interface import make_periodic_knots

    >>> T = make_periodic_knots(3, 8)
    >>> T
    array([-0.6 , -0.4 , -0.2 , 0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2 , 1.4 , 1.6 ])

    """
    T = _core.make_periodic_knots(p, n)
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

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import construct_grid_from_knots

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> grid = construct_grid_from_knots(p, n, T)
    >>> grid
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])

    """
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

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import construct_grid_from_knots
    >>> from psydac.core.interface import construct_quadrature_grid
    >>> from psydac.utilities.quadratures import gauss_legendre

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> grid = construct_grid_from_knots(p, n, T)
    >>> u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
    >>> k = len(u)
    >>> ne = len(grid) - 1        # number of elements
    >>> points, weights = construct_quadrature_grid(ne, k, u, w, grid)

    """
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

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import construct_grid_from_knots
    >>> from psydac.core.interface import construct_quadrature_grid
    >>> from psydac.core.interface import eval_on_grid_splines_ders
    >>> from psydac.utilities.quadratures import gauss_legendre

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

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import compute_spans

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> spans = compute_spans(p, n, T)
    >>> spans
    array([4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0], dtype=int32)
    """
    spans = _core.compute_spans(p, n, T)
    return spans

def compute_greville(p, n, knots):
    """Returns the Greville abscissae associated to a given knot vector.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    T: list, np.array
        knot vector

    Examples

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import compute_greville

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> greville = compute_greville(p, n, T)
    >>> greville
    array([0.        , 0.06666667, 0.2       , 0.4       , 0.6       ,
           0.8       , 0.93333333, 1.        ])

    """
    x = _core.compute_greville(p, n, knots)
    return x

# TODO remove `m` from the fortran code => use pyf file
def collocation_matrix(p, n, T, u):
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

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import collocation_matrix

    >>> p = 3 ; n = 8
    >>> T = make_open_knots(p, n)
    >>> u = np.linspace(0., 1., 7)
    >>> mat = collocation_matrix(p, n, T, u)
    >>> mat.shape
    (7, 8)

    """
    m = len(u)
    mat = _core.collocation_matrix(p, n, m, T, u)
    return mat

# ...
def collocation_cardinal_splines(p, n):
    """Returns the collocation matrix with cardinal B-Splines.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`


    Examples

    >>> from psydac.core.interface import collocation_matrix

    >>> p = 3 ; n = 8
    >>> mat = collocation_cardinal_splines(p, n)
    >>> mat.shape
    (8, 8)

    """
    mat = _core.collocation_cardinal_splines(p, n)
    return mat
# ...


# TODO must be implemented in Fortran
def histopolation_matrix(p, n, T, greville):
    """Returns the histpolation matrix.

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    T: list, np.array
        knot vector

    greville: list, np.array
        greville abscissae

    Examples

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import compute_greville
    >>> from psydac.core.interface import histopolation_matrix

    >>> p = 3 ; n = 8
    >>> T        = make_open_knots(p, n)
    >>> greville = compute_greville(p, n, T)
    >>> D        = histopolation_matrix(p, n, T, greville)

    >>> _print = lambda x: "%.4f" % x
    >>> for i in range(0, D.shape[0]):
    >>>     print ([_print(x) for x in D[i, :]])
    ['0.7037', '0.1389', '0.0062', '-0.0000', '-0.0000', '-0.0000', '-0.0000']
    ['0.2963', '0.6111', '0.1605', '0.0000', '0.0000', '0.0000', '0.0000']
    ['0.0000', '0.2500', '0.6667', '0.1667', '-0.0000', '-0.0000', '-0.0000']
    ['0.0000', '0.0000', '0.1667', '0.6667', '0.1667', '0.0000', '0.0000']
    ['0.0000', '0.0000', '0.0000', '0.1667', '0.6667', '0.2500', '0.0000']
    ['0.0000', '0.0000', '0.0000', '0.0000', '0.1605', '0.6111', '0.2963']
    ['0.0000', '0.0000', '0.0000', '0.0000', '0.0062', '0.1389', '0.7037']

    """
    ng = len(greville)

    # basis[i,j] := Nj(xi)
    basis = collocation_matrix(p, n, T, greville)

    D = np.zeros((ng-1, n-1))
    for i in range(0, ng-1):
        for j in range(max(i-p+1,1),min(i+p+3,n) ):
            s = 0.
            for k in range(0, j):
                s += basis[i,k] - basis[i+1,k]
            D[i, j-1] = s

    return D


# TODO move to Fortran
def mass_matrix(p, n, T):
    """Returns the 1d mass matrix.

    Examples

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import mass_matrix

    >>> p = 3; n = 4
    >>> T = make_open_knots(p, n)
    >>> mass_matrix(p, n, T)
    array([[0.14285714, 0.07142857, 0.02857143, 0.00714286],
           [0.07142857, 0.08571429, 0.06428571, 0.02857143],
           [0.02857143, 0.06428571, 0.08571429, 0.07142857],
           [0.00714286, 0.02857143, 0.07142857, 0.14285714]])

    """
    # constructs the grid from the knot vector
    grid = construct_grid_from_knots(p, n, T)

    ne = len(grid) - 1        # number of elements
    spans = compute_spans(p, n, T)

    u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
    k = len(u)
    points, weights = construct_quadrature_grid(ne, k, u, w, grid)

    d = 1                     # number of derivatives
    basis = eval_on_grid_splines_ders(p, n, k, d, T, points)

    # ...
    mass = np.zeros((n,n))
    # ...

    # ... build matrix
    for ie in range(0, ne):
        i_span = spans[ie]
        for il in range(0, p+1):
            for jl in range(0, p+1):
                i = i_span - p  - 1 + il
                j = i_span - p  - 1 + jl

                v_m = 0.0
                for g in range(0, k):
                    bi_0 = basis[il, 0, g, ie]
                    bj_0 = basis[jl, 0, g, ie]

                    wvol = weights[g, ie]

                    v_m += bi_0 * bj_0 * wvol

                mass[i, j] += v_m
    # ...

    return mass

# ...
def matrix_multi_stages(ts, n, p, knots):

    """
    return the refinement matrix corresponding to the insertion of a given list of knots

    ts: list, np.array
        knots to be inserted

    p: int
        spline degree

    n: int
        number of splines functions i.e. `control points`

    knots: list, np.array
        knot vector

    Examples

    >>> from psydac.core.interface import matrix_multi_stages

    >>> ts = [0.1, 0.2, 0.4, 0.5, 0.7, 0.8]
    >>> tc = [0,0, 0.3, 0.6, 1, 1]
    >>> nc = 4 ; pc = 1

    >>> matrix_multi_stages(ts, nc, pc, tc)
    array([[1.        , 0.        , 0.        , 0.        ],
           [0.66666667, 0.33333333, 0.        , 0.        ],
           [0.33333333, 0.66666667, 0.        , 0.        ],
           [0.        , 1.        , 0.        , 0.        ],
           [0.        , 0.66666667, 0.33333333, 0.        ],
           [0.        , 0.33333333, 0.66666667, 0.        ],
           [0.        , 0.        , 1.        , 0.        ],
           [0.        , 0.        , 0.75      , 0.25      ],
           [0.        , 0.        , 0.5       , 0.5       ],
           [0.        , 0.        , 0.        , 1.        ]])

    """

    m = len(ts)
    mat = _core.matrix_multi_stages(m, ts, n, p, knots)
    return mat

# ...

####################################################################################
#if __name__ == '__main__':

