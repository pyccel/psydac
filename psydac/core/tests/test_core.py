# -*- coding: UTF-8 -*-

import numpy as np

from psydac.core.interface import make_open_knots
from psydac.core.interface import construct_grid_from_knots
from psydac.core.interface import construct_quadrature_grid
from psydac.core.interface import eval_on_grid_splines_ders
from psydac.core.interface import collocation_matrix
from psydac.core.interface import histopolation_matrix
from psydac.core.interface import compute_greville

from psydac.utilities.quadratures import gauss_legendre

def test_open_knots():
    # ...
    n_elements = 8
    p = 2                    # spline degree
    n = n_elements + p - 1   # number of control points
    # ...

    T = make_open_knots(p, n)
    grid = construct_grid_from_knots(p, n, T)
    u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
    k = len(u)
    ne = len(grid) - 1        # number of elements
    points, weights = construct_quadrature_grid(ne, k, u, w, grid)
    d = 1                     # number of derivatives
    basis = eval_on_grid_splines_ders(p, n, k, d, T, points)


def test_collocation():
    # ...
    n_elements = 8
    p = 2                    # spline degree
    n = n_elements + p - 1   # number of control points
    # ...

    T = make_open_knots(p, n)
    m = 7 ; u = np.linspace(0., 1., m)
    mat = collocation_matrix(p, n, T, u)

def test_histopolation():
    # ...
    n_elements = 8
    p = 2                    # spline degree
    n = n_elements + p - 1   # number of control points
    # ...

    T        = make_open_knots(p, n)
    greville = compute_greville(p, n, T)
    D        = histopolation_matrix(p, n, T, greville)

    _print = lambda x: "%.4f" % x

    for i in range(0, D.shape[0]):
        print ([_print(x) for x in D[i, :]])


####################################################################################
if __name__ == '__main__':

    test_open_knots()
    test_collocation()
    test_histopolation()
