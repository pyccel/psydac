# -*- coding: UTF-8 -*-

import numpy as np

from spl.core import make_open_knots
from spl.core import construct_grid_from_knots
from spl.core import construct_quadrature_grid
from spl.core import eval_on_grid_splines_ders
from spl.core import collocation_matrix
from spl.core import compute_greville

from spl.utilities import gauss_legendre

def test_open_knots():
    # ...
    ne = 8       # number of elements
    p  = 2       # spline degree
    n = ne + p   # number of control points
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
    p = 3 ; n = 8
    T = make_open_knots(p, n)
    m = 7 ; u = np.linspace(0., 1., m)
    mat = collocation_matrix(p, n, m, T, u)

def test_histopolation():
    p = 3 ; n = 8

    T        = make_open_knots(p, n)
    breaks   = construct_grid_from_knots(p, n, T)
    greville = compute_greville(p, n, T)


    m = len(greville)
    mat = collocation_matrix(p, n, m, T, greville)
    print (mat)


####################################################################################
if __name__ == '__main__':

    test_open_knots()
    test_collocation()
    test_histopolation()
