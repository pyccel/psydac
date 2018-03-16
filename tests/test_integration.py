# -*- coding: UTF-8 -*-

import numpy as np

from spl.core import make_open_knots
from spl.core import construct_quadrature_grid
from spl.core import compute_greville

from spl.utilities import integrate
from spl.utilities import gauss_legendre

def test_integrate():
    # ...
    n_elements = 8
    p = 2                    # spline degree
    n = n_elements + p - 1   # number of control points
    # ...

    T = make_open_knots(p, n)
    grid = compute_greville(p, n, T)
    u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
    k = len(u)
    ne = len(grid) - 1        # number of elements
    points, weights = construct_quadrature_grid(ne, k, u, w, grid)
    f = lambda u: u*(1.-u)
    f_int = integrate(points, weights, f)

####################################################################################
if __name__ == '__main__':

    test_integrate()
