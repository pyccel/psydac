# -*- coding: UTF-8 -*-

import numpy as np

from psydac.core.interface import make_open_knots
from psydac.core.interface import construct_quadrature_grid
from psydac.core.interface import compute_greville

from psydac.utilities.quadratures import gauss_legendre
from psydac.utilities.integrate   import integrate_1d
from psydac.utilities.integrate   import Integral

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
    F = np.zeros(n)
    f_int = integrate_1d(points, weights, F, f)

def test_integral():
    # ...
    n_elements = 8
    p = 2                    # spline degree
    n = n_elements + p - 1   # number of control points
    # ...

    T = make_open_knots(p, n)

    f = lambda u: u*(1.-u)

    integral = Integral(p, n, T, kind='natural')
    f_int = integral(f)

    integral = Integral(p, n, T, kind='greville')
    f_int = integral(f)

####################################################################################
if __name__ == '__main__':

    test_integrate()
    test_integral()
