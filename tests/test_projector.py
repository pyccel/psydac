# -*- coding: UTF-8 -*-

import numpy as np

from spl.core import make_open_knots
from spl.core import construct_grid_from_knots
from spl.core import construct_quadrature_grid
from spl.core import eval_on_grid_splines_ders
from spl.core import collocation_matrix
from spl.core import histopolation_matrix
from spl.core import compute_greville

from spl.utilities import gauss_legendre
from spl.utilities import Integral
from spl.utilities import Interpolation

from scipy.linalg import inv

def test_projectors_1d():
    # ...
    n_elements = 8
    p = 3                    # spline degree
    n = n_elements + p - 1   # number of control points
    # ...

    T    = make_open_knots(p, n)
    grid = compute_greville(p, n, T)

    m = len(grid)

    M = collocation_matrix(p, n, m, T, grid)
    H = histopolation_matrix(p, n, T, grid)

    histoplate = Integral(p, n, T, kind='greville')
    interpolate = Interpolation(p, n, T)

    f = lambda u: u*(1.-u)

    f_0 = inv(M).dot(interpolate(f))
    f_1 = inv(H).dot(histoplate(f))

    # ... compute the l2 error norm for f_0
    from scipy.interpolate import splev

    fh_0 = lambda x: splev(x, tck=(T, f_0, p), der=0)
    diff = lambda x: (f(x) - fh_0(x))**2

    integrate = Integral(p, n, T)
    err_0 = np.sqrt(np.sum(integrate(diff)))
    print ('> l2 error of `f_0` = {}'.format(err_0))
    # ...

    # ...
    TT = T[1:-1]
    pp = p-1

    # scale fh_1 coefficients
    f_1  = np.array([p/(TT[i+p]-TT[i])*c for (i,c) in enumerate(f_1)])
    fh_1 = lambda x: splev(x, tck=(TT, f_1, pp), der=0)
    diff = lambda x: (f(x) - fh_1(x))**2

    integrate = Integral(p, n, T)
    err_1 = np.sqrt(np.sum(integrate(diff)))
    print ('> l2 error of `f_1` = {}'.format(err_1))
    # ...

####################################################################################
if __name__ == '__main__':

    test_projectors_1d()
