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
from spl.utilities import Contribution

from spl.feec import build_matrices_2d_H1

from scipy.linalg import inv
from scipy import kron
from scipy.linalg import block_diag
from scipy.interpolate import splev


def mass_matrix(p, n, T):
    """Returns the 1d mass matrix."""
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

def test_projectors_1d(verbose=False):
    # ...
    n_elements = 4
    p = 3                    # spline degree
    n = n_elements + p - 1   # number of control points
    # ...

    T    = make_open_knots(p, n)
    grid = compute_greville(p, n, T)

    M = collocation_matrix(p, n, T, grid)
    H = histopolation_matrix(p, n, T, grid)
    mass = mass_matrix(p, n, T)

    histopolation = Integral(p, n, T, kind='greville')
    interpolation = Interpolation(p, n, T)
    contribution = Contribution(p, n, T)

    f = lambda u: u*(1.-u)

    f_0 = inv(M).dot(interpolation(f))
    f_1 = inv(H).dot(histopolation(f))
    f_l2 = inv(mass).dot(contribution(f))

    # ... compute error on H1 for interpolation
    fh_0 = lambda x: splev(x, tck=(T, f_0, p), der=0)
    diff = lambda x: (f(x) - fh_0(x))**2

    integrate = Integral(p, n, T)
    err_0 = np.sqrt(np.sum(integrate(diff)))
    # ...

    # ... compute error on L2
    TT = T[1:-1]
    pp = p-1

    # scale fh_1 coefficients
    f_1  = np.array([p/(TT[i+p]-TT[i])*c for (i,c) in enumerate(f_1)])
    fh_1 = lambda x: splev(x, tck=(TT, f_1, pp), der=0)
    diff = lambda x: (f(x) - fh_1(x))**2

    integrate = Integral(p, n, T)
    err_1 = np.sqrt(np.sum(integrate(diff)))
    # ...

    # ... compute error on H1 for L2 projection
    fh_0 = lambda x: splev(x, tck=(T, f_l2, p), der=0)
    diff = lambda x: (f(x) - fh_0(x))**2

    integrate = Integral(p, n, T)
    err_l2 = np.sqrt(np.sum(integrate(diff)))
    # ...

    # ...
    if verbose:
        print ('==== testing projection in 1d ====')
        print ('> l2 error of `f_0` = {}'.format(err_0))
        print ('> l2 error of `f_1` = {}'.format(err_1))
        print ('> l2 error of `f_l2` = {}'.format(err_l2))
    # ...

def test_projectors_2d(verbose=False):
    # ...
    n_elements = (8, 4)
    p = (2, 2)                                      # spline degree
    n = [_n+_p-1 for (_n,_p) in zip(n_elements, p)] # number of control points
    # ...

    T = [make_open_knots(_p, _n) for (_n,_p) in zip(n, p)]

    M0, M1, M2 = build_matrices_2d_H1(p, n, T)


####################################################################################
if __name__ == '__main__':

    test_projectors_1d(verbose=True)
    test_projectors_2d(verbose=True)
