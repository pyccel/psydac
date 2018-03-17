# -*- coding: UTF-8 -*-

import numpy as np
from numpy import sin, cos, pi
from numpy import bmat

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
from spl.feec import mass_matrix
from spl.feec import Interpolation2D

from scipy.linalg import inv
from scipy import kron
from scipy.linalg import block_diag
from scipy.interpolate import splev


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

    # ...
    interpolate = Interpolation2D(p, n, T)

    interpolate_H1 = lambda f: interpolate('H1', f)
    interpolate_Hcurl = lambda f: interpolate('Hcurl', f)
    interpolate_L2 = lambda f: interpolate('L2', f)
    # ...

    # ... H1
    f = lambda x,y: sin(2.*pi*x) * sin(2.*pi*y)
    F = interpolate_H1(f)
    # ...

    # ... Hcurl
    g0 = lambda x,y: cos(2.*pi*x) * sin(2.*pi*y)
    g1 = lambda x,y: sin(2.*pi*x) * cos(2.*pi*y)
    g  = lambda x,y: [g0(x,y), g1(x,y)]

    G = interpolate_Hcurl(g)
    # ...

    # ... L2
    h = lambda x,y: cos(2.*pi*x) * cos(2.*pi*y)
    H = interpolate_L2(h)
    # ...

    # ...
    if verbose:
        print ('==== testing projection in 2d ====')
        print ('> M0.shape  := {}'.format(M0.shape))
        print ('> M1.shape  := {}'.format(M1.shape))
        print ('> M2.shape  := {}'.format(M2.shape))

        print ('> F.shape  := {}'.format(F.shape))
        print ('> G.shapes := {0} | {1}'.format(G[0].shape, G[1].shape))
        print ('> H.shape  := {}'.format(H.shape))
    # ...





####################################################################################
if __name__ == '__main__':

    test_projectors_1d(verbose=True)
    print('')
    test_projectors_2d(verbose=True)
