# -*- coding: UTF-8 -*-

import numpy as np
from numpy import sin, cos, pi

from spl.core import make_open_knots
from spl.core import collocation_matrix
from spl.core import histopolation_matrix
from spl.core import compute_greville

from spl.utilities import Integral
from spl.utilities import Interpolation
from spl.utilities import Contribution

from spl.feec import mass_matrix
from spl.feec import scaling_matrix
from spl.feec import d_matrix
from spl.feec import get_tck

from scipy.interpolate import splev
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu

# ...
def solve(M, x):
    """Solve y:= Mx using SuperLU."""
    M = csc_matrix(M)
    M_op = splu(M)
    return M_op.solve(x)
# ...

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

    f_0 = solve(M, interpolation(f))
    f_1 = solve(H, histopolation(f))
    f_l2 = solve(mass, contribution(f))

    # ... compute error on H1 for interpolation
    tck = get_tck('H1', p, n, T, f_0)
    fh_0 = lambda x: splev(x, tck)
    diff = lambda x: (f(x) - fh_0(x))**2

    integrate = Integral(p, n, T)
    err_0 = np.sqrt(np.sum(integrate(diff)))
    # ...

    # ... compute error on L2
    # scale fh_1 coefficients
    pp = p-1
    nn = n-1
    TT = T[1:-1]
    S = scaling_matrix(pp, nn, TT)
    f_1  = S.dot(f_1)
    tck = get_tck('L2', p, n, T, f_1)
    fh_1 = lambda x: splev(x, tck)
    diff = lambda x: (f(x) - fh_1(x))**2

    integrate = Integral(p, n, T)
    err_1 = np.sqrt(np.sum(integrate(diff)))
    # ...

    # ... compute error on H1 for L2 projection
    tck = get_tck('H1', p, n, T, f_l2)
    fh_0 = lambda x: splev(x, tck)
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


####################################################################################
if __name__ == '__main__':

    test_projectors_1d(verbose=True)
