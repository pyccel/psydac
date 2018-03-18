# -*- coding: UTF-8 -*-

import numpy as np
from numpy import sin, cos, pi

from spl.core import make_open_knots
from spl.core import compute_greville

from spl.utilities import Integral2D

from spl.feec import build_matrices_2d_H1
from spl.feec import build_mass_matrices
from spl.feec import Interpolation2D
from spl.feec import scaling_matrix
from spl.feec import Grad, Curl
from spl.feec import get_tck

from scipy.interpolate import bisplev
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import splu

# ...
def solve(M, x):
    """Solve y:= Mx using SuperLU."""
    M = csc_matrix(M)
    M_op = splu(M)
    return M_op.solve(x)
# ...


def test_projectors_2d(verbose=False):
    # ...
    n_elements = (8, 4)
    p = (3, 3)                                      # spline degree
    n = [_n+_p-1 for (_n,_p) in zip(n_elements, p)] # number of control points
    # ...

    T = [make_open_knots(_p, _n) for (_n,_p) in zip(n, p)]

    M0, M1, M2 = build_matrices_2d_H1(p, n, T)
    mass_0, mass_1, mass_2 = build_mass_matrices(p, n, T)

    grad = Grad(p, n, T)
    curl = Curl(p, n, T)

    # ...
    interpolate = Interpolation2D(p, n, T)

    interpolate_H1 = lambda f: interpolate('H1', f)
    interpolate_Hcurl = lambda f: interpolate('Hcurl', f)
    interpolate_L2 = lambda f: interpolate('L2', f)
    # ...

    # ... H1
    f = lambda x,y: x*(1.-x)*y*(1.-y)
    F = interpolate_H1(f)
    # ...

    # ... Hcurl
    g0 = lambda x,y: (1.-2.*x)*y*(1.-y)
    g1 = lambda x,y: x*(1.-x)*(1.-2.*y)
    g  = lambda x,y: [g0(x,y), g1(x,y)]

    G = interpolate_Hcurl(g)
    # ...

    # ... L2
    h = lambda x,y: x*(1.-x)*y*(1.-y)
    H = interpolate_L2(h)
    # ...

    # ...
    def to_array_H1(X):
        return X.flatten()

    def to_array_Hcurl(X):
        X0 = X[0] ; X1 = X[1]
        x0 = X0.flatten()
        x1 = X1.flatten()
        return np.concatenate([x0, x1])

    def to_array_L2(X):
        return X.flatten()
    # ...

    # ... compute error on H1 for interpolation
    f_0 = solve(M0, to_array_H1(F))
    tck = get_tck('H1', p, n, T, f_0)
    fh_0 = lambda x,y: bisplev(x, y, tck)
    diff = lambda x,y: (f(x,y) - fh_0(x,y))**2

    integrate = Integral2D(p, n, T)
    err_0 = np.sqrt(np.sum(integrate(diff)))
    # ...

    # ... compute error on L2
    h_2 = solve(M2, to_array_L2(H))
    S = scaling_matrix(p, n, T, kind='L2')
    h_2  = S.dot(h_2)
    # TODO improve and remove
    x = np.zeros(h_2.size)
    for i in range(0, h_2.size):
        x[i] = h_2[0,i]
    h_2 = x
    tck = get_tck('L2', p, n, T, h_2)
    hh_0 = lambda x,y: bisplev(x, y, tck)
    diff = lambda x,y: (h(x,y) - hh_0(x,y))**2

    integrate = Integral2D(p, n, T)
    err_2 = np.sqrt(np.sum(integrate(diff)))
    # ...

    # ... compute error on Hcurl
    g_1 = solve(M1, to_array_Hcurl(G))
    S = scaling_matrix(p, n, T, kind='Hcurl')

    g_1  = S.dot(g_1)
    tck0, tck1 = get_tck('Hcurl', p, n, T, g_1)
    gh_0 = lambda x,y: bisplev(x, y, tck0)
    gh_1 = lambda x,y: bisplev(x, y, tck1)
    diff = lambda x,y: (g(x,y)[0] - gh_0(x,y))**2 + (g(x,y)[1] - gh_1(x,y))**2

    integrate = Integral2D(p, n, T)
    err_1 = np.sqrt(np.sum(integrate(diff)))
    # ...

    # ...
    if verbose:
        print ('==== testing projection in 2d ====')
#        print ('> M0.shape  := {}'.format(M0.shape))
#        print ('> M1.shape  := {}'.format(M1.shape))
#        print ('> M2.shape  := {}'.format(M2.shape))
#        print()
#        print ('> mass_0.shape  := {}'.format(mass_0.shape))
#        print ('> mass_1.shape  := {}'.format(mass_1.shape))
#        print ('> mass_2.shape  := {}'.format(mass_2.shape))
#        print()
#        print ('> grad.shape  := {}'.format(grad.shape))
#        print ('> curl.shape  := {}'.format(curl.shape))
#        print()
#        print ('> F.shape  := {}'.format(F.shape))
#        print ('> G.shapes := {0} | {1}'.format(G[0].shape, G[1].shape))
#        print ('> H.shape  := {}'.format(H.shape))
#        print()
        print ('> l2 error of `f_0` = {}'.format(err_0))
        print ('> l2 error of `g_1` = {}'.format(err_1))
        print ('> l2 error of `h_2` = {}'.format(err_2))
    # ...


####################################################################################
if __name__ == '__main__':

    test_projectors_2d(verbose=True)
