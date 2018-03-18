# -*- coding: UTF-8 -*-

from scipy import kron
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.linalg import block_diag
from scipy.sparse import identity

from numpy import zeros
from numpy import array

# TODO: - docstrings + examples

# ...
def build_kron_matrix(p, n, T, kind):
    """."""
    from spl.core import collocation_matrix
    from spl.core import histopolation_matrix
    from spl.core import compute_greville

    if not isinstance(p, (tuple, list)) or not isinstance(n, (tuple, list)):
        raise TypeError('Wrong type for n and/or p. must be tuple or list')

    assert(len(kind) == len(T))

    grid = [compute_greville(_p, _n, _T) for (_n,_p,_T) in zip(n, p, T)]

    Ms = []
    for i in range(0, len(p)):
        _p = p[i]
        _n = n[i]
        _T = T[i]
        _grid = grid[i]
        _kind = kind[i]

        if _kind == 'interpolation':
            _kind = 'collocation'
        else:
            assert(_kind == 'histopolation')

        func = eval('{}_matrix'.format(_kind))
        M = func(_p, _n, _T, _grid)
        M = csr_matrix(M)

        Ms.append(M.todense()) # kron expects dense matrices

    return kron(*Ms)
# ...


def build_matrices_2d_H1(p, n, T):
    """."""

    # H1
    M0 = build_kron_matrix(p, n, T, kind=['interpolation', 'interpolation'])

    # H-curl
    A = build_kron_matrix(p, n, T, kind=['histopolation', 'interpolation'])
    B = build_kron_matrix(p, n, T, kind=['interpolation', 'histopolation'])
    M1 = block_diag(A, B)

    # L2
    M2 = build_kron_matrix(p, n, T, kind=['histopolation', 'histopolation'])

    return M0, M1, M2


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
    mass = zeros((n,n))
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


class Interpolation2D(object):
    """.

    p: list
        spline degrees

    n: list
        number of splines functions for each direction

    T: list
        knot vectors for each direction

    k: list
        quadrature order for each direction. if not given it will be p+1
    """

    def __init__(self, p, n, T, k=None):

        from spl.utilities import Integral
        from spl.utilities import Interpolation

        if not isinstance(p, (tuple, list)) or not isinstance(n, (tuple, list)):
            raise TypeError('Wrong type for n and/or p. must be tuple or list')

        Is = []
        Hs = []
        for i in range(0, len(p)):
            _k = None
            if not(k is None):
                _k = k[i]

            _interpolation = Interpolation(p[i], n[i], T[i])
            _integration   = Integral(p[i], n[i], T[i], kind='greville', k=_k)

            Is.append(_interpolation)
            Hs.append(_integration)

        self._interpolate = Is
        self._integrate   = Hs

        self._p = p
        self._n = n
        self._T = T

    @property
    def sites(self):
        return [i.sites for i in self._interpolate]

    def __call__(self, kind, f):
        """Computes the integral of the function f over each element of the grid."""
        if kind == 'H1':
            F = zeros(self._n)
            for i,xi in enumerate(self.sites[0]):
                F[i,:] = self._interpolate[1](lambda y: f(xi, y))
            return F

        elif kind == 'Hcurl':
            n0 = (self._n[0]-1, self._n[1])
            n1 = (self._n[0], self._n[1]-1)

            F0 = zeros(n0)
            F1 = zeros(n1)

            _f = lambda x,y: f(x,y)[0]
            for j,yj in enumerate(self.sites[1]):
                F0[:,j] = self._integrate[0](lambda x: _f(x, yj))

            _f = lambda x,y: f(x,y)[1]
            for i,xi in enumerate(self.sites[0]):
                F1[i,:] = self._integrate[1](lambda y: _f(xi, y))

            return F0, F1

        elif kind == 'L2':
            from spl.utilities import integrate_2d

            points = (self._integrate[0]._points, self._integrate[1]._points)
            weights = (self._integrate[0]._weights, self._integrate[1]._weights)

            return integrate_2d(points, weights, f)

        else:
            raise NotImplementedError('Only H1, Hcurl and L2 are available')


def scaling_matrix(p, n, T, kind=None):
    """Returns the scaling matrix for M-splines.
    It is a diagonal matrix whose elements are (p+1)/(T[i+p+1]-T[i])


    """
    if isinstance(p, int):
        x = zeros(n)
        for i in range(0, n):
            x[i] = (p+1)/(T[i+p+1]-T[i])
        return diags(x)

    assert(isinstance(p, (list, tuple)))

    if kind is None:
        Ms = []
        for i in range(0, len(p)):
            M = scaling_matrix(p[i], n[i], T[i])
            # we must convert it to dense, otherwise we get a scipy
            Ms.append(M.todense())
        return kron(*Ms)

    elif kind == 'Hcurl':
        p0 = p[0] ; n0 = n[0] ; T0 = T[0]
        p1 = p[1] ; n1 = n[1] ; T1 = T[1]

        I0 = identity(n0)
        I1 = identity(n1)
        S0 = scaling_matrix(p0-1, n0-1, T0[1:-1])
        S1 = scaling_matrix(p1-1, n1-1, T1[1:-1])

        I0 = I0.todense()
        I1 = I1.todense()
        S0 = S0.todense()
        S1 = S1.todense()

        M0 = kron(S0, I1)
        M1 = kron(I0, S1)
        return block_diag(M0, M1)

    elif kind == 'L2':
        pp = list(p)
        nn = list(n)
        TT = list(T)
        for i in range(0, len(p)):
            pp[i] -= 1
            nn[i] -= 1
            TT[i] = TT[i][1:-1]
        return scaling_matrix(pp, nn, TT)

    raise NotImplementedError('TODO')


def mass_matrix_H1(p, n, T):
    """Returns the 2D/3D mass matrix over H1."""

    assert(isinstance(p, (list, tuple)))

    Ms = []
    for i in range(0, len(p)):
        M = mass_matrix(p[i], n[i], T[i])
        M = csr_matrix(M)
        # we must convert it to dense, otherwise we get a scipy
        Ms.append(M.todense())
    M = kron(*Ms)
    return csr_matrix(M)

def mass_matrix_L2(p, n, T):
    """Returns the 2D/3D mass matrix over L2."""

    assert(isinstance(p, (list, tuple)))

    pp = list(p)
    nn = list(n)
    TT = list(T)
    for i in range(0, len(p)):
        pp[i] -= 1
        nn[i] -= 1
        TT[i] = TT[i][1:-1]

    M = mass_matrix_H1(pp, nn, TT)
    return csr_matrix(M)

def mass_matrix_Hcurl(p, n, T):
    """Returns the 2D mass matrix over Hcurl."""
    pp = list(p) ; pp[0] -= 1
    nn = list(n) ; nn[0] -= 1
    TT = list(T) ; TT[0] = TT[0][1:-1]
    M0 = mass_matrix_H1(pp, nn, TT)

    pp = list(p) ; pp[1] -= 1
    nn = list(n) ; nn[1] -= 1
    TT = list(T) ; TT[1] = TT[1][1:-1]
    M1 = mass_matrix_H1(pp, nn, TT)

    M = block_diag(M0.todense(), M1.todense())
    return csr_matrix(M)

def build_mass_matrices(p, n, T):
    """Returns all mass matrices over the sequence H1 -> Hcurl -> L2."""
    M0 = mass_matrix_H1(p, n, T)
    M1 = mass_matrix_Hcurl(p, n, T)
    M2 = mass_matrix_L2(p, n, T)
    return M0, M1, M2


def _tck_H1_1D(p, n, T, c):
    return (T, c, p)

def _tck_L2_1D(p, n, T, c):
    pp = p-1
    nn = n-1
    TT = T[1:-1]
    return (TT, c, pp)

def _tck_H1_2D(p, n, T, c):
    return (T[0], T[1], c, p[0], p[1])

def _tck_Hcurl_2D(p, n, T, c):
    """."""
    pp = list(p) ; pp[0] -= 1
    nn = list(n) ; nn[0] -= 1
    TT = list(T) ; TT[0] = TT[0][1:-1]
    N = array(nn).prod()
    c0 = c[:N]
    tck0 = (TT[0], TT[1], c0, pp[0], pp[1])

    pp = list(p) ; pp[1] -= 1
    nn = list(n) ; nn[1] -= 1
    TT = list(T) ; TT[1] = TT[1][1:-1]
    c1 = c[N:]
    tck1 = (TT[0], TT[1], c1, pp[0], pp[1])

    return tck0, tck1

def _tck_L2_2D(p, n, T, c):
    return (T[0][1:-1], T[1][1:-1], c, p[0]-1, p[1]-1)

def get_tck(kind, p, n, T, c):
    """Returns the tck for a given space kind."""
    if isinstance(p, int):
        assert(kind in ['H1', 'L2'])
        func = eval('_tck_{}_1D'.format(kind))

    if isinstance(p, (list, tuple)):
        assert(kind in ['H1', 'Hcurl', 'Hdiv', 'L2'])
        if len(p) == 2:
            func = eval('_tck_{}_2D'.format(kind))
        else:
            raise NotImplementedError('Only 2D is available')

    return func(p, n, T, c)

