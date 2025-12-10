# -*- coding: UTF-8 -*-

from scipy import kron
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.linalg import block_diag
from scipy.sparse import identity

from numpy import zeros
from numpy import array

# TODO: - docstrings + examples

def _mass_matrix_H1(p, n, T):
    """Returns the 2D/3D mass matrix over H1."""

    from psydac.core.interface import mass_matrix

    assert(isinstance(p, (list, tuple)))

    Ms = []
    for i in range(0, len(p)):
        M = mass_matrix(p[i], n[i], T[i])
        M = csr_matrix(M)
        # we must convert it to dense, otherwise we get a scipy
        Ms.append(M.toarray())
    M = kron(*Ms)
    return csr_matrix(M)

def _mass_matrix_L2(p, n, T):
    """Returns the 2D/3D mass matrix over L2."""

    assert(isinstance(p, (list, tuple)))

    pp = list(p)
    nn = list(n)
    TT = list(T)
    for i in range(0, len(p)):
        pp[i] -= 1
        nn[i] -= 1
        TT[i] = TT[i][1:-1]

    M = _mass_matrix_H1(pp, nn, TT)
    return csr_matrix(M)

def _mass_matrix_Hcurl_2D(p, n, T):
    """Returns the 2D mass matrix over Hcurl."""
    pp = list(p) ; pp[0] -= 1
    nn = list(n) ; nn[0] -= 1
    TT = list(T) ; TT[0] = TT[0][1:-1]
    M0 = _mass_matrix_H1(pp, nn, TT)

    pp = list(p) ; pp[1] -= 1
    nn = list(n) ; nn[1] -= 1
    TT = list(T) ; TT[1] = TT[1][1:-1]
    M1 = _mass_matrix_H1(pp, nn, TT)

    M = block_diag(M0.toarray(), M1.toarray())
    return csr_matrix(M)


# This is a user-friendly function.
def mass_matrices(p, n, T):
    """Returns all mass matrices.
    """
    # 1d case
    if isinstance(p, int):
        from psydac.core.interface import mass_matrix

        M0 = mass_matrix(p, n, T)

        pp = p-1 ; nn = n-1 ; TT = T[1:-1]
        M1 = mass_matrix(p, n, T)

        return M0, M1

    if not isinstance(p, (list, tuple)):
        raise TypeError('Expecting p to be int or list/tuple')

    if len(p) == 2:
        # TODO improve
        # we only treat the sequence H1 -> Hcurl -> L2
        M0 = _mass_matrix_H1(p, n, T)
        M1 = _mass_matrix_Hcurl_2D(p, n, T)
        M2 = _mass_matrix_L2(p, n, T)
        return M0, M1, M2

    raise NotImplementedError('only 1d and 2D are available')


# ...
def build_kron_matrix(p, n, T, kind):
    """."""
    from psydac.core.interface import collocation_matrix
    from psydac.core.interface import histopolation_matrix
    from psydac.core.interface import compute_greville

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

        Ms.append(M.toarray()) # kron expects dense matrices

    return kron(*Ms)
# ...


def _interpolation_matrices_2d(p, n, T):
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


def interpolation_matrices(p, n, T):
    """Returns all interpolation matrices.
    This is a user-friendly function.
    """
    # 1d case
    if isinstance(p, int):
        from psydac.core.interface import compute_greville
        from psydac.core.interface import collocation_matrix
        from psydac.core.interface import histopolation_matrix

        grid = compute_greville(p, n, T)

        M = collocation_matrix(p, n, T, grid)
        H = histopolation_matrix(p, n, T, grid)

        return M, H

    if not isinstance(p, (list, tuple)):
        raise TypeError('Expecting p to be int or list/tuple')

    if len(p) == 2:
        return _interpolation_matrices_2d(p, n, T)

    raise NotImplementedError('only 1d and 2D are available')



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

        from psydac.utilities.integrate import Integral
        from psydac.utilities.integrate import Interpolation

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
            from psydac.utilities.integrate import integrate_2d

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
        if kind is None:
            x = zeros(n)
            for i in range(0, n):
                x[i] = (p+1)/(T[i+p+1]-T[i])
            return diags(x)
        elif kind == 'L2':
            return scaling_matrix(p-1, n-1, T[1:-1])
        else:
            raise ValueError('Unexpected kind of scaling matrix for 1D')

    assert(isinstance(p, (list, tuple)))

    if kind is None:
        Ms = []
        for i in range(0, len(p)):
            M = scaling_matrix(p[i], n[i], T[i])
            # we must convert it to dense, otherwise we get a scipy
            Ms.append(M.toarray())
        return kron(*Ms)

    elif kind == 'Hcurl':
        p0 = p[0] ; n0 = n[0] ; T0 = T[0]
        p1 = p[1] ; n1 = n[1] ; T1 = T[1]

        I0 = identity(n0)
        I1 = identity(n1)
        S0 = scaling_matrix(p0-1, n0-1, T0[1:-1])
        S1 = scaling_matrix(p1-1, n1-1, T1[1:-1])

        I0 = I0.toarray()
        I1 = I1.toarray()
        S0 = S0.toarray()
        S1 = S1.toarray()

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

