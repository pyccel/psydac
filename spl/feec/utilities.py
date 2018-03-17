# -*- coding: UTF-8 -*-

from scipy import kron
from scipy.linalg import block_diag

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

        Ms.append(M)

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
