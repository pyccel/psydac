# -*- coding: UTF-8 -*-

import numpy as np
from numpy import zeros
from numpy import concatenate
from numpy import block

from scipy import kron
from scipy.sparse import identity
from scipy.sparse import csr_matrix

def d_matrix(n):
    """creates a 1d incidence matrix.
    The final matrix will have a shape of (n,n-1)

    n: int
        number of nodes
    """
    M = zeros((n,n))
    for i in range(0, n):
        M[i,i] = 1.
        if i>0:
            M[i,i-1] = -1.
    return csr_matrix(M[1:n,:])

class Grad(object):
    def __init__(self, p, n, T):
        self._p = p
        self._n = n
        self._T = T

        n0 = n[0]
        n1 = n[1]

        I0 = identity(n0)
        I1 = identity(n1)
        D0 = d_matrix(n0)
        D1 = d_matrix(n1)

        I0 = I0.todense()
        I1 = I1.todense()
        D0 = D0.todense()
        D1 = D1.todense()

        A = kron(D0, I1)
        B = kron(I0, D1)
        self._matrix = concatenate((A, B), axis=0)
        self._matrix = csr_matrix(self._matrix)

    @property
    def shape(self):
        return self._matrix.shape

    def __cal__(self, x):
        return self._matrix.dot(x)

class Curl(object):
    def __init__(self, p, n, T):
        self._p = p
        self._n = n
        self._T = T

        n0 = n[0]
        n1 = n[1]

        I0 = identity(n0-1)
        I1 = identity(n1-1)
        D0 = d_matrix(n0)
        D1 = d_matrix(n1)

        I0 = I0.todense()
        I1 = I1.todense()
        D0 = D0.todense()
        D1 = D1.todense()

        A = kron(D0, I1)
        B = kron(I0, D1)
        self._matrix = block([-B, A])
        self._matrix = csr_matrix(self._matrix)

    @property
    def shape(self):
        return self._matrix.shape

    def __cal__(self, x):
        return self._matrix.dot(x)

