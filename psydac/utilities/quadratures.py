# -*- coding: UTF-8 -*-
#! /usr/bin/python

"""
This module contains some routines to generate quadrature points in 1D
it has also a routine uniform, which generates uniform points
with weights equal to 1
"""

import numpy as np


# ....
def gauss_legendre(ordergl,tol=10e-14):
    """
    Returns nodal abscissas {x} and weights {A} of
    Gauss-Legendre m-point quadrature.
    """
    m = ordergl + 1
    from math import cos,pi
    from numpy import zeros

    def legendre(t,m):
        p0 = 1.0; p1 = t
        for k in range(1,m):
            p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
            p0 = p1; p1 = p
        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p1,dp

    A = zeros(m)
    x = zeros(m)
    nRoots = (m + 1)// 2          # Number of non-neg. roots
    for i in range(nRoots):
        t = cos(pi*(i + 0.75)/(m + 0.5))  # Approx. root
        for j in range(30):
            p,dp = legendre(t,m)          # Newton-Raphson
            dt = -p/dp; t = t + dt        # method
            if abs(dt) < tol:
                x[i] = t; x[m-i-1] = -t
                A[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                A[m-i-1] = A[i]
                break
    return x,A
# ...

# ...
def gauss_lobatto(k):
    """
    Returns nodal abscissas {x} and weights {A} of
    Gauss-Legendre m-point quadrature.
    """
    beta = .5 / np.sqrt(1-(2 * np.arange(1., k + 1)) ** (-2)) #3-term recurrence coeffs
    beta[-1] = np.sqrt((k / (2 * k-1.)))
    T = np.diag(beta, 1) + np.diag(beta, -1) # jacobi matrix
    D, V = np.linalg.eig(T) # eigenvalue decomposition
    xg = np.real(D); i = xg.argsort(); xg.sort() # nodes (= Legendres points)
    w = 2 * (V[0, :]) ** 2; # weights

    return xg, w[i]
# ....

# ....
def quadrature(a, k, method="legendre"):
    """
    this routine generates a quad pts on the grid linspace(a,b,N)
    """

    if method == "legendre":
        x, w = gauss_legendre(k)
    elif method == "lobatto":
        x, w = gauss_lobatto(k)
    else:
        raise NotImplemented("> Only Gauss-Legendre is implemented.")

    grid = a
    N = len(a)
    xgl = np.zeros((N-1, k + 1))
    wgl = np.zeros((N-1, k + 1))
    for i in range (0, N-1):
        xmin = grid[i];xmax = grid[i + 1];dx = 0.5 * (xmax-xmin)
        tab = dx * x + dx + xmin
        xgl[i, :] = tab[::-1]
        wgl[i, :] = 0.5 * ( xmax - xmin ) * w

    return xgl,wgl
# ....
