# -*- coding: UTF-8 -*-
#! /usr/bin/python

import numpy as np

__all__ = ["collocation_cardinal_splines", "collocation_matrix_spline"]

# ...
def collocation_cardinal_splines(p, n):
    """
    Returns the collocation matrix with cardinal bsplines.
    """
    from spl.core.django import m_django_core as core

    mat = core.utilities_collocation_cardinal_splines(p, n)

    from scipy.sparse import csr_matrix
    mat = csr_matrix(mat)

    return mat
# ...

# ...
def collocation_matrix_spline(p, n, m, T, u):
    """
    Returns the collocation matrix spline
    """
    from spl.core.django import m_django_core as core

    mat = core.utilities_collocation_matrix(p, n, m, T, u)

    from scipy.sparse import csr_matrix
    mat = csr_matrix(mat)

    return mat
# ...

# ...
def make_open_knots(p, n):
    """
    Create open knot sequence.

    Parameters
    ----------
    p : int
        Spline degree.

    n : int
        Number of control points
        (= number of degrees of freedom = dimension of linear space).

    Returns
    -------
    T : numpy.ndarray (n+p+1)
        Open knot sequence (1st and last knots are repeated p+1 times).

    """
    from spl.core.django import m_django_core as core

    T = core.utilities_make_open_knots(p, n)

    return T
# ...

# ...
def compute_spans(p, n, T):
    """
    """
    from spl.core.django import m_django_core as core

    n_elements = len(np.unique(T)) - 1
    spans = core.utilities_compute_spans(p, n, T, n_elements)

    return spans
# ...

# ...
def compute_origins_element(p, n, T):
    """
    """
    from spl.core.django import m_django_core as core

    origins_element = core.utilities_compute_origins_element(p, n, T)

    return origins_element
# ...

# ...
def construct_grid_from_knots(p, n, T):
    """
    Extract list of breakpoints from open knot sequence.

    Parameters
    ----------
    p : int
        Spline degree.

    n : int
        Number of control points
        (= number of degrees of freedom = dimension of linear space).

    T : array_like (n+p+1)
        Open knot sequence (1st and last knots are repeated p+1 times).

    Returns
    -------
    grid : numpy.ndarray (n-p+1)
        List of breakpoints (repeated endpoints are eliminated).

    """
    from spl.core.django import m_django_core as core

    n_elements = len(np.unique(T)) - 1
    grid = core.utilities_construct_grid_from_knots(p, n, n_elements, T)

    return grid
# ...

# ...
def construct_quadrature_grid(u, w, grid):
    """
    """
    from spl.core.django import m_django_core as core

    n_elements = len(grid) - 1
    k = len(u)
    points, weights = core.utilities_construct_quadrature_grid(n_elements, k, u, w, grid)
    return points, weights
# ...

# ...
def eval_on_grid_splines_ders(p, n, d, T, points):
    """
    """
    from spl.core.django import m_django_core as core

    k          = points.shape[0]
    n_elements = points.shape[1]

    basis = core.utilities_eval_on_grid_splines_ders(p, n, n_elements, k, d, T, points)
    return basis
# ...
