"""``y = alpha * x + y`` for the data of two stencil vectors, ghost regions included (2D).

The host version of ``stencil_axpy_2d``. Moved from ``feectools.linalg.kernels.axpy_kernels.axpy_2d``; the CUDA version in
``stencil_axpy_2d_cuda.cu`` takes the same arguments in the same order.
"""

from typing import TypeVar

T = TypeVar('T', float, complex)


def stencil_axpy_2d(alpha: 'T', x: 'T[:,:]', y: 'T[:,:]'):
    """
    Compute y = alpha * x + y.

    Parameters
    ----------
    alpha : float | complex
        Scaling coefficient.

    x, y : 2D NumPy arrays of (float | complex) data
        Data of the vectors.
    """
    n0, n1 = x.shape
    for i0 in range(n0):
        for i1 in range(n1):
            y[i0, i1] += alpha * x[i0, i1]
