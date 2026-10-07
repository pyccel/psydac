"""``y = alpha * x + y`` for the data of two stencil vectors, ghost regions included (1D).

The host version of ``stencil_axpy_1d``. Moved from ``feectools.linalg.kernels.axpy_kernels.axpy_1d``; the CUDA version in
``stencil_axpy_1d_cuda.cu`` takes the same arguments in the same order.
"""

from typing import TypeVar

T = TypeVar('T', float, complex)


def stencil_axpy_1d(alpha: 'T', x: 'T[:]', y: 'T[:]'):
    """
    Compute y = alpha * x + y.

    Parameters
    ----------
    alpha : float | complex
        Scaling coefficient.

    x, y : 1D NumPy arrays of (float | complex) data
        Data of the vectors.
    """
    n0, = x.shape
    for i0 in range(n0):
        y[i0] += alpha * x[i0]
