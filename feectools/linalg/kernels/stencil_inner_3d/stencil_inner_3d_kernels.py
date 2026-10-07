"""Inner product of two stencil vectors over their owned (non-ghost) entries (3D).

The host version of ``stencil_inner_3d``. Moved from ``feectools.linalg.kernels.inner_kernels.inner_3d``, with the
result added to ``res[0]`` instead of returned, because a CUDA kernel cannot return a value: the CUDA version in
``stencil_inner_3d_cuda.cu`` takes the same arguments in the same order.

!!! Conjugate on the first argument (see StencilVectorSpace.inner) !!!
"""

from typing import TypeVar

T = TypeVar('T', float, complex)


def stencil_inner_3d(v1: 'T[:,:,:]', v2: 'T[:,:,:]', nghost0: 'int64', nghost1: 'int64', nghost2: 'int64', res: 'T[:]'):
    """
    Add the inner product of two 3D stencil vectors to ``res[0]``.

    Parameters
    ----------
    v1, v2 : 3D NumPy array
        Data of the vectors from which we are computing the inner product.

    nghost0 : int
        Number of ghost cells of the arrays along the index 0.

    nghost1 : int
        Number of ghost cells of the arrays along the index 1.

    nghost2 : int
        Number of ghost cells of the arrays along the index 2.

    res : 1D NumPy array
        The inner product (real or complex) is added to ``res[0]``; the caller sets it to zero first.
    """
    shape0, shape1, shape2 = v1.shape

    part = v1[0, 0, 0] - v1[0, 0, 0]
    for i0 in range(nghost0, shape0 - nghost0):
        for i1 in range(nghost1, shape1 - nghost1):
            for i2 in range(nghost2, shape2 - nghost2):
                part += v1[i0, i1, i2].conjugate() * v2[i0, i1, i2]

    res[0] += part
