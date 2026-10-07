"""Stencil matrix-vector product ``out = mat @ x`` on the owned rows (1D).

The host version of ``stencil_dot_1d``.

Moved from ``feectools.linalg.stencil_dot_kernels.matvec_1d_kernel``. The CUDA version in ``stencil_dot_1d_cuda.cu``
takes the same arguments in the same order.
"""


def stencil_dot_1d(mat: 'float[:, :]',
                   x: 'float[:]',
                   out: 'float[:]',
                   s_in: int,
                   p_in: int,
                   add: int,
                   s_out: int,
                   e_out: int,
                   p_out: int):

    for i1 in range(s_out, e_out):  # global row index
        i1_loc = i1 - s_out  # local row index
        val = 0.
        for d1 in range(2*p_in + 1):
            val += mat[p_out + i1_loc, d1] * x[i1 + d1 - s_in]

        out[p_out + i1_loc] = val

    # last row treated separately
    i1 = e_out
    i1_loc = i1 - s_out  # local row index
    val = 0.
    for d1 in range(2*p_in + add):
        val += mat[p_out + i1_loc, d1] * x[i1 + d1 - s_in]

    out[p_out + i1_loc] = val
