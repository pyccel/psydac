"""Transpose of a stencil matrix, ``matT = mat.T`` on the owned rows of ``matT`` (1D).

The host version of ``stencil_transpose_1d``.

Moved from ``feectools.linalg.stencil_transpose_kernels.transpose_1d_kernel``. The CUDA version in
``stencil_transpose_1d_cuda.cu`` takes the same arguments in the same order. ``e_in`` (the end of the row range of ``mat``) is not
needed here; the CUDA version needs it for the row extents of ``mat``, which it reads through a raw
pointer in 3D.
"""


def stencil_transpose_1d(mat: 'float[:, :]',
                         matT: 'float[:, :]',
                         s_in: int,  # refers to matT
                         e_in: int,  # refers to matT; used by the CUDA version only
                         p_in: int,
                         add: int,
                         s_out: int,
                         e_out: int,
                         p_out: int):

    for i1 in range(s_out, e_out):  # global row index of matT = global column index of mat
        i1_loc = i1 - s_out  # local row index of matT
        for d1 in range(2*p_in + 1):
            j1 = i1 - p_in + d1  # global column index of matT
            j1_loc = j1 - s_in  # local column index of matT = local row index of mat

            matT[p_out + i1_loc, d1] = mat[p_in + j1_loc, p_out + i1 - j1]

    # last row treated separately
    i1 = e_out
    i1_loc = i1 - s_out  # local row index of matT
    for d1 in range(2*p_in + add):
        j1 = i1 - p_in + d1  # global column index of matT
        j1_loc = j1 - s_in  # local column index of matT = local row index of mat

        matT[p_out + i1_loc, d1] = mat[p_in + j1_loc, p_out + i1 - j1]
