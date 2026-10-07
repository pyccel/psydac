// CUDA version of stencil_transpose_3d (stencil_transpose_3d_kernels.py), with the same arguments in the same order.
#include "cunumpy/index.cuh"

/**
 * Transpose of a stencil matrix, matT = mat.T on the owned rows of matT, as the pyccel kernel
 * stencil_transpose_3d.
 *
 * One thread per entry (i1, i2, i3, d1, d2, d3) of `matT` (n_threads = matT.size, see __init__.py), the last
 * axis varying fastest. A thread outside the owned rows of matT, or at a diagonal the pyccel kernel does
 * not write (per direction k: d_k >= 2 * p_in[k] + 1, or >= 2 * p_in[k] + add[k] in the last owned row),
 * returns without writing; every other thread copies one entry, as in the 1D kernel.
 *
 * Both matrices have six axes and cunumpy's array views stop at four, so they are raw pointers to
 * C-contiguous data (checked by cunumpy at the call), with the shapes of stencil matrices without shifts:
 * matT has the rows e_out - s_out + 1 + 2 * p_out and the diagonals 2 * p_in + 1, mat the rows
 * e_in - s_in + 1 + 2 * p_in and the diagonals 2 * p_out + 1 (per direction). StencilMatrix checks this
 * before calling the precompiled kernels (see StencilMatrix.set_backend).
 *
 * @param mat   matrix data, C-contiguous, shape as above
 * @param matT  data of the transposed matrix, C-contiguous, shape as above; the owned rows are written
 * @param s_in, e_in, p_in, add, s_out, e_out, p_out  per direction (length 3), as in the 1D kernel; e_in is
 *              the global end (inclusive) of the rows of mat, used for the shape of mat
 */
extern "C" __global__ void stencil_transpose_3d(const double* mat, double* matT, const long long* s_in,
                                                const long long* e_in, const long long* p_in,
                                                const long long* add, const long long* s_out,
                                                const long long* e_out, const long long* p_out)
{
    // shapes of matT (rows, diagonals) and mat (rows, diagonals), per direction
    long long rows_T[3], diags_T[3], rows[3], diags[3];
    for (int k = 0; k < 3; ++k) {
        rows_T[k] = e_out[k] - s_out[k] + 1 + 2 * p_out[k];
        diags_T[k] = 2 * p_in[k] + 1;
        rows[k] = e_in[k] - s_in[k] + 1 + 2 * p_in[k];
        diags[k] = 2 * p_out[k] + 1;
    }
    const long long size_T = rows_T[0] * rows_T[1] * rows_T[2] * diags_T[0] * diags_T[1] * diags_T[2];
    CUNUMPY_THREAD_1D(thread, size_T);

    long long rest = thread;
    const long long d3 = rest % diags_T[2];
    rest /= diags_T[2];
    const long long d2 = rest % diags_T[1];
    rest /= diags_T[1];
    const long long d1 = rest % diags_T[0];
    rest /= diags_T[0];
    const long long i3_loc = rest % rows_T[2] - p_out[2];  // local row indices of matT
    rest /= rows_T[2];
    const long long i2_loc = rest % rows_T[1] - p_out[1];
    const long long i1_loc = rest / rows_T[1] - p_out[0];
    if (i1_loc < 0 || i1_loc > e_out[0] - s_out[0]) return;
    if (i2_loc < 0 || i2_loc > e_out[1] - s_out[1]) return;
    if (i3_loc < 0 || i3_loc > e_out[2] - s_out[2]) return;
    const long long i1 = s_out[0] + i1_loc;  // global row indices of matT = global column indices of mat
    const long long i2 = s_out[1] + i2_loc;
    const long long i3 = s_out[2] + i3_loc;
    if (d1 >= ((i1 == e_out[0]) ? 2 * p_in[0] + add[0] : 2 * p_in[0] + 1)) return;
    if (d2 >= ((i2 == e_out[1]) ? 2 * p_in[1] + add[1] : 2 * p_in[1] + 1)) return;
    if (d3 >= ((i3 == e_out[2]) ? 2 * p_in[2] + add[2] : 2 * p_in[2] + 1)) return;

    const long long j1 = i1 - p_in[0] + d1;  // global column indices of matT
    const long long j2 = i2 - p_in[1] + d2;
    const long long j3 = i3 - p_in[2] + d3;
    const long long j1_loc = j1 - s_in[0];  // local column indices of matT = local row indices of mat
    const long long j2_loc = j2 - s_in[1];
    const long long j3_loc = j3 - s_in[2];

    const long long index_T =
        ((((((p_out[0] + i1_loc) * rows_T[1] + p_out[1] + i2_loc) * rows_T[2] + p_out[2] + i3_loc) * diags_T[0] +
           d1) * diags_T[1] + d2) * diags_T[2] + d3);
    const long long index =
        ((((((p_in[0] + j1_loc) * rows[1] + p_in[1] + j2_loc) * rows[2] + p_in[2] + j3_loc) * diags[0] +
           p_out[0] + i1 - j1) * diags[1] + p_out[1] + i2 - j2) * diags[2] + p_out[2] + i3 - j3);

    matT[index_T] = mat[index];
}
