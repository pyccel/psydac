// CUDA version of stencil_transpose_2d (stencil_transpose_2d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

/**
 * Transpose of a stencil matrix, matT = mat.T on the owned rows of matT, as the pyccel kernel
 * stencil_transpose_2d.
 *
 * One thread per entry (i1, i2, d1, d2) of `matT` (n_threads = matT.size, see __init__.py), the last axis
 * varying fastest. A thread outside the owned rows of matT, or at a diagonal the pyccel kernel does not
 * write (per direction k: d_k >= 2 * p_in[k] + 1, or >= 2 * p_in[k] + add[k] in the last owned row),
 * returns without writing; every other thread copies one entry, as in the 1D kernel.
 *
 * @param mat   matrix data, shape (rows of the codomain of mat..., diagonals...)
 * @param matT  data of the transposed matrix; the owned rows are written
 * @param s_in, e_in, p_in, add, s_out, e_out, p_out  per direction (length 2), as in the 1D kernel
 *              (e_in is not needed in 2D, `mat` is a view)
 */
extern "C" __global__ void stencil_transpose_2d(Array4D<double> mat, Array4D<double> matT, const long long* s_in,
                                                const long long* e_in, const long long* p_in,
                                                const long long* add, const long long* s_out,
                                                const long long* e_out, const long long* p_out)
{
    CUNUMPY_THREAD_1D(thread, matT.size());

    long long rest = thread;
    const long long d2 = rest % matT.shape[3];
    rest /= matT.shape[3];
    const long long d1 = rest % matT.shape[2];
    rest /= matT.shape[2];
    const long long i2_loc = rest % matT.shape[1] - p_out[1];  // local row indices of matT
    const long long i1_loc = rest / matT.shape[1] - p_out[0];
    if (i1_loc < 0 || i1_loc > e_out[0] - s_out[0]) return;
    if (i2_loc < 0 || i2_loc > e_out[1] - s_out[1]) return;
    const long long i1 = s_out[0] + i1_loc;  // global row indices of matT = global column indices of mat
    const long long i2 = s_out[1] + i2_loc;
    if (d1 >= ((i1 == e_out[0]) ? 2 * p_in[0] + add[0] : 2 * p_in[0] + 1)) return;
    if (d2 >= ((i2 == e_out[1]) ? 2 * p_in[1] + add[1] : 2 * p_in[1] + 1)) return;

    const long long j1 = i1 - p_in[0] + d1;  // global column indices of matT
    const long long j2 = i2 - p_in[1] + d2;
    const long long j1_loc = j1 - s_in[0];  // local column indices of matT = local row indices of mat
    const long long j2_loc = j2 - s_in[1];

    matT(p_out[0] + i1_loc, p_out[1] + i2_loc, d1, d2) =
        mat(p_in[0] + j1_loc, p_in[1] + j2_loc, p_out[0] + i1 - j1, p_out[1] + i2 - j2);
}
