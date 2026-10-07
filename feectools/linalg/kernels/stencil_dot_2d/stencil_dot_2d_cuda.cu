// CUDA version of stencil_dot_2d (stencil_dot_2d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

/**
 * Stencil matrix-vector product out = mat @ x on the owned rows, as the pyccel kernel stencil_dot_2d.
 *
 * One thread per entry of `out` (n_threads = out.size, see __init__.py), the last axis varying fastest. A
 * thread outside the owned rows returns without writing, so the padding of `out` is left as it is, as in
 * pyccel. Along each direction k, interior rows use 2 * p_in[k] + 1 diagonals and the last owned row
 * (i_k == e_out[k]) uses 2 * p_in[k] + add[k]; the pyccel kernel spells out the four combinations, this
 * kernel picks its own per thread. The diagonals are summed in the same order (d1 outer, d2 inner).
 *
 * @param mat   matrix data, shape (rows of `out`..., 2 * p_in + 1...)
 * @param x     data of the domain vector, ghost regions included
 * @param out   data of the codomain vector; the owned rows are written
 * @param s_in, p_in, add, s_out, e_out, p_out  per direction (length 2), as in the 1D kernel
 */
extern "C" __global__ void stencil_dot_2d(Array4D<double> mat, Array2D<double> x, Array2D<double> out,
                                          const long long* s_in, const long long* p_in, const long long* add,
                                          const long long* s_out, const long long* e_out,
                                          const long long* p_out)
{
    CUNUMPY_THREAD_1D(thread, out.size());

    const long long i1_loc = thread / out.shape[1] - p_out[0];  // local row indices
    const long long i2_loc = thread % out.shape[1] - p_out[1];
    if (i1_loc < 0 || i1_loc > e_out[0] - s_out[0]) return;
    if (i2_loc < 0 || i2_loc > e_out[1] - s_out[1]) return;
    const long long i1 = s_out[0] + i1_loc;  // global row indices
    const long long i2 = s_out[1] + i2_loc;

    const long long n_diags1 = (i1 == e_out[0]) ? 2 * p_in[0] + add[0] : 2 * p_in[0] + 1;
    const long long n_diags2 = (i2 == e_out[1]) ? 2 * p_in[1] + add[1] : 2 * p_in[1] + 1;

    double val = 0.;
    for (long long d1 = 0; d1 < n_diags1; ++d1)
        for (long long d2 = 0; d2 < n_diags2; ++d2)
            val += mat(p_out[0] + i1_loc, p_out[1] + i2_loc, d1, d2) * x(i1 + d1 - s_in[0], i2 + d2 - s_in[1]);

    out(p_out[0] + i1_loc, p_out[1] + i2_loc) = val;
}
