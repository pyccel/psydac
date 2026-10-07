// CUDA version of stencil_dot_3d (stencil_dot_3d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

/**
 * Stencil matrix-vector product out = mat @ x on the owned rows, as the pyccel kernel stencil_dot_3d.
 *
 * One thread per entry of `out` (n_threads = out.size, see __init__.py), the last axis varying fastest. A
 * thread outside the owned rows returns without writing, so the padding of `out` is left as it is, as in
 * pyccel. Along each direction k, interior rows use 2 * p_in[k] + 1 diagonals and the last owned row
 * (i_k == e_out[k]) uses 2 * p_in[k] + add[k]; the pyccel kernel spells out the eight combinations, this
 * kernel picks its own per thread. The diagonals are summed in the same order (d1, d2, d3 innermost).
 *
 * `mat` has six axes and cunumpy's array views stop at four, so it is a raw pointer to C-contiguous data
 * (checked by cunumpy at the call) with the shape (out.shape[0], out.shape[1], out.shape[2],
 * 2 * p_in[0] + 1, 2 * p_in[1] + 1, 2 * p_in[2] + 1): the rows of the matrix are those of `out`, and it
 * has exactly the diagonals the pyccel kernel reads. StencilMatrix checks the diagonals before calling
 * the precompiled kernels (see StencilMatrix.set_backend).
 *
 * @param mat   matrix data, C-contiguous, shape as above
 * @param x     data of the domain vector, ghost regions included
 * @param out   data of the codomain vector; the owned rows are written
 * @param s_in, p_in, add, s_out, e_out, p_out  per direction (length 3), as in the 1D kernel
 */
extern "C" __global__ void stencil_dot_3d(const double* mat, Array3D<double> x, Array3D<double> out,
                                          const long long* s_in, const long long* p_in, const long long* add,
                                          const long long* s_out, const long long* e_out,
                                          const long long* p_out)
{
    CUNUMPY_THREAD_1D(thread, out.size());

    const long long i1_loc = thread / (out.shape[1] * out.shape[2]) - p_out[0];  // local row indices
    const long long i2_loc = (thread / out.shape[2]) % out.shape[1] - p_out[1];
    const long long i3_loc = thread % out.shape[2] - p_out[2];
    if (i1_loc < 0 || i1_loc > e_out[0] - s_out[0]) return;
    if (i2_loc < 0 || i2_loc > e_out[1] - s_out[1]) return;
    if (i3_loc < 0 || i3_loc > e_out[2] - s_out[2]) return;
    const long long i1 = s_out[0] + i1_loc;  // global row indices
    const long long i2 = s_out[1] + i2_loc;
    const long long i3 = s_out[2] + i3_loc;

    const long long n_diags1 = (i1 == e_out[0]) ? 2 * p_in[0] + add[0] : 2 * p_in[0] + 1;
    const long long n_diags2 = (i2 == e_out[1]) ? 2 * p_in[1] + add[1] : 2 * p_in[1] + 1;
    const long long n_diags3 = (i3 == e_out[2]) ? 2 * p_in[2] + add[2] : 2 * p_in[2] + 1;

    // element strides of the C-contiguous matrix data
    const long long stride_d3 = 1;
    const long long stride_d2 = (2 * p_in[2] + 1) * stride_d3;
    const long long stride_d1 = (2 * p_in[1] + 1) * stride_d2;
    const long long stride_i3 = (2 * p_in[0] + 1) * stride_d1;
    const long long stride_i2 = out.shape[2] * stride_i3;
    const long long stride_i1 = out.shape[1] * stride_i2;
    const double* row = mat + (p_out[0] + i1_loc) * stride_i1 + (p_out[1] + i2_loc) * stride_i2 +
                        (p_out[2] + i3_loc) * stride_i3;

    double val = 0.;
    for (long long d1 = 0; d1 < n_diags1; ++d1)
        for (long long d2 = 0; d2 < n_diags2; ++d2)
            for (long long d3 = 0; d3 < n_diags3; ++d3)
                val += row[d1 * stride_d1 + d2 * stride_d2 + d3 * stride_d3] *
                       x(i1 + d1 - s_in[0], i2 + d2 - s_in[1], i3 + d3 - s_in[2]);

    out(p_out[0] + i1_loc, p_out[1] + i2_loc, p_out[2] + i3_loc) = val;
}
