// CUDA version of stencil_dot_1d (stencil_dot_1d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

/**
 * Stencil matrix-vector product out = mat @ x on the owned rows, as the pyccel kernel stencil_dot_1d.
 *
 * One thread per entry of `out` (n_threads = out.size, see __init__.py). A thread outside the owned rows
 * (local row index i1_loc outside [0, e_out - s_out]) returns without writing, so the padding of `out` is
 * left as it is, as in pyccel. Interior rows use 2 * p_in + 1 diagonals, the last owned row (i1 == e_out)
 * uses 2 * p_in + add, which is how a rectangular matrix is handled.
 *
 * @param mat   matrix data, shape (rows of `out`, 2 * p_in + 1)
 * @param x     data of the domain vector, ghost regions included
 * @param out   data of the codomain vector; the owned rows are written
 * @param s_in  global start of the domain of this process
 * @param p_in  padding of the domain
 * @param add   1 if the last row uses all 2 * p_in + 1 diagonals, else 0
 * @param s_out global start of the codomain of this process
 * @param e_out global end (inclusive) of the codomain of this process
 * @param p_out padding of the codomain: the owned rows start at index p_out of `mat` and `out`
 */
extern "C" __global__ void stencil_dot_1d(Array2D<double> mat, Array1D<double> x, Array1D<double> out,
                                          long long s_in, long long p_in, long long add, long long s_out,
                                          long long e_out, long long p_out)
{
    CUNUMPY_THREAD_1D(thread, out.size());

    const long long i1_loc = thread - p_out;  // local row index
    if (i1_loc < 0 || i1_loc > e_out - s_out) return;
    const long long i1 = s_out + i1_loc;  // global row index

    const long long n_diags1 = (i1 == e_out) ? 2 * p_in + add : 2 * p_in + 1;

    double val = 0.;
    for (long long d1 = 0; d1 < n_diags1; ++d1)
        val += mat(p_out + i1_loc, d1) * x(i1 + d1 - s_in);

    out(p_out + i1_loc) = val;
}
