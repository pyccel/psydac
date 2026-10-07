// CUDA version of stencil_axpy_1d (stencil_axpy_1d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

/**
 * y = alpha * x + y over the whole arrays (ghost regions included), as the pyccel kernel stencil_axpy_1d.
 *
 * One thread per entry (n_threads = x.size, see __init__.py), the last axis varying fastest.
 *
 * @param alpha scaling coefficient
 * @param x     data of the vector that is added
 * @param y     data of the vector that is updated in place
 */
extern "C" __global__ void stencil_axpy_1d(double alpha, Array1D<double> x, Array1D<double> y)
{
    CUNUMPY_THREAD_1D(thread, x.size());

    const long long i0 = thread;
    y(i0) += alpha * x(i0);
}
