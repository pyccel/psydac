// CUDA version of stencil_axpy_3d (stencil_axpy_3d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

/**
 * y = alpha * x + y over the whole arrays (ghost regions included), as the pyccel kernel stencil_axpy_3d.
 *
 * One thread per entry (n_threads = x.size, see __init__.py), the last axis varying fastest.
 *
 * @param alpha scaling coefficient
 * @param x     data of the vector that is added
 * @param y     data of the vector that is updated in place
 */
extern "C" __global__ void stencil_axpy_3d(double alpha, Array3D<double> x, Array3D<double> y)
{
    CUNUMPY_THREAD_1D(thread, x.size());

    long long rest = thread;
    const long long i2 = rest % x.shape[2];
    rest /= x.shape[2];
    const long long i1 = rest % x.shape[1];
    rest /= x.shape[1];
    const long long i0 = rest;
    y(i0, i1, i2) += alpha * x(i0, i1, i2);
}
