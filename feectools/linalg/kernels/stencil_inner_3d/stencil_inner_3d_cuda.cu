// CUDA version of stencil_inner_3d (stencil_inner_3d_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/index.cuh"

// Threads per block; the block reduction below needs a power of two (block_size in __init__.py).
#define STENCIL_INNER_BLOCK 256

/**
 * Inner product of two stencil vectors over their owned (non-ghost) entries, added to res[0], as the
 * pyccel kernel stencil_inner_3d (real data: the conjugate of v1 is v1).
 *
 * One thread per entry of `v1` (n_threads = v1.size, see __init__.py); a thread on a ghost entry (index
 * i_k < nghost_k or >= shape_k - nghost_k along a direction k) or beyond the array contributes 0. Each
 * block sums the products of its threads in shared memory (a tree reduction, so blockDim.x must be
 * STENCIL_INNER_BLOCK) and adds the block sum to res[0] with one atomicAdd. The caller sets res[0] to zero
 * first, as for the pyccel kernel. The summation order differs from the serial pyccel loop.
 *
 * @param v1, v2  data of the two vectors, ghost regions included
 * @param nghost0 ... number of ghost entries at each end, per direction
 * @param res     one entry; the inner product is added to res[0]
 */
extern "C" __global__ void stencil_inner_3d(Array3D<double> v1, Array3D<double> v2, long long nghost0, long long nghost1, long long nghost2, Array1D<double> res)
{
    __shared__ double part[STENCIL_INNER_BLOCK];

    const long long thread = CUNUMPY_GLOBAL_INDEX_X();
    double product = 0.;
    if (thread < v1.size()) {
        long long rest = thread;
        const long long i2 = rest % v1.shape[2];
        rest /= v1.shape[2];
        const long long i1 = rest % v1.shape[1];
        rest /= v1.shape[1];
        const long long i0 = rest;
        if (i0 >= nghost0 && i0 < v1.shape[0] - nghost0 && i1 >= nghost1 && i1 < v1.shape[1] - nghost1 && i2 >= nghost2 && i2 < v1.shape[2] - nghost2)
            product = v1(i0, i1, i2) * v2(i0, i1, i2);
    }
    part[threadIdx.x] = product;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) part[threadIdx.x] += part[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&res(0), part[0]);
}
