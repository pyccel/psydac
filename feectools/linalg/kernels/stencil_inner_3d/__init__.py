"""The kernel of this folder: pyccel in ``stencil_inner_3d_kernels.py``, CUDA in ``stencil_inner_3d_cuda.cu``."""

from cunumpy.kernels import Kernel

# one thread per entry of `v1` (the first argument), in blocks of 256 threads for the block reduction
stencil_inner_3d = Kernel.from_folder(__name__, block_size=256, n_threads_from=lambda args: args[0].size)
