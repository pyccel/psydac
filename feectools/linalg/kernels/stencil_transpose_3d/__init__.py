"""The kernel of this folder: pyccel in ``stencil_transpose_3d_kernels.py``, CUDA in ``stencil_transpose_3d_cuda.cu``."""

from cunumpy.kernels import Kernel

# one thread per entry of `matT` (the second argument)
stencil_transpose_3d = Kernel.from_folder(__name__, n_threads_from=lambda args: args[1].size)
