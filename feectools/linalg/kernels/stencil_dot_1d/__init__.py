"""The kernel of this folder: pyccel in ``stencil_dot_1d_kernels.py``, CUDA in ``stencil_dot_1d_cuda.cu``."""

from cunumpy.kernels import Kernel

# one thread per entry of `out` (the third argument)
stencil_dot_1d = Kernel.from_folder(__name__, n_threads_from=lambda args: args[2].size)
