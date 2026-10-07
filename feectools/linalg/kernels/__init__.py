"""Kernels of feectools.linalg.

The stencil kernels called in solver loops (``stencil_dot_<n>d``, ``stencil_transpose_<n>d``,
``stencil_inner_<n>d``, ``stencil_axpy_<n>d``) have one folder each, with the pyccel kernel
``<name>_kernels.py`` and its CUDA version ``<name>_cuda.cu`` side by side; the folder's ``__init__.py``
declares the ``cunumpy.kernels.Kernel``, which runs the version of the active backend (see
CUDA_STRATEGY.md). The other modules hold pyccel kernels without CUDA versions.
"""
