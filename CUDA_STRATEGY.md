# CUDA strategy for feectools

Plan for running feectools (the FEEC data and stencil operations used by struphy) on NVIDIA GPUs, next to the
existing pyccel (CPU) kernels. It follows the same rules as struphy's
[CUDA strategy](https://github.com/struphy-hub/struphy/blob/cuda-pr-17-feectools-cuda/CUDA_STRATEGY.md)
(section "feectools" there), and progress is tracked in struphy-hub/struphy#650.

## PR stack

The work is a stack of PRs, set up like struphy's: each PR's base branch is the branch of the PR below it, each
branch contains the one below it, and upstream changes (from `devel-tiny`) come in through merge commits, bottom
up. Each PR's diff on GitHub is therefore only its own step.

| PR | Branch | Base | What | Status |
| --- | --- | --- | --- | --- |
| [#90](https://github.com/struphy-hub/feectools/pull/90) | `cuda-development` | `devel-tiny` | **CUDA 1**: feectools runs on the CuPy backend (the content of [#85](https://github.com/struphy-hub/feectools/pull/85), merged into `cuda-development`) | open |
| [#86](https://github.com/struphy-hub/feectools/pull/86) | `cuda-2-mpi-sync` | `cuda-development` | **CUDA 2**: MPI with device buffers | open |
| [#87](https://github.com/struphy-hub/feectools/pull/87) | `cuda-3-device-binding` | `cuda-2-mpi-sync` | **CUDA 3**: one GPU per MPI rank | open |
| [#88](https://github.com/struphy-hub/feectools/pull/88) | `cuda-4-device-kernels` | `cuda-3-device-binding` | **CUDA 4**: stencil operations on the device, one folder per kernel | open |

Merge bottom up: #90 into `devel-tiny`, then retarget #86 to `devel-tiny` and merge it, and so on. Struphy's CUDA
PRs point their `feectools` submodule at the top of the stack (`cuda-4-device-kernels`) until the stack is merged
and released.

## Goal

feectools runs on either backend with the same code: on the CuPy backend the stencil data are CuPy arrays, MPI
passes device buffers, every rank has its own GPU, and the operations in a solver loop (`dot`, `transpose`,
`inner`, `axpy`) run on the device without host copies.

A developer who adds a GPU version of a feectools kernel only has to write `<name>_cuda.cu` (and
`<name>_test_args.py`) in the folder of its pyccel kernel; loading, dispatch and the parity tests come from
[cunumpy](https://github.com/struphy-hub/cunumpy), as in struphy.

## Principles

- **The backend decides.** The cunumpy backend (`CUNUMPY_BACKEND=cupy` or `cunumpy.set_backend("cupy")`) selects
  CuPy arrays and the CUDA kernels; with NumPy everything runs as before.
- **Data lives where the backend says.** Stencil data are `xp` arrays. Host-only metadata (MPI and index
  bookkeeping, Kronecker solver sizes) stays on NumPy.
- **One folder per kernel** (from CUDA 4 on). A kernel that is called with stencil data in a solver loop lives in
  `feectools/<package>/kernels/<name>/` with `<name>_kernels.py` (pyccel), `<name>_cuda.cu`, `<name>_test_args.py`
  and an `__init__.py` that declares `<name> = Kernel.from_folder(__name__)`. Code imports and calls the kernel.
- **1:1 correspondence.** The CUDA kernel has the same name and the same arguments, in the same order, as its
  pyccel kernel.
- **No silent CPU fallback** for folder kernels: on the CuPy backend a folder kernel without CUDA version raises.
  The other pyccel kernels (B-splines, field evaluation, DOF kernels, used at setup) are wrapped in
  `cunumpy.kernels.PyccelKernel` and copy their arrays to the host and back; see [Open questions](#open-questions).
- **Same cunumpy as struphy.** `cunumpy >= 0.5.0, < 0.6`; names are imported from the submodules
  (`cunumpy.kernels`, `cunumpy.cuda`, `cunumpy.mpi`, `cunumpy.kernel_testing`), not from the deprecated top level.
- **Small steps.** Every PR keeps the NumPy path working and tested.

## CUDA 1 implementation notes (#90, from #85)

feectools works when cunumpy's backend is CuPy, without device kernels:

- Pyccel kernels (stencil, B-splines, field evaluation, DOF kernels) are wrapped in
  `cunumpy.kernels.PyccelKernel`, so they accept CuPy arrays (copied to the host and back).
- Host-only metadata stays on NumPy: MPI and index bookkeeping in `ddm` and `fem.partitioning`, Kronecker solver
  sizes, index arithmetic with Python ints.
- Host-only libraries (LAPACK/SuperLU, SciPy FFT, SciPy sparse) get host copies per array, not by global backend.
- The 1D collocation matrices of the global projectors are built vectorized (element-wise indexing was one device
  round trip per entry: 334 s of a 348 s Derham setup on the GPU).
- Bug fix on both backends: `StencilMatrix._update_ghost_regions_serial` uses a ghost region `pads * shifts` wide.
- `feectools.ddm.mpi` still disables MPI when `CUNUMPY_BACKEND=cupy` (lifted in CUDA 2).
- Depends on `cunumpy >= 0.5.0, < 0.6`; `PyccelKernel` is imported from `cunumpy.kernels`.

## Testing

- Every PR runs the serial tests (`pytest feectools -m "not mpi and not petsc"`) and the MPI tests
  (`mpirun -n 2 pytest feectools -m "mpi and not petsc" --with-mpi`) on the NumPy backend.
- GPU tests are skipped without CuPy and a GPU (`cunumpy.kernel_testing.requires_cupy`). Until a GPU runner exists,
  they are run by hand on an H100 before a PR that touches CUDA code is merged, and the PR description says so.

## Open questions

- **Setup kernels on the GPU.** B-spline, field evaluation and DOF kernels still run on the host through
  `PyccelKernel` (host copies). They run at setup, not in the time loop; they move into kernel folders with CUDA
  versions when a profile shows they matter.
- **GPU CI.** No GPU runner yet; GPU tests are run by hand.
