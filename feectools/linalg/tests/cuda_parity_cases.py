"""Inputs for the pyccel/CUDA parity tests: for every kernel with a CUDA version, the cases it is checked on.

Test-only: ``test_cuda_parity.py`` (on a GPU) and ``test_cuda_emulation.py`` (without one) run each case on both
kernel versions and compare every array. ``build(case)`` returns the kernel's positional arguments on the active
backend; a kernel that is ported to CUDA gets an entry here.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from feectools.linalg.tests.kernel_test_args import (
    MATRIX_CASES,
    VECTOR_CASES,
    axpy_arguments,
    dot_arguments,
    inner_arguments,
    transpose_arguments,
)


@dataclass(frozen=True)
class ParityCases:
    """The cases of one kernel, how to build its arguments, and how to compare."""

    cases: Sequence[Any]
    build: Callable[[Any], tuple]
    """The kernel's positional arguments for one case, on the active backend."""
    rtol: float = 1e-12
    atol: float = 0.0
    n_threads: Callable[[tuple], int] | None = None
    """Launch size, if it is not the one of the declared kernel."""


def numbered(cases):
    """The cases with their index appended (the builders seed their random data with it)."""
    return [(*case, index) for index, case in enumerate(cases)]


PARITY_CASES = {}
for ndim in (1, 2, 3):
    # square and rectangular matrices
    PARITY_CASES[f"stencil_dot_{ndim}d"] = ParityCases(
        numbered(MATRIX_CASES[ndim]), lambda case: dot_arguments(*case), rtol=1e-13, atol=1e-14
    )
    PARITY_CASES[f"stencil_transpose_{ndim}d"] = ParityCases(
        numbered(MATRIX_CASES[ndim]), lambda case: transpose_arguments(*case), rtol=0.0
    )
    # vector spaces with different pads; on the GPU the summation order of inner differs
    PARITY_CASES[f"stencil_inner_{ndim}d"] = ParityCases(
        numbered(VECTOR_CASES[ndim]), lambda case: inner_arguments(*case), rtol=1e-12
    )
    PARITY_CASES[f"stencil_axpy_{ndim}d"] = ParityCases(
        numbered(VECTOR_CASES[ndim]), lambda case: axpy_arguments(*case), rtol=1e-14, atol=1e-15
    )
