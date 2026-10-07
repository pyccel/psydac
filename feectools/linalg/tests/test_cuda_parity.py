"""Parity of every CUDA kernel with its pyccel kernel, on the cases of ``cuda_parity_cases.PARITY_CASES``.

The kernels are the ones the folders declare (``<name> = Kernel.from_folder(...)`` in ``__init__.py``), with
their launch options, i.e. the objects the code calls.
"""
import importlib

import cunumpy as xp
import numpy as np
import pytest
from cunumpy.kernel_testing import assert_kernels_agree, requires_cupy
from cunumpy.kernels import Kernel, KernelCatalog

from feectools.linalg.tests.cuda_parity_cases import PARITY_CASES

# all kernel packages with one folder per kernel, for tests that go through every kernel
PACKAGES = ("feectools.linalg.kernels",)
CATALOGS = {package: KernelCatalog.from_package(package) for package in PACKAGES}
DECLARED = KernelCatalog(
    {
        name: getattr(importlib.import_module(f"{package}.{name}"), name)
        for package, catalog in CATALOGS.items()
        for name in catalog
    }
)
# every kernel with a CUDA version, by name
CUDA_KERNELS = dict(DECLARED.parity_cases())
# (kernel name, case index) of every parity case, as pytest parameters
PARITY_PARAMS = [
    pytest.param(name, index, id=f"{name}-{index}")
    for name in PARITY_CASES
    for index in range(len(PARITY_CASES[name].cases))
]


def check_case(name, index, compare):
    """Run case `index` of kernel `name` with `compare(kernel, make_args, **settings)` (cunumpy's signature)."""
    spec = PARITY_CASES[name]
    case = spec.cases[index]
    return compare(
        CUDA_KERNELS[name],
        lambda backend, seed: spec.build(case),
        n_threads=spec.n_threads,
        rtol=spec.rtol,
        atol=spec.atol,
    )


def test_folders_declare_their_kernels():
    """Each kernel folder's __init__.py declares its kernel under the folder name (the import used in the code)."""
    assert len(DECLARED) > 0
    for catalog in CATALOGS.values():
        for name in catalog:
            kernel = DECLARED[name]
            assert isinstance(kernel, Kernel) and kernel.name == name
            assert kernel.has_cuda == catalog[name].has_cuda


def test_signatures():
    """The pyccel and CUDA versions of every kernel take the same arguments in the same order."""
    DECLARED.check_signatures()


def test_cuda_kernels_have_parity_cases():
    """Every kernel with a CUDA version has parity cases (add them to cuda_parity_cases.PARITY_CASES), and only those."""
    assert set(PARITY_CASES) == set(CUDA_KERNELS)
    assert all(len(spec.cases) > 0 for spec in PARITY_CASES.values())


def test_stencil_kernels_have_cuda():
    """The stencil operations of solver loops (dot, transpose, inner, axpy in 1-3D) all have CUDA versions."""
    for operation in ("dot", "transpose", "inner", "axpy"):
        for ndim in (1, 2, 3):
            assert DECLARED[f"stencil_{operation}_{ndim}d"].has_cuda


@requires_cupy
@pytest.mark.parametrize("name, index", PARITY_PARAMS)
def test_parity(name, index):
    check_case(name, index, assert_kernels_agree)


@requires_cupy
def test_solver_loop_has_no_host_transfers():
    """dot and axpy on the CuPy backend run on the device: no host copies once the kernels are compiled."""
    from cunumpy.profiling import assert_no_transfers

    from feectools.linalg.tests.kernel_test_args import stencil_matrix, stencil_vector

    with xp.use_backend("cupy"):
        rng = np.random.default_rng(0)
        A, V, W = stencil_matrix((12, 10, 8), (12, 10, 8), (2, 2, 3), rng)
        x = stencil_vector(V, rng)
        y = A.dot(x)  # warm up: compiles the CUDA kernels
        V.axpy(0.5, y, x)
        with assert_no_transfers():
            for _ in range(5):
                A.dot(x, out=y)
                V.axpy(0.5, y, x)
