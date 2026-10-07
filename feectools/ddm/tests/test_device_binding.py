"""Each process is bound to its own GPU when feectools.ddm.cart is imported."""
import cunumpy as xp
import pytest
from cunumpy.cuda import device_count
from cunumpy.kernel_testing import requires_cupy
from cunumpy.mpi import local_rank


@requires_cupy
def test_rank_is_bound_to_its_local_device():
    import cupy as cp

    import feectools.ddm.cart  # noqa: F401 -- binds the device on import

    if xp.get_backend() != "cupy":
        pytest.skip("device binding only happens on the CuPy backend")
    expected = local_rank() % device_count()
    assert cp.cuda.runtime.getDevice() == expected
