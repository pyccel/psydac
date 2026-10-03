import numpy as np
import pytest

from feectools.api.essential_bc import apply_essential_bc_stencil
from feectools.ddm.cart import DomainDecomposition
from feectools.ddm.mpi import mpi as MPI
from feectools.fem.splines import SplineSpace
from feectools.fem.tensor import TensorFemSpace


@pytest.mark.parametrize("parallel", [False, pytest.param(True, marks=pytest.mark.mpi)])
def test_essential_bc_marks_ghosts_stale(parallel):
    """After setting boundary coefficients to zero, ghost regions (on all processes) must be refreshed."""
    comm = MPI.COMM_WORLD if parallel else None
    p, nc = 2, 8
    spaces = [SplineSpace(p, grid=np.linspace(0.0, 1.0, nc + 1), periodic=False) for _ in range(2)]
    dd = DomainDecomposition([nc, nc], [False, False], comm=comm)
    V = TensorFemSpace(dd, *spaces).coeff_space

    v = V.zeros()
    v._data[...] = 1.0
    v.update_ghost_regions()
    assert v.ghost_regions_in_sync

    apply_essential_bc_stencil(v, axis=0, ext=-1, order=0)
    assert not v.ghost_regions_in_sync

    v.update_ghost_regions()
    # every copy (owned or ghost) of the boundary coefficients i0 = 0 is zero now
    s0, pad = V.starts[0], V.pads[0]
    for i_loc in range(v._data.shape[0]):
        if s0 - pad + i_loc == 0:
            assert np.all(v._data[i_loc, pad:-pad] == 0.0)
