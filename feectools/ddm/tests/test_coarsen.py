import numpy as np
import pytest

from feectools.ddm.cart import DomainDecomposition
from feectools.ddm.mpi import mpi as MPI


def _comm(parallel):
    return MPI.COMM_WORLD if parallel else None


def _as_list(arrs):
    return [np.asarray(a).tolist() for a in arrs]


@pytest.mark.parametrize("parallel", [False, pytest.param(True, marks=pytest.mark.mpi)])
@pytest.mark.parametrize("periods", [(True, True, True), (False, True, False)])
def test_coarsen(parallel, periods):
    comm = _comm(parallel)
    fine = DomainDecomposition([32, 16, 8], periods, comm=comm)
    coarse = fine.coarsen([2, 4, 1])

    assert coarse.ncells == (16, 4, 8)
    assert coarse.periods == fine.periods
    assert tuple(coarse.nprocs) == tuple(fine.nprocs)
    assert coarse.comm_cart is fine.comm_cart
    assert tuple(coarse.coords) == tuple(fine.coords)

    # every process owns exactly the coarse cells covering its fine cells
    for axis, f in enumerate([2, 4, 1]):
        assert coarse.starts[axis] * f == fine.starts[axis]
        assert (coarse.ends[axis] + 1) * f == fine.ends[axis] + 1
        assert coarse.local_ncells[axis] * f == fine.local_ncells[axis]

    # the original object is untouched
    assert fine.ncells == (32, 16, 8)

    # refining back gives the original partition
    back = coarse.refine(
        fine.ncells,
        [np.asarray(s) * f for s, f in zip(coarse.global_element_starts, [2, 4, 1])],
        [(np.asarray(e) + 1) * f - 1 for e, f in zip(coarse.global_element_ends, [2, 4, 1])],
    )
    assert back.ncells == fine.ncells
    assert _as_list(back.global_element_starts) == _as_list(fine.global_element_starts)
    assert _as_list(back.global_element_ends) == _as_list(fine.global_element_ends)
    assert back.starts == fine.starts
    assert back.ends == fine.ends
    assert back.local_ncells == fine.local_ncells


@pytest.mark.parametrize("parallel", [False, pytest.param(True, marks=pytest.mark.mpi)])
def test_coarsen_repeated(parallel):
    comm = _comm(parallel)
    dd = DomainDecomposition([64, 32, 1], (True, True, True), comm=comm, mpi_dims_mask=[True, True, False])
    for _ in range(3):
        dd = dd.coarsen([2, 2, 1])
    assert dd.ncells == (8, 4, 1)
    assert dd.local_ncells[2] == 1


def test_coarsen_misaligned():
    dd = DomainDecomposition([6, 4, 4], (True, True, True))
    with pytest.raises(ValueError):
        dd.coarsen([4, 1, 1])
    with pytest.raises(AssertionError):
        dd.coarsen([0, 1, 1])
