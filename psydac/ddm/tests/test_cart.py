#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
from mpi4py import MPI

from psydac.ddm.cart import DomainDecomposition

masks = pytest.mark.parametrize('mpi_dims_mask', [None, [False, True], [True, False]])

#==============================================================================
@masks
def test_DomainDecomposition_mpi_dims_mask(mpi_dims_mask) -> None:

    ddm = DomainDecomposition([3, 5], [False, False], mpi_dims_mask=mpi_dims_mask)

    expected = None if mpi_dims_mask is None else tuple(mpi_dims_mask)
    assert ddm.mpi_dims_mask == expected

#==============================================================================
@pytest.mark.mpi
@masks
def test_DomainDecomposition_refine(mpi_dims_mask) -> None:

    # Each direction can be decomposed among all processes
    comm = MPI.COMM_WORLD
    ncells = [2 * comm.size, 3 * comm.size]
    ddm = DomainDecomposition(ncells, [False, False], comm=comm, mpi_dims_mask=mpi_dims_mask)

    # Split each cell into 2 x 2 cells, keeping the subdomain boundaries
    global_starts = [2 * s for s in ddm.global_element_starts]
    global_ends   = [2 * e + 1 for e in ddm.global_element_ends]
    new_ddm = ddm.refine([2 * n for n in ncells], global_starts, global_ends)

    # The refined decomposition has the same process grid (see #622)
    assert new_ddm.mpi_dims_mask == ddm.mpi_dims_mask
    assert list(new_ddm.nprocs) == list(ddm.nprocs)
    assert list(new_ddm.comm_cart.Get_topo()[0]) == list(ddm.nprocs)
    assert new_ddm.starts == tuple(2 * s for s in ddm.starts)
    assert new_ddm.ends   == tuple(2 * e + 1 for e in ddm.ends)
