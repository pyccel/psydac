#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
import pytest
from mpi4py import MPI

from psydac.ddm.cart    import DomainDecomposition
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace

#==============================================================================
# A single process owns all the coefficients, which are at least `degree`
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('periodic', [True, False])
def test_partition_coefficients_too_few(periodic: bool) -> None:

    # Some processes own a single cell, hence less than `degree` coefficients
    comm = MPI.COMM_WORLD
    degree = 3
    ncells = max(comm.size, degree)
    W = SplineSpace(degree=degree, grid=np.linspace(0, 1, ncells + 1), periodic=periodic)
    ddm = DomainDecomposition([ncells], [periodic], comm=comm)

    with pytest.raises(ValueError, match=f'at least {degree} are needed'):
        TensorFemSpace(ddm, W)
