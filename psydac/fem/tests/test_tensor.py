import pytest
from mpi4py import MPI

from psydac.mapping.discrete_gallery import discrete_mapping

#==============================================================================
@pytest.mark.parallel
def test_plot_2d_decomposition():

    # MPI communicator
    mpi_comm = MPI.COMM_WORLD

    # 2D spline mapping and tensor FEM space (distributed)
    F, V = discrete_mapping('target', ncells=(5, 10), degree=(2, 2),
                            comm=mpi_comm, return_space=True)

    # Plot 2D decomposition
    V.plot_2d_decomposition(F, refine=10)