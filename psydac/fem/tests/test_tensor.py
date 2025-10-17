import pytest
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from sympde.topology.domain import Square
from sympde.topology.space import ScalarFunctionSpace
from sympde.topology.analytical_mapping import TargetMapping

from psydac.mapping.discrete_gallery import discrete_mapping
from psydac.api.discretization import discretize

#==============================================================================
@pytest.mark.parallel
@pytest.mark.parametrize('root', ['first', 'last'])
@pytest.mark.parametrize('kind', ['spline', 'analytical'])
def test_plot_2d_decomposition(kind, root):

    # MPI communicator
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.size
    mpi_rank = mpi_comm.rank

    # MPI rank which should make the plot
    if root == 'first':
        mpi_root = 0
    elif root == 'last':
        mpi_root = mpi_size - 1
    else:
        raise ValueError(f'root argument has wrong value {root}')

    # Parameters of tensor-product 2D spline space
    ncells = (6, 9)
    degree = (2, 2)

    if kind == 'spline':
        # 2D spline mapping and tensor FEM space (distributed)
        F, Vh = discrete_mapping('target', ncells=ncells, degree=degree,
                            comm=mpi_comm, return_space=True)
    elif kind == 'analytical':
        Omega = Square('Omega', bounds1=(0, 1), bounds2=(0, 2 * np.pi))
        params = dict(c1=0, c2=0, k=0.3, D=0.2)
        M = TargetMapping('M', dim=2, **params)
        domain = M(Omega)
        V = ScalarFunctionSpace('V', domain)

        # 2D Geometry object
        domain_h = discretize(domain, ncells=ncells, periodic=(False, True),
                              comm=mpi_comm)

        # 2D spline tensor FEM space (distributed)
        Vh = discretize(V, domain_h, degree=degree)

        # 2D callable mapping (analytical)
        F = M.get_callable_mapping()
    else:
        raise ValueError(f'kind argument has wrong value {kind}')

    # Plot 2D decomposition
    # [1] Run without passing (fig, ax)
    Vh.plot_2d_decomposition(F, refine=5, mpi_root=mpi_root)

    # [2] Run with given (fig, ax), compatible
    fig2, ax2 = plt.subplots(1, 1) if mpi_rank == mpi_root else (None, None)
    Vh.plot_2d_decomposition(F, refine=5, fig=fig2, ax=ax2, mpi_root=mpi_root)

    # [3] Run with given (fig, ax), incompatible
    fig3, ax3 = plt.subplots(1, 1) if mpi_rank == mpi_root else (None, None)
    with pytest.raises(AssertionError) as excinfo:
        Vh.plot_2d_decomposition(F, refine=5, fig=fig2, ax=ax3, mpi_root=mpi_root)
    assert "Argument `ax` must be in `fig.axes`" in str(excinfo.value)
    if mpi_rank == mpi_root:
        plt.close(fig3)

#==============================================================================
if __name__ == '__main__':

    test_plot_2d_decomposition('spline', 'first')
    test_plot_2d_decomposition('analytical', 'last')
    plt.show()
