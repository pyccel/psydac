import pytest
import numpy as np
from mpi4py import MPI

from sympde.topology.domain import Square
from sympde.topology.space import ScalarFunctionSpace
from sympde.topology.analytical_mapping import TargetMapping

from psydac.mapping.discrete_gallery import discrete_mapping
from psydac.api.discretization import discretize

#==============================================================================
@pytest.mark.parallel
@pytest.mark.parametrize('kind', ['spline', 'analytical'])
def test_plot_2d_decomposition(kind):

    # MPI communicator
    mpi_comm = MPI.COMM_WORLD

    # Parameters of tensor-product 2D spline space
    ncells = (6, 9)
    degree = (2, 2)

    if kind == 'spline':
        # 2D spline mapping and tensor FEM space (distributed)
        F, Vh = discrete_mapping('target', ncells=ncells, degree=degree,
                            comm=mpi_comm, return_space=True)
    else:
        Omega = Square('Omega', bounds1=(0, 1), bounds2=(0, 2 * np.pi))
        params = dict(c1=0, c2=0, k=0.3, D=0.2)
        map = TargetMapping('M', dim=2, **params)
        domain = map(Omega)
        V = ScalarFunctionSpace('V', domain)

        # 2D Geometry object
        domain_h = discretize(domain, ncells=ncells, periodic=(False, True),
                              comm=mpi_comm)

        # 2D spline tensor FEM space (distributed)
        Vh = discretize(V, domain_h, degree=degree)

        # 2D callable mapping (analytical)
        F = map.get_callable_mapping()

    # Plot 2D decomposition
    Vh.plot_2d_decomposition(F, refine=5)

#==============================================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_plot_2d_decomposition('spline')
    plt.show()

    test_plot_2d_decomposition('analytical')
    plt.show()
