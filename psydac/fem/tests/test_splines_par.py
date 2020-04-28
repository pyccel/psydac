# -*- coding: UTF-8 -*-

import pytest

from psydac.fem.basic   import FemField
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.fem.vector  import VectorFemSpace

from numpy  import linspace
from mpi4py import MPI

@pytest.mark.parametrize( 'reverse_axis', [None, 0, 1] )
def test_2d_1(reverse_axis=None):

    p_1 = 2
    p_2 = 2
    grid_1 = linspace(0., 1., 10)
    grid_2 = linspace(0., 1., 15)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    V1 = SplineSpace(p_1, grid=grid_1)
    V2 = SplineSpace(p_2, grid=grid_2)
    V = TensorFemSpace(V1, V2, comm=comm, reverse_axis=reverse_axis)

    if rank == 0:
        print(V.vector_space.cart.nprocs)
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank ', rank)
            print('TensorFemSpace ', V)
            print('VectorSpace ')
            print('> npts ::', V.vector_space.npts)
            print('> starts ::', V.vector_space.starts)
            print('> ends ::', V.vector_space.ends)
            print('', flush=True)
        comm.Barrier()

    F = FemField(V)

###############################################
if __name__ == '__main__':
    test_2d_2()
