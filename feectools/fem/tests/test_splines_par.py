# -*- coding: UTF-8 -*-

import pytest

from feectools.fem.basic   import FemField
from feectools.fem.splines import SplineSpace
from feectools.fem.tensor  import TensorFemSpace
from feectools.fem.vector  import VectorFemSpace
from feectools.ddm.cart    import DomainDecomposition
from feectools.ddm.mpi import mpi as MPI

from numpy  import linspace

def test_2d_1():

    p_1 = 2
    p_2 = 2
    grid_1 = linspace(0., 1., 10)
    grid_2 = linspace(0., 1., 15)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    V1 = SplineSpace(p_1, grid=grid_1)
    V2 = SplineSpace(p_2, grid=grid_2)

    domain_decomposition = DomainDecomposition([V1.ncells, V2.ncells], [False, False], comm=comm)
    V = TensorFemSpace(domain_decomposition, V1, V2)

    if rank == 0:
        print(V.coeff_space.cart.nprocs)
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank ', rank)
            print('TensorFemSpace ', V)
            print('VectorSpace ')
            print('> npts ::', V.coeff_space.npts)
            print('> starts ::', V.coeff_space.starts)
            print('> ends ::', V.coeff_space.ends)
            print('', flush=True)
        comm.Barrier()

    F = FemField(V)

###############################################
if __name__ == '__main__':
    test_2d_2()
