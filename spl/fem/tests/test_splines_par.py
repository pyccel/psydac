# -*- coding: UTF-8 -*-

import pytest

from spl.fem.splines import SplineSpace
from spl.fem.splines import Spline
from spl.fem.tensor  import TensorFemSpace
from spl.fem.vector  import VectorFemSpace

from numpy  import linspace
from mpi4py import MPI

@pytest.mark.parallel

def test_2d_1():
    print ('>>> test_2d_1')

    knots_1 = [0., 0., 0., 1., 1., 1.]
    knots_2 = [0., 0., 0., 0.5, 1., 1., 1.]
    p_1 = 2
    p_2 = 2

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    V1 = SplineSpace(p_1, knots=knots_1)
    V2 = SplineSpace(p_2, knots=knots_2)
    V = TensorFemSpace(V1, V2, comm=comm)

    for i in range(comm.Get_size()):
        if rank == i:
            print('rank ', rank)
            print('TensorFemSpace ', V)
            print('VectorSpace ')
            print('> npts ::', V.vector_space.npts)
            print('> starts ::', V.vector_space.starts)
            print('> ends ::', V.vector_space.ends)

    F = Spline(V)

def test_2d_2():
    print ('>>> test_2d_2')

    p_1 = 2
    p_2 = 2
    grid_1 = linspace(0., 1., 10)
    grid_2 = linspace(0., 1., 15)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    V1 = SplineSpace(p_1, grid=grid_1)
    V2 = SplineSpace(p_2, grid=grid_2)
    V = TensorFemSpace(V1, V2, comm=comm)

    for i in range(comm.Get_size()):
        if rank == i:
            print('rank ', rank)
            print('TensorFemSpace ', V)
            print('VectorSpace ')
            print('> npts ::', V.vector_space.npts)
            print('> starts ::', V.vector_space.starts)
            print('> ends ::', V.vector_space.ends)

    F = Spline(V)


###############################################
if __name__ == '__main__':
    test_2d_1()
    test_2d_2()
