# coding: utf-8

import pytest
import numpy as np

from spl.linalg.stencil import VectorSpace as StencilVectorSpace
from spl.linalg.stencil import Vector      as StencilVector
from spl.linalg.stencil import Matrix      as StencilMatrix
from spl.ddm.cart       import Cart

#===============================================================================
@pytest.mark.parametrize( 'n1', [4,10,35] )
@pytest.mark.parallel

def test_stencil_1d_parallel_matrix_vector_dot( n1 ):

    from mpi4py import MPI

    p1 = 1

    comm = MPI.COMM_WORLD
    cart = Cart( npts    = [n1,],
                 pads    = [p1,],
                 periods = [True ,],
                 reorder = [False,],
                 comm    = comm )

    V = StencilVectorSpace( cart )
    x = StencilVector( V )
    A = StencilMatrix( V, V )

    x[:] = 1.0
    A[:,-1] = -1.0
    A[:, 0] =  5.0
    A[:,+1] = -2.0

    b = A.dot( x )

    assert isinstance( b, StencilVector )
    assert b.space is x.space
    assert all( b[:] == 2.0 )
