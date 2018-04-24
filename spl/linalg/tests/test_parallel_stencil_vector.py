# coding: utf-8

from mpi4py import MPI

import pytest
import numpy as np

from spl.linalg.stencil import StencilVectorSpace, StencilVector
from spl.ddm.cart       import Cart

#===============================================================================
@pytest.mark.parametrize( 'n1', [2,10,23] )
@pytest.mark.parametrize( 'n2', [2,12,25] )
@pytest.mark.parametrize( 'p1', [1,3,4] )
@pytest.mark.parametrize( 'p2', [1,3,4] )
@pytest.mark.parallel

def test_vector_parallel_dot( n1, n2, p1, p2 ):

    xval  = 2.0
    yval  = 1.0

    comm = MPI.COMM_WORLD
    cart = Cart( npts    = [n1,n2],
                 pads    = [p1,p2],
                 periods = [True ,False],
                 reorder = False,
                 comm    = comm )

    V = StencilVectorSpace( cart )
    x = StencilVector( V )
    y = StencilVector( V )


    x[:,:] = xval
    y[:,:] = yval
    res_ex = n1*n2*xval*yval

    res1 = x.dot( y )
    res2 = y.dot( x )

    assert res1 == res_ex
    assert res2 == res_ex
