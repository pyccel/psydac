# coding: utf-8

import pytest
import numpy as np

from spl.linalg.stencil import StencilVectorSpace, StencilVector

#===============================================================================
def test_1():

    print( "", "-"*40, "Test 1", "-"*40, "", sep="\n" )

    n1 = n2 = 4
    p1 = p2 = 1
    V = StencilVectorSpace( [n1,n2], [p1,p2] )

    x = StencilVector( V )
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1,p1+1):
                for k2 in range(-p2,p2+1):
                    j1 = k1+i1
                    j2 = k2+i2
                    x[j1,j2] = j1+j2
    z = x.copy()

    print('>>> x shape: ', x._data.shape)
    print(">>> x = ")
    print(x)
    print(">>> x.toarray() = ",  x.toarray())
    print(">>> z = x.copy() = ")
    print(z)

#===============================================================================
def test_2():

    print( "", "-"*40, "Test 2", "-"*40, "", sep="\n" )

    n1 = n2 = 3
    p1 = p2 = 1
    V = StencilVectorSpace( [n1,n2], [p1,p2] )

    x = StencilVector( V )
    y = StencilVector( V )

    x[:,:] = 42.
    y[:,:] = 10.

    a = x + y
    b = x - y
    c = 2 * x

    print('>>> x shape: ', x._data.shape)
    print(">>> x = ")
    print(x)
    print('>>> y shape: ', y._data.shape)
    print(">>> y = ")
    print(y)
    print(">>> x+y = ")
    print(a)
    print(">>> x-y = ")
    print(b)
    print(">>> 2*x = ")
    print(c)

#===============================================================================
def test_3():

    print( "", "-"*40, "Test 3", "-"*40, "", sep="\n" )

    n1 = n2 = 3
    p1 = p2 = 1
    V = StencilVectorSpace( [n1,n2], [p1,p2] )

    x = StencilVector( V )
    y = StencilVector( V )

    x[:, :] = 2.
    y[:, :] = 5.

    z = x.dot(y)

    print('>>> x shape: ', x._data.shape)
    print(">>> x.toarray() = ", x.toarray())
    print('>>> y shape: ', y._data.shape)
    print(">>> y.toarray() = ", y.toarray())
    print(">>> dot(x,y) = ")
    print(z)

#===============================================================================
@pytest.mark.parametrize( 'n1', [2,10,23] )
@pytest.mark.parametrize( 'n2', [2,12,25] )
@pytest.mark.parametrize( 'p1', [1,3,4] )
@pytest.mark.parametrize( 'p2', [1,3,4] )
@pytest.mark.parallel

def test_vector_parallel_dot( n1, n2, p1, p2 ):

    from mpi4py       import MPI
    from spl.ddm.cart import Cart

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

#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
