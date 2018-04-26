# coding: utf-8

import pytest
import numpy as np

from spl.linalg.stencil import StencilVectorSpace, StencilVector

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [1,7] )
@pytest.mark.parametrize( 'n2', [1,5] )
@pytest.mark.parametrize( 'p1', [1,2] )
@pytest.mark.parametrize( 'p2', [1,2] )

def test_stencil_vector_2d_serial_init( n1, n2, p1, p2 ):


    V = StencilVectorSpace( [n1,n2], [p1,p2] )
    x = StencilVector( V )

    assert x.space is V
    assert x._data.shape == (n1+2*p1, n2+2*p2)

#===============================================================================
@pytest.mark.parametrize( 'n1', [1,7] )
@pytest.mark.parametrize( 'n2', [1,5] )
@pytest.mark.parametrize( 'p1', [1,2] )
@pytest.mark.parametrize( 'p2', [1,2] )

def test_stencil_vector_2d_serial_copy( n1, n2, p1, p2 ):


    V = StencilVectorSpace( [n1,n2], [p1,p2] )
    x = StencilVector( V )

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1,i2] = 10*i1 + i2

    z = x.copy()

    assert isinstance( z, StencilVector )
    assert z.space is V
    assert z._data is not x._data
    assert np.all( z[:,:] == x[:,:] )

#===============================================================================
@pytest.mark.parametrize( 'n1', [1,7] )
@pytest.mark.parametrize( 'n2', [1,5] )
@pytest.mark.parametrize( 'p1', [1,2] )
@pytest.mark.parametrize( 'p2', [1,2] )

def test_stencil_matrix_2d_serial_toarray( n1, n2, p1, p2 ):

    V = StencilVectorSpace( [n1,n2], [p1,p2] )
    x = StencilVector( V )

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1,i2] = 10*i1 + i2

    xa = x.toarray()

    z = np.zeros( (n1*n2) )
    for i1 in range(n1):
        for i2 in range(n2):
            z[i1*n2+i2] = 10*i1 + i2

    assert xa.shape == (n1*n2,)
    assert np.all( xa == z )

#===============================================================================
@pytest.mark.parametrize( 'n1', [1,7] )
@pytest.mark.parametrize( 'n2', [1,5] )
@pytest.mark.parametrize( 'p1', [1,2] )
@pytest.mark.parametrize( 'p2', [1,2] )

def test_stencil_vector_2d_serial_math( n1, n2, p1, p2 ):


    V = StencilVectorSpace( [n1,n2], [p1,p2] )
    x = StencilVector( V )
    y = StencilVector( V )

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1,i2] = 10*i1 + i2

    y[:,:] = 42.0

    r1 = x + y
    r2 = x - y
    r3 = 2 * x
    r4 = x * 2

    xa = x.toarray()
    ya = y.toarray()

    r1_exact = xa + ya
    r2_exact = xa - ya
    r3_exact = 2  * xa
    r4_exact = xa * 2

    for (r,rex) in zip( [r1,r2,r3,r4], [r1_exact,r2_exact,r3_exact,r4_exact] ):

        assert isinstance( r, StencilVector )
        assert r.space is V
        assert np.all( r.toarray() == rex )

#===============================================================================
@pytest.mark.parametrize( 'n1', [1,7] )
@pytest.mark.parametrize( 'n2', [1,5] )
@pytest.mark.parametrize( 'p1', [1,2] )
@pytest.mark.parametrize( 'p2', [1,2] )

def test_stencil_vector_2d_serial_dot( n1, n2, p1, p2 ):


    V = StencilVectorSpace( [n1,n2], [p1,p2] )
    x = StencilVector( V )
    y = StencilVector( V )

    for i1 in range(n1):
        for i2 in range(n2):
            x[i1,i2] = 10*i1 + i2
            y[i1,i2] = 10*i2 - i1

    z1 = x.dot( y )
    z2 = y.dot( x )

    z_exact = np.dot( x.toarray(), y.toarray() )

    assert z1 == z_exact
    assert z2 == z_exact

#===============================================================================
# PARALLEL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [2,10,23] )
@pytest.mark.parametrize( 'n2', [2,12,25] )
@pytest.mark.parametrize( 'p1', [1,3,4] )
@pytest.mark.parametrize( 'p2', [1,3,4] )
@pytest.mark.parallel

def test_stencil_vector_2d_parallel_dot( n1, n2, p1, p2 ):

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
