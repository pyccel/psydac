# -*- coding: UTF-8 -*-
import pytest
import numpy as np

from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )

def test_stencil_matrix_2d_serial_data( n1, n2, p1, p2 ):

    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values[k1,k2] = 10*k1 + k2

    V = StencilVectorSpace( [0,0], [n1,n2], [p1,p2] )
    M = StencilMatrix( V, V )

    # Fill stencil matrix values
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M[:,:,k1,k2] = nonzero_values[k1,k2]

    Ma = M.toarray()

    print (">>> M shape: ", M._data.shape)
    print (">>> M = ")
    print (Ma)

    # Construct exact matrix by hand
    A = np.zeros( M.shape )
    for i1 in range(1+n1):
        for i2 in range(1+n2):
            for k1 in range(-p1,p1+1):
                for k2 in range(-p2,p2+1):
                    j1 = (i1+k1) % (1+n1)
                    j2 = (i2+k2) % (1+n2)
                    i  = i1*(n2+1) + i2
                    j  = j1*(n2+1) + j2
                    A[i,j] = nonzero_values[k1,k2]

    assert M._data.shape == (1+n1, 1+n2, 1+2*p1, 1+2*p2)
    assert M.shape == ((1+n1)*(1+n2),(1+n1)*(1+n2))
    assert M.shape == Ma.shape
    assert np.all( Ma == A )

#===============================================================================
def test_stencil_matrix_2d_serial_dot():

    nx = ny = 2
    px = py = 1

    V = StencilVectorSpace([0, 0], [nx, ny], [px, py])
    x = StencilVector( V )
    M = StencilMatrix( V, V )

    for ix in range(nx+1):
        for iy in range(ny+1):
            M[ix, iy, 0, 0] =  4.
            M[ix, iy, 1, 0] =  1.
            M[ix, iy,-1, 0] = -1.
            M[ix, iy, 0, 1] =  2.
            M[ix, iy, 0,-1] = -2.

    x[:,:] = 1.

    y = M.dot(x)

    print (">>> M shape = ", M._data.shape)
    print (">>> M = ")
    print ( M.toarray() )

    print (">>> x shape = ", x._data.shape)
    print (">>> x = ", x.toarray() )

    print (">>> y (M dot ones) = ", y.toarray() )

    assert np.all( y.toarray() == 4.0 )

#===============================================================================
if __name__ == "__main__":
    import sys
    import pytest
    pytest.main( sys.argv )
