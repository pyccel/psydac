# -*- coding: UTF-8 -*-

import pytest
import numpy as np

from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )

def test_stencil_matrix_2d_serial_shape( n1, n2, p1, p2 ):

    V = StencilVectorSpace( [n1,n2], [p1,p2] )
    M = StencilMatrix( V, V )

    assert M._data.shape == (n1, n2, 1+2*p1, 1+2*p2)
    assert M.shape == (n1*n2, n1*n2)

#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )

def test_stencil_matrix_2d_serial_toarray( n1, n2, p1, p2 ):

    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values[k1,k2] = 10*k1 + k2

    # Create vector space and stencil matrix
    V = StencilVectorSpace( [n1,n2], [p1,p2] )
    M = StencilMatrix( V, V )

    # Fill in stencil matrix values
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M[:,:,k1,k2] = nonzero_values[k1,k2]

    # Convert stencil matrix to 2D array
    Ma = M.toarray()

    # Construct exact matrix by hand
    A = np.zeros( M.shape )
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1,p1+1):
                for k2 in range(-p2,p2+1):
                    j1 = (i1+k1) % n1
                    j2 = (i2+k2) % n2
                    i  = i1*(n2) + i2
                    j  = j1*(n2) + j2
                    A[i,j] = nonzero_values[k1,k2]

    # Check shape and data in 2D array
    assert Ma.shape == M.shape
    assert np.all( Ma == A )

#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )

def test_stencil_matrix_2d_serial_dot( n1, n2, p1, p2 ):

    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values[k1,k2] = 10*k1 + k2

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( [n1,n2], [p1,p2] )
    M = StencilMatrix( V, V )
    x = StencilVector( V )

    # Fill in stencil matrix values
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M[:,:,k1,k2] = nonzero_values[k1,k2]

    # Fill in vector values
    x[:,:] = 1.

    # Compute matrix-vector product
    y = M.dot(x)

    # Convert stencil objects to Numpy arrays
    Ma = M.toarray()
    xa = x.toarray()
    ya = y.toarray()

    # Exact result using Numpy dot product
    ya_exact = np.dot( Ma, xa )

    # Check data in 1D array
    assert np.all( ya == ya_exact )

#===============================================================================
# PARALLEL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [4,10,35] )
@pytest.mark.parallel

def test_stencil_matrix_1d_parallel_dot( n1 ):

    from mpi4py       import MPI
    from spl.ddm.cart import Cart

    p1 = 1

    comm = MPI.COMM_WORLD
    cart = Cart( npts    = [n1,],
                 pads    = [p1,],
                 periods = [True ,],
                 reorder = False,
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
    assert all( b.toarray() == 2.0 )

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
