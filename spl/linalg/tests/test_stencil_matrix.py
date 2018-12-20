# -*- coding: UTF-8 -*-

import pytest
import numpy as np
from random import random

from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )

def test_stencil_matrix_2d_serial_init( n1, n2, p1, p2, P1=True, P2=False ):

    V = StencilVectorSpace( [n1,n2], [p1,p2], [P1,P1] )
    M = StencilMatrix( V, V )

    assert M._data.shape == (n1+2*p1, n2+2*p2, 1+2*p1, 1+2*p2)
    assert M.shape == (n1*n2, n1*n2)

#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )

def test_stencil_matrix_2d_serial_toarray( n1, n2, p1, p2, P1=False, P2=True ):

    # Select non-zero values based on diagonal index
    nonzero_values = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values[k1,k2] = 10*k1 + k2

    # Create vector space and stencil matrix
    V = StencilVectorSpace( [n1,n2], [p1,p2], [P1,P2] )
    M = StencilMatrix( V, V )

    # Fill in stencil matrix values
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M[:,:,k1,k2] = nonzero_values[k1,k2]

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

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
                    if (P1 or 0 <= i1+k1 < n1) and (P2 or 0 <= i2+k2 < n2):
                        A[i,j] = nonzero_values[k1,k2]

    # Check shape and data in 2D array
    assert Ma.shape == M.shape
    assert np.all( Ma == A )

#===============================================================================
@pytest.mark.parametrize( 'n1', [10,32] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )

def test_stencil_matrix_1d_serial_dot( n1, p1, P1 ):

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( [n1,], [p1,], [P1,] )
    M = StencilMatrix( V, V )
    x = StencilVector( V )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        M[:,k1] = k1

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        x[i1] = 2.0*random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    assert isinstance( y, StencilVector )
    assert y.space is x.space

    # Convert stencil objects to Scipy sparse matrix and 1D Numpy arrays
    Ms = M.tosparse()
    xa = x.toarray()
    ya = y.toarray()

    # Exact result using Scipy sparse dot product
    ya_exact = Ms.dot( xa )

    # Check data in 1D array
    assert np.allclose( ya, ya_exact, rtol=1e-14, atol=1e-14 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [5,15] )
@pytest.mark.parametrize( 'n2', [5,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )

def test_stencil_matrix_2d_serial_dot( n1, n2, p1, p2, P1, P2 ):

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( [n1,n2], [p1,p2], [P1,P2] )
    M = StencilMatrix( V, V )
    x = StencilVector( V )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M[:,:,k1,k2] = 10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x[i1,i2] = 2.0*random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    # Convert stencil objects to Numpy arrays
    Ma = M.toarray()
    xa = x.toarray()
    ya = y.toarray()

    # Exact result using Numpy dot product
    ya_exact = np.dot( Ma, xa )

    # Check data in 1D array
    assert np.allclose( ya, ya_exact, rtol=1e-13, atol=1e-13 )

#===============================================================================
# PARALLEL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [20,67] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_1d_parallel_dot( n1, p1, P1, reorder ):

    from mpi4py       import MPI
    from spl.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    cart = CartDecomposition(
        npts    = [n1,],
        pads    = [p1,],
        periods = [P1,],
        reorder = reorder,
        comm    = comm
    )

    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V )
    x = StencilVector( V )

    s1, = V.starts
    e1, = V.ends

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        M[:,k1] = k1

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(x.starts[0],x.ends[0]+1):
        x[i1] = 2.0*random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot( x )

    assert isinstance( y, StencilVector )
    assert y.space is x.space

    # Convert stencil objects to Scipy sparse matrix and 1D Numpy arrays
    Ms = M.tosparse()
    xa = x.toarray( with_pads=True )
    ya = y.toarray()

    # Exact result using Scipy sparse dot product
    ya_exact = Ms.dot( xa )

    # Check data in 1D array
    assert np.allclose( ya, ya_exact, rtol=1e-14, atol=1e-14 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [ 8,21] )
@pytest.mark.parametrize( 'n2', [13,32] )
@pytest.mark.parametrize( 'p1', [1,3] )
@pytest.mark.parametrize( 'p2', [1,2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_2d_parallel_dot( n1, n2, p1, p2, P1, P2, reorder ):

    from mpi4py       import MPI
    from spl.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    cart = CartDecomposition(
        npts    = [n1,n2],
        pads    = [p1,p2],
        periods = [P1,P2],
        reorder = reorder,
        comm    = comm
    )

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V )
    x = StencilVector( V )

    s1,s2 = V.starts
    e1,e2 = V.ends

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M[:,:,k1,k2] = 10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x[i1,i2] = 2.0*random() - 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)

    assert isinstance( y, StencilVector )
    assert y.space is x.space

    # Convert stencil objects to Numpy arrays
    Ma = M.toarray()
    xa = x.toarray( with_pads=True )
    ya = y.toarray()

    # Exact result using Numpy dot product
    ya_exact = np.dot( Ma, xa )

    # Check data in 1D array
    assert np.allclose( ya, ya_exact, rtol=1e-13, atol=1e-13 )

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
