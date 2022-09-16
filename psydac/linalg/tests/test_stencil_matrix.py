# -*- coding: UTF-8 -*-

import pytest
import numpy as np
from random import random

from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.api.settings   import *
from psydac.ddm.cart import DomainDecomposition, CartDecomposition

#===============================================================================
def compute_global_starts_ends(domain_decomposition, npts):
    ndims         = len(npts)
    global_starts = [None]*ndims
    global_ends   = [None]*ndims

    for axis in range(ndims):
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    return tuple(global_starts), tuple(global_ends)
#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )

def test_stencil_matrix_2d_serial_init( n1, n2, p1, p2, P1=True, P2=False ):

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V )

    assert M._data.shape == (n1+2*p1, n2+2*p2, 1+2*p1, 1+2*p2)
    assert M.shape == (n1*n2, n1*n2)

#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )

def test_stencil_matrix_2d_basic_ops( n1, n2, p1, p2, P1=True, P2=False ):

    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V )

    # take random data, but determinize it
    np.random.seed(2)

    M._data[:] = np.random.random(M._data.shape)

    assert M._data.shape == (n1+2*p1, n2+2*p2, 1+2*p1, 1+2*p2)
    assert M.shape == (n1*n2, n1*n2)

    # we try to go for equality here...
    assert np.array_equal((M * 2)._data, M._data * 2)
    assert np.array_equal((M / 2)._data, M._data / 2)
    assert np.array_equal((M + M)._data, M._data + M._data)
    assert np.array_equal((M - M)._data, M._data - M._data)

    M1 = M.copy()
    M1 *= 2
    M2 = M.copy()
    M2 /= 2
    M3 = M.copy()
    M3 += M
    M4 = M.copy()
    M4 -= M
    assert np.array_equal(M1._data, M._data * 2)
    assert np.array_equal(M2._data, M._data / 2)
    assert np.array_equal(M3._data, M._data + M._data)
    assert np.array_equal(M4._data, M._data - M._data)

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

    # Create domain decomposition
    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace( cart )
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

    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
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

def test_stencil_matrix_2d_serial_dot_1( n1, n2, p1, p2, P1, P2 ):

    # Create domain decomposition
    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
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
@pytest.mark.parametrize( 'n1', [5,15] )
@pytest.mark.parametrize( 'n2', [5,12] )
@pytest.mark.parametrize( 'p1', [2,3] )
@pytest.mark.parametrize( 'p2', [2,3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_dot_2( n1, n2, p1, p2, P1, P2 ):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1,n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1-1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )

    M1 = StencilMatrix( V1, V2 ,pads=(p1,p2))
    M2 = StencilMatrix( V2, V1 ,pads=(p1,p2))
    x1 = StencilVector( V1 )
    x2 = StencilVector( V2 )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    M1._data[p1:-p1, p2:-p2, :, :] = np.random.random(M1._data[p1:-p1, p2:-p2, :, :].shape)
    M2._data[p1:-p1, p2:-p2, :, :] = np.random.random(M2._data[p1:-p1, p2:-p2, :, :].shape)
    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2-1):
            x1[i1,i2] = 2.0*random() - 1.0
    x1.update_ghost_regions()

    for i1 in range(n1-1):
        for i2 in range(n2-1):
            x2[i1,i2] = 2.0*random() - 1.0
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot( M1a, x1a )
    y2a_exact = np.dot( M2a, x2a )

    # Check data in 1D array
    print(y2a-y2a_exact)
    assert np.allclose( y1a, y1a_exact, rtol=1e-13, atol=1e-13 )
    assert np.allclose( y2a, y2a_exact, rtol=1e-13, atol=1e-13 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [5,15] )
@pytest.mark.parametrize( 'n2', [5,12] )
@pytest.mark.parametrize( 'p1', [2,3] )
@pytest.mark.parametrize( 'p2', [2,3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_dot_3( n1, n2, p1, p2, P1, P2 ):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1,n2-1]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1-1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M1 = StencilMatrix( V1, V2 ,pads=(p1,p2-1))
    M2 = StencilMatrix( V2, V1 ,pads=(p1,p2-1))
    x1 = StencilVector( V1 )
    x2 = StencilVector( V2 )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    M1[0:n1-1, 0:n2-1, :, :] = np.random.random((n1-1, n2-1, 2*p1+1, 2*p2-1))
    M2[0:n1, 0:n2-1, :, :]   = np.random.random((n1, n2-1, 2*p1+1, 2*p2-1))

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2-1):
            x1[i1,i2] = 2.0*random() - 1.0
    x1.update_ghost_regions()

    for i1 in range(n1-1):
        for i2 in range(n2-1):
            x2[i1,i2] = 2.0*random() - 1.0
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot( M1a, x1a )
    y2a_exact = np.dot( M2a, x2a )

    # Check data in 1D array

    assert np.allclose( y1a, y1a_exact, rtol=1e-13, atol=1e-13 )
    assert np.allclose( y2a, y2a_exact, rtol=1e-13, atol=1e-13 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [5,15] )
@pytest.mark.parametrize( 'n2', [5,12] )
@pytest.mark.parametrize( 'p1', [2,3] )
@pytest.mark.parametrize( 'p2', [2,3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_dot_4( n1, n2, p1, p2, P1, P2 ):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1-1,n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1-1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M1 = StencilMatrix( V1, V2, pads=(p1-1,p2))
    M2 = StencilMatrix( V2, V1, pads=(p1-1,p2))
    x1 = StencilVector( V1 )
    x2 = StencilVector( V2 )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    M1[0:n1-1, 0:n2-1, :, :] = np.random.random((n1-1, n2-1, 2*p1-1, 2*p2+1))
    M2[0:n1-1, 0:n2  , :, :] = np.random.random((n1-1, n2  , 2*p1-1, 2*p2+1))
    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1-1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*random() - 1.0
    x1.update_ghost_regions()

    for i1 in range(n1-1):
        for i2 in range(n2-1):
            x2[i1,i2] = 2.0*random() - 1.0
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot( M1a, x1a )
    y2a_exact = np.dot( M2a, x2a )

    # Check data in 1D array

    assert np.allclose( y1a, y1a_exact, rtol=1e-13, atol=1e-13 )
    assert np.allclose( y2a, y2a_exact, rtol=1e-13, atol=1e-13 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [5,15] )
@pytest.mark.parametrize( 'n2', [5,12] )
@pytest.mark.parametrize( 'p1', [2,3] )
@pytest.mark.parametrize( 'p2', [2,3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_dot_5( n1, n2, p1, p2, P1, P2 ):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1-1,n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1-1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M1 = StencilMatrix( V1, V2 ,pads=(p1,p2))
    M2 = StencilMatrix( V2, V1 ,pads=(p1,p2))
    x1 = StencilVector( V1 )
    x2 = StencilVector( V2 )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    M1[0:n1-1, 0:n2-1, :, :] = np.random.random((n1-1, n2-1, 2*p1+1, 2*p2+1))
    M2[0:n1-1, 0:n2, :, :] = np.random.random((n1-1, n2, 2*p1+1, 2*p2+1))
    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1-1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*random() - 1.0
    x1.update_ghost_regions()

    for i1 in range(n1-1):
        for i2 in range(n2-1):
            x2[i1,i2] = 2.0*random() - 1.0
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot( M1a, x1a )
    y2a_exact = np.dot( M2a, x2a )

    # Check data in 1D array

    assert np.allclose( y1a, y1a_exact, rtol=1e-13, atol=1e-13 )
    assert np.allclose( y2a, y2a_exact, rtol=1e-13, atol=1e-13 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [5,15] )
@pytest.mark.parametrize( 'n2', [5,12] )
@pytest.mark.parametrize( 'p1', [2,3] )
@pytest.mark.parametrize( 'p2', [2,3] )
@pytest.mark.parametrize( 'P1', [True] )
@pytest.mark.parametrize( 'P2', [True] )

def test_stencil_matrix_2d_serial_dot_6( n1, n2, p1, p2, P1, P2 ):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V , pads=(p1-1, p2-1))
    x = StencilVector( V )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1+1,p1):
        for k2 in range(-p2+1,p2):
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
@pytest.mark.parametrize( 'n1', [4, 10, 32] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )

def test_stencil_matrix_1d_serial_transpose( n1, p1, P1 ):

    # Create domain decomposition
    D = DomainDecomposition([n1-1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V )

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1, -p1:p1+1] = np.random.random( (n1, 2*p1+1) )

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Numpy array
    Ta = M.transpose().toarray()

    # Exact result: convert to Numpy array, then transpose
    Ta_exact = M.toarray().transpose()

    # Check data
    assert np.array_equal( Ta, Ta_exact )

#===============================================================================
@pytest.mark.parametrize( 'n1', [5, 15] )
@pytest.mark.parametrize( 'n2', [5, 12] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'p2', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )

def test_stencil_matrix_2d_serial_transpose_1( n1, n2, p1, p2, P1, P2 ):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V )

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1, 0:n2, :, :] = np.random.random((n1, n2, 2*p1+1, 2*p2+1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
@pytest.mark.parametrize( 'n1', [5, 12] )
@pytest.mark.parametrize( 'n2', [6, 10] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'p2', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_transpose_2( n1, n2, p1, p2, P1, P2 ):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1,n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1-1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M  = StencilMatrix(V1, V2, pads=(p1,p2))

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1-1, 0:n2-1, :, :] = np.random.random((n1-1, n2-1, 2*p1+1, 2*p2+1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
@pytest.mark.parametrize( 'n1', [5, 12] )
@pytest.mark.parametrize( 'n2', [6, 10] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'p2', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_transpose_3( n1, n2, p1, p2, P1, P2 ):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1,n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1-1,n2]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M  = StencilMatrix(V1, V2, pads=(p1,p2))

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1-1, 0:n2, :, :] = np.random.random((n1-1, n2, 2*p1+1, 2*p2+1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
@pytest.mark.parametrize( 'n1', [5, 12] )
@pytest.mark.parametrize( 'n2', [6, 10] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'p2', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_transpose_4( n1, n2, p1, p2, P1, P2 ):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1,n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])


    # Create vector space and stencil matrix
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M  = StencilMatrix(V1, V2, pads=(p1,p2))

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1, 0:n2-1, :, :] = np.random.random((n1, n2-1, 2*p1+1, 2*p2+1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
@pytest.mark.parametrize( 'n1', [7, 12] )
@pytest.mark.parametrize( 'n2', [7, 10] )
@pytest.mark.parametrize( 'p1', [2, 3] )
@pytest.mark.parametrize( 'p2', [2, 3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_transpose_5( n1, n2, p1, p2, P1, P2 ):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1,n2-1]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M  = StencilMatrix(V1, V2, pads=(p1, p2-1))

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1, 0:n2-1, :, :] = np.random.random((n1, n2-1, 2*p1+1, 2*p2-1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
@pytest.mark.parametrize( 'n1', [7, 12] )
@pytest.mark.parametrize( 'n2', [7, 10] )
@pytest.mark.parametrize( 'p1', [2, 3] )
@pytest.mark.parametrize( 'p2', [2, 3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_transpose_6( n1, n2, p1, p2, P1, P2 ):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1-1,n2-1]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M  = StencilMatrix(V1, V2, pads=(p1, p2-1))

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1, 0:n2-1, :, :] = np.random.random((n1, n2-1, 2*p1+1, 2*p2-1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
@pytest.mark.parametrize( 'n1', [7, 12] )
@pytest.mark.parametrize( 'n2', [7, 10] )
@pytest.mark.parametrize( 'p1', [2, 3] )
@pytest.mark.parametrize( 'p2', [2, 3] )
@pytest.mark.parametrize( 'P1', [True] )
@pytest.mark.parametrize( 'P2', [True] )

def test_stencil_matrix_2d_serial_transpose_7( n1, n2, p1, p2, P1, P2 ):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace( cart )
    V2 = StencilVectorSpace( cart )
    M  = StencilMatrix(V1, V2, pads=(p1, p2-1))

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1, 0:n2, :, :] = np.random.random((n1, n2, 2*p1+1, 2*p2-1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
@pytest.mark.parametrize( 'n1', [7, 12] )
@pytest.mark.parametrize( 'n2', [7, 10] )
@pytest.mark.parametrize( 'p1', [2, 3] )
@pytest.mark.parametrize( 'p2', [2, 3] )
@pytest.mark.parametrize( 'P1', [True] )
@pytest.mark.parametrize( 'P2', [True] )

def test_stencil_matrix_2d_serial_transpose_8( n1, n2, p1, p2, P1, P2 ):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace( cart )
    V2 = StencilVectorSpace( cart )
    M  = StencilMatrix(V1, V2, pads=(p1-1, p2-1))

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1, 0:n2, :, :] = np.random.random((n1, n2, 2*p1-1, 2*p2-1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
@pytest.mark.parametrize( 'n1', [7, 12] )
@pytest.mark.parametrize( 'n2', [7, 10] )
@pytest.mark.parametrize( 'p1', [2, 3] )
@pytest.mark.parametrize( 'p2', [2, 3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )

def test_stencil_matrix_2d_serial_transpose_9( n1, n2, p1, p2, P1, P2 ):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts = [n1-1,n2-1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])


    # Create vector space and stencil matrix
    V1 = StencilVectorSpace( cart )
    V2 = StencilVectorSpace( cart )
    M  = StencilMatrix(V1, V2)

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1-1, 0:n2-1, :, :] = np.random.random((n1-1, n2-1, 2*p1+1, 2*p2+1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
@pytest.mark.parametrize( 'n1', [7, 12] )
@pytest.mark.parametrize( 'n2', [7, 10] )
@pytest.mark.parametrize( 'n3', [7, 10] )
@pytest.mark.parametrize( 'p1', [2, 3] )
@pytest.mark.parametrize( 'p2', [2, 3] )
@pytest.mark.parametrize( 'p3', [2, 3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )
@pytest.mark.parametrize( 'P3', [False] )

def test_stencil_matrix_3d_serial_transpose_1( n1, n2, n3, p1, p2, p3, P1, P2, P3 ):
    # This should only work with non periodic boundaries

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1, n3-1], periods=[P1,P2,P3])

    # Partition the points
    npts1 = [n1-1,n2-1, n3-1]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1,n2-1, n3-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2,p3], shifts=[1,1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2,p3], shifts=[1,1,1])

    # Create vector space and stencil matrix
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M  = StencilMatrix(V1, V2, pads=(p1, p2-1, p3-1))

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1, 0:n2-1, 0:n3-1, :, :, :] = np.random.random((n1, n2-1, n3-1, 2*p1+1, 2*p2-1, 2*p3-1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14


#===============================================================================
# BACKENDS TESTS
#===============================================================================

@pytest.mark.parametrize( 'n1', [5,15] )
@pytest.mark.parametrize( 'n2', [5,12] )
@pytest.mark.parametrize( 'p1', [2,3] )
@pytest.mark.parametrize( 'p2', [2,3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )
@pytest.mark.parametrize( 'backend', [PSYDAC_BACKEND_NUMBA, PSYDAC_BACKEND_GPYCCEL] )

def test_stencil_matrix_2d_serial_backend_dot_1( n1, n2, p1, p2, P1, P2 , backend):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1-1,n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1-1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M1 = StencilMatrix( V1, V2 ,pads=(p1,p2), backend=backend)
    M2 = StencilMatrix( V2, V1 ,pads=(p1,p2), backend=backend)
    x1 = StencilVector( V1 )
    x2 = StencilVector( V2 )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    M1[0:n1-1, 0:n2-1, :, :] = np.random.random((n1-1, n2-1, 2*p1+1, 2*p2+1))
    M2[0:n1-1, 0:n2, :, :] = np.random.random((n1-1, n2, 2*p1+1, 2*p2+1))
    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1-1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*random() - 1.0
    x1.update_ghost_regions()

    for i1 in range(n1-1):
        for i2 in range(n2-1):
            x2[i1,i2] = 2.0*random() - 1.0
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot( M1a, x1a )
    y2a_exact = np.dot( M2a, x2a )

    # Check data in 1D array

    assert np.allclose( y1a, y1a_exact, rtol=1e-13, atol=1e-13 )
    assert np.allclose( y2a, y2a_exact, rtol=1e-13, atol=1e-13 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [5,15] )
@pytest.mark.parametrize( 'n2', [5,12] )
@pytest.mark.parametrize( 'p1', [2,3] )
@pytest.mark.parametrize( 'p2', [2,3] )
@pytest.mark.parametrize( 'P1', [True] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.parametrize( 'backend', [None, PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_NUMBA, PSYDAC_BACKEND_GPYCCEL] )

def test_stencil_matrix_2d_serial_backend_dot_2( n1, n2, p1, p2, P1, P2 , backend):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts = [n1-1,n2-1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V , pads=(p1-1, p2-1), backend=backend)
    x = StencilVector( V )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1+1,p1):
        for k2 in range(-p2+1,p2):
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

    # tests for backend propagation
    assert M.backend is backend
    assert M.T.backend is M.backend
    assert (M+M).backend is M.backend
    assert (2*M).backend is M.backend

#===============================================================================
@pytest.mark.parametrize( 'n1', [5,15] )
@pytest.mark.parametrize( 'n2', [5,12] )
@pytest.mark.parametrize( 'p1', [2,3] )
@pytest.mark.parametrize( 'p2', [2,3] )
@pytest.mark.parametrize( 'P1', [False] )
@pytest.mark.parametrize( 'P2', [False] )
@pytest.mark.parametrize( 'backend', [PSYDAC_BACKEND_NUMBA, PSYDAC_BACKEND_GPYCCEL] )

def test_stencil_matrix_2d_serial_backend_dot_4( n1, n2, p1, p2, P1, P2, backend):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts1 = [n1-1,n2]
    global_starts1, global_ends1 = compute_global_starts_ends(D, npts1)

    npts2 = [n1-1,n2-1]
    global_starts2, global_ends2 = compute_global_starts_ends(D, npts2)

    cart1 = CartDecomposition(D, npts1, global_starts1, global_ends1, pads=[p1,p2], shifts=[1,1])
    cart2 = CartDecomposition(D, npts2, global_starts2, global_ends2, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V1 = StencilVectorSpace( cart1 )
    V2 = StencilVectorSpace( cart2 )
    M1 = StencilMatrix( V1, V2, pads=(p1-1,p2), backend=backend)
    M2 = StencilMatrix( V2, V1, pads=(p1-1,p2), backend=backend)
    x1 = StencilVector( V1 )
    x2 = StencilVector( V2 )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    M1[0:n1-1, 0:n2-1, :, :] = np.random.random((n1-1, n2-1, 2*p1-1, 2*p2+1))
    M2[0:n1-1, 0:n2  , :, :] = np.random.random((n1-1, n2  , 2*p1-1, 2*p2+1))
    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1-1):
        for i2 in range(n2):
            x1[i1,i2] = 2.0*random() - 1.0
    x1.update_ghost_regions()

    for i1 in range(n1-1):
        for i2 in range(n2-1):
            x2[i1,i2] = 2.0*random() - 1.0
    x2.update_ghost_regions()

    # Compute matrix-vector product
    y1 = M1.dot(x1)
    y2 = M2.dot(x2)

    # Convert stencil objects to Numpy arrays
    M1a = M1.toarray()
    x1a = x1.toarray()
    y1a = y1.toarray()

    M2a = M2.toarray()
    x2a = x2.toarray()
    y2a = y2.toarray()

    # Exact result using Numpy dot product
    y1a_exact = np.dot( M1a, x1a )
    y2a_exact = np.dot( M2a, x2a )

    # Check data in 1D array

    assert np.allclose( y1a, y1a_exact, rtol=1e-13, atol=1e-13 )
    assert np.allclose( y2a, y2a_exact, rtol=1e-13, atol=1e-13 )

#===============================================================================
@pytest.mark.parametrize( 'n1', [15] )
@pytest.mark.parametrize( 'n2', [12] )
@pytest.mark.parametrize( 'p1', [2] )
@pytest.mark.parametrize( 'p2', [3] )
@pytest.mark.parametrize( 'P1', [True] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.parametrize( 'backend', [None, PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_NUMBA, PSYDAC_BACKEND_GPYCCEL] )
@pytest.mark.parametrize( 'backend2', [None, PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_NUMBA, PSYDAC_BACKEND_GPYCCEL] )

def test_stencil_matrix_2d_serial_backend_switch( n1, n2, p1, p2, P1, P2 , backend, backend2):

    # Create domain decomposition
    D = DomainDecomposition([n1-1,n2-1], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V , pads=(p1-1, p2-1), backend=backend)
    x = StencilVector( V )

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1+1,p1):
        for k2 in range(-p2+1,p2):
            M[:,:,k1,k2] = 10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        for i2 in range(n2):
            x[i1,i2] = 2.0*random() - 1.0
    x.update_ghost_regions()

    assert M.backend is backend
    M.dot(x)
    M.set_backend(backend2)

    assert M.backend is backend2
    M.dot(x)

#===============================================================================
# PARALLEL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [20,67] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_1d_parallel_dot( n1, p1, P1 ):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1-1], periods=[P1], comm=comm)

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[1])

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
@pytest.mark.parallel

def test_stencil_matrix_2d_parallel_dot( n1, n2, p1, p2, P1, P2 ):

    from mpi4py       import MPI

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1,n2], periods=[P1,P2], comm=comm)

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

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
@pytest.mark.parametrize( 'n1', [20,67] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_1d_parallel_sync( n1, p1, P1):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[1])

    V = StencilVectorSpace( cart, dtype=int )
    M = StencilMatrix( V, V )

    s1, = V.starts
    e1, = V.ends

    # Fill-in pattern
    fill_in = lambda i1, k1 : 10*i1+k1

    # Fill in stencil matrix
    for i1 in range(s1, e1+1):
        for k1 in range(-p1, p1+1):
            M[i1,k1] = fill_in( i1, k1 )

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: update ghost regions
    M.update_ghost_regions()

    # Convert stencil object to 1D Numpy array
    Ma = M.toarray( with_pads=True )

    # Create exact solution
    Me = np.zeros( (n1,n1), dtype=V.dtype )

    for i1 in range(n1):
        for k1 in range(-p1, p1+1):

            # Get column index
            j1 = i1 + k1

            # If j1 is outside matrix limits, apply periodic BCs or skip entry
            if not 0 <= j1 < n1:
                if P1:
                    j1 = j1 % n1
                else:
                    continue

            # Fill in matrix element
            Me[i1,j1] = fill_in( i1, k1 )

    # Compare local solution to global
    i1_min = max(0, s1-p1)
    i1_max = min(e1+p1+1, n1)

#    for i in range( comm.size ):
#        if i == comm.rank:
#            print( "RANK {}:".format( i ) )
#            print( M._data.shape )
#            print( Ma.shape )
#            print( Ma )
#            print( "PASSED" )
#            print( flush=True )
#        comm.Barrier()

    assert np.array_equal( Ma[i1_min:i1_max, :], Me[i1_min:i1_max, :] )

#===============================================================================
@pytest.mark.parametrize( 'n1', [21,67] )
@pytest.mark.parametrize( 'n2', [13,32] )
@pytest.mark.parametrize( 'p1', [1,3] )
@pytest.mark.parametrize( 'p2', [1,2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_2d_parallel_sync( n1, n2, p1, p2, P1, P2):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    V = StencilVectorSpace( cart, dtype=int )
    M = StencilMatrix( V, V )

    s1, s2 = V.starts
    e1, e2 = V.ends

    # Fill-in pattern
    fill_in = lambda i1, i2, k1, k2: 1000*i1 + 100*i2 + 10*abs(k1) + abs(k2)

    # Fill in stencil matrix
    for i1 in range(s1, e1+1):
        for i2 in range(s2, e2+1):
            for k1 in range(-p1, p1+1):
                for k2 in range(-p2, p2+1):
                    M[i1, i2, k1, k2] = fill_in( i1, i2, k1, k2 )

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: update ghost regions
    M.update_ghost_regions()

    # Convert stencil object to 1D Numpy array
    Ma = M.toarray( with_pads=True )

    # Create exact solution
    Me = np.zeros( (n1*n2, n1*n2), dtype=V.dtype )

    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1, p1+1):
                for k2 in range(-p2, p2+1):

                    # Get column multi-index
                    j1 = i1 + k1
                    j2 = i2 + k2

                    # If j1 is outside matrix limits,
                    # apply periodic BCs or skip entry
                    if not 0 <= j1 < n1:
                        if P1:
                            j1 = j1 % n1
                        else:
                            continue

                    # If j2 is outside matrix limits,
                    # apply periodic BCs or skip entry
                    if not 0 <= j2 < n2:
                        if P2:
                            j2 = j2 % n2
                        else:
                            continue

                    # Get matrix indices assuming C ordering
                    i = i1 * n2 + i2
                    j = j1 * n2 + j2

                    # Fill in matrix element
                    Me[i,j] = fill_in( i1, i2, k1, k2 )

#    #++++++++++++++++++++++++++++++++++++++
#    # DEBUG
#    #++++++++++++++++++++++++++++++++++++++
#    np.set_printoptions( linewidth=200 )
#
#    if comm.rank == 0:
#        print( 'Me' )
#        print( Me )
#        print( flush=True )
#    comm.Barrier()
#
#    for i in range(comm.size):
#        if i == comm.rank:
#            print( 'RANK {}'.format( i ) )
#            print( Ma )
#            print( flush=True )
#        comm.Barrier()
#    #++++++++++++++++++++++++++++++++++++++

    # Compare local solution to global, row by row
    i1_min = max(0, s1-p1)
    i1_max = min(e1+p1+1, n1)

    i2_min = max(0, s2-p2)
    i2_max = min(e2+p2+1, n2)

    for i1 in range( i1_min, i1_max ):
        for i2 in range( i2_min, i2_max ):
            i = i1 * n2 + i2
            assert np.array_equal( Ma[i,:], Me[i,:] )

#===============================================================================
@pytest.mark.parametrize( 'n1', [20, 67] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_1d_parallel_transpose( n1, p1, P1 ):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1-1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V )

    s1, = V.starts
    e1, = V.ends

    # Fill in matrix values with random numbers between 0 and 1
    M[s1:e1+1, -p1:p1+1] = np.random.random( (e1-s1+1, 2*p1+1) )

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Numpy array
    Ta = M.transpose().toarray()

    # Exact result: convert to Numpy array including padding, then transpose,
    # hence remove entries that do not belong to current process.
    Ta_exact = M.toarray( with_pads=True ).transpose()
    Ta_exact[  :s1, :] = 0.0
    Ta_exact[e1+1:, :] = 0.0

    # Check data
    assert np.array_equal( Ta, Ta_exact )

#===============================================================================
@pytest.mark.parametrize( 'n1', [ 8, 21] )
@pytest.mark.parametrize( 'n2', [13, 32] )
@pytest.mark.parametrize( 'p1', [1, 3] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_2d_parallel_transpose( n1, n2, p1, p2, P1, P2 ):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space and stencil matrix
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V )

    s1, s2 = V.starts
    e1, e2 = V.ends

    # Fill in matrix values with random numbers between 0 and 1
    M[s1:e1+1, s2:e2+1, -p1:p1+1, -p2:p2+1] = np.random.random(
            (e1-s1+1, e2-s2+1, 2*p1+1, 2*p2+1))

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Exact result: convert to Scipy sparse format including padding, then
    # transpose, hence remove entries that do not belong to current process.
    Ts_exact = M.tosparse( with_pads=True ).transpose()

    #...
    Ts_exact = Ts_exact.tocsr()
    for i, j in zip(*Ts_exact.nonzero()):
        i1, i2 = np.unravel_index( i, shape=[n1, n2], order='C' )
        if not (s1 <= i1 <= e1 and s2 <= i2 <= e2):
            Ts_exact[i, j] = 0.0
    Ts_exact = Ts_exact.tocoo()
    Ts_exact.eliminate_zeros()
    #...

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

#===============================================================================
# PARALLEL BACKENDS TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [20,67] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'backend', [PSYDAC_BACKEND_NUMBA, PSYDAC_BACKEND_GPYCCEL] )
@pytest.mark.parallel

def test_stencil_matrix_1d_parallel_backend_dot( n1, p1, P1 , backend):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1-1], periods=[P1])

    # Partition the points
    npts = [n1]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1], shifts=[1])

    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V , backend=backend)
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
@pytest.mark.parametrize( 'backend', [None, PSYDAC_BACKEND_NUMBA, PSYDAC_BACKEND_GPYCCEL] )
@pytest.mark.parallel

def test_stencil_matrix_2d_parallel_backend_dot( n1, n2, p1, p2, P1, P2, backend):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    # Create domain decomposition
    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V , backend=backend)
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

    # tests for backend propagation
    assert M.backend is backend
    assert M.T.backend is M.backend
    assert (M+M).backend is M.backend
    assert (2*M).backend is M.backend

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
