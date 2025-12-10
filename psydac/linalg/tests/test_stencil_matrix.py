# -*- coding: UTF-8 -*-

import pytest
import numpy as np
from random import random

from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix

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
@pytest.mark.parametrize( 'n1', [4, 10, 32] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )

def test_stencil_matrix_1d_serial_transpose( n1, p1, P1 ):

    # Create vector space and stencil matrix
    V = StencilVectorSpace( [n1], [p1], [P1] )
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

def test_stencil_matrix_2d_serial_transpose( n1, n2, p1, p2, P1, P2 ):

    # Create vector space and stencil matrix
    V = StencilVectorSpace( [n1, n2], [p1, p2], [P1, P2] )
    M = StencilMatrix( V, V )

    # Fill in matrix values with random numbers between 0 and 1
    M[0:n1, 0:n2, -p1:p1+1, -p2:p2+1] = np.random.random((n1, n2, 2*p1+1, 2*p2+1))
    M.remove_spurious_entries()

    # If domain is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # TEST: compute transpose, then convert to Scipy sparse format
    Ts = M.transpose().tosparse()

    # Exact result: convert to Scipy sparse format, then transpose
    Ts_exact = M.tosparse().transpose()

    # Check data
    assert abs(Ts - Ts_exact).max() < 1e-14

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
    from psydac.ddm.cart import CartDecomposition

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
    from psydac.ddm.cart import CartDecomposition

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
@pytest.mark.parametrize( 'n1', [20,67] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_1d_parallel_sync( n1, p1, P1, reorder ):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    cart = CartDecomposition(
        npts    = [n1,],
        pads    = [p1,],
        periods = [P1,],
        reorder = reorder,
        comm    = comm
    )

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
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_2d_parallel_sync( n1, n2, p1, p2, P1, P2, reorder ):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    cart = CartDecomposition(
        npts    = [n1, n2],
        pads    = [p1, p2],
        periods = [P1, P2],
        reorder = reorder,
        comm    = comm
    )

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
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_1d_parallel_transpose( n1, p1, P1, reorder ):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    cart = CartDecomposition(
        npts    = [n1,],
        pads    = [p1,],
        periods = [P1,],
        reorder = reorder,
        comm    = comm
    )

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
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel

def test_stencil_matrix_2d_parallel_transpose( n1, n2, p1, p2, P1, P2, reorder ):

    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition

    comm = MPI.COMM_WORLD
    cart = CartDecomposition(
        npts    = [n1, n2],
        pads    = [p1, p2],
        periods = [P1, P2],
        reorder = reorder,
        comm    = comm
    )

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
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
