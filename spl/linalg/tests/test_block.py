# -*- coding: UTF-8 -*-
#
import pytest
import numpy as np
from random import random

from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from spl.linalg.block   import ProductSpace, BlockVector, BlockLinearOperator

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
def test_block_linear_operator_serial_dot( n1, n2, p1, p2, P1, P2  ):
    # Create vector spaces, stencil matrices, and stencil vectors
    V = StencilVectorSpace( [n1,n2], [p1,p2], [P1,P2] )
    M1 = StencilMatrix( V, V )
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    # Fill in stencil matrices based on diagonal index
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = 10*k1+k2
            M2[:,:,k1,k2] = 10*k1+k2+2.
            M3[:,:,k1,k2] = 10*k1+k2+5.
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    W = ProductSpace(V, V)
    # Construct a BlockLinearOperator object containing M1, M2, M, using 3 ways
    #     |M1  M2|
    # L = |      |
    #     |M3  0 |
    dict_blocks = {(0,0):M1, (0,1):M2, (1,0):M3}

    L1 = BlockLinearOperator(W, W, blocks=dict_blocks)

    L2 = BlockLinearOperator(W, W)
    L2.set_blocks(dict_blocks)

    L3 = BlockLinearOperator( W, W )
    L3[(0,0)] = M1
    L3[(0,1)] = M2
    L3[(1,0)] = M3

    # Fill in vector with random values, then update ghost regions
    for i1 in range(n1):
        x1[i1] = 2.0*random() - 1.0
        x2[i1] = 5.0*random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Construct a BlockVector object containing x1 and x2
    #     |x1|
    # X = |  |
    #     |x2|

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Compute BlockLinearOperator product
    Y1 = L1.dot(X)
    Y2 = L2.dot(X)
    Y3 = L3.dot(X)

    # Compute matrix-vector products for each block
    y1 = M1.dot(x1) + M2.dot(x2)
    y2 = M3.dot(x1)

    # Check data in 1D array
    assert np.allclose( Y1.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y1.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y2.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y2.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

    assert np.allclose( Y3.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y3.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

#===============================================================================
# PARALLEL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [ 8,16] )
@pytest.mark.parametrize( 'n2', [8,32] )
@pytest.mark.parametrize( 'p1', [1,3] )
@pytest.mark.parametrize( 'p2', [1,2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'reorder', [True, False] )
@pytest.mark.parallel

def test_block_linear_operator_parallel_dot( n1, n2, p1, p2, P1, P2, reorder ):

    from mpi4py       import MPI
    from spl.ddm.cart import Cart

    comm = MPI.COMM_WORLD
    cart = Cart( npts    = [n1,n2],
                 pads    = [p1,p2],
                 periods = [P1,P2],
                 reorder = reorder,
                 comm    = comm )

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
    M1 = StencilMatrix( V, V )
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )
    M4 = StencilMatrix( V, V )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    s1,s2 = V.starts
    e1,e2 = V.ends

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M1[:,:,k1,k2] = k1+k2+10.
            M2[:,:,k1,k2] = 2.*k1+k2
            M3[:,:,k1,k2] = 5*k1+k2
            M4[:,:,k1,k2] = 10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()
    M4.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x1[i1,i2] = 2.0*random() + 1.0
            x2[i1,i2] = 5.0*random() - 1.0
    x1.update_ghost_regions()
    x2.update_ghost_regions()

    # Create and Fill Block objects
    W = ProductSpace(V, V)
    L = BlockLinearOperator( W, W )
    L[(0,0)] = M1
    L[(0,1)] = M2
    L[(1,0)] = M3
    L[(1,1)] = M4

    X = BlockVector(W)
    X[0] = x1
    X[1] = x2

    # Compute Block-vector product
    Y = L.dot(X)

    # Compute matrix-vector products for each block
    y1 = M1.dot(x1) + M2.dot(x2)
    y2 = M3.dot(x1) + M4.dot(x2)

    # Check data in 1D array
    assert np.allclose( Y.blocks[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y.blocks[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )

