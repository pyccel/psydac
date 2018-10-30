# -*- coding: UTF-8 -*-

import pytest
import numpy as np
from random import random

from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from spl.linalg.block   import ProductSpace, BlockVector, BlockLinearOperator

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', [10,32] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'P1', [True, False] )

def test_block_linear_operator_serial_dot_01( n1, p1, P1 ):
    # Create vector space, stencil matrices, and stencil vectors
    V = StencilVectorSpace( [n1,], [p1,], [P1,] )
    M1 = StencilMatrix( V, V )
    M2 = StencilMatrix( V, V )
    M3 = StencilMatrix( V, V )
    x1 = StencilVector( V )
    x2 = StencilVector( V )

    # Fill in stencil matrices based on diagonal index
    for k1 in range(-p1,p1+1):
        M1[:,k1] = k1+1.
        M2[:,k1] = 10.*k1+1.
        M3[:,k1] = k1+10.
    M1.remove_spurious_entries()
    M2.remove_spurious_entries()
    M3.remove_spurious_entries()

    # Construct a BlockLinearOperator object containing M1, M2, M3
    #     |M1  M2|
    # L = |      |
    #     |M3  0 |
    L = BlockLinearOperator( {(0,0):M1, (0,1):M2, (1,0):M3} )

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
    X = BlockVector([x1, x2])

    # Compute BlockLinearOperator product
    Y = L.dot(X)

    # Compute matrix-vector products for comapraisons
    y1 = M1.dot(x1) + M2.dot(x2)
    y2 = M3.dot(x1)

    # Check data in 1D array
    assert np.allclose( Y.block_list[0].toarray(), y1.toarray(), rtol=1e-14, atol=1e-14 )
    assert np.allclose( Y.block_list[1].toarray(), y2.toarray(), rtol=1e-14, atol=1e-14 )

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
