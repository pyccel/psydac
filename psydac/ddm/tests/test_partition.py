#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest

from psydac.ddm.partition import compute_dims

#==============================================================================
@pytest.mark.parametrize( 'mpi_size', [1,2,5,10] )

def test_partition_1d_uniform( mpi_size ):

    # ...
    # Should pass: all blocks are identical and have size=11
    n1 = 11 * mpi_size
    p1 = 3

    dims, blocksizes = compute_dims( mpi_size, [n1,], [p1,] ) 

    assert dims[0] == mpi_size
    assert blocksizes[0] == 11

    # ...
    # Should fail: minimum block size is too large
    n1 = 4 * mpi_size
    p1 = 5

    with pytest.raises( Exception ):
        dims, blocksizes = compute_dims( mpi_size, [n1,], [p1,] )

#==============================================================================
@pytest.mark.parametrize( 'mpi_size', [1,2,5,10] )

def test_partition_1d_general( mpi_size ):

    # ...
    # Should pass, nominal block size is 11
    n1 = 11 * mpi_size + int( mpi_size > 1 )
    p1 = 4

    dims, blocksizes = compute_dims( mpi_size, [n1,], [p1,] ) 

    assert dims[0] == mpi_size
    assert blocksizes[0] == 11

    # ...
    # Should fail: minimum block size is too large
    n1 = 4 * mpi_size + int( mpi_size > 1 )
    p1 = 5

    with pytest.raises( Exception ):
        dims, blocksizes = compute_dims( mpi_size, [n1,], [p1,] )

#==============================================================================
@pytest.mark.parametrize( 'mpi_size', [1,2,5,10] )
@pytest.mark.parametrize( 'mask', [[True, False], [False, True]] )
@pytest.mark.parametrize( 'npts', [[64, 64], [58, 64], [64, 31]] )

def test_partition_2d_dims_mask( mpi_size, npts, mask ):
  
    # General partition: blocks are not all identical but closer to a cube
    dims, blocksizes = compute_dims( mpi_size, npts, [3,3] )   
    
    # Mask dimensions
    dims, blocksizes = compute_dims( mpi_size, npts, [3,3], mpi_dims_mask=mask )   
    
    # test
    assert dims[0]*dims[1] == mpi_size
    for bsize, n, use_dim in zip(blocksizes, npts, mask):
        if not use_dim:
            assert bsize == n
        else:
            assert bsize == n//mpi_size

#==============================================================================
def test_partition_3d():

    npts = [64,128,50]
    mpi_size = 100

    # ...
    # Uniform partition, yields small block size along 3rd dimension
    dims, blocksizes = compute_dims( mpi_size, npts, try_uniform=True )    

    assert tuple( dims ) == (2, 2, 25)
    assert tuple( blocksizes ) == (32, 64, 2)

    # ...
    # General partition: blocks are not all identical but closer to a cube
    dims, blocksizes = compute_dims( mpi_size, npts, [3,3,3] )    

    assert tuple( dims ) == (5, 5, 4)
    assert tuple( blocksizes ) == (12, 25, 12)
    
#==============================================================================
@pytest.mark.parametrize( 'mpi_size', [1,2,5,10] )
@pytest.mark.parametrize( 'mask', [[True, False, False], 
                                   [False, True, False], 
                                   [False, False, True], 
                                   [True, True, False],
                                   [True, False, True],
                                   [False, True, True]] )
@pytest.mark.parametrize( 'npts', [[32, 64, 128], [62, 59, 41]] )

def test_partition_3d_dims_mask( mpi_size, npts, mask ):
   
    # General partition: blocks are not all identical but closer to a cube
    dims, blocksizes = compute_dims( mpi_size, npts, [3,3,3] )   
    
    # Mask dimensions
    dims, blocksizes = compute_dims( mpi_size, npts, [3,3,3], mpi_dims_mask=mask )   
    
    # test
    assert dims[0]*dims[1]*dims[2] == mpi_size
    for bsize, n, use_dim in zip(blocksizes, npts, mask):
        if not use_dim:
            assert bsize == n
    
    
if __name__ == '__main__':
    # test_partition_2d_dims_mask(10, [64, 64], [True, False])
    # test_partition_2d_dims_mask(10, [58, 64], [True, False])
    
    test_partition_3d_dims_mask(10, [32, 64, 128], [True, False, True])
