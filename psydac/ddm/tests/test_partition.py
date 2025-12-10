import pytest

from psydac.ddm.partition import mpi_compute_dims

#==============================================================================
@pytest.mark.parametrize( 'mpi_size', [1,2,5,10] )

def test_partition_1d_uniform( mpi_size ):

    # ...
    # Should pass: all blocks are identical and have size=11
    n1 = 11 * mpi_size
    p1 = 3

    dims, blocksizes = mpi_compute_dims( mpi_size, [n1,], [p1,] ) 

    assert dims[0] == mpi_size
    assert blocksizes[0] == 11

    # ...
    # Should fail: minimum block size is too large
    n1 = 4 * mpi_size
    p1 = 5

    with pytest.raises( Exception ):
        dims, blocksizes = mpi_compute_dims( mpi_size, [n1,], [p1,] )

#==============================================================================
@pytest.mark.parametrize( 'mpi_size', [1,2,5,10] )

def test_partition_1d_general( mpi_size ):

    # ...
    # Should pass, nominal block size is 11
    n1 = 11 * mpi_size + int( mpi_size > 1 )
    p1 = 4

    dims, blocksizes = mpi_compute_dims( mpi_size, [n1,], [p1,] ) 

    assert dims[0] == mpi_size
    assert blocksizes[0] == 11

    # ...
    # Should fail: minimum block size is too large
    n1 = 4 * mpi_size + int( mpi_size > 1 )
    p1 = 5

    with pytest.raises( Exception ):
        dims, blocksizes = mpi_compute_dims( mpi_size, [n1,], [p1,] )

#==============================================================================
def test_partition_3d():

    npts = [64,128,50]
    mpi_size = 100

    # ...
    # Uniform partition, yields small block size along 3rd dimension
    dims, blocksizes = mpi_compute_dims( mpi_size, npts )    

    assert tuple( dims ) == (2, 2, 25)
    assert tuple( blocksizes ) == (32, 64, 2)

    # ...
    # General partition: blocks are not all identical but closer to a cube
    dims, blocksizes = mpi_compute_dims( mpi_size, npts, [3,3,3] )    

    assert tuple( dims ) == (5, 5, 4)
    assert tuple( blocksizes ) == (12, 25, 12)
