import numpy    as np
import numpy.ma as ma

from sympy.ntheory import factorint

__all__ = ['compute_dims', 'partition_procs_per_patch']

#==============================================================================
def partition_procs_per_patch(npts, size):
    npts       = [np.product(nc) for nc in npts]
    percentage = [nc/sum(npts) for nc in npts]
    sizes      = np.array([int(p*size) for p in percentage])
    diff       = [p*size-s for s,p in zip(sizes, percentage)]
    indices    = np.argsort(diff)[::-1]
    rm         = size-sum(sizes)

    sizes[indices[:rm]] +=1
    assert sum(sizes) == size

    #...
    start  = 0
    ranges = []
    for s in sizes:
        ranges.append([start, start+s-1])
        start += s

    assert start == size

    ranges = np.array(ranges)
    ranks  = [i[0] for i in ranges[indices[:rm]]]

    if len(ranks) == 0:
        if any(s==0 for s in sizes):
            raise ValueError("Cannot compute sizes with given input values!")

    k = 0
    for i,s in enumerate(sizes):
        if s>0:continue
        sizes[i]  = 1
        ranges[i] = [ranks[k], ranks[k]]
        k         = (k+1)%size

    return sizes, ranges

#==============================================================================
def compute_dims( nnodes, gridsizes, min_blocksizes=None, mpi=None ):
    """
    With the aim of distributing a multi-dimensional array on a Cartesian
    topology, compute the number of processes along each dimension.

    Whenever possible, the number of processes is chosen so that the array is
    decomposed into identical blocks.

    Parameters
    ----------
    nnodes : int
        Number of processes in the Cartesian topology.

    gridsizes : list of int
        Number of array elements along each dimension.

    min_blocksizes : list of int
        Minimum acceptable size of a block along each dimension. 

    Returns
    -------
    dims : list of int
        Number of processes along each dimension of the Cartesian topology. 

    blocksizes : list of int
        Nominal block size along each dimension.

    """
    assert nnodes > 0
    assert all( s > 0 for s in gridsizes )
    assert np.prod( gridsizes ) >= nnodes

    if (min_blocksizes is not None):
        assert len( min_blocksizes ) == len( gridsizes )
        assert all( m > 0 for m in gridsizes )
        assert all( s >= m for s,m in zip( gridsizes, min_blocksizes ) )

    # Determine whether uniform decomposition is possible
    uniform = (np.prod( gridsizes ) % nnodes == 0)

    # Compute dimensions of MPI Cartesian topology with most appropriate algorithm
    if uniform:
        dims, blocksizes = compute_dims_uniform( nnodes, gridsizes )
    else:
        dims, blocksizes = compute_dims_general( nnodes, gridsizes )

    # If a minimum block size is given, verify that condition is met

    if min_blocksizes is not None:
        too_small = any( [s < m for (s,m) in zip( blocksizes, min_blocksizes )] )

        # If uniform decomposition yields blocks too small, fall back to general algorithm
        if uniform and too_small:
            dims, blocksizes = compute_dims_general( nnodes, gridsizes )
            too_small = any( [s < m for (s,m) in zip( blocksizes, min_blocksizes )] )

        # If general decomposition yields blocks too small, raise error
        if too_small:
            raise ValueError("Cannot compute dimensions with given input values!")

    return dims, blocksizes

#==============================================================================
def compute_dims_general( mpi_size, npts ):

    nprocs = [1]*len( npts )
    shape  = [n for n in npts]

    f = factorint( mpi_size, multiple=True )
    f.sort( reverse=True )

    for a in f:

        i = np.argmax( shape )
        max_shape = shape[i]

        if shape.count( max_shape ) > 1:
            i = ma.array( nprocs, mask=np.not_equal( shape, max_shape ) ).argmin()

        nprocs[i]  *= a
        shape [i] //= a

    return nprocs, shape

#==============================================================================
def compute_dims_uniform( mpi_size, npts ):

    nprocs = [1]*len( npts )

    mpi_factors   = factorint( int(mpi_size) )
    npts_factors  = [factorint( int(n) ) for n in npts]

    nprocs = [1 for n in npts]

    for a,power in mpi_factors.items():

        exponents = [f.get( a, 0 ) for f in npts_factors]

        for k in range( power ):

            i = np.argmax( exponents )
            max_exp = exponents[i]

            if exponents.count( max_exp ) > 1:
                i = ma.array( nprocs, mask=np.not_equal( exponents, max_exp ) ).argmin()

            nprocs   [i] *= a
            exponents[i] -= 1

            npts_factors[i][a] -= 1

    shape = [np.prod( [key**val for key,val in f.items()] ) for f in npts_factors]

    return nprocs, shape
