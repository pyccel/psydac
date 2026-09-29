import cunumpy as xp
import numpy as np
import numpy.ma as ma

__all__ = ('compute_dims', 'partition_procs_per_patch')

#==============================================================================
def factorint(n, multiple=False):
    """
    Prime factorization of an integer by trial division.

    Drop-in replacement for the subset of ``sympy.ntheory.factorint`` used here.
    Importing sympy takes ~1 s and is not needed for the small integers
    (process counts, number of grid points) that are factorized in this module.

    Parameters
    ----------
    n : int
        Integer to factorize.

    multiple : bool
        If False (default), return a dict {prime: multiplicity}.
        If True, return the list of primes in ascending order, repeated
        according to their multiplicity.
    """
    n = int(n)
    factors = {}

    # same conventions as sympy for non-positive input
    if n == 0:
        factors[0] = 1
    else:
        if n < 0:
            factors[-1] = 1
            n = -n
        p = 2
        while p * p <= n:
            while n % p == 0:
                factors[p] = factors.get(p, 0) + 1
                n //= p
            p += 1 if p == 2 else 2
        if n > 1:
            factors[n] = factors.get(n, 0) + 1

    if multiple:
        return [p for p in sorted(factors) for _ in range(factors[p])]
    return factors

#==============================================================================
def partition_procs_per_patch(npts, size):
    """
    Compute the number of processes in each patch and assign to it an ascending range of processes.
    The processes are distributed porportionally to the patch grid size.

    Parameters
    ----------
    npts : list
        Number of points along each dimension for each patch.

    size : int
        Number of processes.

    Returns
    -------
    sizes : list of int
        Number of processes in each patch.

    ranges: list of list of int
        The assigned ascending range of processes for each patch, 
        the range is represented by a list of ints of size 2 [k1,k2],
        such that k1<=k2.

    """
    npts       = [xp.prod(nc) for nc in npts]
    percentage = [nc / sum(npts) for nc in npts]
    sizes      = xp.array([int(p*size) for p in percentage])
    diff       = [p * size - s for s, p in zip(sizes, percentage)]
    indices    = xp.argsort(diff)[::-1]
    rm         = size - sum(sizes)

    sizes[indices[:rm]] +=1
    assert sum(sizes) == size

    #...
    start  = 0
    ranges = []
    for s in sizes:
        ranges.append([start, start+s-1])
        start += s

    assert start == size

    ranges = xp.array(ranges)
    ranks  = [i[0] for i in ranges[indices[:rm]]]

    if len(ranks) == 0:
        if any(s==0 for s in sizes):
            raise ValueError("Cannot compute sizes with given input values!")

    k = 0
    for i,s in enumerate(sizes):
        if s > 0:
            continue
        sizes[i]  = 1
        ranges[i] = [ranks[k], ranks[k]]
        k         = (k+1) % size

    return sizes, ranges

#==============================================================================
def compute_dims( nnodes, gridsizes, min_blocksizes=None, mpi=None, try_uniform=False, mpi_dims_mask=None ):
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

    try_uniform: bool
        try to decompose the array uniformly.
        
    mpi_dims_mask: list of bool
        True if the dimension is to be used in the domain decomposition (=default for each dimension). 
        If dim_mask[i]=False, the domain decomposition yields blocksizes[i]=gridsizes[i] along the i-th dimension.

    Returns
    -------
    dims : list of int
        Number of processes along each dimension of the Cartesian topology. 

    blocksizes : list of int
        Nominal block size along each dimension.

    """
    assert nnodes > 0
    assert all( s > 0 for s in gridsizes )
    assert xp.prod( xp.asarray(gridsizes) ) >= nnodes

    if (min_blocksizes is not None):
        assert len( min_blocksizes ) == len( gridsizes )
        assert all( m > 0 for m in gridsizes )
        assert all( s >= m for s,m in zip( gridsizes, min_blocksizes ) )

    # Determine whether uniform decomposition is possible
    uniform = (xp.prod( xp.asarray(gridsizes) ) % nnodes == 0)

    # Compute dimensions of MPI Cartesian topology with most appropriate algorithm
    if try_uniform and uniform:
        dims, blocksizes = compute_dims_uniform( nnodes, gridsizes )
    else:
        dims, blocksizes = compute_dims_general( nnodes, gridsizes, mpi_dims_mask=mpi_dims_mask )

    # If a minimum block size is given, verify that condition is met

    if min_blocksizes is not None:
        too_small = any( [s < m for (s,m) in zip( blocksizes, min_blocksizes )] )

        # If uniform decomposition yields blocks too small, fall back to general algorithm
        if uniform and too_small:
            dims, blocksizes = compute_dims_general( nnodes, gridsizes )
            too_small = any( [s < m for (s,m) in zip( blocksizes, min_blocksizes )] )

        # If general decomposition yields blocks too small, raise error
        if too_small:
            raise ValueError("Cannot compute dimensions with the minimum acceptable block sizes {}".format(tuple(min_blocksizes)))

    return dims, blocksizes

#==============================================================================
def compute_dims_general( mpi_size, npts, mpi_dims_mask=None ):

    if mpi_dims_mask is None:
        mpi_dims_mask = [True] * len(npts)
    else:
        assert len(mpi_dims_mask) == len(npts), "mpi_dims_mask must have one entry for each dimension."
        assert all(isinstance(m, bool) for m in mpi_dims_mask), "mpi_dims_mask must only contain True/False values."
        assert any(mpi_dims_mask), "mpi_dims_mask must contain at least one True value."

    nprocs = [1]*len( npts )
    
    shape = []
    for n, use_dim in zip(npts, mpi_dims_mask):
        if use_dim:
            shape += [n]
        else:
            shape += [-1]

    f = factorint( mpi_size, multiple=True )
    f.sort( reverse=True )

    for a in f:

        i = xp.argmax( shape )
        max_shape = shape[i]

        if shape.count( max_shape ) > 1:
            i = ma.array( nprocs, mask=xp.not_equal( shape, max_shape ) ).argmin()

        nprocs[i]  *= a
        shape [i] //= a
        
    for i, use_dim in enumerate(mpi_dims_mask):
        if not use_dim:
            shape[i] = npts[i]

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

            i = xp.argmax( exponents )
            max_exp = exponents[i]

            if exponents.count( max_exp ) > 1:
                i = ma.array( nprocs, mask=xp.not_equal( exponents, max_exp ) ).argmin()

            nprocs   [i] *= a
            exponents[i] -= 1

            npts_factors[i][a] -= 1

    shape = [xp.prod( [key**val for key,val in f.items()] ) for f in npts_factors]

    return nprocs, shape
