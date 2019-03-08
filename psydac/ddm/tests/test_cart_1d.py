# Contents of test_cart_1d.py

#===============================================================================
# TEST CartDecomposition and CartDataExchanger in 1D
#===============================================================================
def run_cart_1d( verbose=False ):

    import numpy as np
    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition, CartDataExchanger

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    # Number of elements
    n1 = 135

    # Padding ('thickness' of ghost region)
    p1 = 3

    # Periodicity
    period1 = True

    #---------------------------------------------------------------------------
    # DOMAIN DECOMPOSITION
    #---------------------------------------------------------------------------

    # Parallel info
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Decomposition of Cartesian domain
    cart = CartDecomposition(
        npts    = [n1+1],
        pads    = [p1],
        periods = [period1],
        reorder = False,
        comm    = comm
    )

    # Local 1D array (extended domain)
    u = np.zeros( cart.shape, dtype=int )

    # Global indices of first and last elements of array
    s1, = cart.starts
    e1, = cart.ends

    # Create object in charge of exchanging data between subdomains
    synchronizer = CartDataExchanger( cart, u.dtype )

    # Print some info
    if verbose:

        if rank == 0:
            print( "" )

        for k in range(size):
            if k == rank:
                print( "RANK = {}".format( rank ) )
                print( "---------" )
                print( ". s1:e1 = {:2d}:{:2d}".format( s1,e1 ) )
                print( "", flush=True )
            comm.Barrier()

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------

    # Fill in true domain with u[i1_loc]=i1_glob
    u[p1:-p1] = [i1 for i1 in range(s1,e1+1)]

    # Update ghost regions
    synchronizer.update_ghost_regions( u )

    #---------------------------------------------------------------------------
    # CHECK RESULTS
    #---------------------------------------------------------------------------

    # Verify that ghost cells contain correct data (note periodic domain!)
    success = all( u[:] == [i1%(n1+1) for i1 in range(s1-p1,e1+p1+1)] )

    # MASTER only: collect information from all processes
    success_global = comm.reduce( success, op=MPI.LAND, root=0 )

    return locals()

#===============================================================================
# RUN TEST WITH PYTEST
#===============================================================================
import pytest

@pytest.mark.parallel
def test_cart_1d():

    namespace = run_cart_1d()

    assert namespace['success']

#===============================================================================
# RUN TEST MANUALLY
#===============================================================================
if __name__=='__main__':

    locals().update( run_cart_1d( verbose=True ) )

    # Print error messages (if any) in orderly fashion
    for k in range(size):
        if k == rank and not success:
            print( "Rank {}: wrong ghost cell data!".format( rank ), flush=True )
        comm.Barrier()

    if rank == 0:
        if success_global:
            print( "PASSED", end='\n\n', flush=True )
        else:
            print( "FAILED", end='\n\n', flush=True )
