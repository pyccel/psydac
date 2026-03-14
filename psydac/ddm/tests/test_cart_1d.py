#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from psydac.ddm.blocking_data_exchanger    import BlockingCartDataExchanger
from psydac.ddm.nonblocking_data_exchanger import NonBlockingCartDataExchanger

#===============================================================================
# TEST CartDecomposition and CartDataExchanger in 1D
#===============================================================================
def run_cart_1d( data_exchanger_type, verbose=False ):

    import numpy as np
    from mpi4py       import MPI
    from psydac.ddm.cart import DomainDecomposition, CartDecomposition

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    # Number of cells
    nc1 = 135

    # Padding ('thickness' of ghost region)
    p1 = 3

    # Periodicity
    period1 = True

    # Number of Points
    n1 = nc1 + p1*(1-period1)
    #---------------------------------------------------------------------------
    # DOMAIN DECOMPOSITION
    #---------------------------------------------------------------------------

    # Parallel info
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    domain_decomposition = DomainDecomposition(ncells=[nc1], periods=[period1], comm=comm)

    es = domain_decomposition.global_element_starts[0]
    ee = domain_decomposition.global_element_ends  [0]

    global_ends        = [ee]
    global_ends[0][-1] = n1-1
    global_starts      = [np.array([0] + (global_ends[0][:-1]+1).tolist())]

    # Decomposition of Cartesian domain
    cart = CartDecomposition(
            domain_decomposition = domain_decomposition,
            npts          = [n1],
            global_starts = global_starts,
            global_ends   = global_ends,
            pads          = [p1],
            shifts        = [1],
    )

    # Local 1D array (extended domain)
    u = np.zeros( cart.shape, dtype=int )

    # Global indices of first and last elements of array
    s1, = cart.starts
    e1, = cart.ends

    # Create object in charge of exchanging data between subdomains
    synchronizer = data_exchanger_type( cart, u.dtype )

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

    request = synchronizer.prepare_communications(u)

    # Update ghost regions
    synchronizer.start_update_ghost_regions(  u, request )
    synchronizer.end_update_ghost_regions(  u, request )

    #---------------------------------------------------------------------------
    # CHECK RESULTS
    #---------------------------------------------------------------------------
    # Verify that ghost cells contain correct data (note periodic domain!)
    success = all( u[:] == [i1%n1 for i1 in range(s1-p1,e1+p1+1)] )

    # MASTER only: collect information from all processes
    success_global = comm.reduce( success, op=MPI.LAND, root=0 )

    return locals()

#===============================================================================
# RUN TEST WITH PYTEST
#===============================================================================
import pytest

@pytest.mark.parametrize( 'data_exchanger_type', [BlockingCartDataExchanger, NonBlockingCartDataExchanger] )
@pytest.mark.mpi
def test_cart_1d( data_exchanger_type ):

    namespace = run_cart_1d( data_exchanger_type )

    assert namespace['success']

#===============================================================================
# RUN TEST MANUALLY
#===============================================================================
if __name__=='__main__':

    locals().update( run_cart_1d( BlockingCartDataExchanger, verbose=True ) )

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
