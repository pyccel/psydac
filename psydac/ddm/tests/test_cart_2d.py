#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from psydac.ddm.blocking_data_exchanger    import BlockingCartDataExchanger
from psydac.ddm.nonblocking_data_exchanger import NonBlockingCartDataExchanger

#===============================================================================
# TEST CartDecomposition and CartDataExchanger in 2D
#===============================================================================
def run_cart_2d( data_exchanger_type, verbose=False , nprocs=None, reverse_axis=None):

    import numpy as np
    from mpi4py       import MPI
    from psydac.ddm.cart import DomainDecomposition, CartDecomposition

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    # Number of cells
    nc1 = 135
    nc2 = 77

    # Padding ('thickness' of ghost region)
    p1 = 3
    p2 = 2

    # Periodicity
    period1 = True
    period2 = False

    # Number of Points
    n1 = nc1 + p1*(1-period1)
    n2 = nc2 + p2*(1-period2)
    #---------------------------------------------------------------------------
    # DOMAIN DECOMPOSITION
    #---------------------------------------------------------------------------

    # Parallel info
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    domain_decomposition = DomainDecomposition(ncells=[nc1,nc2], periods=[period1,period2], comm=comm)
    
    npts          = [n1,n2]
    global_starts = [None]*2
    global_ends   = [None]*2
    for axis in range(2):
        es = domain_decomposition.global_element_starts[axis]
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = (ee+1)-1
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    # Decomposition of Cartesian domain
    cart = CartDecomposition(
            domain_decomposition      = domain_decomposition,
            npts          = [n1,n2],
            global_starts = global_starts,
            global_ends   = global_ends,
            pads          = [p1,p2],
            shifts        = [1,1],
    )

    # Local 2D array with 2D vector data (extended domain)
    shape = list( cart.shape ) + [2]
    u = np.zeros( shape, dtype=int )

    # Global indices of first and last elements of array
    s1,s2 = cart.starts
    e1,e2 = cart.ends

    # Create object in charge of exchanging data between subdomains
    synchronizer = data_exchanger_type( cart, u.dtype, coeff_shape=[2] )

    # Print some info
    if rank == 0:
        print( "" )

    for k in range(size):
        if k == rank:
            print( "Proc. # {}".format( rank ) )
            print( "---------" )
            print( ". s1:e1 = {:2d}:{:2d}".format( s1,e1 ) )
            print( ". s2:e2 = {:2d}:{:2d}".format( s2,e2 ) )
            print( "", flush=True )
        comm.Barrier()

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------

    # Fill in true domain with u[i1_loc,i2_loc,:]=[i1_glob,i2_glob]
    u[p1:-p1,p2:-p2,:] = [[(i1,i2) for i2 in range(s2,e2+1)] for i1 in range(s1,e1+1)]


    request = synchronizer.prepare_communications(u)

    # Update ghost regions
    synchronizer.start_update_ghost_regions(  u, request )
    synchronizer.end_update_ghost_regions( u, request )

    #---------------------------------------------------------------------------
    # CHECK RESULTS
    #---------------------------------------------------------------------------

    # Verify that ghost cells contain correct data (note periodic domain!)
    val = lambda i1,i2: (i1%n1,i2) if 0<=i2<n2 else (0,0)
    uex = [[val(i1,i2) for i2 in range(s2-p2,e2+p2+1)] for i1 in range(s1-p1,e1+p1+1)]

    success = (u == uex).all()

    # MASTER only: collect information from all processes
    success_global = comm.reduce( success, op=MPI.LAND, root=0 )

    return locals()

#===============================================================================
# RUN TEST WITH PYTEST
#===============================================================================
import pytest

@pytest.mark.parametrize( 'data_exchanger_type', [BlockingCartDataExchanger, NonBlockingCartDataExchanger] )
@pytest.mark.mpi
def test_cart_2d(data_exchanger_type):

    namespace = run_cart_2d(data_exchanger_type)

    assert namespace['success']

@pytest.mark.mpi
def test_cart_2d_reverse_axis_0():

    namespace = run_cart_2d(BlockingCartDataExchanger, reverse_axis=0)

    assert namespace['success']

@pytest.mark.mpi
def test_cart_2d_reverse_axis_1():

    namespace = run_cart_2d(BlockingCartDataExchanger, reverse_axis=1)

    assert namespace['success']

#===============================================================================
# RUN TEST MANUALLY
#===============================================================================
if __name__=='__main__':

    locals().update( run_cart_2d( BlockingCartDataExchanger, verbose=True ) )

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
