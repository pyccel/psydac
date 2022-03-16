# File test_multi_carts_2d.py

#===============================================================================
# TEST MultiCartDecomposition in 2D
#===============================================================================
def run_carts_2d():

    import numpy as np
    from mpi4py       import MPI
    from psydac.ddm.cart import MultiCartDecomposition

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    # Number of elements
    n1 = [100,100]
    n2 = [100,100]
    n3 = [100,100]

    # Padding ('thickness' of ghost region)
    p1 = [3,3]
    p2 = [3,3]
    p3 = [3,3]
    # Periodicity
    period1 = [False, False]
    period2 = [False, False]
    period3 = [False, False]

    interfaces = {}
    interfaces[0,1] = True
    interfaces[1,2] = True
    interfaces[2,0] = True
    #---------------------------------------------------------------------------
    # DOMAIN DECOMPOSITION
    #---------------------------------------------------------------------------

    # Parallel info
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Decomposition of Cartesian domain
    cart = MultiCartDecomposition(
        npts       = [n1,n2,n3],
        pads       = [p1,p2,p3],
        periods    = [period1, period2, period3],
        reorder    = False,
        interfaces = interfaces,
        comm    = comm)

    if rank == 0:
        print(cart._size)
        print(cart._rank_ranges)
        print(cart._sizes)
        
#    for k in range(size):
#        if k == rank and not success:
#            print( "Rank {}: wrong ghost cell data!".format( rank ), flush=True )
#        comm.Barrier()

#===============================================================================
# RUN TEST MANUALLY
#===============================================================================
if __name__=='__main__':

    run_carts_2d()

#    # Print error messages (if any) in orderly fashion
#    for k in range(size):
#        if k == rank and not success:
#            print( "Rank {}: wrong ghost cell data!".format( rank ), flush=True )
#        comm.Barrier()

