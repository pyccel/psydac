# File test_multi_carts_2d.py

#===============================================================================
# TEST MultiCartDecomposition in 2D
#===============================================================================
def run_carts_2d():

    import time
    import numpy as np
    from mpi4py          import MPI
    from psydac.ddm.cart import MultiCartDecomposition

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    N = 1000
    # Number of elements
    n = [[200,200] for i in range(N)]

    # Padding ('thickness' of ghost region)
    p = [[3,3] for i in range(N)]
    # Periodicity
    P = [[False, False] for i in range(N)]


    interfaces = {(i,i+1):True for i in range(N-1)}
    #---------------------------------------------------------------------------
    # DOMAIN DECOMPOSITION
    #---------------------------------------------------------------------------

    # Parallel info
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    t1 = time.time()
    # Decomposition of Cartesian domain
    cart = MultiCartDecomposition(
        npts       = n,
        pads       = p,
        periods    = P,
        reorder    = False,
        interfaces = interfaces,
        comm    = comm)
    t2 = time.time()

    T = comm.reduce(t2-t1, root=0, op=MPI.MAX)
    if rank == 0:
        print(cart._size)
        print(cart._rank_ranges)
        print(cart._sizes)
        print("time : ", T)
        
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

