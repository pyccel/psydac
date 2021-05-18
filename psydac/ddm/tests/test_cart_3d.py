# File test_cart_3d.py

#===============================================================================
# TEST CartDecomposition and CartDataExchanger in 3D
#===============================================================================
def run_cart_3d( verbose=False ):

    import numpy as np
    from mpi4py       import MPI
    from psydac.ddm.cart import CartDecomposition, CartDataExchanger

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    # Number of elements
    n1 = 135
    n2 =  77
    n3 =  98

    # Padding ('thickness' of ghost region)
    p1 = 3
    p2 = 2
    p3 = 5

    # Periodicity
    period1 = True
    period2 = False
    period3 = True

    #---------------------------------------------------------------------------
    # DOMAIN DECOMPOSITION
    #---------------------------------------------------------------------------

    # Parallel info
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Decomposition of Cartesian domain
    cart = CartDecomposition(
        npts    = [n1+1,n2+1,n3+1],
        pads    = [p1,p2,p3],
        periods = [period1, period2, period3],
        reorder = False,
        comm    = comm,
    )

    # Local 3D array with 3D vector data (extended domain)
    shape = list( cart.shape ) + [3]
    u = np.zeros( shape, dtype=int )

    # Global indices of first and last elements of array
    s1,s2,s3 = cart.starts
    e1,e2,e3 = cart.ends

    # Create object in charge of exchanging data between subdomains
    synchronizer = CartDataExchanger( cart, u.dtype, coeff_shape=[3] )

    # Print some info
    if rank == 0:
        print( "" )

    for k in range(size):
        if k == rank:
            print( "Proc. # {}".format( rank ) )
            print( "---------" )
            print( ". s1:e1 = {:2d}:{:2d}".format( s1,e1 ) )
            print( ". s2:e2 = {:2d}:{:2d}".format( s2,e2 ) )
            print( ". s3:e3 = {:2d}:{:2d}".format( s3,e3 ) )
            print( "", flush=True )
        comm.Barrier()

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------

    # Fill in true domain with u[i1_loc,i2_loc,i3_loc,:]=[i1_glob,i2_glob,i3_glob]
    u[p1:-p1,p2:-p2,p3:-p3,:] = [[[(i1,i2,i3) for i3 in range(s3,e3+1)] \
                                              for i2 in range(s2,e2+1)] \
                                              for i1 in range(s1,e1+1)]

    # Update ghost regions
    synchronizer.update_ghost_regions( u )

    #---------------------------------------------------------------------------
    # CHECK RESULTS
    #---------------------------------------------------------------------------

    # Verify that ghost cells contain correct data (note periodic domain!)
    val = lambda i1,i2,i3: (i1%(n1+1),i2,i3%(n3+1)) if 0<=i2<=n2 else (0,0,0)

    uex = [[[val(i1,i2,i3) for i3 in range(s3-p3,e3+p3+1)] \
                           for i2 in range(s2-p2,e2+p2+1)] \
                           for i1 in range(s1-p1,e1+p1+1)]

    success = (u == uex).all()

    # MASTER only: collect information from all processes
    success_global = comm.reduce( success, op=MPI.LAND, root=0 )

    return locals()

#===============================================================================
# RUN TEST WITH PYTEST
#===============================================================================
import pytest

@pytest.mark.parallel
def test_cart_3d():

    namespace = run_cart_3d()

    assert namespace['success']

#===============================================================================
# RUN TEST MANUALLY
#===============================================================================
if __name__=='__main__':

    locals().update( run_cart_3d( verbose=True ) )

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
