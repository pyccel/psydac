# File test_cart_2d.py

#===============================================================================
# TEST Cart in 2D
#===============================================================================
def run_cart_2d( verbose=False ):

    import numpy as np
    from mpi4py       import MPI
    from spl.ddm.cart import Cart

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    # Number of elements
    n1 = 135
    n2 = 77

    # Padding ('thickness' of ghost region)
    p1 = 3
    p2 = 2

    # Periodicity
    period1 = True
    period2 = False

    #---------------------------------------------------------------------------
    # DOMAIN DECOMPOSITION
    #---------------------------------------------------------------------------

    # Parallel info
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Decomposition of Cartesian domain
    cart = Cart(
        npts    = [n1+1,n2+1],
        pads    = [p1,p2],
        periods = [period1, period2],
        reorder = False,
        comm    = comm,
    )

    # Local 2D array with 2D vector data (extended domain)
    shape = list( cart.shape ) + [2]
    u = np.zeros( shape, dtype='i' ) # NOTE: 32-bit C INTEGER!

    # Global indices of first and last elements of array
    s1,s2 = cart.starts
    e1,e2 = cart.ends

    # Create MPI subarray datatypes for accessing non-contiguous data
    send_types = {}
    recv_types = {}
    for direction in range(2):
        for disp in [-1,1]:
            info = cart.get_shift_info( direction, disp )

            buf_shape   = list( info[ 'buf_shape' ] ) + [2]
            send_starts = list( info['send_starts'] ) + [0]
            recv_starts = list( info['recv_starts'] ) + [0]

            send_types[direction,disp] = MPI.INT.Create_subarray(
                sizes    = u.shape,
                subsizes = buf_shape,
                starts   = send_starts,
            ).Commit()

            recv_types[direction,disp] = MPI.INT.Create_subarray(
                sizes    = u.shape,
                subsizes = buf_shape,
                starts   = recv_starts,
            ).Commit()

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

    # Choose non-negative invertible function tag(disp) >= 0
    # NOTE: different values of disp must return different tags!
    tag = lambda disp: 42+disp

    # Cycle over dimensions
    for direction in range(2):

        # Requests' handles
        requests = []

        # Start receiving data (MPI_IRECV)
        for disp in [-1,1]:
            info     = cart.get_shift_info( direction, disp )
            recv_buf = (u, 1, recv_types[direction,disp])
            recv_req = cart.comm_cart.Irecv( recv_buf, info['rank_source'], tag(disp) )
            requests.append( recv_req )

        # Start sending data (MPI_ISEND)
        for disp in [-1,1]:
            info     = cart.get_shift_info( direction, disp )
            send_buf = (u, 1, send_types[direction,disp])
            send_req = cart.comm_cart.Isend( send_buf, info['rank_dest'], tag(disp) )
            requests.append( send_req )

        # Wait for end of data exchange (MPI_WAITALL)
        MPI.Request.Waitall( requests )

    #---------------------------------------------------------------------------
    # CHECK RESULTS
    #---------------------------------------------------------------------------

    # Verify that ghost cells contain correct data (note periodic domain!)
    val = lambda i1,i2: (i1%(n1+1),i2) if 0<=i2<=n2 else (0,0)
    uex = [[val(i1,i2) for i2 in range(s2-p2,e2+p2+1)] for i1 in range(s1-p1,e1+p1+1)]

    success = (u == uex).all()

    # MASTER only: collect information from all processes
    success_global = comm.reduce( success, op=MPI.LAND, root=0 )

    return locals()

#===============================================================================
# RUN TEST WITH PYTEST
#===============================================================================
def test_cart_2d():

    namespace = run_cart_2d()

    assert namespace['success']

#===============================================================================
# RUN TEST MANUALLY
#===============================================================================
if __name__=='__main__':

    locals().update( run_cart_2d( verbose=True ) )

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
