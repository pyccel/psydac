# File test_multi_carts_2d.py

#===============================================================================
# TEST MultiCartDecomposition in 2D
#===============================================================================
def run_carts_2d():

    import time
    import numpy as np
    from mpi4py          import MPI
    from psydac.ddm.cart import MultiCartDecomposition, CartDataExchanger

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    N = 10
    # Number of elements
    n1,n2 = 100,100
    p1,p2 = 3,3
    n = [[n1,n2] for i in range(N)]

    # Padding ('thickness' of ghost region)
    p = [[p1,p2] for i in range(N)]
    # Periodicity
    P = [[False, False] for i in range(N)]


    interfaces = {(i,i+1):( ((0,0),(1,-1)) if i%2 ==0 else ((0,0),(-1,1)))  for i in range(N-1)}

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

    carts      = cart._carts
    interfaces = cart._interfaces_carts
    us         = [None]*len(carts)
    syn        = [None]*len(carts)
    dtype      = int

    val = lambda k,i1,i2: k*n1*n2+i1*n1+i2 if (0<=i1<n1 and 0<=i2<n2) else 0

    for i,ci in enumerate(carts):
        if ci:
            s1,s2 = ci.starts
            e1,e2 = ci.ends
            us[i] = np.zeros( ci.shape, dtype=dtype )
            us[i][p1:-p1,p2:-p2] = [[val(i,i1,i2) for i2 in range(s2,e2+1)] for i1 in range(s1,e1+1)]
            synchronizer = CartDataExchanger( ci, us[i].dtype)
            syn[i] = synchronizer
        comm.Barrier()

    uij = {}
    send_types = {}
    recv_types = {}
    displacements = {}
    recv_counts = {}
    indices = {}
    zeros_indices = {}
    for i,j in interfaces:
        if interfaces[i,j] and interfaces[i,j]._intercomm:
            if carts[i] is None:
                shape = interfaces[i,j]._shift_info[interfaces[i,j]._axis]['recv_shape']
                ui    = np.zeros(shape, dtype=dtype)
                uij[i,j] = [ui, us[j]]

            if carts[j] is None:
                shape = interfaces[i,j]._shift_info[interfaces[i,j]._axis]['recv_shape']
                uj    = np.zeros(shape, dtype=dtype)
                uij[i,j] = [us[i], uj]

            intercomm = interfaces[i,j]._intercomm
            send_types[i,j], recv_types[i,j], displacements[i,j], recv_counts[i,j], indices[i,j], zeros_indices[i,j] = interfaces[i,j]._create_buffer_types(interfaces[i,j], dtype)

    for i,ci in enumerate(carts):
        if ci:
            if ci._comm.rank==0:
                print('patch number ', i)
                print('number of elements:', ci.npts)
            ci._comm.Barrier()
            for k in range(ci._comm.size):
                if k==ci._comm.rank:
                    print('rank   :', comm.rank)
                    print('starts :', ci.starts)
                    print('ends   :', ci.ends)
                ci._comm.Barrier()
        comm.Barrier()

    for i,j in interfaces:
        if interfaces[i,j] and interfaces[i,j]._intercomm:
            intercomm = interfaces[i,j]._intercomm
            print('intercomm', comm.rank, intercomm.size)

    for i,ci in enumerate(carts):
        if ci:
            syn[i].update_ghost_regions( us[i] )
            ci._comm.Barrier()
            s1,s2 = ci.starts
            e1,e2 = ci.ends
            uex = [[val(i,i1,i2) for i2 in range(s2-p2,e2+p2+1)] for i1 in range(s1-p1,e1+p1+1)]
            success = (us[i] == uex).all()
            assert success

        comm.Barrier()
    for i,j in interfaces:
        if interfaces[i,j] and interfaces[i,j]._intercomm:
            update_ghost_regions_interface(interfaces[i,j], send_types[i,j], recv_types[i,j], displacements[i,j], recv_counts[i,j], uij[i,j][0], uij[i,j][1], indices[i,j], zeros_indices[i,j])
#            print(uij[i,j][0])
#            print(uij[i,j][1])

def update_ghost_regions_interface(cart, send_type, recv_type, displacements, recv_counts, ui, uj, indices, zeros_indices):
    intercomm = cart._intercomm
    if cart._local_rank_i is not None:
        intercomm.Allgatherv([ui, 1, send_type],[uj, recv_counts, displacements[:-1], recv_type] )
        uj.ravel()[indices] = uj.ravel()[:displacements[-1]]
        uj.ravel()[zeros_indices] = 0
    elif cart._local_rank_j is not None:
        intercomm.Allgatherv([uj, 1, send_type],[ui, recv_counts, displacements[:-1], recv_type] )
        ui.ravel()[indices] = ui.ravel()[:displacements[-1]]
        ui.ravel()[zeros_indices] = 0
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

