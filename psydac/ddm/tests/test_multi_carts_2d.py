# File test_multi_carts_3d.py

#===============================================================================
# TEST MultiCartDecomposition in 2D
#===============================================================================
def get_minus_starts_ends(plus_starts, plus_ends, minus_npts, plus_npts, minus_axis, plus_axis, minus_ext, plus_ext, minus_pads, plus_pads, diff):
    starts = [max(0,s-p) for s,p in zip(plus_starts, minus_pads)]
    ends   = [min(n,e+p) for e,n,p in zip(plus_ends, minus_npts, minus_pads)]
    starts[minus_axis] = 0 if minus_ext == -1 else ends[minus_axis]-minus_pads[minus_axis]
    ends[minus_axis]   = ends[minus_axis] if minus_ext == 1 else minus_pads[minus_axis]
    return starts, ends

def get_plus_starts_ends(minus_starts, minus_ends, minus_npts, plus_npts, minus_axis, plus_axis, minus_ext, plus_ext, minus_pads, plus_pads, diff):
    starts = [max(0,s-p) for s,p in zip(minus_starts, plus_pads)]
    ends   = [min(n,e+p) for e,n,p in zip(minus_ends, plus_npts, plus_pads)]
    starts[plus_axis] = 0 if plus_ext == -1 else ends[plus_axis]-plus_pads[plus_axis]
    ends[plus_axis]   = ends[plus_axis] if plus_ext == 1 else plus_pads[plus_axis]
    return starts, ends
def run_carts_2d():

    import time
    import numpy as np
    from mpi4py          import MPI
    from psydac.ddm.cart import MultiCartDecomposition, InterfacesCartDecomposition, CartDataExchanger, InterfaceCartDataExchanger

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    # Number of patches
    N = 2

    # Number of elements
    n1,n2 = 8,8
    n = [[n1,n2] for i in range(N)]

    # Padding ('thickness' of ghost region)
    p1,p2 = 2,2
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

    cart = MultiCartDecomposition(
        npts       = n,
        pads       = p,
        periods    = P,
        reorder    = False,
        comm    = comm)

    interface_carts = InterfacesCartDecomposition(cart, interfaces)

    timmings_interiors_domains = [0]*N
    timmings_interfaces = {(i,j):0 for i,j in interfaces}

    carts      = cart.carts
    interfaces = interface_carts.carts
    us         = [None]*len(carts)
    syn        = [None]*len(carts)
    syn_interface = {}
    dtype         = int

    for i,j in interfaces:
        if interface_carts.carts[i,j].is_comm_null:
            continue
        interface_carts.carts[i,j].set_communication_info_p2p(get_minus_starts_ends, get_plus_starts_ends)


    val = lambda k,i1,i2: k*n1*n2+i1*n1+i2 if (0<=i1<n1 and 0<=i2<n2) else 0
    for i,ci in enumerate(carts):
        if ci.comm != MPI.COMM_NULL:
            s1,s2 = ci.starts
            e1,e2 = ci.ends
            m1,m2 = ci.shifts
            us[i] = np.zeros( ci.shape, dtype=dtype )
            us[i][m1*p1:-m1*p1,m2*p2:-m2*p2] = [[val(i,i1,i2)for i2 in range(s2,e2+1)] for i1 in range(s1,e1+1)]
            synchronizer = CartDataExchanger( ci, us[i].dtype)
            syn[i] = synchronizer
            print(i, ci.starts, ci.ends, comm.rank, flush=True)
            print(us[i], flush=True)
        comm.Barrier()

    for i,j in interfaces:
        if interfaces[i,j].intercomm != MPI.COMM_NULL:
            if carts[i].comm == MPI.COMM_NULL:
                shape = interfaces[i,j].get_communication_infos_p2p(interfaces[i,j]._axis)['gbuf_recv_shape'][0]
                us[i] = np.zeros(shape, dtype=dtype)

            if carts[j].comm == MPI.COMM_NULL:
                shape = interfaces[i,j].get_communication_infos_p2p(interfaces[i,j]._axis)['gbuf_recv_shape'][0]
                us[j] = np.zeros(shape, dtype=dtype)

            syn_interface[i,j] = InterfaceCartDataExchanger(interfaces[i,j], dtype)


    req = {}
    for minus,plus in interfaces:
        if interfaces[minus,plus].intercomm != MPI.COMM_NULL:
            T1 = time.time()
            syn_interface[minus,plus].update_ghost_regions(us[minus], us[plus])
            T2 = time.time()
            timmings_interfaces[minus,plus] = T2-T1
     
    comm.Barrier()
    print("#####", flush=True)
    comm.Barrier()
#    for minus,plus in interfaces:
#        if interfaces[minus,plus].intercomm == MPI.COMM_NULL:
#            continue
#        if carts[minus].is_comm_null:
#            print(us[minus], flush=True)
#        else:
#            print(us[plus], flush=True)

    raise SystemExit()
    for i,ci in enumerate(carts):
        if ci.comm != MPI.COMM_NULL:
            T1 = time.time()
            syn[i].update_ghost_regions( us[i] )
            T2 = time.time()
            timmings_interiors_domains[i] = T2-T1
            ci._comm.Barrier()
            s1,s2 = ci.starts
            e1,e2 = ci.ends
            m1,m2 = ci.shifts
            uex = [[val(i,i1,i2) for i2 in range(s2-m2*p2,e2+m2*p2+1)] for i1 in range(s1-m1*p1,e1+m1*p1+1)]
            success = (us[i] == uex).all()
            assert success

    for minus,plus in interfaces:
        if interfaces[minus,plus].intercomm != MPI.COMM_NULL:
            T1 = time.time()
            syn_interface[minus,plus].end_update_ghost_regions(req[minus, plus], us[minus], us[plus])
            T2 = time.time()
            timmings_interfaces[minus,plus] = timmings_interfaces[minus,plus] + T2-T1
            axis = interfaces[minus,plus].axis
            I = interfaces[minus,plus]

            if carts[minus].comm != MPI.COMM_NULL:
                ranges = [(0,n) for n,m,p in zip(I.npts_plus, I.shifts_plus, I.pads_plus)]
                coords = I.coords_from_rank_plus[I.boundary_ranks_plus[0]]
                starts = [I.global_starts_plus[d][c] for d,c in enumerate(coords)]
                ends   = [I.global_ends_plus[d][c] for d,c in enumerate(coords)]    
                pads   = I.pads_plus     
                starts[axis] = starts[axis] if I.ext_plus == -1 else ends[axis]-pads[axis]
                ends[axis]   = starts[axis]+pads[axis] if I.ext_plus == -1 else ends[axis]
                ranges[axis] = (starts[axis], ends[axis]+1)
                uex =  [[val(plus,i1,i2)for i2 in range(*ranges[1])] for i1 in range(*ranges[0])]
                uex = np.pad(uex, [(m*p,m*p) for m,p in zip(carts[minus].shifts, carts[minus].pads)])
                u_ij = us[plus]
            elif carts[plus].comm != MPI.COMM_NULL:
                ranges = [(0,n) for n,m,p in zip(I.npts_minus, I.shifts_minus, I.pads_minus)]
                coords = I.coords_from_rank_minus[I.boundary_ranks_minus[0]]
                starts = [I.global_starts_minus[d][c] for d,c in enumerate(coords)]
                ends   = [I.global_ends_minus[d][c] for d,c in enumerate(coords)]
                pads   = I.pads_minus
                starts[axis] = starts[axis] if I.ext_minus == -1 else ends[axis]-pads[axis]
                ends[axis]   = starts[axis]+pads[axis] if I.ext_minus == -1 else ends[axis]
                ranges[axis] = (starts[axis], ends[axis]+1)
                uex =  [[val(minus,i1,i2)for i2 in range(*ranges[1])] for i1 in range(*ranges[0])]
                uex = np.pad(uex, [(m*p,m*p) for m,p in zip(carts[plus].shifts, carts[plus].pads)])
                u_ij = us[minus]
            success = (u_ij == uex).all()
            assert success

    for i,ci in enumerate(carts):
        timmings_interiors_domains[i] = comm.allreduce(timmings_interiors_domains[i], op=MPI.MAX)

    for i,j in timmings_interfaces:
        timmings_interfaces[i,j] = comm.allreduce(timmings_interfaces[i,j], op=MPI.MAX)

    if comm.rank==0:
        print(max(timmings_interiors_domains), flush=True)
        print(max(timmings_interfaces.values()), flush=True)
#===============================================================================
# RUN TEST MANUALLY
#===============================================================================
if __name__=='__main__':

    run_carts_2d()

