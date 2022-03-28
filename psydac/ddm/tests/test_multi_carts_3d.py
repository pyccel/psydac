# File test_multi_carts_3d.py

#===============================================================================
# TEST MultiCartDecomposition in 3D
#===============================================================================
def run_carts_3d():

    import time
    import numpy as np
    from mpi4py          import MPI
    from psydac.ddm.cart import MultiCartDecomposition, InterfacesCartDecomposition, CartDataExchanger, InterfaceCartDataExchanger

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    # Number of patches
    N = 4

    # Number of elements
    n1,n2,n3 = 10,10,10
    n = [[n1,n2,n3] for i in range(N)]

    # Padding ('thickness' of ghost region)
    p1,p2,p3 = 3,3,3
    p = [[p1,p2,p3] for i in range(N)]

    # Periodicity
    P = [[False, False, False] for i in range(N)]


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

    val = lambda k,i1,i2,i3: k*n1*n2*n3+i1*n1+i2*n2+i3 if (0<=i1<n1 and 0<=i2<n2 and 0<=i3<n3) else 0
    for i,ci in enumerate(carts):
        if ci is not None:
            s1,s2,s3 = ci.starts
            e1,e2,e3 = ci.ends
            m1,m2,m3 = ci.shifts
            us[i] = np.zeros( ci.shape, dtype=dtype )
            us[i][m1*p1:-m1*p1,m2*p2:-m2*p2, m3*p3:-m3*p3] = [[[val(i,i1,i2, i3) for i3 in range(s3, e3+1)] for i2 in range(s2,e2+1)] for i1 in range(s1,e1+1)]
            synchronizer = CartDataExchanger( ci, us[i].dtype)
            syn[i] = synchronizer
        comm.Barrier()

    for i,j in interfaces:
        if interfaces[i,j] is not None and interfaces[i,j].intercomm:
            if carts[i] is None:
                shape = interfaces[i,j].get_communication_infos(interfaces[i,j]._axis)['recv_shape']
                us[i] = np.zeros(shape, dtype=dtype)

            if carts[j] is None:
                shape = interfaces[i,j].get_communication_infos(interfaces[i,j]._axis)['recv_shape']
                us[j] = np.zeros(shape, dtype=dtype)

            syn_interface[i,j] = InterfaceCartDataExchanger(interfaces[i,j], dtype)

    req = {}
    for minus,plus in interfaces:
        if interfaces[minus,plus] and interfaces[minus,plus].intercomm:
            T1 = time.time()
            req[minus, plus] = syn_interface[minus,plus].start_update_ghost_regions(us[minus], us[plus])
            T2 = time.time()
            timmings_interfaces[minus,plus] = T2-T1

    for i,ci in enumerate(carts):
        if ci is not None:
            ci.comm.Barrier()
            T1 = time.time()
            syn[i].update_ghost_regions( us[i] )
            T2 = time.time()
            timmings_interiors_domains[i] = T2-T1
            ci._comm.Barrier()
            s1,s2,s3 = ci.starts
            e1,e2,e3 = ci.ends
            m1,m2,m3 = ci.shifts
            uex = [[[val(i,i1,i2,i3) for i3 in range(s3-m3*p3, e3+m3*p3+1)] for i2 in range(s2-m2*p2,e2+m2*p2+1)] for i1 in range(s1-m1*p1,e1+m1*p1+1)]
            success = (us[i] == uex).all()
            assert success

    for minus,plus in interfaces:
        if interfaces[minus,plus] and interfaces[minus,plus].intercomm:
            T1 = time.time()
            syn_interface[minus,plus].end_update_ghost_regions(req[minus, plus], us[minus], us[plus])
            T2 = time.time()
            timmings_interfaces[minus,plus] = T2-T1
            axis = interfaces[minus,plus].axis
            I = interfaces[minus,plus]

            if carts[minus]:
                ranges = [(0,n) for n,m,p in zip(I.npts_plus, I.shifts_plus, I.pads_plus)]
                coords = I.coords_from_rank_plus[I.boundary_ranks_plus[0]]
                starts = [I.global_starts_plus[d][c] for d,c in enumerate(coords)]
                ends   = [I.global_ends_plus[d][c] for d,c in enumerate(coords)]    
                pads   = I.pads_plus     
                starts[axis] = starts[axis] if I.ext_plus == -1 else ends[axis]-pads[axis]
                ends[axis]   = starts[axis]+pads[axis] if I.ext_plus == -1 else ends[axis]
                ranges[axis] = (starts[axis], ends[axis]+1)
                uex =  [[[val(plus,i1,i2,i3) for i3 in range(*ranges[2])] for i2 in range(*ranges[1])] for i1 in range(*ranges[0])]
                uex = np.pad(uex, [(m*p,m*p) for m,p in zip(carts[minus].shifts, carts[minus].pads)])
                u_ij = us[plus]
            elif carts[plus]:
                ranges = [(0,n) for n,m,p in zip(I.npts_minus, I.shifts_minus, I.pads_minus)]
                coords = I.coords_from_rank_minus[I.boundary_ranks_minus[0]]
                starts = [I.global_starts_minus[d][c] for d,c in enumerate(coords)]
                ends   = [I.global_ends_minus[d][c] for d,c in enumerate(coords)]
                pads   = I.pads_minus
                starts[axis] = starts[axis] if I.ext_minus == -1 else ends[axis]-pads[axis]
                ends[axis]   = starts[axis]+pads[axis] if I.ext_minus == -1 else ends[axis]
                ranges[axis] = (starts[axis], ends[axis]+1)
                uex =  [[[val(minus,i1,i2,i3) for i3 in range(*ranges[2])] for i2 in range(*ranges[1])] for i1 in range(*ranges[0])]
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

    run_carts_3d()

