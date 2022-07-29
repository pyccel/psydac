# File test_multicart_2d.py

#===============================================================================
# TEST MultiCartDecomposition in 2D
#===============================================================================

#------------------------------------------------------------------------------
def get_minus_starts_ends(plus_starts, plus_ends, minus_npts, plus_npts, minus_axis, plus_axis,
                          minus_ext, plus_ext, minus_pads, plus_pads, minus_shifts, plus_shifts,
                          diff):
    """
    Compute the coefficients needed by the minus patch in a given interface.
    """
    starts = [max(0,s-m*p) for s,m,p in zip(plus_starts, minus_shifts, minus_pads)]
    ends   = [min(n,e+m*p) for e,n,m,p in zip(plus_ends, minus_npts, minus_shifts, minus_pads)]
    starts[minus_axis] = 0 if minus_ext == -1 else ends[minus_axis]-minus_pads[minus_axis]
    ends[minus_axis]   = ends[minus_axis] if minus_ext == 1 else minus_pads[minus_axis]
    return starts, ends

#------------------------------------------------------------------------------
def get_plus_starts_ends(minus_starts, minus_ends, minus_npts, plus_npts, minus_axis, plus_axis,
                         minus_ext, plus_ext, minus_pads, plus_pads, minus_shifts, plus_shifts,
                         diff):
    """
    Compute the coefficients needed by the plus patch in a given interface.
    """
    starts = [max(0,s-m*p) for s,m,p in zip(minus_starts, plus_shifts, plus_pads)]
    ends   = [min(n,e+m*p) for e,n,m,p in zip(minus_ends, plus_npts, plus_shifts, plus_pads)]
    starts[plus_axis] = 0 if plus_ext == -1 else ends[plus_axis]-plus_pads[plus_axis]
    ends[plus_axis]   = ends[plus_axis] if plus_ext == 1 else plus_pads[plus_axis]
    return starts, ends


#===============================================================================
# TEST MultiPatchDomainDecomposition and CartDataExchanger in 2D
#===============================================================================
def run_carts_2d():
    import numpy as np
    from mpi4py          import MPI
    from psydac.ddm.cart import MultiPatchDomainDecomposition, CartDecomposition, InterfacesCartDecomposition, CartDataExchanger, InterfaceCartDataExchanger

    #---------------------------------------------------------------------------
    # INPUT PARAMETERS
    #---------------------------------------------------------------------------

    # Number of patches
    N = 2

    # Number of cells
    nc1,nc2 = 16,16
    nc = [[nc1,nc2] for i in range(N)]

    # Padding ('thickness' of ghost region)
    p1,p2 = 2,2
    p = [[p1,p2] for i in range(N)]

    # Periodicity
    P = [[False, False] for i in range(N)]

    connectivity = {(i,i+1):( ((0,1),(0,-1)) if i%2 ==0 else ((0,-1),(0,1)))  for i in range(N-1)}

    #---------------------------------------------------------------------------
    # DOMAIN DECOMPOSITION
    #---------------------------------------------------------------------------

    # Parallel info
    comm = MPI.COMM_WORLD

    domain_h =  MultiPatchDomainDecomposition(nc, P, comm=comm)

    # Number of elements
    n = [[ncij + pij*(1-periodij) for ncij,pij,periodij in zip(nci,pi,periodi)] for nci,pi,periodi in zip(nc,p,P)]

    carts = []
    for i in range(N):
        global_starts = [None]*2
        global_ends   = [None]*2
        for axis in range(2):
            es = domain_h.domains[i].global_element_starts[axis]
            ee = domain_h.domains[i].global_element_ends  [axis]

            global_ends  [axis]     = (ee+1)-1
            global_ends  [axis][-1] = n[i][axis]-1
            global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

        carts.append(CartDecomposition(
                        domain_h      = domain_h.domains[i],
                        npts          = n[i],
                        global_starts = global_starts,
                        global_ends   = global_ends,
                        pads          = p[i],
                        shifts        = [1,1]))
    carts = tuple(carts)

    interfaces_cart = InterfacesCartDecomposition(domain_h, carts, connectivity)

    us         = [None]*len(carts)
    syn        = [None]*len(carts)
    syn_interface = {}
    dtype         = int

    for i,j in connectivity:
        if (i,j) in interfaces_cart.carts and not interfaces_cart.carts[i,j].is_comm_null:
            interfaces_cart.carts[i,j].set_interface_communication_infos(get_minus_starts_ends, get_plus_starts_ends)

    val = lambda k,i1,i2: k*n[k][0]*n[k][1]+i1*n[k][0]+i2 if (0<=i1<n[k][0] and 0<=i2<n[k][1]) else 0
    for i,ci in enumerate(carts):
        if not ci.is_comm_null:
            s1,s2 = ci.starts
            e1,e2 = ci.ends
            m1,m2 = ci.shifts
            us[i] = np.zeros( ci.shape, dtype=dtype )
            us[i][m1*p1:-m1*p1,m2*p2:-m2*p2] = [[val(i,i1,i2)for i2 in range(s2,e2+1)] for i1 in range(s1,e1+1)]
            synchronizer = CartDataExchanger( ci, us[i].dtype)
            syn[i] = synchronizer

    for i,j in connectivity:
        if not interfaces_cart.carts[i,j].is_comm_null:
            if carts[i].is_comm_null:
                shape = interfaces_cart.carts[i,j].get_communication_infos(interfaces_cart.carts[i,j]._axis)['gbuf_recv_shape'][0]
                us[i] = np.zeros(shape, dtype=dtype)

            if carts[j].is_comm_null:
                shape = interfaces_cart.carts[i,j].get_communication_infos(interfaces_cart.carts[i,j]._axis)['gbuf_recv_shape'][0]
                us[j] = np.zeros(shape, dtype=dtype)

            syn_interface[i,j] = InterfaceCartDataExchanger(interfaces_cart.carts[i,j], dtype)

    for minus,plus in connectivity:
        if not interfaces_cart.carts[minus,plus].is_comm_null:
            syn_interface[minus,plus].update_ghost_regions(us[minus], us[plus])

    for i,ci in enumerate(carts):
        if not ci.is_comm_null:
            syn[i].update_ghost_regions( us[i] )

    for i,ci in enumerate(carts):
        if not ci.is_comm_null:
            s1,s2 = ci.starts
            e1,e2 = ci.ends
            m1,m2 = ci.shifts
            uex = [[val(i,i1,i2) for i2 in range(s2-m2*p2,e2+m2*p2+1)] for i1 in range(s1-m1*p1,e1+m1*p1+1)]
            success = (us[i] == uex).all()
            assert success

#    for minus,plus in connectivity:
#        if not interfaces[minus,plus].is_comm_null:
#            axis = interfaces[minus,plus].axis
#            I = interfaces[minus,plus]

#            if not carts[minus].is_comm_null:
#                uex =  [[val(plus,i1,i2)for i2 in range(*ranges[1])] for i1 in range(*ranges[0])]
#                uex = np.pad(uex, [(m*p,m*p) for m,p in zip(carts[minus].shifts, carts[minus].pads)])
#                u_ij = us[plus]
#            elif not carts[plus].is_comm_null:
#                uex =  [[val(minus,i1,i2)for i2 in range(*ranges[1])] for i1 in range(*ranges[0])]
#                uex = np.pad(uex, [(m*p,m*p) for m,p in zip(carts[plus].shifts, carts[plus].pads)])
#                u_ij = us[minus]

#            success = (u_ij == uex).all()
#            assert success

#===============================================================================
# RUN TEST MANUALLY
#===============================================================================
if __name__=='__main__':

    run_carts_2d()


