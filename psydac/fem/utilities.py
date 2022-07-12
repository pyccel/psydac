# -*- coding: UTF-8 -*-
import os

from sympde.topology import Interface

from psydac.ddm.cart import CartDecomposition, MultiCartDecomposition, InterfacesCartDecomposition

def create_cart(spaces, comm, reverse_axis=None, nprocs=None):

    num_threads     = int(os.environ.get('OMP_NUM_THREADS',1))

    if len(spaces) == 1:
        spaces = spaces[0]
        npts         = [V.nbasis   for V in spaces]
        pads         = [V._pads    for V in spaces]
        degree       = [V.degree   for V in spaces]
        multiplicity = [V.multiplicity for V in spaces]
        periods      = [V.periodic for V in spaces]

        cart = CartDecomposition(
            npts         = npts,
            pads         = pads,
            periods      = periods,
            reorder      = True,
            comm         = comm,
            shifts       = multiplicity,
            nprocs       = nprocs,
            reverse_axis = reverse_axis,
            num_threads  = num_threads)
    else:
        npts         = [[V.nbasis   for V in space_i] for space_i in spaces]
        pads         = [[V._pads    for V in space_i] for space_i in spaces]
        degree       = [[V.degree   for V in space_i] for space_i in spaces]
        multiplicity = [[V.multiplicity for V in space_i] for space_i in spaces]
        periods      = [[V.periodic for V in space_i] for space_i in spaces]

        cart = MultiCartDecomposition(
            npts         = npts,
            pads         = pads,
            periods      = periods,
            reorder      = True,
            comm         = comm,
            shifts       = multiplicity,
            num_threads  = num_threads)

    return cart

def get_minus_starts_ends(plus_starts, plus_ends, minus_npts, plus_npts, minus_axis, plus_axis, minus_ext, plus_ext, minus_pads, plus_pads, minus_shifts, plus_shifts, diff):
    starts = [max(0,s-m*p) for s,m,p in zip(plus_starts, minus_shifts, minus_pads)]
    ends   = [min(n,e+m*p) for e,n,m,p in zip(plus_ends, minus_npts, minus_shifts, minus_pads)]
    starts[minus_axis] = 0 if minus_ext == -1 else ends[minus_axis]-minus_pads[minus_axis]
    ends[minus_axis]   = ends[minus_axis] if minus_ext == 1 else minus_pads[minus_axis]
    return starts, ends

def get_plus_starts_ends(minus_starts, minus_ends, minus_npts, plus_npts, minus_axis, plus_axis, minus_ext, plus_ext, minus_pads, plus_pads, minus_shifts, plus_shifts, diff):
    starts = [max(0,s-m*p) for s,m,p in zip(minus_starts, plus_shifts, plus_pads)]
    ends   = [min(n,e+m*p) for e,n,m,p in zip(minus_ends, plus_npts, plus_shifts, plus_pads)]
    starts[plus_axis] = 0 if plus_ext == -1 else ends[plus_axis]-plus_pads[plus_axis]
    ends[plus_axis]   = ends[plus_axis] if plus_ext == 1 else plus_pads[plus_axis]
    return starts, ends

def create_interfaces_cart(cart, connectivity=None):
    interfaces_cart = None
    if connectivity:
        connectivity = connectivity.copy()
        interfaces_cart = InterfacesCartDecomposition(cart, connectivity)
        for i,j in connectivity:
            axes   = connectivity[i,j][0]
            exts   = connectivity[i,j][1]
            if (i,j) in interfaces_cart.carts and not interfaces_cart.carts[i,j].is_comm_null:
                interfaces_cart.carts[i,j].set_communication_info(get_minus_starts_ends, get_plus_starts_ends)

    return interfaces_cart

def construct_connectivity(domain):

    interfaces = domain.interfaces if domain.interfaces else []
    if len(domain)==1:
        interiors  = [domain.interior]
    else:
        interiors  = list(domain.interior.args)
        if interfaces:
            interfaces = [interfaces] if isinstance(interfaces, Interface) else list(interfaces.args)

    connectivity = {}
    for e in interfaces:
        i = interiors.index(e.minus.domain)
        j = interiors.index(e.plus.domain)
        connectivity[i, j] = ((e.minus.axis, e.plus.axis),(e.minus.ext, e.plus.ext))

    return connectivity

def construct_interface_spaces(g_spaces, cart, interiors, connectivity):
    if not connectivity:return
    comm = cart.comm if cart is not None else None
    if comm is not None:
        interfaces_cart = create_interfaces_cart(cart, connectivity=connectivity)
        if interfaces_cart:
            interfaces_cart = interfaces_cart.carts

    for i,j in connectivity:
        if comm is None:
            cart_minus = None
            cart_plus  = None
        else:
            if not cart.carts[i].is_comm_null and not cart.carts[j].is_comm_null:
                cart_minus = cart.carts[i]
                cart_plus  = cart.carts[j]
            elif (i,j) in interfaces_cart:
                cart_minus = interfaces_cart[i,j]
                cart_plus  = interfaces_cart[i,j]
            else:
                continue

        ((axis_minus, axis_plus), (ext_minus , ext_plus)) = connectivity[i, j]

        g_spaces[interiors[i]].set_interface_space(axis_minus, ext_minus, cart=cart_minus)
        g_spaces[interiors[j]].set_interface_space(axis_plus , ext_plus , cart=cart_plus)

