# -*- coding: UTF-8 -*-
import os

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

def create_interfaces_cart(cart, interfaces_info=None):
    interfaces_cart = None
    if interfaces_info:
        interfaces_info = interfaces_info.copy()
        interfaces_cart = InterfacesCartDecomposition(cart, interfaces_info)
        for i,j in interfaces_info:
            axes   = interfaces_info[i,j][0]
            exts   = interfaces_info[i,j][1]
            if (i,j) in interfaces_cart.carts and not interfaces_cart.carts[i,j].is_comm_null:
                interfaces_cart.carts[i,j].set_communication_info(get_minus_starts_ends, get_plus_starts_ends)

    return interfaces_cart
