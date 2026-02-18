#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from sympde.topology import Interface

from psydac.ddm.cart   import CartDecomposition, InterfaceCartDecomposition, create_interfaces_cart
from psydac.fem.vector import VectorFemSpace


__all__ = (
    'partition_coefficients',
    'construct_connectivity',
    'get_minus_starts_ends',
    'get_plus_starts_ends',
    'create_cart',
    'construct_interface_spaces',
    'construct_reduced_interface_spaces'
)


def partition_coefficients(domain_decomposition, spaces, min_blocks=None):
    """
    Partition the coefficients starting from the grid decomposition.

    Parameters
    ----------

    domain_decomposition: DomainDecomposition
        The distributed topological domain.

    spaces: list of SplineSpace
        The 1d spline spaces that construct the tensor fem space.

    min_blocks: list of int
        The minimum number of coefficients owned by a process.

    Returns
    -------

    global_starts: list of list of int
        The starts of the coefficients for every process along each direction.

    global_ends: list of list of int
        The ends of the coefficients for every process along each direction.

    """
    npts         = [V.nbasis   for V in spaces]
    multiplicity = [V.multiplicity for V in spaces]

    ndims         = len(npts)
    global_starts = [None] * ndims
    global_ends   = [None] * ndims

    for axis in range(ndims):
        es = domain_decomposition.global_element_starts[axis]
        ee = domain_decomposition.global_element_ends  [axis]
        m  = multiplicity[axis]

        global_ends  [axis]     = m*(ee+1)-1
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    if min_blocks is None:
        min_blocks = [None] * ndims

    for s, e, V, mb in zip(global_starts, global_ends, spaces, min_blocks):
        if V.periodic or mb is None:
            assert all(e-s+1 >= V.degree)
        else:
            assert all(e-s+1 >= mb)

    return global_starts, global_ends


def construct_connectivity(domain):
    """ 
    Compute the connectivity of the multipatch domain.

    Parameters
    ----------
    domain : Sympde.topology.Domain
        The multipatch domain.

    Returns
    -------
    connectivity : dict
        Connectivity between the patches.
        It takes the form of {(i, j):((axis_i, ext_i),(axis_j, ext_j))} for each item of the dictionary,
        where i,j represent the patch indices

    """
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
        connectivity[i, j] = ((e.minus.axis, e.minus.ext),(e.plus.axis, e.plus.ext))

    return connectivity

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

#------------------------------------------------------------------------------
def create_cart(domain_decomposition, spaces):
    """
    Compute the cartesian decomposition of the coefficient space.
    Two different cases are possible:
    
    - Single patch :
        We distribute the coefficients using all the processes provided by the given communicator.
    - Multiple patches :
        We decompose the provided communicator in a list of smaller disjoint intra-communicators,
        and decompose the coefficients of each patch with an assigned intra-communicator.

    Parameters
    ----------
    domain_decomposition : DomainDecomposition | tuple of DomainDecomposition

    spaces : list of list of 1D global Spline spaces
        The 1D global spline spaces that will be distributed.

    Returns
    -------
    cart : tuple of CartDecomposition
        Cartesian decomposition of the coefficient space for each patch in the domain.

    """

    if len(spaces) == 1:
        domain_decomposition = domain_decomposition[0]
        spaces       = spaces[0]
        npts         = [V.nbasis for V in spaces]
        pads         = [V._pads  for V in spaces]
        multiplicity = [V.multiplicity for V in spaces]

        global_starts, global_ends = partition_coefficients(domain_decomposition, spaces)

        carts = [CartDecomposition(
                domain_decomposition = domain_decomposition,
                npts          = npts,
                global_starts = global_starts,
                global_ends   = global_ends,
                pads          = pads,
                shifts        = multiplicity)]
    else:
        carts = []
        for i in range(len(spaces)):
            npts         = [V.nbasis for V in spaces[i]]
            pads         = [V._pads  for V in spaces[i]]
            multiplicity = [V.multiplicity for V in spaces[i]]

            global_starts, global_ends = partition_coefficients(
                    domain_decomposition[i],
                    spaces[i],
                    min_blocks=[p+1 for p in pads])

            new_cart = CartDecomposition(
                            domain_decomposition = domain_decomposition[i],
                            npts          = npts,
                            global_starts = global_starts,
                            global_ends   = global_ends,
                            pads          = pads,
                            shifts        = multiplicity)

            carts.append(new_cart)

        carts = tuple(carts)

    return carts

#------------------------------------------------------------------------------
def construct_interface_spaces(domain_decomposition, g_spaces, carts, interiors, connectivity):
    """ 
    Create the fem spaces for each interface in the domain given by the connectivity.

    Parameters
    ----------
    domain_decomposition : DomainDecomposition

    g_spaces : dict
     dictionary that contains the tensor-fem space for each patch.

    cart: <CartDecomposition|MultiCartDecomposition>
        Cartesian decomposition of the coefficient space.

    interiors: list of Sympde.topology.Domain
      List of the multipatch domain interiors.

    connectivity: dict
       The connectivity of the multipatch domain.
    """
    if not connectivity:return
    comm = domain_decomposition.comm
    interfaces_cart = None
    if comm is not None:
        if connectivity:
            communication_info = (get_minus_starts_ends, get_plus_starts_ends)
            interfaces_cart = create_interfaces_cart(domain_decomposition, carts, connectivity.copy(), communication_info=communication_info)

    for i,j in connectivity:
        if comm is None:
            cart_minus = carts[i]
            cart_plus  = carts[j]
        else:
            if not carts[i].is_comm_null and not carts[j].is_comm_null:
                cart_minus = carts[i]
                cart_plus  = carts[j]
            elif (i,j) in interfaces_cart:
                cart_minus = interfaces_cart[i,j]
                cart_plus  = interfaces_cart[i,j]
            else:
                continue

        ((axis_minus, ext_minus), (axis_plus , ext_plus)) = connectivity[i, j]

        g_spaces[interiors[i]].create_interface_space(axis_minus, ext_minus, cart=cart_minus)
        g_spaces[interiors[j]].create_interface_space(axis_plus , ext_plus , cart=cart_plus)

        max_ncells = tuple(max(ni,nj) for ni,nj in zip(g_spaces[interiors[i]].ncells,g_spaces[interiors[j]].ncells))
        cart_minus = g_spaces[interiors[i]].get_refined_space(max_ncells).coeff_space.cart
        cart_plus  = g_spaces[interiors[j]].get_refined_space(max_ncells).coeff_space.cart
        if isinstance(cart_minus, InterfaceCartDecomposition):
            cart = InterfaceCartDecomposition(cart_minus._cart_minus, cart_plus._cart_plus,
                                              cart_minus._comm, [axis_minus, axis_plus], [ext_minus, ext_plus],
                                              [cart_minus.ranks_in_topo_minus, cart_plus.ranks_in_topo_plus],
                                              [cart_minus._local_group_minus, cart_plus._local_group_plus],
                                              [cart_minus._local_comm_minus, cart_plus._local_comm_plus],
                                              [cart_minus._root_rank_minus, cart_plus._root_rank_plus],
                                              [], reduce_elements=True)

            cart_minus = cart
            cart_plus  = cart
            cart.set_interface_communication_infos(get_minus_starts_ends, get_plus_starts_ends)

        if any(nci!=ncj for nci,ncj in zip(max_ncells, g_spaces[interiors[i]].ncells)):
            g_spaces[interiors[i]].get_refined_space(max_ncells).create_interface_space(axis_minus, ext_minus, cart=cart_minus)
            g_spaces[interiors[j]].get_refined_space(max_ncells).create_interface_space(axis_plus , ext_plus , cart=cart_plus)

#------------------------------------------------------------------------------
def construct_reduced_interface_spaces(spaces, reduced_spaces, interiors, connectivity):
    """
    Create the reduced spaces for the interface coefficients.

    Parameters
    ----------

    spaces: dict
        The tensor FEM spaces that we want to reduce for each patch.

    reduced_spaces: dict
        The reduced coefficient space for each patch.

    interiors: list of Sympde.topology.Domain
        The patches that construct the multipatch domain.
        
    connectivity: dict
       The connectivity of the multipatch domain.

    """
    for i,j in connectivity:
        ((axis_i, ext_i), (axis_j , ext_j)) = connectivity[i, j]

        patch_i = interiors[i]
        patch_j = interiors[j]
        space_i = spaces[patch_i].interfaces.get((axis_i, ext_i), None)
        space_j = spaces[patch_j].interfaces.get((axis_j, ext_j), None)

        if space_i is None or space_j is None: continue

        cart_i  = space_i.coeff_space.cart
        cart_j  = space_j.coeff_space.cart

        if isinstance(cart_i, InterfaceCartDecomposition):
            assert cart_i is cart_j
            if isinstance(reduced_spaces[patch_i], VectorFemSpace):
                for Vi,Vj in zip(reduced_spaces[patch_i].spaces, reduced_spaces[patch_j].spaces):
                    npts_i = [Vik.nbasis for Vik in Vi.spaces]
                    npts_j = [Vik.nbasis for Vik in Vj.spaces]
                    global_starts_i = Vi.coeff_space.cart.global_starts
                    global_starts_j = Vj.coeff_space.cart.global_starts
                    global_ends_i   = Vi.coeff_space.cart.global_ends
                    global_ends_j   = Vj.coeff_space.cart.global_ends
                    shifts_i        = Vi.coeff_space.cart.shifts
                    shifts_j        = Vj.coeff_space.cart.shifts
                    cart_ij         = cart_i.reduce_npts(Vi.coeff_space.cart, Vj.coeff_space.cart)
                    cart_ij.set_interface_communication_infos(get_minus_starts_ends, get_plus_starts_ends)
                    Vi.create_interface_space(axis_i, ext_i, cart=cart_ij)
                    Vj.create_interface_space(axis_j, ext_j, cart=cart_ij)

                    cart = InterfaceCartDecomposition(cart_ij._cart_minus, cart_ij._cart_plus,
                                                      cart_ij._comm, [axis_i, axis_j], [ext_i, ext_j],
                                                      [cart_ij.ranks_in_topo_minus, cart_ij.ranks_in_topo_plus],
                                                      [cart_ij._local_group_minus, cart_ij._local_group_plus],
                                                      [cart_ij._local_comm_minus, cart_ij._local_comm_plus],
                                                      [cart_ij._root_rank_minus, cart_ij._root_rank_plus],
                                                      [], reduce_elements=True)

                    cart.set_interface_communication_infos(get_minus_starts_ends, get_plus_starts_ends)
                    max_ncells = tuple(max(ni,nj) for ni,nj in zip(Vi.ncells,Vj.ncells))
                    Vi.get_refined_space(max_ncells).create_interface_space(axis_i, ext_i, cart=cart)
                    Vj.get_refined_space(max_ncells).create_interface_space(axis_j, ext_j, cart=cart)
            else:
                Vi = reduced_spaces[patch_i]
                Vj = reduced_spaces[patch_j]
                npts_i = [Vik.nbasis for Vik in Vi.spaces]
                npts_j = [Vik.nbasis for Vik in Vj.spaces]
                cart_ij  = cart_i.reduce_npts(Vi.coeff_space.cart, Vj.coeff_space.cart)
                cart_ij.set_interface_communication_infos(get_minus_starts_ends, get_plus_starts_ends)
                Vi.create_interface_space(axis_i, ext_i, cart=cart_ij)
                Vj.create_interface_space(axis_j, ext_j, cart=cart_ij)

                cart = InterfaceCartDecomposition(cart_ij._cart_minus, cart_ij._cart_plus,
                                                  cart_ij._comm, [axis_i, axis_j], [ext_i, ext_j],
                                                  [cart_ij.ranks_in_topo_minus, cart_ij.ranks_in_topo_plus],
                                                  [cart_ij._local_group_minus, cart_ij._local_group_plus],
                                                  [cart_ij._local_comm_minus, cart_ij._local_comm_plus],
                                                  [cart_ij._root_rank_minus, cart_ij._root_rank_plus],
                                                  [], reduce_elements=True)

                cart.set_interface_communication_infos(get_minus_starts_ends, get_plus_starts_ends)
                max_ncells = tuple(max(ni,nj) for ni,nj in zip(Vi.ncells,Vj.ncells))
                Vi.get_refined_space(max_ncells).create_interface_space(axis_i, ext_i, cart=cart)
                Vj.get_refined_space(max_ncells).create_interface_space(axis_j, ext_j, cart=cart)
        else:
            if isinstance(reduced_spaces[patch_i], VectorFemSpace):
                for Vi,Vj in zip(reduced_spaces[patch_i].spaces, reduced_spaces[patch_j].spaces):
                    Vi.create_interface_space(axis_i, ext_i, cart=Vi.coeff_space.cart)
                    Vj.create_interface_space(axis_j , ext_j , cart=Vj.coeff_space.cart)
                    max_ncells = tuple(max(ni,nj) for ni,nj in zip(Vi.ncells,Vj.ncells))
                    cart_i = Vi.get_refined_space(max_ncells).coeff_space.cart
                    cart_j = Vj.get_refined_space(max_ncells).coeff_space.cart
                    Vi.get_refined_space(max_ncells).create_interface_space(axis_i, ext_i, cart=cart_i)
                    Vj.get_refined_space(max_ncells).create_interface_space(axis_j, ext_j, cart=cart_j)
            else:
                Vi = reduced_spaces[patch_i]
                Vj = reduced_spaces[patch_j]
                Vi.create_interface_space(axis_i, ext_i, cart=Vi.coeff_space.cart)
                Vj.create_interface_space(axis_j , ext_j , cart=Vj.coeff_space.cart)
                max_ncells = tuple(max(ni,nj) for ni,nj in zip(Vi.ncells,Vj.ncells))
                cart_i = Vi.get_refined_space(max_ncells).coeff_space.cart
                cart_j = Vj.get_refined_space(max_ncells).coeff_space.cart
                Vi.get_refined_space(max_ncells).create_interface_space(axis_i, ext_i, cart=cart_i)
                Vj.get_refined_space(max_ncells).create_interface_space(axis_j, ext_j, cart=cart_j)
