#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from mpi4py import MPI

from .cart import InterfaceCartDecomposition, find_mpi_type


__all__ = ('InterfaceCartDataExchanger',)

class InterfaceCartDataExchanger:
    """
    This takes care of updating the ghost regions between two sides of an interface for a
    multi-dimensional array distributed according to the given Cartesian
    decomposition of a tensor-product grid of coefficients.

    Parameters
    ----------
    cart : psydac.ddm.InterfaceCartDecomposition
        Object that contains all information about the Cartesian decomposition
        of a tensor-product grid of coefficients.

    dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
        Datatype of single coefficient (if scalar) or of each of its
        components (if vector).

    coeff_shape : [tuple(int) | list(int)]
        Shape of a single coefficient, if this is multi-dimensional
        (optional: by default, we assume scalar coefficients).

    """
    def __init__(self, cart, dtype, *, coeff_shape=()):

        assert isinstance(cart, InterfaceCartDecomposition)

        send_types, recv_types = self._create_buffer_types( cart, dtype , coeff_shape=coeff_shape)

        self._cart          = cart
        self._dtype         = dtype
        self._send_types    = send_types
        self._recv_types    = recv_types
        self._dest_ranks    = cart.get_interface_communication_infos( cart.axis )['dest_ranks']
        self._source_ranks  = cart.get_interface_communication_infos( cart.axis )['source_ranks']


    # ...
    def update_ghost_regions( self, array_minus=None, array_plus=None ):
        req = self.start_update_ghost_regions(array_minus, array_plus)
        self.end_update_ghost_regions(req)

    # ...
    def start_update_ghost_regions( self, array_minus=None, array_plus=None ):
        send_req = []
        recv_req = []
        cart      = self._cart
        intercomm = cart.intercomm

        for i,(st,rank) in enumerate(zip(self._send_types, self._dest_ranks)):

            if cart._local_rank_minus is not None and array_minus is not None:
                send_buf = (array_minus, 1, st)
                send_req.append(intercomm.Isend( send_buf, rank ))
            elif cart._local_rank_plus is not None and array_plus is not None:
                send_buf = (array_plus, 1, st)
                send_req.append(intercomm.Isend( send_buf, rank ))

        for i,(rt,rank) in enumerate(zip(self._recv_types, self._source_ranks)):

            if cart._local_rank_minus is not None and array_plus is not None:
                recv_buf = (array_plus, 1, rt)
                recv_req.append(intercomm.Irecv( recv_buf, rank ))
            elif cart._local_rank_plus is not None and array_minus is not None:
                recv_buf = (array_minus, 1, rt)
                recv_req.append(intercomm.Irecv( recv_buf, rank ))

        return send_req + recv_req

    def end_update_ghost_regions(self, req):
        MPI.Request.Waitall(req)

    @staticmethod
    def _create_buffer_types( cart, dtype , *, coeff_shape=()):

        assert isinstance( cart, InterfaceCartDecomposition )

        mpi_type = find_mpi_type( dtype )
        info     = cart.get_interface_communication_infos( cart.axis )

        # Possibly, each coefficient could have multiple components
        coeff_shape = list( coeff_shape )
        coeff_start = [0] * len( coeff_shape )

        send_types  = [None]*len(info['dest_ranks'])
        axis        = cart.axis
        for i in range(len(info['dest_ranks'])):

            gbuf_shape  = list(info['gbuf_send_shape'][i])  + coeff_shape
            buf_shape   = list(info['buf_send_shape'][i])   + coeff_shape
            send_starts = list(info['gbuf_send_starts'][i]) + coeff_start

            if coeff_shape:
                gbuf_shape[axis]  = info['gbuf_recv_shape'][0][axis]
                buf_shape[axis]   = info['buf_recv_shape'][0][axis]
                send_starts[axis] = info['gbuf_recv_starts'][0][axis]

            send_types[i] = mpi_type.Create_subarray(
                         sizes    = gbuf_shape,
                         subsizes = buf_shape,
                         starts   = send_starts).Commit()

        recv_types = [None]*len(info['source_ranks'])

        for i in range(len(info['source_ranks'])):

            gbuf_shape  = list(info['gbuf_recv_shape'][i])  + coeff_shape
            buf_shape   = list(info['buf_recv_shape'][i])   + coeff_shape
            recv_starts = list(info['gbuf_recv_starts'][i]) + coeff_start

            recv_types[i] = mpi_type.Create_subarray(
                         sizes    = gbuf_shape,
                         subsizes = buf_shape,
                         starts   = recv_starts).Commit()

        return send_types, recv_types
