#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from itertools import product

import numpy as np
from mpi4py import MPI

from .cart import CartDecomposition, find_mpi_type
from .basic import CartDataExchanger


__all__ = ('NonBlockingCartDataExchanger',)

class NonBlockingCartDataExchanger(CartDataExchanger):
    """
    Type that takes care of updating the ghost regions (padding) of a
    multi-dimensional array distributed according to the given Cartesian
    decomposition of a tensor-product grid of coefficients.

    Each coefficient in the decomposed grid may have multiple components,
    contiguous in memory.

    Parameters
    ----------
    cart : psydac.ddm.CartDecomposition
        Object that contains all information about the Cartesian decomposition
        of a tensor-product grid of coefficients.

    dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
        Datatype of single coefficient (if scalar) or of each of its
        components (if vector).

    coeff_shape : [tuple(int) | list(int)]
        Shape of a single coefficient, if this is multi-dimensional
        (optional: by default, we assume scalar coefficients).

    """
    def __init__( self, cart, dtype, *, coeff_shape=(),  assembly=False, axis=None, shape=None ):

        self._send_types, self._recv_types = self._create_buffer_types(
                cart, dtype, coeff_shape=coeff_shape )

        self._cart = cart
        self._comm = cart.comm_cart
        self._axis = axis

        if assembly:
            self._assembly_send_types, self._assembly_recv_types = self._create_assembly_buffer_types(
                cart, dtype, coeff_shape=coeff_shape, axis=axis, shape=shape)
    #---------------------------------------------------------------------------
    # Public interface
    #---------------------------------------------------------------------------
    def get_send_type( self, *args ):
        shift = args[0]
        return self._send_types[shift]

    # ...
    def get_recv_type( self, *args ):
        shift = args[0]
        return self._recv_types[shift]

    # ...
    def get_assembly_send_type( self, *args ):
        direction = args[0]
        disp      = args[1]
        return self._assembly_send_types[direction, disp]

    # ...
    def get_assembly_recv_type( self, *args ):
        direction = args[0]
        disp      = args[1]
        return self._assembly_recv_types[direction, disp]

    # ...
    def prepare_communications(self, u):

        # Requests' handles
        requests = []
        cart = self._cart
        for shift in product( [-1,0,1], repeat=cart._ndims ):
            if all(s==0 for s in shift):
                continue

            info     = cart.get_shift_info_non_blocking( shift )
            comm     = cart._comm_cart

            recv_typ = self.get_recv_type( shift )
            if recv_typ != MPI.DATATYPE_NULL:
                recv_buf = (u, 1, recv_typ)
                recv_req = comm.Recv_init( recv_buf, info['rank_source'], info['tag'] )
                requests.append( recv_req )

            send_typ = self.get_send_type( shift )
            if send_typ != MPI.DATATYPE_NULL:
                send_buf = (u, 1, send_typ)
                send_req = comm.Send_init( send_buf, info['rank_dest'], info['tag'] )
                requests.append( send_req )

        return tuple(requests)

    def start_update_ghost_regions(self, array, requests ):
        MPI.Prequest.Startall( requests )

    def end_update_ghost_regions(self, array, requests):
        MPI.Prequest.Waitall  ( requests )

    # ...
    def start_exchange_assembly_data( self, array ):

        assert isinstance( array, np.ndarray )

        # Shortcuts
        cart  = self._cart
        comm  = self._comm
        gcomm = comm
        ndim  = cart.ndim

        # Choose non-negative invertible function tag(disp) >= 0
        # NOTES:
        #   . different values of disp must return different tags!
        #   . tag at receiver must match message tag at sender
        tag = lambda disp: 42+disp

        # Requests' handles

        for direction in range( ndim ):
            if direction == self._axis: continue
            if self._axis is not None: comm = cart.subcomm[direction]

            # Start receiving data (MPI_IRECV)
            disp        = 1
            info        = cart.get_shift_info( direction, disp )
            recv_typ    = self.get_assembly_recv_type ( direction, disp )
            rank_source = info['rank_source']

            if self._axis is not None:
                rank_source = gcomm.group.Translate_ranks(np.array([rank_source]), comm.group)[0]
            
            recv_buf = (array, 1, recv_typ)
            recv_req = comm.Irecv( recv_buf, rank_source, tag(disp) )

            # Start sending data (MPI_ISEND)
            send_typ = self.get_assembly_send_type ( direction, disp )
            rank_dest = info['rank_dest']

            if self._axis is not None:
                rank_dest = gcomm.group.Translate_ranks(np.array([rank_dest]), comm.group)[0]

            send_buf = (array, 1, send_typ)
            send_req = comm.Isend( send_buf, rank_dest, tag(disp) )

            # Wait for end of data exchange (MPI_WAITALL)
            MPI.Request.Waitall( [recv_req, send_req] )

            if disp == 1:
                info = cart.get_shift_info( direction, disp )
                pads = [0]*ndim
                pads[direction] = cart._pads[direction]*cart._shifts[direction]
                idx_from = tuple(slice(s,s+b) for s,b in zip(info['recv_starts'],info['buf_shape']))
                idx_to   = tuple(slice(s+p,s+b+p) for s,b,p in zip(info['recv_starts'],info['buf_shape'],pads))
                array[idx_to] += array[idx_from]
            else:
                info = cart.get_shift_info( direction, disp )
                pads = [0]*ndim
                pads[direction] = cart._pads[direction]*cart._shifts[direction]
                idx_from = tuple(slice(s,s+b) for s,b in zip(info['recv_starts'],info['buf_shape']))
                idx_to   = tuple(slice(s-p,s+b-p) for s,b,p in zip(info['recv_starts'],info['buf_shape'],pads))
                array[idx_to] += array[idx_from]

    def end_exchange_assembly_data( self, array ):
        pass

    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------
    @staticmethod
    def _create_buffer_types( cart, dtype, *, coeff_shape=() ):
        """
        Create MPI subarray datatypes for updating the ghost regions (padding)
        of a multi-dimensional array distributed according to the given Cartesian
        decomposition of a tensor-product grid of coefficients.

        MPI requires a subarray datatype for accessing non-contiguous slices of
        a multi-dimensional array; this is a typical situation when updating the
        ghost regions.

        Each coefficient in the decomposed grid may have multiple components,
        contiguous in memory.

        Parameters
        ----------
        cart : psydac.ddm.CartDecomposition
            Object that contains all information about the Cartesian decomposition
            of a tensor-product grid of coefficients.

        dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
            Datatype of single coefficient (if scalar) or of each of its
            components (if vector).

        coeff_shape : [tuple(int) | list(int)]
            Shape of a single coefficient, if this is multidimensional
            (optional: by default, we assume scalar coefficients).

        Returns
        -------
        send_types : dict
            Dictionary of MPI subarray datatypes for SEND BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.

        recv_types : dict
            Dictionary of MPI subarray datatypes for RECEIVE BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.

        """
        assert isinstance( cart, CartDecomposition )

        mpi_type = find_mpi_type( dtype )

        # Possibly, each coefficient could have multiple components
        coeff_shape = list( coeff_shape )
        coeff_start = [0] * len( coeff_shape )

        data_shape = list( cart.shape ) + coeff_shape

        send_types = {}
        recv_types = {}
        for shift in product( [-1,0,1], repeat=cart._ndims ):
            if all(s == 0 for s in shift):
                continue
            info = cart.get_shift_info_non_blocking( shift )

            buf_shape   = list( info[ 'buf_shape' ] ) + coeff_shape
            send_starts = list( info['send_starts'] ) + coeff_start
            recv_starts = list( info['recv_starts'] ) + coeff_start

            if info['rank_dest']>=0:
                send_types[shift] = mpi_type.Create_subarray(
                    sizes    = data_shape,
                    subsizes = buf_shape,
                    starts   = send_starts,
                ).Commit()
            else:
                send_types[shift] = MPI.DATATYPE_NULL

            if info['rank_source']>=0:
                recv_types[shift] = mpi_type.Create_subarray(
                    sizes    = data_shape,
                    subsizes = buf_shape,
                    starts   = recv_starts,
                ).Commit()
            else:
                recv_types[shift] = MPI.DATATYPE_NULL

        return send_types, recv_types


    # ...
    @staticmethod
    def _create_assembly_buffer_types( cart, dtype, *, coeff_shape=(), axis=None, shape=None ):
        """
        Create MPI subarray datatypes for updating the ghost regions (padding)
        of a multi-dimensional array distributed according to the given Cartesian
        decomposition of a tensor-product grid of coefficients.
        MPI requires a subarray datatype for accessing non-contiguous slices of
        a multi-dimensional array; this is a typical situation when updating the
        ghost regions.
        Each coefficient in the decomposed grid may have multiple components,
        contiguous in memory.
        Parameters
        ----------
        cart : psydac.ddm.CartDecomposition
            Object that contains all information about the Cartesian decomposition
            of a tensor-product grid of coefficients.
        dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
            Datatype of single coefficient (if scalar) or of each of its
            components (if vector).
        coeff_shape : [tuple(int) | list(int)]
            Shape of a single coefficient, if this is multidimensional
            (optional: by default, we assume scalar coefficients).
        Returns
        -------
        send_types : dict
            Dictionary of MPI subarray datatypes for SEND BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.
        recv_types : dict
            Dictionary of MPI subarray datatypes for RECEIVE BUFFERS, accessed
            through the integer pair (direction, displacement) as key;
            'direction' takes values from 0 to ndim, 'disp' is -1 or +1.
        """
        assert isinstance( cart, CartDecomposition )

        mpi_type = find_mpi_type( dtype )

        # Possibly, each coefficient could have multiple components
        coeff_shape = list( coeff_shape )
        coeff_start = [0] * len( coeff_shape )

        data_shape = list( cart.shape ) + coeff_shape
        send_types = {}
        recv_types = {}

        if axis is not None:
            data_shape[axis] = shape[axis]

        for direction in range( cart.ndim ):
            for disp in [-1, 1]:
                info = cart.get_shift_info( direction, disp )

                buf_shape   = list( info[ 'buf_shape' ] ) + coeff_shape
                send_starts = list( info['send_assembly_starts'] ) + coeff_start
                recv_starts = list( info['recv_assembly_starts'] ) + coeff_start
                if direction == axis:continue
                if axis is not None:
                    buf_shape[axis]   = shape[axis]
                    send_starts[axis] = 0
                    recv_starts[axis] = 0

                send_types[direction,disp] = mpi_type.Create_subarray(
                    sizes    = data_shape ,
                    subsizes =  buf_shape ,
                    starts   = send_starts,
                ).Commit()

                recv_types[direction,disp] = mpi_type.Create_subarray(
                    sizes    = data_shape ,
                    subsizes =  buf_shape ,
                    starts   = recv_starts,
                ).Commit()

        return send_types, recv_types
