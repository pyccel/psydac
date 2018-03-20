import numpy as np
from itertools import product
from mpi4py    import MPI

#===============================================================================
class MPICart2D():

    def __init__( self, npts, pads, periods, reorder, comm=MPI.COMM_WORLD ):

        assert( len( npts ) == len( pads ) == len( periods ) )

        # ...
        self._ndims = len( npts )
        # ...

        # ...
        self._pads    = pads
        self._periods = periods
        self._reorder = reorder
        # ...

        # ...
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()
        # ...

        # ...
        # Know the number of processes along each direction
        self._dims = MPI.Compute_dims( self._size, self._ndims )
        # ...

        # ...
        # Create a 2D MPI cart
        self._comm_cart = comm.Create_cart(
            dims    = self._dims,
            periods = self._periods,
            reorder = self._reorder
        )

        # Know my coordinates in the topology
        self._rank_in_topo = self._comm_cart.Get_rank()
        self._coords       = self._comm_cart.Get_coords( rank=self._rank_in_topo )

        # Start/end values of global indices (without ghost regions)
        self._starts = tuple( ( c   *n)//d   for n,d,c in zip( npts, self._dims, self._coords ) )
        self._ends   = tuple( ((c+1)*n)//d-1 for n,d,c in zip( npts, self._dims, self._coords ) )

        # List of 1D global indices (without ghost regions)
        self._grids = tuple( range(s,e+1) for s,e in zip( self._starts, self._ends ) )

        # N-dimensional global indices (without ghost regions)
        self._indices = product( *self._grids )

        # Compute shape of local arrays in topology (with ghost regions)
        self._shape = tuple( e-s+1+2*p for s,e,p in zip( self._starts, self._ends, self._pads ) )

        # Extended grids with ghost regions
        self._extended_grids = tuple( range(s-p,e+p+1) for s,e,p in zip( self._starts, self._ends, self._pads ) )

        # N-dimensional global indices with ghost regions
        self._extended_indices = product( *self._extended_grids )

        # Create (N-1)-dimensional communicators within the Cartesian topology
        self._subcomm = [None]*self._ndims
        for i in range(self._ndims):
            remain_dims     = [i==j for j in range( self._ndims )]
            self._subcomm[i] = self._comm_cart.Sub( remain_dims )

        # Buffers for communicating with neighbors
        self._send_buffers = {}
        self._recv_buffers = {}
        zero_shift = tuple( [0]*self._ndims )
        for shift in product( [-1,0,1], repeat=self._ndims ):

            if shift == zero_shift:
                continue

            info = self.get_sendrecv_info( shift )
            self._send_buffers[shift] = np.zeros( info['buf_shape'] )
            self._recv_buffers[shift] = np.zeros( info['buf_shape'] )

        # DEBUG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self._rank == 0:
            print( "", flush=True )

        zero_shift = tuple( [0]*self._ndims )
        for k in range( self._size ):
            if self._rank == k:
                print( "="*40 )
                print( "RANK = {}".format( self._rank ) )
                print( "="*40 )
                for shift in product( [-1,0,1], repeat=self._ndims ):
                    if shift == zero_shift:
                        continue
                    print( "sendrecv_info: shift = {}".format( shift ) )
                    print( self.get_sendrecv_info( shift ) )
                    print( "" )
                print( "", flush=True )

            self._comm.Barrier()

        if self._rank == 0:
            print( "", flush=True )
        # <<<<<<<<<<<<<<<<<<<<<<<<<< END DEBUG

    #---------------------------------------------------------------------------
    @property
    def starts( self ):
        return self._starts

    @property
    def ends( self ):
        return self._ends

    @property
    def pads( self ):
        return self._pads

    @property
    def coords( self ):
        return self._coords

    #---------------------------------------------------------------------------
    def coords_exist( self, coords ):

        return all( P or (0 <= c < d) for P,c,d in zip( self._periods, coords, self._dims ) )

    #---------------------------------------------------------------------------
    def get_sendrecv_info( self, shift ):
        assert( len( shift ) == self._ndims )

        # Compute coordinates of destination and source
        coords_dest   = [c+h for c,h in zip( self._coords, shift )]
        coords_source = [c-h for c,h in zip( self._coords, shift )]

        # Convert coordinates to rank, taking care of non-periodic dimensions
        if self.coords_exist( coords_dest ):
            rank_dest = self._comm_cart.Get_cart_rank( coords_dest )
        else:
            rank_dest = MPI.PROC_NULL

        if self.coords_exist( coords_source ):
            rank_source = self._comm_cart.Get_cart_rank( coords_source )
        else:
            rank_source = MPI.PROC_NULL

        # Compute information for exchanging ghost cell data
        indx_recv = []
        indx_send = []
        buf_shape = []
        for s,e,p,h in zip( self._starts, self._ends, self._pads, shift ):
            if h == 0:
                slice_recv = slice( p, -p )
                slice_send = slice( p, -p )
                length     = e-s+1
            elif h == 1:
                slice_recv = slice(    0,  p )
                slice_send = slice( -2*p, -p )
                length     = p
            elif h == -1:
                slice_recv = slice( -p, None )
                slice_send = slice(  p, 2*p  )
                length     = p
            indx_recv.append( slice_recv )
            indx_send.append( slice_send )
            buf_shape.append( length )

        # Store all information into dictionary
        info = {'rank_dest'  : rank_dest,
                'rank_source': rank_source,
                'indx_send'  : tuple( indx_send ),
                'indx_recv'  : tuple( indx_recv ),
                'buf_shape'  : tuple( buf_shape ) }

        # Return dictionary
        return info

    #---------------------------------------------------------------------------
    def __del__(self):

        # Destroy sub-communicators
        for s in self._subcomm:
            s.Free()

        # Destroy Cartesian communicator
        self._comm_cart.Free()

    #---------------------------------------------------------------------------
    def communicate(self, u):

        assert( self._shape == u.shape )

        tag    = 1435
        status = MPI.Status()

        # ... Communication
        zero_shift = tuple( [0]*self._ndims )
        for shift in product( [-1,0,1], repeat=self._ndims ):

            if shift == zero_shift:
                continue

            info = self.get_sendrecv_info( shift )

            sendbuf = self._send_buffers[shift]
            recvbuf = self._recv_buffers[shift]

            # Copy data from u to contiguous send buffer
            sendbuf[...] = u[info['indx_send']]

            # Send and receive data
            self._comm_cart.Sendrecv(
                sendbuf = sendbuf,
                dest    = info['rank_dest'],
                sendtag = tag,
                recvbuf = recvbuf,
                source  = info['rank_source'],
                recvtag = tag,
                status  = status,
            )

            # Copy data from contiguous receive buffer to u
            u[info['indx_recv']] = recvbuf[...]

        # TODO: use non-blocking ISEND + IRECV instead of blocking SENDRECV

        # TODO: increase MPI bandwidth by factor of 2 by using same MPI channel
        #       in both directions (here with dest=source)

    #---------------------------------------------------------------------------
    def reduce(self, x):

        global_x = self._comm_cart.allreduce( x, op=MPI.SUM )

        print(global_x, x)
