import numpy as np
from itertools import product
from mpi4py    import MPI

#===============================================================================
class Cart():

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

#        # Compute/store information for communicating with neighbors
#        self._sendrecv_info = {}
#        zero_shift = tuple( [0]*self._ndims )
#        for shift in product( [-1,0,1], repeat=self._ndims ):
#            if shift == zero_shift:
#                continue
#            self._sendrecv_info[shift] = self._compute_sendrecv_info( shift )

        # Compute/store information for communicating with neighbors
        self._shift_info = {}
        for dimension in range( self._ndims ):
            for disp in [-1,1]:
                self._shift_info[ dimension, disp ] = \
                        self._compute_shift_info( dimension, disp )

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

    @property
    def shape( self ):
        return self._shape

    @property
    def comm_cart( self ):
        return self._comm_cart

    #---------------------------------------------------------------------------
    def coords_exist( self, coords ):

        return all( P or (0 <= c < d) for P,c,d in zip( self._periods, coords, self._dims ) )

#    #---------------------------------------------------------------------------
#    def get_sendrecv_info( self, shift ):
#
#        return self._sendrecv_info[shift]
#
#    #---------------------------------------------------------------------------
#    def _compute_sendrecv_info( self, shift ):
#
#        assert( len( shift ) == self._ndims )
#
#        # Compute coordinates of destination and source
#        coords_dest   = [c+h for c,h in zip( self._coords, shift )]
#        coords_source = [c-h for c,h in zip( self._coords, shift )]
#
#        # Convert coordinates to rank, taking care of non-periodic dimensions
#        if self.coords_exist( coords_dest ):
#            rank_dest = self._comm_cart.Get_cart_rank( coords_dest )
#        else:
#            rank_dest = MPI.PROC_NULL
#
#        if self.coords_exist( coords_source ):
#            rank_source = self._comm_cart.Get_cart_rank( coords_source )
#        else:
#            rank_source = MPI.PROC_NULL
#
#        # Compute information for exchanging ghost cell data
#        buf_shape   = []
#        send_starts = []
#        recv_starts = []
#        for s,e,p,h in zip( self._starts, self._ends, self._pads, shift ):
#
#            if h == 0:
#                buf_length = e-s+1
#                recv_start = p
#                send_start = p
#
#            elif h == 1:
#                buf_length = p
#                recv_start = 0
#                send_start = e-s+1
#
#            elif h == -1:
#                buf_length = p
#                recv_start = e-s+1+p
#                send_start = p
#
#            buf_shape  .append( buf_length )
#            send_starts.append( send_start )
#            recv_starts.append( recv_start )
#
#        # Compute unique identifier for messages traveling along 'shift'
#        tag = sum( (h%3)*(3**n) for h,n in zip(shift,range(self._ndims)) )
#
#        # Store all information into dictionary
#        info = {'rank_dest'  : rank_dest,
#                'rank_source': rank_source,
#                'tag'        : tag,
#                'buf_shape'  : tuple(  buf_shape  ),
#                'send_starts': tuple( send_starts ),
#                'recv_starts': tuple( recv_starts )}
#
#        # return dictionary
#        return info

    #---------------------------------------------------------------------------
    def get_shift_info( self, direction, disp ):

        return self._shift_info[ direction, disp ]

    #---------------------------------------------------------------------------
    def _compute_shift_info( self, direction, disp ):

        assert( 0 <= direction < self._ndims )
        assert( isinstance( disp, int ) )

        # Process ranks for data shifting with MPI_SENDRECV
        (rank_source, rank_dest) = self.comm_cart.Shift( direction, disp )

        # Mesh info info along given direction
        s = self._starts[direction]
        e = self._ends  [direction]
        p = self._pads  [direction]

        # Shape of send/recv subarrays
        buf_shape = np.array( self._shape )
        buf_shape[direction] = p

        # Start location of send/recv subarrays
        send_starts = np.zeros( self._ndims, dtype=int )
        recv_starts = np.zeros( self._ndims, dtype=int )
        if disp > 0:
            recv_starts[direction] = 0
            send_starts[direction] = e-s+1
        elif disp < 0:
            recv_starts[direction] = e-s+1+p
            send_starts[direction] = p

        # Store all information into dictionary
        info = {'rank_dest'  : rank_dest,
                'rank_source': rank_source,
                'buf_shape'  : tuple(  buf_shape  ),
                'send_starts': tuple( send_starts ),
                'recv_starts': tuple( recv_starts )}

        return info

    #---------------------------------------------------------------------------
    def __del__(self):

        # Destroy sub-communicators
        for s in self._subcomm:
            s.Free()

        # Destroy Cartesian communicator
        self._comm_cart.Free()
