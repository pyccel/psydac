import numpy as np
from itertools import product
from mpi4py    import MPI

from spl.ddm.partition import mpi_compute_dims

#===============================================================================
class Cart():

    def __init__( self, npts, pads, periods, reorder, comm=MPI.COMM_WORLD ):

        # Check input arguments
        # TODO: check that arguments are identical across all processes
        assert len( npts ) == len( pads ) == len( periods )
        assert all( n >=1 for n in npts )
        assert all( p >=0 for p in pads )
        assert all( isinstance( period, bool ) for period in periods )
        assert isinstance( reorder, bool )
        assert isinstance( comm, MPI.Comm )

        # Store input arguments
        self._npts    = tuple( npts    )
        self._pads    = tuple( pads    )
        self._periods = tuple( periods )
        self._reorder = reorder
        self._comm    = comm

        # ...
        self._ndims = len( npts )
        # ...

        # ...
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()
        # ...

        # ...
        # Know the number of processes along each direction
#        self._dims = MPI.Compute_dims( self._size, self._ndims )
        mpi_dims, block_shape = mpi_compute_dims( self._size, npts, pads )
        self._dims = mpi_dims
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

        # Compute/store information for communicating with neighbors
        self._shift_info = {}
        for dimension in range( self._ndims ):
            for disp in [-1,1]:
                self._shift_info[ dimension, disp ] = \
                        self._compute_shift_info( dimension, disp )

    #---------------------------------------------------------------------------
    # Global properties (same for each process)
    #---------------------------------------------------------------------------
    @property
    def npts( self ):
        return self._npts

    @property
    def pads( self ):
        return self._pads

    @property
    def periods( self ):
        return self._periods

    @property
    def reorder( self ):
        return self._reorder

    @property
    def comm_cart( self ):
        return self._comm_cart

    @property
    def nprocs( self ):
        return self._dims

    #---------------------------------------------------------------------------
    # Local properties
    #---------------------------------------------------------------------------
    @property
    def starts( self ):
        return self._starts

    @property
    def ends( self ):
        return self._ends

    @property
    def coords( self ):
        return self._coords

    @property
    def shape( self ):
        return self._shape

    @property
    def subcomm( self ):
        return self._subcomm

    #---------------------------------------------------------------------------
    def coords_exist( self, coords ):

        return all( P or (0 <= c < d) for P,c,d in zip( self._periods, coords, self._dims ) )

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
