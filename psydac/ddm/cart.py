# coding: utf-8

import numpy as np
from itertools import product
from mpi4py    import MPI

from psydac.ddm.partition import mpi_compute_dims

__all__ = ['find_mpi_type', 'CartDecomposition', 'CartDataExchanger']

#===============================================================================
def find_mpi_type( dtype ):
    """
    Find correct MPI datatype that corresponds to user-provided datatype.

    Parameters
    ----------
    dtype : [type | str | numpy.dtype | mpi4py.MPI.Datatype]
        Datatype for which the corresponding MPI datatype is requested.

    Returns
    -------
    mpi_type : mpi4py.MPI.Datatype
        MPI datatype to be used for communication.

    """
    if isinstance( dtype, MPI.Datatype ):
        mpi_type = dtype
    else:
        nt = np.dtype( dtype )
        mpi_type = MPI._typedict[nt.char]

    return mpi_type

#===============================================================================
class CartDecomposition():
    """
    Cartesian decomposition of a tensor-product grid of coefficients.
    This is built on top of an MPI communicator with multi-dimensional
    Cartesian topology.

    Parameters
    ----------
    npts : list or tuple of int
        Number of coefficients in the global grid along each dimension.

    pads : list or tuple of int
        Padding along each grid dimension.
        In 1D, this is the number of extra coefficients added at each boundary
        of the local domain to permit non-local operations with compact support;
        this concept extends to multiple dimensions through a tensor product.

    periods : list or tuple of bool
        Periodicity (True|False) along each grid dimension.

    reorder : bool
        Whether individual ranks can be changed in the new Cartesian communicator.

    comm : mpi4py.MPI.Comm
        MPI communicator that will be used to spawn a new Cartesian communicator
        (optional: default is MPI_COMM_WORLD).

    """
    def __init__( self, npts, pads, periods, reorder, comm=MPI.COMM_WORLD, shifts=None, nprocs=None, reverse_axis=None ):

        # Check input arguments
        # TODO: check that arguments are identical across all processes
        assert len( npts ) == len( pads ) == len( periods )
        assert all( n >=1 for n in npts )
        assert all( p >=0 for p in pads )
        assert all( isinstance( period, bool ) for period in periods )
        assert isinstance( reorder, bool )
        assert isinstance( comm, MPI.Comm )

        shifts = tuple(shifts) if shifts else (1,)*len(npts)

        # Store input arguments
        self._npts         = tuple( npts    )
        self._pads         = tuple( pads    )
        self._periods      = tuple( periods )
        self._reduced      = (False,)*len(npts)
        self._shifts       = shifts
        self._reorder      = reorder
        self._comm         = comm

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

        reduced_npts = [(n-1)//m+1 if P else (n-p-1)//m+1 for n,m,p,P in zip(npts, shifts, pads, periods)]

        if nprocs is None:
            nprocs, block_shape = mpi_compute_dims( self._size, reduced_npts, pads )
        else:
            assert len(nprocs) == len(npts)

        self._dims = nprocs
        self._reverse_axis = reverse_axis
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

        if reverse_axis is not None:
            self._coords[reverse_axis] = self._dims[reverse_axis] - self._coords[reverse_axis] - 1

        # Store arrays with all the reduced starts and reduced ends along each direction
        self._reduced_global_starts = [None]*self._ndims
        self._reduced_global_ends   = [None]*self._ndims
        for axis in range( self._ndims ):
            n = reduced_npts[axis]
            d = nprocs[axis]
            p = pads[axis]
            m = shifts[axis]
            P = periods[axis]
            self._reduced_global_starts[axis] = np.array( [( c   *n)//d   for c in range( d )] )
            self._reduced_global_ends  [axis] = np.array( [((c+1)*n)//d-1 for c in range( d )] )
            if not P:self._reduced_global_ends  [axis][0] += p

        # Store arrays with all the starts and ends along each direction
        self._global_starts = [None]*self._ndims
        self._global_ends   = [None]*self._ndims

        for axis in range( self._ndims ):
            n = npts[axis]
            d = nprocs[axis]
            p = pads[axis]
            m = shifts[axis]
            r_starts = self._reduced_global_starts[axis]
            r_ends   = self._reduced_global_ends  [axis]

            global_starts = [0]
            for c in range(1,d):
                global_starts.append(global_starts[c-1] + (r_ends[c-1]-r_starts[c-1]+1)*m)

            global_ends = [global_starts[c+1]-1 for c in range( d-1 )] + [n-1]

            self._global_starts[axis] = np.array( global_starts )
            self._global_ends  [axis] = np.array( global_ends )

        # Start/end values of global indices (without ghost regions)
        self._starts = tuple( self.global_starts[axis][c] for axis,c in zip(range(self._ndims), self._coords) )
        self._ends   = tuple( self.global_ends  [axis][c] for axis,c in zip(range(self._ndims), self._coords) )

        # List of 1D global indices (without ghost regions)
        self._grids = tuple( range(s,e+1) for s,e in zip( self._starts, self._ends ) )

        # Compute shape of local arrays in topology (with ghost regions)
        self._shape = tuple( e-s+1+2*m*p for s,e,p,m in zip( self._starts, self._ends, self._pads, shifts ) )

        # Extended grids with ghost regions
        self._extended_grids = tuple( range(s-m*p,e+m*p+1) for s,e,p,m in zip( self._starts, self._ends, self._pads, shifts ) )

        # Create (N-1)-dimensional communicators within the Cartesian topology
        self._subcomm = [None]*self._ndims
        for i in range(self._ndims):
            remain_dims     = [i==j for j in range( self._ndims )]
            self._subcomm[i] = self._comm_cart.Sub( remain_dims )

        # Compute/store information for communicating with neighbors
        self._shift_info = {}
        for axis in range( self._ndims ):
            for disp in [-1,1]:
                self._shift_info[ axis, disp ] = \
                        self._compute_shift_info( axis, disp )

        self._petsccart     = None
        self._parent_starts = tuple([None]*self._ndims)
        self._parent_ends   = tuple([None]*self._ndims)
    #---------------------------------------------------------------------------
    # Global properties (same for each process)
    #---------------------------------------------------------------------------
    @property
    def ndim( self ):
        return self._ndims

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
    def shifts( self ):
        return self._shifts

    @property
    def reorder( self ):
        return self._reorder

    @property
    def comm( self ):
        return self._comm

    @property
    def comm_cart( self ):
        return self._comm_cart

    @property
    def nprocs( self ):
        return self._dims

    @property
    def reverse_axis(self):
        return self._reverse_axis

    @property
    def global_starts( self ):
        return self._global_starts

    @property
    def global_ends( self ):
        return self._global_ends

    @property
    def reduced_global_starts( self ):
        return self._reduced_global_starts

    @property
    def reduced_global_ends( self ):
        return self._reduced_global_ends

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
    def parent_starts( self ):
        return self._starts

    @property
    def parent_ends( self ):
        return self._parent_ends

    @property
    def reduced( self ):
        return self._reduced

    @property
    def coords( self ):
        return self._coords

    @property
    def shape( self ):
        return self._shape

    @property
    def subcomm( self ):
        return self._subcomm

# NOTE [YG, 09.03.2021]: the equality comparison "==" is removed because we
# prefer using the identity comparison "is" as far as possible.
#    def __eq__( self, a):
#        a = (a.npts, a.pads, a.periods, a.comm)
#        b = (self.npts, self.pads, self.periods, self.comm)
#        return a == b

    #---------------------------------------------------------------------------
    def topetsc( self ):
        """ Convert the cart to a petsc cart.
        """
        if self._petsccart is None:
            from psydac.ddm.petsc import PetscCart
            self._petsccart = PetscCart(self)
        return self._petsccart

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

        reorder = self.reverse_axis == direction
        # Process ranks for data shifting with MPI_SENDRECV
        (rank_source, rank_dest) = self.comm_cart.Shift( direction, disp )

        if reorder:
            (rank_source, rank_dest) = (rank_dest, rank_source)

        # Mesh info info along given direction
        s = self._starts[direction]
        e = self._ends  [direction]
        p = self._pads  [direction]
        m = self._shifts[direction]
        r = self._reduced[direction]

        # Shape of send/recv subarrays
        buf_shape = np.array( self._shape )
        buf_shape[direction] = m*p

        # Start location of send/recv subarrays
        send_starts          = np.zeros( self._ndims, dtype=int )
        recv_starts          = np.zeros( self._ndims, dtype=int )
        send_assembly_starts = np.zeros( self._ndims, dtype=int )
        recv_assembly_starts = np.zeros( self._ndims, dtype=int )

        if disp > 0:
            recv_starts[direction]          = 0
            send_starts[direction]          = e-s+1
            recv_assembly_starts[direction] = m*p-p
            send_assembly_starts[direction] = e-s+1+m*p
        elif disp < 0:
            recv_starts[direction]          = e-s+1+m*p
            send_starts[direction]          = m*p
            recv_assembly_starts[direction] = e-s+1+m*p
            send_assembly_starts[direction] = m*p-p

        # Store all information into dictionary
        info = {'rank_dest'           : rank_dest,
                'rank_source'         : rank_source,
                'buf_shape'           : tuple(  buf_shape  ),
                'send_starts'         : tuple( send_starts ),
                'recv_starts'         : tuple( recv_starts ),
                'send_assembly_starts': tuple( send_assembly_starts ),
                'recv_assembly_starts': tuple( recv_assembly_starts )}

        return info
        
    def reduce_elements( self, axes, n_elements):
        """ Compute the cart of the reduced space.

        Parameters
        ----------
        axes: tuple_like (int)
            The directions to be Reduced.

        n_elements: tuple_like (int)
            Number of elements to substract from the space.

        Returns
        -------
        v: CartDecomposition
            The reduced cart.
        """

        if isinstance(axes, int):
            axes = [axes]

        cart = CartDecomposition(self._npts, self._pads, self._periods, self._reorder, shifts=self.shifts, reverse_axis=self.reverse_axis)

        cart._dims      = self._dims
        cart._comm_cart = self._comm_cart
        cart._coords    = self._coords

        coords          = cart.coords
        nprocs          = cart.nprocs

        cart._shifts = [max(1,m-1) for m in self.shifts]

        for axis in axes:assert(axis<cart._ndims)

        # set pads and npts
        cart._npts    = tuple(n - ne for n,ne in zip(cart.npts, n_elements))
        cart._reduced = tuple(a in axes for a in range(self._ndims))

        # Store arrays with all the starts and ends along each direction
        cart._global_starts = [None]*self._ndims
        cart._global_ends   = [None]*self._ndims
        for axis in range( self._ndims ):
            n = cart._npts[axis]
            d = nprocs[axis]
            m = cart._shifts[axis]
            r_starts = cart._reduced_global_starts[axis]
            r_ends   = cart._reduced_global_ends  [axis]

            global_starts = [0]
            for c in range(1,d):
                global_starts.append(global_starts[c-1] + (r_ends[c-1]-r_starts[c-1]+1)*m)

            global_ends = [global_starts[c+1]-1 for c in range( d-1 )] + [n-1]

            cart._global_starts[axis] = np.array( global_starts )
            cart._global_ends  [axis] = np.array( global_ends )

        # Start/end values of global indices (without ghost regions)
        cart._starts = tuple( cart.global_starts[axis][c] for axis,c in zip(range(self._ndims), self._coords) )
        cart._ends   = tuple( cart.global_ends  [axis][c] for axis,c in zip(range(self._ndims), self._coords) )

        # List of 1D global indices (without ghost regions)
        cart._grids = tuple( range(s,e+1) for s,e in zip( cart._starts, cart._ends ) )

        # N-dimensional global indices (without ghost regions)
        cart._indices = product( *cart._grids )

        # Compute shape of local arrays in topology (with ghost regions)
        cart._shape = tuple( e-s+1+2*m*p for s,e,p,m in zip( cart._starts, cart._ends, cart._pads, cart._shifts ) )

        # Extended grids with ghost regions
        cart._extended_grids = tuple( range(s-m*p,e+m*p+1) for s,e,p,m in zip( cart._starts, cart._ends, cart._pads, cart._shifts ) )

        # N-dimensional global indices with ghost regions
        cart._extended_indices = product( *cart._extended_grids )

        # Create (N-1)-dimensional communicators within the cartsian topology
        cart._subcomm = [None]*cart._ndims
        for i in range(cart._ndims):
            remain_dims      = [i==j for j in range( cart._ndims )]
            cart._subcomm[i] = cart._comm_cart.Sub( remain_dims )

        # Compute/store information for communicating with neighbors
        cart._shift_info = {}
        for axis in range( cart._ndims ):
            for disp in [-1,1]:
                cart._shift_info[ axis, disp ] = \
                        cart._compute_shift_info( axis, disp )

        # Store arrays with all the reduced starts and reduced ends along each direction
        cart._reduced_global_starts = [None]*self._ndims
        cart._reduced_global_ends   = [None]*self._ndims
        for axis in range( self._ndims ):
            cart._reduced_global_starts[axis] = self._reduced_global_starts[axis].copy()
            cart._reduced_global_ends  [axis] = self._reduced_global_ends  [axis].copy()

            # adjust only the end of the last interval
            if not cart.periods[axis]:
                n = cart._npts[axis]
                cart._reduced_global_ends[axis][-1] = n-1

        cart._parent_starts = self.starts
        cart._parent_ends   = self.ends
        return cart

    def reduce_grid(self, global_starts, global_ends):
        """ 
        Returns a new CartDecomposition object with a coarser grid from the original one
        we do that by giving a new global_starts  and global_ends of the coefficients
        in each dimension.
            
        Parameters
        ----------
        global_starts : list/tuple
            the list of the new global_starts  in each dimesion.

        global_ends : list/tuple
            the list of the new global_ends in each dimesion.
 
        """
        # Make a copy
        cart = CartDecomposition(self.npts, self.pads, self.periods, self.reorder, comm=self.comm)

        cart._npts = tuple(end[-1] + 1 for end in global_ends)

        cart._dims = self._dims

        # Create a 2D MPI cart
        cart._comm_cart = self._comm_cart

        # Know my coordinates in the topology
        cart._rank_in_topo = self._rank_in_topo
        cart._coords       = self._coords

        # Start/end values of global indices (without ghost regions)
        cart._starts = tuple( starts[i] for i,starts in zip( self._coords, global_starts) )
        cart._ends   = tuple( ends[i]   for i,ends   in zip( self._coords, global_ends  ) )

        # List of 1D global indices (without ghost regions)
        cart._grids = tuple( range(s,e+1) for s,e in zip( cart._starts, cart._ends ) )

        # N-dimensional global indices (without ghost regions)
        cart._indices = product( *cart._grids )

        # Compute shape of local arrays in topology (with ghost regions)
        cart._shape = tuple( e-s+1+2*p for s,e,p in zip( cart._starts, cart._ends, cart._pads ) )

        # Extended grids with ghost regions
        cart._extended_grids = tuple( range(s-p,e+p+1) for s,e,p in zip( cart._starts, cart._ends, cart._pads ) )

        # N-dimensional global indices with ghost regions
        cart._extended_indices = product( *cart._extended_grids )


        # Compute/store information for communicating with neighbors
        cart._shift_info = {}
        for dimension in range( cart._ndims ):
            for disp in [-1,1]:
                cart._shift_info[ dimension, disp ] = \
                        cart._compute_shift_info( dimension, disp )

        # Store arrays with all the starts and ends along each direction
        cart._global_starts = global_starts
        cart._global_ends   = global_ends

        return cart

#===============================================================================
class CartDataExchanger:
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
    def __init__( self, cart, dtype, *, coeff_shape=(), assembly=False ):

        self._send_types, self._recv_types = self._create_buffer_types(
                cart, dtype, coeff_shape=coeff_shape )

        self._cart = cart
        self._comm = cart.comm_cart

        if assembly:
            self._assembly_send_types, self._assembly_recv_types = self._create_assembly_buffer_types(
                cart, dtype, coeff_shape=coeff_shape )

    #---------------------------------------------------------------------------
    # Public interface
    #---------------------------------------------------------------------------
    def get_send_type( self, direction, disp ):
        return self._send_types[direction, disp]

    # ...
    def get_recv_type( self, direction, disp ):
        return self._recv_types[direction, disp]

    def get_assembly_send_type( self, direction, disp ):
        return self._assembly_send_types[direction, disp]

    # ...
    def get_assembly_recv_type( self, direction, disp ):
        return self._assembly_recv_types[direction, disp]

    # ...
    def update_ghost_regions( self, array, *, direction=None ):
        """
        Update ghost regions in a numpy array with dimensions compatible with
        CartDecomposition (and coeff_shape) provided at initialization.

        Parameters
        ----------
        array : numpy.ndarray
            Multidimensional array corresponding to local subdomain in
            decomposed tensor grid, including padding.

        direction : int
            Index of dimension over which ghost regions should be updated
            (optional: by default all ghost regions are updated).

        """
        if direction is None:
            for d in range( self._cart.ndim ):
                self.update_ghost_regions( array, direction=d )
            return

        assert isinstance( array, np.ndarray )
        assert isinstance( direction, int )

        # Shortcuts
        cart = self._cart
        comm = self._comm

        # Choose non-negative invertible function tag(disp) >= 0
        # NOTES:
        #   . different values of disp must return different tags!
        #   . tag at receiver must match message tag at sender
        tag = lambda disp: 42+disp

        # Requests' handles
        requests = []

        # Start receiving data (MPI_IRECV)
        for disp in [-1,1]:
            info     = cart.get_shift_info( direction, disp )
            recv_typ = self.get_recv_type ( direction, disp )
            recv_buf = (array, 1, recv_typ)
            recv_req = comm.Irecv( recv_buf, info['rank_source'], tag(disp) )
            requests.append( recv_req )

        # Start sending data (MPI_ISEND)
        for disp in [-1,1]:
            info     = cart.get_shift_info( direction, disp )
            send_typ = self.get_send_type ( direction, disp )
            send_buf = (array, 1, send_typ)
            send_req = comm.Isend( send_buf, info['rank_dest'], tag(disp) )
            requests.append( send_req )

        # Wait for end of data exchange (MPI_WAITALL)
        MPI.Request.Waitall( requests )

        comm.Barrier()

    def update_assembly_ghost_regions( self, array ):
        """
        Update ghost regions after the assembly algorithm in a numpy array with dimensions compatible with
        CartDecomposition (and coeff_shape) provided at initialization.


        Parameters
        ----------
        array : numpy.ndarray
            Multidimensional array corresponding to local subdomain in
            decomposed tensor grid, including padding.

        """

        assert isinstance( array, np.ndarray )


        # Shortcuts
        cart = self._cart
        comm = self._comm
        ndim = cart.ndim


        # Choose non-negative invertible function tag(disp) >= 0
        # NOTES:
        #   . different values of disp must return different tags!
        #   . tag at receiver must match message tag at sender
        tag = lambda disp: 42+disp

        # Requests' handles

        disps = [1 if P else -1 for P in cart.periods]
        for direction in range( ndim ):
            # Start receiving data (MPI_IRECV)
            disp     = disps[direction]
            info     = cart.get_shift_info( direction, disp )
            recv_typ = self.get_assembly_recv_type ( direction, disp )
            recv_buf = (array, 1, recv_typ)
            recv_req = comm.Irecv( recv_buf, info['rank_source'], tag(disp) )

            # Start sending data (MPI_ISEND)
            info     = cart.get_shift_info( direction, disp )
            send_typ = self.get_assembly_send_type ( direction, disp )
            send_buf = (array, 1, send_typ)
            send_req = comm.Isend( send_buf, info['rank_dest'], tag(disp) )

            # Wait for end of data exchange (MPI_WAITALL)
            MPI.Request.Waitall( [recv_req, send_req] )

        for direction in range( ndim ):
            disp     = disps[direction]
            if disp == 1:
                info = cart.get_shift_info( direction, disp )
                pads = [0]*ndim
                pads[direction] = cart._pads[direction]
                idx_from = tuple(slice(s,s+b) for s,b in zip(info['recv_starts'],info['buf_shape']))
                idx_to   = tuple(slice(s+p,s+b+p) for s,b,p in zip(info['recv_starts'],info['buf_shape'],pads))
                array[idx_to] += array[idx_from]
            else:
                info = cart.get_shift_info( direction, disp )
                pads = [0]*ndim
                pads[direction] = cart._pads[direction]
                idx_from = tuple(slice(s,s+b) for s,b in zip(info['recv_starts'],info['buf_shape']))
                idx_to   = tuple(slice(s-p,s+b-p) for s,b,p in zip(info['recv_starts'],info['buf_shape'],pads))
                array[idx_to] += array[idx_from]

        comm.Barrier()

    def update_ghost_regions_all_directions_non_blocking( self, array, disp ):

        """
        Update ghost regions for all directions in a numpy array with dimensions compatible with
        CartDecomposition (and coeff_shape) provided at initialization
        using non blocking communications.
        """
   
        assert isinstance( array, np.ndarray )
        assert disp in [-1,1]
  
        # Shortcuts
        cart = self._cart
        comm = self._comm
        ndim = cart.ndim
 
        tag = lambda disp: 42+disp*(direction+1)

        # Requests' handles
        requests = []
        for direction in range(ndim):
            # Start sending data (MPI_ISEND)
            info     = cart.get_shift_info( direction, disp )
            send_typ = self.get_send_type ( direction, disp )
            send_buf = (array, 1, send_typ)
            send_req = comm.Isend( send_buf, info['rank_dest'], tag(disp) )
            requests.append( send_req )

            # Start receiving data (MPI_IRECV)
            info     = cart.get_shift_info( direction, disp )
            recv_typ = self.get_recv_type ( direction, disp )
            recv_buf = (array, 1, recv_typ)
            recv_req = comm.Irecv( recv_buf, info['rank_source'], tag(disp) )
            requests.append( recv_req )

            # Wait for end of data exchange (MPI_WAITALL)
            MPI.Request.Waitall( requests )

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

        for direction in range( cart.ndim ):
            for disp in [-1, 1]:
                info = cart.get_shift_info( direction, disp )

                buf_shape   = list( info[ 'buf_shape' ] ) + coeff_shape
                send_starts = list( info['send_starts'] ) + coeff_start
                recv_starts = list( info['recv_starts'] ) + coeff_start

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

    @staticmethod
    def _create_assembly_buffer_types( cart, dtype, *, coeff_shape=() ):
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

        for direction in range( cart.ndim ):
            for disp in [-1, 1]:
                info = cart.get_shift_info( direction, disp )

                buf_shape   = list( info[ 'buf_shape' ] ) + coeff_shape
                send_starts = list( info['send_assembly_starts'] ) + coeff_start
                recv_starts = list( info['recv_assembly_starts'] ) + coeff_start

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
