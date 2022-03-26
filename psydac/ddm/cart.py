# coding: utf-8

import itertools
import numpy as np
from itertools import product
from mpi4py    import MPI

from psydac.ddm.partition import compute_dims, partition_procs_per_patch


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

#====================================================================================
class MultiCartDecomposition():
    def __init__( self, npts, pads, periods, reorder, comm=MPI.COMM_WORLD, shifts=None, num_threads=None ):

        # Check input arguments
        # TODO: check that arguments are identical across all processes
        assert len( npts ) == len( pads ) == len( periods )
        assert all(all( n >=1 for n in npts_i ) for npts_i in npts)
        assert isinstance( comm, MPI.Comm )

        shifts      = tuple(shifts) if shifts else [(1,)*len(npts[0]) for n in npts]
        num_threads = num_threads if num_threads else 1

        # Store input arguments
        self._npts         = tuple( npts    )
        self._pads         = tuple( pads    )
        self._periods      = tuple( periods )
        self._shifts       = tuple( shifts  )
        self._num_threads  = num_threads
        self._comm         = comm

        # ...
        self._ncarts = len( npts )

        # ...
        size           = comm.Get_size()
        rank           = comm.Get_rank()
        sizes, rank_ranges = partition_procs_per_patch(self._npts, size)

        self._rank   = rank
        self._size   = size
        self._sizes  = sizes
        self._rank_ranges = rank_ranges

        global_group = comm.group
        owned_groups = []

        local_groups        = [None]*self._ncarts
        local_communicators = [None]*self._ncarts

        for i,r in enumerate(rank_ranges):
            if rank>=r[0] and rank<=r[1]:
                local_groups[i]        = global_group.Range_incl([[r[0], r[1], 1]])
                local_communicators[i] = comm.Create_group(local_groups[i], i)
                owned_groups.append(i)

        try:
            carts = [CartDecomposition(n, p, P, reorder, comm=sub_comm, shifts=s, num_threads=num_threads) if sub_comm else None\
                    for n,p,P,sub_comm,s in zip(npts, pads, periods, local_communicators, shifts)]
        except:
            comm.Abort(1)

        self._local_groups        = local_groups
        self._local_communicators = local_communicators
        self._owned_groups        = owned_groups
        self._carts               = carts

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
    def num_threads( self ):
        return self._num_threads

    @property
    def comm( self ):
        return self._comm

    @property
    def ncarts( self ):
        return self._ncarts

    @property
    def size( self ):
        return self._size

    @property
    def rank( self ):
        return self._rank

    @property
    def sizes( self ):
        return self._sizes

    @property
    def rank_ranges( self ):
        return self._rank_ranges

    @property
    def local_groups( self ):
        return self._local_groups

    @property
    def local_communicators( self ):
        return self._local_communicators

    @property
    def owned_groups( self ):
        return self._owned_groups

    @property
    def carts( self ):
        return self._carts

class InterfacesCartDecomposition:
    def __init__(self, carts, interfaces):

        assert isinstance(carts, MultiCartDecomposition)

        npts                = carts.npts
        pads                = carts.pads
        shifts              = carts.shifts
        periods             = carts.periods
        num_threads         = carts.num_threads
        comm                = carts.comm
        global_group        = comm.group
        local_groups        = carts.local_groups
        rank_ranges         = carts.rank_ranges
        local_communicators = carts.local_communicators

        interfaces_groups     = {}
        interfaces_comm       = {}
        interfaces_root_ranks = {}
        interfaces_carts      = {}

        for i,j in interfaces:
            if i in owned_groups or j in owned_groups:
                if not local_groups[i]:
                    local_groups[i] = global_group.Range_incl([[rank_ranges[i][0], rank_ranges[i][1], 1]])
                if not local_groups[j]:
                    local_groups[j] = global_group.Range_incl([[rank_ranges[j][0], rank_ranges[j][1], 1]])

                interfaces_groups[i,j] = local_groups[i].Union(local_groups[i], local_groups[j])
                interfaces_comm  [i,j] = comm.Create_group(interfaces_groups[i,j])
                root_rank_i            = interfaces_groups[i,j].Translate_ranks(local_groups[i], [0], interfaces_groups[i,j])[0]
                root_rank_j            = interfaces_groups[i,j].Translate_ranks(local_groups[j], [0], interfaces_groups[i,j])[0]
                interfaces_root_ranks[i,j] = [root_rank_i, root_rank_j]

        tag   = lambda i,j,disp: (2+disp)*(i+j)
        dtype = find_mpi_type('int64')
        for i,j in interfaces:
            if (i,j) in interfaces_comm:
                ranks_in_topo_i = carts.carts[i].ranks_in_topo if i in owned_groups else np.full(local_groups[i].size, -1)
                ranks_in_topo_j = carts.carts[j].ranks_in_topo if j in owned_groups else np.full(local_groups[j].size, -1)
                req = []
                if interfaces_comm[i,j].rank == interfaces_root_ranks[i,j][0]:
                    req.append(interfaces_comm[i,j].Isend((ranks_in_topo_i, ranks_in_topo_i.size, dtype), interfaces_root_ranks[i,j][1], tag=tag(i,j,1)))
                    req.append(interfaces_comm[i,j].Irecv((ranks_in_topo_j, ranks_in_topo_j.size, dtype), interfaces_root_ranks[i,j][1], tag=tag(i,j,-1)))

                if interfaces_comm[i,j].rank == interfaces_root_ranks[i,j][1]:
                    req.append(interfaces_comm[i,j].Isend((ranks_in_topo_j, ranks_in_topo_j.size, dtype), interfaces_root_ranks[i,j][0], tag=tag(i,j,-1)))
                    req.append(interfaces_comm[i,j].Irecv((ranks_in_topo_i, ranks_in_topo_i.size, dtype), interfaces_root_ranks[i,j][0], tag=tag(i,j,1)))                  

                axes = interfaces[i,j][0]
                exts = interfaces[i,j][1]
                interfaces_carts[i,j] = InterfaceCartDecomposition(npts=[npts[i], npts[j]],
                                                                   pads=[pads[i], pads[j]],
                                                                   periods=[periods[i], periods[j]],
                                                                   comm=interfaces_comm[i,j],
                                                                   shifts=[shifts[i], shifts[j]],
                                                                   axes=axes, exts=exts, 
                                                                   ranks_in_topo=[ranks_in_topo_i, ranks_in_topo_j],
                                                                   local_groups=[local_groups[i], local_groups[j]],
                                                                   local_communicators=[local_communicators[i], local_communicators[j]],
                                                                   root_ranks=interfaces_root_ranks[i,j],
                                                                   requests=req,
                                                                   num_threads=num_threads)

        self._interfaces_groups = interfaces_groups
        self._interfaces_comm   = interfaces_comm
        self._interfaces_carts = interfaces_carts

    @property
    def interfaces_carts( self ):
        return self._interfaces_carts

    @property
    def interfaces_comm( self ):
        return self._interfaces_comm

    @property
    def interfaces_groups( self ):
        return self._interfaces_groups

#===============================================================================
class CartDecomposition():
    """
    Cartesian decomposition of a tensor-product grid of spline coefficients.
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

    shifts: list or tuple of int
        Shifts along each grid dimension.
        It takes values bigger or equal to one, it represents the multiplicity of each knot.

    nprocs: list or tuple of int
       MPI decomposition along each dimension.

    reverse_axis: int
       Reverse the ownership of the processes along the specified axis.

    """
    def __init__( self, npts, pads, periods, reorder, comm=None, shifts=None, nprocs=None, reverse_axis=None, num_threads=None ):

        # Check input arguments
        # TODO: check that arguments are identical across all processes
        assert len( npts ) == len( pads ) == len( periods )
        assert all( n >=1 for n in npts )
        assert all( p >=0 for p in pads )
        assert all( isinstance( period, bool ) for period in periods )
        assert isinstance( reorder, bool )
        assert isinstance( comm, MPI.Comm )

        shifts      = tuple(shifts) if shifts else (1,)*len(npts)
        num_threads = num_threads if num_threads else 1

        # Store input arguments
        self._npts         = tuple( npts    )
        self._pads         = tuple( pads    )
        self._periods      = tuple( periods )
        self._shifts       = shifts
        self._num_threads  = num_threads
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

        reduced_npts = [(n-p-1)//m if m>1 else n if not P else n for n,m,p,P in zip(npts, shifts, pads, periods)]

        if nprocs is None:
            nprocs, block_shape = compute_dims( self._size, reduced_npts, pads )
        else:
            assert len(nprocs) == len(npts)

        assert np.product(nprocs) == self._size

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
        self._rank_in_topo  = self._comm_cart.Get_rank()
        self._coords        = self._comm_cart.Get_coords( rank=self._rank_in_topo )
        self._ranks_in_topo = np.array(comm.group.Translate_ranks(self._comm_cart.group, list(range(self._comm_cart.size)), comm.group))

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
            self._reduced_global_starts[axis] = np.array( [( c   *n)//d   for c in range( d )] )
            self._reduced_global_ends  [axis] = np.array( [((c+1)*n)//d-1 for c in range( d )] )
            if m>1:self._reduced_global_ends  [axis][-1] += p+1

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
    def num_threads( self ):
        return self._num_threads

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

    @property
    def ranks_in_topo( self ):
        return self._ranks_in_topo

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
    def get_shared_memory_subdivision( self, shape ):

        assert len(shape) == self._ndims

        try:
            nthreads , block_shape = compute_dims( self._num_threads, shape , [2*p for p in self._pads])
        except ValueError:
            print("Cannot compute dimensions with given input values!")
            self.comm.Abort(1)

        # compute the coords for all threads
        coords_from_rank = np.array([np.unravel_index(rank, nthreads) for rank in range(self._num_threads)])
        rank_from_coords = np.zeros([n+1 for n in nthreads], dtype=int)
        for r in range(self._num_threads):
            c = coords_from_rank[r]
            rank_from_coords[tuple(c)] = r

        # rank_from_coords is not used in the current version of the assembly code
        # it's used in the commented second version, where we don't use a global barrier, but needs more checks to work

        for i in range(self._ndims):
            ind = [slice(None,None)]*self._ndims
            ind[i] = nthreads[i]
            rank_from_coords[tuple(ind)] = self._num_threads

        # Store arrays with all the starts and ends along each direction for every thread
        thread_global_starts = [None]*self._ndims
        thread_global_ends   = [None]*self._ndims
        for axis in range( self._ndims ):
            n = shape[axis]
            d = nthreads[axis]
            thread_global_starts[axis] = np.array( [( c   *n)//d   for c in range( d )] )
            thread_global_ends  [axis] = np.array( [((c+1)*n)//d-1 for c in range( d )] )

        return coords_from_rank, rank_from_coords, thread_global_starts, thread_global_ends, self._num_threads

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

        # Shape of send/recv subarrays
        buf_shape = np.array( self._shape )
        buf_shape[direction] = m*p

        # Start location of send/recv subarrays
        send_starts = np.zeros( self._ndims, dtype=int )
        recv_starts = np.zeros( self._ndims, dtype=int )
        if disp > 0:
            recv_starts[direction] = 0
            send_starts[direction] = e-s+1
        elif disp < 0:
            recv_starts[direction] = e-s+1+m*p
            send_starts[direction] = m*p

        # Store all information into dictionary
        info = {'rank_dest'  : rank_dest,
                'rank_source': rank_source,
                'buf_shape'  : tuple(  buf_shape  ),
                'send_starts': tuple( send_starts ),
                'recv_starts': tuple( recv_starts )}
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
        cart._npts = tuple(n - ne for n,ne in zip(cart.npts, n_elements))

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

class InterfaceCartDecomposition(CartDecomposition):

    def __init__(self, npts, pads, periods, comm, shifts, axes, exts, ranks_in_topo, local_groups, local_communicators, root_ranks, requests, num_threads):

        npts_i, npts_j       = npts
        pads_i, pads_j       = pads
        periods_i, periods_j = periods
        shifts_i, shifts_j   = shifts
        axis_i, axis_j       = axes
        ext_i,ext_j          = exts
        size_i, size_j       = len(ranks_in_topo[0]), len(ranks_in_topo[1])

        assert axis_i == axis_j

        root_rank_i, root_rank_i         = root_ranks
        local_comm_i, local_comm_j       = local_communicators
        ranks_in_topo_i, ranks_in_topo_j = ranks_in_topo

        self._ndims         = len( npts_i )
        self._npts_i        = npts_i
        self._npts_j        = npts_j
        self._pads_i        = pads_i
        self._pads_j        = pads_j
        self._shifts_i      = shifts_i
        self._shifts_j      = shifts_j
        self._axis          = axis_i
        self._ext_i         = ext_i
        self._ext_j         = ext_j
        self._comm          = comm
        self._local_comm_i  = local_communicators[0]
        self._local_comm_j  = local_communicators[1]
        self._local_rank_i  = None
        self._local_rank_j  = None

        if self._local_comm_i:
            self._local_rank_i = self._local_comm_i.rank

        if self._local_comm_j:
            self._local_rank_j = self._local_comm_j.rank

        reduced_npts_i = [(n-p-1)//m if m>1 else n if not P else n for n,m,p,P in zip(npts_i, shifts_i, pads_i, periods_i)]
        reduced_npts_j = [(n-p-1)//m if m>1 else n if not P else n for n,m,p,P in zip(npts_j, shifts_j, pads_j, periods_j)]

        nprocs_i, block_shape_i = compute_dims( size_i, reduced_npts_i, pads_i )
        nprocs_j, block_shape_j = compute_dims( size_j, reduced_npts_j, pads_j )

        dtype = find_mpi_type('int64')
        MPI.Request.Waitall(requests)

        if local_comm_i:
            local_comm_i.Bcast((ranks_in_topo_j,ranks_in_topo_j.size, dtype), root=0)

        if local_comm_j:
            local_comm_j.Bcast((ranks_in_topo_i,ranks_in_topo_i.size, dtype), root=0)

        self._coords_from_rank_i = np.array([np.unravel_index(rank, nprocs_i) for rank in range(size_i)])
        self._coords_from_rank_j = np.array([np.unravel_index(rank, nprocs_j) for rank in range(size_j)])

        rank_from_coords_i = np.zeros(nprocs_i, dtype=int)
        rank_from_coords_j = np.zeros(nprocs_j, dtype=int)

        for r in range(size_i):
            rank_from_coords_i[tuple(self._coords_from_rank_i[r])] = r

        for r in range(size_j):
            rank_from_coords_j[tuple(self._coords_from_rank_j[r])] = r

        index_i          = [slice(None, None)]*len(npts_i)
        index_j          = [slice(None, None)]*len(npts_i)
        index_i[axis_i]  = 0 if ext_i == -1 else -1
        index_j[axis_j]  = 0 if ext_j == -1 else -1

        self._boundary_ranks_i = rank_from_coords_i[tuple(index_i)]
        self._boundary_ranks_j = rank_from_coords_j[tuple(index_j)]

        boundary_group_i = local_groups[0].Incl(self._boundary_ranks_i)
        boundary_group_j = local_groups[1].Incl(self._boundary_ranks_j)

        comm_i = comm.Create_group(boundary_group_i)
        comm_j = comm.Create_group(boundary_group_j)

        root_i = comm.group.Translate_ranks(boundary_group_i, [0], comm.group)[0]
        root_j = comm.group.Translate_ranks(boundary_group_j, [0], comm.group)[0]

        procs_index_i = boundary_group_i.Translate_ranks(local_groups[0], self._boundary_ranks_i, boundary_group_i)
        procs_index_j = boundary_group_j.Translate_ranks(local_groups[1], self._boundary_ranks_j, boundary_group_j)

        # Reorder procs ranks from 0 to local_group.size-1
        self._boundary_ranks_i = self._boundary_ranks_i[procs_index_i]
        self._boundary_ranks_j = self._boundary_ranks_j[procs_index_j]
        
        if root_i != root_j:
            if not comm_i == MPI.COMM_NULL:
                intercomm = comm_i.Create_intercomm(0, comm, root_j)
                

            elif not comm_j == MPI.COMM_NULL:
                intercomm = comm_j.Create_intercomm(0, comm, root_i)
            else:
                intercomm = None
        else:
            intercomm = None

        self._intercomm = intercomm

        # Store arrays with all the reduced starts and reduced ends along each direction
        self._reduced_global_starts_i = [None]*self._ndims
        self._reduced_global_ends_i   = [None]*self._ndims
        self._reduced_global_starts_j = [None]*self._ndims
        self._reduced_global_ends_j   = [None]*self._ndims
        for axis in range( self._ndims ):
            ni = reduced_npts_i[axis]
            di = nprocs_i[axis]
            pi = pads_i[axis]
            mi = shifts_i[axis]
            nj = reduced_npts_j[axis]
            dj = nprocs_j[axis]
            pj = pads_j[axis]
            mj = shifts_j[axis]

            self._reduced_global_starts_i[axis] = np.array( [( ci   *ni)//di   for ci in range( di )] )
            self._reduced_global_ends_i  [axis] = np.array( [((ci+1)*ni)//di-1 for ci in range( di )] )
            self._reduced_global_starts_j[axis] = np.array( [( cj   *nj)//dj   for cj in range( dj )] )
            self._reduced_global_ends_j  [axis] = np.array( [((cj+1)*nj)//dj-1 for cj in range( dj )] )
            if mi>1:self._reduced_global_ends_i [axis][-1] += pi+1
            if mj>1:self._reduced_global_ends_j [axis][-1] += pj+1

        # Store arrays with all the starts and ends along each direction
        self._global_starts_i = [None]*self._ndims
        self._global_ends_i   = [None]*self._ndims
        self._global_starts_j = [None]*self._ndims
        self._global_ends_j   = [None]*self._ndims

        for axis in range( self._ndims ):
            ni = npts_i[axis]
            di = nprocs_i[axis]
            pi = pads_i[axis]
            mi = shifts_i[axis]
            r_starts_i = self._reduced_global_starts_i[axis]
            r_ends_i   = self._reduced_global_ends_i  [axis]
            nj = npts_j[axis]
            dj = nprocs_j[axis]
            pj = pads_j[axis]
            mj = shifts_j[axis]
            r_starts_j = self._reduced_global_starts_j[axis]
            r_ends_j   = self._reduced_global_ends_j  [axis]

            global_starts_i = [0]
            for ci in range(1,di):
                global_starts_i.append(global_starts_i[ci-1] + (r_ends_i[ci-1]-r_starts_i[ci-1]+1)*mi)

            global_starts_j = [0]
            for cj in range(1,dj):
                global_starts_j.append(global_starts_j[cj-1] + (r_ends_j[cj-1]-r_starts_j[cj-1]+1)*mj)

            global_ends_i   = [global_starts_i[ci+1]-1 for ci in range( di-1 )] + [ni-1]
            global_ends_j   = [global_starts_j[cj+1]-1 for cj in range( dj-1 )] + [nj-1]

            self._global_starts_i[axis] = np.array( global_starts_i )
            self._global_ends_i  [axis] = np.array( global_ends_i )
            self._global_starts_j[axis] = np.array( global_starts_j )
            self._global_ends_j  [axis] = np.array( global_ends_j )

        self._communication_infos = {}
        self._communication_infos[self._axis] = self._compute_communication_infos(self._axis)


    def get_communication_infos( self, axis ):
        return self._communication_infos[ axis ]

    #---------------------------------------------------------------------------
    def _compute_communication_infos( self, axis ):

        if self._intercomm == MPI.COMM_NULL:
            return 

        # Mesh info
        npts_i   = self._npts_i
        npts_j   = self._npts_j
        pads_i   = self._pads_i
        pads_j   = self._pads_j
        shifts_i = self._shifts_i
        shifts_j = self._shifts_j
        indices  = []
        if self._local_rank_i is not None:
            rank_i = self._local_rank_i
            coords = self._coords_from_rank_i[rank_i]
            starts = [self._global_starts_i[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_i[d][c] for d,c in enumerate(coords)]
            send_buf_shape        = [e-s+1 for s,e,p,m in zip(starts, ends, pads_i, shifts_i)]
#            send_buf_shape[axis] = pads_i[axis]+1
            send_starts     = [m*p for m,p in zip(shifts_i, pads_i)]
            send_shape      = [e-s+1+2*m*p for s,e,m,p in zip(starts, ends, shifts_i, pads_i)]

            # ...
            coords = self._coords_from_rank_j[self._boundary_ranks_j[0]]
            starts = [self._global_starts_j[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_j[d][c] for d,c in enumerate(coords)]

            recv_buf_shape       = [1]*len(starts)
            recv_buf_shape[axis] = ends[axis]-starts[axis]+1
            recv_shape           = [n+2*m*p for n,m,p in zip(npts_j, shifts_j, pads_j)]
            recv_shape[axis]     = ends[axis]-starts[axis]+1 + 2*shifts_j[axis]*pads_j[axis]
            recv_starts          = [m*p for m,p in zip(shifts_j, pads_j)]

            displacements   = [0]*(len(self._boundary_ranks_j)+1)
            recv_counts     = [None]*len(self._boundary_ranks_j)
            for k,b in enumerate(self._boundary_ranks_j):
                coords = self._coords_from_rank_j[b]
                starts = [self._global_starts_j[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_j[d][c] for d,c in enumerate(coords)]
                shape_k = [e-s+1 for s,e in zip(starts, ends)]
                recv_counts[k] = np.product(shape_k)
                ranges       = [(s+p*m, p*m+e+1) for s,e,p,m in zip(starts, ends, pads_j, shifts_j)]
                ranges[axis] = (shifts_j[axis]*pads_j[axis], shifts_j[axis]*pads_j[axis]+shape_k[axis])
                indices     += [np.ravel_multi_index( ii, dims=recv_shape, order='C' ) for ii in itertools.product(*[range(*a) for a in ranges])] 

        elif self._local_rank_j is not None:
            rank_j = self._local_rank_j
            coords = self._coords_from_rank_j[rank_j]
            starts = [self._global_starts_j[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_j[d][c] for d,c in enumerate(coords)]
            send_buf_shape        = [e-s+1 for s,e,p,m in zip(starts, ends, pads_j, shifts_j)]
#            send_buf_shape[axis] = pads_j[axis]+1
            send_starts     = [m*p for m,p in zip(shifts_j, pads_j)]
            send_shape      = [e-s+1+2*m*p for s,e,m,p in zip(starts, ends, shifts_j, pads_j)]

            # ...
            coords = self._coords_from_rank_i[self._boundary_ranks_i[0]]
            starts = [self._global_starts_i[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_i[d][c] for d,c in enumerate(coords)]

            recv_buf_shape       = [1]*len(starts)
            recv_buf_shape[axis] = ends[axis]-starts[axis]+1
            recv_shape           = [n+2*m*p for n,m,p in zip(npts_i, shifts_i, pads_i)]
            recv_shape[axis]     = ends[axis]-starts[axis]+1 + 2*shifts_i[axis]*pads_i[axis]
            recv_starts          = [m*p for m,p in zip(shifts_i, pads_i)]

            displacements   = [0]*(len(self._boundary_ranks_i)+1)
            recv_counts     = [None]*len(self._boundary_ranks_i)
            for k,b in enumerate(self._boundary_ranks_i):
                coords = self._coords_from_rank_i[b]
                starts = [self._global_starts_i[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_i[d][c] for d,c in enumerate(coords)]
                shape_k = [e-s+1 for s,e in zip(starts, ends)]
                recv_counts[k] = np.product(shape_k)
                ranges       = [(s+p*m, p*m+e+1) for s,e,p,m in zip(starts, ends, pads_i, shifts_i)]
                ranges[axis] = (shifts_j[axis]*pads_j[axis], shifts_j[axis]*pads_j[axis]+shape_k[axis])
                indices     += [np.ravel_multi_index( ii, dims=recv_shape, order='C' ) for ii in itertools.product(*[range(*a) for a in ranges])] 

        displacements[1:] = np.cumsum(recv_counts)
        global_indices    = list(range(displacements[-1]))
        zeros_indices     = np.setdiff1d(global_indices, indices)
        indices           = indices
        zeros_indices     = np.sort(zeros_indices)
        # Store all information into dictionary
        info = {'send_buf_shape' : tuple( send_buf_shape ),
                'send_starts'    : tuple( send_starts ),
                'send_shape'     : tuple( send_shape  ),
                'displacements'  : tuple( displacements ),
                'recv_counts'    : tuple( recv_counts),
                'indices'        : indices,
                'zeros_indices'  : zeros_indices}

        return info

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
    def __init__( self, cart, dtype, *, coeff_shape=() ):

        self._send_types, self._recv_types = self._create_buffer_types(
                cart, dtype, coeff_shape=coeff_shape )

        self._cart = cart
        self._comm = cart.comm_cart

    #---------------------------------------------------------------------------
    # Public interface
    #---------------------------------------------------------------------------
    def get_send_type( self, direction, disp ):
        return self._send_types[direction, disp]

    # ...
    def get_recv_type( self, direction, disp ):
        return self._recv_types[direction, disp]

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

#===============================================================================
class InterfaceCartDataExchanger:

    def __init__(self, cart, dtype):
        args = self._create_buffer_types( cart, dtype )
        self._send_types    = args[0]
        self._recv_types    = args[1]
        self._displacements = args[2]
        self._recv_counts   = args[3]
        self._indices       = args[4]
        self._zeros_indices = args[5]

    # ...
    def update_ghost_regions( self, array_i, array_j ):
        cart          = self._cart
        send_type     = self._send_types
        recv_type     = self._recv_types
        displacements = self._displacements
        recv_counts   = self._recv_counts
        indices       = self._indices
        zeros_indices = self._zeros_indices
        intercomm     = cart.intercomm

        array_i = array_i.ravel()
        array_j = array_j.ravel()

        if cart.local_rank_i is not None:
            intercomm.Allgatherv([array_i, 1, send_type],[array_j, recv_counts, displacements[:-1], recv_type] )
            array_j[indices]       = array_j[:displacements[-1]]
            array_j[zeros_indices] = 0
        elif cart.local_rank_j is not None:
            intercomm.Allgatherv([array_j, 1, send_type],[array_i, recv_counts, displacements[:-1], recv_type] )
            array_i[indices]       = array_i[:displacements[-1]]
            array_i[zeros_indices] = 0

    @staticmethod
    def _create_buffer_types( cart, dtype ):

        assert isinstance( cart, InterfaceCartDecomposition )

        mpi_type = find_mpi_type( dtype )
        info     = cart.get_communication_infos( cart._axis )

        send_data_shape  = list( info['send_shape' ] )
        send_buf_shape   = list( info['send_buf_shape' ] )
        send_starts      = list( info['send_starts'] )

        send_types = mpi_type.Create_subarray(
                     sizes    = send_data_shape ,
                     subsizes =  send_buf_shape ,
                     starts   = send_starts).Commit()

        displacements    = info['displacements']
        recv_counts      = info['recv_counts']
        recv_types       = mpi_type

        return send_types, recv_types, displacements, recv_counts, info['indices'], info['zeros_indices']

