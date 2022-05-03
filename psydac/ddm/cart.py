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
class MultiCartDecomposition:
    """
    This buids the cartition decomposition of multiple grids.
    In each grid we compute a multi-dimensional cartesian topology using MPI sub-communicators.

    Parameters
    ----------
    npts : list or tuple
        Number of coefficients in the global grid along each dimension for each patch.

    pads : list or tuple
        Padding along each grid dimension for each patch.

    periods : list or tuple of bool
        Periodicity (True|False) along each grid dimension for each patch.

    reorder : bool
        Whether individual ranks can be changed in the new Cartesian communicator.

    comm : mpi4py.MPI.Comm
        MPI communicator that will be used to create the sub-communicators for each grid.
        (optional: default is MPI_COMM_WORLD).

    shifts: list or tuple
        Shifts along each grid dimension for each patch.
        It takes values bigger or equal to one, it represents the multiplicity of each knot.

    num_threads: int
       Number of threads for each MPI rank.
    """
    def __init__( self, npts, pads, periods, reorder, comm=MPI.COMM_WORLD, shifts=None, num_threads=None ):

        # Check input arguments
        # TODO: check that arguments are identical across all processes
        assert len( npts ) == len( pads ) == len( periods )
        assert all(all( n >=1 for n in npts_k ) for npts_k in npts)
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
        local_communicators = [MPI.COMM_NULL]*self._ncarts

        for i,r in enumerate(rank_ranges):
            if rank>=r[0] and rank<=r[1]:
                local_groups[i]        = global_group.Range_incl([[r[0], r[1], 1]])
                local_communicators[i] = comm.Create_group(local_groups[i], i)
                owned_groups.append(i)

        try:
            carts = [CartDecomposition(n, p, P, reorder, comm=sub_comm, global_comm=comm, shifts=s, num_threads=num_threads)\
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

#==================================================================================
class InterfacesCartDecomposition:
    """
    This Connects the Cartesian grids when they share an interface.

    Parameters
    ----------

    carts: MultiCartDecomposition
        The cartition decomposition of multiple grids.

    interfaces: dict
        The connectivity of the grids.
        It contains the grids that share an interface along with their axes and extremities.

    """
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
        owned_groups        = carts.owned_groups

        interfaces_groups     = {}
        interfaces_comm       = {}
        interfaces_root_ranks = {}
        interfaces_carts      = {}

        for i,j in interfaces:
            interfaces_comm[i,j] = MPI.COMM_NULL

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
            req = []
            ranks_in_topo_i = None
            ranks_in_topo_j = None
            axes   = interfaces[i,j][0]
            exts   = interfaces[i,j][1]
            if interfaces_comm[i,j] != MPI.COMM_NULL:
                ranks_in_topo_i = carts.carts[i].ranks_in_topo if i in owned_groups else np.full(local_groups[i].size, -1)
                ranks_in_topo_j = carts.carts[j].ranks_in_topo if j in owned_groups else np.full(local_groups[j].size, -1)

                if interfaces_comm[i,j].rank == interfaces_root_ranks[i,j][0]:
                    req.append(interfaces_comm[i,j].Isend((ranks_in_topo_i, ranks_in_topo_i.size, dtype), interfaces_root_ranks[i,j][1], tag=tag(i,j,1)))
                    req.append(interfaces_comm[i,j].Irecv((ranks_in_topo_j, ranks_in_topo_j.size, dtype), interfaces_root_ranks[i,j][1], tag=tag(i,j,-1)))

                if interfaces_comm[i,j].rank == interfaces_root_ranks[i,j][1]:
                    req.append(interfaces_comm[i,j].Isend((ranks_in_topo_j, ranks_in_topo_j.size, dtype), interfaces_root_ranks[i,j][0], tag=tag(i,j,-1)))
                    req.append(interfaces_comm[i,j].Irecv((ranks_in_topo_i, ranks_in_topo_i.size, dtype), interfaces_root_ranks[i,j][0], tag=tag(i,j,1)))

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
        self._carts             = interfaces_carts

    @property
    def carts( self ):
        return self._carts

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

    num_threads: int
       Number of threads for each MPI rank.

    """
    def __init__( self, npts, pads, periods, reorder, comm=None, global_comm=None, shifts=None, nprocs=None, reverse_axis=None, num_threads=None ):

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
        self._local_comm   = comm
        self._global_comm  = comm if global_comm is None else global_comm
        self._reverse_axis = reverse_axis
        # ...
        self._ndims = len( npts )
        # ...

        if comm == MPI.COMM_NULL:
            return

        self._size = comm.Get_size()
        self._rank = comm.Get_rank()
        # ...

        # ...
        # Know the number of processes along each direction
#        self._dims = MPI.Compute_dims( self._size, self._ndims )

        reduced_npts = [(n-p-1)//m if m>1 else n if not P else n for n,m,p,P in zip(npts, shifts, pads, periods)]

        if nprocs is None:
            nprocs, block_shape = compute_dims( self._size, reduced_npts, [p+1 for p in pads] )
        else:
            assert len(nprocs) == len(npts)

        assert np.product(nprocs) == self._size

        self._dims = nprocs
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
    def local_comm( self ):
        return self._local_comm

    @property
    def global_comm( self ):
        return self._global_comm

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
    def is_comm_null( self ):
        return self.comm == MPI.COMM_NULL

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
    def ranks_in_topo( self ):
        return self._ranks_in_topo

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

        for axis in axes:assert(axis<self._ndims)

        cart = CartDecomposition(self._npts, self._pads, self._periods, self._reorder, comm=self.comm, global_comm=self._global_comm, shifts=self.shifts, reverse_axis=self.reverse_axis)

        # set pads and npts
        cart._npts   = tuple(n - ne for n,ne in zip(cart.npts, n_elements))
        cart._shifts = [max(1,m-1) for m in self.shifts]

        if cart.is_comm_null:
            return cart

        cart._dims      = self._dims
        cart._comm_cart = self._comm_cart
        cart._coords    = self._coords

        coords          = cart.coords
        nprocs          = cart.nprocs

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

    #---------------------------------------------------------------------------
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

#===============================================================================
class InterfaceCartDecomposition(CartDecomposition):
    """
    The Cartesian decomposition of an interface constucted from the Cartesian decomposition of the patches that shares an interface.
    This is built using a new inter-communicator between the cartition grids.

    Parameters
    ----------
    npts : list
        Number of coefficients in the global grid along each dimension for the patches that shares the interface.

    pads : list
        Padding along each grid dimension for the patches that shares the interface.

    periods : list or tuple of bool
        Periodicity (True|False) along each grid dimension for the patches that shares the interface.

    comm : mpi4py.MPI.Comm
        MPI communicator that will be used to spawn the cart decomposition

    shifts: list or tuple of int
        Shifts along each grid dimension.
        It takes values bigger or equal to one, it represents the multiplicity of each knot.

    axes: list of ints
        The axes of the patches that constucts the interface.

    exts: list of ints
        The extremities of the patches that constucts the interface.

    ranks_in_topo:
        The ranks of the processes that shares the interface. 

    local_groups: list of MPI.Group
        The groups that constucts the patches that shares the interface.

    local_communicators: list of intra-communicators
        The communicators of the patches that shares the interface.

    root_ranks: list of ints
        The root ranks in the global communicator of the patches.

    requests: list of MPI.Request
        the requests of the communications between the cartesian topologies that constucts the interface.

    num_threads: int
       Number of threads for each MPI rank.

    """
    def __init__(self, npts, pads, periods, comm, shifts, axes, exts, ranks_in_topo, local_groups, local_communicators, root_ranks, requests, num_threads, reduce_elements=False):

        npts_minus, npts_plus       = npts
        pads_minus, pads_plus       = pads
        periods_minus, periods_plus = periods
        shifts_minus, shifts_plus   = shifts
        axis_minus, axis_plus       = axes
        ext_minus, ext_plus         = exts
        size_minus, size_plus       = len(ranks_in_topo[0]), len(ranks_in_topo[1])

        assert axis_minus == axis_plus
        num_threads = num_threads if num_threads else 1

        root_rank_minus, root_rank_plus         = root_ranks
        local_comm_minus, local_comm_plus       = local_communicators
        ranks_in_topo_minus, ranks_in_topo_plus = ranks_in_topo

        self._ndims         = len( npts_minus )
        self._npts_minus    = npts_minus
        self._npts_plus     = npts_plus
        self._pads_minus    = pads_minus
        self._pads_plus     = pads_plus
        self._periods_minus = periods_minus
        self._periods_plus  = periods_plus
        self._shifts_minus  = shifts_minus
        self._shifts_plus   = shifts_plus
        self._axis          = axis_minus
        self._ext_minus     = ext_minus
        self._ext_plus      = ext_plus
        self._comm          = comm
        self._local_comm_minus = local_comm_minus
        self._local_comm_plus  = local_comm_plus
        self._root_rank_minus  = root_rank_minus
        self._root_rank_plus   = root_rank_plus
        self._ranks_in_topo_minus = ranks_in_topo_minus
        self._ranks_in_topo_plus  = ranks_in_topo_plus
        self._local_group_minus = local_groups[0]
        self._local_group_plus  = local_groups[1]
        self._local_rank_minus = None
        self._local_rank_plus  = None
        self._intercomm        = MPI.COMM_NULL
        self._num_threads      = num_threads

        if comm == MPI.COMM_NULL:
            return

        if local_comm_minus != MPI.COMM_NULL :
            self._local_rank_minus = local_comm_minus.rank

        if local_comm_plus != MPI.COMM_NULL:
            self._local_rank_plus = local_comm_plus.rank

        reduced_npts_minus = [(n-p-1)//m if m>1 else n if not P else n for n,m,p,P in zip(npts_minus, shifts_minus, pads_minus, periods_minus)]
        reduced_npts_plus  = [(n-p-1)//m if m>1 else n if not P else n for n,m,p,P in zip(npts_plus, shifts_plus, pads_plus, periods_plus)]

        nprocs_minus, block_shape_minus = compute_dims( size_minus, reduced_npts_minus, pads_minus )
        nprocs_plus, block_shape_plus   = compute_dims( size_plus, reduced_npts_plus, pads_plus )

        self._nprocs_minus = nprocs_minus
        self._nprocs_plus  = nprocs_plus

        if requests:MPI.Request.Waitall(requests)
        dtype = find_mpi_type('int64')
        if local_comm_minus != MPI.COMM_NULL and reduce_elements == False:
            local_comm_minus.Bcast((ranks_in_topo_plus,ranks_in_topo_plus.size, dtype), root=0)

        if local_comm_plus != MPI.COMM_NULL and reduce_elements == False:
            local_comm_plus.Bcast((ranks_in_topo_minus,ranks_in_topo_minus.size, dtype), root=0)

        self._coords_from_rank_minus = np.array([np.unravel_index(rank, nprocs_minus) for rank in range(size_minus)])
        self._coords_from_rank_plus  = np.array([np.unravel_index(rank, nprocs_plus)  for rank in range(size_plus)])

        rank_from_coords_minus = np.zeros(nprocs_minus, dtype=int)
        rank_from_coords_plus  = np.zeros(nprocs_plus, dtype=int)

        for r in range(size_minus):
            rank_from_coords_minus[tuple(self._coords_from_rank_minus[r])] = r

        for r in range(size_plus):
            rank_from_coords_plus[tuple(self._coords_from_rank_plus[r])] = r

        index_minus      = [slice(None, None)]*len(npts_minus)
        index_plus       = [slice(None, None)]*len(npts_minus)
        index_minus[axis_minus] = 0 if ext_minus == -1 else -1
        index_plus[axis_plus]   = 0 if ext_plus  == -1 else -1

        self._boundary_ranks_minus = rank_from_coords_minus[tuple(index_minus)].ravel()
        self._boundary_ranks_plus  = rank_from_coords_plus[tuple(index_plus)].ravel()

        boundary_group_minus = local_groups[0].Incl(self._boundary_ranks_minus)
        boundary_group_plus  = local_groups[1].Incl(self._boundary_ranks_plus)

        comm_minus = comm.Create_group(boundary_group_minus)
        comm_plus  = comm.Create_group(boundary_group_plus)

        root_minus = comm.group.Translate_ranks(boundary_group_minus, [0], comm.group)[0]
        root_plus  = comm.group.Translate_ranks(boundary_group_plus, [0], comm.group)[0]

        procs_index_minus = boundary_group_minus.Translate_ranks(local_groups[0], self._boundary_ranks_minus, boundary_group_minus)
        procs_index_plus  = boundary_group_plus.Translate_ranks(local_groups[1],  self._boundary_ranks_plus,  boundary_group_plus)

        # Reorder procs ranks from 0 to local_group.size-1
        self._boundary_ranks_minus = self._boundary_ranks_minus[procs_index_minus]
        self._boundary_ranks_plus  = self._boundary_ranks_plus[procs_index_plus]

        if root_minus != root_plus:
            if not comm_minus == MPI.COMM_NULL:
                self._intercomm = comm_minus.Create_intercomm(0, comm, root_plus)
                self._local_comm = comm_minus

            elif not comm_plus == MPI.COMM_NULL:
                self._intercomm = comm_plus.Create_intercomm(0, comm, root_minus)
                self._local_comm = comm_plus

        if self._intercomm == MPI.COMM_NULL:
            return

        self._local_boundary_ranks_minus = boundary_group_minus.Translate_ranks(local_groups[0], self._boundary_ranks_minus, boundary_group_minus)
        self._local_boundary_ranks_plus  = boundary_group_plus.Translate_ranks(local_groups[1], self._boundary_ranks_plus, boundary_group_plus)

#        high = self._local_rank_plus is not None
#        self._intercomm = self._intercomm.Merge(high=high)
        # Store arrays with all the reduced starts and reduced ends along each direction
        self._reduced_global_starts_minus = [None]*self._ndims
        self._reduced_global_ends_minus   = [None]*self._ndims
        self._reduced_global_starts_plus  = [None]*self._ndims
        self._reduced_global_ends_plus    = [None]*self._ndims
        for axis in range( self._ndims ):
            ni = reduced_npts_minus[axis]
            di = nprocs_minus[axis]
            pi = pads_minus[axis]
            mi = shifts_minus[axis]
            nj = reduced_npts_plus[axis]
            dj = nprocs_plus[axis]
            pj = pads_plus[axis]
            mj = shifts_plus[axis]

            self._reduced_global_starts_minus[axis] = np.array( [( ci   *ni)//di   for ci in range( di )] )
            self._reduced_global_ends_minus  [axis] = np.array( [((ci+1)*ni)//di-1 for ci in range( di )] )
            self._reduced_global_starts_plus[axis] = np.array( [( cj   *nj)//dj   for cj in range( dj )] )
            self._reduced_global_ends_plus  [axis] = np.array( [((cj+1)*nj)//dj-1 for cj in range( dj )] )
            if mi>1:self._reduced_global_ends_minus [axis][-1] += pi+1
            if mj>1:self._reduced_global_ends_plus [axis][-1] += pj+1

        # Store arrays with all the starts and ends along each direction
        self._global_starts_minus = [None]*self._ndims
        self._global_ends_minus   = [None]*self._ndims
        self._global_starts_plus  = [None]*self._ndims
        self._global_ends_plus    = [None]*self._ndims

        for axis in range( self._ndims ):
            ni = npts_minus[axis]
            di = nprocs_minus[axis]
            pi = pads_minus[axis]
            mi = shifts_minus[axis]
            r_starts_minus = self._reduced_global_starts_minus[axis]
            r_ends_minus   = self._reduced_global_ends_minus  [axis]
            nj = npts_plus[axis]
            dj = nprocs_plus[axis]
            pj = pads_plus[axis]
            mj = shifts_plus[axis]
            r_starts_plus = self._reduced_global_starts_plus[axis]
            r_ends_plus   = self._reduced_global_ends_plus  [axis]

            global_starts_minus = [0]
            for ci in range(1,di):
                global_starts_minus.append(global_starts_minus[ci-1] + (r_ends_minus[ci-1]-r_starts_minus[ci-1]+1)*mi)

            global_starts_plus = [0]
            for cj in range(1,dj):
                global_starts_plus.append(global_starts_plus[cj-1] + (r_ends_plus[cj-1]-r_starts_plus[cj-1]+1)*mj)

            global_ends_minus   = [global_starts_minus[ci+1]-1 for ci in range( di-1 )] + [ni-1]
            global_ends_plus   = [global_starts_plus[cj+1]-1 for cj in range( dj-1 )] + [nj-1]

            self._global_starts_minus[axis] = np.array( global_starts_minus )
            self._global_ends_minus  [axis] = np.array( global_ends_minus )
            self._global_starts_plus[axis] = np.array( global_starts_plus )
            self._global_ends_plus  [axis] = np.array( global_ends_plus )

        if self._local_rank_minus is not None:
            # Store input arguments
            self._npts    = tuple( npts_minus    )
            self._pads    = tuple( pads_minus    )
            self._periods = tuple( periods_minus )
            self._shifts  = tuple( shifts_minus  )
            self._dims    = nprocs_minus

            self._reduced_global_starts = self._reduced_global_starts_minus
            self._reduced_global_ends   = self._reduced_global_ends_minus
            self._global_starts         = self._global_starts_minus
            self._global_ends           = self._global_ends_minus

            # Start/end values of global indices (without ghost regions)
            coords       = self._coords_from_rank_minus[self._local_rank_minus]
            self._starts = tuple( self._global_starts[d][c] for d,c in enumerate(coords) )
            self._ends   = tuple( self._global_ends  [d][c] for d,c in enumerate(coords) )

        if self._local_rank_plus is not None:
            # Store input arguments
            self._npts    = tuple( npts_minus    )
            self._pads    = tuple( pads_minus    )
            self._periods = tuple( periods_minus )
            self._shifts  = tuple( shifts_minus  )
            self._dims    = nprocs_plus

            self._reduced_global_starts = self._reduced_global_starts_plus
            self._reduced_global_ends   = self._reduced_global_ends_plus
            self._global_starts         = self._global_starts_plus
            self._global_ends           = self._global_ends_plus

            # Start/end values of global indices (without ghost regions)
            coords       = self._coords_from_rank_plus[self._local_rank_plus]
            self._starts = tuple( self._global_starts[d][c] for d,c in enumerate(coords) )
            self._ends   = tuple( self._global_ends  [d][c] for d,c in enumerate(coords) )

        # List of 1D global indices (without ghost regions)
        self._grids = tuple( range(s,e+1) for s,e in zip( self._starts, self._ends ) )

        self._petsccart         = None
        self._parent_starts     = tuple([None]*self._ndims)
        self._parent_ends       = tuple([None]*self._ndims)
        self._parent_npts_minus = tuple([None]*self._ndims)
        self._parent_npts_plus  = tuple([None]*self._ndims)
        self._get_minus_starts_ends = None
        self._get_plus_starts_ends  = None

        self._communication_infos = {}

    #---------------------------------------------------------------------------
    # Global properties (same for each process)
    #---------------------------------------------------------------------------
    @property
    def ndims( self ):
        return self._ndims

    @property
    def npts_minus( self ):
        return self._npts_minus

    @property
    def npts_plus( self ):
        return self._npts_plus

    @property
    def pads_minus( self ):
        return self._pads_minus

    @property
    def pads_plus( self ):
        return self._pads_plus

    @property
    def periods_minus( self ):
        return self._periods_minus

    @property
    def periods_plus( self ):
        return self._periods_plus

    @property
    def shifts_minus( self ):
        return self._shifts_minus

    @property
    def shifts_plus( self ):
        return self._shifts_plus

    @property
    def ext_minus( self ):
        return self._ext_minus

    @property
    def ext_plus( self ):
        return self._ext_plus

    @property
    def root_rank_minus( self ):
        return self._root_rank_minus

    @property
    def root_rank_plus( self ):
        return self._root_rank_plus

    @property
    def ranks_in_topo_minus( self ):
        return self._ranks_in_topo_minus

    @property
    def coords_from_rank_minus( self ):
        return self._coords_from_rank_minus

    @property
    def coords_from_rank_plus( self ):
        return self._coords_from_rank_plus

    @property
    def boundary_ranks_minus( self ):
        return self._boundary_ranks_minus

    @property
    def boundary_ranks_plus( self ):
        return self._boundary_ranks_plus

    @property
    def reduced_global_starts_minus( self ):
        return self._reduced_global_starts_minus

    @property
    def reduced_global_starts_plus( self ):
        return self._reduced_global_starts_plus

    @property
    def reduced_global_ends_minus( self ):
        return self._reduced_global_ends_minus

    @property
    def reduced_global_ends_plus( self ):
        return self._reduced_global_ends_plus

    @property
    def global_starts_minus( self ):
        return self._global_starts_minus

    @property
    def global_starts_plus( self ):
        return self._global_starts_plus

    @property
    def global_ends_minus( self ):
        return self._global_ends_minus

    @property
    def global_ends_plus( self ):
        return self._global_ends_plus

    @property
    def axis( self ):
        return self._axis

    @property
    def comm( self ):
        return self._comm

    @property
    def intercomm( self ):
        return self._intercomm

    @property
    def is_comm_null( self ):
        return self._intercomm == MPI.COMM_NULL

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
    def ranks_in_topo_plus( self ):
        return self._ranks_in_topo_plus

    @property
    def local_group_minus( self ):
        return self._local_group_minus

    @property
    def local_group_plus( self ):
        return self._local_group_plus

    @property
    def local_comm_minus( self ):
        return self._local_comm_minus

    @property
    def local_comm_plus( self ):
        return self._local_comm_plus

    @property
    def local_comm( self ):
        return self._local_comm

    @property
    def local_rank_minus( self ):
        return self._local_rank_minus

    @property
    def local_rank_plus( self ):
        return self._local_rank_plus

    #---------------------------------------------------------------------------
    def reduce_elements( self, axes, n_elements):

        if isinstance(axes, int):
            axes = [axes]

        npts    = [self.npts_minus, self.npts_plus]
        pads    = [self.pads_minus, self.pads_plus]
        periods = [self.periods_minus, self.periods_plus]
        comm    = self.comm
        shifts  = [self.shifts_minus, self.shifts_plus]
        axes    = [self.axis, self.axis]
        exts    = [self.ext_minus, self.ext_plus]
        ranks_in_topo       = [self.ranks_in_topo_minus, self.ranks_in_topo_plus]
        local_groups        = [self.local_group_minus, self.local_group_plus]
        local_communicators = [self.local_comm_minus, self.local_comm_plus]
        root_ranks   = [self.root_rank_minus, self.root_rank_plus]
        requests     = []
        num_threads  = self.num_threads

        cart = InterfaceCartDecomposition(npts, pads, periods, comm,
                                         shifts, axes, exts,
                                         ranks_in_topo, local_groups,
                                         local_communicators,
                                         root_ranks, requests, num_threads, reduce_elements=True)


        cart._npts_minus = tuple(n - ne for n,ne in zip(cart.npts_minus, n_elements))
        cart._npts_plus  = tuple(n - ne for n,ne in zip(cart.npts_plus, n_elements))

        cart._shifts_minus = [max(1,m-1) for m in self.shifts_minus]
        cart._shifts_plus  = [max(1,m-1) for m in self.shifts_plus]

        assert all(axis<cart._ndims for axis in axes)

        if cart.is_comm_null:
            return cart

        # Store arrays with all the starts and ends along each direction
        cart._global_starts_minus = [None]*self._ndims
        cart._global_ends_minus   = [None]*self._ndims
        cart._global_starts_plus  = [None]*self._ndims
        cart._global_ends_plus    = [None]*self._ndims

        for axis in range( self._ndims ):
            ni = cart._npts_minus[axis]
            di = cart._nprocs_minus[axis]
            pi = cart._pads_minus[axis]
            mi = cart._shifts_minus[axis]
            r_starts_minus = cart._reduced_global_starts_minus[axis]
            r_ends_minus   = cart._reduced_global_ends_minus  [axis]
            nj = cart._npts_plus[axis]
            dj = cart._nprocs_plus[axis]
            pj = cart._pads_plus[axis]
            mj = cart._shifts_plus[axis]
            r_starts_plus = cart._reduced_global_starts_plus[axis]
            r_ends_plus   = cart._reduced_global_ends_plus  [axis]

            global_starts_minus = [0]
            for ci in range(1,di):
                global_starts_minus.append(global_starts_minus[ci-1] + (r_ends_minus[ci-1]-r_starts_minus[ci-1]+1)*mi)

            global_starts_plus = [0]
            for cj in range(1,dj):
                global_starts_plus.append(global_starts_plus[cj-1] + (r_ends_plus[cj-1]-r_starts_plus[cj-1]+1)*mj)

            global_ends_minus   = [global_starts_minus[ci+1]-1 for ci in range( di-1 )] + [ni-1]
            global_ends_plus   = [global_starts_plus[cj+1]-1 for cj in range( dj-1 )] + [nj-1]

            cart._global_starts_minus[axis] = np.array( global_starts_minus )
            cart._global_ends_minus  [axis] = np.array( global_ends_minus )
            cart._global_starts_plus[axis] = np.array( global_starts_plus )
            cart._global_ends_plus  [axis] = np.array( global_ends_plus )

        # Start/end values of global indices (without ghost regions)
        if self._local_rank_minus is not None:
            coords       = self._coords_from_rank_minus[self._local_rank_minus]
            cart._starts = tuple( cart._global_starts_minus[d][c] for d,c in enumerate(coords) )
            cart._ends   = tuple( cart._global_ends_minus  [d][c] for d,c in enumerate(coords) )

        if self._local_rank_plus is not None:
            coords       = self._coords_from_rank_plus[self._local_rank_plus]
            cart._starts = tuple( cart._global_starts_plus[d][c] for d,c in enumerate(coords) )
            cart._ends   = tuple( cart._global_ends_plus  [d][c] for d,c in enumerate(coords) )

        cart._parent_starts = self.starts
        cart._parent_ends   = self.ends
        cart._parent_npts_minus = tuple(npts[0])
        cart._parent_npts_plus  = tuple(npts[1])

        cart._communication_infos = {}
        cart._get_minus_starts_ends = self._get_minus_starts_ends
        cart._get_plus_starts_ends  = self._get_plus_starts_ends
        cart._communication_infos[cart._axis] = cart._compute_communication_infos_p2p(cart._axis)

        return cart

    def set_communication_info( self, get_minus_starts_ends, get_plus_starts_ends ):
        self._communication_infos[self._axis] = self._compute_communication_infos_p2p(self._axis, get_minus_starts_ends, get_plus_starts_ends)

    def get_communication_infos( self, axis ):
        return self._communication_infos[ axis ]

    #---------------------------------------------------------------------------
    def _compute_communication_infos( self, axis ):

        if self._intercomm == MPI.COMM_NULL:
            return

        # Mesh info
        npts_minus   = self._npts_minus
        npts_plus    = self._npts_plus
        p_npts_minus = self._parent_npts_minus
        p_npts_plus  = self._parent_npts_plus
        pads_minus   = self._pads_minus
        pads_plus    = self._pads_plus
        shifts_minus = self._shifts_minus
        shifts_plus  = self._shifts_plus
        ext_minus    = self._ext_minus
        ext_plus     = self._ext_plus
        indices  = []

        diff = 0
        if p_npts_minus[axis] is not None:
            diff = p_npts_minus[axis]-npts_minus[axis]

        if self._local_rank_minus is not None:
            rank_minus = self._local_rank_minus
            coords = self._coords_from_rank_minus[rank_minus]
            starts = [self._global_starts_minus[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_minus[d][c] for d,c in enumerate(coords)]
            send_shape   = [e-s+1+2*m*p for s,e,m,p in zip(starts, ends, shifts_minus, pads_minus)]
            send_starts  = [m*p for m,p in zip(shifts_minus, pads_minus)]
            m,p,s,e      = shifts_minus[axis], pads_minus[axis], starts[axis], ends[axis]
            send_starts[axis] = m*p if ext_minus == -1 else m*p+e-s+1-p-1+diff
            starts[axis] = starts[axis] if ext_minus == -1 else ends[axis]-pads_minus[axis]+diff
            ends[axis]   = starts[axis]+pads_minus[axis]-diff if ext_minus == -1 else ends[axis]
            send_buf_shape    = [e-s+1 for s,e,p,m in zip(starts, ends, pads_minus, shifts_minus)]

            # ...
            coords = self._coords_from_rank_plus[self._boundary_ranks_plus[0]]
            starts = [self._global_starts_plus[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_plus[d][c] for d,c in enumerate(coords)]

            recv_shape       = [n+2*m*p for n,m,p in zip(npts_plus, shifts_plus, pads_plus)]
            recv_shape[axis] = pads_plus[axis]+1-diff + 2*shifts_plus[axis]*pads_plus[axis]

            displacements   = [0]*(len(self._boundary_ranks_plus)+1)
            recv_counts     = [None]*len(self._boundary_ranks_plus)
            for k,b in enumerate(self._boundary_ranks_plus):
                coords = self._coords_from_rank_plus[b]
                starts = [self._global_starts_plus[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_plus[d][c] for d,c in enumerate(coords)]
                starts[axis] = starts[axis] if ext_plus == -1 else ends[axis]-pads_plus[axis]+diff
                ends[axis]   = starts[axis]+pads_plus[axis]-diff if ext_plus == -1 else ends[axis]
                shape_k = [e-s+1 for s,e in zip(starts, ends)]
                recv_counts[k] = np.product(shape_k)
                ranges         = [(s+p*m, p*m+e+1) for s,e,p,m in zip(starts, ends, pads_plus, shifts_plus)]
                ranges[axis]   = (shifts_plus[axis]*pads_plus[axis], shifts_plus[axis]*pads_plus[axis]+shape_k[axis])
                indices       += [np.ravel_multi_index( ii, dims=recv_shape, order='C' ) for ii in itertools.product(*[range(*a) for a in ranges])]

        elif self._local_rank_plus is not None:
            rank_plus = self._local_rank_plus
            coords = self._coords_from_rank_plus[rank_plus]
            starts = [self._global_starts_plus[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_plus[d][c] for d,c in enumerate(coords)]
            send_shape   = [e-s+1+2*m*p for s,e,m,p in zip(starts, ends, shifts_plus, pads_plus)]
            send_starts  = [m*p for m,p in zip(shifts_plus, pads_plus)]
            m,p,s,e = shifts_plus[axis], pads_plus[axis], starts[axis], ends[axis]
            send_starts[axis] = m*p if ext_plus == -1 else m*p+e-s+1-p-1+diff
            starts[axis] = starts[axis] if ext_plus == -1 else ends[axis]-pads_plus[axis]+diff
            ends[axis]   = starts[axis]+pads_plus[axis]-diff if ext_plus == -1 else ends[axis]
            send_buf_shape  = [e-s+1 for s,e,p,m in zip(starts, ends, pads_plus, shifts_plus)]

            # ...
            coords = self._coords_from_rank_minus[self._boundary_ranks_minus[0]]
            starts = [self._global_starts_minus[d][c] for d,c in enumerate(coords)]
            ends   = [self._global_ends_minus[d][c] for d,c in enumerate(coords)]

            recv_shape           = [n+2*m*p for n,m,p in zip(npts_minus, shifts_minus, pads_minus)]
            recv_shape[axis]     = pads_minus[axis]+1-diff + 2*shifts_minus[axis]*pads_minus[axis]

            displacements   = [0]*(len(self._boundary_ranks_minus)+1)
            recv_counts     = [None]*len(self._boundary_ranks_minus)
            for k,b in enumerate(self._boundary_ranks_minus):
                coords = self._coords_from_rank_minus[b]
                starts = [self._global_starts_minus[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_minus[d][c] for d,c in enumerate(coords)]
                starts[axis] = starts[axis] if ext_minus == -1 else ends[axis]-pads_minus[axis]+diff
                ends[axis]   = starts[axis]+pads_minus[axis]-diff if ext_minus == -1 else ends[axis]
                shape_k = [e-s+1 for s,e in zip(starts, ends)]
                recv_counts[k] = np.product(shape_k)
                ranges       = [(s+p*m, p*m+e+1) for s,e,p,m in zip(starts, ends, pads_minus, shifts_minus)]
                ranges[axis] = (shifts_minus[axis]*pads_minus[axis], shifts_minus[axis]*pads_minus[axis]+shape_k[axis])
                indices     += [np.ravel_multi_index( ii, dims=recv_shape, order='C' ) for ii in itertools.product(*[range(*a) for a in ranges])]

        displacements[1:] = np.cumsum(recv_counts)
        # Store all information into dictionary
        info = {'send_buf_shape' : tuple( send_buf_shape ),
                'send_starts'    : tuple( send_starts ),
                'send_shape'     : tuple( send_shape  ),
                'recv_shape'     : tuple( recv_shape ),
                'displacements'  : tuple( displacements ),
                'recv_counts'    : tuple( recv_counts),
                'indices'        : indices}

        return info
    #---------------------------------------------------------------------------
    def _compute_communication_infos_p2p( self, axis , get_minus_starts_ends=None, get_plus_starts_ends=None):

        if self._intercomm == MPI.COMM_NULL:
            return

        # Mesh info
        npts_minus   = self._npts_minus
        npts_plus    = self._npts_plus
        p_npts_minus = self._parent_npts_minus
        p_npts_plus  = self._parent_npts_plus
        pads_minus   = self._pads_minus
        pads_plus    = self._pads_plus
        shifts_minus = self._shifts_minus
        shifts_plus  = self._shifts_plus
        ext_minus    = self._ext_minus
        ext_plus     = self._ext_plus
        indices      = []

        if get_minus_starts_ends is not None:
            self._get_minus_starts_ends = get_minus_starts_ends

        if get_plus_starts_ends is not None:
            self._get_plus_starts_ends  = get_plus_starts_ends

        diff = 0
        if p_npts_minus[axis] is not None:
            diff = p_npts_minus[axis]-npts_minus[axis]

        if self._local_rank_minus is not None:
            rank_minus = self._local_rank_minus
            coords = self._coords_from_rank_minus[rank_minus]
            starts_minus = [self._global_starts_minus[d][c] for d,c in enumerate(coords)]
            ends_minus   = [self._global_ends_minus[d][c] for d,c in enumerate(coords)]
            starts_extended_minus = [s-p for s,p in zip(starts_minus, pads_minus)]
            ends_extended_minus = [min(n, e+p) for e,p,n in zip(ends_minus, pads_minus, npts_minus)]
            buf_shape   = [e-s+1+2*m*p for s,e,m,p in zip(starts_minus, ends_minus, shifts_minus, pads_minus)]
            dest_ranks       = []
            buf_send_shape   = []
            gbuf_send_shape  = []
            gbuf_send_starts = []
            for k,rank_plus in enumerate(self._boundary_ranks_plus):
                coords = self._coords_from_rank_plus[rank_plus]
                starts = [self._global_starts_plus[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_plus[d][c] for d,c in enumerate(coords)]
                starts_m, ends_m = self._get_minus_starts_ends(starts, ends, npts_minus, npts_plus, axis, axis,
                                                         ext_minus, ext_plus, pads_minus, pads_plus, diff)
                starts_inter = [max(s1,s2) for s1,s2 in zip(starts_minus, starts_m)]
                ends_inter   = [min(e1,e2) for e1,e2 in zip(ends_minus, ends_m)]
                if any(s>=e if i!=axis else False for i,(s,e) in enumerate(zip(starts_inter, ends_inter))):
                    continue

                starts_inter[axis] = starts_minus[axis] if ext_minus == -1 else ends_minus[axis]-pads_minus[axis]+diff
                ends_inter[axis]   = starts_minus[axis]+pads_minus[axis]-diff if ext_minus == -1 else ends_minus[axis]

                dest_ranks.append(self._local_boundary_ranks_plus[k])
                buf_send_shape.append([e-s+1 for s,e in zip(starts_inter, ends_inter)])
                gbuf_send_shape.append(buf_shape)
                gbuf_send_starts.append([si-s+p for si,s,p in zip(starts_inter, starts_minus, pads_minus)])

            buf_shape   = [e-s+1+2*m*p for s,e,m,p in zip(starts_minus, ends_minus, shifts_plus, pads_plus)]
            buf_shape[axis] = 3*pads_plus[axis]+1-diff
            source_ranks     = []
            buf_recv_shape   = []
            gbuf_recv_shape  = []
            gbuf_recv_starts = []
            for k,rank_plus in enumerate(self._boundary_ranks_plus):
                coords = self._coords_from_rank_plus[rank_plus]
                starts = [self._global_starts_plus[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_plus[d][c] for d,c in enumerate(coords)]

                starts_inter = [max(s1,s2) for s1,s2 in zip(starts_extended_minus, starts)]
                ends_inter   = [min(e1,e2) for e1,e2 in zip(ends_extended_minus, ends)]
                if any(s>=e if i!=axis else False for i,(s,e) in enumerate(zip(starts_inter, ends_inter))):
                    continue

                starts_extended_minus[axis] = 0
                starts_inter[axis]          = pads_plus[axis]
                ends_inter[axis]            = 2*pads_plus[axis]-diff

                source_ranks.append(self._local_boundary_ranks_plus[k])
                buf_recv_shape.append([e-s+1 for s,e in zip(starts_inter, ends_inter)])
                gbuf_recv_shape.append(buf_shape)
                gbuf_recv_starts.append([si-s for si,s in zip(starts_inter, starts_extended_minus)])

        elif self._local_rank_plus is not None:
            rank_plus = self._local_rank_plus
            coords = self._coords_from_rank_plus[rank_plus]
            starts_plus = [self._global_starts_plus[d][c] for d,c in enumerate(coords)]
            ends_plus   = [self._global_ends_plus[d][c] for d,c in enumerate(coords)]
            starts_extended_plus = [s-p for s,p in zip(starts_plus, pads_plus)]
            ends_extended_plus = [min(n, e+p) for e,p,n in zip(ends_plus, pads_plus, npts_plus)]
            buf_shape   = [e-s+1+2*m*p for s,e,m,p in zip(starts_plus, ends_plus, shifts_plus, pads_plus)]
            dest_ranks       = []
            buf_send_shape   = []
            gbuf_send_shape  = []
            gbuf_send_starts = []

            for k,rank_minus in enumerate(self._boundary_ranks_minus):
                coords = self._coords_from_rank_minus[rank_minus]
                starts = [self._global_starts_minus[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_minus[d][c] for d,c in enumerate(coords)]
                starts_p, ends_p = self._get_plus_starts_ends(starts, ends, npts_minus, npts_plus, axis, axis,
                                                         ext_minus, ext_plus, pads_minus, pads_plus, diff)
                starts_inter = [max(s1,s2) for s1,s2 in zip(starts_plus, starts_p)]
                ends_inter   = [min(e1,e2) for e1,e2 in zip(ends_plus, ends_p)]
                if any(s>=e if i!=axis else False for i,(s,e) in enumerate(zip(starts_inter, ends_inter))):
                    continue

                starts_inter[axis] = starts_plus[axis] if ext_plus == -1 else ends_plus[axis]-pads_plus[axis]+diff
                ends_inter[axis]   = starts_plus[axis]+pads_plus[axis]-diff if ext_plus == -1 else ends_plus[axis]

                dest_ranks.append(self._local_boundary_ranks_minus[k])
                buf_send_shape.append([e-s+1 for s,e in zip(starts_inter, ends_inter)])
                gbuf_send_shape.append(buf_shape)
                gbuf_send_starts.append([si-s+p for si,s,p in zip(starts_inter, starts_plus, pads_plus)])

            buf_shape        = [e-s+1+2*m*p for s,e,m,p in zip(starts_plus, ends_plus, shifts_minus, pads_minus)]
            buf_shape[axis] = 3*pads_minus[axis]+1-diff
            source_ranks     = []
            buf_recv_shape   = []
            gbuf_recv_shape  = []
            gbuf_recv_starts = []

            for k,rank_minus in enumerate(self._boundary_ranks_minus):
                coords = self._coords_from_rank_minus[rank_minus]
                starts = [self._global_starts_minus[d][c] for d,c in enumerate(coords)]
                ends   = [self._global_ends_minus[d][c] for d,c in enumerate(coords)]

                starts_inter = [max(s1,s2) for s1,s2 in zip(starts_extended_plus, starts)]
                ends_inter   = [min(e1,e2) for e1,e2 in zip(ends_extended_plus, ends)]
                if any(s>=e if i!=axis else False for i,(s,e) in enumerate(zip(starts_inter, ends_inter))):
                    continue

                starts_extended_plus[axis] = 0
                starts_inter[axis]          = pads_plus[axis]
                ends_inter[axis]            = 2*pads_plus[axis]-diff

                source_ranks.append(self._local_boundary_ranks_minus[k])
                buf_recv_shape.append([e-s+1 for s,e in zip(starts_inter, ends_inter)])
                gbuf_recv_shape.append(buf_shape)
                gbuf_recv_starts.append([si-s for si,s in zip(starts_inter, starts_extended_plus)])

        # Store all information into dictionary
        info = {'dest_ranks'       : tuple( dest_ranks ),
                'buf_send_shape'   : tuple( buf_send_shape ),
                'gbuf_send_shape'  : tuple( gbuf_send_shape  ),
                'gbuf_send_starts' : tuple( gbuf_send_starts ),
                'source_ranks'     : tuple( source_ranks ),
                'buf_recv_shape'   : tuple( buf_recv_shape),
                'gbuf_recv_shape'  : tuple( gbuf_recv_shape ),
                'gbuf_recv_starts' : tuple( gbuf_recv_starts )
                }

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

    """
    def __init__(self, cart, dtype):

        assert isinstance(cart, InterfaceCartDecomposition)

        send_types, recv_types = self._create_buffer_types( cart, dtype )

        self._cart          = cart
        self._dtype         = dtype
        self._send_types    = send_types
        self._recv_types    = recv_types
        self._dest_ranks    = cart.get_communication_infos( cart.axis )['dest_ranks']
        self._source_ranks  = cart.get_communication_infos( cart.axis )['source_ranks']

    # ...
    def update_ghost_regions( self, array_minus, array_plus ):
        req = self.start_update_ghost_regions(array_minus, array_plus)
        self.end_update_ghost_regions(req)

    # ...
    def start_update_ghost_regions( self, array_minus, array_plus ):
        send_req = []
        recv_req = []
        cart      = self._cart
        intercomm = cart.intercomm

        for i,(st,rank) in enumerate(zip(self._send_types, self._dest_ranks)):
            if cart._local_rank_minus is not None:
                send_buf = (array_minus, 1, st)
            else:
                send_buf = (array_plus, 1, st)

            send_req.append(intercomm.Isend( send_buf, rank ))

        for i,(st,rank) in enumerate(zip(self._recv_types, self._source_ranks)):
            if cart._local_rank_minus is not None:
                recv_buf = (array_plus, 1, st)
            else:
                recv_buf = (array_minus, 1, st)

            recv_req.append(intercomm.Irecv( recv_buf, rank ))
        return send_req + recv_req

    def end_update_ghost_regions(self, req):
        MPI.Request.Waitall(req)

    @staticmethod
    def _create_buffer_types( cart, dtype ):

        assert isinstance( cart, InterfaceCartDecomposition )

        mpi_type = find_mpi_type( dtype )
        info     = cart.get_communication_infos( cart.axis )

        send_types = [None]*len(info['dest_ranks'])

        for i in range(len(info['dest_ranks'])):
            send_types[i] = mpi_type.Create_subarray(
                         sizes    = info['gbuf_send_shape'][i] ,
                         subsizes = info['buf_send_shape'][i] ,
                         starts   = info['gbuf_send_starts'][i]).Commit()

        recv_types = [None]*len(info['source_ranks'])
        for i in range(len(info['source_ranks'])):
            recv_types[i] = mpi_type.Create_subarray(
                         sizes    = info['gbuf_recv_shape'][i] ,
                         subsizes = info['buf_recv_shape'][i] ,
                         starts   = info['gbuf_recv_starts'][i]).Commit()

        return send_types, recv_types

