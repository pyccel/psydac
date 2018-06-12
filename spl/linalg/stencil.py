# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from spl.linalg.basic import VectorSpace, Vector, LinearOperator
from spl.ddm.cart     import Cart

__all__ = ['StencilVectorSpace','StencilVector','StencilMatrix']

#===============================================================================
class StencilVectorSpace( VectorSpace ):
    """
    Vector space for n-dimensional stencil format. Two different initializations
    are possible:

    - serial  : StencilVectorSpace( npts, pads, dtype=float )
    - parallel: StencilVectorSpace( cart, dtype=float )

    Parameters
    ----------
    npts : tuple-like
        Number of entries along each direction
        (= global dimensions of vector space).

    pads : tuple-like
        Padding p along each direction (number of diagonals is 2*p+1).

    dtype : type
        Type of scalar entries.

    cart : spl.ddm.cart.Cart
        MPI Cartesian topology.

    """
    def __init__( self, *args, **kwargs ):

        if len(args) == 1 or hasattr( kwargs, 'cart' ):
            self._init_parallel( *args, **kwargs )
        else:
            self._init_serial  ( *args, **kwargs )

    # ...
    def _init_serial( self, npts, pads, periods, dtype=float ):

        assert len(npts) == len(pads) == len(periods)
        self._parallel = False

        # Sequential attributes
        self._starts  = tuple( 0   for n in npts )
        self._ends    = tuple( n-1 for n in npts )
        self._pads    = tuple( pads )
        self._periods = tuple( periods )
        self._dtype   = dtype
        self._ndim    = len( npts )

        # Global dimensions of vector space
        self._npts   = tuple( npts )

    # ...
    def _init_parallel( self, cart, dtype=float ):

        assert isinstance( cart, Cart )
        self._parallel = True

        # Sequential attributes
        self._starts  = cart.starts
        self._ends    = cart.ends
        self._pads    = cart.pads
        self._periods = cart.periods
        self._dtype   = dtype
        self._ndim    = len(cart.starts)

        # Global dimensions of vector space
        self._npts   = cart.npts

        # Parallel attributes
        mpi_type = self._find_mpi_type( dtype )
        send_types, recv_types = self._create_buffer_types( cart, mpi_type )

        self._cart       = cart
        self._mpi_type   = mpi_type
        self._send_types = send_types
        self._recv_types = recv_types

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def dimension( self ):
        """ The dimension of a vector space V is the cardinality
            (i.e. the number of vectors) of a basis of V over its base field.
        """
        from numpy import prod
        return prod( self._npts )

    # ...
    def zeros( self ):
        """
        Get a copy of the null element of the StencilVectorSpace V.

        Returns
        -------
        null : StencilVector
            A new vector object with all components equal to zero.

        """
        return StencilVector( self )

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def parallel( self ):
        return self._parallel

    # ...
    @property
    def cart( self ):
        return self._cart if self._parallel else None

    # ...
    @property
    def npts( self ):
        return self._npts

    # ...
    @property
    def starts( self ):
        return self._starts

    # ...
    @property
    def ends( self ):
        return self._ends

    # ...
    @property
    def pads( self ):
        return self._pads

    # ...
    @property
    def periods( self ):
        return self._periods

    # ...
    @property
    def dtype( self ):
        return self._dtype

    # ...
    @property
    def ndim( self ):
        return self._ndim

    #---------------------------------------------------------------------------
    # PARALLEL FACILITIES
    #---------------------------------------------------------------------------
    @staticmethod
    def _find_mpi_type( dtype ):
        from mpi4py import MPI

        if dtype == float:
            mpi_type = MPI.DOUBLE
        elif dtype == 'i':
            mpi_type = MPI.INT
        else:
            raise ValueError( 'dtype not understood' )

        return mpi_type

    # ...
    @staticmethod
    def _create_buffer_types( cart, mpi_type ):
        """ Create MPI subarray datatypes for accessing non-contiguous data.
        """
        data_shape = cart.shape
        send_types = {}
        recv_types = {}

        for direction in range( len(data_shape) ):
            for disp in [-1, 1]:
                info = cart.get_shift_info( direction, disp )

                buf_shape   = info[ 'buf_shape' ]
                send_starts = info['send_starts']
                recv_starts = info['recv_starts']

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

    # ...
    def get_send_type( self, direction, disp ):
        return self._send_types[direction,disp] if self._parallel else None

    # ...
    def get_recv_type( self, direction, disp ):
        return self._recv_types[direction,disp] if self._parallel else None

#===============================================================================
class StencilVector( Vector ):
    """
    Vector in n-dimensional stencil format.

    Parameters
    ----------
    V : spl.linalg.stencil.StencilVectorSpace
        Space to which the new vector belongs.

    """
    def __init__( self, V ):
        from numpy import zeros

        assert isinstance( V, StencilVectorSpace )

        sizes = [e-s+2*p+1 for s,e,p in zip(V.starts, V.ends, V.pads)]
        self._data  = zeros( sizes, dtype=V.dtype )
        self._space = V

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    #...
    def dot( self, v ):
        from numpy import dot

        assert isinstance( v, StencilVector )
        assert v._space is self._space

        index = tuple( slice(p,-p) for p in self.pads )
        res   = dot( self._data[index].flat, v._data[index].flat )

        if self._space.parallel:
            from mpi4py import MPI
            res = self._space.cart.comm_cart.allreduce( res, op=MPI.SUM )

        return res

    #...
    def copy( self ):
        w = StencilVector( self._space )
        w._data[:] = self._data[:]
        return w

    #...
    def __mul__( self, a ):
        w = StencilVector( self._space )
        w._data = self._data * a
        return w

    #...
    def __rmul__( self, a ):
        w = StencilVector( self._space )
        w._data = a * self._data

        return w

    #...
    def __add__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        w = StencilVector( self._space )
        w._data = self._data + v._data
        return w

    #...
    def __sub__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        w = StencilVector( self._space )
        w._data = self._data - v._data
        return w

    #...
    def __imul__( self, a ):
        self._data *= a
        return self

    #...
    def __iadd__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        self._data += v._data
        return self

    #...
    def __isub__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        self._data -= v._data
        return self

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def starts(self):
        return self._space.starts

    # ...
    @property
    def ends(self):
        return self._space.ends

    # ...
    @property
    def pads(self):
        return self._space.pads

    # ...
    def __str__(self):
        txt  = '\n'
        txt += '> starts  :: {starts}\n'.format(starts=self.starts)
        txt += '> ends    :: {ends}\n'.format(ends=self.ends)
        txt += '> pads    :: {pads}\n'.format(pads=self.pads)
        txt += '> data    :: {data}\n'.format(data=self._data)

        return txt

    # ...
    def toarray( self, *, with_pads=False ):
        """
        Return a numpy 1D array corresponding to the given StencilVector,
        with or without pads.

        Parameters
        ----------
        with_pads : bool
            If True, include pads in output array.

        Returns
        -------
        array : numpy.ndarray
            A copy of the data array collapsed into one dimension.

        """
        # In parallel case, call different functions based on 'with_pads' flag
        if self.space.parallel:
            if with_pads:
                return self._toarray_parallel_with_pads()
            else:
                return self._toarray_parallel_no_pads()

        # In serial case, ignore 'with_pads' flag
        index = tuple( slice(p,-p) for p in self.pads )
        return self._data[index].flatten()

    # ...
    def _toarray_parallel_no_pads( self ):
        from numpy import zeros
        a         = zeros( self.space.npts )
        idx_from  = tuple( slice(p,-p) for p in self.pads )
        idx_to    = tuple( slice(s,e+1) for s,e in zip(self.starts,self.ends) )
        a[idx_to] = self._data[idx_from]
        return a.reshape(-1)

    # ...
    def _toarray_parallel_with_pads( self ):
        from numpy import zeros

        # Step 0: create extended n-dimensional array with zero values
        shape = tuple( n+2*p for n,p in zip( self.space.npts, self.pads ) )
        a = zeros( shape )

        # Step 1: write extended data chunk (local to process) onto array
        idx = tuple( slice(s,e+2*p+1) for s,e,p in
            zip( self.starts, self.ends, self.pads ) )
        a[idx] = self._data

        # Step 2: if necessary, apply periodic boundary conditions to array
        ndim = self.space.ndim

        for direction in range( ndim ):

            periodic = self.space.cart.periods[direction]
            coord    = self.space.cart.coords [direction]
            nproc    = self.space.cart.nprocs [direction]

            if periodic:

                p = self.pads[direction]

                # Left-most process: copy data from left to right
                if coord == 0:
                    idx_from = tuple(
                        (slice(None,p) if d == direction else slice(None))
                        for d in range( ndim )
                    )
                    idx_to = tuple(
                        (slice(-2*p,-p) if d == direction else slice(None))
                        for d in range( ndim )
                    )
                    a[idx_to] = a[idx_from]

                # Right-most process: copy data from right to left
                if coord == nproc-1:
                    idx_from = tuple(
                        (slice(-p,None) if d == direction else slice(None))
                        for d in range( ndim )
                    )
                    idx_to = tuple(
                        (slice(p,2*p) if d == direction else slice(None))
                        for d in range( ndim )
                    )
                    a[idx_to] = a[idx_from]

        # Step 3: remove ghost regions from global array
        idx = tuple( slice(p,-p) for p in self.pads )
        out = a[idx]

        # Step 4: return flattened array
#        return out.flatten()
        return out.reshape(-1)

    # ...
    def __getitem__(self, key):
        index = self._getindex( key )
        return self._data[index]

    # ...
    def __setitem__(self, key, value):
        index = self._getindex( key )
        self._data[index] = value

    # ...
    # TODO: maybe change name to 'exchange'
    def update_ghost_regions( self, *, direction=None ):
        """
        Update ghost regions before performing non-local access to vector
        elements (e.g. in matrix-vector product).

        Parameters
        ----------
        direction : int
            Single direction along which to operate (if not specified, all of them).

        """
        ndim     = self._space.ndim
        parallel = self._space.parallel

        if parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
           update_ghost_regions = self._update_ghost_regions_parallel
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            update_ghost_regions = self._update_ghost_regions_serial

        if direction is not None:
            assert isinstance( direction, int )
            assert 0 <= direction < ndim
            update_ghost_regions( direction )
        else:
            for direction in range(ndim):
                update_ghost_regions( direction )

    # ...
    def _update_ghost_regions_serial( self, direction: int ):

        ndim     = self._space.ndim
        periodic = self._space.periods[direction]
        p        = self._space.pads   [direction]

        idx_front = [slice(None)]*direction
        idx_back  = [slice(None)]*(ndim-direction-1)

        if periodic:

            # Copy data from left to right
            idx_from = tuple( idx_front + [slice( p, 2*p)] + idx_back )
            idx_to   = tuple( idx_front + [slice(-p,None)] + idx_back )
            self._data[idx_to] = self._data[idx_from]

            # Copy data from right to left
            idx_from = tuple( idx_front + [slice(-2*p,-p)] + idx_back )
            idx_to   = tuple( idx_front + [slice(None, p)] + idx_back )
            self._data[idx_to] = self._data[idx_from]

        else:

            # Set left ghost region to zero
            idx_ghost = tuple( idx_front + [slice(None, p)] + idx_back )
            self._data[idx_ghost] = 0

            # Set right ghost region to zero
            idx_ghost = tuple( idx_front + [slice(-p,None)] + idx_back )
            self._data[idx_ghost] = 0

    # ...
    def _update_ghost_regions_parallel( self, direction: int ):

        from mpi4py import MPI

        u         = self._data
        space     = self._space
        cart      = space.cart
        comm_cart = cart.comm_cart

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
            recv_typ = space.get_recv_type( direction, disp )
            recv_buf = (u, 1, recv_typ)
            recv_req = comm_cart.Irecv( recv_buf, info['rank_source'], tag(disp) )
            requests.append( recv_req )

        # Start sending data (MPI_ISEND)
        for disp in [-1,1]:
            info     = cart.get_shift_info( direction, disp )
            send_typ = space.get_send_type( direction, disp )
            send_buf = (u, 1, send_typ)
            send_req = comm_cart.Isend( send_buf, info['rank_dest'], tag(disp) )
            requests.append( send_req )

        # Wait for end of data exchange (MPI_WAITALL)
        MPI.Request.Waitall( requests )

    #--------------------------------------
    # Private methods
    #--------------------------------------
    def _getindex( self, key ):

        # TODO: check if we should ignore padding elements

        if not isinstance( key, tuple ):
            key = (key,)
        index = []
        for (i,s,p) in zip(key, self.starts, self.pads):
            if isinstance(i, slice):
                start = None if i.start is None else i.start - s + p
                stop  = None if i.stop  is None else i.stop  - s + p
                l = slice(start, stop, i.step)
            else:
                l = i - s + p
            index.append(l)
        return tuple(index)

#===============================================================================
class StencilMatrix( LinearOperator ):
    """
    Matrix in n-dimensional stencil format.

    This is a linear operator that maps elements of stencil vector space V to
    elements of stencil vector space W.

    For now we only accept V==W.

    Parameters
    ----------
    V : spl.linalg.stencil.StencilVectorSpace
        Domain of the new linear operator.

    W : spl.linalg.stencil.StencilVectorSpace
        Codomain of the new linear operator.

    """
    def __init__( self, V, W ):

        from numpy import zeros

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W, StencilVectorSpace )
        assert V is W

        dims        = [e-s+1 for s,e in zip(V.starts, V.ends)]
        diags       = [2*p+1 for p in V.pads]
        self._data  = zeros( dims+diags, dtype=V.dtype )
        self._space = V

        self._dims  = dims
        self._ndim  = len( dims )

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain( self ):
        return self._space

    # ...
    @property
    def codomain( self ):
        return self._space

    # ...
    def dot( self, v, out=None ):

        from numpy import ndindex, dot

        assert isinstance( v, StencilVector )
        assert v.space is self.domain

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self.codomain
        else:
            out = StencilVector( self.codomain )

        ss = self.starts
        pp = self.pads
        kk = [slice(None)] * self._ndim

        for xx in ndindex( *self._dims ):

            ii    = tuple( s+x for s,x in zip(ss,xx) )
            jj    = tuple( slice(i-p,i+p+1) for i,p in zip(ii,pp) )
            ii_kk = tuple( list(ii) + kk )

            out[ii] = dot( self[ii_kk].flat, v[jj].flat )

        # Necessary if vector space is distributed across processes
        out.update_ghost_regions()

        return out

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------
    @property
    def starts( self ):
        return self._space.starts

    # ...
    @property
    def ends( self ):
        return self._space.ends

    # ...
    @property
    def pads( self ):
        return self._space.pads

    # ...
    def __getitem__(self, key):
        index = self._getindex( key )
        return self._data[index]

    # ...
    def __setitem__(self, key, value):
        index = self._getindex( key )
        self._data[index] = value

    #...
    def tocoo( self ):

        from numpy        import ndenumerate, ravel_multi_index, prod
        from scipy.sparse import coo_matrix

        # Shortcuts
        nn = self._space.npts
        nd = self._ndim

        ss = self.starts
        pp = self.pads

        # COO storage
        rows = []
        cols = []
        data = []

        for (index,value) in ndenumerate( self._data ):

            # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

            xx = index[:nd]  # x=i-s
            ll = index[nd:]  # l=p+k

            ii = [s+x for s,x in zip(ss,xx)]
            jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nn,pp)]

            I = ravel_multi_index( ii, dims=nn, order='C' )
            J = ravel_multi_index( jj, dims=nn, order='C' )

            rows.append( I )
            cols.append( J )
            data.append( value )

        M = coo_matrix(
                (data,(rows,cols)),
                shape = [prod(nn)]*2,
                dtype = self._space.dtype
        )

        M.eliminate_zeros()

        return M

    #...
    def tocsr( self ):
        return self.tocoo().tocsr()

    #...
    def toarray( self ):
        return self.tocoo().toarray()

    #...
    def max( self ):
        return self._data.max()

    #...
    def copy( self ):
        M = Matrix( self.domain, self.codomain )
        M._data[:] = self._data[:]
        return M

    #...
    def remove_spurious_entries( self ):
        """
        If any dimension is NOT periodic, make sure that the corresponding
        periodic corners are set to zero.

        """
        # TODO: access 'self._data' directly for increased efficiency
        # TODO: add unit tests

        ndim  = self._space.ndim

        for direction in range(ndim):

            periodic = self._space.periods[direction]

            if not periodic:

                n = self._space.npts[direction]

                s = self.starts[direction]
                e = self.ends  [direction]
                p = self.pads  [direction]

                idx_front = [slice(None)]*direction
                idx_back  = [slice(None)]*(ndim-direction-1)

                # Top-right corner
                for i in range( max(0,s), min(p,e+1) ):
                    index = tuple( idx_front + [i]            + idx_back +
                                   idx_front + [slice(-p,-i)] + idx_back )
                    self[index] = 0

                # Bottom-left corner
                for i in range( max(n-p,s), min(n,e+1) ):
                    index = tuple( idx_front + [i]              + idx_back +
                                   idx_front + [slice(n-i,p+1)] + idx_back )
                    self[index] = 0

    #--------------------------------------
    # Private methods
    #--------------------------------------
    def _getindex( self, key ):

        nd = self._ndim
        ii = key[:nd]
        kk = key[nd:]

        index = []

        for i,s in zip( ii, self.starts ):
            x = self._shift_index( i,-s )
            index.append( x )

        for k,p in zip( kk, self.pads ):
            l = self._shift_index( k,p )
            index.append( l )

        return tuple(index)

    # ...
    @staticmethod
    def _shift_index( index, shift ):
        if isinstance( index, slice ):
            start = None if index.start is None else index.start + shift
            stop  = None if index.stop  is None else index.stop  + shift
            return slice(start, stop, index.step)
        else:
            return index + shift

#===============================================================================
del VectorSpace, Vector, LinearOperator
