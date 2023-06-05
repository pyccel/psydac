# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import os
import warnings

import numpy as np

from types        import MappingProxyType
from scipy.sparse import coo_matrix
from mpi4py       import MPI

from psydac.linalg.basic   import VectorSpace, Vector, LinearOperator
from psydac.ddm.cart       import find_mpi_type, CartDecomposition, InterfaceCartDecomposition
from psydac.ddm.utilities  import get_data_exchanger
from .kernels              import *

__all__ = ('StencilVectorSpace','StencilVector','StencilMatrix', 'StencilInterfaceMatrix')

#===============================================================================
def compute_diag_len(pads, shifts_domain, shifts_codomain, return_padding=False):
    """ Compute the diagonal length and the padding of the stencil matrix for each direction,
        using the shifts of the domain and the codomain.

        Parameters
        ----------
        pads : tuple-like (int)
         Padding along each direction

        shifts_domain : tuple_like (int)
         Shifts of the domain along each direction

        shifts_codomain : tuple_like (int)
         Shifts of the codomain along each direction

        return_padding : bool
            Return the new padding if True
    
        Returns
        -------
        n : (int)
         Diagonal length of the stencil matrix

        ep : (int)
          Padding that constitutes the starting index of the non zero elements 
    """
    n = ((np.ceil((pads+1)/shifts_codomain)-1)*shifts_domain).astype('int')
    ep = -np.minimum(0, n-pads)
    n = n+ep + pads+1
    if return_padding:
        return n.astype('int'), (ep).astype('int')
    else:
        return n.astype('int')

#===============================================================================
class StencilVectorSpace( VectorSpace ):
    """
    Vector space for n-dimensional stencil format. Two different initializations
    are possible:

    - serial  : StencilVectorSpace( npts, pads, periods, shifts=None, starts=None, ends=None, dtype=float )
    - parallel: StencilVectorSpace( cart, dtype=float )

    Parameters
    ----------
    npts : tuple-like (int)
        Number of entries along each direction
        (= global dimensions of vector space).

    pads : tuple-like (int)
        Padding p along each direction needed for the ghost regions.

    periods : tuple-like (bool)
        Periodicity along each direction.

    shifts : tuple-like (int)
        shift m of the coefficients in each direction.

    starts : tuple-like (int)
        Index of the first coefficient local to the space in each direction.

    ends : tuple-like (int)
        Index of the last coefficient local to the space in each direction.

    dtype : type
        Type of scalar entries.

    cart : psydac.ddm.cart.CartDecomposition
        Tensor-product grid decomposition according to MPI Cartesian topology.

    """

    def __init__( self, cart, dtype=float ):

        assert isinstance( cart, (CartDecomposition, InterfaceCartDecomposition) )

        # Sequential attributes
        self._parallel   = cart.is_parallel
        self._cart       = cart
        self._ndim       = cart._ndims
        self._npts       = cart.npts
        self._pads       = cart.pads
        self._periods    = cart.periods
        self._shifts     = cart.shifts
        self._dtype      = dtype
        self._starts     = cart.starts
        self._ends       = cart.ends

        # The shape of the allocated numpy array
        self._shape         = cart.shape
        self._parent_starts = cart.parent_starts
        self._parent_ends   = cart.parent_ends
        self._mpi_type      = find_mpi_type(dtype)

        # The dictionary follows the structure {(axis, ext): StencilVectorSpace()}
        # where axis and ext represent the boundary shared by two patches
        self._interfaces    = {}
        self._interfaces_readonly = MappingProxyType(self._interfaces)


        # Parallel attributes
        if cart.is_parallel and not cart.is_comm_null:
            self._mpi_type      = find_mpi_type(dtype)
            if isinstance(cart, InterfaceCartDecomposition):
                self._shape = cart.get_interface_communication_infos(cart.axis)['gbuf_recv_shape'][0]
            else:
                self._synchronizer = get_data_exchanger( cart, dtype , assembly=True, blocking=False)

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def dimension( self ):
        """ The dimension of a vector space V is the cardinality
            (i.e. the number of vectors) of a basis of V over its base field.
        """
        return np.prod( self._npts )

    # ...
    @property
    def dtype( self ):
        return self._dtype

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
    def mpi_type( self ):
        return self._mpi_type

    @property
    def shape( self ):
        return self._shape

    @property
    def parallel( self ):
        return self._parallel

    # ...
    @property
    def cart( self ):
        return self._cart

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
    def parent_starts( self ):
        return self._parent_starts

    # ...
    @property
    def parent_ends( self ):
        return self._parent_ends

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
    def shifts( self ):
        return self._shifts

    # ...
    @property
    def ndim( self ):
        return self._ndim

    @property
    def interfaces( self ):
        return self._interfaces_readonly

    def set_interface(self, axis, ext, cart):

        """ Set the interface space along a given axis and extremity.

        Parameters
        ----------
         axis : int
          The axis of the new Interface space.

         ext: int
          The extremity of the new Interface space.
          the values must be 1 or -1.

         cart: CartDecomposition
          The cart of the new space.
        """

        assert int(ext) in [-1,1]

        # Create the interface space in the parallel case using the new cart
        assert isinstance(cart, (CartDecomposition, InterfaceCartDecomposition))
        if cart.is_comm_null: return
        if isinstance(cart, InterfaceCartDecomposition):
            # Case where the patches that share the interface are owned by different intra-communicators
            space = StencilVectorSpace(cart)
            self._interfaces[axis, ext] = space
        else:
            # Case where the patches that share the interface are owned by the same intra-communicator
            if self.parent_ends[axis] is not None:
                diff = min(1,self.parent_ends[axis]-self.ends[axis])
            else:
                diff = 0

            starts = list(cart._starts)
            ends   = list(cart._ends)
            parent_starts = list(cart._parent_starts)
            parent_ends   = list(cart._parent_ends)
            if ext == 1:
                starts[axis] = self.ends[axis]-self.pads[axis]+diff
                if parent_starts[axis] is not None:
                    parent_starts[axis] = parent_ends[axis]-self.pads[axis]
            else:
                ends[axis] = self.pads[axis]-diff
                if parent_ends[axis] is not None:
                    parent_ends[axis] = self.pads[axis]

            cart = cart.change_starts_ends(tuple(starts), tuple(ends), tuple(parent_starts), tuple(parent_ends))
            space = StencilVectorSpace(cart, self.dtype)

            self._interfaces[axis, ext] = space

#===============================================================================
class StencilVector( Vector ):
    """
    Vector in n-dimensional stencil format.

    Parameters
    ----------
    V : psydac.linalg.stencil.StencilVectorSpace
        Space to which the new vector belongs.

    """
    def __init__(self, V):

        assert isinstance(V, StencilVectorSpace)

        self._space          = V
        self._sizes          = V.shape
        self._ndim           = len(V.npts)
        self._data           = np.zeros( V.shape, dtype=V.dtype )
        self._dot_send_data  = np.zeros((1,), dtype=V.dtype)
        self._dot_recv_data  = np.zeros((1,), dtype=V.dtype)
        self._interface_data = {}
        self._requests       = None

        # allocate data for the boundary that shares an interface
        for axis, ext in V.interfaces:
            self._interface_data[axis, ext] = np.zeros( V.interfaces[axis, ext].shape, dtype=V.dtype )

        #prepare communications
        if V.cart.is_parallel and not V.cart.is_comm_null and isinstance(V.cart, CartDecomposition):
            self._requests = V._synchronizer.prepare_communications(self._data)

        # TODO: distinguish between different directions
        self._sync  = False

    def __del__(self):
        # Release memory of persistent MPI communication channels
        if self._requests:
            for request in self._requests:
                request.Free()

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space(self):
        return self._space

    #...
    @property
    def dtype(self):
        return self._space.dtype

    #...
    def dot(self, v):
        """
        Return the inner vector product between self and v.

        If the values are real, it returns the classical scalar product.
        If the values are complex, it returns the classical sesquilinear product with linearity on the vector v.

        Parameters
        ----------
        v : StencilVector
            Vector of the same space than self needed for the scalar product

        Returns
        -------
        null: self._space.dtype
            Scalar containing scalar product of v and self

        """

        assert isinstance(v, StencilVector)
        assert v._space is self._space

        if self._space.parallel:
            self._dot_send_data[0] = self._dot(self._data, v._data , self._space.pads, self._space.shifts)
            self._space.cart.global_comm.Allreduce((self._dot_send_data, self._space.mpi_type),
                                                   (self._dot_recv_data, self._space.mpi_type),
                                                   op=MPI.SUM )
            return self._dot_recv_data[0]
        else:
            return self._dot(self._data, v._data , self._space.pads, self._space.shifts)

    #...
    @staticmethod
    def _dot(v1, v2, pads, shifts):
        index = tuple( slice(m*p,-m*p) for p,m in zip(pads, shifts))
        return np.vdot(v1[index].flat, v2[index].flat)

    def conjugate(self, out=None):
        if out is not None:
            assert isinstance(out, StencilVector)
            assert out.space is self.space
        else:
            out = StencilVector(self.space)
        np.conjugate(self._data, out=out._data, casting='no')
        for axis, ext in self._space.interfaces:
            np.conjugate(self._interface_data[axis, ext], out=out._interface_data[axis, ext], casting='no')
        out._sync = self._sync
        return out

    #...
    def copy(self, out=None):
        if self is out:
            return self
        w = out or StencilVector( self._space )
        np.copyto(w._data, self._data, casting='no')
        for axis, ext in self._space.interfaces:
            np.copyto(w._interface_data[axis, ext], self._interface_data[axis, ext], casting='no')
        w._sync = self._sync
        return w

    #...
    def __neg__(self):
        w = StencilVector( self._space )
        np.negative(self._data, out=w._data)
        for axis, ext in self._space.interfaces:
            np.negative(self._interface_data[axis, ext], out=w._interface_data[axis, ext])
        w._sync = self._sync
        return w

    #...
    def __mul__(self, a):
        w = StencilVector( self._space )
        np.multiply(self._data, a, out=w._data)
        for axis, ext in self._space.interfaces:
            np.multiply(self._interface_data[axis, ext], a, out=w._interface_data[axis, ext])
        w._sync = self._sync
        return w

    #...
    def __add__(self, v):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        w = StencilVector( self._space )
        np.add(self._data, v._data, out=w._data)
        for axis, ext in self._space.interfaces:
            np.add(self._interface_data[axis, ext], v._interface_data[axis, ext], out=w._interface_data[axis, ext])
        w._sync = self._sync and v._sync
        return w

    #...
    def __sub__(self, v):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        w = StencilVector( self._space )
        np.subtract(self._data, v._data, out=w._data)
        for axis, ext in self._space.interfaces:
            np.subtract(self._interface_data[axis, ext], v._interface_data[axis, ext], out=w._interface_data[axis, ext])
        w._sync = self._sync and v._sync
        return w

    #...
    def __imul__(self, a):
        self._data *= a
        for axis, ext in self._space.interfaces:
            self._interface_data[axis, ext] *= a
        return self

    #...
    def __iadd__(self, v):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        self._data += v._data
        for axis, ext in self._space.interfaces:
            self._interface_data[axis, ext] += v._interface_data[axis, ext]
        self._sync  = v._sync and self._sync
        return self

    #...
    def __isub__(self, v):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        self._data -= v._data
        for axis, ext in self._space.interfaces:
            self._interface_data[axis, ext] -= v._interface_data[axis, ext]
        self._sync  = v._sync and self._sync
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
        txt += '> starts  :: {starts}\n'.format( starts= self.starts )
        txt += '> ends    :: {ends}\n'  .format( ends  = self.ends   )
        txt += '> pads    :: {pads}\n'  .format( pads  = self.pads   )
        txt += '> data    :: {data}\n'  .format( data  = self._data  )
        txt += '> sync    :: {sync}\n'  .format( sync  = self._sync  )
        return txt

    # ...
    def toarray(self, *, order='C', with_pads=False):
        """
        Return a numpy 1D array corresponding to the given StencilVector,
        with or without pads.

        Parameters
        ----------
        with_pads : bool
            If True, include pads in output array.

        order: {'C','F'}
             Memory representation of the data ‘C’ for row-major ordering (C-style), ‘F’ column-major ordering (Fortran-style).

        Returns
        -------
        array : numpy.ndarray
            A copy of the data array collapsed into one dimension.

        """


        # In parallel case, call different functions based on 'with_pads' flag
        if self.space.parallel:
            if with_pads:
                return self._toarray_parallel_with_pads(order=order)
            else:
                return self._toarray_parallel_no_pads(order=order)

        # In serial case, ignore 'with_pads' flag
        return self.toarray_local(order=order)

    # ...
    def toarray_local(self , *, order='C'):
        """ return the local array without the padding"""

        idx = tuple( slice(m*p,-m*p) for p,m in zip(self.pads, self.space.shifts) )
        return self._data[idx].flatten( order=order)

    # ...
    def _toarray_parallel_no_pads(self, order='C'):
        a         = np.zeros( self.space.npts, self.dtype )
        idx_from  = tuple( slice(m*p,-m*p) for p,m in zip(self.pads, self.space.shifts) )
        idx_to    = tuple( slice(s,e+1) for s,e in zip(self.starts,self.ends) )
        a[idx_to] = self._data[idx_from]
        return a.flatten( order=order)

    # ...
    def _toarray_parallel_with_pads(self, order='C'):

        pads = [m*p for m,p in zip(self.space.shifts, self.pads)]
        # Step 0: create extended n-dimensional array with zero values
        shape = tuple( n+2*p for n,p in zip( self.space.npts, pads ) )
        a = np.zeros( shape, self.dtype )

        # Step 1: write extended data chunk (local to process) onto array
        idx = tuple( slice(s,e+2*p+1) for s,e,p in
            zip( self.starts, self.ends, pads) )
        a[idx] = self._data

        # Step 2: if necessary, apply periodic boundary conditions to array
        ndim = self.space.ndim

        for direction in range( ndim ):

            periodic = self.space.cart.periods[direction]
            coord    = self.space.cart.coords [direction]
            nproc    = self.space.cart.nprocs [direction]

            if periodic:

                p = pads[direction]

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
        idx = tuple( slice(p,-p) for p in pads )
        out = a[idx]

        # Step 4: return flattened array
        return out.flatten( order=order)

    def topetsc(self):
        """ Convert to petsc data structure.
        """
        from psydac.linalg.topetsc import vec_topetsc
        vec = vec_topetsc( self )
        return vec

    # ...
    def __getitem__(self, key):
        index = self._getindex(key)
        return self._data[index]

    # ...
    def __setitem__(self, key, value):
        index = self._getindex(key)
        self._data[index] = value

    # ...
    @property
    def ghost_regions_in_sync(self):
        return self._sync

    # ...
    # NOTE: this property must be set collectively
    @ghost_regions_in_sync.setter
    def ghost_regions_in_sync(self, value):
        assert isinstance(value, bool)
        self._sync = value

    # ...
    # TODO: maybe change name to 'exchange'
    def update_ghost_regions(self):
        """
        Update ghost regions before performing non-local access to vector
        elements (e.g. in matrix-vector product).

        Parameters
        ----------
        direction : int
            Single direction along which to operate (if not specified, all of them).

        """

        # Update interior ghost regions
        if self.space.parallel:
            if not self.space.cart.is_comm_null:
                # PARALLEL CASE: fill in ghost regions with data from neighbors
                self.space._synchronizer.start_update_ghost_regions( self._data, self._requests )
                self.space._synchronizer.end_update_ghost_regions( self._data, self._requests )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._update_ghost_regions_serial()

        # Update interface ghost regions
        if self.space.parallel:

            for axis, ext in self.space.interfaces:
                V      = self.space.interfaces[axis, ext]
                if isinstance(V.cart, InterfaceCartDecomposition):continue
                slices = [slice(s, e+2*m*p+1) for s,e,m,p in zip(V.starts, V.ends, V.shifts, V.pads)]
                self._interface_data[axis, ext][...] = self._data[tuple(slices)]
        else:

            for axis, ext in self.space.interfaces:
                V      = self.space.interfaces[axis, ext]
                slices = [slice(s, e+2*m*p+1) for s,e,m,p in zip(V.starts, V.ends, V.shifts, V.pads)]
                self._interface_data[axis, ext][...] = self._data[tuple(slices)]

        # Flag ghost regions as up-to-date
        self._sync = True

    # ...
    def _update_ghost_regions_serial(self):

        ndim = self._space.ndim
        for direction in range(ndim):
            periodic = self._space.periods[direction]
            p        = self._space.pads   [direction]*self._space.shifts[direction]

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
    def exchange_assembly_data(self):
        """
        Exchange assembly data.
        """

        if self.space.parallel and not self.space.cart.is_comm_null:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self.space._synchronizer.start_exchange_assembly_data( self._data )
            self.space._synchronizer.end_exchange_assembly_data( self._data )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._exchange_assembly_data_serial()

        ndim     = self._space.ndim
        for direction in range(ndim):
            idx_front = [slice(None)]*direction
            idx_back  = [slice(None)]*(ndim-direction-1)

            p        = self._space.pads   [direction]
            m        = self._space.shifts[direction]
            idx_from = tuple( idx_front + [ slice(-m*p,None) if (-m*p+p)!=0 else slice(-m*p,None)] + idx_back )
            self._data[idx_from] = 0.
            idx_from = tuple( idx_front + [ slice(0,m*p)] + idx_back )
            self._data[idx_from] = 0.

    # ...
    def _exchange_assembly_data_serial(self):

        ndim = self._space.ndim
        for direction in range(ndim):

            periodic = self._space.periods[direction]
            p        = self._space.pads   [direction]
            m        = self._space.shifts[direction]

            if periodic:
                idx_front = [slice(None)]*direction
                idx_back  = [slice(None)]*(ndim-direction-1)

                # Copy data from left to right
                idx_to   = tuple( idx_front + [slice( m*p, m*p+p)] + idx_back )
                idx_from = tuple( idx_front + [ slice(-m*p,-m*p+p) if (-m*p+p)!=0 else slice(-m*p,None)] + idx_back )
                self._data[idx_to] += self._data[idx_from]

    #--------------------------------------
    # Private methods
    #--------------------------------------
    def _getindex(self, key):

        # TODO: check if we should ignore padding elements

        if not isinstance( key, tuple ):
            key = (key,)
        index = []
        for (i,s,p,m) in zip(key, self.starts, self.pads,self.space.shifts):
            if isinstance(i, slice):
                start = None if i.start is None else i.start - s + m*p
                stop  = None if i.stop  is None else i.stop  - s + m*p
                l = slice(start, stop, i.step)
            else:
                l = i - s + m*p
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
    V : psydac.linalg.stencil.StencilVectorSpace
        Domain of the new linear operator.

    W : psydac.linalg.stencil.StencilVectorSpace
        Codomain of the new linear operator.

    """
    def __init__( self, V, W, pads=None , backend=None):

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W, StencilVectorSpace )
        assert W.pads == V.pads
        if not W.dtype==V.dtype:
            raise NotImplementedError("The domain and the codomain should have the same data type.")

        if pads is not None:
            for p,vp in zip(pads, V.pads):
                assert p<=vp

        self._pads     = pads or tuple(V.pads)
        dims           = list(W.shape)
        diags          = [compute_diag_len(p, md, mc) for p,md,mc in zip(self._pads, V.shifts, W.shifts)]
        self._data     = np.zeros( dims+diags, dtype=W.dtype )
        self._domain   = V
        self._codomain = W
        self._ndim     = len( dims )
        self._backend  = backend
        self._is_T     = False
        self._diag_indices = None
        self._requests = None

        # Parallel attributes
        if W.parallel:
            if W.cart.is_comm_null:return
            # Create data exchanger for ghost regions
            self._synchronizer = get_data_exchanger(
                cart        = W.cart,
                dtype       = W.dtype,
                coeff_shape = diags,
                assembly    = True
            )

        # Flag ghost regions as not up-to-date (conservative choice)
        self._sync = False

        # Prepare the arguments for the dot product method
        nd  = [(ej-sj+2*gp*mj-mj*p-gp)//mj*mi+1 for sj,ej,mj,mi,p,gp in zip(V.starts, V.ends, V.shifts, W.shifts, self._pads, V.pads)]
        nc  = [ei-si+1 for si,ei,mj,p in zip(W.starts, W.ends, V.shifts, self._pads)]

        # Number of rows in matrix (along each dimension)
        nrows        = [min(ni,nj) for ni,nj  in zip(nc, nd)]
        nrows_extra  = [max(0,ni-nj) for ni,nj in zip(nc, nd)]

        args                 = {}
        args['starts']       = tuple(V.starts)
        args['nrows']        = tuple(nrows)
        args['nrows_extra']  = tuple(nrows_extra)
        args['gpads']        = tuple(V.pads)
        args['pads']         = tuple(self._pads)
        args['dm']           = tuple(V.shifts)
        args['cm']           = tuple(W.shifts)

        self._dotargs_null = args
        self._args         = args.copy()
        self._func         = self._dot

        self._transpose_args_null = self._prepare_transpose_args()
        self._transpose_args      = self._transpose_args_null.copy()
        self._transpose_func      = self._transpose

        if backend is None:
            backend = PSYDAC_BACKENDS.get(os.environ.get('PSYDAC_BACKEND'))

        if backend:
            self.set_backend(backend)

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain( self ):
        return self._domain

    # ...
    @property
    def codomain( self ):
        return self._codomain

    # ...
    @property
    def dtype( self ):
        return self._domain.dtype

    # ...
    def dot( self, v, out=None):
        """
        Return the matrix/vector product between self and v.
        This function optimized this product.

        Parameters
        ----------
        v   : StencilVector
            Vector of the domain of self needed for the Matrix/Vector product

        out : StencilVector
            Vector of the codomain of self

        Returns
        -------
        out : StencilVector
            Vector of the codomain of self, contain the result of the product

        """


        assert isinstance( v, StencilVector )
        assert v.space is self.domain

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self.codomain
        else:
            out = StencilVector( self.codomain )

        # Necessary if vector space is distributed across processes
        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        self._func(self._data, v._data, out._data, **self._args)

        # IMPORTANT: flag that ghost regions are not up-to-date
        out.ghost_regions_in_sync = False
        return out

    def vdot( self, v, out=None):
        """
        Return the matrix/vector product between the conjugate of self and v.
        This function optimized this product.

        Parameters
        ----------
        v   : StencilVector
            Vector of the domain of self needed for the Matrix/Vector product

        out : StencilVector
            Vector of the codomain of self

        Returns
        -------
        out : StencilVector
            Vector of the codomain of self, contain the result of the product

        """

        assert isinstance( v, StencilVector )
        assert v.space is self.domain

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self.codomain
        else:
            out = StencilVector( self.codomain )

        # Necessary if vector space is distributed across processes
        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        # Instead of computing A_*x, this function computes (A*x_)_
        self._func(self._data, np.conjugate(v._data), out._data, **self._args)
        np.conjugate(out._data, out=out._data)

        # IMPORTANT: flag that ghost regions are not up-to-date
        out.ghost_regions_in_sync = False
        return out

    # ...
    @staticmethod
    def _dot(mat, x, out, starts, nrows, nrows_extra, gpads, pads, dm, cm):

        # Index for k=i-j
        ndim = len(x.shape)
        kk   = [slice(None)]*ndim

        # pads are <= gpads
        diff = [gp-p for gp,p in zip(gpads, pads)]

        ndiags, _ = list(zip(*[compute_diag_len(p,mj,mi, return_padding=True) for p,mi,mj in zip(pads,cm,dm)]))

        bb = [p*m+p+1-n-s%m for p,m,n,s in zip(gpads, dm, ndiags, starts)]

        for xx in np.ndindex( *nrows ):

            ii    = tuple( mi*pi + x for mi,pi,x in zip(cm, gpads, xx) )
            jj    = tuple( slice(b-d+(x+s%mj)//mi*mj,b-d+(x+s%mj)//mi*mj+n) for x,mi,mj,b,s,n,d in zip(xx,cm,dm,bb,starts,ndiags,diff) )
            ii_kk = tuple( list(ii) + kk )
            out[ii] = np.dot( mat[ii_kk].flat, x[jj].flat )

        new_nrows = list(nrows).copy()

        for d,er in enumerate(nrows_extra):

            rows = new_nrows.copy()
            del rows[d]

            for n in range(er):
                for xx in np.ndindex(*rows):
                    xx = list(xx)
                    xx.insert(d, nrows[d]+n)

                    ii     = tuple(mi*pi + x for mi,pi,x in zip(cm, gpads, xx))
                    ee     = [max(x-l+1,0) for x,l in zip(xx, nrows)]
                    jj     = tuple( slice(b-d+(x+s%mj)//mi*mj, b-d+(x+s%mj)//mi*mj+n-e) for x,mi,mj,d,e,b,s,n in zip(xx, cm, dm, diff, ee,bb,starts, ndiags) )
                    kk     = [slice(None,n-e) for n,e in zip(ndiags, ee)]
                    ii_kk  = tuple( list(ii) + kk )
                    out[ii] = np.dot( mat[ii_kk].flat, x[jj].flat )

            new_nrows[d] += er
            
    # ...
    def transpose(self, conjugate=False):
        """ Create new StencilMatrix Mt, where domain and codomain are swapped
            with respect to original matrix M, and Mt_{ij} = M_{ji}.
        """
        # For clarity rename self
        M = self

        # If necessary, update ghost regions of original matrix M
        if not M.ghost_regions_in_sync:
            M.update_ghost_regions()

        # Create new matrix where domain and codomain are swapped
        Mt = StencilMatrix(M.codomain, M.domain, pads=self._pads, backend=self._backend)

        # Call low-level '_transpose' function (works on Numpy arrays directly)
        if conjugate:
            self._transpose_func(np.conjugate(M._data), Mt._data, **self._transpose_args)
        else:
            self._transpose_func(M._data, Mt._data, **self._transpose_args)
        return Mt

    @staticmethod
    def _transpose( M, Mt, nrows, ncols, gpads, pads, dm, cm, ndiags, ndiagsT, si, sk, sl):

        # NOTE:
        #  . Array M  index by [i1, i2, ..., k1, k2, ...]
        #  . Array Mt index by [j1, j2, ..., l1, l2, ...]

        #M[i,j-i+p]
        #Mt[j,i-j+p]

        diff   = [gp-p for gp,p in zip(gpads, pads)]
        for xx in np.ndindex( *nrows ):

            jj = tuple(m*p + x for m,p,x in zip(dm, gpads, xx) )

            for ll in np.ndindex( *ndiags ):

                ii = tuple( s + mi*(x//mj) + l + d for mj,mi,x,l,d,s in zip(dm,cm, xx, ll, diff, si))

                kk = tuple( s + x%mj-mj*(l//mi) for mj,mi,l,x,s in zip(dm, cm, ll, xx, sk))
                ll = tuple(l+s for l,s in zip(ll, sl))

                if all(k<n  and k>-1 for k,n in zip(kk,ndiagsT)) and\
                   all(l<n for l,n in zip(ll, ndiags)) and\
                   all(i<n for i,n in zip(ii, ncols)):
                    Mt[(*jj, *ll)] = M[(*ii, *kk)]

    # ...
    def toarray( self, **kwargs ):
        """ Convert to Numpy 2D array. """

        order     = kwargs.pop('order', 'C')
        with_pads = kwargs.pop('with_pads', False)

        if self.codomain.parallel and with_pads:
            coo = self._tocoo_parallel_with_pads(order=order)
        else:
            coo = self._tocoo_no_pads(order=order)

        return coo.toarray()

    # ...
    def tosparse( self, **kwargs ):
        """ Convert to any Scipy sparse matrix format. """

        order     = kwargs.pop('order', 'C')
        with_pads = kwargs.pop('with_pads', False)

        if self.codomain.parallel and with_pads:
            coo = self._tocoo_parallel_with_pads(order=order)
        else:
            coo = self._tocoo_no_pads(order=order)

        return coo

    #--------------------------------------
    # Overridden properties/methods
    #--------------------------------------
    def __neg__(self):
        return self.__mul__(-1)

    # ...
    def __mul__( self, a ):
        w = StencilMatrix( self._domain, self._codomain, self._pads, self._backend )
        w._data = self._data * a
        w._func = self._func
        w._args = self._args
        w._sync = self._sync
        return w

    #...
    def __add__(self, m):
        if isinstance(m, StencilMatrix):
            #assert isinstance(m, StencilMatrix)
            assert m._domain   is self._domain
            assert m._codomain is self._codomain
            assert m._pads     == self._pads

            if m._backend is not self._backend:
                msg = 'Adding two matrices with different backends is ambiguous - defaulting to backend of first addend'
                warnings.warn(msg, category=RuntimeWarning)
            
            w = StencilMatrix(self._domain, self._codomain, self._pads, self._backend)
            w._data = self._data  +  m._data
            w._func = self._func
            w._args = self._args
            w._sync = self._sync and m._sync
            return w
        else:
            return LinearOperator.__add__(self, m)

    #...
    def __sub__(self, m):
        if isinstance(m, StencilMatrix):
            #assert isinstance(m, StencilMatrix)
            assert m._domain   is self._domain
            assert m._codomain is self._codomain
            assert m._pads     == self._pads

            if m._backend is not self._backend:
                msg = 'Subtracting two matrices with different backends is ambiguous - defaulting to backend of the matrix we subtract from'
                warnings.warn(msg, category=RuntimeWarning)

            w = StencilMatrix(self._domain, self._codomain, self._pads, self._backend)
            w._data = self._data  -  m._data
            w._func = self._func
            w._args = self._args
            w._sync = self._sync and m._sync
            return w
        else:
            return LinearOperator.__sub__(self, m)

    #--------------------------------------
    # New properties/methods
    #--------------------------------------

    # TODO: check if this method is really needed!!
    def conjugate(self, out=None):
        if out is not None:
            assert isinstance(out, StencilMatrix)
            assert out.domain is self.domain
            assert out.codomain is self.codomain
        else:
            out = StencilMatrix(self.domain, self.codomain, pads=self.pads)
            out._func    = self._func
            out._args    = self._args
        np.conjugate(self._data, out=out._data, casting='no')
        return out

    # ...
    # TODO: check if this method is really needed!!
    def conj(self, out=None):
        return self.conjugate(out=out)

    # ...
    @property
    def pads( self ):
        return self._pads

    # ...
    @property
    def backend( self ):
        return self._backend

    # ...
    def __getitem__(self, key):
        index = self._getindex( key )
        return self._data[index]

    # ...
    def __setitem__(self, key, value):
        index = self._getindex( key )
        self._data[index] = value

    #...
    def max( self ):
        return self._data.max()

    #...
    def copy( self ):
        M = StencilMatrix( self.domain, self.codomain, self._pads, self._backend )
        M._data[:] = self._data[:]
        M._func    = self._func
        M._args    = self._args
        return M

    #...
    def __imul__(self, a):
        self._data *= a
        return self

    #...
    def __iadd__(self, m):
        if isinstance(m, StencilMatrix):
            #assert isinstance(m, StencilMatrix)
            assert m._domain   is self._domain
            assert m._codomain is self._codomain
            assert m._pads     == self._pads
            self._data += m._data
            self._sync  = m._sync and self._sync
            return self
        else:
            return LinearOperator.__add__(self, m)

    #...
    def __isub__(self, m):
        if isinstance(m, StencilMatrix):
            #assert isinstance(m, StencilMatrix)
            assert m._domain   is self._domain
            assert m._codomain is self._codomain
            assert m._pads     == self._pads
            self._data -= m._data
            self._sync  = m._sync and self._sync
            return self
        else:
            return LinearOperator.__sub__(self, m)

    #...
    def __abs__( self ):
        w = StencilMatrix( self._domain, self._codomain, self._pads, self._backend )
        w._data = abs(self._data)
        w._func = self._func
        w._args = self._args
        w._sync = self._sync
        return w

    #...
    def remove_spurious_entries( self ):
        """
        If any dimension is NOT periodic, make sure that the corresponding
        periodic corners are set to zero.

        """
        # TODO: access 'self._data' directly for increased efficiency

        ndim  = self._domain.ndim

        for direction in range(ndim):

            periodic = self._domain.periods[direction]

            if not periodic:

                nc = self._codomain.npts[direction]
                nd = self._domain.npts[direction]

                s = self._codomain.starts[direction]
                e = self._codomain.ends  [direction]
                p = self.pads  [direction]

                idx_front = [slice(None)]*direction
                idx_back  = [slice(None)]*(ndim-direction-1)

                # Top-right corner
                for i in range( max(0,s), min(p,e+1) ):
                    index = tuple( idx_front + [i]            + idx_back +
                                   idx_front + [slice(-p,-i)] + idx_back )
                    self[index] = 0

                # Bottom-left corner
                for i in range( max(nd-p,s), min(nc,e+1) ):
                    index = tuple( idx_front + [i]               + idx_back +
                                   idx_front + [slice(nd-i,p+1)] + idx_back )
                    self[index] = 0

    # ...
    def update_ghost_regions( self ):
        """
        Update ghost regions before performing non-local access to matrix
        elements (e.g. in matrix transposition).
        """
        ndim     = self._codomain.ndim
        parallel = self._codomain.parallel

        if parallel:
            if not self._codomain.cart.is_comm_null:
                # PARALLEL CASE: fill in ghost regions with data from neighbors
                self._synchronizer.start_update_ghost_regions( self._data, self._requests )
                self._synchronizer.end_update_ghost_regions( self._data , self._requests)
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._update_ghost_regions_serial()

        # Flag ghost regions as up-to-date
        self._sync = True

    # ...
    def exchange_assembly_data( self ):
        """
        Exchange assembly data.
        """
        ndim     = self._codomain.ndim
        parallel = self._codomain.parallel

        if self._codomain.parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self._synchronizer.start_exchange_assembly_data( self._data )
            self._synchronizer.end_exchange_assembly_data( self._data )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._exchange_assembly_data_serial()

        ndim = self._codomain.ndim
        for direction in range(ndim):
            idx_front = [slice(None)]*direction
            idx_back  = [slice(None)]*(ndim-direction-1)

            p        = self._codomain.pads   [direction]
            m        = self._codomain.shifts[direction]
            idx_from = tuple( idx_front + [ slice(-m*p,None) if (-m*p+p)!=0 else slice(-m*p,None)] + idx_back )
            self._data[idx_from] = 0.
            idx_from = tuple( idx_front + [ slice(0,m*p)] + idx_back )
            self._data[idx_from] = 0.

    # ...
    def _exchange_assembly_data_serial( self ):

        ndim     = self._codomain.ndim
        for direction in range(ndim):

            periodic = self._codomain.periods[direction]
            p        = self._codomain.pads   [direction]
            m        = self._codomain.shifts[direction]

            if periodic:
                idx_front = [slice(None)]*direction
                idx_back  = [slice(None)]*(ndim-direction-1)

                # Copy data from left to right
                idx_to   = tuple( idx_front + [slice( m*p, m*p+p)] + idx_back )
                idx_from = tuple( idx_front + [ slice(-m*p,-m*p+p) if (-m*p+p)!=0 else slice(-m*p,None)] + idx_back )
                self._data[idx_to] += self._data[idx_from]

    def diagonal(self):
        if self._diag_indices is None:
            cm    = self.codomain.shifts
            dm    = self.domain.shifts
            ss    = self.codomain.starts
            pp    = [compute_diag_len(p,mj,mi)-(p+1) for p,mi,mj in zip(self._pads, cm, dm)]
            nrows = tuple(e-s+1 for s,e in zip(self.codomain.starts, self.codomain.ends))
            indices = [np.zeros(np.product(nrows), dtype=int) for _ in range(2*len(nrows))]
            l = 0
            for xx in np.ndindex(*nrows):
                ii = [m*p+x for m,p,x in zip(self.domain.shifts, self.domain.pads, xx)]
                jj = [p+x+s-((x+s)//mi)*mj for (x,mi,mj,p,s) in zip(xx,cm,dm,pp,ss)]
                for k in range(len(nrows)):
                    indices[k][l] = ii[k]
                    indices[k+len(nrows)][l] = jj[k]
                l += 1
            self._diag_indices = tuple(indices)
        else:
            nrows   = tuple(e-s+1 for s,e in zip(self.codomain.starts, self.codomain.ends))

        return self._data[self._diag_indices].reshape(nrows)

    # ...
    def topetsc( self ):
        """ Convert to petsc data structure.
        """
        from psydac.linalg.topetsc import mat_topetsc
        mat = mat_topetsc( self )
        return mat

    #--------------------------------------
    # Private methods
    #--------------------------------------

    # ...
    def _getindex( self, key ):

        nd = self._ndim
        ii = key[:nd]
        kk = key[nd:]

        index = []

        for i,s,p,m in zip( ii, self._codomain.starts, self._codomain.pads, self._codomain.shifts ):
            x = self._shift_index( i, m*p-s )
            index.append( x )

        for k,p in zip( kk, self._pads ):
            l = self._shift_index( k, p )
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

    def tocoo_local( self, order='C' ):

        # Shortcuts
        sc = self._codomain.starts
        ec = self._codomain.ends
        pc = self._codomain.pads

        sd = self._domain.starts
        ed = self._domain.ends
        pd = self._domain.pads

        nd = self._ndim

        nr = [e-s+1 +2*p for s,e,p in zip(sc, ec, pc)]
        nc = [e-s+1 +2*p for s,e,p in zip(sd, ed, pd)]

        ravel_multi_index = np.ravel_multi_index

        # COO storage
        rows = []
        cols = []
        data = []

        local = tuple( [slice(p,-p) for p in pc] + [slice(None)] * nd )

        dd = [pdi-ppi for pdi,ppi in zip(pd, self._pads)]

        for (index, value) in np.ndenumerate( self._data[local] ):

            # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

            xx = index[:nd]  # ii is local
            ll = index[nd:]  # l=p+k

            ii = [x+p for x,p in zip(xx, pc)]
            jj = [(l+i+d)%n for (i,l,d,n) in zip(xx,ll,dd,nc)]

            I = ravel_multi_index( ii, dims=nr,  order=order )
            J = ravel_multi_index( jj, dims=nc,  order=order )

            rows.append( I )
            cols.append( J )
            data.append( value )

        M = coo_matrix(
                (data,(rows,cols)),
                shape = [np.prod(nr),np.prod(nc)],
                dtype = self._domain.dtype
        )

        M.eliminate_zeros()

        return M

    #...
    def _tocoo_no_pads( self , order='C'):

        # Shortcuts
        nr    = self._codomain.npts
        nd    = self._ndim
        nc    = self._domain.npts
        ss    = self._codomain.starts
        cpads = self._codomain.pads
        dm    = self._domain.shifts
        cm    = self._codomain.shifts

        pp = [np.int64(compute_diag_len(p,mj,mi)-(p+1)) for p,mi,mj in zip(self._pads, cm, dm)]

        # Range of data owned by local process (no ghost regions)
        local = tuple( [slice(mi*p,-mi*p) for p,mi in zip(cpads, cm)] + [slice(None)] * nd )
        size  = self._data[local].size

        # COO storage
        rows = np.zeros(size, dtype='int64')
        cols = np.zeros(size, dtype='int64')
        data = np.zeros(size, dtype=self.dtype)
        nrl = [np.int64(e-s+1) for s,e in zip(self.codomain.starts, self.codomain.ends)]
        ncl = [np.int64(i) for i in self._data.shape[nd:]]
        ss = [np.int64(i) for i in ss]
        nr = [np.int64(i) for i in nr]
        nc = [np.int64(i) for i in nc]
        dm = [np.int64(i) for i in dm]
        cm = [np.int64(i) for i in cm]
        cpads = [np.int64(i) for i in cpads]
        pp = [np.int64(i) for i in pp]

        func        = 'stencil2coo_{dim}d_{order}'.format(dim=nd, order=order)
        stencil2coo = eval(func)

        ind = stencil2coo(self._data, data, rows, cols, *nrl, *ncl, *ss, *nr, *nc, *dm, *cm, *cpads, *pp)
        M = coo_matrix(
                (data[:ind],(rows[:ind],cols[:ind])),
                shape = [np.prod(nr),np.prod(nc)],
                dtype = self.dtype)
        return M

    #...
    def _tocoo_parallel_with_pads( self , order='C'):

        # If necessary, update ghost regions
        if not self.ghost_regions_in_sync:
            self.update_ghost_regions()

        # Shortcuts
        nr = self._codomain.npts
        nc = self._domain.npts
        nd = self._ndim

        ss = self._codomain.starts
        ee = self._codomain.ends
        pp = self._pads
        pc = self._codomain.pads
        pd = self._domain.pads
        cc = self._codomain.periods

        ravel_multi_index = np.ravel_multi_index

        # COO storage
        rows = []
        cols = []
        data = []

        # List of rows (to avoid duplicate updates)
        I_list = []

        # Shape of row and diagonal spaces
        xx_dims = self._data.shape[:nd]
        ll_dims = self._data.shape[nd:]

        # Cycle over rows (x = p + i - s)
        for xx in np.ndindex( *xx_dims ):

            # Compute row multi-index with simple shift
            ii = [s + x - p for (s, x, p) in zip(ss, xx, pc)]

            # Apply periodicity where appropriate
            ii = [i - n if (c and i >= n and i - n < s) else
                  i + n if (c and i <  0 and i + n > e) else i
                  for (i, s, e, n, c) in zip(ii, ss, ee, nr, cc)]

            # Compute row flat index
            # Exclude values outside global limits of matrix
            try:
                I = ravel_multi_index( ii, dims=nr,  order=order )
            except ValueError:
                continue

            # If I is a new row, append it to list of rows
            # DO NOT update same row twice!
            if I not in I_list:
                I_list.append( I )
            else:
                continue

            # Cycle over diagonals (l = p + k)
            for ll in np.ndindex( *ll_dims ):

                # Compute column multi-index (k = j - i)
                jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,pp)]

                # Compute column flat index
                J = ravel_multi_index( jj, dims=nc,  order=order )

                # Extract matrix value
                value = self._data[(*xx, *ll)]

                # Append information to COO arrays
                rows.append( I )
                cols.append( J )
                data.append( value )

        # Create Scipy COO matrix
        M = coo_matrix(
                (data,(rows,cols)),
                shape = [np.prod(nr), np.prod(nc)],
                dtype = self._domain.dtype
        )

        M.eliminate_zeros()

        return M

    # ...
    @property
    def ghost_regions_in_sync( self ):
        return self._sync

    # ...
    # NOTE: this property must be set collectively
    @ghost_regions_in_sync.setter
    def ghost_regions_in_sync( self, value ):
        assert isinstance( value, bool )
        self._sync = value

    # ...
    def _update_ghost_regions_serial( self ):

        ndim     = self._codomain.ndim
        for direction in range( self._codomain.ndim ):

            periodic = self._codomain.periods[direction]
            p        = self._codomain.pads   [direction]

            idx_front = [slice(None)]*direction
            idx_back  = [slice(None)]*(ndim-direction-1 + ndim)

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
    def _prepare_transpose_args(self):

        #prepare the arguments for the transpose method
        V     = self.domain
        W     = self.codomain
        ssc   = W.starts
        eec   = W.ends
        ssd   = V.starts
        eed   = V.ends
        pads    = self._pads
        gpads = V.pads

        dm    = V.shifts
        cm    = W.shifts

        # Number of rows in the transposed matrix (along each dimension)
        nrows       = [e-s+1 for s,e in zip(ssd, eed)]
        ncols       = [e-s+2*m*p+1 for s,e,m,p in zip(ssc, eec, cm, gpads)]

        pp = pads
        ndiags, starts = list(zip(*[compute_diag_len(p,mi,mj, return_padding=True) for p,mi,mj in zip(pp,cm,dm)]))
        ndiagsT, _     = list(zip(*[compute_diag_len(p,mj,mi, return_padding=True) for p,mi,mj in zip(pp,cm,dm)]))

        diff   = [gp-p for gp,p in zip(gpads, pp)]

        sl   = [(s if mi>mj else 0) + (s%mi+mi//mj if mi<mj else 0)+(s if mi==mj else 0)\
                 for s,p,mi,mj in zip(starts,pp,cm,dm)]

        si   = [(mi*p-mi*(int(np.ceil((p+1)/mj))-1) if mi>mj else 0)+\
                 (mi*p-mi*(p//mi)+ d*(mi-1) if mi<mj else 0)+\
                 (mj*p-mj*(p//mi)+ d*(mi-1) if mi==mj else 0)\
                  for mi,mj,p,d in zip(cm, dm, pp, diff)]

        sk   = [n-1\
                 + (-(p%mj) if mi>mj else 0)\
                 + (-p+mj*(p//mi) if mi<mj  else 0)\
                 + (-p+mj*(p//mi) if mi==mj else 0)\
                 for mi,mj,n,p in zip(cm, dm, ndiagsT, pp)]

        args = {}
        args['nrows']   = tuple(nrows)
        args['ncols']   = tuple(ncols)
        args['gpads']   = tuple(gpads)
        args['pads']    = tuple(pads)
        args['dm']      = tuple(dm)
        args['cm']      = tuple(cm)
        args['ndiags']  = tuple(ndiags)
        args['ndiagsT'] = tuple(ndiagsT)
        args['si']      = tuple(si)
        args['sk']      = tuple(sk)
        args['sl']      = tuple(sl)
        return args

    # ...
    def set_backend(self, backend):
        from psydac.api.ast.linalg import LinearOperatorDot, TransposeOperator
        self._backend         = backend
        self._args            = self._dotargs_null.copy()
        self._transpose_args  = self._transpose_args_null.copy()

        if self._backend is None:
            self._func           = self._dot
            self._transpose_func = self._transpose
        else:
            transpose = TransposeOperator(self._ndim, backend=frozenset(backend.items()))
            self._transpose_func = transpose.func

            nrows   = self._transpose_args.pop('nrows')
            ncols   = self._transpose_args.pop('ncols')
            gpads   = self._transpose_args.pop('gpads')
            pads    = self._transpose_args.pop('pads')
            dm      = self._transpose_args.pop('dm')
            cm      = self._transpose_args.pop('cm')
            ndiags  = self._transpose_args.pop('ndiags')
            ndiagsT = self._transpose_args.pop('ndiagsT')
            si      = self._transpose_args.pop('si')
            sk      = self._transpose_args.pop('sk')
            sl      = self._transpose_args.pop('sl')

            args = dict([('n{i}',nrows),('nc{i}', ncols),('gp{i}', gpads),('p{i}',pads ),
                                ('dm{i}', dm),('cm{i}', cm),('nd{i}', ndiags),
                                ('ndT{i}', ndiagsT),('si{i}', si),('sk{i}', sk),('sl{i}', sl)])

            for arg_name, arg_val in args.items():
                for i in range(len(nrows)):
                    self._transpose_args[arg_name.format(i=i+1)] = np.int64(arg_val[i]) if isinstance(arg_val[i], int) else arg_val[i]

            if self.domain.parallel:
                comm = self.codomain.cart.comm
                if self.domain == self.codomain:
                    # In this case nrows_extra[i] == 0 for all i
                    dot = LinearOperatorDot(self._ndim,
                                    block_shape = (1,1),
                                    keys = ((0,0),),
                                    comm = comm,
                                    backend=frozenset(backend.items()),
                                    nrows_extra = (self._args['nrows_extra'],),
                                    gpads=(self._args['gpads'],),
                                    pads=(self._args['pads'],),
                                    dm = (self._args['dm'],),
                                    cm = (self._args['cm'],),
                                    dtype=self.dtype)

                    starts = self._args.pop('starts')
                    nrows  = self._args.pop('nrows')

                    self._args.pop('nrows_extra')
                    self._args.pop('gpads')
                    self._args.pop('pads')
                    self._args.pop('dm')
                    self._args.pop('cm')

                    for i in range(len(nrows)):
                        self._args['s00_{i}'.format(i=i+1)] = np.int64(starts[i])

                    for i in range(len(nrows)):
                        self._args['n00_{i}'.format(i=i+1)] = np.int64(nrows[i])

                else:
                    dot = LinearOperatorDot(self._ndim,
                                            block_shape = (1,1),
                                            keys = ((0,0),),
                                            comm = comm,
                                            backend=frozenset(backend.items()),
                                            gpads=(self._args['gpads'],),
                                            pads=(self._args['pads'],),
                                            dm = (self._args['dm'],),
                                            cm = (self._args['cm'],),
                                            dtype=self.dtype)

                    starts      = self._args.pop('starts')
                    nrows       = self._args.pop('nrows')
                    nrows_extra = self._args.pop('nrows_extra')

                    self._args.pop('gpads')
                    self._args.pop('pads')
                    self._args.pop('dm')
                    self._args.pop('cm')

                    for i in range(len(nrows)):
                        self._args['s00_{i}'.format(i=i+1)] = np.int64(starts[i])

                    for i in range(len(nrows)):
                        self._args['n00_{i}'.format(i=i+1)] = np.int64(nrows[i])

                    for i in range(len(nrows)):
                        self._args['ne00_{i}'.format(i=i+1)] = np.int64(nrows_extra[i])

            else:
                dot = LinearOperatorDot(self._ndim,
                                        block_shape = (1,1),
                                        keys = ((0,0),),
                                        comm = None,
                                        backend=frozenset(backend.items()),
                                        starts = (tuple(self._args['starts']),),
                                        nrows=(tuple(self._args['nrows']),),
                                        nrows_extra=(self._args['nrows_extra'],),
                                        gpads=(self._args['gpads'],),
                                        pads=(self._args['pads'],),
                                        dm = (self._args['dm'],),
                                        cm = (self._args['cm'],),
                                        dtype=self.dtype)
                self._args.pop('nrows')
                self._args.pop('nrows_extra')
                self._args.pop('gpads')
                self._args.pop('pads')
                self._args.pop('starts')
                self._args.pop('dm')
                self._args.pop('cm')

            self._func = dot.func

#===============================================================================
# TODO [YG, 28.01.2021]:
# - Check if StencilMatrix should be subclassed
# - Reimplement magic methods (some are simply copied from StencilMatrix)
def flip_axis(index, n):
    s = n-index.start-1
    e = n-index.stop-1 if n>index.stop else None
    return slice(s,e,-1)

class StencilInterfaceMatrix(LinearOperator):
    """
    Matrix in n-dimensional stencil format for an interface.

    This is a linear operator that maps elements of stencil vector space V to
    elements of stencil vector space W.

    Parameters
    ----------
    V   : psydac.linalg.stencil.StencilVectorSpace
          Domain of the new linear operator.

    W   : psydac.linalg.stencil.StencilVectorSpace
          Codomain of the new linear operator.

    s_d : int
          The starting index of the domain.

    s_c : int
          The starting index of the codomain.

    d_axis : int
          The axis of the Interface of the domain.

    c_axis : int
          The axis of the Interface of the codomain.

    d_ext : int
          The extremity of the domain Interface space.
          the values must be 1 or -1.

    c_ext : int
          The extremity of the codomain Interface space.
          the values must be 1 or -1.

    dim : int
          The axis of the interface.

    pads: <list|tuple>
          Padding of the linear operator.

    """
    def __init__( self, V, W, s_d, s_c, d_axis, c_axis, d_ext, c_ext, *, flip=None, pads=None, backend=None ):

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W, StencilVectorSpace )
        assert W.pads == V.pads

        Vin = V.interfaces[d_axis, d_ext]

        if pads is not None:
            for p,vp in zip(pads, Vin.pads):
                assert p<=vp

        self._pads     = pads or tuple(Vin.pads)
        dims           = list(W.shape)

        if W.parent_ends[c_axis] is not None:
            diff = min(1, W.parent_ends[c_axis]-W.ends[c_axis])
        else:
            diff = 0

        dims[c_axis]   = W.pads[c_axis] + 1-diff + 2*W.shifts[c_axis]*W.pads[c_axis]
        diags          = [compute_diag_len(p, md, mc) for p,md,mc in zip(self._pads, Vin.shifts, W.shifts)]
        self._data     = np.zeros( dims+diags, dtype=W.dtype )

        # Parallel attributes
        if W.parallel and not isinstance(W.cart, InterfaceCartDecomposition):
            if W.cart.is_comm_null:return
            # Create data exchanger for ghost regions
            self._synchronizer = get_data_exchanger(
                cart        = W.cart,
                dtype       = W.dtype,
                coeff_shape = diags,
                assembly    = True,
                axis        = c_axis,
                shape       = self._data.shape
            )

        self._flip        = tuple([1]*len(dims) if flip is None else flip)
        self._permutation = list(range(len(dims)))
        self._permutation[d_axis], self._permutation[c_axis] = self._permutation[c_axis], self._permutation[d_axis]
        self._domain         = V
        self._codomain       = W
        self._domain_axis    = d_axis
        self._codomain_axis  = c_axis
        self._domain_ext     = d_ext
        self._codomain_ext   = c_ext
        self._domain_start   = s_d
        self._codomain_start = s_c
        self._ndim        = len( dims )
        self._backend     = None


        # Prepare the arguments for the dot product method
        nd  = [(ej-sj+2*gp*mj-mj*p-gp)//mj*mi+1 for sj,ej,mj,mi,p,gp in zip(Vin.starts, Vin.ends, Vin.shifts, W.shifts, self._pads, Vin.pads)]
        nc  = [ei-si+1 for si,ei,mj,p in zip(W.starts, W.ends, Vin.shifts, self._pads)]

        # Number of rows in matrix (along each dimension)
        nrows         = [min(ni,nj) for ni,nj  in zip(nc, nd)]
        nrows_extra   = [max(0,ni-nj) for ni,nj in zip(nc, nd)]
        nrows_extra[c_axis] = max(W.npts[c_axis]-Vin.npts[c_axis], 0)
        nrows[c_axis] = W.pads[c_axis] + 1-diff-nrows_extra[c_axis]


        args                 = {}
        args['starts']       = tuple(Vin.starts)
        args['nrows']        = tuple(nrows)
        args['nrows_extra']  = tuple(nrows_extra)
        args['gpads']        = tuple(Vin.pads)
        args['pads']         = tuple(self._pads)
        args['dm']           = tuple(Vin.shifts)
        args['cm']           = tuple(W.shifts)
        args['c_axis']       = c_axis
        args['d_start']      = self._domain_start
        args['c_start']      = self._codomain_start
        args['flip']         = self._flip
        args['permutation']  = self._permutation

        self._dotargs_null = args
        self._args         = args.copy()
        self._func         = self._dot

        self._transpose_args_null = self._prepare_transpose_args()
        self._transpose_args      = self._transpose_args_null.copy()
        self._transpose_func      = self._transpose

        if backend is None:
            backend = PSYDAC_BACKENDS.get(os.environ.get('PSYDAC_BACKEND'))

        if backend:
            self.set_backend(backend)

        # Flag ghost regions as not up-to-date (conservative choice)
        self._sync = False

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def domain( self ):
        return self._domain

    # ...
    @property
    def codomain( self ):
        return self._codomain

    # ...
    @property
    def dtype( self ):
        return self.domain.dtype

    # ...
    def dot( self, v, out=None ):

        assert isinstance( v, StencilVector )
        assert v.space is self.domain

        # Necessary if vector space is distributed across processes

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self.codomain
            out[(slice(None,None),)*v.space.ndim] = 0.
        else:
            out = StencilVector( self.codomain )

        # Necessary if vector space is distributed across processes
        if not v.ghost_regions_in_sync and not v.space.parallel:
            v.update_ghost_regions()

        self._func(self._data, v._interface_data[self._domain_axis, self._domain_ext], out._data, **self._args)
        # IMPORTANT: flag that ghost regions are not up-to-date
        out.ghost_regions_in_sync = False
        return out

    # ...
    @staticmethod
    def _dot(mat, v, out, starts, nrows, nrows_extra, gpads, pads, dm, cm, c_axis, d_start, c_start, flip, permutation):

        # Index for k=i-j
        nrows      = list(nrows)
        ndim       = len(v.shape)
        kk         = [slice(None)]*ndim
        diff       = [xp-p for xp,p in zip(gpads, pads)]

        ndiags, _ = list(zip(*[compute_diag_len(p,mj,mi, return_padding=True) for p,mi,mj in zip(pads,cm,dm)]))
        bb        = [p*m+p+1-n-s%m for p,m,n,s in zip(gpads, dm, ndiags, starts)]
        nn        = v.shape

        for xx in np.ndindex( *nrows ):
            ii    = [ mi*pi + x for mi,pi,x in zip(cm, gpads, xx) ]
            jj    = tuple( slice(b-d+(x+s%mj)//mi*mj,b-d+(x+s%mj)//mi*mj+n) for x,mi,mj,b,s,n,d in zip(xx,cm,dm,bb,starts,ndiags,diff) )
            jj    = [flip_axis(i,n) if f==-1 else i for i,f,n in zip(jj,flip,nn)]
            jj    = tuple(jj[i] for i in permutation)
            ii_kk = tuple( ii + kk )

            ii[c_axis] += c_start
            out[tuple(ii)] = np.dot( mat[ii_kk].flat, v[jj].flat )


        new_nrows = nrows.copy()
        for d,er in enumerate(nrows_extra):

            rows = new_nrows.copy()
            del rows[d]

            for n in range(er):
                for xx in np.ndindex(*rows):
                    xx = list(xx)
                    xx.insert(d, nrows[d]+n)

                    ii     = [mi*pi + x for mi,pi,x in zip(cm, gpads, xx)]
                    ee     = [max(x-l+1,0) for x,l in zip(xx, nrows)]
                    jj     = tuple( slice(b-d+(x+s%mj)//mi*mj, b-d+(x+s%mj)//mi*mj+n-e) for x,mi,mj,d,e,b,s,n in zip(xx, cm, dm, diff, ee, bb, starts, ndiags) )
                    jj     = [flip_axis(i,n) if f==-1 else i for i,f,n in zip(jj, flip, nn)]
                    jj     = tuple(jj[i] for i in permutation)
                    kk     = [slice(None,n-e) for n,e in zip(ndiags, ee)]
                    ii_kk  = tuple( ii + kk )
                    ii[c_axis] += c_start
                    out[tuple(ii)] = np.dot( mat[ii_kk].flat, v[jj].flat )

            new_nrows[d] += er

    # ...
    def transpose( self, conjugate=False, Mt=None):
        """ Create new StencilInterfaceMatrix Mt, where domain and codomain are swapped
            with respect to original matrix M, and Mt_{ij} = M_{ji}.
        """

        # For clarity rename self
        M = self

        if Mt is None:
            # Create new matrix where domain and codomain are swapped

            Mt = StencilInterfaceMatrix(M.codomain, M.domain, M.codomain_start, M.domain_start, M.codomain_axis, M.domain_axis, M.codomain_ext, M.domain_ext,
                                        flip=M.flip, pads=M.pads, backend=M.backend)

        # Call low-level '_transpose' function (works on Numpy arrays directly)
        if conjugate:
            M._transpose_func(np.conjugate(M._data), Mt._data, **M._transpose_args)
        else:
            M._transpose_func(M._data, Mt._data, **M._transpose_args)
        return Mt

    @staticmethod
    def _transpose( M, Mt, nrows, ncols, gpads, pads, dm, cm, ndiags, ndiagsT, si, sk, sl):

        # NOTE:
        #  . Array M  index by [i1, i2, ..., k1, k2, ...]
        #  . Array Mt index by [j1, j2, ..., l1, l2, ...]

        #M[i,j-i+p]
        #Mt[j,i-j+p]

        diff       = [gp-p for gp,p in zip(gpads, pads)]

        for xx in np.ndindex( *nrows ):
            jj = tuple(m*p + x for m,p,x in zip(dm, gpads, xx) )
            for ll in np.ndindex( *ndiags ):
                ii = tuple( s + mi*(x//mj) + l + d for mj,mi,x,l,d,s in zip(dm,cm, xx, ll, diff, si))
                kk = tuple( s + x%mj-mj*(l//mi) for mj,mi,l,x,s in zip(dm, cm, ll, xx, sk))
                ll = tuple(l+s for l,s in zip(ll, sl))

                if all(k<n  and k>-1 for k,n in zip(kk,ndiagsT)) and\
                   all(l<n for l,n in zip(ll, ndiags)) and\
                   all(i<n for i,n in zip(ii, ncols)):
                    Mt[(*jj, *ll)] = M[(*ii, *kk)]

    def _prepare_transpose_args(self):

        #prepare the arguments for the transpose method
        V     = self.domain
        W     = self.codomain
        ssc   = W.starts
        eec   = W.ends
        ssd   = V.interfaces[self._domain_axis, self._domain_ext].starts
        eed   = V.interfaces[self._domain_axis, self._domain_ext].ends
        pads  = self._pads
        gpads = V.pads
        dm    = V.shifts
        cm    = W.shifts
        dim   = self._codomain_axis

        # Number of rows in the transposed matrix (along each dimension)
        nrows       = [e-s+1 for s,e in zip(ssd, eed)]
        ncols       = [e-s+1+2*m*p for s,e,m,p in zip(ssc, eec, cm, gpads)]

        pp = pads
        ndiags, starts = list(zip(*[compute_diag_len(p,mi,mj, return_padding=True) for p,mi,mj in zip(pp,cm,dm)]))
        ndiagsT, _     = list(zip(*[compute_diag_len(p,mj,mi, return_padding=True) for p,mi,mj in zip(pp,cm,dm)]))

        diff   = [gp-p for gp,p in zip(gpads, pp)]

        sl   = [(s if mi>mj else 0) + (s%mi+mi//mj if mi<mj else 0)+(s if mi==mj else 0)\
                 for s,p,mi,mj in zip(starts,pp,cm,dm)]

        si   = [(mi*p-mi*(int(np.ceil((p+1)/mj))-1) if mi>mj else 0)+\
                 (mi*p-mi*(p//mi)+ d*(mi-1) if mi<mj else 0)+\
                 (mj*p-mj*(p//mi)+ d*(mi-1) if mi==mj else 0)\
                  for mi,mj,p,d in zip(cm, dm, pp, diff)]

        sk   = [n-1\
                 + (-(p%mj) if mi>mj else 0)\
                 + (-p+mj*(p//mi) if mi<mj  else 0)\
                 + (-p+mj*(p//mi) if mi==mj else 0)\
                 for mi,mj,n,p in zip(cm, dm, ndiagsT, pp)]


        if V.parent_ends[dim] is not None:
            diff_r = min(1,V.parent_ends[dim]-V.ends[dim])
        else:
            diff_r = 0

        if W.parent_ends[dim] is not None:
            diff_c = min(1,W.parent_ends[dim]-W.ends[dim])
        else:
            diff_c = 0

        nrows[dim]  = pads[dim] + 1 - diff_r
        ncols[dim]  = pads[dim] + 1 - diff_c + 2*cm[dim]*pads[dim]

        args = {}
        args['nrows']   = tuple(nrows)
        args['ncols']   = tuple(ncols)
        args['gpads']   = tuple(gpads)
        args['pads']    = tuple(pads)
        args['dm']      = tuple(dm)
        args['cm']      = tuple(cm)
        args['ndiags']  = tuple(ndiags)
        args['ndiagsT'] = tuple(ndiagsT)
        args['si']      = tuple(si)
        args['sk']      = tuple(sk)
        args['sl']      = tuple(sl)
        return args
    # ...
    def toarray( self, **kwargs ):

        order     = kwargs.pop('order', 'C')
        with_pads = kwargs.pop('with_pads', False)

        if self.codomain.parallel and with_pads:
            coo = self._tocoo_parallel_with_pads()
        else:
            coo = self._tocoo_no_pads()

        return coo.toarray()

    # ...
    def tosparse( self, **kwargs ):

        order     = kwargs.pop('order', 'C')
        with_pads = kwargs.pop('with_pads', False)

        if self.codomain.parallel and with_pads:
            coo = self._tocoo_parallel_with_pads()
        else:
            coo = self._tocoo_no_pads()

        return coo

    #...
    def copy( self ):
        M = StencilInterfaceMatrix( self._domain, self._codomain,
                                    self._domain_start, self._codomain_start,
                                    self._domain_axis, self._codomain_axis,
                                    self._domain_ext, self._codomain_ext,
                                    flip=self._flip, pads=self._pads,
                                    backend=self._backend )
        M._data[:] = self._data[:]
        return M

    # ...
    def __neg__(self):
        return self.__mul__(-1)

    #...
    def __mul__( self, a ):
        w = self.copy()
        w._data *= a
        w._sync = self._sync
        return w

    #...
    def __add__(self, m):
        raise NotImplementedError('TODO: StencilInterfaceMatrix.__add__')

    #...
    def __sub__(self, m):
        raise NotImplementedError('TODO: StencilInterfaceMatrix.__sub__')

    #...
    def __imul__(self, a):
        self._data *= a

    #...
    def __iadd__(self, m):
        raise NotImplementedError('TODO: StencilInterfaceMatrix.__iadd__')

    #...
    def __isub__(self, m):
        raise NotImplementedError('TODO: StencilInterfaceMatrix.__isub__')

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------

    # ...
    @property
    def domain_axis( self ):
        return self._domain_axis

    # ...
    @property
    def codomain_axis( self ):
        return self._codomain_axis

    # ...
    @property
    def domain_ext( self ):
        return self._domain_ext

    # ...
    @property
    def codomain_ext( self ):
        return self._codomain_ext

    # ...
    @property
    def domain_start( self ):
        return self._domain_start

    # ...
    @property
    def codomain_start( self ):
        return self._codomain_start

    # ...
    @property
    def dim( self ):
        return self._ndim

    # ...
    @property
    def flip( self ):
        return self._flip

    # ...
    @property
    def permutation( self ):
        return self._permutation

    # ...
    @property
    def pads( self ):
        return self._pads

    # ...
    @property
    def T(self):
        return self.transpose()

    # ...
    def __getitem__(self, key):
        index = self._getindex( key )
        return self._data[index]

    # ...
    def __setitem__(self, key, value):
        index = self._getindex( key )
        self._data[index] = value

    #...
    def max( self ):
        return self._data.max()

    # ...
    @property
    def backend( self ):
        return self._backend

    #--------------------------------------
    # Private methods
    #--------------------------------------
    def _getindex( self, key ):

        nd = self._ndim
        ii = key[:nd]
        kk = key[nd:]

        index = []

        for i,s,p in zip( ii, self._codomain.starts, self._codomain.pads ):
            x = self._shift_index( i, p-s )
            index.append( x )

        for k,p in zip( kk, self._pads ):
            l = self._shift_index( k, p )
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

    #...
    def _tocoo_no_pads( self ):
        # Shortcuts
        nr  = self.codomain.npts
        nc  = self.domain.npts
        ss  = self.codomain.starts
        pp  = self.codomain.pads
        nd  = len(pp)

        dim = self._codomain_axis

        flip        = self.flip
        permutation = self.permutation
        c_start     = self.codomain_start
        d_start     = self.domain_start
        dm          = self.domain.shifts
        cm          = self.codomain.shifts

        ravel_multi_index = np.ravel_multi_index

        # COO storage
        rows = []
        cols = []
        data = []
        # Range of data owned by local process (no ghost regions)
        local = tuple( [slice(m*p,-m*p) for m,p in zip(cm, pp)] + [slice(None)] * nd )
        pp = [compute_diag_len(p,mj,mi)-(p+1) for p,mi,mj in zip(self._pads, cm, dm)]

        for (index,value) in np.ndenumerate( self._data[local] ):
            if value:
                # index = [i1, i2, ..., p1+j1-i1, p2+j2-i2, ...]

                xx = index[:nd]  # x=i-s
                ll = index[nd:]  # l=p+k

                ii = [s+x for s,x in zip(ss,xx)]
                di = [i//m for i,m in zip(ii,cm)]

                jj = [(i*m+l-p)%n for (i,m,l,n,p) in zip(di,dm,ll,nc,pp)]

                ii[dim] += c_start
                jj[dim] += d_start

                jj = [n-j-1 if f==-1 else j for j,f,n in zip(jj,flip,nc)]

                jj = [jj[i] for i in permutation]

                I = ravel_multi_index( ii, dims=nr, order='C' )
                J = ravel_multi_index( jj, dims=nc, order='C' )

                rows.append( I )
                cols.append( J )
                data.append( value )

        M = coo_matrix(
                    (data,(rows,cols)),
                    shape = [np.prod(nr),np.prod(nc)],
                    dtype = self.domain.dtype)

        return M

    # ...
    @property
    def ghost_regions_in_sync( self ):
        return self._sync

    # ...
    # NOTE: this property must be set collectively
    @ghost_regions_in_sync.setter
    def ghost_regions_in_sync( self, value ):
        assert isinstance( value, bool )
        self._sync = value

    # ...
    def _update_ghost_regions_serial( self, direction: int ):

        if direction is None:
            for d in range( self._codomain.ndim ):
                self._update_ghost_regions_serial( d )
            return

        ndim     = self._codomain.ndim
        periodic = self._codomain.periods[direction]
        p        = self._codomain.pads   [direction]

        idx_front = [slice(None)]*direction
        idx_back  = [slice(None)]*(ndim-direction-1 + ndim)

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
    def exchange_assembly_data( self ):
        """
        Exchange assembly data.
        """
        ndim     = self._codomain.ndim
        parallel = self._codomain.parallel

        if self._codomain.parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self._synchronizer.start_exchange_assembly_data( self._data )
            self._synchronizer.end_exchange_assembly_data( self._data )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._exchange_assembly_data_serial()

    # ...
    def _exchange_assembly_data_serial( self ):

        ndim     = self._codomain.ndim
        for direction in range(ndim):
            if direction == self._codomain_axis:continue
            periodic = self._codomain.periods[direction]
            p        = self._codomain.pads   [direction]
            m        = self._codomain.shifts[direction]

            if periodic:
                idx_front = [slice(None)]*direction
                idx_back  = [slice(None)]*(ndim-direction-1)

                # Copy data from left to right
                idx_to   = tuple( idx_front + [slice( m*p, m*p+p)] + idx_back )
                idx_from = tuple( idx_front + [ slice(-m*p,-m*p+p) if (-m*p+p)!=0 else slice(-m*p,None)] + idx_back )
                self._data[idx_to] += self._data[idx_from]

    def set_backend(self, backend):
        from psydac.api.ast.linalg import LinearOperatorDot, InterfaceTransposeOperator
        self._backend         = backend
        self._args            = self._dotargs_null.copy()
        self._transpose_args  = self._transpose_args_null.copy()

        if self._backend is None:
            self._func           = self._dot
            self._transpose_func = self._transpose
        else:
            transpose = InterfaceTransposeOperator(self._ndim, backend=frozenset(backend.items()))
            self._transpose_func = transpose.func

            nrows   = self._transpose_args.pop('nrows')
            ncols   = self._transpose_args.pop('ncols')
            gpads   = self._transpose_args.pop('gpads')
            pads    = self._transpose_args.pop('pads')
            dm      = self._transpose_args.pop('dm')
            cm      = self._transpose_args.pop('cm')
            ndiags  = self._transpose_args.pop('ndiags')
            ndiagsT = self._transpose_args.pop('ndiagsT')
            si      = self._transpose_args.pop('si')
            sk      = self._transpose_args.pop('sk')
            sl      = self._transpose_args.pop('sl')

            args = dict([('n{i}',nrows),('nc{i}', ncols),('gp{i}', gpads),('p{i}',pads ),
                          ('dm{i}', dm),('cm{i}', cm), ('nd{i}', ndiags),('ndT{i}', ndiagsT),('si{i}', si),
                          ('sk{i}', sk),('sl{i}', sl)])

            self._transpose_args            = {}
            for arg_name, arg_val in args.items():
                for i in range(len(nrows)):
                    self._transpose_args[arg_name.format(i=i+1)] =  np.int64(arg_val[i])

            if self.domain.parallel:

                comm = self.domain.interfaces[self._domain_axis, self._domain_ext].cart.local_comm

                if self.domain == self.codomain:
                    # In this case nrows_extra[i] == 0 for all i
                    dot = LinearOperatorDot(self._ndim,
                                    block_shape = (1,1),
                                    keys = ((0,0),),
                                    comm = comm,
                                    backend=frozenset(backend.items()),
                                    nrows_extra=(self._args['nrows_extra'],),
                                    gpads=(self._args['gpads'],),
                                    pads=(self._args['pads'],),
                                    dm = (self._args['dm'],),
                                    cm = (self._args['cm'],),
                                    interface=True,
                                    flip_axis=self._flip,
                                    interface_axis=self._codomain_axis,
                                    d_start=(self._domain_start,),
                                    c_start=(self._codomain_start,))

                    starts = self._args.pop('starts')
                    nrows  = self._args.pop('nrows')

                    self._args = {}
                    for i in range(len(nrows)):
                        self._args['s00_{i}'.format(i=i+1)] = np.int64(starts[i])

                    for i in range(len(nrows)):
                        self._args['n00_{i}'.format(i=i+1)] = np.int64(nrows[i])

                else:
                    dot = LinearOperatorDot(self._ndim,
                                            block_shape = (1,1),
                                            keys = ((0,0),),
                                            comm = comm,
                                            backend=frozenset(backend.items()),
                                            gpads=(self._args['gpads'],),
                                            pads=(self._args['pads'],),
                                            dm = (self._args['dm'],),
                                            cm = (self._args['cm'],),
                                            interface=True,
                                            flip_axis=self._flip,
                                            interface_axis=self._codomain_axis,
                                            d_start=(self._domain_start,),
                                            c_start=(self._codomain_start,))

                    starts      = self._args.pop('starts')
                    nrows       = self._args.pop('nrows')
                    nrows_extra = self._args.pop('nrows_extra')

                    self._args = {}

                    for i in range(len(nrows)):
                        self._args['s00_{i}'.format(i=i+1)] = np.int64(starts[i])

                    for i in range(len(nrows)):
                        self._args['n00_{i}'.format(i=i+1)] =  np.int64(nrows[i])

                    for i in range(len(nrows)):
                        self._args['ne00_{i}'.format(i=i+1)] =  np.int64(nrows_extra[i])

            else:
                dot = LinearOperatorDot(self._ndim,
                                        block_shape = (1,1),
                                        keys = ((0,0),),
                                        comm = None,
                                        backend=frozenset(backend.items()),
                                        starts = (tuple(self._args['starts']),),
                                        nrows=(self._args['nrows'],),
                                        nrows_extra=(self._args['nrows_extra'],),
                                        gpads=(self._args['gpads'],),
                                        pads=(self._args['pads'],),
                                        dm = (self._args['dm'],),
                                        cm = (self._args['cm'],),
                                        interface=True,
                                        flip_axis=self._flip,
                                        interface_axis=self._codomain_axis,
                                        d_start=(self._domain_start,),
                                        c_start=(self._codomain_start,))

                self._args = {}

            self._func = dot.func

#===============================================================================
from psydac.api.settings   import PSYDAC_BACKENDS
del VectorSpace, Vector
