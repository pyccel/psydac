# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import os
from collections import OrderedDict
import warnings

import numpy as np
from scipy.sparse import coo_matrix
from mpi4py import MPI

from psydac.linalg.basic   import VectorSpace, Vector, Matrix
from psydac.ddm.cart       import find_mpi_type, CartDecomposition, CartDataExchanger

__all__ = ['StencilVectorSpace','StencilVector','StencilMatrix', 'StencilInterfaceMatrix']

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

    - serial  : StencilVectorSpace( npts, pads, periods, dtype=float )
    - parallel: StencilVectorSpace( cart, dtype=float )

    Parameters
    ----------
    npts : tuple-like (int)
        Number of entries along each direction
        (= global dimensions of vector space).

    pads : tuple-like (int)
        Padding p along each direction (number of diagonals is 2*p+1).

    periods : tuple-like (bool)
        Periodicity along each direction.

    dtype : type
        Type of scalar entries.

    cart : psydac.ddm.cart.CartDecomposition
        Tensor-product grid decomposition according to MPI Cartesian topology.

    """
    def __init__( self, *args, **kwargs ):

        if len(args) == 1 or ('cart' in kwargs):
            self._init_parallel( *args, **kwargs )
        else:
            self._init_serial  ( *args, **kwargs )

    # ...
    def _init_serial( self, npts, pads, periods, shifts=None, dtype=float ):

        if shifts is None:shifts = tuple(1 for _ in pads)

        assert len(npts) == len(pads) == len(periods) == len(shifts)
        self._parallel = False

        # Sequential attributes
        self._starts        = tuple( 0   for n in npts )
        self._ends          = tuple( n-1 for n in npts )
        self._pads          = tuple( pads )
        self._periods       = tuple( periods )
        self._shifts        = tuple( shifts )
        self._dtype         = dtype
        self._ndim          = len( npts )
        self._parent_starts = tuple([None]*self._ndim)
        self._parent_ends   = tuple([None]*self._ndim)
        self._reduced       = [False]*self._ndim

        # Global dimensions of vector space
        self._npts       = tuple( npts )
        # Local dimensions of vector space
        self._local_npts = tuple( npts )

    # ...
    def _init_parallel( self, cart, dtype=float ):

        assert isinstance( cart, CartDecomposition )
        self._parallel = True

        # Sequential attributes
        self._starts        = cart.starts
        self._ends          = cart.ends
        self._parent_starts = cart.parent_starts
        self._parent_ends   = cart.parent_ends
        self._reduced       = cart.reduced
        self._pads          = cart.pads
        self._periods       = cart.periods
        self._shifts        = cart.shifts
        self._dtype         = dtype
        self._ndim          = len(cart.starts)

        # Global dimensions of vector space
        self._npts       = cart.npts
        # Local dimensions of vector space
        self._local_npts = tuple( e-s+1 for s,e in zip(cart.starts, cart.ends) )

        # Parallel attributes
        self._cart         = cart
        self._mpi_type     = find_mpi_type( dtype )
        self._synchronizer = CartDataExchanger( cart, dtype , assembly=True)

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

# NOTE [YG, 09.03.2021]: the equality comparison "==" is removed because we
# prefer using the identity comparison "is" as far as possible.
#    # ...
#    def __eq__(self, V):
#
#        if self.parallel and V.parallel:
#            cond = self._dtype == V._dtype
#            cond = cond and self._cart ==  V._cart
#            return cond
#
#        elif not self.parallel and not V.parallel:
#            cond = self.npts == V.npts
#            cond = cond and self.pads == V.pads
#            cond = cond and self.periods == V.periods
#            cond = cond and self.dtype == V.dtype
#            return cond
#        else:
#            return False

    def reduce_elements(self, axes, n_elements):
        """ Compute the reduced space.

        Parameters
        ----------
        axes: tuple_like (int)
            Reduced directions.

        n_elements: tuple_like (int)
            Number of elements to substract from the space.

        Returns
        -------
        v: StencilVectorSpace
            The reduced space.
        """
        assert not self.parallel
        npts   = [n-ne for n,ne in zip(self.npts, n_elements)]
        shifts = [max(1,m-1) for m in self.shifts]

        v = StencilVectorSpace(npts=npts, pads=self.pads, periods=self.periods, shifts=shifts)
        v._parent_starts = self.starts
        v._parent_ends   = self.ends
        v._reduced       = tuple(a in axes for a in range(self._ndim))
        return v
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
    def local_npts( self ):
        return self._local_npts

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
        return self._starts

    # ...
    @property
    def parent_ends( self ):
        return self._parent_ends

    @property
    def reduced( self ):
        return self._reduced

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

#===============================================================================
class StencilVector( Vector ):
    """
    Vector in n-dimensional stencil format.

    Parameters
    ----------
    V : psydac.linalg.stencil.StencilVectorSpace
        Space to which the new vector belongs.

    """
    def __init__( self, V ):

        assert isinstance( V, StencilVectorSpace )

        sizes = [e-s+1 + 2*m*p for s,e,p,m in zip(V.starts, V.ends, V.pads, V.shifts)]

        self._sizes = tuple(sizes)
        self._ndim  = len(V.starts)
        self._data  = np.zeros( sizes, dtype=V.dtype )
        self._space = V

        # TODO: distinguish between different directions
        self._sync  = False

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    #...
    @property
    def dtype( self ):
        return self.space.dtype

    #...
    def dot( self, v ):

        assert isinstance( v, StencilVector )
        assert v._space is self._space

        res = self._dot(self._data, v._data , self.space.pads, self.space.shifts)
        if self._space.parallel:
            res = self._space.cart.comm_cart.allreduce( res, op=MPI.SUM )

        return res

    #...
    @staticmethod
    def _dot(v1, v2, pads, shifts):
        ndim = len(v1.shape)
        index = tuple( slice(m*p,-m*p) for p,m in zip(pads, shifts))
        return np.dot(v1[index].flat, v2[index].flat)

    #...
    def copy( self ):
        w = StencilVector( self._space )
        w._data[:] = self._data[:]
        w._sync    = self._sync
        return w

    #...
    def __neg__( self ):
        w = StencilVector( self._space )
        w._data[:] = -self._data[:]
        w._sync    =  self._sync
        return w

    #...
    def __mul__( self, a ):
        w = StencilVector( self._space )
        w._data = self._data * a
        w._sync = self._sync
        return w

    #...
    def __rmul__( self, a ):
        w = StencilVector( self._space )
        w._data = a * self._data
        w._sync = self._sync
        return w

    #...
    def __add__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        w = StencilVector( self._space )
        w._data = self._data  +  v._data
        w._sync = self._sync and v._sync
        return w

    #...
    def __sub__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        w = StencilVector( self._space )
        w._data = self._data  -  v._data
        w._sync = self._sync and v._sync
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
        self._sync  = v._sync and self._sync
        return self

    #...
    def __isub__( self, v ):
        assert isinstance( v, StencilVector )
        assert v._space is self._space
        self._data -= v._data
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
    def toarray( self, *, order='C', with_pads=False ):
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
    def toarray_local( self , *, order='C'):
        """ return the local array without the padding"""

        idx = tuple( slice(m*p,-m*p) for p,m in zip(self.pads, self.space.shifts) )
        return self._data[idx].flatten( order=order)

    # ...
    def _toarray_parallel_no_pads( self, order='C' ):
        a         = np.zeros( self.space.npts )
        idx_from  = tuple( slice(m*p,-m*p) for p,m in zip(self.pads, self.space.shifts) )
        idx_to    = tuple( slice(s,e+1) for s,e in zip(self.starts,self.ends) )
        a[idx_to] = self._data[idx_from]
        return a.flatten( order=order)

    # ...
    def _toarray_parallel_with_pads( self, order='C' ):

        pads = [m*p for m,p in zip(self.space.shifts, self.pads)]
        # Step 0: create extended n-dimensional array with zero values
        shape = tuple( n+2*p for n,p in zip( self.space.npts, pads ) )
        a = np.zeros( shape )

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

    def topetsc( self ):
        """ Convert to petsc data structure.
        """

        space = self.space
        assert space.parallel
        cart      = space.cart
        petsccart = cart.topetsc()
        petsc     = petsccart.petsc
        lgmap     = petsccart.l2g_mapping

        size  = (petsccart.local_size, None)
        gvec  = petsc.Vec().createMPI(size, comm=cart.comm)
        gvec.setLGMap(lgmap)
        gvec.setUp()

        idx = tuple( slice(m*p,-m*p) for m,p in zip(self.pads, self.space.shifts) )
        gvec.setArray(self._data[idx])

        return gvec
    # ...
    def __getitem__(self, key):
        index = self._getindex( key )
        return self._data[index]

    # ...
    def __setitem__(self, key, value):
        index = self._getindex( key )
        self._data[index] = value

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
        if self.space.parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self.space._synchronizer.update_ghost_regions( self._data, direction=direction )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._update_ghost_regions_serial( direction )

        # Flag ghost regions as up-to-date
        self._sync = True

    # ...
    def update_assembly_ghost_regions( self ):
        """
        Update ghost regions before performing non-local access to vector
        elements (e.g. in matrix-vector product).

        Parameters
        ----------
        direction : int
            Single direction along which to operate (if not specified, all of them).

        """
        if self.space.parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self.space._synchronizer.update_assembly_ghost_regions( self._data )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._update_assembly_ghost_regions_serial()
    # ...
    def _update_ghost_regions_serial( self, direction=None ):

        if direction is None:
            for d in range( self._space.ndim ):
                self._update_ghost_regions_serial( d )
            return

        ndim     = self._space.ndim
        periodic = self._space.periods[direction]
        p        = self._space.pads   [direction]
        m        = self._space.shifts[direction]

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

    def _update_assembly_ghost_regions_serial( self ):

        ndim     = self._space.ndim
        for direction in range(ndim):

            periodic = self._space.periods[direction]
            p        = self._space.pads   [direction]
            m        = self._space.shifts[direction]
            r        = self._space.reduced[direction]

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
    def _getindex( self, key ):

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
class StencilMatrix( Matrix ):
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

        if pads is not None:
            for p,vp in zip(pads, V.pads):
                assert p<=vp

        self._pads     = pads or tuple(V.pads)
        dims           = [e-s+2*mi*p+1 for s,e,p,mi in zip(W.starts, W.ends, W.pads, W.shifts)]
        diags          = [compute_diag_len(p, md, mc) for p,md,mc in zip(self._pads, V.shifts, W.shifts)]
        self._data     = np.zeros( dims+diags, dtype=W.dtype )
        self._domain   = V
        self._codomain = W
        self._ndim     = len( dims )
        self._backend  = backend
        self._is_T     = False

        # Parallel attributes
        if W.parallel:
            # Create data exchanger for ghost regions
            self._synchronizer = CartDataExchanger(
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

        args                 = OrderedDict()
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
        self._transpose_func     = self._transpose

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
        return self.domain.dtype

    # ...
    def dot( self, v, out=None):

        assert isinstance( v, StencilVector )
        assert v.space is self.domain

        # Necessary if vector space is distributed across processes
        if not v.ghost_regions_in_sync:
            v.update_ghost_regions()

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self.codomain
        else:
            out = StencilVector( self.codomain )


        self._func(self._data, v._data, out._data, **self._args)

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
    def transpose( self ):
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
    # Other properties/methods
    #--------------------------------------

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
    def __mul__( self, a ):
        w = StencilMatrix( self._domain, self._codomain, self._pads, self._backend )
        w._data = self._data * a
        w._func = self._func
        w._args = self._args
        w._sync = self._sync
        return w

    #...
    def __rmul__( self, a ):
        w = StencilMatrix( self._domain, self._codomain, self._pads, self._backend )
        w._data = a * self._data
        w._func = self._func
        w._args = self._args
        w._sync = self._sync
        return w

    # ...
    def __neg__(self):
        return self.__mul__(-1)

    #...
    def __add__(self, m):
        assert isinstance(m, StencilMatrix)
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

    #...
    def __sub__(self, m):
        assert isinstance(m, StencilMatrix)
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

    #...
    def __imul__(self, a):
        self._data *= a
        return self

    #...
    def __iadd__(self, m):
        assert isinstance(m, StencilMatrix)
        assert m._domain   is self._domain
        assert m._codomain is self._codomain
        assert m._pads     == self._pads
        self._data += m._data
        self._sync  = m._sync and self._sync
        return self

    #...
    def __isub__(self, m):
        assert isinstance(m, StencilMatrix)
        assert m._domain   is self._domain
        assert m._codomain is self._codomain
        assert m._pads     == self._pads
        self._data -= m._data
        self._sync  = m._sync and self._sync
        return self

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
        # TODO: add unit tests

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
    def update_ghost_regions( self, *, direction=None ):
        """
        Update ghost regions before performing non-local access to matrix
        elements (e.g. in matrix transposition).

        Parameters
        ----------
        direction : int
            Single direction along which to operate (if not specified, all of them).

        """
        ndim     = self._codomain.ndim
        parallel = self._codomain.parallel

        if self._codomain.parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self._synchronizer.update_ghost_regions( self._data, direction=direction )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._update_ghost_regions_serial( direction )

        # Flag ghost regions as up-to-date
        self._sync = True

    def update_assembly_ghost_regions( self ):
        """
        Update ghost regions after the assembly algorithm.
        """
        ndim     = self._codomain.ndim
        parallel = self._codomain.parallel

        if self._codomain.parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self._synchronizer.update_assembly_ghost_regions( self._data )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._update_assembly_ghost_regions_serial()

    # ...
    @property
    def T(self):
        return self.transpose()

    # ...
    def topetsc( self ):
        """ Convert to petsc data structure.
        """

        dspace = self.domain
        cspace = self.codomain
        assert cspace.parallel and dspace.parallel

        ccart      = cspace.cart
        cpetsccart = ccart.topetsc()
        clgmap     = cpetsccart.l2g_mapping

        dcart      = dspace.cart
        dpetsccart = dcart.topetsc()
        dlgmap     = dpetsccart.l2g_mapping

        petsc      = dpetsccart.petsc

        r_size  = (cpetsccart.local_size, None)
        c_size  = (dpetsccart.local_size, None)

        gmat = petsc.Mat().create(comm=dcart.comm)
        gmat.setSizes((r_size, c_size))
        gmat.setType('mpiaij')
        gmat.setLGMap(clgmap, dlgmap)
        gmat.setUp()

        mat_csr = self.tocoo_local().tocsr()

        gmat.setValuesLocalCSR(mat_csr.indptr,mat_csr.indices,mat_csr.data)
        gmat.assemble()
        return gmat

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

        ravel_multi_index = np.ravel_multi_index

        # COO storage
        rows = []
        cols = []
        data = []

        pp = [compute_diag_len(p,mj,mi)-(p+1) for p,mi,mj in zip(self._pads, cm, dm)]
        # Range of data owned by local process (no ghost regions)
        local = tuple( [slice(mi*p,-mi*p) for p,mi in zip(cpads, cm)] + [slice(None)] * nd )

        for (index,value) in np.ndenumerate( self._data[local] ):

            # index = [i1-s1, i2-s2, ..., p1+j1-i1, p2+j2-i2, ...]

            xx = index[:nd]  # x=i-s
            ll = index[nd:]  # l=p+k

            ii = [s+x for s,x in zip(ss,xx)]
            di = [i//m for i,m in zip(ii,cm)]

            jj = [(i*m+l-p)%n for (i,m,l,n,p) in zip(di,dm,ll,nc,pp)]

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

    def _update_assembly_ghost_regions_serial( self ):

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

        args = OrderedDict()
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
        self._backend        = backend
        self._args           = self._dotargs_null.copy()
        self._transpose_args  = self._transpose_args_null.copy()

        if self._backend is None:
            self._func           = self._dot
            self._transpose_func = self._transpose
        else:
            transpose            = TransposeOperator(self._ndim, backend=frozenset(backend.items()))
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

            args = OrderedDict([('n{i}',nrows),('nc{i}', ncols),('gp{i}', gpads),('p{i}',pads ),
                                ('dm{i}', dm),('cm{i}', cm),('nd{i}', ndiags),
                                ('ndT{i}', ndiagsT),('si{i}', si),('sk{i}', sk),('sl{i}', sl)])

            for arg_name, arg_val in args.items():
                for i in range(len(nrows)):
                    self._transpose_args[arg_name.format(i=i+1)] = arg_val[i]

            if self.domain.parallel:
                if self.domain == self.codomain:
                    # In this case nrows_extra[i] == 0 for all i
                    dot = LinearOperatorDot(self._ndim,
                                    backend=frozenset(backend.items()),
                                    nrows_extra = self._args['nrows_extra'],
                                    gpads=self._args['gpads'],
                                    pads=self._args['pads'],
                                    dm = self._args['dm'],
                                    cm = self._args['cm'])

                    starts = self._args.pop('starts')
                    nrows  = self._args.pop('nrows')

                    self._args.pop('nrows_extra')
                    self._args.pop('gpads')
                    self._args.pop('pads')
                    self._args.pop('dm')
                    self._args.pop('cm')

                    for i in range(len(nrows)):
                        self._args['s{i}'.format(i=i+1)] = starts[i]

                    for i in range(len(nrows)):
                        self._args['n{i}'.format(i=i+1)] = nrows[i]

                else:
                    dot = LinearOperatorDot(self._ndim,
                                            backend=frozenset(backend.items()),
                                            gpads=self._args['gpads'],
                                            pads=self._args['pads'],
                                            dm = self._args['dm'],
                                            cm = self._args['cm'])

                    starts      = self._args.pop('starts')
                    nrows       = self._args.pop('nrows')
                    nrows_extra = self._args.pop('nrows_extra')

                    self._args.pop('gpads')
                    self._args.pop('pads')
                    self._args.pop('dm')
                    self._args.pop('cm')

                    for i in range(len(nrows)):
                        self._args['s{i}'.format(i=i+1)] = starts[i]

                    for i in range(len(nrows)):
                        self._args['n{i}'.format(i=i+1)] = nrows[i]

                    for i in range(len(nrows)):
                        self._args['ne{i}'.format(i=i+1)] = nrows_extra[i]

            else:
                dot = LinearOperatorDot(self._ndim,
                                        backend=frozenset(backend.items()),
                                        starts = tuple(self._args['starts']),
                                        nrows=tuple(self._args['nrows']),
                                        nrows_extra=self._args['nrows_extra'],
                                        gpads=self._args['gpads'],
                                        pads=self._args['pads'],
                                        dm = self._args['dm'],
                                        cm = self._args['cm'])
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
class StencilInterfaceMatrix(Matrix):
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

    dim : int
          The axis of the interface.

    pads: <list|tuple>
          Padding of the linear operator.

    """
    def __init__( self, V, W, s_d, s_c, dim, pads=None ):

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W, StencilVectorSpace )
        assert W.pads == V.pads

        if pads is not None:
            for p,vp in zip(pads, V.pads):
                assert p<=vp

        self._pads     = pads or tuple(V.pads)
        dims           = [e-s+2*p+1 for s,e,p in zip(W.starts, W.ends, W.pads)]
        dims[dim]      = 3*W.pads[dim] + 1
        diags          = [2*p+1 for p in self._pads]
        self._data     = np.zeros( dims+diags, dtype=W.dtype )
        self._domain   = V
        self._codomain = W
        self._dim      = dim
        self._d_start  = s_d
        self._c_start  = s_c
        self._ndim     = len( dims )

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
        if not v.ghost_regions_in_sync:
            raise ValueError('ghost regions are not updated')

        if out is not None:
            assert isinstance( out, StencilVector )
            assert out.space is self.codomain
        else:
            out = StencilVector( self.codomain )

        # Shortcuts
        ssc   = self.codomain.starts
        eec   = self.codomain.ends
        ssd   = self.domain.starts
        eed   = self.domain.ends
        dpads = self.domain.pads
        dim   = self.dim

        c_start = self.c_start
        d_start = self.d_start
        pads    = self.pads

        # Number of rows in matrix (along each dimension)
        nrows        = [ed-s+1 for s,ed in zip(ssd, eed)]
        nrows_extra  = [0 if ec<=ed else ec-ed for ec,ed in zip(eec,eed)]
        nrows[dim]   = self._pads[dim] + 1 - nrows_extra[dim]

        self._dot(self._data, v._data, out._data, nrows, nrows_extra, dpads, pads, dim, d_start, c_start)

        # IMPORTANT: flag that ghost regions are not up-to-date
        out.ghost_regions_in_sync = False
        return out

    # ...
    @staticmethod
    def _dot(mat, v, out, nrows, nrows_extra, dpads, pads, dim, d_start, c_start):
        # Index for k=i-j
        ndim = len(v.shape)
        kk = [slice(None)]*ndim
        diff = [xp-p for xp,p in zip(dpads, pads)]

        diff[dim] += d_start

        for xx in np.ndindex( *nrows ):
            ii    = [ p+x for p,x in zip(dpads,xx) ]
            jj    = tuple( slice(d+x,d+x+2*p+1) for x,p,d in zip(xx,pads,diff) )
            ii_kk = tuple( ii + kk )

            ii[dim] += c_start
            out[tuple(ii)] = np.dot( mat[ii_kk].flat, v[jj].flat )

        new_nrows = nrows.copy()
        for d,er in enumerate(nrows_extra):

            rows = new_nrows.copy()
            del rows[d]

            for n in range(er):
                for xx in np.ndindex(*rows):
                    xx = list(xx)
                    xx.insert(d, nrows[d]+n)

                    ii     = [x+xp for x,xp in zip(xx, dpads)]
                    ee     = [max(x-l+1,0) for x,l in zip(xx, nrows)]
                    jj     = tuple( slice(x+d, x+d+2*p+1-e) for x,p,d,e in zip(xx, pads, diff, ee) )

                    ndiags = [2*p + 1-e for p,e in zip(pads,ee)]
                    kk     = [slice(None,diag) for diag in ndiags]
                    ii_kk  = tuple( list(ii) + kk )
                    ii[dim] += c_start
                    out[tuple(ii)] = np.dot( mat[ii_kk].flat, v[jj].flat )
            new_nrows[d] += er
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
                                    self._d_start, self._c_start,
                                    self._dim, self._pads )
        M._data[:] = self._data[:]
        return M

    # ...
    def __neg__(self):
        return self.__mul__(-1)

    #...
    def __mul__( self, a ):
        w = StencilInterfaceMatrix( self._domain, self._codomain,
                                    self._d_start, self._c_start,
                                    self._dim, self._pads )
        w._data = self._data * a
        w._sync = self._sync
        return w

    #...
    def __rmul__( self, a ):
        w = StencilInterfaceMatrix( self._domain, self._codomain,
                                    self._d_start, self._c_start,
                                    self._dim, self._pads )
        w._data = a * self._data
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
        raise NotImplementedError('TODO: StencilInterfaceMatrix.__imul__')

    #...
    def __iadd__(self, m):
        raise NotImplementedError('TODO: StencilInterfaceMatrix.__iadd__')

    #...
    def __isub__(self, m):
        raise NotImplementedError('TODO: StencilInterfaceMatrix.__isub__')

    #--------------------------------------
    # Other properties/methods
    #--------------------------------------

    @property
    def dim( self ):
        return self._dim

    # ...
    @property
    def d_start( self ):
        return self._d_start

    # ...
    @property
    def c_start( self ):
        return self._c_start

    # ...
    @property
    def pads( self ):
        return self._pads

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
    def update_ghost_regions( self, *, direction=None ):
        """
        Update ghost regions before performing non-local access to matrix
        elements (e.g. in matrix transposition).

        Parameters
        ----------
        direction : int
            Single direction along which to operate (if not specified, all of them).

        """
        ndim     = self._codomain.ndim
        parallel = self._codomain.parallel

        if self._codomain.parallel:
            # PARALLEL CASE: fill in ghost regions with data from neighbors
            self._synchronizer.update_ghost_regions( self._data, direction=direction )
        else:
            # SERIAL CASE: fill in ghost regions along periodic directions, otherwise set to zero
            self._update_ghost_regions_serial( direction )

        # Flag ghost regions as up-to-date
        self._sync = True

    def update_assembly_ghost_regions( self ):
        pass

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
        nr = self.codomain.npts
        nc = self.domain.npts
        ss = self.codomain.starts
        pp = self.codomain.pads
        nd = len(pp)
        dim = self.dim
        c_start = self.c_start
        d_start = self.d_start

        ravel_multi_index = np.ravel_multi_index

        # COO storage
        rows = []
        cols = []
        data = []
        # Range of data owned by local process (no ghost regions)
        local = tuple( [slice(p,-p) for p in pp] + [slice(None)] * nd )
        for (index,value) in np.ndenumerate( self._data[local] ):
            if value:
                # index = [i1, i2, ..., p1+j1-i1, p2+j2-i2, ...]

                xx = index[:nd]  # x=i-s
                ll = index[nd:]  # l=p+k

                ii = [s+x for s,x in zip(ss,xx)]
                jj = [(i+l-p) % n for (i,l,n,p) in zip(ii,ll,nc,self.pads)]

                ii[dim] += c_start
                jj[dim] += d_start

                I = ravel_multi_index( ii, dims=nr,  order='C' )
                J = ravel_multi_index( jj, dims=nc,  order='C' )

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


#===============================================================================
from psydac.api.settings   import PSYDAC_BACKENDS
del VectorSpace, Vector, Matrix
