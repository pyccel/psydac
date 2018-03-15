# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from .basic import (VectorSpace as VectorSpaceBase,
                    Vector      as VectorBase,
                    LinearOperator)

#===============================================================================
class VectorSpace( VectorSpaceBase ):
    """
    Vector space for n-dimensional stencil format.

    Parameters
    ----------
    starts : tuple-like
        Start index along each direction.

    ends : tuple-like
        End index along each direction.

    pads : tuple-like
        Padding p along each direction (number of diagonals is 2*p+1).

    cart : <not defined>
        MPI Cartesian topology (not used for now).

    """
    def __init__( self, *args, **kwargs ):

        if len(args) == 1 or hasattr( kwargs, 'cart' ):
            self._init_parallel( *args, **kwargs )
        else:
            self._init_serial  ( *args, **kwargs )

    # ...
    def _init_serial( self, starts, ends, pads, dtype=float ):
        from numpy import prod

        assert( len(starts) == len(ends) == len(pads) )

        self._starts = tuple(starts)
        self._ends   = tuple(ends)
        self._pads   = tuple(pads)
        self._dtype  = dtype
        self._ndim   = len(starts)

        self._dimension = prod( [e-s+1 for s,e in zip(starts,ends)] )

    # ...
    def _init_parallel( self, cart, dtype=float ):

        raise NotImplementedError( "Parallel version not yet available." )

    # ...
    @property
    def dimension( self ):
        return self._dimension

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
    def dtype( self ):
        return self._dtype

    # ...
    @property
    def ndim( self ):
        return self._ndim

#===============================================================================
class Vector( VectorBase ):
    """
    Vector in n-dimensional stencil format.

    Parameters
    ----------
    V : spl.linalg.stencil.VectorSpace
        Space to which the new vector belongs.

    """
    def __init__( self, V ):
        from numpy import zeros

        assert( isinstance( V, VectorSpace ) )

        sizes = [e-s+2*p+1 for s,e,p in zip(V.starts, V.ends, V.pads)]
        self._data  = zeros(sizes)
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

        assert( isinstance( v, Vector ) )
        assert( v._space is self._space )

        index = tuple( slice(p,-p) for p in self.pads )
        return dot( self._data[index].flat, v._data[index].flat )

    #...
    def copy( self ):
        w = Vector( self._space )
        w._data[:] = self._data[:]
        return w

    #...
    def __mul__( self, a ):
        w = Vector( self._space )
        w._data = self._data * a
        return w

    #...
    def __rmul__( self, a ):
        w = Vector( self._space )
        w._data = a * self._data

        return w

    #...
    def __add__( self, v ):
        assert( isinstance( v, Vector ) )
        assert( v._space is self._space )
        w = Vector( self._space )
        w._data = self._data + v._data
        return w

    #...
    def __sub__( self, v ):
        assert( isinstance( v, Vector ) )
        assert( v._space is self._space )
        w = Vector( self._space )
        w._data = self._data - v._data
        return w

    #...
    def __imul__( self, a ):
        self._data *= a
        return self

    #...
    def __iadd__( self, v ):
        assert( isinstance( v, Vector ) )
        assert( v._space is self._space )
        self._data += v._data
        return self

    #...
    def __isub__( self, v ):
        assert( isinstance( v, Vector ) )
        assert( v._space is self._space )
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
        return str(self._data)

    # ...
    def toarray(self):
        """
        Return a numpy 1D array corresponding to the given Vector, without pads.

        """
        index = tuple( slice(p,-p) for p in self.pads )
        return self._data[index].flatten()

    # ...
    def __getitem__(self, key):
        index = self._getindex( key )
        return self._data[index]

    # ...
    def __setitem__(self, key, value):
        index = self._getindex( key )
        self._data[index] = value

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
class Matrix( LinearOperator ):
    """
    Matrix in n-dimensional stencil format.

    This is a linear operator that maps elements of stencil vector space V to
    elements of stencil vector space W.

    For now we only accept V==W.

    Parameters
    ----------
    V : spl.linalg.stencil.VectorSpace
        Domain of the new linear operator.

    W : spl.linalg.stencil.VectorSpace
        Codomain of the new linear operator.

    """
    def __init__( self, V, W ):

        from numpy import zeros

        assert( isinstance( V, VectorSpace ) )
        assert( isinstance( W, VectorSpace ) )
        assert( V is W )

        dims        = [e-s+1 for s,e in zip(V.starts, V.ends)]
        diags       = [2*p+1 for p in V.pads]
        self._data  = zeros( dims+diags )
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

        assert( isinstance( v, Vector ) )
        assert( v.space is self.domain )

        if out is not None:
            assert( isinstance( out, Vector ) )
            assert( out.space is self.codomain )
        else:
            out = Vector( self.codomain )

        ss = self.starts
        pp = self.pads
        kk = [slice(None)] * self._ndim

        for xx in ndindex( *self._dims ):

            ii    = tuple( s+x for s,x in zip(ss,xx) )
            jj    = tuple( slice(i-p,i+p+1) for i,p in zip(ii,pp) )
            ii_kk = tuple( list(ii) + kk )

            out[ii] = dot( self[ii_kk].flat, v[jj].flat )

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
        nn = self._dims
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

    #--------------------------------------
    # Private methods
    #--------------------------------------
    def _getindex( self, key ):

        nd = self._ndim
        ii = key[:nd]
        kk = key[nd:]

        index = []

        for i,s in zip( ii, self.starts ):
            x = self._shift_index( i,s )
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
del VectorSpaceBase, VectorBase, LinearOperator
