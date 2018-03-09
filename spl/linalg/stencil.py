# coding: utf-8

from .basic import (VectorSpace as VectorSpaceBase,
                    Vector      as VectorBase,
                    LinearOperator)

#===============================================================================
class VectorSpace( VectorSpaceBase ):
    """
    Vector space for stencil format.

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

        assert( len(starts) == len(ends) == len(pads) )

        self._starts = tuple(starts)
        self._ends   = tuple(ends)
        self._pads   = tuple(pads)
        self._dtype  = dtype
        self._ndim   = len(starts)

    # ...
    def _init_parallel( self, cart, dtype=float ):

        raise NotImplementedError( "Parallel version not yet available." )

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

#===============================================================================
class Vector( VectorBase ):
    """
    Stencil vector.

    """
    def __init__( self, V ):

        assert( isinstance( V, VectorSpace ) )

        import numpy as np
        sizes = [e-s+2*p+1 for s,e,p in zip(V.starts, V.ends, V.pads)]
        self._data  = np.zeros(sizes)
        self._space = V

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    #...
    def dot( self, v ):

        assert( isinstance( v, Vector ) )
        assert( v._space is self._space )

        import numpy as np

        # TODO: verify this
        return np.dot( self._data.flat, v._data.flat )

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
class Matrix(object):
    """
    Class that represents a stencil matrix.
    """

    def __init__(self, starts, ends, pads):

        assert( len(starts) == len(ends) == len(pads) )

        self._starts = tuple(starts)
        self._ends   = tuple(ends)
        self._pads   = tuple(pads)
        self._ndim   = len(starts)

        sizes = [e-s+1 for s,e in zip(starts, ends)]
        pads  = [2*p+1 for p in pads]
        shape =  sizes + pads

        import numpy as np
        self._data = np.zeros(shape)

    @property
    def starts(self):
        return self._starts

    @property
    def ends(self):
        return self._ends

    @property
    def pads(self):
        return self._pads

    @property
    def ndim(self):
        return self._ndim

    # ...
    def __getitem__(self, key):
        nd = self._ndim
        ii = key[:nd]
        kk = key[nd:]

        index = []

        for (i,s) in zip(ii, self._starts):
            if isinstance(i, slice):
                start = None if i.start is None else s + i.start
                stop  = None if i.stop  is None else s + i.stop
                l = slice(start, stop, i.step)
            else:
                l = s + i
            index.append(l)

        for (k,p) in zip(kk, self._pads):
            if isinstance(k, slice):
                start = None if k.start is None else p + k.start
                stop  = None if k.stop  is None else p + k.stop
                l = slice(start, stop, k.step)
            else:
                l = p + k
            index.append(l)

        return self._data[tuple(index)]

    # ...
    def __setitem__(self, key, value):
        nd = self._ndim
        ii = key[:nd]
        kk = key[nd:]

        index = []

        for (i,s) in zip(ii, self._starts):
            if isinstance( i, slice ):
                start = None if i.start is None else s+i.start
                stop  = None if i.stop  is None else s+i.stop
                l = slice(start, stop, i.step)
            else:
                l = s + i
            index.append(l)

        for (k,p) in zip(kk, self._pads):
            if isinstance(k, slice):
                start = None if k.start is None else p+k.start
                stop  = None if k.stop  is None else p+k.stop
                l = slice(start, stop, k.step)
            else:
                l = p + k
            index.append(l)

        self._data[tuple(index)] = value

    def __str__(self):
        return str(self._data)
    # ...

    # ...
    def dot(self, v):

        if not isinstance(v, Vector):
            raise TypeError("v must be a Vector")

        import numpy as np

        # TODO check shapes

        [s1, s2] = self.starts
        [e1, e2] = self.ends
        [p1, p2] = self.pads

        # ...
        res  = v.copy()
        res *= 0.0

        for i1 in range(s1, e1+1):
            for i2 in range(s2, e2+1):
                    res[i1,i2] = np.dot(
                            self[i1,i2,:,:].flat,
                            v[i1-p1:i1+p1+1,i2-p2:i2+p2+1].flat
                            )

#                for k1 in range(-p1, p1+1):
#                    for k2 in range(-p2, p2+1):
#                        j1 = k1+i1
#                        j2 = k2+i2
#                        res[i1,i2] = res[i1,i2] + self[i1,i2,k1,k2] * v[j1,j2]
        # ...

        return res
    # ...

    # ...
    def tocoo(self):
        """
        Convert the stencil data to sparce matrix in the COOrdinate form
        """

        from scipy.sparse import coo_matrix

        s1 = self.starts[0]
        e1 = self.ends[0]

        s2 = self.starts[0]
        e2 = self.ends[1]

        p1 = self.pads[0]
        p2 = self.pads[1]

        n1 = e1 - s1 + 1
        n2 = e2 - s2 + 1

        rows = []
        cols = []
        vals = []

        # ...
        for i1 in range(s1, e1+1):
            for i2 in range(s2, e2+1):
                for k1 in range(-p1, p1+1):
                    for k2 in range(-p2, p2+1):
                        j1 = (k1 + i1)%n1
                        j2 = (k2 + i2)%n2
                        irow = i1 + n1*i2
                        icol = j1 + n1*j2

                        rows.append(irow)
                        cols.append(icol)
                        vals.append(self[i1, i2, k1, k2])
        # ...

#        rows = np.array(rows)
#        cols = np.array(cols)
#        vals = np.array(vals)
        mat  = coo_matrix((vals, (rows, cols)), shape=(n1*n2, n1*n2))

        mat.eliminate_zeros()

        return mat
    # ...

#===============================================================================
del VectorSpaceBase, VectorBase, LinearOperator
