# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import numpy as np

from spl.linalg.basic   import VectorSpace, Vector, LinearOperator
from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from spl.polar .dense   import DenseVectorSpace, DenseVector, DenseLinearOperator

__all__ = ['LinearOperator_StencilToDense', 'LinearOperator_DenseToStencil']

#==============================================================================
class LinearOperator_StencilToDense( LinearOperator ):

    def __init__( self, V, W, data ):

        assert isinstance( V, StencilVectorSpace )
        assert isinstance( W,   DenseVectorSpace )

        # V space must be 2D for now (TODO: extend to higher dimensions)
        # W space must have 3 components for now (TODO: change to arbitrary n)
        # Only works in serial (TODO: extend to MPI setting)

        s1, s2 = V.starts
        e1, e2 = V.ends
        p1, p2 = V.pads
        n0     = W.dimension

        data = np.asarray( data )
        assert data.shape == (n0, p1, e2-s2+1)

        # Store information in object
        self._domain   = V
        self._codomain = W
        self._data     = data

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
    def dot( self, v, out=None ):

        assert isinstance( v, StencilVector )
        assert v.space is self._domain

        if out:
            assert isinstance( out, DenseVector )
            assert out.space is self._codomain
        else:
            out = self._codomain.zeros()

        # TODO: verify if range(s1,e1+1) contains degrees of freedom
        # TODO: implement parallel version
        # TODO: do we care about having a 'dot_incr' option?

        V = self._domain
        W = self._codomain

        s1, s2 = V.starts
        e1, e2 = V.ends
        p1, p2 = V.pads
        n0     = W.dimension

        B_sd = self._data
        y    =  out._data

        for i in range( n0 ):
            y[i] = np.dot( B_sd[i,:,:].flat, v[0:p1,s2:e2+1].flat )
    
        return out

    #-------------------------------------
    # Other properties/methods
    #-------------------------------------
    def toarray( self ):

        n0     = self.codomain.dimension

        n1, n2 = self.domain.npts
        p1, p2 = self.domain.pads
        s1, s2 = self.domain.starts
        e1, e2 = self.domain.ends

        a  = np.zeros( (n0,n1*n2), dtype=self.codomain.dtype )
        d  = self._data

        for i in range( n0 ):
            for j1 in range( p1 ):
                j_start = j1*n2 + s2
                j_stop  = j1*n2 + e2 + 1
                a[i,j_start:j_stop] = d[i,j1,:]

        return a

    # ...
    def tocsr( self ):
        # TODO!!!!
        pass

#==============================================================================
class LinearOperator_DenseToStencil( LinearOperator ):

    def __init__( self, V, W, data ):

        assert isinstance( V,   DenseVectorSpace )
        assert isinstance( W, StencilVectorSpace )

        # V space must have 3 components for now (TODO: change to arbitrary n)
        # W space must be 2D for now (TODO: extend to higher dimensions)
        # Only works in serial (TODO: extend to MPI setting)

        s1, s2 = W.starts
        e1, e2 = W.ends
        p1, p2 = W.pads
        n0     = V.dimension

        data = np.asarray( data )
        assert data.shape == (p1, e2-s2+1, n0)

        # Store information in object
        self._domain   = V
        self._codomain = W
        self._data     = data

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
    def dot( self, v, out=None ):

        assert isinstance( v, DenseVector )
        assert v.space is self._domain

        if out:
            assert isinstance( out, StencilVector )
            assert out.space is self._codomain
        else:
            out = self._codomain.zeros()

        V = self._domain
        W = self._codomain

        s1, s2 = W.starts
        e1, e2 = W.ends
        p1, p2 = W.pads
        n0     = V.dimension

        B_ds = self._data
        x    =    v._data

        out[0:p1,s2:e2+1] = np.dot( B_ds, x )

        return out

    #-------------------------------------
    # Other properties/methods
    #-------------------------------------
    def toarray( self ):

        n0     = self.domain.dimension

        n1, n2 = self.codomain.npts
        p1, p2 = self.codomain.pads
        s1, s2 = self.codomain.starts
        e1, e2 = self.codomain.ends

        a  = np.zeros( (n1*n2,n0), dtype=self.codomain.dtype )
        d  = self._data

        for i1 in range( p1 ):
            for i2 in range( s2, e2+1 ):
                i = i1*n2 + i2
                a[i,:] = d[i1,i2,:]

        return a

    # ...
    def tocsr( self ):
        # TODO!!!
        pass
