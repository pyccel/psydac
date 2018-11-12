# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import numpy as np

from spl.linalg.basic import VectorSpace, Vector, LinearOperator

__all__ = ['DenseVectorSpace', 'DenseVector', 'DenseLinearOperator']

#==============================================================================
class DenseVectorSpace( VectorSpace ):
    """
    Space of one-dimensional global arrays, NOT distributed across processes.

    Typical examples are the right-hand side vector b and solution vector x
    of a linear system A*x=b.
 
    Parameters
    ----------
    n : int
        Number of vector components.

    dtype : data-type, optional
        Desired data-type for the arrays (default is numpy.float64).
    
    """
    def __init__( self, n, dtype=np.float64 ):
        self._n     = n
        self._dtype = dtype
        pass

    #-------------------------------------
    # Abstract interface
    #-------------------------------------
    @property
    def dimension( self ):
        return self._n

    # ...
    def zeros( self ):
        data = np.zeros( self.dimension, dtype=self.dtype )
        return DenseVector( self, data )

    #-------------------------------------
    # Other properties/methods
    #-------------------------------------
    @property
    def dtype( self ):
        return self._dtype

#==============================================================================
class DenseVector( Vector ):

    def __init__( self, V, data ):

        assert isinstance( V, DenseVectorSpace )

        data = np.asarray( data )
        assert data.ndim  == 1
        assert data.shape ==(V.dimension,)
        assert data.dtype == V.dtype

        self._space = V
        self._data  = data

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    # ...
    def dot( self, v ):
        assert isinstance( v, DenseVector )
        assert v._space is self._space
        return np.dot( self._data, v._data )

    # ...
    def copy( self ):
        return DenseVector( self._space, self._data.copy() )

    # ...
    def __mul__( self, a ):
        return DenseVector( self._space, self._data * a )

    # ...
    def __rmul__( self, a ):
        return DenseVector( self._space, a * self._data )

    # ...
    def __add__( self, v ):
        assert isinstance( v, DenseVector )
        assert v._space is self._space
        return DenseVector( self._space, self._data + v._data )

    # ...
    def __sub__( self, v ):
        assert isinstance( v, DenseVector )
        assert v._space is self._space
        return DenseVector( self._space, self._data - v._data )

    # ...
    def __imul__( self, a ):
        self._data *= a
        return self

    # ...
    def __iadd__( self, v ):
        assert isinstance( v, DenseVector )
        assert v._space is self._space
        self._data += v._data
        return self

    # ...
    def __isub__( self, v ):
        assert isinstance( v, DenseVector )
        assert v._space is self._space
        self._data -= v._data
        return self

    #-------------------------------------
    # Other properties/methods
    #-------------------------------------
    def toarray( self ):
        return self._data.copy()

#==============================================================================
class DenseLinearOperator( LinearOperator ):

    def __init__( self, V, W, data ):

        assert isinstance( V, DenseVectorSpace )
        assert isinstance( W, DenseVectorSpace )

        data = np.asarray( data )
        assert data.ndim  == 2
        assert data.shape == (W.dimension, V.dimension)
#        assert data.dfype == #???

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
            assert isinstance( out, DenseVector )
            assert out.space is self._codomain
            np.dot( self._data, v._data, out=out._data )
        else:
            W    = self._codomain
            data = np.dot( self._data, v._data )
            out  = DenseVector( W, data )

        return out

