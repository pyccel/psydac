# coding: utf-8

import numpy as np
from scipy.sparse import coo_matrix
from mpi4py import MPI

from psydac.linalg.basic   import VectorSpace, Vector, Matrix
from psydac.ddm.cart       import find_mpi_type, CartDecomposition, CartDataExchanger
from psydac.linalg.stencil import StencilVector, StencilVectorSpace

__all__ = ['DenseVectorSpace' , 'DenseVector', 'DenseMatrix']

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
    def __init__( self, n, dtype=np.float64):

        self._ncoeff = n
        self._dtype  = dtype

    #-------------------------------------
    # Abstract interface
    #-------------------------------------
    @property
    def ncoeff( self ):
        """ The number of  coeff of a vector space V is the cardinality
        """
        return self._ncoeff
        
    @property
    def dimension(self):
        return None

    # ...
    def zeros( self ):
        """
        Get a copy of the null element of the DenseVectorSpace V.

        Returns
        -------
        null : DenseVector
            A new vector object with all components equal to zero.

        """
        return DenseVector( self )

    #-------------------------------------
    # Other properties/methods
    #-------------------------------------
    @property
    def dtype( self ):
        return self._dtype



#==============================================================================
class DenseVector( Vector ):

    def __init__( self, V ):

        assert isinstance( V, DenseVectorSpace )

        self._space = V
        self._data  = np.zeros( V.ncoeff, dtype=V.dtype )

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

        res = np.dot( self._data, v._data )

        V = self._space
        if V.parallel:
            if V.radial_comm.rank == V.radial_root:
                res = V.tensor_comm.allreduce( res )
            res = V.radial_comm.bcast( res, root=V.radial_root )

        return res

    # ...
    def copy( self ):
        return DenseVector( self._space, self._data.copy() )

    # ...
    def __mul__( self, a ):
        w = DenseVector( self._space )
        w._data = self._data * a
        return w

    def __rmul__( self, a ):
        w = DenseVector( self._space )
        w._data = self._data * a
        return w
    # ...
    def __add__( self, v ):
        assert isinstance( v, DenseVector )
        assert v._space is self._space
        w = DenseVector( self._space )
        w._data = self._data + v._data
        return w

    # ...
    def __sub__( self, v ):
        assert isinstance( v, DenseVector )
        assert v._space is self._space
        w = DenseVector( self._space )
        w._data = self._data - v._data
        return w
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


#===============================================================================
class DenseMatrix( Matrix ):

    def __init__( self, V, W ):

        if isinstance(V, DenseVectorSpace):
            coeffs = V.ncoeff
            data = [StencilVector(W) for i in range(coeffs)]
            
        elif isinstance(W , DenseVectorSpace):
            coeffs = W.ncoeff
            data = [StencilVector(V) for i in range(coeffs)]
        else:
            raise ValueError('domain or codomain must be a DenseVectorSpace') 

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
        assert v.space is self._domain

        if out:
            assert out.space is self._codomain
            if isinstance(self._codomain, DenseVectorSpace):
                assert isinstance( out, DenseVector )
                out._data[:] = [x.dot(v) for x in self._data]
            else:
                assert isinstance( out, StencilVector )
                for i in range(v.ncoeff):
                    out += self._data[i] * v._data[i]  
        else:
            if isinstance(self._codomain, DenseVectorSpace):
                out = DenseVector(self._codomain)
                out._data[:] = [x.dot(v) for x in self._data]
            else:
                out = StencilVector(self._codomain)
                for i in range(v.ncoeff):
                    out += self._data[i] * v._data[i] 

        return out

    # ...
    def toarray( self ):
        return self.tosparse().todense()

    # ...
    def tosparse( self ):
        
        if isinstance( self._domain, StencilVectorSpace):
            out = np.vstack([v.toarray() for v in self._data])
            return coo_matrix(out)
        else:
            return coo_matrix(out).T
