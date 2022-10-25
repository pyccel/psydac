# coding: utf-8
# Copyright 2018 Jalal Lakhlili, Yaman Güçlü

from abc                 import abstractmethod
from numpy               import ndarray
from scipy.linalg.lapack import dgbtrf, dgbtrs
from scipy.sparse        import spmatrix
from scipy.sparse.linalg import splu

from psydac.linalg2.basic import VectorSpace
from psydac.linalg2.basic import LinearSolver
from psydac.linalg2.ndarray import NdarrayVector

__all__ = ['DirectSolver', 'BandedSolver', 'SparseSolver']

#===============================================================================
class DirectSolver( LinearSolver ):
    """
    Abstract class for direct linear solvers.

    """

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def space( self ):
        pass

    @property
    @abstractmethod
    def domain( self ):
        pass

    @property
    @abstractmethod
    def codomain( self ):
        pass

    @property
    @abstractmethod
    def dtype( self ):
        pass

    @abstractmethod
    def solve( self, rhs, out=None, transposed=False ):
        pass

    #@abstractmethod
    #def dot( self, rhs, out=None, transposed=False):
    #    pass

#===============================================================================
class BandedSolver ( DirectSolver ):
    """
    Solve the equation Ax = b for x, assuming A is banded matrix.

    Parameters
    ----------
    u : integer
        Number of non-zero upper diagonal.

    l : integer
        Number of non-zero lower diagonal.

    bmat : nd-array
        Banded matrix.

    """
    def __init__( self, space, u, l, bmat ):

        self._u    = u
        self._l    = l

        # ... LU factorization
        self._bmat, self._ipiv, self._finfo = dgbtrf(bmat, l, u)

        self._sinfo = None

        assert isinstance(space, VectorSpace)
        self._domain = space
        self._codomain = space
        self._space = space

    @property
    def finfo( self ):
        return self._finfo

    @property
    def sinfo( self ):
        return self._sinfo

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return None

    #...
    def solve( self, rhs, out=None, transposed=False ):
        """
        Solves for the given right-hand side.

        Parameters
        ----------
        rhs : ndarray
            The right-hand sides to solve for. The vectors are assumed to be given in C-contiguous order,
            i.e. if multiple right-hand sides are given, then rhs is a two-dimensional array with the 0-th
            index denoting the number of the right-hand side, and the 1-st index denoting the element inside
            a right-hand side.
        
        out : ndarray | NoneType
            Output vector. If given, it has to have the same shape and datatype as rhs.
        
        transposed : bool
            If and only if set to true, we solve against the transposed matrix. (supported by the underlying solver)
        """
        rhs = rhs.data
        assert rhs.T.shape[0] == self._bmat.shape[1]

        if out is None:
            preout, self._sinfo = dgbtrs(self._bmat, self._l, self._u, rhs.T, self._ipiv, trans=transposed)
            out = preout.T

        else :
            assert out.shape == rhs.shape
            assert out.dtype == rhs.dtype

            # support in-place operations
            if rhs is not out:
                out[:] = rhs

            # TODO: handle non-contiguous views?
            
            # we want FORTRAN-contiguous data (default is assumed to be C contiguous)
            _, self._sinfo = dgbtrs(self._bmat, self._l, self._u, out.T, self._ipiv, overwrite_b=True, trans=transposed)
        out = NdarrayVector(self._space, data=out)
        return out

    #def dot( self, rhs, out=None, transposed=False):
    #    return self.solve(rhs, out, transposed)

#===============================================================================
class SparseSolver ( DirectSolver ):
    """
    Solve the equation Ax = b for x, assuming A is scipy sparse matrix.

    Parameters
    ----------
    spmat : scipy.sparse.spmatrix
        Generic sparse matrix.

    """
    def __init__( self, spmat ):

        assert isinstance( spmat, spmatrix )

        self._space = ndarray
        self._splu  = splu( spmat.tocsc() )

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
        return self._space

    @property
    def domain( self ):
        return self._space

    @property
    def codomain( self ):
        return self._space

    #...
    def solve( self, rhs, out=None, transposed=False ):
        """
        Solves for the given right-hand side.

        Parameters
        ----------
        rhs : ndarray
            The right-hand sides to solve for. The vectors are assumed to be given in C-contiguous order,
            i.e. if multiple right-hand sides are given, then rhs is a two-dimensional array with the 0-th
            index denoting the number of the right-hand side, and the 1-st index denoting the element inside
            a right-hand side.
        
        out : ndarray | NoneType
            Output vector. If given, it has to have the same shape and datatype as rhs.
        
        transposed : bool
            If and only if set to true, we solve against the transposed matrix. (supported by the underlying solver)
        """
        
        assert rhs.T.shape[0] == self._splu.shape[1]

        if out is None:
            out = self._splu.solve( rhs.T, trans='T' if transposed else 'N' ).T

        else:
            assert out.shape == rhs.shape
            assert out.dtype == rhs.dtype

            # currently no in-place solve exposed
            out[:] = self._splu.solve( rhs.T, trans='T' if transposed else 'N' ).T

        return out

    def dot( self, rhs, out=None, transposed=False):
        return self.solve(rhs, out, transposed)