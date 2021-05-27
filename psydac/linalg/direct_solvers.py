# coding: utf-8
# Copyright 2018 Jalal Lakhlili, Yaman Güçlü

from abc                 import abstractmethod
from numpy               import ndarray
from scipy.linalg.lapack import dgbtrf, dgbtrs
from scipy.sparse        import spmatrix
from scipy.sparse.linalg import splu

from psydac.linalg.basic    import LinearSolver

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

    @abstractmethod
    def solve( self, rhs, out=None, transposed=False ):
        pass

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
    def __init__( self, u, l, bmat ):

        self._u    = u
        self._l    = l

        # ... LU factorization
        self._bmat, self._ipiv, self._finfo = dgbtrf(bmat, l, u)

        self._sinfo = None

        self._space = ndarray

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

    #...
    def solve( self, rhs, out=None, transposed=False ):

        assert rhs.shape[0] == self._bmat.shape[1]

        if out is None:
            preout, self._sinfo = dgbtrs(self._bmat, self._l, self._u, rhs.T, self._ipiv, trans=transposed)
            out = preout.T

        else :
            assert out.shape == rhs.shape

            # support in-place operations
            if rhs is not out:
                out[:] = rhs

            # TODO: handle non-contiguous views?
            
            # we want FORTRAN-contiguous data (default is assumed to be C contiguous)
            _, self._sinfo = dgbtrs(self._bmat, self._l, self._u, out.T, self._ipiv, overwrite_b=True, trans=transposed)

        return out

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

    #...
    def solve( self, rhs, out=None, transposed=False ):

        assert rhs.shape[0] == self._splu.shape[1]

        if out is None:
            out = self._splu.solve( rhs.T, trans='T' if transposed else 'N' ).T

        else:
            assert out.shape == rhs.shape

            # currently no in-place solve exposed
            out[:] = self._splu.solve( rhs.T, trans='T' if transposed else 'N' ).T

        return out

#===============================================================================


