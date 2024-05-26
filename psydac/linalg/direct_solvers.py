# coding: utf-8
# Copyright 2018 Jalal Lakhlili, Yaman Güçlü

from abc                 import abstractmethod
import numpy               as np
from scipy.linalg.lapack import dgbtrf, dgbtrs, sgbtrf, sgbtrs, cgbtrf, cgbtrs, zgbtrf, zgbtrs
from scipy.sparse        import spmatrix
from scipy.sparse.linalg import splu,inv

from psydac.linalg.basic    import LinearSolver

__all__ = ('DirectSolver', 'BandedSolver', 'SparseSolver')

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
        if bmat.dtype == np.float32:
            self._factor_function = sgbtrf
            self._solver_function = sgbtrs
        elif bmat.dtype == np.float64:
            self._factor_function = dgbtrf
            self._solver_function = dgbtrs
        elif bmat.dtype == np.complex64:
            self._factor_function = cgbtrf
            self._solver_function = cgbtrs
        elif bmat.dtype == np.complex128:
            self._factor_function = zgbtrf
            self._solver_function = zgbtrs
        else:
            msg = f'Cannot create a DirectSolver for bmat.dtype = {bmat.dtype}'
            raise NotImplementedError(msg)

        self._bmat, self._ipiv, self._finfo = self._factor_function(bmat, l, u)

        self._sinfo = None

        self._space = np.ndarray
        self._dtype = bmat.dtype

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
        assert rhs.T.shape[0] == self._bmat.shape[1]

        if out is None:
            preout, self._sinfo = self._solver_function(self._bmat, self._l, self._u, rhs.T, self._ipiv,
                                                        trans=transposed)
            out = preout.T

        else:
            assert out.shape == rhs.shape
            assert out.dtype == rhs.dtype

            # support in-place operations
            if rhs is not out:
                out[:] = rhs

            # TODO: handle non-contiguous views?

            # we want FORTRAN-contiguous data (default is assumed to be C contiguous)
            _, self._sinfo = self._solver_function(self._bmat, self._l, self._u, out.T, self._ipiv, overwrite_b=True,
                                                   trans=transposed)

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

        self._space = np.ndarray
        self._inv  = inv( spmat.tocsc() )

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space( self ):
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
        
        assert rhs.T.shape[0] == self._inv.shape[1]

        if out is None:
            out = self._inv.dot(rhs.T).T

        else:
            assert out.shape == rhs.shape
            assert out.dtype == rhs.dtype

            # currently no in-place solve exposed
            out[:] = self._inv.dot(rhs.T).T

        return out
