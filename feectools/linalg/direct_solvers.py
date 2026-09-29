#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from abc                 import abstractmethod
import cunumpy as xp
from cunumpy.xp import array_backend
from scipy.sparse        import spmatrix, dia_matrix

from feectools.linalg.basic    import LinearSolver

__all__ = ('to_bnd', 'BandedSolver', 'SparseSolver')

#===============================================================================
def to_bnd(A):
    """Converts a 1D StencilMatrix to a band matrix"""

    dmat = dia_matrix(A.toarray(), dtype=A.dtype)
    la   = abs(dmat.offsets.min())
    ua   = dmat.offsets.max()
    cmat = dmat.tocsr()

    A_bnd = xp.zeros((1+ua+2*la, cmat.shape[1]), dtype=A.dtype)

    for i,j in zip(*cmat.nonzero()):
        A_bnd[la+ua+i-j, j] = cmat[i,j]

    return A_bnd, la, ua

#===============================================================================
class BandedSolver(LinearSolver):
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
    def __init__(self, u, l, bmat, transposed=False):

        self._u    = u
        self._l    = l
        self._transposed = transposed

        # imported here: scipy.linalg costs ~0.3 s to import and is only needed once a solver is built
        from scipy.linalg.lapack import dgbtrf, dgbtrs, sgbtrf, sgbtrs, cgbtrf, cgbtrs, zgbtrf, zgbtrs

        # ... LU factorization
        if bmat.dtype == xp.float32:
            self._factor_function = sgbtrf
            self._solver_function = sgbtrs
        elif bmat.dtype == xp.float64:
            self._factor_function = dgbtrf
            self._solver_function = dgbtrs
        elif bmat.dtype == xp.complex64:
            self._factor_function = cgbtrf
            self._solver_function = cgbtrs
        elif bmat.dtype == xp.complex128:
            self._factor_function = zgbtrf
            self._solver_function = zgbtrs
        else:
            msg = f'Cannot create a BandedSolver for bmat.dtype = {bmat.dtype}'
            raise NotImplementedError(msg)
        # print(f"{bmat = } {type(bmat) = }")
        if hasattr(bmat, "get"):  # CuPy array
            bmat = bmat.get()
        else:
            bmat = xp.asanyarray(bmat)
        self._bmat, self._ipiv, self._finfo = self._factor_function(bmat, l, u)

        self._sinfo = None

        self._space = xp.ndarray
        self._dtype = bmat.dtype

    @staticmethod
    def from_stencil_mat_1d(A):
        """Converts a 1D StencilMatrix to a BandedSolver."""

        A.remove_spurious_entries()
        A_bnd, la, ua = to_bnd(A)
        return BandedSolver(ua, la, A_bnd)

    @property
    def finfo(self):
        return self._finfo

    @property
    def sinfo(self):
        return self._sinfo

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space(self):
        return self._space

    def transpose(self):
        cls = type(self)
        obj = super().__new__(cls)

        obj._u = self._l
        obj._l = self._u
        obj._bmat = self._bmat
        obj._ipiv = self._ipiv
        obj._finfo = self._finfo
        obj._factor_function = self._factor_function
        obj._solver_function = self._solver_function
        obj._sinfo = None
        obj._space = self._space
        obj._dtype = self._dtype
        obj._transposed = not self._transposed

        return obj

    #...
    def solve(self, rhs, out=None):
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
        """
        assert rhs.T.shape[0] == self._bmat.shape[1]

        transposed = self._transposed

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
            from cunumpy.xp import array_backend
            if array_backend.backend == "numpy":
                _, self._sinfo = self._solver_function(self._bmat, self._l, self._u, out.T, self._ipiv, overwrite_b=True,
                                                   trans=transposed)
            else:
                # GPU
                out_cpu = out.get()
                _, self._sinfo = self._solver_function(self._bmat, self._l, self._u, out_cpu.T, self._ipiv, overwrite_b=True,
                                                   trans=transposed)
                out.set(out_cpu)

        return out

#===============================================================================
class SparseSolver (LinearSolver):
    """
    Solve the equation Ax = b for x, assuming A is scipy sparse matrix.

    Parameters
    ----------
    spmat : scipy.sparse.spmatrix
        Generic sparse matrix.

    """
    def __init__(self, spmat, transposed=False):

        assert isinstance(spmat, spmatrix)

        from scipy.sparse.linalg import splu  # deferred, see BandedSolver

        self._space = xp.ndarray
        self._splu  = splu(spmat.tocsc())
        self._transposed = transposed

    #--------------------------------------
    # Abstract interface
    #--------------------------------------
    @property
    def space(self):
        return self._space

    def transpose(self):
        cls = type(self)
        obj = super().__new__(cls)

        obj._space = self._space
        obj._splu = self._splu
        obj._transposed = not self._transposed

        return obj

    #...
    def solve(self, rhs, out=None):
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
        """
        
        assert rhs.T.shape[0] == self._splu.shape[1]
        transposed = self._transposed

        if out is None:
            out = self._splu.solve(rhs.T, trans='T' if transposed else 'N').T

        else:
            assert out.shape == rhs.shape
            assert out.dtype == rhs.dtype

            # currently no in-place solve exposed
            if array_backend.backend == "numpy":
                out[:] = self._splu.solve(rhs.T, trans='T' if transposed else 'N').T
            else:
                rhs_cpu = rhs.get()
                result_cpu = self._splu.solve(rhs_cpu.T, trans='T' if transposed else 'N').T
                out[:] = xp.asarray(result_cpu)

        return out
