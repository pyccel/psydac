#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os

import numpy as np
import scipy.fft as scifft

from psydac.linalg.basic import LinearOperator, LinearSolver
from psydac.linalg.stencil import StencilVectorSpace
from psydac.linalg.kron import KroneckerLinearSolver


class DistributedFFTBase(LinearOperator):
    """
    A base class for the distributed FFT, DCT and DST.
    Internally calls a KroneckerLinearSolver on a solver which just applies the FFT or some other function.

    Parameters
    ----------
    space : StencilVectorSpace
        The vector space needed for the KroneckerLinearSolver internally.
    
    function : callable | list/tuple of callables
        A list/tuple of callables function, each with one parameter x which applies some function in-place on x.
        The function at position i is applied to the i-th tensor direction.
        If only a single callable is given, it is used for all directions.
    """
    def toarray(self):
        raise NotImplementedError('toarray() is not defined for DistributedFFTBase.')

    def tosparse(self):
        raise NotImplementedError('tosparse() is not defined for DistributedFFTBase.')

    # Possible additions for the future:
    # * split off the LinearSolver class when used with the space ndarray (as used in the KroneckerLinearSolver),
    #   and make it state if it works in-place (or if it needs temporary memory), and what its optimal
    #   size is (FFT might work faster with padding)
    # * include FFTW support (e.g. pyfftw)

    class OneDimSolver(LinearSolver):
        """
        A one-dimensional solver which just applies a given function.

        Parameters
        ----------
        function : Callable
            The given function.
        """
        def __init__(self, function):
            self._function = function

        @property
        def space(self):
            return np.ndarray

        def transpose(self):
            raise NotImplementedError('transpose() is not implemented for OneDimSolvers')
        
        def solve(self, rhs, out=None):
            if out is None:
                out = np.empty_like(rhs)
            
            if out is not rhs:
                out[:] = rhs
            
            self._function(out)

            return out

    def __init__(self, space, functions):
        assert isinstance(space, StencilVectorSpace)
        if isinstance(functions, list) or isinstance(functions, tuple):
            solvers = [DistributedFFTBase.OneDimSolver(function) for function in functions]
        else:
            onedimsolver = DistributedFFTBase.OneDimSolver(functions)
            solvers = [onedimsolver] * space.ndim
        self._isolver = KroneckerLinearSolver(space, space, solvers)

    # ...
    @property
    def domain(self):
        return self._isolver.space

    # ...
    @property
    def codomain(self):
        return self._isolver.space

    # ...
    @property
    def dtype( self ):
        return self._isolver.dtype

    # ...
    def dot(self, v, out=None):
        # just call the KroneckerLinearSolver
        return self._isolver.solve(v, out=out)

    def transpose(self, conjugate=False):
        raise NotImplementedError()

# IMPORTANT NOTE: All of these scifft.fft functions currently trust that overwrite_x=True will yield an in-place fft...
# (this is not completely given to hold forever, so in case these tests fail in some future version, change this)

class DistributedFFT(DistributedFFTBase):
    """
    Equals an n-dimensional FFT operation, except that it works on a distributed/parallel StencilVector.

    Internally calls scipy.fft.fft for each direction.

    Parameters
    ----------
    space : StencilVectorSpace
        The space the n-dimensional FFT should be run on. Must have a complex data type (i.e. space.dtype.kind == 'c').
    
    norm : str
        Specifies the normalization factor. See the documentation of the corresponding scipy.fft.fft parameter.
    
    workers : Union[int, NoneType]
        Specifies the number of worker threads. By default set to the number of OpenMP threads, if given.
        See also the documentation of the corresponding scipy.fft.fft parameter.
    """
    def __init__(self, space, norm=None, workers=os.environ.get('OMP_NUM_THREADS', None)):
        # only allow complex data types
        assert isinstance(space, StencilVectorSpace)
        assert np.dtype(space.dtype).kind == 'c'
        workers = int(workers) if workers is not None else None

        super().__init__(space, lambda out: scifft.fft(
                out, axis=1, overwrite_x=True, workers=workers, norm=norm))


class DistributedIFFT(DistributedFFTBase):
    """
    Equals an n-dimensional IFFT operation, except that it works on a distributed/parallel StencilVector.

    Internally calls scipy.fft.ifft for each direction.

    Parameters
    ----------
    space : StencilVectorSpace
        The space the n-dimensional IFFT should be run on. Must have a complex data type (i.e. space.dtype.kind == 'c').
    
    norm : str
        Specifies the normalization factor. See the documentation of the corresponding scipy.fft.ifft parameter.
    
    workers : Union[int, NoneType]
        Specifies the number of worker threads. By default set to the number of OpenMP threads, if given.
        See also the documentation of the corresponding scipy.fft.ifft parameter.
    """
    def __init__(self, space, norm=None, workers=os.environ.get('OMP_NUM_THREADS', None)):
        # only allow complex data types
        assert isinstance(space, StencilVectorSpace)
        assert np.dtype(space.dtype).kind == 'c'
        workers = int(workers) if workers is not None else None
        
        super().__init__(space, lambda out: scifft.ifft(
                out, axis=1, overwrite_x=True, workers=workers, norm=norm))

class DistributedDCT(DistributedFFTBase):
    """
    Equals an n-dimensional DCT operation, except that it works on a distributed/parallel StencilVector.

    Internally calls scipy.fft.dct for each direction.

    Parameters
    ----------
    space : StencilVectorSpace
        The space the n-dimensional DCT should be run on.
    
    norm : str
        Specifies the normalization factor. See the documentation of the corresponding scipy.fft.dct parameter.
    
    workers : Union[int, NoneType]
        Specifies the number of worker threads. By default set to the number of OpenMP threads, if given.
        See also the documentation of the corresponding scipy.fft.dct parameter.
    
    ttype : int
        The DCT type to use. (the name of this parameter in the underlying method is actually `type`).
    """
    def __init__(self, space, norm=None, workers=os.environ.get('OMP_NUM_THREADS', None), ttype=2):
        workers = int(workers) if workers is not None else None
        super().__init__(space, lambda out: scifft.dct(
                out, axis=1, overwrite_x=True, workers=workers, norm=norm, type=ttype))

class DistributedIDCT(DistributedFFTBase):
    """
    Equals an n-dimensional IDCT operation, except that it works on a distributed/parallel StencilVector.

    Internally calls scipy.fft.idct for each direction.

    Parameters
    ----------
    space : StencilVectorSpace
        The space the n-dimensional IDCT should be run on.
    
    norm : str
        Specifies the normalization factor. See the documentation of the corresponding scipy.fft.idct parameter.
    
    workers : Union[int, NoneType]
        Specifies the number of worker threads. By default set to the number of OpenMP threads, if given.
        See also the documentation of the corresponding scipy.fft.idct parameter.
    
    ttype : int
        The DCT type to use. (the name of this parameter in the underlying method is actually `type`).
    """
    def __init__(self, space, norm=None, workers=os.environ.get('OMP_NUM_THREADS', None), ttype=2):
        workers = int(workers) if workers is not None else None
        super().__init__(space, lambda out: scifft.idct(
                out, axis=1, overwrite_x=True, workers=workers, norm=norm, type=ttype)) 

class DistributedDST(DistributedFFTBase):
    """
    Equals an n-dimensional DST operation, except that it works on a distributed/parallel StencilVector.

    Internally calls scipy.fft.dst for each direction.

    Parameters
    ----------
    space : StencilVectorSpace
        The space the n-dimensional DST should be run on.
    
    norm : str
        Specifies the normalization factor. See the documentation of the corresponding scipy.fft.dst parameter.
    
    workers : Union[int, NoneType]
        Specifies the number of worker threads. By default set to the number of OpenMP threads, if given.
        See also the documentation of the corresponding scipy.fft.dst parameter.
    
    ttype : int
        The DCT type to use. (the name of this parameter in the underlying method is actually `type`).
    """
    def __init__(self, space, norm=None, workers=os.environ.get('OMP_NUM_THREADS', None), ttype=2):
        workers = int(workers) if workers is not None else None
        super().__init__(space, lambda out: scifft.dst(
                out, axis=1, overwrite_x=True, workers=workers, norm=norm, type=ttype))

class DistributedIDST(DistributedFFTBase):
    """
    Equals an n-dimensional IDST operation, except that it works on a distributed/parallel StencilVector.

    Internally calls scipy.fft.idst for each direction.

    Parameters
    ----------
    space : StencilVectorSpace
        The space the n-dimensional IDST should be run on.
    
    norm : str
        Specifies the normalization factor. See the documentation of the corresponding scipy.fft.idst parameter.
    
    workers : Union[int, NoneType]
        Specifies the number of worker threads. By default set to the number of OpenMP threads, if given.
        See also the documentation of the corresponding scipy.fft.idst parameter.
    
    ttype : int
        The DCT type to use. (the name of this parameter in the underlying method is actually `type`).
    """
    def __init__(self, space, norm=None, workers=os.environ.get('OMP_NUM_THREADS', None), ttype=2):
        workers = int(workers) if workers is not None else None
        super().__init__(space, lambda out: scifft.idst(
                out, axis=1, overwrite_x=True, workers=workers, norm=norm, type=ttype)) 
