#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import scipy.fft as scifft
import numpy as np
from mpi4py import MPI

from psydac.linalg.fft import *
from psydac.ddm.cart               import DomainDecomposition, CartDecomposition
from psydac.linalg.stencil import StencilVector
#===============================================================================
def compute_global_starts_ends(domain_decomposition, npts):
    ndims         = len(npts)
    global_starts = [None]*ndims
    global_ends   = [None]*ndims

    for axis in range(ndims):
        es = domain_decomposition.global_element_starts[axis]
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    return global_starts, global_ends
#===============================================================================
def decode_fft_type(ffttype):
    if ffttype == 'fft':
        return (complex, DistributedFFT, scifft.fftn)
    elif ffttype == 'ifft':
        return (complex, DistributedIFFT, scifft.ifftn)
    elif ffttype == 'dct':
        return (complex, DistributedDCT, scifft.dctn)
    elif ffttype == 'idct':
        return (complex, DistributedIDCT, scifft.idctn)
    elif ffttype == 'dst':
        return (complex, DistributedDST, scifft.dstn)
    elif ffttype == 'idst':
        return (complex, DistributedIDST, scifft.idstn)
    else:
        raise NotImplementedError()

def method_test(seed, comm, config, dtype, classtype, comparison, verbose=False):
    np.random.seed(seed)

    if comm is None:
        rank = -1
    else:
        rank = comm.Get_rank()
    
    npts, pads, periods = config

    if verbose:
        print(f'[{rank}] Test start', flush=True)

    comm = MPI.COMM_WORLD
    D = DomainDecomposition(npts, periods=periods, comm=comm)

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=pads, shifts=[1]*len(pads))

    # vector spaces
    V = StencilVectorSpace(cart, dtype=dtype)
    localslice = tuple([slice(s, e+1) for s, e in zip(V.starts, V.ends)])

    if verbose:
        print(f'[{rank}] Vector spaces built', flush=True)

    if np.dtype(dtype).kind == 'c':
        Y_glob = np.random.random(V.npts) + np.random.random(V.npts) * 1j
    else:
        Y_glob = np.random.random(V.npts)

    # vector to solve for (Y)
    Y = StencilVector(V)
    Y[localslice] = Y_glob[localslice]
    Y.update_ghost_regions()

    if verbose:
        print(f'[{rank}] Vector built', flush=True)

    X_glob = comparison(Y_glob)

    compare = classtype(V)
    X = compare.dot(Y)

    if verbose:
        print(f'[{rank}] Functions have been run', flush=True)

    assert np.allclose(X_glob[localslice], X[localslice], 1e-10, 1e-10)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'params', [([8], [2], [False]), ([8,9], [2,3], [False,True]), ([8,9,17], [2,3,7], [False,True,False])] )
@pytest.mark.parametrize( 'ffttype', ['fft', 'ifft', 'dct', 'idct', 'dst', 'idst'] )
def test_kron_fft_ser(seed, params, ffttype):
    method_test(seed, None, params, *decode_fft_type(ffttype), verbose=False)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'params', [([16], [2], [False]), ([16,18], [2,3], [False,True]), ([16,18,37], [2,3,7], [False,True,False])] )
@pytest.mark.parametrize( 'ffttype', ['fft', 'ifft', 'dct', 'idct', 'dst', 'idst'] )
@pytest.mark.mpi
def test_kron_fft_par(seed, params, ffttype):
    method_test(seed, MPI.COMM_WORLD, params, *decode_fft_type(ffttype), verbose=False)

if __name__ == '__main__':
    method_test(0, MPI.COMM_WORLD, ([8,9,5], [2,3,7], [False,True,False]), *decode_fft_type('fft'), verbose=True)
