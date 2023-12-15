# -*- coding: UTF-8 -*-
#
import pytest
import numpy as np
import scipy.sparse as spa

from psydac.linalg.basic import LinearOperator, ZeroOperator, IdentityOperator, ComposedLinearOperator, InverseLinearOperator, SumLinearOperator, PowerLinearOperator, ScaledLinearOperator
from psydac.linalg.direct_solvers import SparseSolver
from psydac.linalg.stencil        import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.block          import BlockVectorSpace, BlockVector
from psydac.linalg.block          import BlockLinearOperator
from psydac.linalg.utilities      import array_to_psydac
from psydac.linalg.kron           import KroneckerLinearSolver
from psydac.api.settings          import PSYDAC_BACKEND_GPYCCEL
from psydac.ddm.cart              import DomainDecomposition, CartDecomposition

#===============================================================================
def compare_arrays(arr_psy, arr, rank, atol=1e-14, verbose=False):
    '''Assert equality of distributed psydac array and corresponding fraction of cloned numpy array.
    Arrays can be block-structured as nested lists/tuples.

    Parameters
    ----------
        arr_psy : psydac object
            Stencil/Block Vector/Matrix.

        arr : array
            Numpy array with same tuple/list structure as arr_psy. If arr is a matrix it can be stencil or band format.

        rank : int
            Rank of mpi process

        atol : float
            Absolute tolerance used in numpy.allclose()
    '''
    if isinstance(arr_psy, StencilVector):

        s = arr_psy.starts
        e = arr_psy.ends

        tmp1 = arr_psy[s[0]: e[0] + 1, s[1]: e[1] + 1]

        if arr.ndim == 2:
            tmp2 = arr[s[0]: e[0] + 1, s[1]: e[1] + 1]
        else:
            tmp2 = arr.reshape(arr_psy.space.npts[0],
                               arr_psy.space.npts[1])[s[0]: e[0] + 1, s[1]: e[1] + 1]

        assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, BlockVector):

        if not (isinstance(arr, tuple) or isinstance(arr, list)):
            arrs = np.split(arr, [arr_psy.blocks[0].shape[0],
                            arr_psy.blocks[0].shape[0] + arr_psy.blocks[1].shape[0]])
        else:
            arrs = arr

        for vec_psy, vec in zip(arr_psy.blocks, arrs):
            s = vec_psy.starts
            e = vec_psy.ends

            tmp1 = vec_psy[s[0]: e[0] + 1, s[1]: e[1] + 1]

            if vec.ndim == 2:
                tmp2 = vec[s[0]: e[0] + 1, s[1]: e[1] + 1]
            else:
                tmp2 = vec.reshape(vec_psy.space.npts[0], vec_psy.space.npts[1])[
                    s[0]: e[0] + 1, s[1]: e[1] + 1]

            assert np.allclose(tmp1, tmp2, atol=atol)
    else:
        raise AssertionError('Wrong input type.')

    if verbose:
        print(
            f'Rank {rank}: Assertion for array comparison passed with atol={atol}.')

def compute_global_starts_ends(domain_decomposition, npts):
    global_starts = [None]*len(npts)
    global_ends   = [None]*len(npts)

    for axis in range(len(npts)):
        es = domain_decomposition.global_element_starts[axis]
        ee = domain_decomposition.global_element_ends  [axis]

        global_ends  [axis]     = ee.copy()
        global_ends  [axis][-1] = npts[axis]-1
        global_starts[axis]     = np.array([0] + (global_ends[axis][:-1]+1).tolist())

    return global_starts, global_ends

def create_equal_random_arrays(W, seedv =123):
    np.random.seed(seedv)
    arr = []
    if isinstance(W, BlockVectorSpace):
        arr_psy = BlockVector(W)

        for d, block in enumerate(arr_psy.blocks):

            dims = W.spaces[d].npts

            arr += [np.random.rand(*dims)]

            s = block.starts
            e = block.ends

            arr_psy[d][s[0]:e[0] + 1, s[1]:e[1] + 1] = arr[-1][s[0]:e[0] + 1, s[1]:e[1] + 1]
        arr_psy.update_ghost_regions()
    elif isinstance(W, StencilVectorSpace):
        arr_psy = StencilVector(W)

        dims = W.npts

        arr += [np.random.rand(*dims)]

        s = arr_psy.starts
        e = arr_psy.ends

        arr_psy[s[0]:e[0] + 1, s[1]:e[1] + 1] = arr[-1][s[0]:e[0] + 1, s[1]:e[1] + 1]

    else:
        raise Exception(
                'W must be a BlockVectorSpace or a StencilVectorSpace.')

    return arr, arr_psy



@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'n1', [16] )
@pytest.mark.parametrize( 'n2', [16, 32] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True] )
@pytest.mark.parallel

def test_block_linear_operator_parallel_dot( dtype, n1, n2, p1, p2, P1, P2 ):
    # Define a factor for the data
    if dtype == complex:
        factor = 1j
    else:
        factor = 1

    # set seed for reproducibility
    np.random.seed(n1*n2*p1*p2)

    from mpi4py       import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    D = DomainDecomposition([n1,n2], periods=[P1,P2], comm=comm)

    # Partition the points
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart, dtype=dtype )

    v0arr , v0 = create_equal_random_arrays(V, seedv=4568)
    
    v0arr = v0arr[0].flatten()
    
    # Create and Fill Block objects
    W = BlockVectorSpace(V, V)
    
    v1arr1, v1 = create_equal_random_arrays(W, seedv=4568)
    v1arr = []
    for i in v1arr1:
        aux = i.flatten()
        for j in aux:
            v1arr.append(j)
 
    # Create random matrix 
    N1 = StencilMatrix( V, V )
    N2 = StencilMatrix( V, V )
    N3 = StencilMatrix( V, V )
    N4 = StencilMatrix( V, V )

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            N1[:,:,k1,k2] = factor*np.random.random()
            N2[:,:,k1,k2] = factor*np.random.random()
            N3[:,:,k1,k2] = factor*np.random.random()
            N4[:,:,k1,k2] = factor*np.random.random()

    K = BlockLinearOperator( W, W )

    K[0,0] = N1
    K[0,1] = N2
    K[1,0] = N3
    K[1,1] = N4
    
    
    #####
    #Testing tosparse() for LinearOperators with domain being StencilVectorSpace
    
    #We take a power of the Stencil Matrix N1
    N1P = N1**3
    #We turn our PowerLinearOperator into a sparse matrix, and then into a numpy array
    N1arrloc = N1P.tosparse().toarray()
    #We get the global matrix of the PowerLinearOperator
    N1arr = np.zeros(np.shape(N1arrloc), dtype=dtype) 
    comm.Allreduce(N1arrloc, N1arr, op=MPI.SUM)
    
    assert isinstance(N1P, PowerLinearOperator)
    assert isinstance(N1P.domain, StencilVectorSpace)
    compare_arrays(N1P.dot(v0), N1arr.dot(v0arr), rank)
    
    #####
    #Testing tosparse() for LinearOperators with domain being BlockVectorSpace
    
    #We take a power of the BlockLinearOperator
    KP = K**3
    #We turn our PowerLinearOperator into a sparse matrix, and then into a numpy array
    KLarrloc = KP.tosparse().toarray()
    #We get the global matrix of the operator
    KLarr = np.zeros(np.shape(KLarrloc), dtype=dtype)
    comm.Allreduce(KLarrloc, KLarr, op=MPI.SUM)

    assert isinstance(KP, PowerLinearOperator)
    assert isinstance(KP.domain, BlockVectorSpace)
    compare_arrays(KP.dot(v1), KLarr.dot(v1arr), rank)
    
    

    

    
#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )