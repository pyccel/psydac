import pytest
import numpy as np
from mpi4py import MPI

from psydac.linalg.block   import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.basic   import InverseLinearOperator, PowerLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.solvers import inverse
from psydac.ddm.cart       import DomainDecomposition, CartDecomposition

#===============================================================================
n1array = [2, 7]
n2array = [2, 3]
p1array = [1, 3]
p2array = [1, 3]

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

def get_StencilVectorSpace(n1, n2, p1, p2, P1, P2):
    comm = MPI.COMM_WORLD
    npts = [n1, n2]
    pads = [p1, p2]
    periods = [P1, P2]
    D = DomainDecomposition(npts, periods=periods, comm = comm)
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=pads, shifts=[1,1])
    V = StencilVectorSpace(C)
    return V

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize('n1', n1array)
@pytest.mark.parametrize('n2', n2array)
@pytest.mark.parametrize('p1', p1array)
@pytest.mark.parametrize('p2', p2array)

def test_square_stencil_basic(n1, n2, p1, p2, P1=False, P2=False):

    ###
    ### 1. Initiation
    ###

    # Initiate StencilVectorSpace
    V = get_StencilVectorSpace(n1, n2, p1, p2, P1, P2)

    S = StencilMatrix(V, V)

    # Initiate a StencilVector
    v = StencilVector(V)
    for i in range(n1):
        for j in range(n2):
            v[i,j] = 1

    nonzero_values = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values[k1,k2] = 1 + k1*n2 + k2
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1==0:
                if k2<0:
                    nonzero_values[k1,k2] = nonzero_values[-k1,-k2]
            elif k1<0:
                nonzero_values[k1,k2] = nonzero_values[-k1,-k2]

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S[:,:,k1,k2] = nonzero_values[k1,k2]
    S.remove_spurious_entries()

    # Construct exact matrices by hand
    A1 = np.zeros( S.shape )
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1,p1+1):
                for k2 in range(-p2,p2+1):
                    j1 = (i1+k1) % n1
                    j2 = (i2+k2) % n2
                    i  = i1*(n2) + i2
                    j  = j1*(n2) + j2
                    if (P1 or 0 <= i1+k1 < n1) and (P2 or 0 <= i2+k2 < n2):
                        A1[i,j] = nonzero_values[k1,k2]

    ###
    ### Use PowerLinearOperator to test toarray function.
    ###
    ###
    # Raising a StencilMatrix to a power works
    Spo3 = S**3
    
    #We call toarray on the PowerLinearOperator
    Spo3arr = Spo3.toarray()
    
    #Again we know that A1 is the matrix representation of S. Thus, if we take A1 A1 A1 it should be the same as Sp3arr
    Apoarr = np.matmul(A1, A1)
    Apoarr = np.matmul(Apoarr, A1)
    
    assert isinstance(Spo3, PowerLinearOperator)
    assert np.array_equal(Spo3arr, Apoarr)
        
    ###
    ### Use InverseLinearOperator to test toarray function.
    ###
    ###
    #We invert S
    Sinv = inverse(S, "bicgstab")
    Sinvarr = Sinv.toarray()
    
    assert isinstance(Sinv, InverseLinearOperator)
    
    Itest = np.matmul(A1,Sinvarr)
    assert np.allclose(Itest, np.eye(np.size(Itest[0])),atol= 10**(-5))

#===============================================================================
@pytest.mark.parametrize('n1', n1array)
@pytest.mark.parametrize('n2', n2array)
@pytest.mark.parametrize('p1', p1array)
@pytest.mark.parametrize('p2', p2array)

def test_square_block_basic(n1, n2, p1, p2, P1=False, P2=False):

    # Initiate StencilVectorSpace
    V = get_StencilVectorSpace(n1, n2, p1, p2, P1, P2)
       
    # Initiate Linear Operators    
    S = StencilMatrix(V, V)

    # Initiate a StencilVector
    v = StencilVector(V)
    for i in range(n1):
        for j in range(n2):
            v[i,j] = float(i*3.5 + j*10.0)    

    nonzero_values = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values[k1,k2] = 1 + k1*n2 + k2
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1==0:
                if k2<0:
                    nonzero_values[k1,k2] = nonzero_values[-k1,-k2]
            elif k1<0:
                nonzero_values[k1,k2] = nonzero_values[-k1,-k2]

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S[:,:,k1,k2] = nonzero_values[k1,k2]
    S.remove_spurious_entries()  

    # Initiate a BlockVectorSpace
    Vb = BlockVectorSpace(V,V)

    # Initiate BlockLOs and LOs acting on BlockVectorSpaces
    B  = BlockLinearOperator(Vb, Vb, ((S, None), (None, S)))
    
    # Initiate a BlockVector
    vb = BlockVector(Vb, (v, v))
    vbarr = vb.toarray()
    
    ###
    ### Test toarray with power of one BlockLinearOperators and taking dot product with a block vector
    ### 
    ###
    Bpow = B**3
    Bpowarr = Bpow.toarray()
    assert isinstance(Bpow, PowerLinearOperator)
    assert np.array_equal(np.matmul(Bpowarr,vbarr),Bpow.dot(vb).toarray())

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )