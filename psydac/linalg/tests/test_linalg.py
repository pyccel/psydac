#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.basic import LinearOperator, ZeroOperator, IdentityOperator, ComposedLinearOperator, SumLinearOperator, PowerLinearOperator, ScaledLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.solvers import ConjugateGradient, inverse
from psydac.ddm.cart       import DomainDecomposition, CartDecomposition

#===============================================================================

n1array = [2, 7]
n2array = [2, 3]
p1array = [1, 3]
p2array = [1, 3]

def array_equal(a, b):
    return np.array_equal(a.toarray(), b.toarray())

def sparse_equal(a, b):
    return (a.tosparse() != b.tosparse()).nnz == 0

def assert_pos_def(A):
    assert isinstance(A, LinearOperator)
    A_array = A.toarray()
    assert np.all(np.linalg.eigvals(A_array) > 0)

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

def get_StencilVectorSpace(npts, pads, periods):
    assert len(npts) == len(pads) == len(periods)
    shifts = [1] * len(npts)
    D = DomainDecomposition(npts, periods=periods)
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=pads, shifts=shifts)
    V = StencilVectorSpace(C)
    return V

def get_positive_definite_StencilMatrix(V):

    np.random.seed(2)
    assert isinstance(V, StencilVectorSpace)
    [n1, n2] = V._npts
    [p1, p2] = V._pads
    [P1, P2] = V._periods
    assert (P1 == False) and (P2 == False)
    
    S = StencilMatrix(V, V)

    for i in range(0, p1+1):
        if i != 0:
            for j in range(-p2, p2+1):
                S[:, :, i, j] = 2*np.random.random()-1
        else:
            for j in range(1, p2+1):
                S[:, :, i, j] = 2*np.random.random()-1
    S += S.T
    S[:, :, 0, 0] = ((n1 * n2) - 1) / np.random.random()
    S /= S[0, 0, 0, 0]
    S.remove_spurious_entries()

    return S

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize('n1', n1array)
@pytest.mark.parametrize('n2', n2array)
@pytest.mark.parametrize('p1', p1array)
@pytest.mark.parametrize('p2', p2array)

def test_square_stencil_basic(n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs S,S1 (StencilMatrix), I (IdentityOperator), Z (ZeroOperator) and a Stencilvector v
    # 2. Test general basic operations
    # 3. Test special cases

    ###
    ### 1. Initiation
    ###

    # Initiate StencilVectorSpace
    V = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    
    # Initiate Linear Operators
    Z = ZeroOperator(V, V)
    I = IdentityOperator(V, V)
    S = StencilMatrix(V, V)
    S1 = StencilMatrix(V, V)
    # a non-symmetric StencilMatrix for transpose testing
    S2 = StencilMatrix(V, V)

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
    Sa = S.toarray()

    nonzero_values1 = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1 == 0:
                if k2 == 0:
                    nonzero_values1[k1,k2] = 1
                else:
                    nonzero_values1[k1,k2] = 0
            else:
                nonzero_values1[k1,k2] = 0
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S1[:,:,k1,k2] = nonzero_values1[k1,k2]
    S1.remove_spurious_entries()
    S1a = S1.toarray()

    nonzero_values2 = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values2[k1,k2] = 1 + k1*n2 + k2
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1==0:
                if k2<0:
                    nonzero_values2[k1,k2] = 0
            elif k1<0:
                nonzero_values2[k1,k2] = 0
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S2[:,:,k1,k2] = nonzero_values2[k1,k2]
    S2.remove_spurious_entries()
    S2a = S2.toarray()

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

    A2 = np.zeros( S.shape )
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1,p1+1):
                for k2 in range(-p2,p2+1):
                    j1 = (i1+k1) % n1
                    j2 = (i2+k2) % n2
                    i  = i1*(n2) + i2
                    j  = j1*(n2) + j2
                    if (P1 or 0 <= i1+k1 < n1) and (P2 or 0 <= i2+k2 < n2):
                        A2[i,j] = nonzero_values1[k1,k2]

    # Check shape and data in 2D array
    assert np.array_equal(v.toarray(), np.ones(n1 * n2))

    assert Sa.shape == S.shape
    assert np.array_equal( Sa, A1 )
    assert S1a.shape == S1.shape
    assert np.array_equal( S1a, A2 )

    ###
    ### 2. Test general basic operations
    ### Addition, Substraction, Negation, Multiplication, Composition, Raising to a Power, Transposing
    ###

    ## ___Addition and Substraction, also Negation___

    # Adding and Substracting StencilMatrices returns a StencilMatrix
    assert isinstance(S + S1, StencilMatrix)
    assert isinstance(S - S1, StencilMatrix)

    # Adding and Substracting StencilMatrices and other LOs returns a SumLinearOperator object
    assert isinstance(S + I, SumLinearOperator)
    assert isinstance(I + S, SumLinearOperator)
    assert isinstance(S - I, SumLinearOperator)
    assert isinstance(I - S, SumLinearOperator)

    # Negating a StencilMatrix works as intended
    assert isinstance(-S, StencilMatrix)
    assert array_equal((-S).dot(v), -S.dot(v))

    ## ___Multiplication, Composition, Raising to a Power___

    # Multiplying and Dividing a StencilMatrix by a scalar returns a StencilMatrix
    assert isinstance(np.pi * S, StencilMatrix)
    assert isinstance(S * np.pi, StencilMatrix)
    assert isinstance(S / np.pi, StencilMatrix)

    # Composing StencilMatrices works
    assert isinstance(S @ S1, ComposedLinearOperator)

    # Raising a StencilMatrix to a power works
    assert isinstance(S**3, PowerLinearOperator)

    ## ___Transposing___
    
    assert not np.array_equal(S2a, S2a.T) # using a nonsymmetric matrix throughout
    assert isinstance(S2.T, StencilMatrix)
    assert np.array_equal(S2.T.toarray(), S2a.T)
    assert np.array_equal(S2.T.T.toarray(), S2a)

    ###
    ### 3. Test special cases
    ### Add. und Sub. with ZeroO's, Composition with Zero- and IdentityO's, Raising to the power of 0 and 1
    ###

    ## ___Addition and Substraction with ZeroOperators___

    # Adding a ZeroOperator does not change the StencilMatrix
    assert (S + Z) is S
    assert (Z + S) is S

    # Substracting a ZeroOperator and substracting from a ZeroOperator work as intended
    assert (S - Z) is S
    assert array_equal(-S, Z - S)

    ## ___Composing with Zero- and IdentityOperators___   

    # Composing a StencilMatrix with a ZeroOperator returns a ZeroOperator
    assert isinstance(S @ Z, ZeroOperator)
    assert isinstance(Z @ S, ZeroOperator)

    # Composing a StencilMatrix with the IdentityOperator does not change the object
    assert (S @ I) is S
    assert (I @ S) is S

    ## ___Raising to the power of 0 and 1___
    
    # Raising a StencilMatrix to the power of 1 or 0 does not change the object / returns an IdentityOperator
    assert S**1 is S
    assert isinstance(S**0, IdentityOperator)

#===============================================================================
@pytest.mark.parametrize('n1', n1array)
@pytest.mark.parametrize('n2', n2array)
@pytest.mark.parametrize('p1', p1array)
@pytest.mark.parametrize('p2', p2array)

def test_square_block_basic(n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs S,S1 (StencilMatrix), Z (ZeroOperator) and a Stencilvector v
    #    Initiate square LOs B,B1 (BlockLO), BZ (ZeroOperator), BI (IdentityOperator) and a BlockVector vb
    # 2. Test general basic operations
    # 3. Test special cases

    # Initiate StencilVectorSpace
    V = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
       
    # Initiate Linear Operators
    Z = ZeroOperator(V, V)    
    S = StencilMatrix(V, V)
    S1 = StencilMatrix(V, V)
    # a non-symmetric StencilMatrix for transpose testing
    S2 = StencilMatrix(V,V)

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

    nonzero_values1 = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1 == 0:
                if k2 == 0:
                    nonzero_values1[k1,k2] = 1
                else:
                    nonzero_values1[k1,k2] = 0
            else:
                nonzero_values1[k1,k2] = 0
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S1[:,:,k1,k2] = nonzero_values1[k1,k2]
    S1.remove_spurious_entries()

    nonzero_values2 = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values2[k1,k2] = 1 + k1*n2 + k2
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1==0:
                if k2<0:
                    nonzero_values2[k1,k2] = 0
            elif k1<0:
                nonzero_values2[k1,k2] = 0
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S2[:,:,k1,k2] = nonzero_values2[k1,k2]
    S2.remove_spurious_entries()

    # Initiate a BlockVectorSpace
    Vb = BlockVectorSpace(V,V)

    # Initiate BlockLOs and LOs acting on BlockVectorSpaces
    BZ = ZeroOperator(Vb, Vb)
    BI = IdentityOperator(Vb, Vb)
    B  = BlockLinearOperator(Vb, Vb, ((S, None), (None, S)))
    B1 = BlockLinearOperator(Vb, Vb, ((S1, None), (Z, S1)))
    B2 = BlockLinearOperator(Vb, Vb, ((S2, None), (None, S2)))

    # Initiate a BlockVector
    vb = BlockVector(Vb, (v, v))

    ###
    ### 2. Test general basic operations
    ### Addition, Substraction, Negation, Multiplication, Composition, Raising to a Power, Transposing
    ###

    ## ___Addition and Substraction, also Negation___

    # Adding and Substracting BlockLOs returns a BlockLO
    assert isinstance(B + B1, BlockLinearOperator)
    assert isinstance(B - B1, BlockLinearOperator)

    # Adding and Substracting BlockLOs and other LOs returns a SumLinearOperator object
    assert isinstance(B + BI, SumLinearOperator)
    assert isinstance(BI + B, SumLinearOperator)
    assert isinstance(B - BI, SumLinearOperator)
    assert isinstance(BI - B, SumLinearOperator)

    # Negating a BlockLO works as intended
    assert isinstance(-B, BlockLinearOperator)
    assert array_equal((-B).dot(vb), -B.dot(vb))

    ## ___Multiplication, Composition, Raising to a Power___

    # Multiplying and Dividing a BlockLO by a scalar returns a BlockLO
    assert isinstance(np.pi * B, BlockLinearOperator)
    assert isinstance(B * np.pi, BlockLinearOperator)
    assert isinstance(B / np.pi, BlockLinearOperator)

    # Composing BlockLOs works
    assert isinstance(B @ B1, ComposedLinearOperator)

    # Raising a BlockLO to a power works
    assert isinstance(B**3, PowerLinearOperator)

    ## ___Transposing___
    assert not np.array_equal(B2.toarray(), B2.toarray().T) # using a nonsymmetric matrix throughout
    assert isinstance(B2.T, BlockLinearOperator)
    assert np.array_equal(B2.T.toarray(), B2.toarray().T)
    assert np.array_equal(B2.T.T.toarray(), B2.toarray())

    ###
    ### 3. Test special cases
    ### Add. und Sub. with ZeroO's, Composition with Zero- and IdentityO's, Raising to the power of 0 and 1
    ###

    ## ___Addition and Substraction with ZeroOperators___

    # Adding a ZeroOperator does not change the BlockLO
    BBZ = B + BZ
    BZB = BZ + B
    assert sparse_equal(BBZ, B)
    assert sparse_equal(BZB, B)

    # Substracting a ZeroOperator and substracting from a ZeroOperator work as intended
    BmBZ = B - BZ
    BZmB = BZ - B
    assert sparse_equal(BmBZ,  B)
    assert sparse_equal(BZmB, -B)

    ## ___Composing with Zero- and IdentityOperators___ 

    # Composing a BlockLO with a ZeroOperator returns a ZeroOperator
    # Update 21.12.: ZeroLOs and IdentityLOs from and/or to BlockVectorSpaces are now BlockLOs
    # thus B@BZ is now a ComposedLO.
    assert isinstance(B @ BZ, ComposedLinearOperator)
    assert isinstance(BZ @ B, ComposedLinearOperator)

    # Composing a BlockLO with the IdentityOperator does not change the object
    assert B @ BI == B
    assert BI @ B == B

    ## ___Raising to the power of 1___

    # Raising a BlockLO to the power of 1 does not change the object
    assert B**1 is B

    ## ___Raising to the power of 0___

    # Raising a BlockLO to the power of 0 returns an IdentityOperator
    assert isinstance(B**0, IdentityOperator)
    assert sparse_equal(B**0, BI)

#===============================================================================
@pytest.mark.parametrize('n1', n1array)
@pytest.mark.parametrize('n2', n2array)
@pytest.mark.parametrize('p1', p1array)
@pytest.mark.parametrize('p2', p2array)

def test_in_place_operations(n1, n2, p1, p2, P1=False, P2=False):

    # testing __imul__ although not explicitly implemented (in the LinearOperator class)

    # Initiate StencilVectorSpace
    V  = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    Vc = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    Vc._dtype = complex
    v = StencilVector(V)
    vc = StencilVector(Vc)
    v_array = np.zeros(n1*n2)

    for i in range(n1):
        for j in range(n2):
            v[i,j] = i+1
            v_array[i*n2+j] = i+1
            vc[i,j] = i+1       
    
    I1 = IdentityOperator(V,V)
    I2 = IdentityOperator(V,V)
    I3 = IdentityOperator(V,V)
    I4 = IdentityOperator(Vc,Vc)

    I1 *= 0
    I2 *= 1
    I3 *= 3
    v3 = I3.dot(v)
    I4 *= 3j
    v4 = I4.dot(vc)

    assert np.array_equal(v.toarray(), v_array)
    assert isinstance(I1, ZeroOperator)
    assert isinstance(I2, IdentityOperator)
    assert isinstance(I3, ScaledLinearOperator)
    assert np.array_equal(v3.toarray(), np.dot(v_array, 3))
    assert isinstance(I4, ScaledLinearOperator)
    assert np.array_equal(v4.toarray(), np.dot(v_array, 3j))

    # testing __iadd__ and __isub__ although not explicitly implemented (in the LinearOperator class)

    nonzero_values1 = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values1[k1,k2] = 1 + k1*n2 + k2
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1==0:
                if k2<0:
                    nonzero_values1[k1,k2] = nonzero_values1[-k1,-k2]
            elif k1<0:
                nonzero_values1[k1,k2] = nonzero_values1[-k1,-k2]

    S = StencilMatrix(V,V)
    Z1 = ZeroOperator(V,V)
    Z2 = ZeroOperator(V,V)
    Z3 = Z1.copy()

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S[:,:,k1,k2] = nonzero_values1[k1,k2]
    S.remove_spurious_entries()
    T = S.copy()
    Sa = S.toarray()

    Z1 += S
    S += Z2

    assert isinstance(Z1, StencilMatrix)
    assert isinstance(S, StencilMatrix)

    S += Z1

    w = S.dot(v)

    assert isinstance(S, StencilMatrix)
    assert np.array_equal(w.toarray(), np.dot(np.dot(2, Sa), v_array))

    Z3 -= T
    T -= Z2
    T -= S+3*Z3

    w2 = T.dot(v)

    assert isinstance(Z3, StencilMatrix)
    assert isinstance(T, StencilMatrix)
    assert np.array_equal(w2.toarray(), np.dot(np.dot(2, Sa), v_array))
 
#===============================================================================
@pytest.mark.parametrize('n1', n1array)
@pytest.mark.parametrize('n2', n2array)
@pytest.mark.parametrize('p1', p1array)
@pytest.mark.parametrize('p2', p2array)

def test_inverse_transpose_interaction(n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs: S (V->V, StencilMatrix), S1 (W->W, StencilMatrix)
        #Initiate BlockLO: B (VxW -> VxW) and a BlockVector u element of VxW
    # 2. For both B and S, check whether all possible combinations of the transpose and the inverse behave as expected

    # Initiate StencilVectorSpace
    V  = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    V2 = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    W  = get_StencilVectorSpace([n1+2, n2], [p1, p2+1], [P1, P2])
    
    # Initiate positive definite StencilMatrices for which the cg inverse works (necessary for certain tests)
    S = StencilMatrix(V, V)
    S1 = StencilMatrix(W, W)
    S2 = StencilMatrix(V, V2)

    # Initiate StencilVectors 
    v = StencilVector(V)
    for i in range(n1):
        for j in range(n2):
            v[i,j] = 1
    v2 = StencilVector(V2)
    for i in range(n1):
        for j in range(n2):
            v2[i,j] = 1 
    w = StencilVector(W)
    for i in range(n1+2):
        for j in range(n2):
            w[i,j] = 1   
    
    # Fill the matrices S and S1 (both upper triangular)
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
    
    nonzero_values1 = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2-1,p2+2):
            nonzero_values1[k1,k2] = 1 + k1*n2 + k2
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2-1,p2+2):
            if k1==0:
                if k2<0:
                    nonzero_values1[k1,k2] = nonzero_values1[-k1,-k2]
            elif k1<0:
                nonzero_values1[k1,k2] = nonzero_values1[-k1,-k2]

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S[:,:,k1,k2] = nonzero_values[k1,k2]
            S2[:,:,k1,k2] = nonzero_values[k1,k2]
    S.remove_spurious_entries()
    S2.remove_spurious_entries()

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2-1,p2+2):
            S1[:,:,k1,k2] = nonzero_values1[k1,k2]
    S1.remove_spurious_entries()

    # Initiate a BlockVectorSpace, a BlockLO and a BlockVector
    U = BlockVectorSpace(V, W)
    B = BlockLinearOperator(U, U, ((S, None), (None, S1)))
    u = BlockVector(U, (v,w))

    ###
    ### Test whether pre-allocated storage in InverseLinearoperator subclasses belong to the right space.
    ### Not working so far as algorithms implicitely assume domain == codomain.
    ###

    #S2_inv_pcg = inverse(S2, 'pcg', pc='jacobi', tol=1e-9)
    #S2_inv_lsmr = inverse(S2, 'lsmr', tol=1e-9)
    #x_pcg = S2_inv_pcg @ v2
    #x_lsmr = S2_inv_lsmr @ v2
    #assert isinstance(x_pcg, V)
    #assert isinstance(x_lsmr, V)

    ###
    ### BlockLO Transpose - Inverse Tests
    ### -1,T & T,-1 --- -1,T,T --- -1,T,-1 --- T,-1,-1 --- T,-1,T (the combinations I test)
    ###

    # Square root test
    scaled_matrix = B * np.random.random() # Ensure the diagonal elements != 1
    diagonal_values = scaled_matrix.diagonal(sqrt=False).toarray()
    sqrt_diagonal_values = scaled_matrix.diagonal(sqrt=True).toarray()
    assert np.array_equal(sqrt_diagonal_values, np.sqrt(diagonal_values))

    tol = 1e-5
    C = inverse(B, 'cg', tol=tol)
    P = B.diagonal(inverse=True)

    B_T = B.T
    C_T = C.T

    # -1,T & T,-1 -> equal
    assert isinstance(C_T, ConjugateGradient)
    assert isinstance(inverse(B_T, 'cg', tol=tol), ConjugateGradient)
    diff = C_T @ u - inverse(B_T, 'cg', tol=tol) @ u
    assert diff.inner(diff) == 0

    # -1,T,T -> equal -1
    diff = C_T.T @ u - C @ u
    assert diff.inner(diff) == 0

    # -1,T,-1 -> equal T
    assert isinstance(inverse(C_T, 'bicg'), BlockLinearOperator)
    diff = inverse(C_T, 'bicg') @ u - B_T @ u
    assert diff.inner(diff) == 0

    # T,-1,-1 -> equal T
    assert isinstance(inverse(inverse(B_T, 'cg', tol=tol), 'cg', pc=P), BlockLinearOperator)
    diff = inverse(inverse(B_T, 'cg', tol=tol), 'cg', pc=P) @ u - B_T @ u
    assert diff.inner(diff) == 0

    # T,-1,T -> equal -1
    assert isinstance(inverse(B_T, 'cg', tol=tol).T, ConjugateGradient)
    diff = inverse(B_T, 'cg', tol=tol) @ u - C @ u
    assert diff.inner(diff) == 0

    ###
    ### StencilMatrix Transpose - Inverse Tests
    ### -1,T & T,-1 --- -1,T,T --- -1,T,-1 --- T,-1,-1 --- T,-1,T (the combinations I test)
    ###

    tol = 1e-5
    C = inverse(S, 'cg', tol=tol)
    P = S.diagonal(inverse=True)

    S_T = S.T
    C_T = C.T

    # -1,T & T,-1 -> equal
    assert isinstance(C_T, ConjugateGradient)
    assert isinstance(inverse(S_T, 'cg', tol=tol), ConjugateGradient)
    diff = C_T @ v - inverse(S_T, 'cg', tol=tol) @ v
    assert diff.inner(diff) == 0

    # -1,T,T -> equal -1
    diff = C_T.T @ v - C @ v
    assert diff.inner(diff) == 0

    # -1,T,-1 -> equal T
    assert isinstance(inverse(C_T, 'bicg'), StencilMatrix)
    diff = inverse(C_T, 'bicg') @ v - S_T @ v
    assert diff.inner(diff) == 0

    # T,-1,-1 -> equal T
    assert isinstance(inverse(inverse(S_T, 'cg', tol=tol), 'cg', pc=P), StencilMatrix)
    diff = inverse(inverse(S_T, 'cg', tol=tol), 'cg', pc=P) @ v - S_T @ v
    assert diff.inner(diff) == 0

    # T,-1,T -> equal -1
    assert isinstance(inverse(S_T, 'cg', tol=tol).T, ConjugateGradient)
    diff = inverse(S_T, 'cg', tol=tol) @ v - C @ v
    assert diff.inner(diff) == 0

#===============================================================================
@pytest.mark.parametrize('n1', [3, 5])
@pytest.mark.parametrize('n2', [4, 7])
@pytest.mark.parametrize('p1', [2, 6])
@pytest.mark.parametrize('p2', [3, 9])

def test_positive_definite_matrix(n1, n2, p1, p2):
    P1 = False
    P2 = False
    V = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    S = get_positive_definite_StencilMatrix(V)

    assert_pos_def(S)

#===============================================================================
@pytest.mark.parametrize('n1', [3, 5])
@pytest.mark.parametrize('n2', [4, 7])
@pytest.mark.parametrize('p1', [2, 6])
@pytest.mark.parametrize('p2', [3, 9])

def test_operator_evaluation(n1, n2, p1, p2):

    # 1. Initiate StencilVectorSpace V, pos. def. Stencil Matrix S and StencilVector v = (1,1,1,1)
    #    Initiate a BlockVectorSpace U = VxV, a BlockLO B = [[V, None], [None, V]] and a BlockVector u = (v,v)
    #    as well as 2 "uncompatible" LOs: Z = ZeroO(U,U), I = IdentityO(V,V)
    #    Further create the conjugate gradient InverseLOs of S and B, S_ILO and B_ILO
    # 2.1 PowerLO test 
    #       Test B**(0,1,2), B_ILO**(0,1,2), Z**(0,1,2)
    #            S**(0,1,2), S_ILO**(0,1,2), I**(0,1,2)
    # 2.2 SumLO test
    #       Test B + B_ILO + B + B_ILO
    #            S + S_ILO + S + S_ILO
    # 2.3 CompositionLO test
    #       Test B @ (-B) = -B**2
    #            S @ (-S) = -S**2
    # 2.4 Huge Composition
    #       H1 = S_ILO . T, testing inverse transpose interaction
    #       H2 = (S_ILO) ⁻¹, testing inverse inverse interaction
    #       H3 = (2 * S_ILO) @ (S**2), testing composition of container classes
    #       H4 = 2 * ( S¹ @ S⁰ ), testing power 1 and 0, composition with Identity, scaling of container class
    #       H5 = ZV @ I, ZV a ZeroO(V,V), testing composition with a ZeroO
    #       H = H1 @ ( H2 + H3 - H4 + H5 ) . T, testing all together
    # 2.5 InverseLO test (explicit)

    ###
    ### 1.
    ###

    P1 = False
    P2 = False

    # Initiate StencilVectorSpace V
    V = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    
    # Initiate positive definite StencilMatrices for which the cg inverse works (necessary for certain tests)
    S = get_positive_definite_StencilMatrix(V)

    # Initiate StencilVectors 
    v = StencilVector(V)
    for i in range(n1):
        for j in range(n2):
            v[i,j] = 1   
    
    # Initiate a BlockVectorSpace, a BlockLO and a BlockVector
    U = BlockVectorSpace(V, V)
    B = BlockLinearOperator(U, U, ((S, None), (None, S)))
    Z = ZeroOperator(U, U)
    I = IdentityOperator(V, V)
    u = BlockVector(U, (v,v))
    tol = 1e-6
    S_ILO = inverse(S, 'cg', tol=tol)
    B_ILO = inverse(B, 'cg', tol=tol)

    ###
    ### 2.
    ###

    ### 2.1 PowerLO test
    Bmat = B.toarray()
    assert_pos_def(B)
    uarr = u.toarray()
    b0 = ( B**0 @ u ).toarray()
    b1 = ( B**1 @ u ).toarray()
    b2 = ( B**2 @ u ).toarray()
    assert np.array_equal(uarr, b0)
    assert np.linalg.norm( np.dot(Bmat, uarr) - b1 ) < 1e-10
    assert np.linalg.norm( np.dot(Bmat, np.dot(Bmat, uarr)) - b2 ) < 1e-10

    bi0 = ( B_ILO**0 @ u ).toarray()
    bi1 = ( B_ILO**1 @ u ).toarray()
    bi2 = ( B_ILO**2 @ u ).toarray()
    B_inv_mat = np.linalg.inv(Bmat)
    b_inv_arr = np.matrix.flatten(B_inv_mat)
    error_est = 2 + n1 * n2 * np.max( [ np.abs(b_inv_arr[i]) for i in range(len(b_inv_arr)) ] )
    assert np.array_equal(uarr, bi0)
    bi12 = np.linalg.solve(Bmat, uarr)
    bi22 = np.linalg.solve(Bmat, bi12)
    assert np.linalg.norm( (Bmat @ bi12) - uarr ) < tol
    assert np.linalg.norm( (Bmat @ bi22) - bi12 ) < error_est * tol

    zeros = U.zeros().toarray()
    z0 = ( Z**0 @ u ).toarray()
    z1 = ( Z**1 @ u ).toarray()
    z2 = ( Z**2 @ u ).toarray()
    assert np.array_equal(uarr, z0)
    assert np.array_equal(zeros, z1)
    assert np.array_equal(zeros, z2)

    Smat = S.toarray()
    assert_pos_def(S)
    varr = v.toarray()
    s0 = ( S**0 @ v ).toarray()
    s1 = ( S**1 @ v ).toarray()
    s2 = ( S**2 @ v ).toarray()
    assert np.array_equal(varr, s0)
    assert np.linalg.norm( np.dot(Smat, varr) - s1 ) < 1e-10
    assert np.linalg.norm( np.dot(Smat, np.dot(Smat, varr)) - s2 ) < 1e-10

    si0 = ( S_ILO**0 @ v ).toarray()
    si1 = ( S_ILO**1 @ v ).toarray()
    si2 = ( S_ILO**2 @ v ).toarray()
    S_inv_mat = np.linalg.inv(Smat)
    s_inv_arr = np.matrix.flatten(S_inv_mat)
    error_est = 2 + n1 * n2 * np.max( [ np.abs(s_inv_arr[i]) for i in range(len(s_inv_arr)) ] )
    assert np.array_equal(varr, si0)
    si12 = np.linalg.solve(Smat, varr)
    si22 = np.linalg.solve(Smat, si12)
    assert np.linalg.norm( (Smat @ si12) - varr ) < tol
    assert np.linalg.norm( (Smat @ si22) - si12 ) < error_est * tol

    i0 = ( I**0 @ v ).toarray()
    i1 = ( I**1 @ v ).toarray()
    i2 = ( I**2 @ v ).toarray()
    assert np.array_equal(varr, i0)
    assert np.array_equal(varr, i1)
    assert np.array_equal(varr, i2)

    ### 2.2 SumLO tests
    Sum1 = B + B_ILO + B + B_ILO
    Sum2 = S + S_ILO + S + S_ILO
    sum1 = Sum1 @ u
    sum2 = Sum2 @ v
    u_approx = B @ (0.5*(sum1 - 2*B@u))
    v_approx = S @ (0.5*(sum2 - 2*S@v))
    assert np.linalg.norm( (u_approx - u).toarray() ) < tol
    assert np.linalg.norm( (v_approx - v).toarray() ) < tol

    ### 2.3 CompLO tests
    C1 = B @ (-B)
    C2 = S @ (-S)
    c1 = ( C1 @ u ).toarray()
    c2 = ( C2 @ v ).toarray()
    assert np.array_equal(-c1, b2)
    assert np.array_equal(-c2, s2)

    ### 2.4 Huge composition
    ZV = ZeroOperator(V, V)
    H1 = S_ILO.T
    H2 = inverse(S_ILO, 'bicg', tol=tol)
    H3 = (2 * S_ILO) @ S**2
    H4 = 2 * (S**1 @ S**0)
    H5 = ZV @ I
    H = H1 @ ( H2 + H3 - H4 + H5 ).T
    assert np.linalg.norm( (H @ v).toarray() - v.toarray() ) < 10 * tol

    ### 2.5 InverseLO test

    S_cg = inverse(S, 'cg', tol=tol)
    B_cg = inverse(B, 'cg', tol=tol)
    S_pcg = inverse(S, 'cg', pc=S.diagonal(inverse=True), tol=tol)
    B_pcg = inverse(B, 'cg', pc=B.diagonal(inverse=True), tol=tol)
    S_bicg = inverse(S, 'bicg', tol=tol)
    B_bicg = inverse(B, 'bicg', tol=tol)
    S_lsmr = inverse(S, 'lsmr', tol=tol)
    B_lsmr = inverse(B, 'lsmr', tol=tol)
    S_mr = inverse(S, 'minres', tol=tol)
    B_mr = inverse(B, 'minres', tol=tol)

    xs_cg = S_cg @ v
    xs_pcg = S_pcg @ v
    xs_bicg = S_bicg @ v
    xs_lsmr = S_lsmr @ v
    xs_mr = S_mr @ v

    xb_cg = B_cg @ u
    xb_pcg = B_pcg @ u
    xb_bicg = B_bicg @ u
    xb_lsmr = B_lsmr @ u
    xb_mr = B_mr @ u

    # Several break-criteria in the LSMR algorithm require different way to determine success
    # than asserting rnorm < tol, as that is not required. Even though it should?

    assert np.linalg.norm( (S @ xs_cg - v).toarray() ) < tol
    assert np.linalg.norm( (S @ xs_pcg - v).toarray() ) < tol
    assert np.linalg.norm( (S @ xs_bicg - v).toarray() ) < tol
    assert S_lsmr.get_success() == True
    assert np.linalg.norm( (S @ xs_mr - v).toarray() ) < tol

    assert np.linalg.norm( (B @ xb_cg - u).toarray() ) < tol
    assert np.linalg.norm( (B @ xb_pcg - u).toarray() ) < tol
    assert np.linalg.norm( (B @ xb_bicg - u).toarray() ) < tol
    assert B_lsmr.get_success() == True
    assert np.linalg.norm( (B @ xb_mr - u).toarray() ) < tol

#===============================================================================

def test_internal_storage():

    # Create LinearOperator Z = A @ A.T @ A @ A.T @ A, where the domain and codomain of A are of different dimension.
    # Prior to a fix, operator would not have enough preallocated storage defined.

    n1=2
    n2=1
    p1=1
    p2=1
    P1=False
    P2=False
    V = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    U1 = BlockVectorSpace(V, V)
    U2 = BlockVectorSpace(V, V, V)

    x1 = StencilVector(V)
    x1[0] = 1
    x1[1] = 1
    x = BlockVector(U1, (x1, x1))
    xx = BlockVector(U2, (x1, x1, x1))

    A1 = StencilMatrix(V, V)
    A1[0, 0, 0, 0] = 1
    A1[1, 0, 0, 0] = 1
    A = BlockLinearOperator(U1, U2, ((A1, A1), (A1, A1), (A1, A1)))
    B = A.T
    C = A
    D = A.T

    Z1_1 = A @ (B @ C)
    Z1_2 = (A @ B) @ C
    Z1_3 = A @ B @ C
    y1_1 = Z1_1 @ x
    y1_2 = Z1_2 @ x
    y1_3 = Z1_3 @ x

    Z2_1 = (A @ B) @ (C @ D)
    Z2_2 = (A @ B @ C) @ D
    Z2_3 = A @ (B @ C @ D)
    y2_1 = Z2_1 @ xx
    y2_2 = Z2_2 @ xx
    y2_3 = Z2_3 @ xx

    assert len(Z1_1.tmp_vectors) == 2
    assert len(Z1_2.tmp_vectors) == 2
    assert len(Z1_3.tmp_vectors) == 2
    assert len(Z2_1.tmp_vectors) == 3
    assert len(Z2_2.tmp_vectors) == 3
    assert len(Z2_3.tmp_vectors) == 3
    assert np.array_equal( y1_1.toarray(), y1_2.toarray() ) & np.array_equal( y1_2.toarray(), y1_3.toarray() )
    assert np.array_equal( y2_1.toarray(), y2_2.toarray() ) & np.array_equal( y2_2.toarray(), y2_3.toarray() )

#===============================================================================
@pytest.mark.parametrize(("solver", "use_jacobi_pc"),
    [('CG'      , False), ('CG', True),
     ('BiCG'    , False),
     ('BiCGSTAB', False), ('BiCGSTAB', True),
     ('MINRES'  , False),
     ('LSMR'    , False),
     ('GMRES'   , False)]
 )
def test_x0update(solver, use_jacobi_pc):
    n1 = 4
    n2 = 3
    p1 = 5
    p2 = 2
    P1 = False
    P2 = False
    V = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    A = get_positive_definite_StencilMatrix(V)
    assert_pos_def(A)
    b = StencilVector(V)
    for n in range(n1):
        b[n, :] = 1.
    assert np.array_equal(b.toarray(), np.ones(n1*n2, dtype=float))

    # Create Inverse
    tol = 1e-6
    pc = A.diagonal(inverse=True) if use_jacobi_pc else None
    A_inv = inverse(A, solver, pc=pc, tol=tol)

    # Check whether x0 is not None
    x0_init = A_inv.get_options("x0")
    assert x0_init is not None

    # Apply inverse and check x0
    x = A_inv @ b
    x0_new1 = A_inv.get_options("x0")
    assert x0_new1 is x0_init

    # Change x0, apply A_inv and check for x0
    A_inv.set_options(x0 = b)
    assert A_inv.get_options("x0") is b

    x = A_inv @ b
    assert A_inv.get_options("x0") is b

    # Apply inverse using out=x0 and check for updated x0
    x = A_inv.dot(b, out=b)
    assert A_inv.get_options('x0') is x

#===============================================================================
def test_dot_inner():

    n1, n2 = 4, 7
    p1, p2 = 2, 3
    P1, P2 = False, False

    V = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    M = get_positive_definite_StencilMatrix(V)
    N = get_positive_definite_StencilMatrix(V)

    U1 = BlockVectorSpace(V, V)
    U2 = BlockVectorSpace(V, V, V)
    A  = BlockLinearOperator(U1, U2, ((M, None),
                                      (M,    N),
                                      (None, N)))

    b = A.domain.zeros()
    c = A.codomain.zeros()

    # Set the values of b and c randomly from a uniform distribution over the
    # interval [0, 1)
    rng = np.random.default_rng(seed=42)
    for bj in b:
        Vj = bj.space
        rng.random(size=Vj.shape, dtype=Vj.dtype, out=bj._data)
    for ci in c:
        Vi = ci.space
        rng.random(size=Vi.shape, dtype=Vi.dtype, out=ci._data)

    # Create a work vector for the dot product, needed to compare results
    work_vec = A.codomain.zeros()

    # Result of dot product is a temporary vector, which is allocated and then
    # discarded. This is the default behavior of the dot method.
    r0 = A.dot(b).inner(c)

    # Result of dot product is stored in work_vec and used in the next line
    A.dot(b, out=work_vec)
    r1 = work_vec.inner(c)

    # Result of dot product is stored in work_vec and used in the same line
    r2 = A.dot(b, out=work_vec).inner(c)

    # Calling the dot_inner method, which uses an internal work vector to store
    # the result of the dot product, and then uses it for the inner product.
    r3 = A.dot_inner(b, c)

    # Check if the results are equal
    assert r0 == r1
    assert r0 == r2
    assert r0 == r3

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
