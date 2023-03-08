import pytest
import numpy as np

from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.basic import ZeroOperator, IdentityOperator, ComposedLinearOperator, SumLinearOperator, PowerLinearOperator, InverseLinearOperator, ScaledLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.solvers import BiConjugateGradient, ConjugateGradient, PConjugateGradient, inverse
from psydac.ddm.cart         import DomainDecomposition, CartDecomposition
#===============================================================================

# 11.12.22: Check for redundancy(?) in pytests 1,2 and 4, document 1,2

n1array = [2, 7]
n2array = [2, 3]
p1array = [1, 3]
p2array = [1, 3]

def is_pos_def(x):
    assert np.all(np.linalg.eigvals(x) > 0)

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
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def test_square_stencil_basic(n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs S,S1 (StencilMatrix), I (IdentityOperator), Z (ZeroOperator) and a Stencilvector v
    # 2. Test general basic operations
    # 3. Test special cases

    ###
    ### 1. Initiation
    ###

    # Initiate StencilVectorSpace
    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    V = StencilVectorSpace( C )
    
    # Initiate Linear Operators
    Z = ZeroOperator(V, V)
    I = IdentityOperator(V, V)
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
    assert np.all(v.toarray() == np.ones(n1*n2))

    assert Sa.shape == S.shape
    assert np.all( Sa == A1 )
    assert S1a.shape == S1.shape
    assert np.all( S1a == A2 )

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
    assert np.all((-S).dot(v).toarray() == -( S.dot(v).toarray() ))

    ## ___Multiplication, Composition, Raising to a Power___

    # Multiplying and Dividing a StencilMatrix by a scalar returns a StencilMatrix
    assert isinstance(np.pi*S, StencilMatrix)
    assert isinstance(S*np.pi, StencilMatrix)
    assert isinstance(S/np.pi, StencilMatrix)

    # Composing StencilMatrices works
    assert isinstance(S@S1, ComposedLinearOperator)

    # Raising a StencilMatrix to a power works
    assert isinstance(S**3, PowerLinearOperator)

    ## ___Transposing___
    
    assert not np.all(S2a == np.transpose(S2a))
    assert isinstance(S2.T, StencilMatrix)
    assert np.all(S2.T.toarray() == np.transpose(S2a)) # using a nonsymmetric matrix throughout
    assert np.all(S2.T.T.toarray() == S2a)

    ###
    ### 3. Test special cases
    ### Add. und Sub. with ZeroO's, Composition with Zero- and IdentityO's, Raising to the power of 0 and 1
    ###

    ## ___Addition and Substraction with ZeroOperators___

    # Adding a ZeroOperator does not change the StencilMatrix
    assert S+Z == S
    assert Z+S == S

    # Substracting a ZeroOperator and substracting from a ZeroOperator work as intended
    assert S-Z == S
    assert np.all((Z-S).toarray() == (-S).toarray())

    ## ___Composing with Zero- and IdentityOperators___   

    # Composing a StencilMatrix with a ZeroOperator returns a ZeroOperator
    assert isinstance(S@Z, ZeroOperator)
    assert isinstance(Z@S, ZeroOperator)

    # Composing a StencilMatrix with the IdentityOperator does not change the object
    assert S@I == S
    assert I@S == S

    ## ___Raising to the power of 0 and 1___
    
    # Raising a StencilMatrix to the power of 1 or 0 does not change the object / returns an IdentityOperator
    assert S**1 == S
    assert isinstance(S**0, IdentityOperator)  

#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def test_square_block_basic(n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs S,S1 (StencilMatrix), Z (ZeroOperator) and a Stencilvector v
    #    Initiate square LOs B,B1 (BlockLO), BZ (ZeroOperator), BI (IdentityOperator) and a BlockVector vb
    # 2. Test general basic operations
    # 3. Test special cases

    # Initiate StencilVectorSpace
    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    V = StencilVectorSpace( C )
       
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
    B = BlockLinearOperator(Vb, Vb, ((S, None), (None, S)))
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
    # Update 21.12.: ZeroLOs and IdentityLOs from and/or to BlockVectorSpaces are now BlockLOs
    # thus the sums/differences below should be BlockLOs again
    assert isinstance(B + BI, BlockLinearOperator)
    assert isinstance(BI + B, BlockLinearOperator)
    assert isinstance(B - BI, BlockLinearOperator)
    assert isinstance(BI - B, BlockLinearOperator)

    # Negating a BlockLO works as intended
    assert isinstance(-B, BlockLinearOperator)
    assert np.all((-B).dot(vb).toarray() == -( B.dot(vb).toarray() ))

    ## ___Multiplication, Composition, Raising to a Power___

    # Multiplying and Dividing a BlockLO by a scalar returns a BlockLO
    assert isinstance(np.pi*B, BlockLinearOperator)
    assert isinstance(B*np.pi, BlockLinearOperator)
    assert isinstance(B/np.pi, BlockLinearOperator)

    # Composing BlockLOs works
    assert isinstance(B@B1, ComposedLinearOperator)

    # Raising a BlockLO to a power works
    assert isinstance(B**3, PowerLinearOperator)

    ## ___Transposing___
    assert not np.all(B2.toarray() == np.transpose(B2.toarray()))
    assert isinstance(B2.T, BlockLinearOperator)
    assert np.all(B2.T.toarray() == np.transpose(B2.toarray())) # using a nonsymmetric matrix throughout
    assert np.all(B2.T.T.toarray() == B2.toarray())

    ###
    ### 3. Test special cases
    ### Add. und Sub. with ZeroO's, Composition with Zero- and IdentityO's, Raising to the power of 0 and 1
    ###

    ## ___Addition and Substraction with ZeroOperators___

    # Adding a ZeroOperator does not change the BlockLO
    # Update 21.12.: ZeroLOs and IdentityLOs from and/or to BlockVectorSpaces are now BlockLOs
    # thus while B+BZ = B is still true, B+BZ is a new object now.
    BBZ = B+BZ
    BZB = BZ+B
    for i in range(2):
        for j in range(2):
            if isinstance(BBZ.blocks[i][j], ZeroOperator):
                assert isinstance(B.blocks[i][j], ZeroOperator) or B.blocks[i][j] == None
            else:
                assert BBZ.blocks[i][j] == B.blocks[i][j]
            if isinstance(BZB.blocks[i][j], ZeroOperator):
                assert isinstance(B.blocks[i][j], ZeroOperator) or B.blocks[i][j] == None
            else:
                assert BZB.blocks[i][j] == B.blocks[i][j]
    #assert BBZ.blocks == B.blocks
    #assert BZ+B == B

    # Substracting a ZeroOperator and substracting from a ZeroOperator work as intended
    BmBZ = B-BZ
    for i in range(2):
        for j in range(2):
            if isinstance(BmBZ.blocks[i][j], ZeroOperator):
                assert isinstance(B.blocks[i][j], ZeroOperator) or B.blocks[i][j] == None
            else:
                assert BmBZ.blocks[i][j] == B.blocks[i][j]
    #assert B-BZ == B
    assert np.all((BZ-B).toarray() == (-B).toarray()) # replaces assert BZ-B == -B

    ## ___Composing with Zero- and IdentityOperators___ 

    # Composing a BlockLO with a ZeroOperator returns a ZeroOperator
    # Update 21.12.: ZeroLOs and IdentityLOs from and/or to BlockVectorSpaces are now BlockLOs
    # thus B@BZ is now a ComposedLO.
    assert isinstance(B@BZ, ComposedLinearOperator)
    assert isinstance(BZ@B, ComposedLinearOperator)

    # Composing a BlockLO with the IdentityOperator does not change the object
    # due to the 21.12. change not valid anymore
    #assert B@BI == B
    #assert BI@B == B
    # but: 
    assert np.all(((B@BI)@vb).toarray() == (B@vb).toarray())
    assert np.all(((BI@B)@vb).toarray() == (B@vb).toarray())

    ## ___Raising to the power of 0 and 1___

    # Raising a BlockLO to the power of 1 or 0 does not change the object / returns an IdentityOperator
    # 21.12. change: B**0 a BlockLO with IdentityLOs at the diagonal
    assert B**1 == B
    #assert isinstance(B**0, IdentityOperator)
    assert isinstance(B**0, BlockLinearOperator)
    assert np.all(((B**0)@vb).toarray() == vb.toarray())

#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def test_in_place_operations(n1, n2, p1, p2, P1=False, P2=False):

    # testing __imul__ although not explicitly implemented (in the LinearOperator class)

    # Initiate StencilVectorSpace
    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    V = StencilVectorSpace( C )

    v = StencilVector(V)
    v_array = np.zeros(n1*n2)

    for i in range(n1):
        for j in range(n2):
            v[i,j] = i+1
            v_array[i*n2+j] = i+1
    
    I1 = IdentityOperator(V,V)
    I2 = IdentityOperator(V,V)
    I3 = IdentityOperator(V,V)

    I1 *= 0
    I2 *= 1
    I3 *= 3
    v3 = I3.dot(v)

    assert np.all(v.toarray() == v_array)
    assert isinstance(I1, ZeroOperator)
    assert isinstance(I2, IdentityOperator)
    assert isinstance(I3, ScaledLinearOperator)
    assert np.all(v3.toarray() == np.dot(v_array, 3))

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
    assert np.all(w.toarray() == np.dot(np.dot(2, Sa), v_array))

    Z3 -= T
    T -= Z2
    T -= S+3*Z3

    w2 = T.dot(v)

    assert isinstance(Z3, StencilMatrix)
    assert isinstance(T, StencilMatrix)
    assert np.all(w2.toarray() == np.dot(np.dot(2, Sa), v_array))
    
#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def test_inverse_transpose_interaction(n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs: S (V->V, StencilMatrix), S1 (W->W, StencilMatrix)
        #Initiate BlockLO: B (VxW -> VxW) and a BlockVector u element of VxW
    # 2. For both B and S, check whether all possible combinations of the transpose and the inverse behave as expected

    # Initiate StencilVectorSpace
    D = DomainDecomposition([n1,n2], periods=[P1,P2])

    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])

    V = StencilVectorSpace( C )
    D = DomainDecomposition([n1+2,n2], periods=[P1,P2])

    npts = [n1+2,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2+1], shifts=[1,1])
    W = StencilVectorSpace( C )
    
    # Initiate positive definite StencilMatrices for which the cg inverse works (necessary for certain tests)
    S = StencilMatrix(V,V)
    S1 = StencilMatrix(W,W)

    # Initiate StencilVectors 
    v = StencilVector(V)
    for i in range(n1):
        for j in range(n2):
            v[i,j] = 1   
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
    S.remove_spurious_entries()

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2-1,p2+2):
            S1[:,:,k1,k2] = nonzero_values1[k1,k2]
    S1.remove_spurious_entries()

    # Initiate a BlockVectorSpace, a BlockLO and a BlockVector
    U = BlockVectorSpace(V, W)
    B = BlockLinearOperator(U, U, ((S, None), (None, S1)))
    u = BlockVector(U, (v,w))

    ###
    ### BlockLO Transpose - Inverse Tests
    ### -1,T & T,-1 --- -1,T,T --- -1,T,-1 --- T,-1,-1 --- T,-1,T (the combinations I test)
    ###

    tol = 1e-5
    C = inverse(B, 'cg', tol=tol)

    # -1,T & T,-1 -> equal
    assert isinstance(C.T, ConjugateGradient)
    assert isinstance(inverse(B.T, 'cg', tol=tol), ConjugateGradient)
    assert np.sqrt(sum(((C.T.dot(u) - inverse(B.T, 'cg', tol=tol).dot(u)).toarray())**2)) < 2*tol
    # -1,T,T -> equal -1
    assert np.sqrt(sum(((C.T.T.dot(u) - C.dot(u)).toarray())**2)) < 2*tol
    # -1,T,-1 -> equal T
    assert isinstance(inverse(C.T, 'bicg'), BlockLinearOperator)
    assert np.all(inverse(C.T, 'bicg').dot(u).toarray() == B.T.dot(u).toarray())
    # T,-1,-1 -> equal T
    assert isinstance(inverse(inverse(B.T, 'cg', tol=tol), 'pcg', pc='jacobi'), BlockLinearOperator)
    assert np.all(inverse(inverse(B.T, 'cg', tol=tol), 'pcg', pc='jacobi').dot(u).toarray() == B.T.dot(u).toarray())
    # T,-1,T -> equal -1
    assert isinstance(inverse(B.T, 'cg', tol=tol).T, ConjugateGradient)
    assert np.sqrt(sum(((inverse(B.T, 'cg', tol=tol).dot(u) - C.dot(u)).toarray())**2)) < tol

    ###
    ### StencilMatrix Transpose - Inverse Tests
    ### -1,T & T,-1 --- -1,T,T --- -1,T,-1 --- T,-1,-1 --- T,-1,T (the combinations I test)
    ###

    tol = 1e-5
    C = inverse(S, 'cg', tol=tol)

    # -1,T & T,-1 -> equal
    assert isinstance(C.T, ConjugateGradient)
    assert isinstance(inverse(S.T, 'cg', tol=tol), ConjugateGradient)
    assert np.sqrt(sum(((C.T.dot(v) - inverse(S.T, 'cg', tol=tol).dot(v)).toarray())**2)) < 2*tol
    # -1,T,T -> equal -1
    assert np.sqrt(sum(((C.T.T.dot(v) - C.dot(v)).toarray())**2)) < 2*tol
    # -1,T,-1 -> equal T
    assert isinstance(inverse(C.T, 'bicg'), StencilMatrix)
    assert np.all(inverse(C.T, 'bicg').dot(v).toarray() == S.T.dot(v).toarray())
    # T,-1,-1 -> equal T
    assert isinstance(inverse(inverse(S.T, 'cg', tol=tol), 'pcg', pc='jacobi'), StencilMatrix)
    assert np.all(inverse(inverse(S.T, 'cg', tol=tol), 'pcg', pc='jacobi').dot(v).toarray() == S.T.dot(v).toarray())
    # T,-1,T -> equal -1
    assert isinstance(inverse(S.T, 'cg', tol=tol).T, ConjugateGradient)
    assert np.sqrt(sum(((inverse(S.T, 'cg', tol=tol).dot(v) - C.dot(v)).toarray())**2)) < tol

#===============================================================================
# 'cg' inverse requires a positive definite matrix.
# I did not yet come up with a way to create positive definite matrices for arbitrary n1, n2, p1, p2, P1, P2
# Thus until changed: Only one test with a simple 4x4 positive definite matrix

def test_operator_evaluation():#n1, n2, p1, p2, P1=False, P2=False):

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

    ###
    ### 1.
    ###

    # See comment above method regarding this explicit definition
    n1 = 2
    n2 = 2
    p1 = 1
    p2 = 1
    P1 = False
    P2 = False

    nonzero_values = dict()
    nonzero_values[0, -1] = 1
    nonzero_values[0, 0] = 2
    nonzero_values[0, 1] = 1
    nonzero_values[-1, -1] = 0
    nonzero_values[-1, 0] = 0
    nonzero_values[-1, 1] = 0
    nonzero_values[1, -1] = 0
    nonzero_values[1, 0] = 0
    nonzero_values[1, 1] = 0

    # Initiate StencilVectorSpace V
    D = DomainDecomposition([n1,n2], periods=[P1,P2])
    npts = [n1,n2]
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    C = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1,p2], shifts=[1,1])
    V = StencilVectorSpace( C )
    
    # Initiate positive definite StencilMatrices for which the cg inverse works (necessary for certain tests)
    S = StencilMatrix(V,V)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S[:,:,k1,k2] = nonzero_values[k1,k2]
    S.remove_spurious_entries()

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
    is_pos_def(Bmat)
    uarr = u.toarray()
    b0 = ((B**0)@u).toarray()
    b1 = ((B**1)@u).toarray()
    b2 = ((B**2)@u).toarray()
    assert np.all(uarr == b0)
    assert np.all(np.dot(Bmat, uarr) == b1)
    assert np.all(np.dot(Bmat, np.dot(Bmat, uarr)) == b2)

    bi0 = ((B_ILO**0)@u).toarray()
    bi1 = ((B_ILO**1)@u).toarray()
    bi2 = ((B_ILO**2)@u).toarray()
    B_inv_mat = np.linalg.inv(Bmat)
    b_inv_arr = np.matrix.flatten(B_inv_mat)
    error_est = 2 + n1*n2*np.max([np.abs(b_inv_arr[i]) for i in range(len(b_inv_arr))])
    assert np.all(uarr == bi0)
    assert np.linalg.norm(np.linalg.solve(Bmat, uarr) - bi1, ord=2) < tol
    assert np.linalg.norm(np.linalg.solve(Bmat, np.linalg.solve(Bmat, uarr)) - bi2, ord=2) < error_est * tol

    zeros = U.zeros().toarray()
    z0 = ((Z**0)@u).toarray()
    z1 = ((Z**1)@u).toarray()
    z2 = ((Z**2)@u).toarray()
    assert np.all(uarr == z0)
    assert np.all(zeros == z1)
    assert np.all(zeros == z2)

    Smat = S.toarray()
    is_pos_def(Smat)
    varr = v.toarray()
    s0 = ((S**0)@v).toarray()
    s1 = ((S**1)@v).toarray()
    s2 = ((S**2)@v).toarray()
    assert np.all(varr == s0)
    assert np.all(np.dot(Smat, varr) == s1)
    assert np.all(np.dot(Smat, np.dot(Smat, varr)) == s2)

    si0 = ((S_ILO**0)@v).toarray()
    si1 = ((S_ILO**1)@v).toarray()
    si2 = ((S_ILO**2)@v).toarray()
    S_inv_mat = np.linalg.inv(Smat)
    s_inv_arr = np.matrix.flatten(S_inv_mat)
    error_est = 2 + n1*n2*np.max([np.abs(s_inv_arr[i]) for i in range(len(s_inv_arr))])
    assert np.all(varr == si0)
    assert np.linalg.norm(np.linalg.solve(Smat, varr) - si1, ord=2) < tol
    assert np.linalg.norm(np.linalg.solve(Smat, np.linalg.solve(Smat, varr)) - si2, ord=2) < error_est * tol

    i0 = ((I**0)@v).toarray()
    i1 = ((I**1)@v).toarray()
    i2 = ((I**2)@v).toarray()
    assert np.all(varr == i0)
    assert np.all(varr == i1)
    assert np.all(varr == i2)

    ### 2.2 SumLO tests
    Sum1 = B + B_ILO + B + B_ILO
    Sum2 = S + S_ILO + S + S_ILO
    sum1 = (Sum1@u).toarray()
    sum2 = (Sum2@v).toarray()
    Sum1_mat = 2*Bmat + 2*B_inv_mat
    Sum2_mat = 2*Smat + 2*S_inv_mat
    assert np.linalg.norm(sum1 - np.dot(Sum1_mat, uarr), ord=2) < 2 * tol
    assert np.linalg.norm(sum2 - np.dot(Sum2_mat, varr), ord=2) < 2 * tol

    ### 2.3 CompLO tests
    C1 = B@(-B)
    C2 = S@(-S)
    c1 = (C1@u).toarray()
    c2 = (C2@v).toarray()
    assert np.all(-c1 == b2)
    assert np.all(-c2 == s2)

    ### 2.4 Huge composition
    ZV = ZeroOperator(V, V)
    H1 = S_ILO.T
    H2 = inverse(S_ILO, 'bicg', tol=tol)
    H3 = (2*S_ILO) @ (S**2)
    H4 = 2*( (S**1) @ (S**0) )
    H5 = ZV @ I
    H = H1 @ ( H2 + H3 - H4 + H5 ).T
    assert np.linalg.norm((H@v).toarray() - v.toarray(), ord=2) < 10 * tol

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
