import pytest
import numpy as np

from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.basic import ZeroOperator, IdentityOperator, ComposedLinearOperator, SumLinearOperator, PowerLinearOperator, InverseLinearOperator, ScaledLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.solvers import BiConjugateGradient, ConjugateGradient, PConjugateGradient
#===============================================================================

n1array = [2, 7]
n2array = [2, 3]
p1array = [1, 3]
p2array = [1, 3]

def is_pos_def(x):
    assert np.all(np.linalg.eigvals(x) > 0)

def l2_norm_diff(x,y):
    l = len(x)
    assert l == len(y)
    vec = [(x[i]-y[i])**2 for i in range(l)]
    norm = np.sqrt(sum(vec))
    return norm

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def hitest_square_stencil_basic(n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs S,S1 (StencilMatrix), I (IdentityOperator), Z (ZeroOperator) and a Stencilvector v
    # 2. Test general basic operations
    # 3. Test special cases

    ###
    ### 1. Initiation
    ###

    # Initiate StencilVectorSpace
    V = StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    
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

    #################################

    # Inverting StencilMatrices works
    tol = 1e-6
    M = S.toarray()
    Mi = np.linalg.inv(M)
    realsol = np.dot(Mi,v.toarray())

    ### Conjugate Gradient test
    S_inv1 = S.inverse('cg', tol=tol)
    assert isinstance(S_inv1, ConjugateGradient)
    sol = S_inv1.dot(v).toarray()
    errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    err = sum(errors)
    assert err < tol**2

    ### Preconditioned Conjugate Gradient test, pc = 'jacobi'
    S_inv2 = S.inverse('pcg', pc='jacobi', tol=tol)
    assert isinstance(S_inv2, PConjugateGradient)
    sol = S_inv2.dot(v).toarray()
    errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    err = sum(errors)
    assert err < tol**2
    
    # weighted jacobi doesn't work yet
    S_inv3 = S.inverse('pcg', pc='weighted_jacobi', tol=tol)
    assert isinstance(S_inv3, PConjugateGradient)
    #sol = S_inv3.dot(v)[0].toarray()
    #errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    #err = sum(errors)
    #assert err < tol**2

    ### Biconjugate Gradient test
    S_inv4 = S.inverse('bicg', tol=tol)
    assert isinstance(S_inv4, BiConjugateGradient)
    sol = S_inv4.dot(v).toarray()
    errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    err = sum(errors)
    assert err < tol**2

    # Transposing the inverse of a StencilMatrix works
    T = S_inv4.T
    assert isinstance(T, BiConjugateGradient)
    assert np.all(T.linop.toarray() == S.T.toarray()) # not a good test as S is symmetric     

#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def hitest_square_block_basic(n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs S,S1 (StencilMatrix), Z (ZeroOperator) and a Stencilvector v
    #    Initiate square LOs B,B1 (BlockLO), BZ (ZeroOperator), BI (IdentityOperator) and a BlockVector vb
    # 2. Test general basic operations
    # 3. Test special cases

    # Initiate StencilVectorSpace
    V = StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
       
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
    assert isinstance(B + BI, SumLinearOperator)
    assert isinstance(BI + B, SumLinearOperator)
    assert isinstance(B - BI, SumLinearOperator)
    assert isinstance(BI - B, SumLinearOperator)

    # Negating a BlockLO works as intended
    assert isinstance(-B, BlockLinearOperator)
    assert np.all((-B).dot(vb).toarray() == -( B.dot(vb).toarray() ))

    ## ___Multiplication, Composition, Raising to a Power___

    # Multiplying and Dividing a BlockLO by a scalar returns a BlockLO
    assert isinstance(np.pi*B, BlockLinearOperator)
    assert isinstance(B*np.pi, BlockLinearOperator)
    assert isinstance(B/np.pi, BlockLinearOperator)

    # Composing StencilMatrices works
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
    assert B+BZ == B
    assert BZ+B == B

    # Substracting a ZeroOperator and substracting from a ZeroOperator work as intended
    assert B-BZ == B
    assert np.all((BZ-B).toarray() == (-B).toarray()) # replaces assert BZ-B == -B

    ## ___Composing with Zero- and IdentityOperators___ 

    # Composing a BlockLO with a ZeroOperator returns a ZeroOperator
    assert isinstance(B@BZ, ZeroOperator)
    assert isinstance(BZ@B, ZeroOperator)

    # Composing a BlockLO with the IdentityOperator does not change the object
    assert B@BI == B
    assert BI@B == B

    ## ___Raising to the power of 0 and 1___

    # Raising a BlockLO to the power of 1 or 0 does not change the object / returns an IdentityOperator
    assert B**1 == B
    assert isinstance(B**0, IdentityOperator)

    #################################
    

    

    

    

    

    

    

    

    # Inverting StencilMatrices works
    tol = 1e-6
    M = S.toarray()
    Mi = np.linalg.inv(M)
    realsol = np.dot(Mi,v.toarray())
    realBsol = np.append(realsol, realsol)

    ### Conjugate Gradient test
    B_inv1 = B.inverse('cg', tol=tol)
    assert isinstance(B_inv1, InverseLinearOperator)
    sol = B_inv1.dot(vb).toarray()
    errors = [(sol[i]-realBsol[i])**2 for i in range(len(realBsol))]
    err = sum(errors)
    assert err < tol**2

    ### Preconditioned Conjugate Gradient test, pc = 'jacobi'
    B_inv2 = B.inverse('pcg', pc='jacobi', tol=tol)
    assert isinstance(B_inv2, InverseLinearOperator)
    sol = B_inv2.dot(vb).toarray()
    errors = [(sol[i]-realBsol[i])**2 for i in range(len(realBsol))]
    err = sum(errors)
    assert err < tol**2
    
    # weighted jacobi doesn't work yet
    B_inv3 = B.inverse('pcg', pc='weighted_jacobi', tol=tol)
    assert isinstance(B_inv3, InverseLinearOperator)
    #sol = S_inv3.dot(v)[0].toarray()
    #errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    #err = sum(errors)
    #assert err < tol**2

    ### Biconjugate Gradient test
    B_inv4 = B.inverse('bicg', tol=tol)
    assert isinstance(B_inv4, InverseLinearOperator)
    sol = B_inv4.dot(vb).toarray()
    errors = [(sol[i]-realBsol[i])**2 for i in range(len(realBsol))]
    err = sum(errors)
    assert err < tol**2

#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def hitest_in_place_operations(n1, n2, p1, p2, P1=False, P2=False):

    # testing __imul__ although not explicitly implemented (in the LinearOperator class)

    V = StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])

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
    T = S.copy() # also testing whether copy does what it supposed to
    Sa = S.toarray()

    Z1 += S
    S += Z2 # now equiv. to S = LinearOperator.__add__(S, Z2) = S + Z2 = S

    assert isinstance(Z1, StencilMatrix)
    assert isinstance(S, StencilMatrix)

    S += Z1

    w = S.dot(v)

    assert isinstance(S, StencilMatrix)
    assert np.all(w.toarray() == np.dot(np.dot(2, Sa), v_array))

    Z3 -= T # should be -T = -Sa if ZeroOperator.copy() works
    T -= Z2 # should still be T = Sa
    T -= S+3*Z3 # should be Sa - 2*Sa -(-3*Sa) = 2*Sa if StencilMatrix.copy() works

    w2 = T.dot(v)

    assert isinstance(Z3, StencilMatrix)
    assert isinstance(T, StencilMatrix)
    assert np.all(w2.toarray() == np.dot(np.dot(2, Sa), v_array))
    
#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def hitest_inverse_transpose_interaction(n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs: S (V->V, StencilMatrix), S1 (W->W, StencilMatrix)
        #Initiate BlockLO: B (VxW -> VxW) and a BlockVector u element of VxW
    # 2. For both B and S, check whether all possible combinations of the transpose and the inverse behave as expected

    # Initiate VectorSpaces
    V = StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    W = StencilVectorSpace([n1+2, n2], [p1, p2+1], [P1, P2])
    
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
    C = B.inverse('cg', tol=tol)

    # -1,T & T,-1 -> equal
    assert isinstance(C.T, ConjugateGradient)
    assert isinstance(B.T.inverse('cg', tol=tol), ConjugateGradient)
    assert np.sqrt(sum(((C.T.dot(u) - B.T.inverse('cg', tol=tol).dot(u)).toarray())**2)) < 2*tol
    # -1,T,T -> equal -1
    assert np.sqrt(sum(((C.T.T.dot(u) - C.dot(u)).toarray())**2)) < 2*tol
    # -1,T,-1 -> equal T
    assert isinstance(C.T.inverse('bicg'), BlockLinearOperator)
    assert np.all(C.T.inverse('bicg').dot(u).toarray() == B.T.dot(u).toarray())
    # T,-1,-1 -> equal T
    assert isinstance(B.T.inverse('cg', tol=tol).inverse('pcg', pc='jacobi'), BlockLinearOperator)
    assert np.all(B.T.inverse('cg', tol=tol).inverse('pcg', pc='jacobi').dot(u).toarray() == B.T.dot(u).toarray())
    # T,-1,T -> equal -1
    assert isinstance(B.T.inverse('cg', tol=tol).T, ConjugateGradient)
    assert np.sqrt(sum(((B.T.inverse('cg', tol=tol).dot(u) - C.dot(u)).toarray())**2)) < tol

    ###
    ### StencilMatrix Transpose - Inverse Tests
    ### -1,T & T,-1 --- -1,T,T --- -1,T,-1 --- T,-1,-1 --- T,-1,T (the combinations I test)
    ###

    tol = 1e-5
    C = S.inverse('cg', tol=tol)

    # -1,T & T,-1 -> equal
    assert isinstance(C.T, ConjugateGradient)
    assert isinstance(S.T.inverse('cg', tol=tol), ConjugateGradient)
    assert np.sqrt(sum(((C.T.dot(v) - S.T.inverse('cg', tol=tol).dot(v)).toarray())**2)) < 2*tol
    # -1,T,T -> equal -1
    assert np.sqrt(sum(((C.T.T.dot(v) - C.dot(v)).toarray())**2)) < 2*tol
    # -1,T,-1 -> equal T
    assert isinstance(C.T.inverse('bicg'), StencilMatrix)
    assert np.all(C.T.inverse('bicg').dot(v).toarray() == S.T.dot(v).toarray())
    # T,-1,-1 -> equal T
    assert isinstance(S.T.inverse('cg', tol=tol).inverse('pcg', pc='jacobi'), StencilMatrix)
    assert np.all(S.T.inverse('cg', tol=tol).inverse('pcg', pc='jacobi').dot(v).toarray() == S.T.dot(v).toarray())
    # T,-1,T -> equal -1
    assert isinstance(S.T.inverse('cg', tol=tol).T, ConjugateGradient)
    assert np.sqrt(sum(((S.T.inverse('cg', tol=tol).dot(v) - C.dot(v)).toarray())**2)) < tol

#===============================================================================
#@pytest.mark.parametrize( 'n1', n1array)
#@pytest.mark.parametrize( 'n2', n2array)
#@pytest.mark.parametrize( 'p1', p1array)
#@pytest.mark.parametrize( 'p2', p2array)

def hitest_operator_evaluation():#n1, n2, p1, p2, P1=False, P2=False):

    # 1. Initiate square LOs: S (V->V, StencilMatrix), S1 (W->W, StencilMatrix)
        #Initiate BlockLO: B (VxW -> VxW) and a BlockVector u element of VxW
    # 2. For both B and S, check whether all possible combinations of the transpose and the inverse behave as expected

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
    #for k1 in range(-p1,p1+1):
    #    for k2 in range(-p2,p2+1):
    #        nonzero_values[k1,k2] = 1 + k1*n2 + k2
    #for k1 in range(-p1,p1+1):
    #    for k2 in range(-p2,p2+1):
    #        if k1==0:
    #            if k2<0:
    #                nonzero_values[k1,k2] = nonzero_values[-k1,-k2]
    #        elif k1<0:
    #            nonzero_values[k1,k2] = nonzero_values[-k1,-k2]

    # Initiate VectorSpaces
    V = StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    #W = StencilVectorSpace([n1+2, n2], [p1, p2+1], [P1, P2])
    
    # Initiate positive definite StencilMatrices for which the cg inverse works (necessary for certain tests)
    S = StencilMatrix(V,V)
    #S1 = StencilMatrix(W,W)

    # Initiate StencilVectors 
    v = StencilVector(V)
    for i in range(n1):
        for j in range(n2):
            v[i,j] = 1   
    #w = StencilVector(W)
    #for i in range(n1+2):
    #    for j in range(n2):
    #        w[i,j] = 1   
    
    # Fill the matrices S and S1 (both dense, pos. def.)
    #nonzero_values = dict()
    #for k1 in range(-p1,p1+1):
    #    for k2 in range(-p2,p2+1):
    #        nonzero_values[k1,k2] = 1 + k1*n2 + k2
    #for k1 in range(-p1,p1+1):
    #    for k2 in range(-p2,p2+1):
    #        if k1==0:
    #            if k2<0:
    #                nonzero_values[k1,k2] = nonzero_values[-k1,-k2]
    #        elif k1<0:
    #            nonzero_values[k1,k2] = nonzero_values[-k1,-k2]
    
    #nonzero_values1 = dict()
    #for k1 in range(-p1,p1+1):
    #    for k2 in range(-p2-1,p2+2):
    #        nonzero_values1[k1,k2] = 1 + k1*n2 + k2
    #for k1 in range(-p1,p1+1):
    #    for k2 in range(-p2-1,p2+2):
    #        if k1==0:
    #            if k2<0:
    #                nonzero_values1[k1,k2] = nonzero_values1[-k1,-k2]
    #        elif k1<0:
    #            nonzero_values1[k1,k2] = nonzero_values1[-k1,-k2]

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S[:,:,k1,k2] = nonzero_values[k1,k2]
    S.remove_spurious_entries()

    #for k1 in range(-p1,p1+1):
    #    for k2 in range(-p2-1,p2+2):
    #        S1[:,:,k1,k2] = nonzero_values1[k1,k2]
    #S1.remove_spurious_entries()

    # Initiate a BlockVectorSpace, a BlockLO and a BlockVector
    U = BlockVectorSpace(V, V)#W)
    B = BlockLinearOperator(U, U, ((S, None), (None, S)))#S1)))
    Z = ZeroOperator(U, U)
    I = IdentityOperator(V, V)
    u = BlockVector(U, (v,v))#w))

    tol = 1e-6
    S_ILO = S.inverse('cg', tol=tol)
    B_ILO = B.inverse('cg', tol=tol)

    ### PowerLO tests
    # tests using (S, v), (S_ILO, v), (I, v) and (B, u), (B_ILO, u), (Z, u)

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
    assert l2_norm_diff(np.linalg.solve(Bmat, uarr), bi1) < tol
    assert l2_norm_diff(np.linalg.solve(Bmat, np.linalg.solve(Bmat, uarr)), bi2) < error_est*tol

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
    #assert np.all(np.array([9, 9, 9, 9], dtype=float) == s2)

    si0 = ((S_ILO**0)@v).toarray()
    si1 = ((S_ILO**1)@v).toarray()
    si2 = ((S_ILO**2)@v).toarray()
    S_inv_mat = np.linalg.inv(Smat)
    s_inv_arr = np.matrix.flatten(S_inv_mat)
    error_est = 2 + n1*n2*np.max([np.abs(s_inv_arr[i]) for i in range(len(s_inv_arr))])
    assert np.all(varr == si0)
    assert l2_norm_diff(np.linalg.solve(Smat, varr), si1) < tol
    assert l2_norm_diff(np.linalg.solve(Smat, np.linalg.solve(Smat, varr)), si2) < error_est*tol

    i0 = ((I**0)@v).toarray()
    i1 = ((I**1)@v).toarray()
    i2 = ((I**2)@v).toarray()
    assert np.all(varr == i0)
    assert np.all(varr == i1)
    assert np.all(varr == i2)

    ### SumLO tests
    # tests using (S, v), (S_ILO, v) and (B, u), (B_ILO, u)
    assert np.all(B.toarray() == Bmat)
    Sum1 = B + B_ILO + B + B_ILO
    assert np.all(B.toarray() == Bmat)
    Sum2 = S + S_ILO + S + S_ILO
    sum1 = (Sum1@u).toarray()
    sum2 = (Sum2@v).toarray()
    Sum1_mat = 2*Bmat + 2*B_inv_mat
    Sum2_mat = 2*Smat + 2*S_inv_mat

    assert l2_norm_diff(sum1, np.dot(Sum1_mat, uarr)) < 2*tol
    assert l2_norm_diff(sum2, np.dot(Sum2_mat, varr)) < 2*tol

    ### CompLO tests
    C1 = B@(-B)
    C2 = S@(-S)
    c1 = (C1@u).toarray()
    c2 = (C2@v).toarray()

    assert np.all(-c1 == b2)
    assert np.all(-c2 == s2)

    ### Huge composition

    ZV = ZeroOperator(V, V)

    H1 = S.inverse('cg', tol=tol).T
    assert l2_norm_diff((H1@v).toarray(), si1) < tol
    H2 = S.inverse('cg', tol=tol).inverse('bicg', tol=tol)
    assert np.all((H2@v).toarray() == s1)
    H31 = S**2
    assert np.all((H31@v).toarray() == np.array([9,9,9,9], dtype=float))
    H32 = 2*(S.inverse('cg', tol=tol))

    assert np.all((S.inverse('cg', tol=tol)@v).toarray() == np.linalg.inv(S.toarray())*varr)

    assert np.all(varr == np.array([1,1,1,1], dtype=float))
    assert np.all( np.array([ [2,1,0,0], [1,2,0,0], [0,0,2,1], [0,0,1,2] ], dtype=float) == S.toarray())
    assert np.all((H32@v).toarray() == 2*np.linalg.inv(S.toarray())*varr)
    H3 = (2*(S.inverse('cg', tol=tol))) @ (S**2)
    
    assert np.all(np.array([3, 3, 3, 3], dtype=float) == s1)
    assert l2_norm_diff((H3@v).toarray(), 2*s1) < 2*tol
    H4 = 2*( (S**1) @ (S**0) )
    assert np.all((H4@v).toarray() == 2*s1)
    H5 = ZV @ I
    assert np.all((H5@v).toarray() == H5.codomain.zeros().toarray())

    assert isinstance(H3._codomain, StencilVectorSpace)
    assert isinstance(H3.T, ComposedLinearOperator)
    H = H1 @ ( H2 + H3 - H4 + H5 ).T
    #assert l2_norm_diff((H@v).toarray(), v.toarray()) < 10*tol


#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )