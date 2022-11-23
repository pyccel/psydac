import pytest
import numpy as np

from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.basic import ZeroOperator, IdentityOperator, ComposedLinearOperator, SumLinearOperator, PowerLinearOperator, InverseLinearOperator, ScaledLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg2.iterative_solvers import BiConjugateGradient
#===============================================================================

n1array = [2, 7]
n2array = [2, 3]
p1array = [1, 3]
p2array = [1, 3]

#===============================================================================
# SERIAL TESTS
#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def test_square_stencil_basic_operations(n1, n2, p1, p2, P1=False, P2=False):

    # initiate StencilVectorSpace
    V = StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    
    # Initiate Linear Operators
    Z = ZeroOperator(V, V)
    I = IdentityOperator(V, V)
    SM1 = StencilMatrix(V, V)
    SM2 = StencilMatrix(V, V)

    # Initiate a StencilVector
    v = StencilVector(V)
    for i in range(n1):
        for j in range(n2):
            v[i,j] = 1

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

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            SM1[:,:,k1,k2] = nonzero_values1[k1,k2]
    SM1.remove_spurious_entries()
    SM1a = SM1.toarray()

    nonzero_values2 = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1 == 0:
                if k2 == 0:
                    nonzero_values2[k1,k2] = 1
                else:
                    nonzero_values2[k1,k2] = 0
            else:
                nonzero_values2[k1,k2] = 0
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            SM2[:,:,k1,k2] = nonzero_values2[k1,k2]
    SM2.remove_spurious_entries()
    SM2a = SM2.toarray()

    # Construct exact matrices by hand
    A1 = np.zeros( SM1.shape )
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1,p1+1):
                for k2 in range(-p2,p2+1):
                    j1 = (i1+k1) % n1
                    j2 = (i2+k2) % n2
                    i  = i1*(n2) + i2
                    j  = j1*(n2) + j2
                    if (P1 or 0 <= i1+k1 < n1) and (P2 or 0 <= i2+k2 < n2):
                        A1[i,j] = nonzero_values1[k1,k2]

    A2 = np.zeros( SM1.shape )
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1,p1+1):
                for k2 in range(-p2,p2+1):
                    j1 = (i1+k1) % n1
                    j2 = (i2+k2) % n2
                    i  = i1*(n2) + i2
                    j  = j1*(n2) + j2
                    if (P1 or 0 <= i1+k1 < n1) and (P2 or 0 <= i2+k2 < n2):
                        A2[i,j] = nonzero_values2[k1,k2]

    # Check shape and data in 2D array
    assert np.all(v.toarray() == np.ones(n1*n2))

    assert SM1a.shape == SM1.shape
    assert np.all( SM1a == A1 )
    assert SM2a.shape == SM2.shape
    assert np.all( SM2a == A2 )

    # Adding a ZeroOperator does not change the StencilMatrix
    assert SM1+Z == SM1
    assert Z+SM1 == SM1

    # Substracting a ZeroOperator and substracting from a ZeroOperator work as intended
    assert SM1-Z == SM1
    assert np.all((Z-SM1).toarray() == (-SM1).toarray()) # replaces assert Z-SM1 == -SM1

    # Adding and Substracting StencilMatrices returns a StencilMatrix
    assert isinstance(SM1 + SM2, StencilMatrix)
    assert isinstance(SM1 - SM2, StencilMatrix)

    # Multiplying and Dividing a StencilMatrix by a scalar returns a StencilMatrix
    assert isinstance(np.pi*SM1, StencilMatrix)
    assert isinstance(SM1*np.pi, StencilMatrix)
    assert isinstance(SM1/np.pi, StencilMatrix)

    # Composing StencilMatrices works
    assert isinstance(SM1@SM2, ComposedLinearOperator)

    # Composing a StencilMatrix with a ZeroOperator returns a ZeroOperator
    assert isinstance(SM1@Z, ZeroOperator)
    assert isinstance(Z@SM1, ZeroOperator)

    # Composing a StencilMatrix with the IdentityOperator does not change the object
    assert SM1@I == SM1
    assert I@SM1 == SM1

    # Raising a StencilMatrix to a power works
    assert isinstance(SM1**3, PowerLinearOperator)

    # Raising a StencilMatrix to the power of 1 or 0 does not change the object / returns an IdentityOperator
    assert SM1**1 == SM1
    assert isinstance(SM1**0, IdentityOperator)

    # Inverting StencilMatrices works
    tol = 1e-6
    M = SM1.toarray()
    Mi = np.linalg.inv(M)
    realsol = np.dot(Mi,v.toarray())

    ### Conjugate Gradient test
    SM1_inv1 = SM1.inverse('cg', tol=tol)
    assert isinstance(SM1_inv1, InverseLinearOperator)
    sol = SM1_inv1.dot(v)[0].toarray()
    errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    err = sum(errors)
    assert err < tol**2

    ### Preconditioned Conjugate Gradient test, pc = 'jacobi'
    SM1_inv2 = SM1.inverse('pcg', pc='jacobi', tol=tol)
    assert isinstance(SM1_inv2, InverseLinearOperator)
    sol = SM1_inv2.dot(v)[0].toarray()
    errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    err = sum(errors)
    assert err < tol**2
    
    # weighted jacobi doesn't work yet
    SM1_inv3 = SM1.inverse('pcg', pc='weighted_jacobi', tol=tol)
    assert isinstance(SM1_inv3, InverseLinearOperator)
    #sol = SM1_inv3.dot(v)[0].toarray()
    #errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    #err = sum(errors)
    #assert err < tol**2

    ### Biconjugate Gradient test
    SM1_inv4 = SM1.inverse('bicg', tol=tol)
    assert isinstance(SM1_inv4, InverseLinearOperator)
    sol = SM1_inv4.dot(v)[0].toarray()
    errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    err = sum(errors)
    assert err < tol**2

    # Transposing the inverse of a StencilMatrix works
    T = SM1_inv4.T
    assert isinstance(T, BiConjugateGradient)
    assert np.all(T.linop.toarray() == SM1.T.toarray()) # not a good test as SM1 is symmetric 

    # Adding and Substracting StencilMatrices and other LOs returns a SumLinearOperator object
    assert isinstance(SM1 + I, SumLinearOperator)
    assert isinstance(I + SM1, SumLinearOperator)
    assert isinstance(SM1 - I, SumLinearOperator)
    assert isinstance(I - SM1, SumLinearOperator)

#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def test_square_block_basic_operations(n1, n2, p1, p2, P1=False, P2=False):

    # initiate StencilVectorSpace
    V = StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    Vb = BlockVectorSpace(V,V)
    
    # Initiate Linear Operators
    Z = ZeroOperator(V, V)
    #BZ = BlockLinearOperator(Vb, Vb, ((Z, Z), (Z, Z)))
    BZ = ZeroOperator(Vb, Vb)
    I = IdentityOperator(V, V)
    #BI = BlockLinearOperator(Vb, Vb, ((I, None), (None, I)))
    BI = IdentityOperator(Vb, Vb)
    SM1 = StencilMatrix(V, V)
    SM2 = StencilMatrix(V, V) # is actually an identity matrix

    # Initiate a StencilVector
    v = StencilVector(V)
    for i in range(n1):
        for j in range(n2):
            v[i,j] = 1
    vb = BlockVector(Vb, (v, v))

    nonzero_values1 = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values1[k1,k2] = 1 + k1*n2 + k2
    #print(nonzero_values1)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1==0:
                if k2<0:
                    nonzero_values1[k1,k2] = nonzero_values1[-k1,-k2]
            elif k1<0:
                nonzero_values1[k1,k2] = nonzero_values1[-k1,-k2]

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            SM1[:,:,k1,k2] = nonzero_values1[k1,k2]
    SM1.remove_spurious_entries()
    SM1a = SM1.toarray()
    BM1 = BlockLinearOperator(Vb, Vb, ((SM1, Z), (None, SM1)))

    nonzero_values2 = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            if k1 == 0:
                if k2 == 0:
                    nonzero_values2[k1,k2] = 1
                else:
                    nonzero_values2[k1,k2] = 0
            else:
                nonzero_values2[k1,k2] = 0
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            SM2[:,:,k1,k2] = nonzero_values2[k1,k2]
    SM2.remove_spurious_entries()
    SM2a = SM2.toarray()
    #print(n1, n2, p1, p2)
    #print(SM1a)
    #print(SM2a)
    BM2 = BlockLinearOperator(Vb, Vb, ((SM2, None), (Z, SM2)))

    # Construct exact matrices by hand
    A1 = np.zeros( SM1.shape )
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1,p1+1):
                for k2 in range(-p2,p2+1):
                    j1 = (i1+k1) % n1
                    j2 = (i2+k2) % n2
                    i  = i1*(n2) + i2
                    j  = j1*(n2) + j2
                    if (P1 or 0 <= i1+k1 < n1) and (P2 or 0 <= i2+k2 < n2):
                        A1[i,j] = nonzero_values1[k1,k2]

    A2 = np.zeros( SM1.shape )
    for i1 in range(n1):
        for i2 in range(n2):
            for k1 in range(-p1,p1+1):
                for k2 in range(-p2,p2+1):
                    j1 = (i1+k1) % n1
                    j2 = (i2+k2) % n2
                    i  = i1*(n2) + i2
                    j  = j1*(n2) + j2
                    if (P1 or 0 <= i1+k1 < n1) and (P2 or 0 <= i2+k2 < n2):
                        A2[i,j] = nonzero_values2[k1,k2]

    # Check shape and data in 2D array
    assert np.all(vb.toarray() == np.ones(2*n1*n2))

    assert SM1a.shape == SM1.shape
    assert np.all( SM1a == A1 )
    assert SM2a.shape == SM2.shape
    assert np.all( SM2a == A2 )

    # Adding a ZeroOperator does not change the StencilMatrix
    # ### it prolly creates an SLO object!
    assert BM1+BZ == BM1
    assert BZ+BM1 == BM1

    # Adding StencilMatrices returns a StencilMatrix
    assert isinstance(BM1 + BM2, BlockLinearOperator)

    # Multiplying a StencilMatrix returns a StencilMatrix
    assert isinstance(np.pi*BM1, BlockLinearOperator)
    assert isinstance(BM1*np.pi, BlockLinearOperator)

    # Composing StencilMatrices works
    assert isinstance(BM1@BM2, ComposedLinearOperator)

    # Composing a StencilMatrix with a ZeroOperator returns a ZeroOperator
    # ### see adding a ZeroOperator ...
    assert isinstance(BM1@BZ, ZeroOperator)
    assert isinstance(BZ@BM1, ZeroOperator)

    # Composing a StencilMatrix with the IdentityOperator does not change the object
    ### 
    assert BM1@BI == BM1
    assert BI@BM1 == BM1

    # Raising a StencilMatrix to a power works
    assert isinstance(BM1**3, PowerLinearOperator)

    # Raising a StencilMatrix to the power of 1 or 0 does not change the object / returns an IdentityOperator
    assert BM1**1 == BM1
    assert isinstance(BM1**0, IdentityOperator)

    # Inverting StencilMatrices works
    tol = 1e-6
    M = SM1.toarray()
    Mi = np.linalg.inv(M)
    realsol = np.dot(Mi,v.toarray())
    realBsol = np.append(realsol, realsol)

    ### Conjugate Gradient test
    BM1_inv1 = BM1.inverse('cg', tol=tol)
    assert isinstance(BM1_inv1, InverseLinearOperator)
    sol = BM1_inv1.dot(vb)[0].toarray()
    errors = [(sol[i]-realBsol[i])**2 for i in range(len(realBsol))]
    err = sum(errors)
    assert err < tol**2

    ### Preconditioned Conjugate Gradient test, pc = 'jacobi'
    BM1_inv2 = BM1.inverse('pcg', pc='jacobi', tol=tol)
    assert isinstance(BM1_inv2, InverseLinearOperator)
    sol = BM1_inv2.dot(vb)[0].toarray()
    errors = [(sol[i]-realBsol[i])**2 for i in range(len(realBsol))]
    err = sum(errors)
    assert err < tol**2
    
    # weighted jacobi doesn't work yet
    BM1_inv3 = BM1.inverse('pcg', pc='weighted_jacobi', tol=tol)
    assert isinstance(BM1_inv3, InverseLinearOperator)
    #sol = SM1_inv3.dot(v)[0].toarray()
    #errors = [(sol[i]-realsol[i])**2 for i in range(len(realsol))]
    #err = sum(errors)
    #assert err < tol**2

    ### Biconjugate Gradient test
    BM1_inv4 = BM1.inverse('bicg', tol=tol)
    assert isinstance(BM1_inv4, InverseLinearOperator)
    sol = BM1_inv4.dot(vb)[0].toarray()
    errors = [(sol[i]-realBsol[i])**2 for i in range(len(realBsol))]
    err = sum(errors)
    assert err < tol**2

    # Adding StencilMatrices and other LOs returns a SumLinearOperator object
    assert isinstance(BM1 + BI, SumLinearOperator)
    assert isinstance(BI + BM1, SumLinearOperator)

#===============================================================================
@pytest.mark.parametrize( 'n1', n1array)
@pytest.mark.parametrize( 'n2', n2array)
@pytest.mark.parametrize( 'p1', p1array)
@pytest.mark.parametrize( 'p2', p2array)

def test_in_place_operations(n1, n2, p1, p2, P1=False, P2=False):

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
    #T = StencilMatrix(V,V)
    Z1 = ZeroOperator(V,V)
    Z2 = ZeroOperator(V,V)
    Z3 = Z1.copy()

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            S[:,:,k1,k2] = nonzero_values1[k1,k2]
            #T[:,:,k1,k2] = nonzero_values1[k1,k2]
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
    pass

# also test inverse-inverse interaction (inverse transpose inverse transpose)

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )