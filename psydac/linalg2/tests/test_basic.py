#from abc   import ABCMeta, abstractmethod

import numpy as np
from numpy import ndarray

#from psydac.linalg2.basic import VectorSpace, Vector, LinearOperator
from psydac.linalg2.ndarray import NdarrayVectorSpace, NdarrayVector, NdarrayLinearOperator
from psydac.linalg2.expr import ZeroOperator, IdOperator

#===============================================================================
if __name__ == "__main__":
    """
    # NdarrayVector and NdarrayVectorSpace __init__
    V = NdarrayVectorSpace(dim=6, dtype=float)
    a = NdarrayVector(space=V)
    b = NdarrayVector(space=V, data=1)
    c = NdarrayVector(space=V, data=np.array([1,2,3,4,5,6], dtype=float))
    try:
        NdarrayVector(space=V, data=np.array([1,2,3,4,5,6,7], dtype=float))
    except AssertionError as msg:
        print('NdarrayVector: Error if wrong dimension (too high)')
        print('data neither scalar nor of right dimension' == str(msg))
    try:
        NdarrayVector(space=V, data=np.array([1,2], dtype=float))
    except AssertionError as msg:
        print('NdarrayVector: Error if wrong dimension (too low)')
        print('data neither scalar nor of right dimension' == str(msg))
    try:
        NdarrayVector(space=V, data=np.array([1,2,3,4,5,6], dtype=int))
    except AssertionError as msg:
        print('NdarrayVector: Error if wrong data type')
        print('data of wrong dtype' == str(msg))
    
    W = NdarrayVectorSpace(dim=6, dtype=int)
    x = NdarrayVector(space=W, data=1)
    y = np.array([2,3,2,3,2,3], dtype=float)

    # dot
    b.dot(c)
    try:
        b.dot(x)
    except AssertionError as msg:
        print('NdarrayVector.dot: Error if wrong space')
        print('v and self dont belong to the same space' == str(msg))

    try:
        b.dot(y)
    except AssertionError as msg:
        print('NdarrayVector.dot: Error if np.array instead of NdarrayVector')
        print('v is not a NdarrayVector' == str(msg))

    # __mul__
    u = 3*x
    v = x*3
    print('NdarrayVector.__mul__: Scalar multiplication from left and right works as expected')
    print(u._data == v._data)
    print(u._data == 3*np.ones(6, dtype=int))
    try:
        x*x
    except AssertionError as msg:
        print('NdarrayVector.__mul__: Error if not scalar')
        print('a is not a scalar' == str(msg))

    # __add__
    a+b
    try:
        c+y
    except AssertionError as msg:
        print('NdarrayVector.__add__: Error if not NdarrayVector')
        print('v is not NdarrayVector' == str(msg))

    try:
        a+x
    except AssertionError as msg:
        print('NdarrayVector.__add__: Error if wrong space')
        print('v space is not self space' == str(msg))

    # NdarrayLinearOperator __init__
    U = NdarrayVectorSpace( dim = 2 )
    u = NdarrayVector(U,1)
    A = NdarrayLinearOperator( domain = V, codomain = U, matrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]], dtype=float) )
    Z = ZeroOperator( domain = V, codomain = U )
    print(A.dot(b)._data == np.array([1,1], dtype=float))
    print(A.idot(b,u)._data == np.array([2,2], dtype=float))
    
    S = A+Z
    print('Basic SumLinearOperator Test:')
    S1 = S.addends[0]
    S2 = S.addends[1]
    print(S1.dot(b)._data == np.ones(2, dtype=float))
    print(S1.idot(b,u)._data == np.array([2,2], dtype=float))
    print(S2.dot(b)._data == np.zeros(2, dtype=float))
    print(S2.idot(b,u)._data == np.ones(2, dtype=float))
    print(S.domain)
    print(S.codomain)
    print(S.shape == (2,6))
    print(S.addends[0]._matrix)
    print(S.addends[1]._matrix)
    print(S.length)
    index = np.where([S._addends[i]._matrix is not None for i in range(S.length) ])
    print(index[0])

    try:
        S.dtype
    except NotImplementedError as msg:
        print('Class does not provide a dtype method without a matrix' == str(msg))

    #T = A+A
    #print(isinstance(T,SumLinearOperator))
    #print(T.length)
    #print("B is SLO")
    #R = A+T
    #print(isinstance(R,SumLinearOperator))
    #print(R.length)
    #print("A is SLO")
    #H = T+A
    #print(isinstance(H,SumLinearOperator))
    #print(H.length)
    #T = A+A+Z+A
    #print("First SLO.dot test:")
    #print(T.dot(b)._data)
    #print(T.length)
    """
    print('Implementation Test:')
    print()

    print('1. Creating VectorSpaces:')
    U = NdarrayVectorSpace(1)
    V = NdarrayVectorSpace(2)
    W = NdarrayVectorSpace(3)
    print('VectorSpaces U,V,W of dim 1,2,3 successfully created.')
    print()

    print('2. Creating LOs from V->W with matrix representation:')
    a = np.array([[1,0],[0,1],[1,1]], dtype=float)
    b = np.array([[0,2],[2,0],[3,3]], dtype=float)
    A = NdarrayLinearOperator(domain=V, codomain=W, matrix=a)
    B = NdarrayLinearOperator(domain=V, codomain=W, matrix=b)
    print('Successfully created LOs A and B, matrix.dimesion = (3,2).')
    print()

    print('3. Creating LOs from V->W without matrix representation:')
    Z = ZeroOperator(domain=V, codomain=W)
    I1 = IdOperator(V)
    I2 = IdOperator(W)
    print('Sucessfully created three LOs without matrix representation, namely two IdOperator and one ZeroOperator.')
    print()

    print('4. Creating compositions of LOs from V->W:')
    c = np.array([[1],[0],[1]], dtype=float)
    d = np.array([[1,-1]], dtype=float)
    e = np.array([[1,-1],[0,-2],[3,0]], dtype=float)
    f = np.array([[0,1],[-1,0]], dtype=float)
    C = NdarrayLinearOperator(domain=U, codomain=W, matrix=c)
    D = NdarrayLinearOperator(domain=V, codomain=U, matrix=d)    
    E = NdarrayLinearOperator(domain=V, codomain=W, matrix=e)
    F = NdarrayLinearOperator(domain=V, codomain=V, matrix=f)
    G = C@D@I1
    H = I2@E@I1@F
    print('Convolution LOs G and H have been successfully created, both including identity operators without matrix representation')
    print()

    print('5. Testing both creation of ~arbitrary~ combinations of the given operators A,B,Z,I1,I2,G,H as well as')
    print('the evaluation of such operator at v=(1,1):')
    print()
    T1 = 2*(A+Z+G+Z+H+B)
    print('5.1 Successfully created operator T1 = 2*(A+Z+G+Z+H+B)')
    v = NdarrayVector(space=V, data=np.ones(2,dtype=float))
    print('    Successfully created vector (1,1) of space V')
    y1 = T1.dot(v)
    print('    Successfully evaluated T1.dot(v) if [ True True True]')
    print(y1._data==np.array([10,10,22],dtype=float))
    print()
    T2 = 1*(T1+0*T1) + 2*0.5*I2@T1 + Z@(I1@I1+I1)
    print('5.2 Successfully created operator T2 = 1*(T1+0*T1) + 2*0.5*Id2@T1 + Z1@(Id1@Id1+Id1)')
    y2 = T2.dot(v)
    print('    Successfully evaluated T2.dot(v) if [ True True True]')
    print(y2._data==np.array([20,20,44],dtype=float))
    print()