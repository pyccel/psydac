import numpy as np

from psydac.linalg2.ndarray import NdarrayVectorSpace, NdarrayVector, NdarrayLinearOperator
from psydac.linalg2.basic import ZeroOperator, IdOperator

#===============================================================================
if __name__ == "__main__":
    
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
    I1 = IdOperator(V,V)
    I2 = IdOperator(W,W)
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
    G = C @ D @ I1
    H = I2 @ E @ I1 @ F
    print('Convolution LOs G and H have been successfully created, both including identity operators without matrix representation')
    print()

    print('5. Testing both creation of ~arbitrary~ combinations of the given operators A,B,Z,I1,I2,G,H as well as')
    print('the evaluation of such operator at v=(1,1):')
    print()
    T1 = 2*(A + Z + G + Z + H + B)
    print('5.1 Successfully created operator T1 = 2*(A + Z + G + Z + H + B)')
    v = NdarrayVector(space=V, data=np.ones(2,dtype=float))
    print('    Successfully created vector (1,1) of space V')
    y1 = T1.dot(v)
    print('    Successfully evaluated T1.dot(v) if [ True True True]')
    print(y1._data==np.array([10,10,22],dtype=float))
    print()
    T2 = 1*(T1 + 0*T1) + 2*0.5*I2 @ T1 + Z @ (I1 @ I1 + I1)  
    print('5.2 Successfully created operator T2 = 1*(T1 + 0*T1) + 2*0.5*Id2 @ T1 + Z1 @ (Id1 @ Id1 + Id1)')
    y2 = T2.dot(v)
    print('    Successfully evaluated T2.dot(v) if [ True True True]')
    print(y2.data)
    print(y2._data==np.array([20,20,44],dtype=float))
    print()