import numpy as np
from scipy.sparse        import coo_matrix

from psydac.linalg2.block import BlockMatrix, BlockVector, BlockVectorSpace
from psydac.linalg2.direct_solvers import BandedSolver, SparseSolver
from psydac.linalg2.ndarray import NdarrayVectorSpace, NdarrayVector, NdarrayLinearOperator
from psydac.linalg2.basic import InverseLinearOperator, ZeroOperator, IdentityOperator
from psydac.linalg2.stencil import StencilVectorSpace, StencilVector, StencilMatrix

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
    sh = np.array([[1,1],[2,0],[0,1]], dtype=float)
    Sh = NdarrayLinearOperator(domain=V, codomain=W, matrix=sh)
    A = NdarrayLinearOperator(domain=V, codomain=W, matrix=a)
    B = NdarrayLinearOperator(domain=V, codomain=W, matrix=b)
    print('Successfully created LOs A and B, matrix.dimesion = (3,2).')
    print()

    print('3. Creating LOs from V->W without matrix representation:')
    Z = ZeroOperator(domain=V, codomain=W)
    Z2 = ZeroOperator(domain=V, codomain=V)
    I1 = IdentityOperator(V,V)
    I2 = IdentityOperator(W,W)
    bmat = np.array([[0,1,1], [1,1,1], [0,0,0], [0,0,0]])
    S = BandedSolver(W, 1, 0, bmat)
    v = [1, 1, 1]
    i = [0, 1, 2]
    j = [0, 1, 2]
    spmat = coo_matrix((v, (i,j)), shape=(3,3))
    S2 = SparseSolver(W, spmat)
    print('Sucessfully created six LOs without matrix representation, namely two IdentityOperators, two ZeroOperators and two LinearSolvers.')
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
    LS = S @ Sh
    LS2 = S2 @ Sh

    print('Composition LOs G, H, LS and LS2 have been successfully created, G and H including identity operators without matrix representation and LS and LS2 as LinearSolver compositions')
    print()

    print('5. Testing both creation of ~arbitrary~ combinations of the given operators A,B,Z,I1,I2,G,H as well as')
    print('the evaluation of such operator at v=(1,1):')

    print()
    T1 = 2*(A + Z + G + Z + H + B)
    print('5.1 Successfully created operator T1 = 2*(A + Z + G + Z + H + B)')
    v = NdarrayVector(space=V, data=np.ones(2,dtype=float))
    print('    Successfully created vector (1,1) of space V')
    y1 = T1.dot(v)

    print()
    ops = T1.operator.addends
    classes = [ops[i].__class__.__name__ for i in range(len(ops))]
    print(classes)
    print()

    print('    Successfully evaluated T1.dot(v) if [ True True True]')
    print(y1.data==np.array([10,10,22],dtype=float))
    print()
    T2 = 1*(T1 + 0*T1) + 2*0.5*I2 @ T1 + Z @ (I1 @ I1 + I1)  
    print('5.2 Successfully created operator T2 = 1*(T1 + 0*T1) + 2*0.5*Id2 @ T1 + Z1 @ (Id1 @ Id1 + Id1)')
    y2 = T2.dot(v)

    print()
    ops = T2.addends
    classes = [ops[i].__class__.__name__ for i in range(len(ops))]
    print(classes)
    print(ops[0].scalar)
    print(ops[0].operator.__class__.__name__)
    print([ops[0].operator.addends[i].__class__.__name__ for i in range(len(ops[0].operator.addends))])
    print(ops[1].scalar)
    print(ops[1].operator.__class__.__name__)
    print([ops[1].operator.addends[i].__class__.__name__ for i in range(len(ops[1].operator.addends))])
    print()

    print('    Successfully evaluated T2.dot(v) if [ True True True]')
    print(y2.data)
    print(y2.data==np.array([20,20,44],dtype=float))
    print()

    print('6. Testing PowLinearOperator:')
    P1 = F ** 2
    P2 = F ** 1
    P3 = F ** 0
    P4 = I1 ** 2
    P5 = I1 ** 1
    P6 = I1 ** 0
    P7 = Z2 ** 2
    P8 = Z2 ** 1
    P9 = Z2 ** 0
    print('Successfully created PowLinearOperators using factorials 0, 1 and 2 of a matrix represented LO as well as an IdLO and a ZeroLO')
    print('Checking for right implementation:')
    print()
    print(P1.dot(v).data == -v.data)
    print(P2.dot(v).data == np.array([1, -1], dtype=float))
    print(P3.dot(v).data == v.data)
    print(P4.dot(v).data == v.data)
    print(P5.dot(v).data == v.data)
    print(P6.dot(v).data == v.data)
    print(P7.dot(v).data == V.zeros().data)
    print(P8.dot(v).data == V.zeros().data)
    print(P9.dot(v).data == np.ones(2, dtype=float))
    print()

    print('Testing the implementation of LinearSolvers:')
    print()
    
    T3 = 2*(A + Z + G + Z + LS + 0.5*LS2 + H + B)

    ops = T3.operator.addends
    classes = [ops[i].__class__.__name__ for i in range(len(ops))]
    print(classes)

    y3 = T3.dot(v)

    ops = T3.operator.addends
    classes = [ops[i].__class__.__name__ for i in range(len(ops))]
    print(classes)

    print()
    print(y3.data == np.array([14,14,25],dtype=float))

    T4 = 1*(T3 + 0*T3) + 2*0.5*I2 @ T3 + Z @ (I1 @ I1 + I1)  
    y4 = T4.dot(v)
    print(y4.data==np.array([28,28,50],dtype=float))

    ############################################
    print()
    print('InverseLinearOperator tests: throughout b = [1, 1, 1, 1]')
    print()
    print('1. "cg"-inverse of an NdarrayLinearOperator')
    print('2. "bicg"-inverse of an NdarrayLinearOperator')
    print('3. "pcg"-"jacobi"-inverse of a StencilMatrix')
    print('4. "pcg"-"jacobi"-inverse of a Blockmatrix consisting of StencilMatrices')
    print()
    print('1. Creating the cg inverse P_cg of the 4x4 NdarrayLinearOperator P = ')
    print('[1, 0, 2, 0]')
    print('[0, 1, 0, 2]')
    print('[3, 0, 1, 0]')
    print('[0, 3, 0, 1]')
    print('Passing kwargs = {"x0": x0, "verbose": True}, x0 an NdarrayVector [0, 0, 0, 0]')
    mat = np.array([[1, 0, 2, 0], [0, 1, 0, 2], [3, 0, 1, 0], [0, 3, 0, 1]], dtype=float)
    S = NdarrayVectorSpace(4)
    P = NdarrayLinearOperator(domain=S, codomain=S, matrix=mat)
    x0_vec = np.array([0, 0, 0, 0], dtype=float)
    x0 = NdarrayVector(space=S, data=x0_vec)
    P_cg = P.inverse('cg', verbose = True, x0 = x0)
    #P_cg.getoptions()
    print('Setting verbose=False')
    P_cg.setoptions(verbose=False)
    #P_cg.getoptions()
    #print('Success')
    #print()

    #print('Creating a r.h.s. NdarrayVector b = [1, 1, 1, 1] and calling P_cg.dot(b)')
    b_vec = np.array([1, 1, 1, 1], dtype=float)
    b = NdarrayVector(space=S, data=b_vec)
    x, info = P_cg.dot(b)
    print('cg NdarrayLinearOperator results:')
    print('Expected: x = [0.2, 0.2, 0.4, 0.4]')
    print(info)
    print("x = ", x.data)
    #print('Success')
    print()

    print('2. Creating the bicg inverse P_bicg of the 4x4 NdarrayLinearOperator P')
    #x0_vec = np.zeros(4, dtype=float)
    #x0 = NdarrayVector(space=S, data=x0_vec)
    print('Verifying that x0 does not get changed anymore:')
    print('x0: ', x0.data)
    #print('P matrix:')
    #print(P.matrix)
    #print('dot test')
    #print(P.dot())
    print('Passing kwargs = {"x0": x0, "verbose": True}, x0 an NdarrayVector [0, 0, 0, 0]')
    P_bicg = P.inverse('bicg', x0=x0, verbose=True)
    #print('Setting verbose=False')
    #P_bicg.setoptions(verbose=False)
    x, info = P_bicg.dot(b)
    print('bicg NdarrayLinearOperator results:')
    print('Expected: x = [0.2, 0.2, 0.4, 0.4]')
    print(info)
    print("x = ", x.data)
    print()

    print('3. Creating the pcg jacobi inverse M_inv of the 4x4 StencilMatrix M =')
    print('[1, 0, 2, 0]')
    print('[0, 1, 0, 2]')
    print('[3, 0, 1, 0]')
    print('[0, 3, 0, 1]')
    print('Passing kwargs = {"pc": "jacobi", "x0": x0, "verbose": True}, x0 a StencilVector [0, 0, 0, 0]')

    n1=2
    n2=2
    p1=1
    p2=1
    P1=False
    P2=False
    nonzero_values = dict()
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            nonzero_values[k1,k2] = ((k1%3)+1)*((k2+1)%2)
    V = StencilVectorSpace( [n1,n2], [p1,p2], [P1,P2] )
    M = StencilMatrix( V, V )
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M[:,:,k1,k2] = nonzero_values[k1,k2]
    M.remove_spurious_entries()

    x0 = StencilVector(V)

    M_inv = M.inverse('pcg', pc='jacobi', x0=x0, verbose=True)
    print('Setting verbose=False')
    M_inv.setoptions(verbose=False)
    #M_inv2 = M.inverse('pcg', pc='weighted_jacobi', x0=x0, verbose=False)
    #print('Success')
    #print()

    #print('Creating a r.h.s. StencilVector b = [1, 1, 1, 1] and calling M_inv.dot(b) and M_inv2.dot(b)')
    print('pcg jacobi StencilMatrix results:')
    print('Expected: x = [0.2, 0.2, 0.4, 0.4]')
    b = StencilVector(V)
    b[0] = 1
    b[1] = 1    
    x, info = M_inv.dot(b)
    #x2, info2 = M_inv2.dot(b)
    #print('jacobi output:')
    print(info)
    print("x = ", x.toarray())
    #print('weighted_jacobi output:')
    #print(info2)
    #print("x2 = ", x2.toarray())
    #print('Success (w/o weighted jacobi)')
    print()

    print('4. Creating the pcg jacobi inverse M_block_inv of a BlockMatrix M_block = ')
    print('[M, None]')
    print('[None, M]')
    print('Passing kwargs = {"pc": "jacobi", "x0": x0, "verbose": True}, x0 an BlockVector [x0, x0]')
    #print('a BlockVector b_block = [b, b] as well as x0_block = [x0, x0]')

    X = BlockVectorSpace(V, V)
    b_block = BlockVector(X, (b, b))
    x0_block = BlockVector(X, (x0, x0))
    M_block = BlockMatrix(X, X, ((M, None), (None, M)))

    M_block_inv = M_block.inverse('pcg', pc='jacobi', x0=x0_block, verbose=True)
    print('Setting verbose=False')
    M_block_inv.setoptions(verbose=False)
    x_block, info = M_block_inv.dot(b_block)
    print('pcg jacobi BlockMatrix results:')
    print('Expected: x = [0.2, 0.2, 0.4, 0.4, 0.2, 0.2, 0.4, 0.4]')
    print(info)
    print("x_block = ", x_block.toarray())
    print()

    print('Testing basic transpose implementation:')
    ET = E.T # [[1,-1],[0,-2],[3,0]] -> [[1, 0, 3], [-1, -2, 0]]
    vt = NdarrayVector(W, data=np.ones(3, dtype=float))
    print(ET.dot(vt).data == np.array([4, -3], dtype=float))