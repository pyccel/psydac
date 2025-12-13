#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.basic import LinearOperator, ZeroOperator, IdentityOperator, ComposedLinearOperator, SumLinearOperator, PowerLinearOperator, ScaledLinearOperator
from psydac.linalg.basic import MatrixFreeLinearOperator
from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.solvers import ConjugateGradient, inverse
from psydac.ddm.cart       import DomainDecomposition, CartDecomposition

from psydac.linalg.tests.test_linalg import get_StencilVectorSpace, get_positive_definite_StencilMatrix, assert_pos_def

def get_random_StencilMatrix(domain, codomain):

    np.random.seed(2)
    V = domain
    W = codomain
    assert isinstance(V, StencilVectorSpace)
    assert isinstance(W, StencilVectorSpace)
    [n1, n2] = V._npts
    [p1, p2] = V._pads
    [P1, P2] = V._periods
    assert (P1 == False) and (P2 == False)

    [m1, m2] = W._npts
    [q1, q2] = W._pads
    [Q1, Q2] = W._periods
    assert (Q1 == False) and (Q2 == False)

    S = StencilMatrix(V, W)

    for i in range(0, q1+1):
        if i != 0:
            for j in range(-q2, q2+1):
                S[:, :, i, j] = 2*np.random.random()-1
        else:
            for j in range(1, q2+1):
                S[:, :, i, j] = 2*np.random.random()-1
    S.remove_spurious_entries()

    return S

def get_random_StencilVector(V):
    np.random.seed(3)
    assert isinstance(V, StencilVectorSpace)
    [n1, n2] = V._npts
    v = StencilVector(V)
    for i in range(n1):
        for j in range(n2):
            v[i,j] = np.random.random()
    return v

#===============================================================================
@pytest.mark.parametrize('n1', [3, 5])
@pytest.mark.parametrize('n2', [4, 7])
@pytest.mark.parametrize('p1', [2, 6])
@pytest.mark.parametrize('p2', [3, 9])

def test_fake_matrix_free(n1, n2, p1, p2):
    P1 = False
    P2 = False
    m1 = (n2+n1)//2
    m2 = n1+1
    q1 = p1 # using same degrees because both spaces must have same padding for now
    q2 = p2 
    V1 = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    V2 = get_StencilVectorSpace([m1, m2], [q1, q2], [P1, P2])
    S = get_random_StencilMatrix(codomain=V2, domain=V1)
    O = MatrixFreeLinearOperator(codomain=V2, domain=V1, dot=lambda v: S @ v)

    print(f'O.domain = {O.domain}')
    print(f'S.domain = {S.domain}')
    print(f'V1:      = {V1}')
    v = get_random_StencilVector(V1)
    tol = 1e-10
    y = S.dot(v)
    x = O.dot(v)
    print(f'error = {np.linalg.norm( (x - y).toarray() )}')
    assert np.linalg.norm( (x - y).toarray() ) < tol
    O.dot(v, out=x)
    print(f'error = {np.linalg.norm( (x - y).toarray() )}')
    assert np.linalg.norm( (x - y).toarray() ) < tol


@pytest.mark.parametrize(("solver", "use_jacobi_pc"),
    [('CG'      , False), ('CG', True),
     ('BiCG'    , False),
     ('BiCGSTAB', False), ('BiCGSTAB', True),
     ('MINRES'  , False),
     ('LSMR'    , False),
     ('GMRES'   , False)]
 )
def test_solvers_matrix_free(solver, use_jacobi_pc):
    print(f'solver = {solver}')
    n1 = 4
    n2 = 3
    p1 = 5
    p2 = 2
    P1 = False
    P2 = False
    V = get_StencilVectorSpace([n1, n2], [p1, p2], [P1, P2])
    A_SM = get_positive_definite_StencilMatrix(V)
    assert_pos_def(A_SM)
    AT_SM = A_SM.transpose()
    A = MatrixFreeLinearOperator(domain=V, codomain=V, dot=lambda v: A_SM @ v, dot_transpose=lambda v: AT_SM @ v)

    # get rhs and solution
    b = get_random_StencilVector(V)
    x = A.dot(b)

    # Create Inverse with A
    tol = 1e-6
    pc = A_SM.diagonal(inverse=True) if use_jacobi_pc else None
    A_inv = inverse(A, solver, pc=pc, tol=tol)  

    AA = A_inv._A
    xx = AA.dot(b)
    print(f'norm(xx) = {np.linalg.norm( xx.toarray() )}')
    print(f'norm(x)  = {np.linalg.norm( x.toarray() )}')

    # Apply inverse and check
    y = A_inv @ x
    error = np.linalg.norm( (b - y).toarray())
    assert np.linalg.norm( (b - y).toarray() ) < tol

#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
