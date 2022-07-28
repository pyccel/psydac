# -*- coding: UTF-8 -*-

import pytest
import time
import numpy as np
from mpi4py                     import MPI
from psydac.ddm.cart               import CartDecomposition
from scipy.sparse               import csc_matrix, dia_matrix, kron
from scipy.sparse.linalg        import splu
from psydac.linalg.stencil         import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.kron            import KroneckerLinearSolver
from psydac.linalg.direct_solvers  import SparseSolver, BandedSolver

# ... solve AX==Y on the conventional way, where A=\bigotimes_i A_i
def kron_solve_seq_ref(Y, A, transposed):

    # ...
    assert len(A) > 0
    preC = A[0].tosparse().tocsr()
    for i in range(1, len(A)):
        preC = kron(preC, A[i].tosparse().tocsr())
    if transposed:
        preC = preC.T
    C = csc_matrix(preC)

    C_op  = splu(C)
    X = C_op.solve(Y.flatten())

    return X.reshape(Y.shape)
# ...

# ... convert a 1D stencil matrix to band matrix
def to_bnd(A):

    dmat = dia_matrix(A.toarray())
    la   = abs(dmat.offsets.min())
    ua   = dmat.offsets.max()
    cmat = dmat.tocsr()

    A_bnd = np.zeros((1+ua+2*la, cmat.shape[1]))

    for i,j in zip(*cmat.nonzero()):
        A_bnd[la+ua+i-j, j] = cmat[i,j]

    return A_bnd, la, ua
# ...

def matrix_to_bandsolver(A):
    A.remove_spurious_entries()
    A_bnd, la, ua = to_bnd(A)
    return BandedSolver(ua, la, A_bnd)

def matrix_to_sparse(A):
    A.remove_spurious_entries()
    return SparseSolver(A.tosparse())

def random_matrix(seed, space):
    A = StencilMatrix(space, space)
    p = space.pads[0]

    # for now, take matrices like this (as in the other tests)
    A[:,-p:0   ] = 1
    A[:, 0 :1   ] = (seed+10)*p
    A[:, 1 :p+1] = -1

    return A

def random_vectordata(seed, npts):
    # for now, take vectors like this (as in the other tests)
    return np.fromfunction(lambda *point: sum([10**i*d+seed for i,d in enumerate(point)]), npts)

def compare_solve(seed, comm, npts, pads, periods, direct_solver, transposed=False, verbose=False):
    if comm is None:
        rank = -1
    else:
        rank = comm.Get_rank()

    if verbose:
        print(f'[{rank}] Test start', flush=True)

    # vector spaces
    if comm is None:
        V = StencilVectorSpace(npts, pads, periods)
    else:
        cart = CartDecomposition(
            npts    = npts,
            pads    = pads,
            periods = periods,
            reorder = True,
            comm    = comm
        )
        V = StencilVectorSpace(cart)
    Vs = [StencilVectorSpace([n], [p], [P]) for n,p,P in zip(npts, pads, periods)]
    localslice = tuple([slice(s, e+1) for s, e in zip(V.starts, V.ends)])

    if verbose:
        print(f'[{rank}] Vector spaces built', flush=True)

    # bulid matrices (A)
    A = [random_matrix(seed+i+1, Vi) for i,Vi in enumerate(Vs)]
    solvers = [direct_solver(Ai) for Ai in A]

    if verbose:
        print(f'[{rank}] Matrices built', flush=True)

    # vector to solve for (Y)
    Y = StencilVector(V)
    Y_glob = random_vectordata(seed, npts)
    Y[localslice] = Y_glob[localslice]
    Y.update_ghost_regions()

    if verbose:
        print(f'[{rank}] RHS vector built', flush=True)

    # solve in two different ways
    X_glob = kron_solve_seq_ref(Y_glob, A, transposed)
    Xout = StencilVector(V)

    X = KroneckerLinearSolver(V, solvers).solve(Y, out=Xout, transposed=transposed)
    assert X is Xout

    if verbose:
        print(f'[{rank}] Systems solved', flush=True)

    # debug output
    if verbose and comm is not None:
        for i in range(comm.Get_size()):
            if rank == i:
                print(f'[{rank}] Output for rank {rank}')
                print(f'[{rank}] X_glob  = {X_glob}')
                print(f'[{rank}] X  = {X.toarray().reshape(npts)}', )
                print(f'[{rank}]', flush=True)
            comm.Barrier()

    # compare for equality
    assert np.allclose( X[localslice], X_glob[localslice], rtol=1e-8, atol=1e-8 )

# tests of the direct solvers

@pytest.mark.parametrize( 'seed', [0,2,10] )
@pytest.mark.parametrize( 'n', [8, 16, 17, 64] )
@pytest.mark.parametrize( 'p', [1, 3] )
@pytest.mark.parametrize( 'P', [True, False] )
@pytest.mark.parametrize( 'nrhs', [1,3] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
@pytest.mark.parametrize( 'transposed', [True, False] )
def test_direct_solvers(seed, n, p, P, nrhs, direct_solver, transposed):
    # space (V)
    V = StencilVectorSpace([n], [p], [P])

    # bulid matrices (A)
    A = random_matrix(seed+1, V)
    solver = direct_solver(A)

    # vector to solve for (Y)
    Y_glob = np.stack([random_vectordata(seed + i, [n]) for i in range(nrhs)], axis=0)

    # ref solve
    preC = A.tosparse().tocsc()
    if transposed:
        preC = preC.T
    C = csc_matrix(preC)

    C_op  = splu(C)

    X_glob = C_op.solve(Y_glob.T).T

    # new vector allocation
    X_glob2 = solver.solve(Y_glob, transposed=transposed)

    # solve with out vector
    X_glob3 = Y_glob.copy()
    X_glob4 = solver.solve(Y_glob, out=X_glob3, transposed=transposed)

    # solve in-place
    X_glob5 = Y_glob.copy()
    X_glob6 = solver.solve(X_glob5, out=X_glob5, transposed=transposed)

    # compare results
    assert X_glob4 is X_glob3
    assert X_glob6 is X_glob5

    assert np.allclose( X_glob, X_glob2, rtol=1e-8, atol=1e-8 )
    assert np.allclose( X_glob, X_glob3, rtol=1e-8, atol=1e-8 )
    assert np.allclose( X_glob, X_glob5, rtol=1e-8, atol=1e-8 )

# right now, the maximum tested number for MPI_COMM_WORLD.size is 4; some test sizes failed with size 8 for now.

# tests without MPI

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'params', [([8], [2], [False]), ([8,9], [2,3], [False,True])] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
def test_kron_solver_nompi(seed, params, direct_solver):
    compare_solve(seed, None, params[0], params[1], params[2], direct_solver, transposed=False, verbose=False)

# low-dimensional tests

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [8, 9, 16, 17] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
def test_kron_solver_1d_ser(seed, n1, p1, P1, direct_solver):
    compare_solve(seed, MPI.COMM_SELF, [n1], [p1], [P1], direct_solver, transposed=False, verbose=False)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [8, 9, 16, 17] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
@pytest.mark.parallel
def test_kron_solver_1d_par(seed, n1, p1, P1, direct_solver):
    # we take n1*p1 here to prevent MPI topology problems
    compare_solve(seed, MPI.COMM_WORLD, [n1*p1], [p1], [P1], direct_solver, transposed=False, verbose=False)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [5, 8, 16, 17] )
@pytest.mark.parametrize( 'n2', [4, 9] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
def test_kron_solver_2d_ser(seed, n1, n2, p1, p2, P1, P2, direct_solver):
    compare_solve(seed, MPI.COMM_SELF, [n1,n2], [p1,p2], [P1,P2], direct_solver, transposed=False, verbose=False)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [8, 12, 16, 17] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
@pytest.mark.parallel
def test_kron_solver_2d_par(seed, n1, n2, p1, p2, P1, P2, direct_solver):
    compare_solve(seed, MPI.COMM_WORLD, [n1,n2], [p1,p2], [P1,P2], direct_solver, transposed=False, verbose=False)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [5, 8, 16, 17] )
@pytest.mark.parametrize( 'n2', [4, 9] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
def test_kron_solver_2d_transposed_ser(seed, n1, n2, p1, p2, P1, P2, direct_solver):
    compare_solve(seed, MPI.COMM_SELF, [n1,n2], [p1,p2], [P1,P2], direct_solver, transposed=True, verbose=False)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [8, 16, 17] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
@pytest.mark.parallel
def test_kron_solver_2d_transposed_par(seed, n1, n2, p1, p2, P1, P2, direct_solver):
    compare_solve(seed, MPI.COMM_WORLD, [n1,n2], [p1,p2], [P1,P2], direct_solver, transposed=True, verbose=False)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [5, 8, 16, 17] )
@pytest.mark.parametrize( 'n2', [4, 9] )
@pytest.mark.parametrize( 'n3', [4, 5] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'p3', [1, 2] )
def test_kron_solver_3d_ser(seed, n1, n2, n3, p1, p2, p3, P1=False, P2=True, P3=False, direct_solver=matrix_to_sparse):
    compare_solve(seed, MPI.COMM_SELF, [n1,n2,n3], [p1,p2,p3], [P1,P2,P3], direct_solver, transposed=False, verbose=False)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [6, 8, 16, 17] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'n3', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'p3', [1, 2] )
@pytest.mark.parallel
def test_kron_solver_3d_par(seed, n1, n2, n3, p1, p2, p3, P1=False, P2=True, P3=False, direct_solver=matrix_to_sparse):
    compare_solve(seed, MPI.COMM_WORLD, [n1,n2,n3], [p1,p2,p3], [P1,P2,P3], direct_solver, transposed=False, verbose=False)

# higher-dimensional tests

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'dim', [4,7,10] )
def test_kron_solver_nd_ser(seed, dim):
    if dim < 5:
        npts_base = 4
    else:
        npts_base = 2
    compare_solve(seed, MPI.COMM_SELF, [npts_base]*dim, [1]*dim, [False]*dim, matrix_to_sparse, transposed=False, verbose=False)

@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'dim', [4,5,6] )
@pytest.mark.parallel
def test_kron_solver_nd_par(seed, dim):
    # for now, avoid too high dim's, since we solve the matrix completely on each rank as well...

    npts_base = 4
    compare_solve(seed, MPI.COMM_WORLD, [npts_base]*dim, [1]*dim, [False]*dim, matrix_to_sparse, transposed=False, verbose=False)

if __name__ == '__main__':
    # showcase testcase
    compare_solve(0, MPI.COMM_WORLD, [4,4,5], [1,2,3], [False,True,False], matrix_to_bandsolver, transposed=False, verbose=True)
    #compare_solve(0, MPI.COMM_WORLD, [2]*10, [1]*10, [False]*10, matrix_to_sparse, verbose=True)
