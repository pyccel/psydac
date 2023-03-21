# -*- coding: UTF-8 -*-

import pytest
import time
import numpy as np
from mpi4py                     import MPI
from psydac.ddm.cart               import DomainDecomposition, CartDecomposition
from scipy.sparse               import csc_matrix, dia_matrix, kron
from scipy.sparse.linalg        import splu
from psydac.linalg.stencil         import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.kron            import KroneckerLinearSolver
from psydac.linalg.direct_solvers  import SparseSolver, BandedSolver

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

    dmat = dia_matrix(A.toarray(), dtype=A.dtype)
    la   = abs(dmat.offsets.min())
    ua   = dmat.offsets.max()
    cmat = dmat.tocsr()

    A_bnd = np.zeros((1+ua+2*la, cmat.shape[1]), A.dtype)

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
    dtype = space.dtype

    # for now, take matrices like this (as in the other tests)
    if dtype==complex:
        factor=1j
    else:
        factor=1
    A[:,-p:0   ] = 1
    A[:, 0 :1   ] = (seed*factor+10)*p
    A[:, 1 :p+1] = -1

    return A

def random_vectordata(seed, npts, dtype):
    #define our factor to choose our data type
    if dtype==complex:
        factor=1j
    else:
        factor=1
    # for now, take vectors like this (as in the other tests)
    return np.fromfunction(lambda *point: sum([10**i*d+factor*seed for i,d in enumerate(point)]), npts, dtype=dtype)

def compare_solve(seed, comm, npts, pads, periods, direct_solver, dtype=float, transposed=False, verbose=False):
    if comm is None:
        rank = -1
    else:
        rank = comm.Get_rank()

    if verbose:
        print(f'[{rank}] Test start', flush=True)

    # vector spaces
    comm = MPI.COMM_WORLD
    D = DomainDecomposition(npts, periods=periods, comm=comm)

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=pads, shifts=[1]*len(pads))

    V = StencilVectorSpace(cart, dtype=dtype)

    Ds    = [DomainDecomposition([n], periods=[P]) for n,P in zip(npts, periods)]
    carts = [CartDecomposition(Di, [n],  *compute_global_starts_ends(Di, [n]), pads=[p], shifts=[1]) for Di,n,p in zip(Ds, npts, pads)]
    Vs = [StencilVectorSpace(carti, dtype=dtype) for carti in carts]
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
    Y_glob = random_vectordata(seed, npts, dtype=dtype)
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

#===============================================================================
# tests of the direct solvers
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2, 10] )
@pytest.mark.parametrize( 'n', [8, 64] )
@pytest.mark.parametrize( 'p', [1, 3] )
@pytest.mark.parametrize( 'P', [True, False] )
@pytest.mark.parametrize( 'nrhs', [1, 3] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
@pytest.mark.parametrize( 'transposed', [True, False] )
def test_direct_solvers(dtype, seed, n, p, P, nrhs, direct_solver, transposed):

    D    = DomainDecomposition([n], periods=[P])
    cart = CartDecomposition(D, [n],  *compute_global_starts_ends(D, [n]), pads=[p], shifts=[1])

    # space (V)
    V = StencilVectorSpace( cart, dtype=dtype )

    # bulid matrices (A)
    A = random_matrix(seed+1, V)
    solver = direct_solver(A)

    # vector to solve for (Y)
    Y_glob = np.stack([random_vectordata(seed + i, [n], dtype) for i in range(nrhs)], axis=0)

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

#===============================================================================
# tests without MPI
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'params', [([8], [2], [False]), ([8,9], [2,3], [False,True])] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
def test_kron_solver_nompi(seed, params, direct_solver, dtype):
    compare_solve(seed, None, params[0], params[1], params[2], direct_solver, dtype=dtype, transposed=False, verbose=False)


#===============================================================================
# SERIAL TESTS
#===============================================================================

# low-dimensional tests

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [8, 17] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
def test_kron_solver_1d_ser(dtype, seed, n1, p1, P1, direct_solver):
    compare_solve(seed, MPI.COMM_SELF, [n1], [p1], [P1], direct_solver, dtype=dtype, transposed=False, verbose=False)
#===============================================================================


@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [5, 17] )
@pytest.mark.parametrize( 'n2', [4, 9] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
def test_kron_solver_2d_ser(dtype, seed, n1, n2, p1, p2, P1, P2, direct_solver):
    compare_solve(seed, MPI.COMM_SELF, [n1,n2], [p1,p2], [P1,P2], direct_solver, dtype=dtype, transposed=False, verbose=False)
#===============================================================================

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [5, 17] )
@pytest.mark.parametrize( 'n2', [4, 9] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
def test_kron_solver_2d_transposed_ser(seed, n1, n2, p1, p2, P1, P2, direct_solver, dtype):
    compare_solve(seed, MPI.COMM_SELF, [n1,n2], [p1,p2], [P1,P2], direct_solver, dtype=dtype, transposed=True, verbose=False)
#===============================================================================

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [5, 17] )
@pytest.mark.parametrize( 'n2', [4, 9] )
@pytest.mark.parametrize( 'n3', [4, 5] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'p3', [1, 2] )
def test_kron_solver_3d_ser(seed, n1, n2, n3, p1, p2, p3, dtype, P1=False, P2=True, P3=False, direct_solver=matrix_to_sparse):
    compare_solve(seed, MPI.COMM_SELF, [n1,n2,n3], [p1,p2,p3], [P1,P2,P3], direct_solver, dtype=dtype, transposed=False, verbose=False)
#===============================================================================

# higher-dimensional tests

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'dim', [4, 10] )
def test_kron_solver_nd_ser(seed, dim, dtype):
    if dim < 5:
        npts_base = 4
    else:
        npts_base = 2
    compare_solve(seed, MPI.COMM_SELF, [npts_base]*dim, [1]*dim, [False]*dim, matrix_to_sparse,dtype=dtype, transposed=False, verbose=False)
#===============================================================================
# PARALLEL TESTS
#===============================================================================

# low-dimensional tests
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [8, 17] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
@pytest.mark.parallel
def test_kron_solver_1d_par(seed, n1, p1, P1, direct_solver, dtype):
    # we take n1*p1 here to prevent MPI topology problems
    compare_solve(seed, MPI.COMM_WORLD, [n1*p1], [p1], [P1], direct_solver, dtype=dtype, transposed=False, verbose=False)
#===============================================================================

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [8, 17] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
@pytest.mark.parallel
def test_kron_solver_2d_par(seed, n1, n2, p1, p2, P1, P2, direct_solver, dtype):
    compare_solve(seed, MPI.COMM_WORLD, [n1,n2], [p1,p2], [P1,P2], direct_solver, dtype=dtype, transposed=False, verbose=False)
#===============================================================================

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [8, 17] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'P1', [True, False] )
@pytest.mark.parametrize( 'P2', [True, False] )
@pytest.mark.parametrize( 'direct_solver', [matrix_to_bandsolver, matrix_to_sparse] )
@pytest.mark.parallel
def test_kron_solver_2d_transposed_par(seed, n1, n2, p1, p2, P1, P2, direct_solver, dtype):
    compare_solve(seed, MPI.COMM_WORLD, [n1,n2], [p1,p2], [P1,P2], direct_solver, dtype=dtype, transposed=True, verbose=False)
#===============================================================================

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'n1', [6, 17] )
@pytest.mark.parametrize( 'n2', [8, 12] )
@pytest.mark.parametrize( 'n3', [8, 12] )
@pytest.mark.parametrize( 'p1', [1, 2] )
@pytest.mark.parametrize( 'p2', [1, 2] )
@pytest.mark.parametrize( 'p3', [1, 2] )
@pytest.mark.parallel
def test_kron_solver_3d_par(seed, n1, n2, n3, p1, p2, p3, dtype, P1=False, P2=True, P3=False, direct_solver=matrix_to_sparse):
    compare_solve(seed, MPI.COMM_WORLD, [n1,n2,n3], [p1,p2,p3], [P1,P2,P3], direct_solver, dtype=dtype, transposed=False, verbose=False)
#===============================================================================

# higher-dimensional tests

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'dim', [4, 6] )
@pytest.mark.parallel
def test_kron_solver_nd_par(seed, dim, dtype):
    # for now, avoid too high dim's, since we solve the matrix completely on each rank as well...

    npts_base = 4
    compare_solve(seed, MPI.COMM_WORLD, [npts_base]*dim, [1]*dim, [False]*dim, matrix_to_sparse, dtype=dtype, transposed=False, verbose=False)
#===============================================================================

if __name__ == '__main__':
    # showcase testcase
    compare_solve(0, MPI.COMM_WORLD, [4,4,5], [1,2,3], [False,True,False], matrix_to_bandsolver, dtype=[float, complex], transposed=False, verbose=True)
    #compare_solve(0, MPI.COMM_WORLD, [2]*10, [1]*10, [False]*10, matrix_to_sparse, verbose=True)
