# -*- coding: UTF-8 -*-

import pytest
import time
import numpy as np
from mpi4py                     import MPI
from psydac.ddm.cart               import CartDecomposition
from scipy.sparse               import csc_matrix, dia_matrix, kron
from scipy.sparse.linalg        import splu
from psydac.linalg.stencil         import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.kron            import kronecker_solve_2d_par
from psydac.linalg.direct_solvers  import SparseSolver, BandedSolver

# ... return X, solution of (A1 kron A2)X = Y
def kron_solve_seq_ref(A1, A2, Y):

    # ...
    A1_csr = A1.tosparse().tocsr()
    A2_csr = A2.tosparse().tocsr()
    C = csc_matrix(kron(A1_csr, A2_csr))

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

#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parallel
def test_kron_solver_2d_band_par( n1, n2, p1, p2, P1=False, P2=False ):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ... 2D MPI cart
    cart = CartDecomposition(
        npts    = [n1, n2],
        pads    = [p1, p2],
        periods = [P1, P2],
        reorder = True,
        comm    = comm
    )

    # ... Vector Spaces
    V = StencilVectorSpace(cart)

    [s1, s2] = V.starts
    [e1, e2] = V.ends


   # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    V1 = StencilVectorSpace([n1], [p1], [P1])
    V2 = StencilVectorSpace([n2], [p2], [P2])

    # ... Matrices and Direct solvers
    A1 = StencilMatrix(V1, V1)
    A1[:,-p1:0   ] = -4
    A1[:, 0 :1   ] = 10*p1
    A1[:, 1 :p1+1] = -4
    A1.remove_spurious_entries()
    A1_bnd, la1, ua1 = to_bnd(A1)
    solver_1 = BandedSolver(ua1, la1, A1_bnd)

    A2 = StencilMatrix(V2, V2)
    A2[:,-p2:0   ] = -1
    A2[:, 0 :1   ] = 2*p2
    A2[:, 1 :p2+1] = -1
    A2.remove_spurious_entries()
    A2_bnd, la2, ua2 = to_bnd(A2)
    solver_2 = BandedSolver(ua2, la2, A2_bnd)

    #  ... RHS
    Y = StencilVector(V)
    Y_glob = np.array([[(i1+1)*10+(i2+1) for i2 in range(n2)] for i1 in range(n1)])
    Y[s1:e1+1, s2:e2+1] = Y_glob[s1:e1+1, s2:e2+1]
    Y.update_ghost_regions()

    # ...
    X_glob = kron_solve_seq_ref(A1, A2, Y_glob)

    X = kronecker_solve_2d_par(solver_1, solver_2, Y)

    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print('X_glob  = \n', X_glob)
            print('X  = \n', X.toarray().reshape(n1, n2))
            print('', flush=True)
            time.sleep(0.1)
        comm.Barrier()
    # ...

    # ... Check data
    #assert np.allclose( X[s1:e1+1, s2:e2+1], X_glob[s1:e1+1, s2:e2+1], rtol=1e-13, atol=1e-13 )
#===============================================================================

#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parallel
def test_kron_solver_3d_sparse_par( n1, n2, p1, p2, P1=False, P2=False ):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ... 2D MPI cart
    cart = CartDecomposition(
        npts    = [n1, n2],
        pads    = [p1, p2],
        periods = [P1, P2],
        reorder = True,
        comm    = comm
    )

    # ... Vector Spaces
    V = StencilVectorSpace(cart)

    [s1, s2] = V.starts
    [e1, e2] = V.ends


   # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    V1 = StencilVectorSpace([n1], [p1], [P1])
    V2 = StencilVectorSpace([n2], [p2], [P2])

    # ... Matrices and Direct solvers
    A1 = StencilMatrix(V1, V1)
    A1[:,-p1:0   ] = -4
    A1[:, 0 :1   ] = 10*p1
    A1[:, 1 :p1+1] = -4
    A1.remove_spurious_entries()
    solver_1 = SparseSolver(A1.tosparse())

    A2 = StencilMatrix(V2, V2)
    A2[:,-p2:0   ] = -1
    A2[:, 0 :1   ] = 2*p2
    A2[:, 1 :p2+1] = -1
    A2.remove_spurious_entries()
    solver_2 = SparseSolver(A2.tosparse())

    #  ... RHS
    Y = StencilVector(V)
    Y_glob = np.array([[(i1+1)*10+(i2+1) for i2 in range(n2)] for i1 in range(n1)])
    Y[s1:e1+1, s2:e2+1] = Y_glob[s1:e1+1, s2:e2+1]
    Y.update_ghost_regions()

    # ...
    X_glob = kron_solve_seq_ref(A1, A2, Y_glob)
    X = kronecker_solve_2d_par(solver_1, solver_2, Y)

    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print('X_glob  = \n', X_glob)
            print('X  = \n', X.toarray().reshape(n1, n2))
            print('', flush=True)
            time.sleep(0.1)
        comm.Barrier()
    # ...

    # ... Check data
    assert np.allclose( X[s1:e1+1, s2:e2+1], X_glob[s1:e1+1, s2:e2+1], rtol=1e-13, atol=1e-13 )
#===============================================================================


#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
