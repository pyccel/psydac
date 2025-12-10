# -*- coding: UTF-8 -*-

import pytest
import time
import numpy as np
from mpi4py                     import MPI
from scipy.sparse               import csc_matrix, dia_matrix, kron
from scipy.sparse.linalg        import splu

from psydac.ddm.cart               import CartDecomposition
from psydac.linalg.stencil         import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.kron            import kronecker_solve_3d_par
from psydac.linalg.direct_solvers  import SparseSolver, BandedSolver

# ... return X, solution of (A1 kron A2 kron A3)X = Y
def kron_solve_seq_ref(A1, A2, A3, Y):

    # ...
    A1_csr = A1.tosparse().tocsr()
    A2_csr = A2.tosparse().tocsr()
    A3_csr = A3.tosparse().tocsr()
    C = csc_matrix(kron(kron(A1_csr, A2_csr), A3_csr))

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
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [4,8] )
@pytest.mark.parametrize( 'n3', [4] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'p2', [2] )
@pytest.mark.parametrize( 'p3', [2] )
@pytest.mark.parallel
def test_kron_solver_3d_band_par( n1, n2, n3, p1, p2, p3, P1=False, P2=False, P3=False ):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ... 2D MPI cart
    cart = CartDecomposition(
        npts    = [n1, n2, n3],
        pads    = [p1, p2, p3],
        periods = [P1, P2, P3],
        reorder = True,
        comm    = comm
    )

    # ...
    sizes1 = cart.global_ends[0] - cart.global_starts[0] + 1
    sizes2 = cart.global_ends[1] - cart.global_starts[1] + 1
    sizes3 = cart.global_ends[2] - cart.global_starts[2] + 1

    disps = cart.global_starts[0]*n2*n3 + cart.global_starts[1]*n3 + cart.global_starts[2]
    sizes = sizes1*sizes2*sizes3
    # ...

    # ... Vector Spaces
    V = StencilVectorSpace(cart)

    [s1, s2, s3] = V.starts
    [e1, e2, e3] = V.ends


   # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    V1 = StencilVectorSpace([n1], [p1], [P1])
    V2 = StencilVectorSpace([n2], [p2], [P2])
    V3 = StencilVectorSpace([n3], [p3], [P3])

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

    A3 = StencilMatrix(V3, V3)
    A3[:,-p3:0   ] = -2
    A3[:, 0 :1   ] = 3*p2
    A3[:, 1 :p3+1] = -2
    A3.remove_spurious_entries()
    A3_bnd, la3, ua3 = to_bnd(A3)
    solver_3 = BandedSolver(ua3, la3, A3_bnd)

    #  ... RHS
    Y = StencilVector(V)
    Y_glob = np.array([[[(i1+1)*100+(i2+1)*10 +(i3+1) for i3 in range(n3)] for i2 in range(n2)] for i1 in range(n1)])
    Y[s1:e1+1, s2:e2+1, s3:e3+1] = Y_glob[s1:e1+1, s2:e2+1, s3:e3+1]
    Y.update_ghost_regions()

    # ...
    X_glob = kron_solve_seq_ref(A1, A2, A3, Y_glob)
    X = kronecker_solve_3d_par(solver_1, solver_2, solver_3, Y)

    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print('X_glob  = \n', X_glob)
            print('X  = \n', X.toarray().reshape(n1, n2, n3))
            print('', flush=True)
            time.sleep(0.1)
        comm.Barrier()
    # ...

    # ... Check data
    assert np.allclose( X[s1:e1+1, s2:e2+1, s3:e3+1], X_glob[s1:e1+1, s2:e2+1, s3:e3+1], rtol=1e-8, atol=1e-8 )
#===============================================================================

#===============================================================================
@pytest.mark.parametrize( 'n1', [8,16] )
@pytest.mark.parametrize( 'n2', [4,8] )
@pytest.mark.parametrize( 'n3', [4] )
@pytest.mark.parametrize( 'p1', [1, 2, 3] )
@pytest.mark.parametrize( 'p2', [2] )
@pytest.mark.parametrize( 'p3', [2] )
@pytest.mark.parallel
def test_kron_solver_3d_sparse_par( n1, n2, n3, p1, p2, p3, P1=False, P2=False, P3=False ):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ... 2D MPI cart
    cart = CartDecomposition(
        npts    = [n1, n2, n3],
        pads    = [p1, p2, p3],
        periods = [P1, P2, P3],
        reorder = True,
        comm    = comm
    )

    # ...
    sizes1 = cart.global_ends[0] - cart.global_starts[0] + 1
    sizes2 = cart.global_ends[1] - cart.global_starts[1] + 1
    sizes3 = cart.global_ends[2] - cart.global_starts[2] + 1

    disps = cart.global_starts[0]*n2*n3 + cart.global_starts[1]*n3 + cart.global_starts[2]
    sizes = sizes1*sizes2*sizes3
    # ...

    # ... Vector Spaces
    V = StencilVectorSpace(cart)

    [s1, s2, s3] = V.starts
    [e1, e2, e3] = V.ends


   # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    V1 = StencilVectorSpace([n1], [p1], [P1])
    V2 = StencilVectorSpace([n2], [p2], [P2])
    V3 = StencilVectorSpace([n3], [p3], [P3])

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

    A3 = StencilMatrix(V3, V3)
    A3[:,-p3:0   ] = -2
    A3[:, 0 :1   ] = 3*p2
    A3[:, 1 :p3+1] = -2
    A3.remove_spurious_entries()
    solver_3 = SparseSolver(A3.tosparse())

    #  ... RHS
    Y = StencilVector(V)
    Y_glob = np.array([[[(i1+1)*100+(i2+1)*10 +(i3+1) for i3 in range(n3)] for i2 in range(n2)] for i1 in range(n1)])
    Y[s1:e1+1, s2:e2+1, s3:e3+1] = Y_glob[s1:e1+1, s2:e2+1, s3:e3+1]
    Y.update_ghost_regions()

    # ...
    X_glob = kron_solve_seq_ref(A1, A2, A3, Y_glob)
    X = kronecker_solve_3d_par(solver_1, solver_2, solver_3, Y)

    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print('X_glob  = \n', X_glob)
            print('X  = \n', X.toarray().reshape(n1, n2, n3))
            print('', flush=True)
            time.sleep(0.1)
        comm.Barrier()
    # ...

    # ... Check data
    assert np.allclose( X[s1:e1+1, s2:e2+1, s3:e3+1], X_glob[s1:e1+1, s2:e2+1, s3:e3+1], rtol=1e-8, atol=1e-8 )
#===============================================================================


#===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================
if __name__ == "__main__":
    import sys
#    pytest.main( sys.argv )

    test_kron_solver_3d_band_par( 8, 4, 4, 2, 1, 1)
