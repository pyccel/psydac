# -*- coding: UTF-8 -*-

import pytest
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
import time
from scipy.sparse import csc_matrix, kron
from scipy.sparse.linalg import splu
from spl.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from spl.linalg.kron    import KroneckerStencilMatrix_2D, kronecker_solve_2d_par
from spl.linalg.direct_solvers import SparseSolver

# ... return X, solution of (A1 kron A2)X = Y
def kron_solve_seq_ref(A1, A2, Y):

    # ...
    A1_csr = A1.tocsr()
    A2_csr = A2.tocsr()
    C = csc_matrix(kron(A1_csr, A2_csr))

    C_op  = splu(C)
    X = C_op.solve(Y.flatten())

    return X.reshape(Y.shape)
# ...

#===============================================================================
@pytest.mark.parametrize( 'n1', [7,15] )
@pytest.mark.parametrize( 'n2', [8,12] )
@pytest.mark.parametrize( 'p1', [1,2,3] )
@pytest.mark.parametrize( 'p2', [1,2,3] )
@pytest.mark.parallel
def test_kron_solver_2d_par( n1, n2, p1, p2, P1=True, P2=False ):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ... 2D MPI cart
    cart = Cart(npts = [n1, n2], pads = [p1, p2], periods = [False, False],\
                reorder = True, comm = comm)

    # ...
    sizes1 = cart.global_ends[0] - cart.global_starts[0] + 1
    sizes2 = cart.global_ends[1] - cart.global_starts[1] + 1

    disps = cart.global_starts[0]*n2 + cart.global_starts[1]
    sizes = sizes1*sizes2
    # ...

    # ... Vector Spaces
    V = StencilVectorSpace(cart)

    [s1, s2] = V.starts
    [e1, e2] = V.ends


   # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    V1 = StencilVectorSpace([n1], [p1], [False])
    V2 = StencilVectorSpace([n2], [p2], [False])

    # ... Matrices and Direct solvers
    A1 = StencilMatrix(V1, V1)
    A1[:,-p1:0   ] = -4
    A1[:, 0 :1   ] = 10*p1
    A1[:, 1 :p1+1] = -4
    A1.remove_spurious_entries()
    solver_1 = SparseSolver(A1.tocsr())

    A2 = StencilMatrix(V2, V2)
    A2[:,-p2:0   ] = -1
    A2[:, 0 :1   ] = 2*p2
    A2[:, 1 :p2+1] = -1
    A2.remove_spurious_entries()
    solver_2 = SparseSolver(A2.tocsr())

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
