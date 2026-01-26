#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import time

import pytest
import numpy as np
from mpi4py              import MPI
from scipy.sparse        import csc_matrix, kron
from scipy.sparse.linalg import splu

from sympde.calculus import dot
from sympde.expr     import BilinearForm, integral
from sympde.topology import Line
from sympde.topology import Cube, Square
from sympde.topology import Derham
from sympde.topology import elements_of

from psydac.api.discretization     import discretize
from psydac.ddm.cart               import DomainDecomposition, CartDecomposition
from psydac.linalg.block           import BlockLinearOperator
from psydac.linalg.direct_solvers  import SparseSolver, BandedSolver
from psydac.linalg.kron            import KroneckerLinearSolver
from psydac.linalg.solvers         import inverse
from psydac.linalg.stencil         import StencilVectorSpace, StencilVector, StencilMatrix

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

def matrix_to_sparse(A):
    A.remove_spurious_entries()
    return SparseSolver(A.tosparse())

def random_matrix(seed, domain, codomain):
    A = StencilMatrix(domain, codomain)
    p = domain.pads[0] # by definition of the StencilMatrix, domain.pads == codomain.pads
    dtype = domain.dtype # by definition of the StencilMatrix, domain.dtype == codomain.dtype

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
    D2 = DomainDecomposition(npts, periods=periods, comm=comm)

    # Partition the points
    global_starts, global_ends = compute_global_starts_ends(D, npts)
    global_starts2, global_ends2 = compute_global_starts_ends(D2, npts)

    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=pads, shifts=[1]*len(pads))
    cart2 = CartDecomposition(D2, npts, global_starts2, global_ends2, pads=pads, shifts=[1]*len(pads))

    V = StencilVectorSpace(cart, dtype=dtype)
    W = StencilVectorSpace(cart2, dtype=dtype)

    Ds    = [DomainDecomposition([n], periods=[P]) for n,P in zip(npts, periods)]
    carts = [CartDecomposition(Di, [n],  *compute_global_starts_ends(Di, [n]), pads=[p], shifts=[1]) for Di,n,p in zip(Ds, npts, pads)]
    Vs = [StencilVectorSpace(carti, dtype=dtype) for carti in carts]
    Ds2    = [DomainDecomposition([n], periods=[P]) for n,P in zip(npts, periods)]
    carts2 = [CartDecomposition(Di, [n],  *compute_global_starts_ends(Di, [n]), pads=[p], shifts=[1]) for Di,n,p in zip(Ds2, npts, pads)]
    Ws = [StencilVectorSpace(carti, dtype=dtype) for carti in carts2]
    localslice = tuple([slice(s, e+1) for s, e in zip(V.starts, V.ends)])

    if verbose:
        print(f'[{rank}] Vector spaces built', flush=True)

    # bulid matrices (A)
    A = [random_matrix(seed+i+1, Vi, Wi) for i,(Vi, Wi) in enumerate(zip(Vs, Ws))]
    solvers = [direct_solver(Ai) for Ai in A]

    if verbose:
        print(f'[{rank}] Matrices built', flush=True)

    # vector to solve for (Y)
    if transposed:
        Y = StencilVector(W)
    else:
        Y = StencilVector(V)
    Y_glob = random_vectordata(seed, npts, dtype=dtype)
    Y[localslice] = Y_glob[localslice]
    Y.update_ghost_regions()

    if verbose:
        print(f'[{rank}] RHS vector built', flush=True)

    # solve in two different ways
    X_glob = kron_solve_seq_ref(Y_glob, A, transposed)
    if transposed:
        Xout = StencilVector(V)
    else:
        Xout = StencilVector(W)

    solver = KroneckerLinearSolver(V, W, solvers)
    if transposed:
        solver = solver.T
    
    X = solver.solve(Y, out=Xout)
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

def get_M1_block_kron_solver(V1, ncells, degree, periodic):
    """
    Given a 3D DeRham sequenece (V0 = H(grad) --grad--> V1 = H(curl) --curl--> V2 = H(div) --div--> V3 = L2)
    discretized using ncells, degree and periodic,

        domain = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
        derham = Derham(domain)
        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree),

    returns the inverse of the mass matrix M1 as a BlockLinearOperator consisting of three KroneckerLinearSolvers on the diagonal.
    """
    # assert 3D
    assert len(ncells) == 3
    assert len(degree) == 3
    assert len(periodic) == 3

    # 1D domain to be discreticed using the respective values of ncells, degree, periodic
    domain_1d = Line('L', bounds=(0,1))
    derham_1d = Derham(domain_1d)

    # storage for the 1D mass matrices
    M0_matrices = []
    M1_matrices = []

    # assembly of the 1D mass matrices
    for (n, p, P) in zip(ncells, degree, periodic):

        domain_1d_h = discretize(domain_1d, ncells=[n], periodic=[P])
        derham_1d_h = discretize(derham_1d, domain_1d_h, degree=[p])

        u_1d_0, v_1d_0 = elements_of(derham_1d.V0, names='u_1d_0, v_1d_0')
        u_1d_1, v_1d_1 = elements_of(derham_1d.V1, names='u_1d_1, v_1d_1')

        a_1d_0 = BilinearForm((u_1d_0, v_1d_0), integral(domain_1d, u_1d_0 * v_1d_0))
        a_1d_1 = BilinearForm((u_1d_1, v_1d_1), integral(domain_1d, u_1d_1 * v_1d_1))

        a_1d_0_h = discretize(a_1d_0, domain_1d_h, (derham_1d_h.V0, derham_1d_h.V0))
        a_1d_1_h = discretize(a_1d_1, domain_1d_h, (derham_1d_h.V1, derham_1d_h.V1))

        M_1d_0 = a_1d_0_h.assemble()
        M_1d_1 = a_1d_1_h.assemble()

        M0_matrices.append(M_1d_0)
        M1_matrices.append(M_1d_1)

    V1_1 = V1[0]
    V1_2 = V1[1]
    V1_3 = V1[2]

    B1_mat = [M1_matrices[0], M0_matrices[1], M0_matrices[2]]
    B2_mat = [M0_matrices[0], M1_matrices[1], M0_matrices[2]]
    B3_mat = [M0_matrices[0], M0_matrices[1], M1_matrices[2]]

    B1_solvers = [BandedSolver.from_stencil_mat_1d(Ai) for Ai in B1_mat]
    B2_solvers = [BandedSolver.from_stencil_mat_1d(Ai) for Ai in B2_mat]
    B3_solvers = [BandedSolver.from_stencil_mat_1d(Ai) for Ai in B3_mat]

    B1_kron_inv = KroneckerLinearSolver(V1_1, V1_1, B1_solvers)
    B2_kron_inv = KroneckerLinearSolver(V1_2, V1_2, B2_solvers)
    B3_kron_inv = KroneckerLinearSolver(V1_3, V1_3, B3_solvers)

    M1_block_kron_solver = BlockLinearOperator(V1, V1, ((B1_kron_inv, None, None), 
                                                        (None, B2_kron_inv, None), 
                                                        (None, None, B3_kron_inv)))

    return M1_block_kron_solver



def get_inverse_mass_matrices(derham_h, domain_h):
    """
    Given a 2D DeRham sequence (V0 = H(grad) --curl--> V1 = H(div) --div--> V2 = L2)
    and its discrete domain, which shall be rectangular,
    returns the inverse of the mass matrices for all three spaces using kronecker solvers.
    """
    # assert 2D
    # Maybe should add more assert regarding the types and the domain form in case this get used in general context.
    
    V0h   = derham_h.V0.coeff_space
    V1h   = derham_h.V1.coeff_space
    V2h   = derham_h.V2.coeff_space
    
    bounds1 = domain_h.domain.bounds1
    bounds2 = domain_h.domain.bounds2
        
    ncells = domain_h.ncells[domain_h.domain.name]
    degree = derham_h.V0.degree
    periodic = domain_h.periodic[domain_h.domain.name]
    
    assert len(ncells) == 2
    assert len(degree) == 2
    assert len(periodic) == 2

    # 1D domain to be discreticed using the respective values of ncells, degree, periodic
    list_domain_1d = [Line('L1', bounds=bounds1), Line('L2', bounds=bounds2)]
    list_derham_1d = [Derham(domain_1d) for domain_1d in list_domain_1d]

    # storage for the 1D mass matrices
    M0_matrices = []
    M1_matrices = []

    # assembly of the 1D mass matrices
    for (n, p, P, domain_1d, derham_1d) in zip(ncells, degree, periodic, list_domain_1d, list_derham_1d):

        domain_1d_h = discretize(domain_1d, ncells=[n], periodic=[P])
        derham_1d_h = discretize(derham_1d, domain_1d_h, degree=[p])

        u_1d_0, v_1d_0 = elements_of(derham_1d.V0, names='u_1d_0, v_1d_0')
        u_1d_1, v_1d_1 = elements_of(derham_1d.V1, names='u_1d_1, v_1d_1')

        a_1d_0 = BilinearForm((u_1d_0, v_1d_0), integral(domain_1d, u_1d_0 * v_1d_0))
        a_1d_1 = BilinearForm((u_1d_1, v_1d_1), integral(domain_1d, u_1d_1 * v_1d_1))

        a_1d_0_h = discretize(a_1d_0, domain_1d_h, (derham_1d_h.V0, derham_1d_h.V0))
        a_1d_1_h = discretize(a_1d_1, domain_1d_h, (derham_1d_h.V1, derham_1d_h.V1))

        M_1d_0 = a_1d_0_h.assemble()
        M_1d_1 = a_1d_1_h.assemble()

        M0_matrices.append(M_1d_0)
        M1_matrices.append(M_1d_1)

    #V0 H1 space
    
    B_mat_V0 = [M0_matrices[0], M0_matrices[1]]
    
    B_solvers_V0 = [BandedSolver.from_stencil_mat_1d(Ai) for Ai in B_mat_V0]
    
    M0_kron_solver = KroneckerLinearSolver(V0h, V0h, B_solvers_V0)


    #V1 Hdiv space
    V1_1 = V1h[0]
    V1_2 = V1h[1]

    B1_mat_V1 = [M0_matrices[0], M1_matrices[1]]
    B2_mat_V1 = [M1_matrices[0], M0_matrices[1]]

    B1_solvers_V1 = [BandedSolver.from_stencil_mat_1d(Ai) for Ai in B1_mat_V1]
    B2_solvers_V1 = [BandedSolver.from_stencil_mat_1d(Ai) for Ai in B2_mat_V1]

    B1_kron_inv_V1 = KroneckerLinearSolver(V1_1, V1_1, B1_solvers_V1)
    B2_kron_inv_V1 = KroneckerLinearSolver(V1_2, V1_2, B2_solvers_V1)

    M1_block_kron_solver = BlockLinearOperator(V1h, V1h, ((B1_kron_inv_V1, None), 
                                                          (None, B2_kron_inv_V1)))
    
    #V2 L2 space
    
    B_mat_V2 = [M1_matrices[0], M1_matrices[1]]
    
    B_solvers_V2 = [BandedSolver.from_stencil_mat_1d(Ai) for Ai in B_mat_V2]
    
    M2_kron_solver = KroneckerLinearSolver(V2h, V2h, B_solvers_V2)
    
    
    return M0_kron_solver, M1_block_kron_solver, M2_kron_solver

#===============================================================================
# tests of the direct solvers
@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2, 10] )
@pytest.mark.parametrize( 'n', [8, 64] )
@pytest.mark.parametrize( 'p', [1, 3] )
@pytest.mark.parametrize( 'P', [True, False] )
@pytest.mark.parametrize( 'nrhs', [1, 3] )
@pytest.mark.parametrize( 'direct_solver', [BandedSolver.from_stencil_mat_1d, matrix_to_sparse] )
@pytest.mark.parametrize( 'transposed', [True, False] )
def test_direct_solvers(dtype, seed, n, p, P, nrhs, direct_solver, transposed):

    D    = DomainDecomposition([n], periods=[P])
    cart = CartDecomposition(D, [n],  *compute_global_starts_ends(D, [n]), pads=[p], shifts=[1])

    # domain V and codomain W
    V = StencilVectorSpace( cart, dtype=dtype )

    # bulid matrices (A)
    A = random_matrix(seed+1, V, V)
    solver = direct_solver(A)
    if transposed:
        solver = solver.T

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
    X_glob2 = solver.solve(Y_glob)

    # solve with out vector
    X_glob3 = Y_glob.copy()
    X_glob4 = solver.solve(Y_glob, out=X_glob3)

    # solve in-place
    X_glob5 = Y_glob.copy()
    X_glob6 = solver.solve(X_glob5, out=X_glob5)

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
@pytest.mark.parametrize( 'direct_solver', [BandedSolver.from_stencil_mat_1d, matrix_to_sparse] )
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
@pytest.mark.parametrize( 'direct_solver', [BandedSolver.from_stencil_mat_1d, matrix_to_sparse] )
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
@pytest.mark.parametrize( 'direct_solver', [BandedSolver.from_stencil_mat_1d, matrix_to_sparse] )
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
@pytest.mark.parametrize( 'direct_solver', [BandedSolver.from_stencil_mat_1d, matrix_to_sparse] )
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
@pytest.mark.parametrize( 'direct_solver', [BandedSolver.from_stencil_mat_1d, matrix_to_sparse] )
@pytest.mark.mpi
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
@pytest.mark.parametrize( 'direct_solver', [BandedSolver.from_stencil_mat_1d, matrix_to_sparse] )
@pytest.mark.mpi
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
@pytest.mark.parametrize( 'direct_solver', [BandedSolver.from_stencil_mat_1d, matrix_to_sparse] )
@pytest.mark.mpi
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
@pytest.mark.mpi
def test_kron_solver_3d_par(seed, n1, n2, n3, p1, p2, p3, dtype, P1=False, P2=True, P3=False, direct_solver=matrix_to_sparse):
    compare_solve(seed, MPI.COMM_WORLD, [n1,n2,n3], [p1,p2,p3], [P1,P2,P3], direct_solver, dtype=dtype, transposed=False, verbose=False)
#===============================================================================

# higher-dimensional tests

@pytest.mark.parametrize( 'dtype', [float, complex] )
@pytest.mark.parametrize( 'seed', [0, 2] )
@pytest.mark.parametrize( 'dim', [4, 6] )
@pytest.mark.mpi
def test_kron_solver_nd_par(seed, dim, dtype):
    # for now, avoid too high dim's, since we solve the matrix completely on each rank as well...

    npts_base = 4
    compare_solve(seed, MPI.COMM_WORLD, [npts_base]*dim, [1]*dim, [False]*dim, matrix_to_sparse, dtype=dtype, transposed=False, verbose=False)

#===============================================================================

# test Kronecker solver of the M1 mass matrix of our 3D DeRham sequence, as described in the get_M1_block_kron_solver method
@pytest.mark.parametrize( 'ncells', [[8, 8, 8], [8, 16, 8]] )
@pytest.mark.parametrize( 'degree', [[2, 2, 2]] )
@pytest.mark.parametrize( 'periodic', [[True, True, True]] )
@pytest.mark.mpi
def test_3d_m1_solver(ncells, degree, periodic):

    comm = MPI.COMM_WORLD
    domain = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
    derham = Derham(domain)
    domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V1 = derham_h.V1.coeff_space
    P0, P1, P2, P3  = derham_h.projectors()

    # obtain an iterative M1 solver the usual way
    u1, v1 = elements_of(derham.V1, names='u1, v1')
    a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    a1_h = discretize(a1, domain_h, (derham_h.V1, derham_h.V1))
    M1 = a1_h.assemble()
    tol = 1e-12
    maxiter = 1000
    M1_iterative_solver = inverse(M1, 'cg', tol = tol, maxiter=maxiter)

    # obtain a direct M1 solver utilizing the Block-Kronecker structure of M1
    M1_direct_solver = get_M1_block_kron_solver(V1, ncells, degree, periodic)

    # obtain x and rhs = M1 @ x, both elements of derham_h.V1
    def get_A_fun(n=1, m=1, A0=1e04):
        """Get the tuple A = (A1, A2, A3), where each entry is a function taking x,y,z as input."""

        mu_tilde = np.sqrt(m**2 + n**2)  

        eta = lambda x, y, z: x**2 * (1-x)**2 * y**2 * (1-y)**2 * z**2 * (1-z)**2

        u1  = lambda x, y, z:  A0 * (n/mu_tilde) * np.sin(np.pi * m * x) * np.cos(np.pi * n * y)
        u2  = lambda x, y, z: -A0 * (m/mu_tilde) * np.cos(np.pi * m * x) * np.sin(np.pi * n * y)
        u3  = lambda x, y, z:  A0 * np.sin(np.pi * m * x) * np.sin(np.pi * n * y)

        A1 = lambda x, y, z: eta(x, y, z) * u1(x, y, z)
        A2 = lambda x, y, z: eta(x, y, z) * u2(x, y, z)
        A3 = lambda x, y, z: eta(x, y, z) * u3(x, y, z)

        A = (A1, A2, A3)
        return A
    x = P1(get_A_fun()).coeffs
    rhs = M1 @ x

    # solve M1 @ x = rhs for x two ways
    # pass -s to see timings
    # on my local machine, executing 
    # mpirun -n 4 python -m pytest test_kron_direct_solver.py::test_3d_m1_solver -s
    # I can report the following data:

    ### 4 processes, test case 1 (ncells=[8, 8, 8]):

    # Solving for x using the iterative solver: 23.73982548713684 seconds
    # Solving for x using the iterative solver: 23.820897102355957 seconds
    # Solving for x using the iterative solver: 23.783425092697144 seconds
    # Solving for x using the iterative solver: 23.71373987197876 seconds
    # Solving for x using the direct solver: 0.3333120346069336 seconds
    # Solving for x using the direct solver: 0.3369138240814209 seconds
    # Solving for x using the direct solver: 0.33652329444885254 seconds
    # Solving for x using the direct solver: 0.34088802337646484 seconds

    ###4 processes, test case 2 (ncells=[8, 16, 8]):
    # Solving for x using the iterative solver: 82.10541296005249 seconds
    # Solving for x using the iterative solver: 81.88263297080994 seconds
    # Solving for x using the iterative solver: 82.07102465629578 seconds
    # Solving for x using the iterative solver: 82.00282955169678 seconds
    # Solving for x using the direct solver: 0.1675126552581787 seconds
    # Solving for x using the direct solver: 0.17473626136779785 seconds
    # Solving for x using the direct solver: 0.15992450714111328 seconds
    # Solving for x using the direct solver: 0.17931437492370605 seconds

    # Note that on consecutive solves, with only a slightly changing rhs and recycle=True, the iterative solver won't perform as bad anymore.

    start = time.time()
    x_iterative = M1_iterative_solver @ rhs
    stop = time.time()
    print(f"Solving for x using the iterative solver: {stop-start} seconds")

    start = time.time()
    x_direct = M1_direct_solver @ rhs
    stop = time.time()
    print(f"Solving for x using the direct solver: {stop-start} seconds")

    # assert rhs_iterative is within the tolerance close to rhs, and so is rhs_direct
    rhs_iterative = M1 @ x_iterative
    rhs_direct = M1 @ x_direct
    assert np.linalg.norm((rhs-rhs_iterative).toarray()) < tol
    assert np.linalg.norm((rhs-rhs_direct).toarray()) < tol
    
    
#===============================================================================

# test Kronecker solver of the mass matrices of our 2D (H1,Hdiv,L2) DeRham sequence, using get_inverse_mass_matrices function

@pytest.mark.parametrize( 'ncells', [[8, 8], [8, 16]] )
@pytest.mark.parametrize( 'degree', [[2, 2], [2,3]] )
@pytest.mark.parametrize( 'bounds', [[(0,1), (0,1)], [(0,0.5), (0,2.)]] )
@pytest.mark.parametrize( 'periodic', [[True, True], [False,False]] )
@pytest.mark.mpi
def test_2d_mass_solver(ncells, degree, bounds, periodic):

    comm = MPI.COMM_WORLD
    domain = Square('Omega', bounds1=bounds[0], bounds2=bounds[1])
    derham = Derham(domain, ["H1", "Hdiv", "L2"])
    domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)

    P0, P1, P2 = derham_h.projectors()
    
    # obtain an iterative M0 solver the usual way
    u0, v0 = elements_of(derham.V0, names='u0, v0')
    a0 = BilinearForm((u0, v0), integral(domain, u0*v0))
    a0_h = discretize(a0, domain_h, (derham_h.V0, derham_h.V0))
    M0 = a0_h.assemble()
    tol = 1e-10
    maxiter = 1000
    M0_iterative_solver = inverse(M0, 'cg', tol = tol, maxiter=maxiter)

    # obtain an iterative M1 solver the usual way
    u1, v1 = elements_of(derham.V1, names='u1, v1')
    a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    a1_h = discretize(a1, domain_h, (derham_h.V1, derham_h.V1))
    M1 = a1_h.assemble()
    tol = 1e-10
    maxiter = 1000
    M1_iterative_solver = inverse(M1, 'cg', tol = tol, maxiter=maxiter)
    
    # obtain an iterative M2 solver the usual way
    u2, v2 = elements_of(derham.V2, names='u0, v0')
    a2 = BilinearForm((u2, v2), integral(domain, u2*v2))
    a2_h = discretize(a2, domain_h, (derham_h.V2, derham_h.V2))
    M2 = a2_h.assemble()
    tol = 1e-10
    maxiter = 1000
    M2_iterative_solver = inverse(M2, 'cg', tol = tol, maxiter=maxiter)

    # obtain a direct M1 solver utilizing the Block-Kronecker structure of M1
    M0_direct_solver, M1_direct_solver, M2_direct_solver = get_inverse_mass_matrices(derham_h, domain_h)

    # obtain x and rhs = M1 @ x, both elements of derham_h.V1
    def get_A_fun_vec(n=1, m=1, A0=1e04):
        """Get the tuple A = (A1, A2), where each entry is a function taking x,y,z as input."""

        mu_tilde = np.sqrt(m**2 + n**2)  

        eta = lambda x, y: x**2 * (1-x)**2 * y**2 * (1-y)**2

        u1  = lambda x, y:  A0 * (n/mu_tilde) * np.sin(np.pi * m * x) * np.cos(np.pi * n * y)
        u2  = lambda x, y: -A0 * (m/mu_tilde) * np.cos(np.pi * m * x) * np.sin(np.pi * n * y)

        A1 = lambda x, y: eta(x, y) * u1(x, y)
        A2 = lambda x, y: eta(x, y) * u2(x, y)

        A = (A1, A2)
        return A
    
    def get_A_fun_scalar(n=1, m=1, A0=1e04):
        """Get the tuple A = (A1, A2), where each entry is a function taking x,y,z as input."""

        mu_tilde = np.sqrt(m**2 + n**2)  

        eta = lambda x, y: x**2 * (1-x)**2 * y**2 * (1-y)**2

        u  = lambda x, y:  A0 * (n/mu_tilde) * np.sin(np.pi * m * x) * np.cos(np.pi * n * y)

        A = lambda x, y: eta(x, y) * u(x, y)
        return A
    
    
    x0 = P0(get_A_fun_scalar()).coeffs
    rhs = M0 @ x0

    # solve M0 @ x9 = rhs for x two ways
    x_iterative = M0_iterative_solver @ rhs
    x_direct = M0_direct_solver @ rhs

    # assert rhs_iterative is within the tolerance close to rhs, and so is rhs_direct
    rhs_iterative = M0 @ x_iterative
    rhs_direct = M0 @ x_direct
    assert np.linalg.norm((rhs-rhs_iterative).toarray()) < tol
    assert np.linalg.norm((rhs-rhs_direct).toarray()) < tol
    
    x1 = P1(get_A_fun_vec()).coeffs
    rhs = M1 @ x1

    # solve M1 @ x = rhs for x two ways
    x_iterative = M1_iterative_solver @ rhs
    x_direct = M1_direct_solver @ rhs

    # assert rhs_iterative is within the tolerance close to rhs, and so is rhs_direct
    rhs_iterative = M1 @ x_iterative
    rhs_direct = M1 @ x_direct
    assert np.linalg.norm((rhs-rhs_iterative).toarray()) < tol
    assert np.linalg.norm((rhs-rhs_direct).toarray()) < tol
    
    x2 = P2(get_A_fun_scalar()).coeffs
    rhs = M2 @ x2

    # solve M2 @ x = rhs for x two ways
    x_iterative = M2_iterative_solver @ rhs
    x_direct = M2_direct_solver @ rhs

    # assert rhs_iterative is within the tolerance close to rhs, and so is rhs_direct
    rhs_iterative = M2 @ x_iterative
    rhs_direct = M2 @ x_direct
    assert np.linalg.norm((rhs-rhs_iterative).toarray()) < tol
    assert np.linalg.norm((rhs-rhs_direct).toarray()) < tol
    
#===============================================================================

if __name__ == '__main__':
    # showcase testcase
    compare_solve(0, MPI.COMM_WORLD, [4,4,5], [1,2,3], [False,True,False], BandedSolver.from_stencil_mat_1d, dtype=[float, complex], transposed=False, verbose=True)
    #compare_solve(0, MPI.COMM_WORLD, [2]*10, [1]*10, [False]*10, matrix_to_sparse, verbose=True)
