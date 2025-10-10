import  time
import  numpy as np
import  pytest

from    sympde.calculus     import inner
from    sympde.expr         import integral, BilinearForm
from    sympde.topology     import elements_of, Derham, Mapping, Line, Square, Cube

from    psydac.api.discretization   import discretize
from    psydac.api.settings         import PSYDAC_BACKEND_GPYCCEL
from    psydac.ddm.cart             import DomainDecomposition, CartDecomposition
from    psydac.linalg.basic         import LinearOperator, LinearSolver
from    psydac.linalg.block         import BlockVectorSpace, BlockLinearOperator
from    psydac.linalg.kron          import KroneckerLinearSolver
from    psydac.linalg.solvers       import inverse
from    psydac.linalg.stencil       import StencilVectorSpace, StencilMatrix, StencilVector
from    psydac.linalg.tests.test_kron_direct_solver import matrix_to_bandsolver

def define_data_hermitian(n, p, dtype=float):
    domain_decomposition = DomainDecomposition([n - p], [False])
    cart = CartDecomposition(domain_decomposition, [n], [np.array([0])], [np.array([n - 1])], [p], [1])
    # ... Vector Spaces
    V = StencilVectorSpace(cart,dtype=dtype)
    e = V.ends[0]
    s = V.starts[0]

    # Build banded matrix with 2p+1 diagonals: must be symmetric and positive definite
    # Here we assign value 2*p on main diagonal and -1 on other diagonals
    if dtype==complex:
        factor=1+1j
    else:
        factor=1
    A = StencilMatrix(V, V)
    A[:, -p:0] = 1-1*factor
    A[:, 0:1] = 2 * p
    A[:, 1:p + 1] = 1-1*factor.conjugate()
    A.remove_spurious_entries()

    # Build exact solution
    xe = StencilVector(V)
    xe[s:e + 1] = factor*np.random.random(e + 1 - s)
    return(V, A, xe)

def define_data(n, p, matrix_data, dtype=float):
    domain_decomposition = DomainDecomposition([n - p], [False])
    cart = CartDecomposition(domain_decomposition, [n], [np.array([0])], [np.array([n - 1])], [p], [1])
    # ... Vector Spaces
    V = StencilVectorSpace(cart, dtype=dtype)
    e = V.ends[0]
    s = V.starts[0]

    # Build banded matrix with 2p+1 diagonals: must be symmetric and positive definite
    # Here we assign value 2*p on main diagonal and -1 on other diagonals

    A = StencilMatrix(V, V)
    A[:, -p:0] = -matrix_data[0]
    A[:, 0:1] = matrix_data[1]
    A[:, 1:p + 1] = matrix_data[2]
    A.remove_spurious_entries()

    # Build exact solution
    xe = StencilVector(V)
    xe[s:e + 1] = np.random.random(e + 1 - s)
    return(V, A, xe)

class SquareTorus(Mapping):

    _expressions = {'x': 'x1 * cos(x2)',
                    'y': 'x1 * sin(x2)',
                    'z': 'x3'}
    
    _ldim        = 3
    _pdim        = 3

class Annulus(Mapping):

    _expressions = {'x': 'x1 * cos(x2)',
                    'y': 'x1 * sin(x2)'}
    
    _ldim        = 2
    _pdim        = 2

def _test_LO_equality_using_rng(A, B):

    assert isinstance(A, LinearOperator)
    assert isinstance(B, LinearOperator)
    assert A.domain is B.domain
    assert A.codomain is B.codomain

    rng = np.random.default_rng(42)

    x   = A.domain.zeros()
    y1  = A.codomain.zeros()
    y2  = y1.copy()

    n   = 10

    for _ in range(n):

        x *= 0.

        if isinstance(A.domain, BlockVectorSpace):
            for block in x.blocks:
                rng.random(size=block._data.shape, dtype="float64", out=block._data)
        else:
            rng.random(size=x._data.shape, dtype="float64", out=x._data)

        A.dot(x, out=y1)
        B.dot(x, out=y2)

        diff = y1 - y2
        err  = A.codomain.inner(diff, diff)

        #print(A.codomain.inner(y1, y1))
        #print(err)
        
        assert err == 0

def get_LST_preconditioner(derham_h, M0=None, M1=None, M2=None, M3=None, backend=None):
    """
    LST (Loli, Sangalli, Tani) preconditioners are mass matrix preconditioners of the form

    pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt,

    where

    D_inv_sqrt          is the diagonal matrix of the square roots of the inverse diagonal entries of the mass matrix M,
    D_log_sqrt          is the diagonal matrix of the square roots of the diagonal entries of the mass matrix on the logical domain,
    M_log_kron_solver   is the Kronecker Solver of the mass matrix on the logical domain.

    These preconditioners work very well even on complex domains as numerical experiments have shown.
    
    """

    assert derham_h is not None

    dim = derham_h.dim
    assert dim in (2, 3)

    domain_h    = derham_h.domain_h

    domain      = domain_h.domain
    if dim == 2:
        derham  = Derham(domain, derham_h.sequence)
    else:
        derham  = Derham(domain)

    comm        = domain_h.comm
    backend     = backend

    ncells,     = domain_h.ncells.values()
    degree      = derham_h.V0.degree
    periodic,   = domain_h.periodic.values()
    
    logical_domain      = domain.logical_domain
    logical_domain_h    = discretize(logical_domain, ncells=ncells, periodic=periodic, comm=comm)

    if dim == 2:
        Ms = [M0, M1, M2]
    else:
        Ms = [M0, M1, M2, M3]

    # -----

    D_inv_sqrt_arr = []

    for M in Ms:
        if M is not None:
            D_inv_sqrt_arr.append(M.diagonal(inverse=True, sqrt=True))
        else:
            D_inv_sqrt_arr.append(None)

    # -----

    D_log_sqrt_arr = []

    for M, V, Vh in zip(Ms, derham.spaces, derham_h.spaces):
        if M is not None:
            u, v = elements_of(V, names='u, v')
            expr = inner(u, v) if isinstance(M.domain, BlockVectorSpace) else u*v
            a = BilinearForm((u, v), integral(logical_domain, expr))
            ah = discretize(a, logical_domain_h, (Vh, Vh), backend=backend)
            M_log = ah.assemble()
            D_log_sqrt_arr.append(M_log.diagonal(inverse=False, sqrt=True))
        else:
            D_log_sqrt_arr.append(None)

    # -----

    M_log_kron_solver_arr = []

    logical_domain_1d_x = Line('L', bounds=logical_domain.bounds1)
    logical_domain_1d_y = Line('L', bounds=logical_domain.bounds2)
    if dim == 3:
        logical_domain_1d_z = Line('L', bounds=logical_domain.bounds3)

    logical_domain_1d_list = [logical_domain_1d_x, logical_domain_1d_y]
    if dim == 3:
        logical_domain_1d_list += [logical_domain_1d_z]

    M0_1d_list = []
    M1_1d_list = []

    for ncells_1d, degree_1d, periodic_1d, logical_domain_1d in zip(ncells, degree, periodic, logical_domain_1d_list):

        derham_1d = Derham(logical_domain_1d)

        logical_domain_1d_h = discretize(logical_domain_1d, ncells=[ncells_1d, ], periodic=[periodic_1d, ])
        derham_1d_h = discretize(derham_1d, logical_domain_1d_h, degree=[degree_1d, ])

        V0_1d,  V1_1d  = derham_1d.spaces
        V0h_1d, V1h_1d = derham_1d_h.spaces

        u0, v0 = elements_of(V0_1d, names='u0, v0')
        u1, v1 = elements_of(V1_1d, names='u1, v1')

        a0_1d = BilinearForm((u0, v0), integral(logical_domain_1d, u0*v0))
        a1_1d = BilinearForm((u1, v1), integral(logical_domain_1d, u1*v1))

        a0h_1d = discretize(a0_1d, logical_domain_1d_h, (V0h_1d, V0h_1d))
        a1h_1d = discretize(a1_1d, logical_domain_1d_h, (V1h_1d, V1h_1d))

        M0_1d_list.append(a0h_1d.assemble())
        M1_1d_list.append(a1h_1d.assemble())

    M0_solvers  = [matrix_to_bandsolver(M) for M in M0_1d_list]
    M1_solvers  = [matrix_to_bandsolver(M) for M in M1_1d_list]

    if dim == 2:
        V0_cs = derham_h.V0.coeff_space
        V1_cs = derham_h.V1.coeff_space
        V2_cs = derham_h.V2.coeff_space

        if M0 is not None:
            M0_log_kron_solver = KroneckerLinearSolver(V0_cs, V0_cs, (M0_solvers[0], M0_solvers[1]))
            M_log_kron_solver_arr.append(M0_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

        if M1 is not None:
            if derham_h.sequence[1] == 'hcurl':
                M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M1_solvers[0], M0_solvers[1]))
                M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M0_solvers[0], M1_solvers[1]))
                M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None],
                                                                        [None, M1_1_log_kron_solver]])
            elif derham_h.sequence[1] == 'hdiv':
                M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M0_solvers[0], M1_solvers[1]))
                M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M1_solvers[0], M0_solvers[1]))
                M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None],
                                                                        [None, M1_1_log_kron_solver]])
            else:
                raise ValueError(f'The second space in the sequence {derham_h.sequence} must be either "hcurl" or "hdiv".')
            M_log_kron_solver_arr.append(M1_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

        if M2 is not None:
            M2_log_kron_solver = KroneckerLinearSolver(V2_cs, V2_cs, (M1_solvers[0], M1_solvers[1]))
            M_log_kron_solver_arr.append(M2_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)
    else:
        V0_cs = derham_h.V0.coeff_space
        V1_cs = derham_h.V1.coeff_space
        V2_cs = derham_h.V2.coeff_space
        V3_cs = derham_h.V3.coeff_space

        if M0 is not None:
            M0_log_kron_solver = KroneckerLinearSolver(V0_cs, V0_cs, (M0_solvers[0], M0_solvers[1], M0_solvers[2]))
            M_log_kron_solver_arr.append(M0_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

        if M1 is not None:
            M1_0_log_kron_solver = KroneckerLinearSolver(V1_cs[0], V1_cs[0], (M1_solvers[0], M0_solvers[1], M0_solvers[2]))
            M1_1_log_kron_solver = KroneckerLinearSolver(V1_cs[1], V1_cs[1], (M0_solvers[0], M1_solvers[1], M0_solvers[2]))
            M1_2_log_kron_solver = KroneckerLinearSolver(V1_cs[2], V1_cs[2], (M0_solvers[0], M0_solvers[1], M1_solvers[2]))
            M1_log_kron_solver = BlockLinearOperator(V1_cs, V1_cs, [[M1_0_log_kron_solver, None, None],
                                                                    [None, M1_1_log_kron_solver, None],
                                                                    [None, None, M1_2_log_kron_solver]])
            M_log_kron_solver_arr.append(M1_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)
        
        if M2 is not None:
            M2_0_log_kron_solver = KroneckerLinearSolver(V2_cs[0], V2_cs[0], (M0_solvers[0], M1_solvers[1], M1_solvers[2]))
            M2_1_log_kron_solver = KroneckerLinearSolver(V2_cs[1], V2_cs[1], (M1_solvers[0], M0_solvers[1], M1_solvers[2]))
            M2_2_log_kron_solver = KroneckerLinearSolver(V2_cs[2], V2_cs[2], (M1_solvers[0], M1_solvers[1], M0_solvers[2]))
            M2_log_kron_solver = BlockLinearOperator(V2_cs, V2_cs, [[M2_0_log_kron_solver, None, None],
                                                                    [None, M2_1_log_kron_solver, None],
                                                                    [None, None, M2_2_log_kron_solver]])
            M_log_kron_solver_arr.append(M2_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

        if M3 is not None:
            M3_log_kron_solver = KroneckerLinearSolver(V3_cs, V3_cs, (M1_solvers[0], M1_solvers[1], M1_solvers[2]))
            M_log_kron_solver_arr.append(M3_log_kron_solver)
        else:
            M_log_kron_solver_arr.append(None)

    # --------------------------------

    M_pc_arr = []

    for M, D_inv_sqrt, D_log_sqrt, M_log_kron_solver in zip(Ms, D_inv_sqrt_arr, D_log_sqrt_arr, M_log_kron_solver_arr):
        #print(M, D_inv_sqrt, D_log_sqrt, M_log_kron_solver)
        if M is not None:
            M_pc = D_inv_sqrt @ D_log_sqrt @ M_log_kron_solver @ D_log_sqrt @ D_inv_sqrt
            M_pc_arr.append(M_pc)

    return M_pc_arr

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('solver', ['cg', 'pcg', 'bicg', 'bicgstab', 'pbicgstab', 'minres', 'lsmr', 'gmres'])

def test_solver_tridiagonal(n, p, dtype, solver, verbose=False):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    if solver in ['bicg', 'bicgstab', 'pbicgstab', 'lsmr']:
        if dtype==complex:
            diagonals = [1-10j,6+9j,3+5j]
        else:
            diagonals = [1,6,3]
            
        if solver == 'pbicgstab' and dtype == complex:
            # pbicgstab only works for real matrices
            return
    elif solver == 'gmres':
        if dtype==complex:
            diagonals = [-7-2j,-6-2j,-1-10j]
        else:
            diagonals = [-7,-1,-3]

    if solver in ['cg', 'pcg', 'minres']:
        # pcg runs with Jacobi preconditioner
        V, A, xe = define_data_hermitian(n, p, dtype=dtype)
        if solver == 'minres' and dtype == complex:
            # minres only works for real matrices
            return
    else:
        V, A, xe = define_data(n, p, diagonals, dtype=dtype)

    # Tolerance for success: 2-norm of error in solution
    tol = 1e-8

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------
    if verbose:
        # Title
        print()
        print( "="*80 )
        print( f"SERIAL TEST: solve linear system A*x = b using {solver}")
        print( "="*80 )
        print()

    #Create the solvers
    if solver in ['pcg', 'pbicgstab']:
        pc = A.diagonal(inverse=True)
        solv = inverse(A, solver, pc=pc, tol=1e-13, verbose=verbose, recycle=True)
    else:
        solv = inverse(A, solver, tol=1e-13, verbose=verbose, recycle=True)
    solvt = solv.transpose()
    solvh = solv.H
    solv2 = inverse(A@A, solver, tol=1e-13, verbose=verbose, recycle=True) # Test solver of composition of operators

    # Manufacture right-hand-side vector from exact solution
    be  = A @ xe
    be2 = A @ be # Test solver with consecutive solves
    bet = A.T @ xe
    beh = A.H @ xe

    # Solve linear system
    # Assert x0 got updated correctly and is not the same object as the previous solution, but just a copy
    x = solv @ be
    info = solv.get_info()
    solv_x0 = solv._options["x0"]
    assert np.array_equal(x.toarray(), solv_x0.toarray())
    assert x is not solv_x0

    x2 = solv @ be2
    solv_x0 = solv._options["x0"]
    assert np.array_equal(x2.toarray(), solv_x0.toarray())
    assert x2 is not solv_x0

    xt = solvt.solve(bet)
    solvt_x0 = solvt._options["x0"]
    assert np.array_equal(xt.toarray(), solvt_x0.toarray())
    assert xt is not solvt_x0

    xh = solvh.dot(beh)
    solvh_x0 = solvh._options["x0"]
    assert np.array_equal(xh.toarray(), solvh_x0.toarray())
    assert xh is not solvh_x0

    if solver != 'pcg':
        # PCG only works with operators with diagonal
        xc = solv2 @ be2
        solv2_x0 = solv2._options["x0"]
        assert np.array_equal(xc.toarray(), solv2_x0.toarray())
        assert xc is not solv2_x0


    # Verify correctness of calculation: 2-norm of error
    b = A @ x
    b2 = A @ x2
    bt = A.T @ xt
    bh = A.H @ xh
    if solver != 'pcg':
        bc = A @ A @ xc

    err = b - be
    err_norm = np.linalg.norm( err.toarray() )
    err2 = b2 - be2
    err2_norm = np.linalg.norm( err2.toarray() )
    errt = bt - bet
    errt_norm = np.linalg.norm( errt.toarray() )
    errh = bh - beh
    errh_norm = np.linalg.norm( errh.toarray() )

    if solver != 'pcg': 
        errc = bc - be2
        errc_norm = np.linalg.norm( errc.toarray() )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------
    if verbose:
        print()
        print( 'A  =', A, sep='\n' )
        print( 'b  =', b )
        print( 'x  =', x )
        print( 'xe =', xe )
        print( 'info =', info )
        print()

        print( "-"*40 )
        print( f"2-norm of error in solution = {err_norm:.2e}" )
        if err_norm < tol:
            print( "PASSED" )
        else:
            print( "FAIL" )
        print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    # The lsmr solver does not consistently produce outputs x whose error ||Ax - b|| is less than tol.
    if solver != 'lsmr':
        assert err_norm < tol
        assert err2_norm < tol
        assert errt_norm < tol
        assert errh_norm < tol
        assert solver == 'pcg' or errc_norm < tol

#===============================================================================
def test_LST_preconditioner():

    ncells_3d      = [16, 7, 11]
    degree_3d      = [1, 4, 2]
    periodic_3d    = [False, True, False]

    comm    = None
    backend = PSYDAC_BACKEND_GPYCCEL

    dimensions = [2, 3]

    maxiter = 20000
    tol     = 1e-13

    # Test both in 2D and 3D
    for dim in dimensions:
        print(f' ----- Start {dim}D test -----')

        ncells      = ncells_3d  [0:2] if dim == 2 else ncells_3d
        degree      = degree_3d  [0:2] if dim == 2 else degree_3d
        periodic    = periodic_3d[0:2] if dim == 2 else periodic_3d

        if dim == 2:
            logical_domain = Square('S', bounds1=(0.5, 1), bounds2=(0, 2*np.pi))
            mapping = Annulus('A')
            sequence = ['h1', 'hcurl', 'l2']
        else:
            logical_domain = Cube  ('C', bounds1=(0.5, 1), bounds2=(0, 2*np.pi), bounds3=(0, 1))
            mapping = SquareTorus('ST')

        domain  = mapping(logical_domain)

        derham = Derham(domain, sequence=sequence) if dim == 2 else Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree)

        Vs                      = derham.spaces
        Vhs                     = derham_h.spaces

        mass_matrices = []

        for V, Vh in zip(Vs, Vhs):
            print(f"V = {V} is a {type(V)}")
            print(V.shape, V.kind, type(V.kind), V.domain, type(V.domain), V.kind.name)
            u, v = elements_of(V, names='u, v')
            expr = inner(u, v) if isinstance(Vh.coeff_space, BlockVectorSpace) else u*v
            a = BilinearForm((u, v), integral(domain, expr))
            ah = discretize(a, domain_h, (Vh, Vh), backend=backend)
            mass_matrices.append(ah.assemble())

        if dim == 2:
            M0, M1, M2 = mass_matrices
        else:
            M0, M1, M2, M3 = mass_matrices

        if dim == 2:
            mass_matrix_preconditioners   = get_LST_preconditioner(derham_h, M0=M0, M1=M1, M2=M2)
        else:
            mass_matrix_preconditioners   = get_LST_preconditioner(derham_h, M0=M0, M1=M1, M2=M2, M3=M3)

        # Prepare testing whether obtaining only a subset of preconditioners works
        mass_matrix_preconditioners_1,  = get_LST_preconditioner(derham_h, M1=M1)
        mass_matrix_preconditioners_2 = get_LST_preconditioner(derham_h, M0=M0, M2=M2)
        if dim == 3:
            mass_matrix_preconditioners_3,  = get_LST_preconditioner(derham_h, M3=M3)

        test_pcs = [mass_matrix_preconditioners_2[0], mass_matrix_preconditioners_1, mass_matrix_preconditioners_2[1]]
        if dim == 3:
            test_pcs += [mass_matrix_preconditioners_3, ]

        # Test 1: Test whether obtaining only a subset of all possible preconditioners works
        for pc, test_pc in zip(mass_matrix_preconditioners, test_pcs):
            _test_LO_equality_using_rng(pc, test_pc)

        print(f' Accessing a subset of all possible preconditioners works.')

        rng = np.random.default_rng(42)

        # For comparison and testing: Number of iterations required, not using and using a preconditioner
        # More information via "-s" when running the test
        true_cg_niter  = [[90, 681, 62], [486, 7970, 5292, 147]]
        true_pcg_niter = [[ 6,   6,  2], [  6,    7,    6,   2]]

        for i, (M, Mpc) in enumerate(zip(mass_matrices, mass_matrix_preconditioners)):

            M_inv_cg  = inverse(M, 'cg',          maxiter=maxiter, tol=tol)
            M_inv_pcg = inverse(M, 'pcg', pc=Mpc, maxiter=maxiter, tol=tol)

            y = M.codomain.zeros()
            if isinstance(M.codomain, BlockVectorSpace):
                for block in y.blocks:
                    rng.random(size=block._data.shape, dtype="float64", out=block._data)
            else:
                rng.random(size=y._data.shape, dtype="float64", out=y._data)

            t0 = time.time()
            x_cg = M_inv_cg @ y
            t1 = time.time()

            y_cg     = M @ x_cg
            diff_cg  = y - y_cg
            err_cg   = np.sqrt(M.codomain.inner(diff_cg, diff_cg))
            time_cg  = t1 - t0
            info_cg  = M_inv_cg.get_info()

            t0 = time.time()
            x_pcg = M_inv_pcg @ y
            t1 = time.time()

            y_pcg    = M @ x_pcg
            diff_pcg = y - y_pcg
            err_pcg  = np.sqrt(M.codomain.inner(diff_pcg, diff_pcg))
            time_pcg = t1 - t0
            info_pcg = M_inv_pcg.get_info()

            print(f' - M{i} test -')
            print(f' CG : {info_cg} in {time_cg:.3g}s       - err.: {err_cg:.3g}')
            print(f' PCG: {info_pcg} in {time_pcg:.3g}s     - err.: {err_pcg:.3g}')

            if dim == 2:
                assert info_pcg['niter'] == true_pcg_niter[0][i]
            else:
                assert info_pcg['niter'] == true_pcg_niter[1][i]

        print()

# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
