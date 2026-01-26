#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import  time
import  numpy as np
import  pytest
from    mpi4py import MPI

from    sympde.topology               import elements_of, Mapping, Derham, Square, Cube
from    sympde.calculus               import inner
from    sympde.expr                   import integral, BilinearForm

from    psydac.api.discretization     import discretize
from    psydac.api.settings           import PSYDAC_BACKEND_GPYCCEL
from    psydac.ddm.cart               import DomainDecomposition, CartDecomposition
from    psydac.fem.lst_preconditioner import construct_LST_preconditioner
from    psydac.linalg.basic           import IdentityOperator, ScaledLinearOperator, InverseLinearOperator, MatrixFreeLinearOperator
from    psydac.linalg.block           import BlockVectorSpace
from    psydac.linalg.solvers         import inverse
from    psydac.linalg.stencil         import StencilVectorSpace, StencilMatrix, StencilVector
from    psydac.linalg.tests.utilities import check_linop_equality_using_rng, Annulus, SquareTorus


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

#===============================================================================
@pytest.mark.parametrize('n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize(('solver', 'use_jacobi_pc'),
    [('CG'      , False), ('CG', True),
     ('BiCG'    , False),
     ('BiCGSTAB', False), ('BiCGSTAB', True),
     ('MINRES'  , False),
     ('LSMR'    , False),
     ('GMRES'   , False)]
 )
def test_solver_tridiagonal(n, p, dtype, solver, use_jacobi_pc, verbose=False):

    # Quickly skip tests that are not relevant
    if solver == 'BiCGSTAB' and use_jacobi_pc and dtype == complex:
        pytest.skip("Preconditioned BiCGSTAB only works for real matrices")
    elif solver == 'MINRES' and dtype == complex:
        pytest.skip("MINRES only works for real matrices")
    
    # Also skip some problematic tests for now -- see Issue #557
    if solver == 'LSMR' and dtype == complex:
        pytest.skip("LSMR currently failing for complex matrices. Please investigate")
    elif solver == 'BiCG' and dtype == complex:
        pytest.skip("BiCG currently failing for complex matrices. Please investigate")

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    #... Define data for the test
    if solver in ['CG', 'MINRES']:
        # CG and MINRES require symmetric(hermitian) positive definite matrices
        hermitian = True
        diagonals = None
    elif solver in ['BiCG', 'BiCGSTAB', 'LSMR']:
        hermitian = False
        diagonals = [1-10j, 6+9j, 3+5j] if dtype==complex else [1, 6, 3]
    elif solver == 'GMRES':
        hermitian = False
        diagonals = [-7-2j, -6-2j, -1-10j] if dtype==complex else [-7, -6, -1]

    if hermitian:
        V, A, xe = define_data_hermitian(n, p, dtype=dtype)
    else:
        V, A, xe = define_data(n, p, diagonals, dtype=dtype)
    #...

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
    pc = A.diagonal(inverse=True) if use_jacobi_pc else None
    solv = inverse(A, solver, pc=pc, tol=1e-13, verbose=verbose, recycle=True)
    solvt = solv.transpose()
    solvh = solv.H
    # Test solver on composition of operators
    solv2 = inverse(A@A, solver, pc=pc, tol=1e-13, verbose=verbose, recycle=True) 
    # Test solver on scaled linear operators
    A_mf = MatrixFreeLinearOperator(domain=A.codomain, codomain=A.domain, dot=lambda x:A@x, dot_transpose=lambda x:(A.T)@x)
    A3 = 3*A_mf
    assert isinstance(A3, ScaledLinearOperator)
    solv3 = inverse(A3, solver, pc=pc, tol=1e-13, verbose=verbose, recycle=True) 
    # Test solver on inverse linear operators
    assert isinstance(solv, InverseLinearOperator)
    solv4 = inverse(solv, solver, pc=pc, tol=1e-13, verbose=verbose, recycle=True) 

    # Manufacture right-hand-side vector from exact solution
    be  = A @ xe
    be2 = A @ be # Test solver with consecutive solves
    be3 = 3 * be
    be4 = xe
    bet = A.T @ xe
    beh = A.H @ xe

    # Solve linear system
    # Assert x0 got updated correctly and is not the same object as the previous solution, but just a copy
    x = solv @ be  # A^{-1} @ A @ xe  = xe
    info = solv.get_info()
    solv_x0 = solv._options["x0"]
    assert np.array_equal(x.toarray(), solv_x0.toarray())
    assert x is not solv_x0

    x2 = solv @ be2  # A^{-1} @ A @ A @ xe = A @ xe
    solv_x0 = solv._options["x0"]
    assert np.array_equal(x2.toarray(), solv_x0.toarray())
    assert x2 is not solv_x0

    x3 = solv3 @ be3  # (3A)^{-1} @ (3 * A @ xe ) = xe
    assert isinstance(solv3, ScaledLinearOperator)
    # a ScaledLinearOperator has no attribute '_option'

    x4 = solv4 @ be4  # ((A)^{-1))^{-1} @ xe = A @ xe
    assert isinstance(solv4, StencilMatrix)
    # a StencilMatrix has no attribute '_option'

    xt = solvt.solve(bet) 
    solvt_x0 = solvt._options["x0"]
    assert np.array_equal(xt.toarray(), solvt_x0.toarray())
    assert xt is not solvt_x0

    xh = solvh.dot(beh)
    solvh_x0 = solvh._options["x0"]
    assert np.array_equal(xh.toarray(), solvh_x0.toarray())
    assert xh is not solvh_x0

    if not (solver == 'CG' and use_jacobi_pc):
        # PCG only works with operators with diagonal
        xc = solv2 @ be2
        solv2_x0 = solv2._options["x0"]
        assert np.array_equal(xc.toarray(), solv2_x0.toarray())
        assert xc is not solv2_x0

    # Verify correctness of calculation: 2-norm of error
    b = A @ x
    b2 = A @ x2
    b3 = A @ x3
    b4 = A @ x4
    bt = A.T @ xt
    bh = A.H @ xh
    if not (solver == 'CG' and use_jacobi_pc):
        bc = A @ A @ xc

    err = b - be
    err_norm = np.linalg.norm( err.toarray() )
    err2 = b2 - be2
    err2_norm = np.linalg.norm( err2.toarray() )
    err3 = b3 - be
    err3_norm = np.linalg.norm( err3.toarray() )
    err4 = b4 - be2
    err4_norm = np.linalg.norm( err4.toarray() )
    errt = bt - bet
    errt_norm = np.linalg.norm( errt.toarray() )
    errh = bh - beh
    errh_norm = np.linalg.norm( errh.toarray() )

    if not (solver == 'CG' and use_jacobi_pc):
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
    assert err_norm < tol
    assert err2_norm < tol
    assert err3_norm < tol
    assert err4_norm < tol
    assert errt_norm < tol
    assert errh_norm < tol
    assert (solver == 'CG' and use_jacobi_pc) or errc_norm < tol

#===============================================================================
def test_LST_preconditioner(comm=None):

    ncells_3d   = [16, 7, 11]
    degree_3d   = [1, 4, 2]
    periodic_3d = [False, True, False]

    prin = True if ((comm is None) or (comm.rank == 0)) else False
    backend = PSYDAC_BACKEND_GPYCCEL

    dimensions = [2, 3]

    maxiter = 20000
    tol     = 1e-13

    if prin:
        print()
    # Test both in 2D and 3D
    for dim in dimensions:
        if prin:
            print(f' ----- Start {dim}D test -----')

        ncells   = ncells_3d  [0:2] if dim == 2 else ncells_3d
        degree   = degree_3d  [0:2] if dim == 2 else degree_3d
        periodic = periodic_3d[0:2] if dim == 2 else periodic_3d

        if dim == 2:
            logical_domain = Square('S', bounds1=(0.5, 1), bounds2=(0, 2*np.pi))
            mapping = Annulus('A')
            sequence = ['h1', 'hcurl', 'l2']
        else:
            logical_domain = Cube  ('C', bounds1=(0.5, 1), bounds2=(0, 2*np.pi), bounds3=(0, 1))
            mapping = SquareTorus('ST')

        domain = mapping(logical_domain)

        derham = Derham(domain, sequence=sequence) if dim == 2 else Derham(domain)

        domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
        derham_h = discretize(derham, domain_h, degree=degree)

        Vs  = derham.spaces
        Vhs = derham_h.spaces

        d_projectors = derham_h.dirichlet_projectors(kind='linop')

        mass_matrices   = []
        mass_0_matrices = []

        for i, (V, Vh) in enumerate(zip(Vs, Vhs)):
            u, v = elements_of(V, names='u, v')
            expr = inner(u, v) if isinstance(Vh.coeff_space, BlockVectorSpace) else u*v
            a    = BilinearForm((u, v), integral(domain, expr))
            ah   = discretize(a, domain_h, (Vh, Vh), backend=backend)
            M    = ah.assemble()
            mass_matrices.append(M)
            if i < dim:
                DP  = d_projectors[i]
                I   = IdentityOperator(Vhs[i].coeff_space)
                M_0 = DP @ M @ DP + (I - DP)
                mass_0_matrices.append(M_0)

        if dim == 2:
            M0, M1, M2 = mass_matrices
        else:
            M0, M1, M2, M3 = mass_matrices

        if dim == 2:
            mass_matrix_preconditioners   = derham_h.LST_preconditioners(M0=M0, M1=M1, M2=M2             )
            mass_0_matrix_preconditioners = derham_h.LST_preconditioners(M0=M0, M1=M1,        hom_bc=True)
        else:
            mass_matrix_preconditioners   = derham_h.LST_preconditioners(M0=M0, M1=M1, M2=M2, M3=M3,            )
            mass_0_matrix_preconditioners = derham_h.LST_preconditioners(M0=M0, M1=M1, M2=M2,        hom_bc=True)

        # Prepare testing whether obtaining only a subset of preconditioners works
        M1_pc,       = derham_h.LST_preconditioners(M1=M1       )
        M0_pc, M2_pc = derham_h.LST_preconditioners(M0=M0, M2=M2)
        if dim == 3:
            M3_pc,   = derham_h.LST_preconditioners(M3=M3       )

        test_pcs = [M0_pc, M1_pc, M2_pc]
        if dim == 3:
            test_pcs += [M3_pc]

        # Test whether obtaining only a subset of all possible preconditioners works
        for pc, test_pc in zip(mass_matrix_preconditioners, test_pcs):
            check_linop_equality_using_rng(pc, test_pc)#, tol=1e-13)

        if prin:
            print(f' Accessing a subset of all possible preconditioners works.')

        rng = np.random.default_rng(42)

        # For comparison and testing: Number of iterations required, not using and using a preconditioner
        # More information via " -s" when running the test
        #                            dim 2                           dim 3
        #                   M0   M1  M2  M0_0  M1_0     M0    M1    M2   M3  M0_0  M1_0  M2_0
        true_cg_niter   = [[90, 681, 62,   77,  600], [486, 7970, 5292, 147,  356, 5892, 4510]]
        true_pcg_niter  = [[ 6,   6,  2,    5,    5], [  6,    7,    6,   2,    5,    5,    5]]
        # M{i}_0 matrices preconditioned with a LST preconditioner designed for M{i} instead:
        #                                M0_0  M1_0                          M0_0  M1_0  M2_0
        true_pcg_niter2 = [[               23,   24], [                       367, 2867,  220]]

        mass_matrices               += mass_0_matrices
        mass_matrix_preconditioners += mass_0_matrix_preconditioners
        extended_fem_spaces          = Vhs + Vhs[:-1]

        for i, (M, Mpc, Vh) in enumerate(zip(mass_matrices, mass_matrix_preconditioners, extended_fem_spaces)):

            cg = False # Set to True to compare iterations and time with not-preconditioned Conjugate Gradient solver

            # hom_bc = False for M0 M1 M2 (M3), then hom_bc = True for M0_0 M1_0 (M2_0)
            hom_bc = True if i > dim else False

            # In order to obtain an LST for M{i}_0, we still have to pass M{i} to `construct_LST_preconditioner`.`
            # M2 = M{i} if M = M{i}_0 and hence can be used to obtain the pc for M{i}_0
            M2 = mass_matrices[i-dim-1] if i > dim else M
            Mpc2 = construct_LST_preconditioner(M2, domain_h, Vh, hom_bc=hom_bc)
            check_linop_equality_using_rng(Mpc, Mpc2, tol=1e-12)
            if prin:
                print(' The LST pc obtained using derham_h.LST_preconditioners is the same as the one obtained from construct_LST_preconditioner.')

            if cg:
                M_inv_cg = inverse(M, 'cg',          maxiter=maxiter, tol=tol)
            M_inv_pcg = inverse(M, 'cg', pc=Mpc, maxiter=maxiter, tol=tol)

            y = M.codomain.zeros()
            if isinstance(M.codomain, BlockVectorSpace):
                for block in y.blocks:
                    rng.random(size=block._data.shape, dtype="float64", out=block._data)
            else:
                rng.random(size=y._data.shape, dtype="float64", out=y._data)

            if (i > dim):
                if prin:
                    print(f' Projecting rhs vector into space of functions satisfying hom. DBCs')
                DP = d_projectors[i-(dim+1)]
                y  = DP @ y

            if cg:
                t0 = time.time()
                x_cg = M_inv_cg @ y
                t1 = time.time()

                y_cg    = M @ x_cg
                diff_cg = y - y_cg
                err_cg  = np.sqrt(M.codomain.inner(diff_cg, diff_cg))
                time_cg = t1 - t0
                info_cg = M_inv_cg.get_info()

            t0 = time.time()
            x_pcg = M_inv_pcg @ y
            t1 = time.time()

            y_pcg    = M @ x_pcg
            diff_pcg = y - y_pcg
            err_pcg  = np.sqrt(M.codomain.inner(diff_pcg, diff_pcg))
            time_pcg = t1 - t0
            info_pcg = M_inv_pcg.get_info()

            if dim == 2:
                mat_txt = f'M{i}' if i <= 2 else f'M{i-3}_0'
            else:
                mat_txt = f'M{i}' if i <= 3 else f'M{i-4}_0'

            if prin:
                print(f' - {mat_txt} test -')
                if cg:
                    print(f' CG : {info_cg} in {time_cg:.3g}s       - err.: {err_cg:.3g}')
                print(f' PCG: {info_pcg} in {time_pcg:.3g}s     - err.: {err_pcg:.3g}')

            if dim == 2:
                assert info_pcg['niter'] == true_pcg_niter[0][i]
            else:
                assert info_pcg['niter'] == true_pcg_niter[1][i]

        print()

#===============================================================================
@pytest.mark.mpi
def test_LST_preconditioner_parallel():
    comm = MPI.COMM_WORLD
    test_LST_preconditioner(comm=comm)

# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
