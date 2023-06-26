
import numpy as np
import pytest
from psydac.linalg.solvers import inverse
from psydac.linalg.stencil import StencilVectorSpace, StencilMatrix, StencilVector
from psydac.linalg.basic import LinearSolver
from psydac.ddm.cart import DomainDecomposition, CartDecomposition

def compute_global_starts_ends(domain_decomposition, npts, pads):
    ndims = len(npts)
    global_starts = [None] * ndims
    global_ends = [None] * ndims

    for axis in range(ndims):
        ee = domain_decomposition.global_element_ends[axis]

        global_ends[axis] = ee.copy()
        global_ends[axis][-1] = npts[axis] - 1
        global_starts[axis] = np.array([0] + (global_ends[axis][:-1] + 1).tolist())

    for s, e, p in zip(global_starts, global_ends, pads):
        assert all(e - s + 1 >= p)

    return tuple(global_starts), tuple(global_ends)

def define_data_hermitian(n, p, dtype=float, comm=None):
    if comm:
        domain_decomposition = DomainDecomposition([n - 1], periods=[False], comm=comm)
    else:
        domain_decomposition = DomainDecomposition([n - 1], periods=[False])

    global_starts, global_ends = compute_global_starts_ends(domain_decomposition, [n], [p])
    cart = CartDecomposition(domain_decomposition, [n], global_starts, global_ends, pads=[p], shifts=[1])
    # ... Vector Spaces
    V = StencilVectorSpace(cart, dtype=dtype)
    e = V.ends[0]
    s = V.starts[0]

    # Build banded matrix with 2p+1 diagonals: must be symmetric and positive definite
    # Here we assign value 2*p on main diagonal and -1 on other diagonals
    if dtype == complex:
        factor = 1+1j
    else:
        factor = 1
    A = StencilMatrix(V, V)
    A[:, -p:0] = 1-1*factor
    A[:, 0:1] = 2 * p
    A[:, 1:p + 1] = 1-1*factor.conjugate()
    A.remove_spurious_entries()

    # Build exact solution
    xe = StencilVector(V)
    xe[s:e + 1] = factor*np.random.random(e + 1 - s)
    return(V, A, xe)

def define_data(n, p, matrix_data, dtype=float, comm=None):
    if comm:
        domain_decomposition = DomainDecomposition([n - p], periods=[False], comm=None)
    else:
        domain_decomposition = DomainDecomposition([n - p], periods=[False])
    global_starts, global_ends = compute_global_starts_ends(domain_decomposition, [n], [p])
    cart = CartDecomposition(domain_decomposition, [n], global_starts, global_ends, [p], [1])
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
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('solver', ['cg', 'pcg', 'bicg', 'bicgstab', 'minres', 'lsmr', 'gmres'])

def test_solver_tridiagonal(n, p, dtype, solver, verbose=False):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    if solver in ['bicg', 'bicgstab', 'lsmr']:
        if dtype==complex:
            diagonals = [1-10j,6+9j,3+5j]
        else:
            diagonals = [1,6,3]
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
    tol = 1e-10

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
    solv  = inverse(A, solver, tol=1e-13, verbose=True)
    solvt = solv.transpose()
    solvh = solv.H

    # Manufacture right-hand-side vector from exact solution
    b  = A.dot( xe )
    b2 = A.dot( b ) # Test solver with consecutive solves
    bt = A.T.dot( xe )
    bh = A.H.dot( xe )

    # Solve linear system
    x = solv @ b
    info = solv.get_info()
    x2 = solv @ b2
    xt = solvt.solve(bt)
    xh = solvh.dot(bh)

    # Verify correctness of calculation: 2-norm of error
    err = x - xe
    err_norm = np.linalg.norm( err.toarray() )
    err2 = x2 - b
    err2_norm = np.linalg.norm( err2.toarray() )
    errt = xt - xe
    errt_norm = np.linalg.norm( errt.toarray() )
    errh = xh - xe
    errh_norm = np.linalg.norm( errh.toarray() )

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
    assert errt_norm < tol
    assert errh_norm < tol

@pytest.mark.parametrize( 'n', [15, 20] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('solver', ['cg', 'pcg', 'bicg', 'bicgstab', 'lsmr', 'gmres'])
@pytest.mark.parallel

def test_solver_tridiagonal_parallel(n, p, dtype, solver, verbose=False):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------
    from mpi4py import MPI
    if solver in ['bicg', 'bicgstab', 'lsmr']:
        if dtype==complex:
            diagonals = [1-10j,6+9j,3+5j]
        else:
            diagonals = [1,6,3]
    elif solver == 'gmres':
        if dtype==complex:
            diagonals = [-7-2j,-6-2j,-1-10j]
        else:
            diagonals = [-7,-1,-3]

    if solver in ['cg', 'pcg', 'minres']:
        # pcg runs with Jacobi preconditioner
        V, A, xe = define_data_hermitian(n, p, dtype=dtype, comm=MPI.COMM_WORLD)
        if solver == 'minres' and dtype == complex:
            # minres only works for real matrices
            return
    else:
        V, A, xe = define_data(n, p, diagonals, dtype=dtype, comm=MPI.COMM_WORLD)

    # Tolerance for success: 2-norm of error in solution
    tol = 1e-10

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
    solv  = inverse(A, solver, tol=1e-13, verbose=True)
    solvt = solv.transpose()
    solvh = solv.H

    # Manufacture right-hand-side vector from exact solution
    b  = A.dot( xe )
    b2 = A.dot( b ) # Test solver with consecutive solves
    bt = A.T.dot( xe )
    bh = A.H.dot( xe )

    # Solve linear system
    x = solv @ b
    info = solv.get_info()
    x2 = solv @ b2
    xt = solvt.solve(bt)
    xh = solvh.dot(bh)

    # Verify correctness of calculation: 2-norm of error
    err = x - xe
    err_norm = np.linalg.norm( err.toarray() )
    err2 = x2 - b
    err2_norm = np.linalg.norm( err2.toarray() )
    errt = xt - xe
    errt_norm = np.linalg.norm( errt.toarray() )
    errh = xh - xe
    errh_norm = np.linalg.norm( errh.toarray() )

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
    assert errt_norm < tol
    assert errh_norm < tol


# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
