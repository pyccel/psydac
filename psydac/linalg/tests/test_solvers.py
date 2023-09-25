
import numpy as np
import pytest
from psydac.linalg.solvers import inverse
from psydac.linalg.stencil import StencilVectorSpace, StencilMatrix, StencilVector
from psydac.linalg.basic import LinearSolver
from psydac.ddm.cart import DomainDecomposition, CartDecomposition


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
    solv  = inverse(A, solver, tol=tol, verbose=False, recycle=True)
    solvt = solv.transpose()
    solvh = solv.H

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

    # Verify correctness of calculation: 2-norm of error
    b = A @ x
    b2 = A @ x2
    bt = A.T @ xt
    bh = A.H @ xh

    
    err = b - be
    err_norm = np.linalg.norm( err.toarray() )
    err2 = b2 - be2
    err2_norm = np.linalg.norm( err2.toarray() )
    errt = bt - bet
    errt_norm = np.linalg.norm( errt.toarray() )
    errh = bh - beh
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
    # The lsmr solver does not consistently produce outputs x whose error ||Ax - b|| is less than tol.
    if solver != 'lsmr':
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
