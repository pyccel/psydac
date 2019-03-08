import numpy as np
import pytest

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )

def test_bicg_tridiagonal( n ):
    """
    Test generic Biconjugate Gradient algorithm on tridiagonal linear system.

    Parameters
    ----------
    n : int
        Dimension of linear system (number of rows = number of columns).

    """
    from psydac.linalg.iterative_solvers import bicg

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    # Build generic non-singular matrix
    # Here tridiagonal matrix with values [-1, 3, -2] on diagonals
    A = np.diag([-1.0]*(n-1),-1) + np.diag([3.0]*n,0) + np.diag([-2.0]*(n-1),1)

    # Build exact solution: here with random values in [-1,1]
    xe = 2.0 * np.random.random( n ) - 1.0

    # Tolerance for success: L2-norm of error in solution
    tol = 1e-12

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------

    # Title
    print()
    print( "="*80 )
    print( "SERIAL TEST: solve linear system A*x = b using biconjugate gradient" )
    print( "="*80 )
    print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )

    # Solve linear system using BiCG
    x, info = bicg( A, A.transpose(), b, tol=1e-13, verbose=True )

    # Verify correctness of calculation: L2-norm of error
    err = x-xe
    err_norm = np.linalg.norm( err )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------

    print()
    print( 'A  =', A, sep='\n' )
    print( 'b  =', b )
    print( 'x  =', x )
    print( 'xe =', x )
    print( 'info =', info )
    print()

    print( "-"*40 )
    print( "L2-norm of error in solution = {:.2e}".format( err_norm ) )
    if err_norm < tol:
        print( "PASSED" )
    else:
        print( "FAIL" )
    print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    assert err_norm < tol
