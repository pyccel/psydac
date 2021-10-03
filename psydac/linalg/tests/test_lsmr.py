import numpy as np
import pytest

#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )

def test_lsmr_tridiagonal( n ):
    """
    Test generic LSMR algorithm on tridiagonal linear system.

    Parameters
    ----------
    n : int
        Dimension of linear system (number of rows = number of columns).

    """
    from psydac.linalg.iterative_solvers import lsmr

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    # Build generic non-singular matrix
    sdiag = np.random.random( n - 1 )
    diag  = np.random.random( n )
    A = np.diag(sdiag,-1) + np.diag(diag,0) + np.diag(sdiag,1)

    # Build exact solution: here with random values in [-1,1]
    xe = 2.0 * np.random.random( n ) - 1.0

    # Tolerance for success: L2-norm of error in solution
    tol = 1e-10

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------

    # Title
    print()
    print( "="*80 )
    print( "SERIAL TEST: solve linear system A*x = b using lsmr" )
    print( "="*80 )
    print()

    # Manufacture right-hand-side vector from exact solution
    b = A.dot( xe )

    # Solve linear system using BiCG
    x, info = lsmr( A, A.T, b, tol=1e-13, verbose=True )

    # Verify correctness of calculation: L2-norm of error
    res = A.dot(x)-b
    res_norm = np.linalg.norm( res )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------

    print()
    print( 'A  =', A, sep='\n' )
    print( 'b  =', b )
    print( 'x  =', x )
    print( 'xe =', xe )
    print( 'info =', info )
    print()

    print( "-"*40 )
    print( "L2-norm of error in solution = {:.2e}".format( res_norm ) )
    if res_norm < tol:
        print( "PASSED" )
    else:
        print( "FAIL" )
    print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    assert res_norm < tol
