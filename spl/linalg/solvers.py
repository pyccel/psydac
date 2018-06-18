# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

def cg( A, b, x0=None, tol=1e-5, maxiter=1000, verbose=False ):
    """
    Conjugate gradient algorithm for solving linear system Ax=b.
    Implementation from [1], page 137.

    Parameters
    ----------
    A : spl.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    b : spl.linalg.basic.Vector
        Right-hand-side vector of linear system. Individual entries b[i] need
        not be accessed, but b has 'shape' attribute and provides 'copy()' and
        'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
        scalar multiplication and sum operations are available.

    x0 : spl.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for L2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, L2-norm of residual r is printed at each iteration.

    Returns
    -------
    x : spl.linalg.basic.Vector
        Converged solution.

    See also
    --------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    from math import sqrt

    n = A.shape[0]

    assert( A.shape == (n,n) )
    assert( b.shape == (n, ) )

    # First guess of solution
    if x0 is None:
        x = 0.0 * b.copy()
    else:
        assert( x0.shape == (n,) )
        x = x0.copy()

    # First values
    r  = b - A.dot( x )
    am = r.dot( r )
    p  = r.copy()

    tol_sqr = tol**2

    if verbose:
        print( "CG solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Iterate to convergence
    for m in range( 1, maxiter+1 ):

        if am < tol_sqr:
            m -= 1
            break

        v   = A.dot( p )
        l   = am / v.dot( p )
        x  += l*p
        r  -= l*v
        am1 = r.dot( r )
        p   = r + (am1/am)*p
        am  = am1

        if verbose:
            print( template.format( m, sqrt( am ) ) )

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': m, 'success': am < tol_sqr, 'res_norm': sqrt( am ) }

    return x, info

