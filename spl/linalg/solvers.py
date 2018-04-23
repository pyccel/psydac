# coding: utf-8

import numpy as np

# ... Solver: CGL performs maxit CG iterations on the linear system Ax = b
#     starting from x = x0
def cgl(mat, b, x0, maxit, tol):
    xk = x0.zeros_like()
    mx = x0.zeros_like()
    p  = x0.zeros_like()
    q  = x0.zeros_like()
    r  = x0.zeros_like()

    # xk = x0
    xk = x0.copy()
    mx = mat.dot(x0)

    # r = b - mx
    r = b.copy()
    b.sub(mx)

    # p = r
    p = r.copy()

    rdr = r.dot(r)

    for i_iter in range(1, maxit+1):
        q = mat.dot(p)
        alpha = rdr / p.dot(q)

        # xk = xk + alpha * p
        ap = p.copy()
        ap.mul(alpha)
        xk.add(ap)

        # r  = r - alpha * q
        aq = q.copy()
        aq.mul(alpha)
        r.sub(aq)

        # ...
        if r.dot(r) >= 0.:
            norm_err = np.sqrt(r.dot(r))
            print (i_iter, norm_err )

            if norm_err < tol:
                x0 = xk.copy()
                break

        rdrold = rdr
        rdr = r.dot(r)
        beta = rdr / rdrold

        #p = r + beta * p
        bp = p.copy()
        bp.mul(beta)
        p  = r.copy()
        p.add(bp)

    x0 = xk.copy()
    # ...

    return x0
# ....

#===============================================================================
#
# Copyright 2018 Yaman Güçlü
#
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
        Maximum number of iterations

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

#===============================================================================
del np
