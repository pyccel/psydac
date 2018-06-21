# coding: utf-8
"""
    This modile provides iterative solvers and precond ...
"""

__all__ = ['cg','pcg', 'weighted_jacobi']

# ...
def cg( A, b, x0=None, tol=1e-6, maxiter=1000, verbose=False ):
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
# ...

# ...
def pcg(A, b, pc='weighted_jacobi', x0=None, tol=1e-6, maxiter=1000,
        verbose=True):
    """
    Preconditioned Conjugate Gradient (PCG) solves the symetric positive definte
    system Ax = b. It assumes that pc(r) returns the solution to Ps = r,
    where P is positive definite.

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

    pc: string
        Preconditioner for A, it should approximate the inverse of A. Default
        vlaue is "weighted_jacobi".

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

    """

    from math import sqrt

    psolve = eval(pc)

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
    r = b - A.dot(x)

    nrmr0 = sqrt(r.dot(r))

    s = psolve(A, r)
    p = s
    sr = s.dot(r)

    if verbose:
        print( "CG solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Iterate to convergence
    for k in range(1, maxiter+1):

        q = A.dot(p)
        alpha  = sr / p.dot(q)

        x  = x + alpha*p
        r  = r - alpha*q

        s = A.dot(r)

        nrmr = r.dot(r)

        if nrmr < tol*nrmr0:
            k -= 1
            break

        s = psolve(A, r)

        srold = sr
        sr = s.dot(r)

        beta = sr/srold

        p = s + beta*p

        if verbose:
            print( template.format(k, sqrt(nrmr)))

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': k, 'success': nrmr < tol*nrmr0, 'res_norm': sqrt(nrmr) }

    return x, info
# ...

# ...
def weighted_jacobi(A, b, x0=None, omega= 2./3, tol=1e-6, maxiter=100, verbose=False):
    """

    Preconditioning  improves the rate of convergence, which implies that fewer iterations are needed to reach a given error tolerance.
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

    omega : float
        The weight parameter (optional). Default value equal to 2/3.

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

    """
    from math import sqrt
    from spl.linalg.stencil import StencilVector

    n = A.shape[0]

    assert(A.shape == (n,n))
    assert(b.shape == (n, ))

    V  = b.space
    s = V.starts
    e = V.ends

    dr = 0.0 * b.copy()

    # First guess of solution
    if x0 is None:
        x = 0.0 * b.copy()
    else:
        assert( x0.shape == (n,) )
        x = x0.copy()

    tol_sqr = tol**2
    if verbose:
        print( "Weighted Jacobi iterative method:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Iterate to convergence
    for k in range(1, maxiter+1):

        r = b - A.dot(x)

        # TODO build new external method get_diagonaland add 3d
        if V.ndim ==1:
            dr[s[0]:e[0]+1] = omega*r[s[0]:e[0]+1]/A[s[0]:e[0]+1, 0]
        elif V.ndim ==2:
            dr[s[0]:e[0]+1, s[1]:e[1]+1] = omega*r[s[0]:e[0]+1, s[1]:e[1]+1]/A[s[0]:e[0]+1, s[1]:e[1]+1, 0, 0]

        dr.update_ghost_regions()

        x  = x + dr

        nrmr = dr.dot(dr)
        if nrmr < tol_sqr:
            k -= 1
            break


        if verbose:
            print( template.format(k, sqrt(nrmr)))

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': k, 'success': nrmr < tol_sqr, 'res_norm': sqrt(nrmr) }

    return x
# ...
