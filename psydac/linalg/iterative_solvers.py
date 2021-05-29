# coding: utf-8
"""
This module provides iterative solvers and precondionners.

"""
from math import sqrt
from psydac.linalg.basic import LinearSolver

__all__ = ['cg', 'pcg', 'bicg', 'jacobi', 'weighted_jacobi']

# ...
def cg( A, b, x0=None, tol=1e-6, maxiter=1000, verbose=False ):
    """
    Conjugate gradient algorithm for solving linear system Ax=b.
    Implementation from [1], page 137.

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    b : psydac.linalg.basic.Vector
        Right-hand-side vector of linear system. Individual entries b[i] need
        not be accessed, but b has 'shape' attribute and provides 'copy()' and
        'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
        scalar multiplication and sum operations are available.

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for L2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, L2-norm of residual r is printed at each iteration.

    Results
    -------
    x : psydac.linalg.basic.Vector
        Converged solution.

    info : dict
        Dictionary containing convergence information:
          - 'niter'    = (int) number of iterations
          - 'success'  = (boolean) whether convergence criteria have been met
          - 'res_norm' = (float) 2-norm of residual vector r = A*x - b.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    n = A.shape[0]

    assert( A.shape == (n,n) )
    assert( b.shape == (n, ) )

    # First guess of solution
    if x0 is None:
        x  = b.copy()
        x *= 0.0
    else:
        assert( x0.shape == (n,) )
        x = x0.copy()

    # First values
    v  = A.dot(x)
    r  = b - v
    am = r.dot( r )
    p  = r.copy()

    tol_sqr = tol**2

    if verbose:
        print( "CG solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"
        print( template.format( 1, sqrt( am ) ) )

    # Iterate to convergence
    for m in range( 2, maxiter+1 ):

        if am < tol_sqr:
            m -= 1
            break

        v   = A.dot(p, out=v)
        l   = am / v.dot( p )
        x  += l*p
        r  -= l*v
        am1 = r.dot( r )
        p  *= (am1/am)
        p  += r
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
def pcg(A, b, pc, x0=None, tol=1e-6, maxiter=1000, verbose=False):
    """
    Preconditioned Conjugate Gradient (PCG) solves the symetric positive definte
    system Ax = b. It assumes that pc(r) returns the solution to Ps = r,
    where P is positive definite.

    Parameters
    ----------
    A : psydac.linalg.stencil.StencilMatrix
        Left-hand-side matrix A of linear system

    b : psydac.linalg.stencil.StencilVector
        Right-hand-side vector of linear system.

    pc: string
        Preconditioner for A, it should approximate the inverse of A.
        "jacobi" and "weighted_jacobi" are available in this module.

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for L2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, L2-norm of residual r is printed at each iteration.

    Returns
    -------
    x : psydac.linalg.basic.Vector
        Converged solution.

    """
    from math import sqrt

    n = A.shape[0]

    assert( A.shape == (n,n) )
    assert( b.shape == (n, ) )

    # First guess of solution
    if x0 is None:
        x  = b.copy()
        x *= 0.0
    else:
        assert( x0.shape == (n,) )
        x = x0.copy()

    # Preconditioner
    if pc is None:
        # for now, call the cg method here
        return cg(A, b, x0=x0, tol=tol, maxiter=maxiter, verbose=verbose)
    elif isinstance(pc, str):
        pcfun = globals()[pc]
        psolve = lambda r: pcfun(A, r)
    elif isinstance(pc, LinearSolver):
        s = b.space.zeros()
        psolve = lambda r: pc.solve(r, out=s)
    elif hasattr(pc, '__call__'):
        psolve = lambda r: pc(A, r)

    # First values
    v = A.dot(x)
    r = b - v
    nrmr_sqr = r.dot(r)

    s  = psolve(r)
    am = s.dot(r)
    p  = s.copy()

    tol_sqr = tol**2

    if verbose:
        print( "Pre-conditioned CG solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"
        print( template.format(1, sqrt(nrmr_sqr)))

    # Iterate to convergence
    for k in range(2, maxiter+1):

        if nrmr_sqr < tol_sqr:
            k -= 1
            break

        v  = A.dot(p, out=v)
        l  = am / v.dot(p)
        x += l*p
        r -= l*v

        nrmr_sqr = r.dot(r)
        s = psolve(r)

        am1 = s.dot(r)
        p  *= (am1/am)
        p  += s
        am  = am1

        if verbose:
            print( template.format(k, sqrt(nrmr_sqr)))

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': k, 'success': nrmr_sqr < tol_sqr, 'res_norm': sqrt(nrmr_sqr) }

    return x, info
# ...

# ...
def jacobi(A, b):
    """
    Jacobi preconditioner.
    ----------
    A : psydac.linalg.stencil.StencilMatrix | psydac.linalg.block.BlockMatrix
        Left-hand-side matrix A of linear system.

    b : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
        Right-hand-side vector of linear system.

    Returns
    -------
    x : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
        Preconditioner solution

    """
    from psydac.linalg.block   import BlockMatrix, BlockVector
    from psydac.linalg.stencil import StencilMatrix, StencilVector

    # Sanity checks
    assert isinstance(A, (StencilMatrix, BlockMatrix))
    assert isinstance(b, (StencilVector, BlockVector))
    assert A.codomain == A.domain
    assert A.codomain == b.space

    #-------------------------------------------------------------
    # Handle the case of a block linear system
    if isinstance(A, BlockMatrix):
        x = [jacobi(A[i, i], bi) for i, bi in enumerate(b.blocks)]
        return BlockVector(b.space, blocks=x)
    #-------------------------------------------------------------

    V = b.space
    i = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
    ii = i + (0,) * V.ndim

    x = b.copy()
    x[i] /= A[ii]
    x.update_ghost_regions()

    return x
# ...

# ...
def weighted_jacobi(A, b, x0=None, omega= 2./3, tol=1e-10, maxiter=100, verbose=False):
    """
    Weighted Jacobi iterative preconditioner.

    Parameters
    ----------
    A : psydac.linalg.stencil.StencilMatrix
        Left-hand-side matrix A of linear system.

    b : psydac.linalg.stencil.StencilVector
        Right-hand-side vector of linear system.

    x0 : psydac.linalg.basic.Vector
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
    x : psydac.linalg.stencil.StencilVector
        Converged solution.

    """
    from math import sqrt
    from psydac.linalg.stencil import StencilVector

    n = A.shape[0]

    assert(A.shape == (n,n))
    assert(b.shape == (n, ))

    V  = b.space
    s = V.starts
    e = V.ends

    # First guess of solution
    if x0 is None:
        x = 0.0 * b.copy()
    else:
        assert( x0.shape == (n,) )
        x = x0.copy()

    dr = 0.0 * b.copy()
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

        # TODO build new external method get_diagonal and add 3d case
        if V.ndim ==1:
            for i1 in range(s[0], e[0]+1):
                dr[i1] = omega*r[i1]/A[i1, 0]

        elif V.ndim ==2:
            for i1 in range(s[0], e[0]+1):
                for i2 in range(s[1], e[1]+1):
                    dr[i1, i2] = omega*r[i1, i2]/A[i1, i2, 0, 0]
        # ...
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

# ...
def bicg(A, At, b, x0=None, tol=1e-6, maxiter=1000, verbose=False):
    """
    Biconjugate gradient (BCG) algorithm for solving linear system Ax=b.
    Implementation from [1], page 175.

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    At : psydac.linalg.basic.LinearOperator
        Matrix transpose of A, with 'shape' attribute and 'dot(p)' function.

    b : psydac.linalg.basic.Vector
        Right-hand-side vector of linear system. Individual entries b[i] need
        not be accessed, but b has 'shape' attribute and provides 'copy()' and
        'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
        scalar multiplication and sum operations are available.

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for 2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, 2-norm of residual r is printed at each iteration.

    Results
    -------
    x : psydac.linalg.basic.Vector
        Numerical solution of linear system.

    info : dict
        Dictionary containing convergence information:
          - 'niter'    = (int) number of iterations
          - 'success'  = (boolean) whether convergence criteria have been met
          - 'res_norm' = (float) 2-norm of residual vector r = A*x - b.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    TODO
    ----
    Add optional preconditioner

    """
    n = A.shape[0]

    assert A .shape == (n, n)
    assert At.shape == (n, n)
    assert b .shape == (n,)

    # First guess of solution
    if x0 is None:
        x = 0.0 * b.copy()
    else:
        assert x0.shape == (n,)
        x = x0.copy()

    # First values
    r  = b - A.dot( x )
    p  = r.copy()
    v  = 0.0 * b.copy()

    rs = r.copy()
    ps = p.copy()
    vs = 0.0 * b.copy()

    res_sqr = r.dot(r)
    tol_sqr = tol**2

    if verbose:
        print( "BiCG solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Iterate to convergence
    for m in range(1, maxiter + 1):

        if res_sqr < tol_sqr:
            m -= 1
            break

        #-----------------------
        # MATRIX-VECTOR PRODUCTS
        #-----------------------
        A .dot(p , out=v )
        At.dot(ps, out=vs)
        #-----------------------

        # c := (r, rs)
        c = r.dot(rs)

        # a := (r, rs) / (v, ps)
        a = c / v.dot(ps)

        #-----------------------
        # SOLUTION UPDATE
        #-----------------------
        # x := x + a*p
        p *= a
        x += p
        #-----------------------

        # r := r - a*v
        v *= a
        r -= v

        # rs := rs - a*vs
        vs *= a
        rs -= vs

        # b := (r, rs)_{m+1} / (r, rs)_m
        b = r.dot(rs) / c

        # p := r + b*p
        p *= (b/a)
        p += r

        # ps := rs + b*ps
        ps *= b
        ps += rs

        # ||r||_2 := (r, r)
        res_sqr = r.dot( r )

        if verbose:
            print( template.format(m, sqrt(res_sqr)) )

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': m, 'success': res_sqr < tol_sqr, 'res_norm': sqrt( res_sqr ) }

    return x, info
# ...
