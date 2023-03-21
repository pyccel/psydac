# coding: utf-8
"""
This module provides iterative solvers and precondionners.

"""
import numpy as np

from math import sqrt
from psydac.linalg.basic     import LinearSolver, LinearOperator
from psydac.linalg.utilities import _sym_ortho


__all__ = ['cg', 'pcg', 'bicg', 'lsmr', 'minres', 'jacobi', 'weighted_jacobi']

# ...
def cg( A, b, x0=None, tol=1e-6, maxiter=1000, verbose=False ):
    """
    Conjugate gradient algorithm for solving linear system Ax=b.
    Only working if A is an hermitian and positive-definite linear operator.
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
        print( template.format( 1, sqrt( am.real ) ) )

    m = 1
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
            print( template.format( m, sqrt( am.real ) ) )

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': m, 'success': am < tol_sqr, 'res_norm': sqrt( am.real ) }

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

    pc: NoneType | str | psydac.linalg.basic.LinearSolver | Callable
        Preconditioner for A, it should approximate the inverse of A.
        Can either be:
        * None, i.e. not pre-conditioning (this calls the standard `cg` method)
        * The strings 'jacobi' or 'weighted_jacobi'. (rather obsolete, supply a callable instead, if possible)
        * A LinearSolver object (in which case the out parameter is used)
        * A callable with two parameters (A, r), where A is the LinearOperator from above, and r is the residual.

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
        print( template.format(1, sqrt(nrmr_sqr.real)))

    k = 1
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
            print( template.format(k, sqrt(nrmr_sqr.real)))

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': k, 'success': nrmr_sqr < tol_sqr, 'res_norm': sqrt(nrmr_sqr.real) }

    return x, info
# ...

# ...
def jacobi(A, b):
    """
    Jacobi preconditioner.
    In case A is None we return a zero vector of the same dimensions as b

    Parameters
    ----------
    A : psydac.linalg.stencil.StencilMatrix | psydac.linalg.block.BlockLinearOperator
        Left-hand-side matrix A of linear system.

    b : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
        Right-hand-side vector of linear system.

    Returns
    -------
    x : psydac.linalg.stencil.StencilVector | psydac.linalg.block.BlockVector
        Preconditioner solution

    """
    from psydac.linalg.block   import BlockLinearOperator, BlockVector
    from psydac.linalg.stencil import StencilMatrix, StencilVector

    # In case A is None we return a zero vector
    if A is None:
        return b.space.zeros()

    # Sanity checks
    assert isinstance(A, (StencilMatrix, BlockLinearOperator))
    assert isinstance(b, (StencilVector, BlockVector))
    assert A.codomain == A.domain
    assert A.codomain == b.space

    #-------------------------------------------------------------
    # Handle the case of a block linear system
    if isinstance(A, BlockLinearOperator):
        x = [jacobi(A[i, i], bi) for i, bi in enumerate(b.blocks)]
        return BlockVector(b.space, blocks=x)
    #-------------------------------------------------------------

    V = b.space
    i = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))

    x = b.copy()
    x[i] /= A.diagonal()
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
            print( template.format(k, sqrt(nrmr.real)))

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': k, 'success': nrmr < tol_sqr, 'res_norm': sqrt(nrmr.real) }

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
        v  = A .dot(p , out=v)
        vs = At.dot(ps, out=vs)
        #-----------------------

        # c := (rs, r)
        c = rs.dot(r)

        # a := (rs, r) / (ps, v)
        a = c / ps.dot(v)

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
        vs *= a.conj()
        rs -= vs

        # b := (rs, r)_{m+1} / (r, rs)_m
        b = rs.dot(r) / c

        # p := r + b*p
        p *= (b/a)
        p += r

        # ps := rs + b*ps
        ps *= b.conj()
        ps += rs

        # ||r||_2 := (r, r)
        res_sqr = r.dot( r )

        if verbose:
            print( template.format(m, sqrt(res_sqr.real)) )

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': m, 'success': res_sqr < tol_sqr, 'res_norm': sqrt( res_sqr.real ) }

    return x, info
# ...
def bicgstab(A, b, x0=None, tol=1e-6, maxiter=1000, verbose=False):
    """
    Biconjugate gradient stabilized method (BCGSTAB) algorithm for solving linear system Ax=b.
    Implementation from [1], page 175.

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
    [1] H. A. van der Vorst. Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the
    solution of nonsymmetric linear systems. SIAM J. Sci. Stat. Comp., 13(2):631–644, 1992

    TODO
    ----
    Add optional preconditioner

    """
    n = A.shape[0]

    assert A .shape == (n, n)
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
    vs  = 0.0 * b.copy()

    r0 = r.copy()
    s = 0.0 * r.copy()

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
        v  = A .dot(p , out=v)
        #-----------------------

        # c := (r0, r)
        c = r0.dot(r)

        # a := (r0, r) / (r0, v)
        a = c / (r0.dot(v))

        # s := r - a*v
        s *= 0
        v *= a
        s += r
        s -= v

        # vs :=  A*s
        vs = A.dot(s, out=vs)

        # w := (s, A*s) / (A*s, A*s)
        w = s.dot(vs) / vs.dot(vs)

        #-----------------------
        # SOLUTION UPDATE
        #-----------------------
        # x := x + a*p +w*s
        p *= a
        s *= w
        x += p
        x += s
        #-----------------------

        # r := s - w*vs
        vs *= w
        s *= 1/w
        r *= 0
        r += s
        r -= vs

        # ||r||_2 := (r, r)
        res_sqr = r.dot( r )

        if res_sqr<tol_sqr:
            break

        # b := a / w * (r0, r)_{m+1} / (r0, r)_m
        b = r0.dot(r)*a / (c * w)

        # p := r + b*p- b*w*v
        v *= (b*w/a)
        p *= (b/a)
        p -= v
        p += r

        if verbose:
            print( template.format(m, sqrt(res_sqr.real)) )

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': m, 'success': res_sqr < tol_sqr, 'res_norm': sqrt( res_sqr.real ) }

    return x, info
# ...

# ...
def minres(A, b, x0=None, tol=1e-6, maxiter=1000, verbose=False):
    """
    Use MINimum RESidual iteration to solve Ax=b

    MINRES minimizes norm(A*x - b) for a real symmetric matrix A.  Unlike
    the Conjugate Gradient method, A can be indefinite or singular.

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

    Notes
    -----
    This is an adaptation of the MINRES Solver in Scipy, where the method is modified to accept Psydac data structures,
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/sparse/linalg/isolve/minres.py

    References
    ----------
    Solution of sparse indefinite systems of linear equations,
        C. C. Paige and M. A. Saunders (1975),
        SIAM J. Numer. Anal. 12(4), pp. 617-629.
        https://web.stanford.edu/group/SOL/software/minres/

    """

    n = A.shape[0]

    assert A .shape == (n, n)
    assert b .shape == (n,)
    assert A.dtype ==float
    # First guess of solution
    if x0 is None:
        x = 0.0 * b.copy()
    else:
        assert x0.shape == (n,)
        x = x0.copy()

    istop = 0
    itn   = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0

    eps = np.finfo(b.dtype).eps

    res1 = b - A.dot(x)
    y  = res1

    beta = sqrt(res1.dot(res1))

    # Initialize other quantities
    oldb    = 0
    dbar    = 0
    epsln   = 0
    qrnorm  = beta
    phibar  = beta
    rhs1    = beta
    rhs2    = 0
    tnorm2  = 0
    gmax    = 0
    gmin    = np.finfo(b.dtype).max
    cs      = -1
    sn      = 0
    w       = 0.0 * b.copy()
    w2      = 0.0 * b.copy()
    res2    = res1

    if verbose:
        print( "MINRES solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"


    for itn in range(1, maxiter + 1 ):

        s = 1.0/beta
        v = s*y
        y = A.dot(v)

        if itn >= 2:y = y - (beta/oldb)*res1

        alfa = v.dot(y)
        res1 = res2
        y    = y - (alfa/beta)*res2
        res2 = y
        oldb = beta
        beta = sqrt(y.dot(y))
        tnorm2 += alfa**2 + oldb**2 + beta**2

        # Apply previous rotation Qk-1 to get
        #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

        oldeps = epsln
        delta  = cs * dbar + sn * alfa      # delta1 = 0         deltak
        gbar   = sn * dbar - cs * alfa      # gbar 1 = alfa1     gbar k
        epsln  = sn * beta                  # epsln2 = 0         epslnk+1
        dbar   = - cs * beta                # dbar 2 = beta2     dbar k+1
        root   = sqrt(gbar**2 + dbar**2)
        Arnorm = phibar * root

        # Compute the next plane rotation Qk

        gamma  = sqrt(gbar**2 + beta**2)  # gammak
        gamma  = max(gamma, eps)
        cs     = gbar / gamma                # ck
        sn     = beta / gamma                # sk
        phi    = cs * phibar                 # phik
        phibar = sn * phibar                 # phibark+1

        # Update  x.

        denom = 1.0/gamma
        w1    = w2
        w2    = w
        w     = (v - oldeps*w1 - delta*w2) * denom
        x     = x + phi*w

        # Go round again.

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z    = rhs1 / gamma
        rhs1 = rhs2 - delta*z
        rhs2 = - epsln*z

        # Estimate various norms and test for convergence.

        Anorm = sqrt(tnorm2)
        ynorm = sqrt(x.dot(x))
        epsa  = Anorm * eps
        epsx  = Anorm * ynorm * eps
        epsr  = Anorm * ynorm * tol
        diag  = gbar

        if diag == 0:diag = epsa

        rnorm  = phibar
        if ynorm == 0 or Anorm == 0:test1 = inf
#        else:test1 = rnorm / (Anorm*ynorm)  # ||r||  / (||A|| ||x||)
        else:test1 = rnorm                   # ||r||

        if Anorm == 0:test2 = inf
        else:test2 = root / Anorm           # ||Ar|| / (||A|| ||r||)

        # Estimate  cond(A).
        # In this version we look at the diagonals of  R  in the
        # factorization of the lower Hessenberg matrix,  Q * H = R,
        # where H is the tridiagonal matrix from Lanczos with one
        # extra row, beta(k+1) e_k^T.

        Acond = gmax/gmin

        if verbose:
            print( template.format(itn, rnorm ))

        # See if any of the stopping criteria are satisfied.
        if istop == 0:
            t1 = 1 + test1      # These tests work if tol < eps
            t2 = 1 + test2
            if t2 <= 1:istop = 2
            if t1 <= 1:istop = 1

            if Acond >= 0.1/eps:istop = 4

            if test2 <= tol:istop = 2
            if test1 <= tol:istop = 1

        if istop != 0:
            break

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': itn, 'success': rnorm<tol, 'res_norm': rnorm }
    return x, info
# ...

# ...
def lsmr(A, At, b, x0=None, tol=None, atol=None, btol=None, maxiter=1000, conlim=1e8, verbose=False):
    """Iterative solver for least-squares problems.
    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    ``A`` is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. ``b`` is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).

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

    atol : float
        Absolute tolerance for 2-norm of residual r = A*x - b.

    btol : float
        Relative tolerance for 2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    conlim : float
        lsmr terminates if an estimate of cond(A) exceeds
        conlim.

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

    Notes
    -----
    This is an adaptation of the LSMR Solver in Scipy, where the method is modified to accept Psydac data structures,
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/sparse/linalg/isolve/lsmr.py

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders,
           "LSMR: An iterative algorithm for sparse least-squares problems",
           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
           :arxiv:`1006.0758`
    .. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/
    """

    m, n = A.shape

    if atol is None:atol = 1e-6
    if btol is None:btol = 1e-6
    if tol is not None: 
        atol = tol
        btol = tol
        
    u = b
    normb = sqrt((b.dot(b)).real)
    if x0 is None:
        if not isinstance(A, np.ndarray):
            x = A.domain.zeros()
        else:
            x = np.zeros(n, dtype=A.dtype)
        beta = normb
    else:
        x = x0.copy()
        assert x0.shape == (n,)
        u = u - A.dot(x)
        beta = sqrt(u.dot(u))

    if beta > 0:
        u = (1 / beta) * u
        v = At.dot(u)
        alpha = sqrt((v.dot(v)).real)
    else:
        v = x.copy()
        alpha = 0

    if alpha > 0:v = (1 / alpha) * v

    # Initialize variables for 1st iteration.

    itn      = 0
    zetabar  = alpha * beta
    alphabar = alpha
    rho      = 1
    rhobar   = 1
    cbar     = 1
    sbar     = 0

    h    = v.copy()
    hbar = 0. * x.copy()

    # Initialize variables for estimation of ||r||.

    betadd      = beta
    betad       = 0
    rhodold     = 1
    tautildeold = 0
    thetatilde  = 0
    zeta        = 0
    d           = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2  = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA   = sqrt(normA2)
    condA   = 1
    normx   = 0

    # Items for use in stopping rules, normb set earlier
    istop = 0
    ctol  = 0
    if conlim > 0:ctol = 1 / conlim
    normr = beta

    # Reverse the order here from the original matlab code because
    normar = alpha * beta

    if verbose:
        print( "LSMR solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Main iteration loop.
    for itn in range(1, maxiter + 1):

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  a*v   -  alpha*u,
        #        alpha*v  =  A'*u  -  beta*v.

        u *= -alpha
        u += A.dot(v)
        beta = sqrt((u.dot(u)).real)

        if beta > 0:
            u     *= (1 / beta)
            v     *= -beta
            v     += At.dot(u)
            alpha = sqrt((v.dot(v)).real)
            if alpha > 0:v *= (1 / alpha)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = _sym_ortho(alphabar, 0.)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold    = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew  = s*alpha
        alphabar  = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold          = rhobar
        zetaold            = zeta
        thetabar           = sbar * rho
        rhotemp            = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta               = cbar * zetabar
        zetabar            = - sbar * zetabar

        # Update h, h_hat, x.

        hbar *= - (thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x    += (zeta / (rho * rhobar)) * hbar
        h    *= - (thetanew / rho)
        h    += v

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold                     = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde                        = stildeold * rhobar
        rhodold                           = ctildeold * rhobar
        betad                             = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud        = (zeta - thetatilde * tautildeold) / rhodold
        d           = d + betacheck * betacheck
        normr       = sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA  = sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx  = sqrt((x.dot(x)).real)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        if (normA * normr) != 0:test2 = normar / (normA * normr)
        else:test2 = np.infty
        test3 = 1 / condA
        t1    = test1 / (1 + normA * normx / normb)
        rtol  = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:istop = 7
        if 1 + test3 <= 1:istop = 6
        if 1 + test2 <= 1:istop = 5
        if 1 + t1 <= 1:istop = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol:istop = 3
        if test2 <= atol:istop = 2
        if test1 <= rtol:istop = 1

        if verbose:
            print( template.format(itn, normr ))

        if istop > 0:
            break


    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': itn, 'success': istop in [1,2,3], 'res_norm': normr }
    return x, info
# ...
