# coding: utf-8
"""
This module provides iterative solvers and precondionners.

"""
from psydac.linalg2.basic     import LinearSolver, LinearOperator, InverseLinearOperator


__all__ = ['ConjugateGradient', 'PConjugateGradient']

#===============================================================================
class ConjugateGradient( InverseLinearOperator ):
    """
    

    """
    def __init__( self, linop, *, solver=None, x0=None, tol=1e-6, maxiter=1000, verbose=False ):

        assert isinstance(linop, LinearOperator)
        assert linop.domain == linop.codomain
        self._linop = linop
        self._domain = linop.codomain
        self._codomain = linop.domain
        self._space = linop.domain
        self._x0 = x0
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose
        self._options = {"_x0":self._x0, "_tol":self._tol, "_maxiter": self._maxiter, "_verbose": self._verbose}

    @property
    def options( self ):
        return self._options

    def _update_options( self ):
        self._options = {"_x0":self._x0, "_tol":self._tol, "_maxiter": self._maxiter, "_verbose": self._verbose}

    @property
    def space( self ):
        return self._space

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return None

    def solve(self, b):
        """
        Conjugate gradient algorithm for solving linear system Ax=b.
        Implementation from [1], page 137.

        Parameters
        ----------
        A = self._operator : psydac.linalg.basic.LinearOperator
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
        from math import sqrt

        A = self._linop
        n = A.shape[0]
        x0 = self._x0
        tol = self._tol
        maxiter = self._maxiter
        verbose = self._verbose
        

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

    def dot(self, b):
        return self.solve(b)

#===============================================================================
class PConjugateGradient( InverseLinearOperator ):
    """
    

    """
    def __init__( self, linop, *, solver=None, pc=None, x0=None, tol=1e-6, maxiter=1000, verbose=False ):

        assert isinstance(linop, LinearOperator)
        assert linop.domain == linop.codomain
        self._operator = linop
        self._domain = linop.codomain
        self._codomain = linop.domain
        self._space = linop.domain
        self._pc = pc
        self._x0 = x0
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose

    @property
    def space( self ):
        return self._space

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return None

    def solve(self, b):
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

        A = self._operator
        n = A.shape[0]
        pc = self._pc
        x0 = self._x0
        tol = self._tol
        maxiter = self._maxiter
        verbose = self._verbose

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
        assert pc is not None
        if pc == 'jacobi':
            psolve = lambda r: InverseLinearOperator.jacobi(A, r)
        elif pc == 'weighted_jacobi':
            psolve = lambda r: InverseLinearOperator.weighted_jacobi(A, r) # allows for further specification not callable like this!
        elif isinstance(pc, str):
            pcfun = getattr(InverseLinearOperator, str)
            #pcfun = globals()[pc]
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

    def dot(self, b):
        return self.solve(b)