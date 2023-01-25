# coding: utf-8
"""
This module provides iterative solvers and precondionners.

"""
from math import sqrt
import numpy as np

from psydac.linalg.basic     import Vector, LinearOperator, InverseLinearOperator

__all__ = ['ConjugateGradient', 'PConjugateGradient', 'BiConjugateGradient', 'MinimumResidual']

#===============================================================================
class ConjugateGradient(InverseLinearOperator):
    """
    

    """
    def __init__(self, linop, *, x0=None, tol=1e-6, maxiter=1000, verbose=False):

        assert isinstance(linop, LinearOperator)
        assert linop.domain == linop.codomain
        self._solver = 'cg'
        self._linop = linop
        self._domain = linop.codomain
        self._codomain = linop.domain
        self._space = linop.domain
        self._x0 = x0
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose
        self._options = {"x0":self._x0, "tol":self._tol, "maxiter": self._maxiter, "verbose": self._verbose}
        self._check_options(**self._options)
        self._tmps = {"v":linop.domain.zeros(), "r":linop.domain.zeros(), "p":linop.domain.zeros(), 
                      "lp":linop.domain.zeros(), "lv":linop.domain.zeros()}

    def _check_options(self, **kwargs):
        keys = ('x0', 'tol', 'maxiter', 'verbose')
        for key, value in kwargs.items():
            idx = [key == keys[i] for i in range(len(keys))]
            assert any(idx), "key not supported, check options"
            true_idx = idx.index(True)
            if true_idx == 0:
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._domain, "x0 belongs to the wrong VectorSpace"
            elif true_idx == 1:
                assert value is not None, "tol may not be None"
                # don't know if that one works -want to check if value is a number
                assert value*0 == 0, "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif true_idx == 2:
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif true_idx == 3:
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"

    def _update_options(self):
        self._options = {"x0":self._x0, "tol":self._tol, "maxiter": self._maxiter, "verbose": self._verbose}

    def solve(self, b, out=None):
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

        A = self._linop
        n = A.shape[0]
        x0 = self._x0
        tol = self._tol
        maxiter = self._maxiter
        verbose = self._verbose
        

        assert( A.shape == (n,n) )
        assert( b.shape == (n, ) )

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._domain
            out *= 0
            if x0 is None:
                x = out
            else:
                assert( x0.shape == (n,) )
                out += x0
                x = out
        else:
            if x0 is None:
                x  = b.copy()
                x *= 0.0
            else:
                assert( x0.shape == (n,) )
                x = x0.copy()

        # Extract local storage
        v = self._tmps["v"]
        r = self._tmps["r"]
        p = self._tmps["p"]
        # Not strictly needed by the conjugate gradient, but necessary to avoid temporaries
        lp = self._tmps["lp"]
        lv = self._tmps["lv"]

        # First values
        A.dot(x, out=v)
        b.copy(out=r)
        r -= v
        am = r.dot( r )
        r.copy(out=p)

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
            A.dot(p, out=v)
            l   = am / v.dot( p )
            p.copy(out=lp)
            lp *= l
            x  += lp # this was x += l*p
            v.copy(out=lv)
            lv *= l
            r  -= lv # this was r -= l*v
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

        return x#, info

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class PConjugateGradient(InverseLinearOperator):
    """
    

    """
    def __init__(self, linop, *, pc=None, x0=None, tol=1e-6, maxiter=1000, verbose=False):

        assert isinstance(linop, LinearOperator)
        assert linop.domain == linop.codomain
        self._solver = 'pcg'
        self._linop = linop
        self._domain = linop.codomain
        self._codomain = linop.domain
        self._space = linop.domain
        self._pc = pc
        self._x0 = x0
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose
        self._options = {"pc": self._pc, "x0": self._x0, "tol": self._tol, "maxiter": self._maxiter, "verbose": self._verbose}
        self._check_options(**self._options)
        self._tmps = {"v":linop.domain.zeros(), "r":linop.domain.zeros(), "p":linop.codomain.zeros(), 
                      "s":linop.codomain.zeros(), "lp":linop.codomain.zeros(), "lv":linop.domain.zeros()}

    def _check_options(self, **kwargs):
        keys = ('pc', 'x0', 'tol', 'maxiter', 'verbose')
        for key, value in kwargs.items():
            idx = [key == keys[i] for i in range(len(keys))]
            assert any(idx), "key not supported, check options"
            true_idx = idx.index(True)
            if true_idx == 0:
                assert value is not None, "pc may not be None"
                assert value == 'jacobi' or value == 'weighted_jacobi', "unsupported preconditioner"
            elif true_idx == 1:
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._domain, "x0 belongs to the wrong VectorSpace"
            elif true_idx == 2:
                assert value is not None, "tol may not be None"
                # don't know if that one works -want to check if value is a number
                assert value*0 == 0, "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif true_idx == 3:
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif true_idx == 4:
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"

    def _update_options( self ):
        self._options = {"pc": self._pc, "x0": self._x0, "tol": self._tol, "maxiter": self._maxiter, "verbose": self._verbose}

    def solve(self, b, out=None):
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

        A = self._linop
        n = A.shape[0]
        pc = self._pc
        x0 = self._x0
        tol = self._tol
        maxiter = self._maxiter
        verbose = self._verbose

        assert( A.shape == (n,n) )
        assert( b.shape == (n, ) )

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._domain
            out *= 0
            if x0 is None:
                x  = out
            else:
                assert( x0.shape == (n,) )
                out += x0
                x = out
        else:
            if x0 is None:
                x  = b.copy()
                x *= 0.0
            else:
                assert( x0.shape == (n,) )
                x = x0.copy()

        # Preconditioner
        assert pc is not None
        if pc == 'jacobi':
            psolve = lambda r, out: InverseLinearOperator.jacobi(A, r, out)
        elif pc == 'weighted_jacobi':
            psolve = lambda r: InverseLinearOperator.weighted_jacobi(A, r) # allows for further specification not callable like this!
        #elif isinstance(pc, str):
        #    pcfun = getattr(InverseLinearOperator, str)
        #    #pcfun = globals()[pc]
        #    psolve = lambda r: pcfun(A, r)
        #elif isinstance(pc, LinearSolver):
        #    s = b.space.zeros()
        #    psolve = lambda r: pc.solve(r, out=s)
        #elif hasattr(pc, '__call__'):
        #    psolve = lambda r: pc(A, r)

        # Extract local storage
        v = self._tmps["v"]
        r = self._tmps["r"]
        p = self._tmps["p"]
        s = self._tmps["s"]
        # Not strictly needed by the conjugate gradient, but necessary to avoid temporaries
        lp = self._tmps["lp"]
        lv = self._tmps["lv"]

        # First values
        A.dot(x, out=v)
        b.copy(out=r)
        r -= v
        nrmr_sqr = r.dot(r)
        psolve(r, out=s)
        am = s.dot(r)
        s.copy(out=p)

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
            p.copy(out=lp)
            lp *= l
            x  += lp # this was x += l*p
            v.copy(out=lv)
            lv *= l
            r  -= lv # this was r -= l*v

            nrmr_sqr = r.dot(r)
            psolve(r, out=s)

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
        return x#, info

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class BiConjugateGradient(InverseLinearOperator):
    """
    

    """
    def __init__(self, linop, *, x0=None, tol=1e-6, maxiter=1000, verbose=False):

        assert isinstance(linop, LinearOperator)
        assert linop.domain == linop.codomain
        self._solver = 'bicg'
        self._linop = linop
        self._domain = linop.codomain
        self._codomain = linop.domain
        self._space = linop.domain
        self._x0 = x0
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose
        self._options = {"x0":self._x0, "tol":self._tol, "maxiter": self._maxiter, "verbose": self._verbose}
        self._check_options(**self._options)
        self._tmps = {"v":linop.domain.zeros(), "r":linop.domain.zeros(), "p":linop.domain.zeros(), 
                      "vs":linop.domain.zeros(), "rs":linop.domain.zeros(), "ps":linop.domain.zeros()}

    def _check_options(self, **kwargs):
        keys = ('x0', 'tol', 'maxiter', 'verbose')
        for key, value in kwargs.items():
            idx = [key == keys[i] for i in range(len(keys))]
            assert any(idx), "key not supported, check options"
            true_idx = idx.index(True)
            if true_idx == 0:
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._domain, "x0 belongs to the wrong VectorSpace"
            elif true_idx == 1:
                assert value is not None, "tol may not be None"
                # don't know if that one works -want to check if value is a number
                assert value*0 == 0, "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif true_idx == 2:
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif true_idx == 3:
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"

    def _update_options( self ):
        self._options = {"x0":self._x0, "tol":self._tol, "maxiter": self._maxiter, "verbose": self._verbose}

    def solve(self, b, out=None):
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
        A = self._linop
        At = A.T
        n = A.shape[0]
        x0 = self._x0
        tol = self._tol
        maxiter = self._maxiter
        verbose = self._verbose

        assert A .shape == (n, n)
        assert At.shape == (n, n)
        assert b .shape == (n,)

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._domain
            out *= 0
            if x0 is None:
                x  = out
            else:
                assert( x0.shape == (n,) )
                out += x0
                x = out
        else:
            if x0 is None:
                x  = b.copy()
                x *= 0.0
            else:
                assert( x0.shape == (n,) )
                x = x0.copy()

        # Extract local storage
        v = self._tmps["v"]
        r = self._tmps["r"]
        p = self._tmps["p"]
        vs = self._tmps["vs"]
        rs = self._tmps["rs"]
        ps = self._tmps["ps"]

        # First values
        A.dot(x, out=v)
        b.copy(out=r)
        r -= v
        r.copy(out=p)
        v *= 0

        r.copy(out=rs)
        p.copy(out=ps)
        v.copy(out=vs)

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
            A.dot(p, out=v)
            At.dot(ps, out=vs)
            #v  = A.dot(p , out=v) # overwriting v, then saving in v. Necessary?
            #vs = At.dot(ps, out=vs) # same story
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
            p *= (b/a) # *= (b/a) why a? or update description
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

        return x#, info

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class MinimumResidual(InverseLinearOperator):
    """
    

    """
    def __init__(self, linop, *, x0=None, tol=1e-6, maxiter=1000, verbose=False):

        assert isinstance(linop, LinearOperator)
        assert linop.domain == linop.codomain
        self._solver = 'minres'
        self._linop = linop
        self._domain = linop.codomain
        self._codomain = linop.domain
        self._space = linop.domain
        self._x0 = x0
        self._tol = tol
        self._maxiter = maxiter
        self._verbose = verbose
        self._options = {"x0":self._x0, "tol":self._tol, "maxiter": self._maxiter, "verbose": self._verbose}
        self._check_options(**self._options)
        self._tmps = {"rs":linop.domain.zeros(), "y":linop.domain.zeros(), "w":linop.domain.zeros(), 
                      "w2":linop.domain.zeros(), "res1":linop.domain.zeros(), "res2":linop.domain.zeros()}

    def _check_options(self, **kwargs):
        keys = ('x0', 'tol', 'maxiter', 'verbose')
        for key, value in kwargs.items():
            idx = [key == keys[i] for i in range(len(keys))]
            assert any(idx), "key not supported, check options"
            true_idx = idx.index(True)
            if true_idx == 0:
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._domain, "x0 belongs to the wrong VectorSpace"
            elif true_idx == 1:
                assert value is not None, "tol may not be None"
                # don't know if that one works -want to check if value is a number
                assert value*0 == 0, "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif true_idx == 2:
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif true_idx == 3:
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"

    def _update_options(self):
        self._options = {"x0":self._x0, "tol":self._tol, "maxiter": self._maxiter, "verbose": self._verbose}

    def solve(self, b, out=None):
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

        A = self._linop
        n = A.shape[0]
        x0 = self._x0
        tol = self._tol
        maxiter = self._maxiter
        verbose = self._verbose

        # Extract local storage
        rs = self._tmps["rs"]
        y = self._tmps["y"]
        w = self._tmps["w"]
        w2 = self._tmps["w2"]
        res1 = self._tmps["res1"]
        res2 = self._tmps["res2"]

        assert A .shape == (n, n)
        assert b .shape == (n,)

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._domain
            out *= 0
            if x0 is None:
                x  = out
            else:
                assert( x0.shape == (n,) )
                out += x0
                x = out
        else:
            if x0 is None:
                x  = b.copy()
                x *= 0.0
            else:
                assert( x0.shape == (n,) )
                x = x0.copy()

        istop = 0
        itn   = 0
        Anorm = 0
        Acond = 0
        rnorm = 0
        ynorm = 0

        eps = np.finfo(b.dtype).eps

        A.dot(x, out=rs)
        b.copy(out=res1)
        res1 -= rs
        #res1 = b - A.dot(x)
        res1.copy(out=y)
        #y  = res1

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
        b.copy(out=w)
        w *= 0
        b.copy(out=w2)
        w2 *= 0
        #w       = 0.0 * b.copy()
        #w2      = 0.0 * b.copy()
        res1.copy(out=res2)
        #res2    = res1

        if verbose:
            print( "MINRES solver:" )
            print( "+---------+---------------------+")
            print( "+ Iter. # | L2-norm of residual |")
            print( "+---------+---------------------+")
            template = "| {:7d} | {:19.2e} |"


        for itn in range(1, maxiter + 1 ):

            s = 1.0/beta
            v = s*y
            A.dot(v, out=y)
            #y = A.dot(v)

            if itn >= 2:y = y - (beta/oldb)*res1 # <--- pot. source of tmp

            alfa = v.dot(y)
            res2.copy(out=res1)
            #res1 = res2
            res2 *= alfa/beta
            y -= res2
            y.copy(out=res2)
            #y    = y - (alfa/beta)*res2
            #res2 = y
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

            w *= phi
            x += w
            #x     = x + phi*w

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
        #info = {'niter': itn, 'success': rnorm<tol, 'res_norm': rnorm }
        return x#, info

    def dot(self, b, out=None):
        return self.solve(b, out=out)