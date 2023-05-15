# coding: utf-8
"""
This module provides iterative solvers and precondionners.

"""
from math import sqrt
import numpy as np

from psydac.linalg.basic     import Vector, LinearOperator, InverseLinearOperator, IdentityOperator, ScaledLinearOperator
from psydac.linalg.utilities import _sym_ortho

__all__ = ['ConjugateGradient', 'PConjugateGradient', 'BiConjugateGradient', 'BiConjugateGradientStabilized', 'MinimumResidual', 'LSMR', 'GMRES']

def inverse(A, solver, **kwargs):
    """
    A function to create objects of all InverseLinearOperator subclasses.
    14.02.23: ConjugateGradient, PConjugateGradient, BiConjugateGradient, MinimumResidual, LSMR
    The ''kwargs given must be compatible with the chosen solver subclass, see
    :func:~`solvers.ConjugateGradient`
    :func:~`solvers.PConjugateGradient`
    :func:~`solvers.BiConjugateGradient`
    :func:~`solvers.MinimumResidual`
    :func:~`solvers.LSMR`
    :func:~`solvers.GMRES`
    
    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    solver : str
        14.02.23: Either 'cg', 'pcg', 'bicg', 'minres', 'lsmr' or 'gmres'
        Indicating the preferred iterative solver.

    Returns
    -------
    obj : psydac.linalg.basic.InverseLinearOperator
        More specifically: Returns the chosen subclass, for example psydac.linalg.solvers.ConjugateGradient
        A linear operator acting as the inverse of A.

    """
    # Check solver input
    solvers = ('cg', 'pcg', 'bicg', 'bicgstab', 'minres', 'lsmr', 'gmres')
    if solver not in solvers:
        raise ValueError(f"Required solver '{solver}' not understood.")

    assert isinstance(A, LinearOperator)
    if isinstance(A, IdentityOperator):
        return A
    elif isinstance(A, ScaledLinearOperator):
        return ScaledLinearOperator(domain=A.codomain, codomain=A.domain, c=1/A.scalar, A=inverse(A, solver, **kwargs))
    elif isinstance(A, InverseLinearOperator):
        return A.linop

    # Instantiate object of correct solver class
    if solver == 'cg':
        obj = ConjugateGradient(A, **kwargs)
    elif solver == 'pcg':
        obj = PConjugateGradient(A, **kwargs)
    elif solver == 'bicg':
        obj = BiConjugateGradient(A, **kwargs)
    elif solver == 'bicgstab':
        obj = BiConjugateGradientStabilized(A, **kwargs)
    elif solver == 'minres':
        obj = MinimumResidual(A, **kwargs)
    elif solver == 'lsmr':
        obj = LSMR(A, **kwargs)
    elif solver == 'gmres':
        obj = GMRES(A, **kwargs)
    return obj

#===============================================================================
class ConjugateGradient(InverseLinearOperator):
    """
    A LinearOperator subclass. Objects of this class are meant to be created using :func:~`solvers.inverse`.

    The .dot (and also the .solve) function are based on the 
    Conjugate gradient algorithm for solving linear system Ax=b.
    Implementation from [1], page 137.

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for L2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, L2-norm of residual r is printed at each iteration.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=1000, verbose=False):

        assert isinstance(A, LinearOperator)
        assert A.domain.dimension == A.codomain.dimension
        domain = A.codomain
        codomain = A.domain

        if x0 is not None:
            assert isinstance(x0, Vector)
            assert x0.space is codomain
        else:
            x0 = codomain.zeros()

        self._A = A
        self._domain = domain
        self._codomain = codomain
        self._solver = 'cg'
        self._options = {"x0":x0, "tol":tol, "maxiter":maxiter, "verbose":verbose}
        self._check_options(**self._options)
        self._tmps = {key: domain.zeros() for key in ("v", "r", "p", "lp", "lv")}
        self._info = None

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert value is not None, "tol may not be None"
                assert value*0 == 0, "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"
            else:
                raise ValueError(f"Key '{key}' not understood. See self._options for allowed keys.")

    def transpose(self, conjugate=False):
        At = self._A.transpose(conjugate=conjugate)
        solver = self._solver
        options = self._options
        return inverse(At, solver, **options)

    def solve(self, b, out=None):
        """
        Conjugate gradient algorithm for solving linear system Ax=b.
        Only working if A is an hermitian and positive-definite linear operator.
        Implementation from [1], page 137.
        Info can be accessed using get_info(), see :func:~`basic.InverseLinearOperator.get_info`.

        Parameters
        ----------
        b : psydac.linalg.basic.Vector
            Right-hand-side vector of linear system Ax = b. Individual entries b[i] need
            not be accessed, but b has 'shape' attribute and provides 'copy()' and
            'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
            scalar multiplication and sum operations are available.

        out : psydac.linalg.basic.Vector | NoneType
            The output vector, or None (optional).

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

        A = self._A
        domain = self._domain
        codomain = self._codomain
        options = self._options
        x0 = options["x0"]
        tol = options["tol"]
        maxiter = options["maxiter"]
        verbose = options["verbose"]
        
        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)

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
        am = r.dot(r).real
        r.copy(out=p)

        tol_sqr = tol**2

        if verbose:
            print( "CG solver:" )
            print( "+---------+---------------------+")
            print( "+ Iter. # | L2-norm of residual |")
            print( "+---------+---------------------+")
            template = "| {:7d} | {:19.2e} |"
            print(template.format(1, sqrt(am)))

        # Iterate to convergence
        for m in range(2, maxiter+1):
            if am < tol_sqr:
                m -= 1
                break
            A.dot(p, out=v)
            l   = am / v.dot(p)
            p.copy(out=lp)
            lp *= l
            x  += lp # this was x += l*p
            v.copy(out=lv)
            lv *= l
            r  -= lv # this was r -= l*v
            am1 = r.dot(r).real
            p  *= (am1/am)
            p  += r
            am  = am1
            if verbose:
                print(template.format(m, sqrt(am)))

        if verbose:
            print( "+---------+---------------------+")

        # Convergence information
        self._info = {'niter': m, 'success': am < tol_sqr, 'res_norm': sqrt(am) }

        return x

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class PConjugateGradient(InverseLinearOperator):
    """
    A LinearOperator subclass. Objects of this class are meant to be created using :func:~`solvers.inverse`.
    The .dot (and also the .solve) function are based on a preconditioned conjugate gradient method.

    Preconditioned Conjugate Gradient (PCG) solves the symetric positive definte
    system Ax = b. It assumes that pc(r) returns the solution to Ps = r,
    where P is positive definite.

    Parameters
    ----------
    A : psydac.linalg.stencil.StencilMatrix
        Left-hand-side matrix A of linear system

    pc: str
        Preconditioner for A, it should approximate the inverse of A.
        Can currently only be:
        * The strings 'jacobi' or 'weighted_jacobi'. (rather obsolete, supply a callable instead, if possible)(14.02.: test weighted_jacobi)
        The following should also be possible
        * None, i.e. not pre-conditioning (this calls the standard `cg` method)
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

    """
    def __init__(self, A, *, pc='jacobi', x0=None, tol=1e-6, maxiter=1000, verbose=False):

        assert isinstance(A, LinearOperator)
        assert A.domain.dimension == A.codomain.dimension
        domain = A.codomain
        codomain = A.domain

        if x0 is not None:
            assert isinstance(x0, Vector)
            assert x0.space is codomain
        else:
            x0 = codomain.zeros()

        self._A = A
        self._domain = domain
        self._codomain = codomain
        self._solver = 'pcg'
        self._options = {"x0":x0, "pc":pc, "tol":tol, "maxiter":maxiter, "verbose":verbose}
        self._check_options(**self._options)
        tmps_codomain = {key: codomain.zeros() for key in ("p", "s", "lp")}
        tmps_domain = {key: domain.zeros() for key in ("v", "r", "lv")}
        self._tmps = {**tmps_codomain, **tmps_domain}
        self._info = None

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'pc':
                assert value is not None, "pc may not be None"
                assert value == 'jacobi', "unsupported preconditioner"
            elif key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert value is not None, "tol may not be None"
                assert value*0 == 0, "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"
            else:
                raise ValueError(f"Key '{key}' not understood. See self._options for allowed keys.")

    def transpose(self, conjugate=False):
        At = self._A.transpose(conjugate=conjugate)
        solver = self._solver
        options = self._options
        return inverse(At, solver, **options)

    def solve(self, b, out=None):
        """
        Preconditioned Conjugate Gradient (PCG) solves the symetric positive definte
        system Ax = b. It assumes that pc(r) returns the solution to Ps = r,
        where P is positive definite.
        Info can be accessed using get_info(), see :func:~`basic.InverseLinearOperator.get_info`.

        Parameters
        ----------
        b : psydac.linalg.stencil.StencilVector
            Right-hand-side vector of linear system.

        out : psydac.linalg.basic.Vector | NoneType
            The output vector, or None (optional).

        Returns
        -------
        x : psydac.linalg.basic.Vector
            Converged solution.

        info : dict
            Dictionary containing convergence information:
            - 'niter'    = (int) number of iterations
            - 'success'  = (boolean) whether convergence criteria have been met
            - 'res_norm' = (float) 2-norm of residual vector r = A*x - b.

        """

        A = self._A
        domain = self._domain
        codomain = self._codomain
        options = self._options
        x0 = options["x0"]
        pc = options["pc"]
        tol = options["tol"]
        maxiter = options["maxiter"]
        verbose = options["verbose"]

        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)

        # Preconditioner
        assert pc is not None
        if pc == 'jacobi':
            psolve = lambda r, out: InverseLinearOperator.jacobi(A, r, out)
        #elif pc == 'weighted_jacobi':
        #    psolve = lambda r, out: InverseLinearOperator.weighted_jacobi(A, r, out) # allows for further specification not callable like this!
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
        r       -= v
        nrmr_sqr = r.dot(r).real
        psolve(r, out=s)
        am       = s.dot(r)
        s.copy(out=p)

        tol_sqr  = tol**2

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

            nrmr_sqr = r.dot(r).real
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
        self._info = {'niter': k, 'success': nrmr_sqr < tol_sqr, 'res_norm': sqrt(nrmr_sqr) }

        return x

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class BiConjugateGradient(InverseLinearOperator):
    """
    A LinearOperator subclass. Objects of this class are meant to be created using :func:~`solvers.inverse`.

    The .dot (and also the .solve) function are based on the 
    Biconjugate gradient (BCG) algorithm for solving linear system Ax=b.
    Implementation from [1], page 175.

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for 2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, 2-norm of residual r is printed at each iteration.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=1000, verbose=False):

        assert isinstance(A, LinearOperator)
        assert A.domain.dimension == A.codomain.dimension
        domain = A.codomain
        codomain = A.domain

        if x0 is not None:
            assert isinstance(x0, Vector)
            assert x0.space is codomain
        else:
            x0 = codomain.zeros()

        self._A = A
        self._Ah = A.H
        self._domain = domain
        self._codomain = codomain
        self._solver = 'bicg'
        self._options = {"x0":x0, "tol":tol, "maxiter":maxiter, "verbose":verbose}
        self._check_options(**self._options)
        self._tmps = {key: domain.zeros() for key in ("v", "r", "p", "vs", "rs", "ps")}
        self._info = None

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert value is not None, "tol may not be None"
                assert value*0 == 0, "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"
            else:
                raise ValueError(f"Key '{key}' not understood. See self._options for allowed keys.")

    def transpose(self, conjugate=False):
        At = self._A.transpose(conjugate=conjugate)
        solver = self._solver
        options = self._options
        return inverse(At, solver, **options)

    def solve(self, b, out=None):
        """
        Biconjugate gradient (BCG) algorithm for solving linear system Ax=b.
        Implementation from [1], page 175.
        Info can be accessed using get_info(), see :func:~`basic.InverseLinearOperator.get_info`.

        Parameters
        ----------
        b : psydac.linalg.basic.Vector
            Right-hand-side vector of linear system. Individual entries b[i] need
            not be accessed, but b has 'shape' attribute and provides 'copy()' and
            'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
            scalar multiplication and sum operations are available.

        out : psydac.linalg.basic.Vector | NoneType
            The output vector, or None (optional).

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
        A = self._A
        Ah = self._Ah
        domain = self._domain
        codomain = self._codomain
        options = self._options
        x0 = options["x0"]
        tol = options["tol"]
        maxiter = options["maxiter"]
        verbose = options["verbose"]

        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)

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

        res_sqr = r.dot(r).real
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
            Ah.dot(ps, out=vs)
            #v  = A.dot(p , out=v) # overwriting v, then saving in v. Necessary?
            #vs = At.dot(ps, out=vs) # same story
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

            # rs := rs - conj(a)*vs
            vs *= a.conj()
            rs -= vs

            # b := (rs, r)_{m+1} / (rs, r)_m
            b = rs.dot(r) / c

            # p := r + b*p
            p *= (b/a) # *= (b/a) why a? or update description
            p += r

            # ps := rs + conj(b)*ps
            ps *= b.conj()
            ps += rs

            # ||r||_2 := (r, r)
            res_sqr = r.dot(r).real

            if verbose:
                print( template.format(m, sqrt(res_sqr)) )

        if verbose:
            print( "+---------+---------------------+")

        # Convergence information
        self._info = {'niter': m, 'success': res_sqr < tol_sqr, 'res_norm': sqrt(res_sqr)}

        return x

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class BiConjugateGradientStabilized(InverseLinearOperator):
    """
    A LinearOperator subclass. Objects of this class are meant to be created using :func:~`solvers.inverse`.

    The .dot (and also the .solve) function are based on the
    Biconjugate gradient Stabilized (BCGSTAB) algorithm for solving linear system Ax=b.
    Implementation from [1], page 175.

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for 2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, 2-norm of residual r is printed at each iteration.

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=1000, verbose=False):

        assert isinstance(A, LinearOperator)
        assert A.domain.dimension == A.codomain.dimension
        domain = A.codomain
        codomain = A.domain

        if x0 is not None:
            assert isinstance(x0, Vector)
            assert x0.space is codomain
        else:
            x0 = codomain.zeros()

        self._A = A
        self._domain = domain
        self._codomain = codomain
        self._solver = 'bicgstab'
        self._options = {"x0": x0, "tol": tol, "maxiter": maxiter, "verbose": verbose}
        self._check_options(**self._options)
        self._tmps = {key: domain.zeros() for key in ("v", "r", "p", "vs", "r0", "s")}
        self._info = None

    def _check_options(self, **kwargs):
        keys = ('x0', 'tol', 'maxiter', 'verbose')
        for key, value in kwargs.items():
            idx = [key == keys[i] for i in range(len(keys))]
            assert any(idx), "key not supported, check options"
            true_idx = idx.index(True)
            if true_idx == 0:
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
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

    def transpose(self, conjugate=False):
        At = self._A.transpose(conjugate=conjugate)
        solver = self._solver
        options = self._options
        return inverse(At, solver, **options)

    def solve(self, b, out=None):
        """
        Biconjugate gradient stabilized method (BCGSTAB) algorithm for solving linear system Ax=b.
        Implementation from [1], page 175.

        Parameters
        ----------
        b : psydac.linalg.basic.Vector
            Right-hand-side vector of linear system. Individual entries b[i] need
            not be accessed, but b has 'shape' attribute and provides 'copy()' and
            'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
            scalar multiplication and sum operations are available.
        out : psydac.linalg.basic.Vector | NoneType
            The output vector, or None (optional).

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

        A = self._A
        domain = self._domain
        codomain = self._codomain
        options = self._options
        x0 = options["x0"]
        tol = options["tol"]
        maxiter = options["maxiter"]
        verbose = options["verbose"]

        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)

        # Extract local storage
        v = self._tmps["v"]
        r = self._tmps["r"]
        p = self._tmps["p"]
        vs = self._tmps["vs"]
        r0 = self._tmps["r0"]
        s = self._tmps["s"]

        # First values
        A.dot(x, out=v)
        b.copy(out=r)
        r -= v
        #r = b - A.dot(x)
        r.copy(out=p)
        v *= 0.0
        vs *= 0.0

        r.copy(out=r0)
        r.copy(out=s)
        s *= 0.0

        res_sqr = r.dot(r).real
        tol_sqr = tol ** 2

        if verbose:
            print("BiCGSTAB solver:")
            print("+---------+---------------------+")
            print("+ Iter. # | L2-norm of residual |")
            print("+---------+---------------------+")
            template = "| {:7d} | {:19.2e} |"

        # Iterate to convergence
        for m in range(1, maxiter + 1):

            if res_sqr < tol_sqr:
                m -= 1
                break

            # -----------------------
            # MATRIX-VECTOR PRODUCTS
            # -----------------------
            v = A.dot(p, out=v)
            # -----------------------

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

            # -----------------------
            # SOLUTION UPDATE
            # -----------------------
            # x := x + a*p +w*s
            p *= a
            s *= w
            x += p
            x += s
            # -----------------------

            # r := s - w*vs
            vs *= w
            s *= 1 / w
            r *= 0
            r += s
            r -= vs

            # ||r||_2 := (r, r)
            res_sqr = r.dot(r).real

            if res_sqr < tol_sqr:
                break

            # b := a / w * (r0, r)_{m+1} / (r0, r)_m
            b = r0.dot(r) * a / (c * w)

            # p := r + b*p- b*w*v
            v *= (b * w / a)
            p *= (b / a)
            p -= v
            p += r

            if verbose:
                print(template.format(m, sqrt(res_sqr)))

        if verbose:
            print("+---------+---------------------+")

        # Convergence information
        self._info = {'niter': m, 'success': res_sqr < tol_sqr, 'res_norm': sqrt(res_sqr)}

        return x

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class MinimumResidual(InverseLinearOperator):
    """
    A LinearOperator subclass. Objects of this class are meant to be created using :func:~`solvers.inverse`.

    The .dot (and also the .solve) function
    Use MINimum RESidual iteration to solve Ax=b

    MINRES minimizes norm(A*x - b) for a real symmetric matrix A.  Unlike
    the Conjugate Gradient method, A can be indefinite or singular.

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for 2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, 2-norm of residual r is printed at each iteration.

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
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=1000, verbose=False):

        assert isinstance(A, LinearOperator)
        assert A.domain.dimension == A.codomain.dimension
        assert A.dtype == float
        domain = A.codomain
        codomain = A.domain

        if x0 is not None:
            assert isinstance(x0, Vector)
            assert x0.space is codomain
        else:
            x0 = codomain.zeros()

        self._A = A
        self._domain = domain
        self._codomain = codomain
        self._solver = 'minres'
        self._options = {"x0":x0, "tol":tol, "maxiter":maxiter, "verbose":verbose}
        self._check_options(**self._options)
        self._tmps = {key: domain.zeros() for key in ("res1", "res2", "w", "w2", "yc",
                      "v", "resc", "res2c", "ycc", "res1c", "wc", "w2c")}
        self._info = None

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert value is not None, "tol may not be None"
                assert value*0 == 0, "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"
            else:
                raise ValueError(f"Key '{key}' not understood. See self._options for allowed keys.")

    def transpose(self, conjugate=False):
        At = self._A.transpose(conjugate=conjugate)
        solver = self._solver
        options = self._options
        return inverse(At, solver, **options)

    def solve(self, b, out=None):
        """
        Use MINimum RESidual iteration to solve Ax=b
        MINRES minimizes norm(A*x - b) for a real symmetric matrix A.  Unlike
        the Conjugate Gradient method, A can be indefinite or singular.
        Info can be accessed using get_info(), see :func:~`basic.InverseLinearOperator.get_info`.

        Parameters
        ----------
        b : psydac.linalg.basic.Vector
            Right-hand-side vector of linear system. Individual entries b[i] need
            not be accessed, but b has 'shape' attribute and provides 'copy()' and
            'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
            scalar multiplication and sum operations are available.

        out : psydac.linalg.basic.Vector | NoneType
            The output vector, or None (optional).

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

        A = self._A
        domain = self._domain
        codomain = self._codomain
        options = self._options
        x0 = options["x0"]
        tol = options["tol"]
        maxiter = options["maxiter"]
        verbose = options["verbose"]

        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)

        # Extract local storage
        v = self._tmps["v"]
        w = self._tmps["w"]
        w2 = self._tmps["w2"]
        res1 = self._tmps["res1"]
        res2 = self._tmps["res2"]
        # auxiliary to minimzize temps, optimal solution until proven wrong
        wc = self._tmps["wc"]
        w2c = self._tmps["w2c"]
        yc = self._tmps["yc"]
        ycc = self._tmps["ycc"]
        resc = self._tmps["resc"]
        res1c = self._tmps["res1c"]
        res2c = self._tmps["res2c"]

        istop = 0
        itn   = 0
        Anorm = 0
        Acond = 0
        rnorm = 0
        ynorm = 0

        eps = np.finfo(b.dtype).eps

        A.dot(x, out=res1)
        res1 -= b
        res1 *= -1.0
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
        b.copy(out=w)
        w *= 0.0
        b.copy(out=w2)
        w2 *= 0.0
        res1.copy(out=res2)

        if verbose:
            print( "MINRES solver:" )
            print( "+---------+---------------------+")
            print( "+ Iter. # | L2-norm of residual |")
            print( "+---------+---------------------+")
            template = "| {:7d} | {:19.2e} |"


        for itn in range(1, maxiter + 1 ):

            s = 1.0/beta
            y.copy(out=v)
            v *= s
            A.dot(v, out=yc)
            y = yc

            if itn >= 2:
                res1 *= (beta/oldb)
                y -= res1

            alfa = v.dot(y)
            res1 = res2

            res2.copy(out=resc)
            resc *= (alfa/beta)
            y.copy(out=ycc)
            ycc -= resc
            y = ycc
            res1.copy(out=res1c)
            res1 = res1c
            y.copy(out=res2c)
            res2 = res2c

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

            w1.copy(out=yc)
            yc *= oldeps
            w2.copy(out=w2c)
            w2.copy(out=wc)
            w = wc
            w2 = w2c
            w *= delta
            w += yc
            w -= v
            w *= -denom
            w.copy(out=yc)
            yc *= phi
            x += yc

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
            #else:test1 = rnorm / (Anorm*ynorm)  # ||r||  / (||A|| ||x||)
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
        self._info = {'niter': itn, 'success': rnorm<tol, 'res_norm': rnorm }

        return x

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class LSMR(InverseLinearOperator):
    """
    A LinearOperator subclass. Objects of this class are meant to be created using :func:~`solvers.inverse`.

    The .dot (and also the .solve) function are based on the 
    Iterative solver for least-squares problems.
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
    def __init__(self, A, *, x0=None, tol=None, atol=None, btol=None, maxiter=1000, conlim=1e8, verbose=False):

        assert isinstance(A, LinearOperator)
        assert A.domain.dimension == A.codomain.dimension
        domain = A.codomain
        codomain = A.domain

        if x0 is not None:
            assert isinstance(x0, Vector)
            assert x0.space is codomain
        else:
            x0 = codomain.zeros()

        self._A = A
        self._domain = domain
        self._codomain = codomain
        self._solver = 'lsmr'
        self._options = {"x0":x0, "tol":tol, "atol":atol, "btol":btol,
                         "maxiter":maxiter, "conlim":conlim, "verbose":verbose}
        self._check_options(**self._options)
        self._info = None
        self._successful = None
        tmps_domain = {key: domain.zeros() for key in ("uh", "uc")}
        tmps_codomain = {key: codomain.zeros() for key in ("v", "vh", "h", "hbar")}
        self._tmps = {**tmps_codomain, **tmps_domain}

    def get_success(self):
        return self._successful

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                if value is not None:
                    assert value*0 == 0, "tol must be a real number"
                    assert value > 0, "tol must be positive" # suppose atol/btol must also be positive numbers
            elif key == 'atol' or key == 'btol':
                if value is not None:
                    assert value*0 == 0, "atol/btol must be a real number"
                    assert value >= 0, "atol/btol must not be negative"
            elif key == 'maxiter':
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'conlim':
                assert value is not None, "conlim may not be None"
                assert value*0 == 0, "conlim must be a real number" # actually an integer?
                assert value > 0, "conlim must be positive" # supposedly
            elif key == 'verbose':
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"
            else:
                raise ValueError(f"Key '{key}' not understood. See self._options for allowed keys.")

    def transpose(self, conjugate=False):
        At = self._A.transpose(conjugate=conjugate)
        solver = self._solver
        options = self._options
        return inverse(At, solver, **options)

    def solve(self, b, out=None):
        """Iterative solver for least-squares problems.
        lsmr solves the system of linear equations ``Ax = b``. If the system
        is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
        ``A`` is a rectangular matrix of dimension m-by-n, where all cases are
        allowed: m = n, m > n, or m < n. ``b`` is a vector of length m.
        The matrix A may be dense or sparse (usually sparse).
        Info can be accessed using get_info(), see :func:~`basic.InverseLinearOperator.get_info`.

        Parameters
        ----------
        b : psydac.linalg.basic.Vector
            Right-hand-side vector of linear system. Individual entries b[i] need
            not be accessed, but b has 'shape' attribute and provides 'copy()' and
            'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
            scalar multiplication and sum operations are available.

        out : psydac.linalg.basic.Vector | NoneType
            The output vector, or None (optional).

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

        A = self._A
        At = A.H
        domain = self._domain
        codomain = self._codomain
        options = self._options
        x0 = options["x0"]
        tol = options["tol"]
        atol = options["atol"]
        btol = options["btol"]
        maxiter = options["maxiter"]
        conlim = options["conlim"]
        verbose = options["verbose"]

        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)

        # Extract local storage
        v = self._tmps["v"]
        h = self._tmps["h"]
        hbar = self._tmps["hbar"]
        # Not strictly needed by the LSMR, but necessary to avoid temporaries
        uh = self._tmps["uh"]
        vh = self._tmps["vh"]
        uc = self._tmps["uc"]

        if atol is None:atol = 1e-6
        if btol is None:btol = 1e-6
        if tol is not None: 
            atol = tol
            btol = tol

        u = b
        normb = sqrt(b.dot(b).real)

        A.dot(x, out=uh)
        u -= uh
        beta = sqrt(u.dot(u).real)

        if beta > 0:
            u.copy(out = uc)
            uc *= (1 / beta)
            u = uc
            At.dot(u, out=v)
            alpha = sqrt(v.dot(v).real)
        else:
            x.copy(out=v)
            alpha = 0

        if alpha > 0:
            v *= (1 / alpha)

        # Initialize variables for 1st iteration.
        itn      = 0
        zetabar  = alpha * beta
        alphabar = alpha
        rho      = 1
        rhobar   = 1
        cbar     = 1
        sbar     = 0

        v.copy(out=h)
        x.copy(out=hbar)
        hbar *= 0.0

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
            A.dot(v, out=uh)
            u += uh
            beta = sqrt(u.dot(u).real)

            if beta > 0:
                u     *= (1 / beta)
                v     *= -beta
                At.dot(u, out=vh)
                v     += vh
                alpha = sqrt(v.dot(v).real)
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

            hbar.copy(out=uh)
            uh *= (zeta / (rho * rhobar))
            x += uh

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
            normx  = sqrt(x.dot(x).real)

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
        self._info = {'niter': itn, 'success': istop in [1,2,3], 'res_norm': normr }
        # Seems necessary, as algorithm might terminate even though rnorm > tol.
        self._successful = istop in [1,2,3]

        return x

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class GMRES(InverseLinearOperator):
    """
    A LinearOperator subclass. Objects of this class are meant to be created using :func:~`solvers.inverse`.

    The .dot (and also the .solve) function are based on the 
    Generalized minimum residual algorithm for solving linear system Ax=b.
    Implementation from Wikipedia

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).

    tol : float
        Absolute tolerance for L2-norm of residual r = A*x - b.

    maxiter: int
        Maximum number of iterations.

    verbose : bool
        If True, L2-norm of residual r is printed at each iteration.

    References
    ----------
    [1] Y. Saad and M.H. Schultz, "GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems", SIAM J. Sci. Stat. Comput., 7:856–869, 1986.

    """
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=100, verbose=False):

        assert isinstance(A, LinearOperator)
        assert A.domain.dimension == A.codomain.dimension
        domain = A.codomain
        codomain = A.domain

        if x0 is not None:
            assert isinstance(x0, Vector)
            assert x0.space is codomain
        else:
            x0 = codomain.zeros()

        self._A = A
        self._domain = domain
        self._codomain = codomain
        self._solver = 'gmres'
        self._options = {"x0":x0, "tol":tol, "maxiter":maxiter, "verbose":verbose}
        self._check_options(**self._options) 
        self._tmps = {key: domain.zeros() for key in ("r", "p", "v", "lv")}

        # Initialize upper Hessenberg matrix
        self._H = np.zeros((self._options["maxiter"] + 1, self._options["maxiter"]), dtype=A.dtype)

        self._info = None
        

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert value is not None, "tol may not be None"
                assert value*0 == 0, "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert value is not None, "maxiter may not be None"
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert value is not None, "verbose may not be None"
                assert isinstance(value, bool), "verbose must be a bool"
            else:
                raise ValueError(f"Key '{key}' not understood. See self._options for allowed keys.")

    def transpose(self, conjugate=False):
        At = self._A.transpose(conjugate=conjugate)
        solver = self._solver
        options = self._options
        return inverse(At, solver, **options)

    def solve(self, b, out=None):
        """
        Generalized minimum residual algorithm for solving linear system Ax=b.
        Implementation from Wikipedia
        Info can be accessed using get_info(), see :func:~`basic.InverseLinearOperator.get_info`.

        Parameters
        ----------
        b : psydac.linalg.basic.Vector
            Right-hand-side vector of linear system Ax = b. Individual entries b[i] need
            not be accessed, but b has 'shape' attribute and provides 'copy()' and
            'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
            scalar multiplication and sum operations are available.

        out : psydac.linalg.basic.Vector | NoneType
            The output vector, or None (optional).

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

        """

        A = self._A
        domain = self._domain
        codomain = self._codomain
        options = self._options
        x0 = options["x0"]
        tol = options["tol"]
        maxiter = options["maxiter"]
        verbose = options["verbose"]
        

        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)
        
        # Extract local storage
        r = self._tmps["r"]
        v = self._tmps["v"]

        # Internal objects of GMRES
        H = self._H
        Q = []
        beta = []
        sn = []
        cn = []

        # First values
        A.dot( x , out=r)
        r -= b

        am = r.dot(r).real ** 0.5
        beta.append(am)
        Q.append(- r / am)       

        if verbose:
            print( "GMRES solver:" )
            print( "+---------+---------------------+")
            print( "+ Iter. # | L2-norm of residual |")
            print( "+---------+---------------------+")
            template = "| {:7d} | {:19.2e} |"
            print( template.format( 1, am ) )

        # Iterate to convergence

        for k in range(maxiter):
            if am < tol:
                break

            # run Arnoldi
            self.arnoldi(k, Q)

            # make the last diagonal entry in H equal to 0, so that H becomes upper triangular
            self.apply_givens_rotation(k, sn, cn)

            # update the residual vector
            beta.append(- sn[k] * beta[k])
            beta[k] *= cn[k]

            am = abs(beta[k+1])
            if verbose:
                print( template.format( k+2, am ) )

        if verbose:
            print( "+---------+---------------------+")        
        # calculate result
        y = self.solve_triangular(H[:k, :k], beta[:k]) # system of upper triangular matrix

        for i in range(k):
            Q[i].copy(out=v)
            v *= y[i]
            x += v

        # Convergence information
        self._info = {'niter': k+1, 'success': am < tol, 'res_norm': am }
        
        return x
    
    def solve_triangular(self, T, d):
        # Backwards substitution. Assumes T is upper triangular
        k = T.shape[0]
        y = np.zeros((k,), dtype=self._A.dtype)

        for k1 in range(k):
            temp = 0.
            for k2 in range(1, k1 + 1):
                temp += T[k - 1 - k1, k - 1 - k1 + k2] * y[k - 1 - k1 + k2]
            y[k - 1 - k1] = ( d[k - 1 - k1] - temp ) / T[k - 1 - k1, k - 1 - k1]
        
        return y

    def arnoldi(self, k, Q):
        h = self._H[:k+2, k]

        p = self._tmps["p"]
        self._A.dot( Q[k] , out=p) # Krylov vector

        lv = self._tmps["lv"]

        for i in range(k + 1): # Modified Gram-Schmidt, keeping Hessenberg matrix
            h[i] = p.dot(Q[i])
            Q[i].copy(out=lv)
            lv *= h[i]
            p -= lv
        
        h[k+1] = p.dot(p) ** 0.5
        p /= h[k+1] # Normalize vector

        Q.append(p.copy())


    def apply_givens_rotation(self, k, sn, cn):
        # Apply Givens rotation to last column of H
        h = self._H[:k+2, k]

        for i in range(k):
            temp = cn[i] * h[i] + sn[i] * h[i+1]
            h[i+1] = - sn[i] * h[i] + cn[i] * h[i+1]
            h[i] = temp
        
        mod = (h[k]**2 + h[k+1]**2)**0.5
        cn.append( h[k] / mod )
        sn.append( h[k+1] / mod )

        h[k] = cn[k] * h[k] + sn[k] * h[k+1]
        h[k+1] = 0. # becomes triangular

    def dot(self, b, out=None):
        return self.solve(b, out=out)