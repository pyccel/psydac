# coding: utf-8
"""
This module provides iterative solvers and precondionners.

"""
from math import sqrt
import numpy as np

from psydac.linalg.basic     import Vector, LinearOperator, InverseLinearOperator, IdentityOperator, ScaledLinearOperator
from psydac.linalg.utilities import _sym_ortho

__all__ = ('ConjugateGradient', 'PConjugateGradient', 'BiConjugateGradient', 'BiConjugateGradientStabilized', 'MinimumResidual', 'LSMR', 'GMRES')

def is_real(x):
    from numbers import Number
    return isinstance(x, Number) and np.isrealobj(x) and not isinstance(x, bool)

def inverse(A, solver, **kwargs):
    """
    A function to create objects of all InverseLinearOperator subclasses.

    These are, as of June 06, 2023:
    ConjugateGradient, PConjugateGradient, BiConjugateGradient,
    BiConjugateGradientStabilized, MinimumResidual, LSMR, GMRES.

    The kwargs given must be compatible with the chosen solver subclass.
    
    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system; individual entries A[i,j]
        can't be accessed, but A has 'shape' attribute and provides 'dot(p)'
        function (i.e. matrix-vector product A*p).

    solver : str
        Preferred iterative solver. Options are: 'cg', 'pcg', 'bicg',
        'bicgstab', 'minres', 'lsmr', 'gmres'.

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

    maxiter : int
        Maximum number of iterations.

    verbose : bool
        If True, L2-norm of residual r is printed at each iteration.

    recycle : bool
        Stores a copy of the output in x0 to speed up consecutive calculations of slightly altered linear systems

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=1000, verbose=False, recycle=False):

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
        self._options = {"x0":x0, "tol":tol, "maxiter":maxiter, "verbose":verbose, "recycle":recycle}
        self._check_options(**self._options)
        self._tmps = {key: domain.zeros() for key in ("v", "r", "p")}
        self._info = None

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert is_real(value), "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert isinstance(value, bool), "verbose must be a bool"
            elif key == 'recycle':
                assert isinstance(value, bool), "recycle must be a bool"
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

        Returns
        -------
        x : psydac.linalg.basic.Vector
            Numerical solution of the linear system. To check the convergence of the solver,
            use the method InverseLinearOperator.get_info().

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
        recycle = options["recycle"]
        
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

            x.mul_iadd(l, p)  # this is x += l*p
            r.mul_iadd(-l, v) # this is r -= l*v

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

        if recycle:
            x.copy(out=self._options["x0"])

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

    recycle : bool
        Stores a copy of the output in x0 to speed up consecutive calculations of slightly altered linear systems

    """
    def __init__(self, A, *, pc='jacobi', x0=None, tol=1e-6, maxiter=1000, verbose=False, recycle=False):

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
        self._options = {"x0":x0, "pc":pc, "tol":tol, "maxiter":maxiter, "verbose":verbose, "recycle":recycle}
        self._check_options(**self._options)
        tmps_codomain = {key: codomain.zeros() for key in ("p", "s")}
        tmps_domain = {key: domain.zeros() for key in ("v", "r")}
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
                assert is_real(value), "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert isinstance(value, bool), "verbose must be a bool"
            elif key == 'recycle':
                assert isinstance(value, bool), "recycle must be a bool"
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
            Numerical solution of the linear system. To check the convergence of the solver,
            use the method InverseLinearOperator.get_info().

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
        recycle = options["recycle"]

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

            x.mul_iadd(l, p) # this is x += l*p
            r.mul_iadd(-l, v) # this is r -= l*v

            nrmr_sqr = r.dot(r).real
            psolve(r, out=s)

            am1 = s.dot(r)

            # we are computing p = (am1 / am) * p + s by using axpy on s and exchanging the arrays
            s.mul_iadd((am1/am), p)
            s, p = p, s

            am  = am1

            if verbose:
                print( template.format(k, sqrt(nrmr_sqr)))

        if verbose:
            print( "+---------+---------------------+")

        # Convergence information
        self._info = {'niter': k, 'success': nrmr_sqr < tol_sqr, 'res_norm': sqrt(nrmr_sqr) }

        if recycle:
            x.copy(out=self._options["x0"])

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

    recycle : bool
        Stores a copy of the output in x0 to speed up consecutive calculations of slightly altered linear systems

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=1000, verbose=False, recycle=False):

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
        self._options = {"x0":x0, "tol":tol, "maxiter":maxiter, "verbose":verbose, "recycle":recycle}
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
                assert is_real(value), "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert isinstance(value, bool), "verbose must be a bool"
            elif key == 'recycle':
                assert isinstance(value, bool), "recycle must be a bool"
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
        ToDo: Add optional preconditioner

        Parameters
        ----------
        b : psydac.linalg.basic.Vector
            Right-hand-side vector of linear system. Individual entries b[i] need
            not be accessed, but b has 'shape' attribute and provides 'copy()' and
            'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
            scalar multiplication and sum operations are available.

        out : psydac.linalg.basic.Vector | NoneType
            The output vector, or None (optional).

        Returns
        -------
        x : psydac.linalg.basic.Vector
            Numerical solution of linear system. To check the convergence of the solver,
            use the method InverseLinearOperator.get_info().

        References
        ----------
        [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

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
        recycle = options["recycle"]

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
            #-----------------------

            # c := (rs, r)
            c = rs.dot(r)

            # a := (rs, r) / (ps, v)
            a = c / ps.dot(v)

            #-----------------------
            # SOLUTION UPDATE
            #-----------------------
            # x := x + a*p
            x.mul_iadd(a, p)
            #-----------------------

            # r := r - a*v
            r.mul_iadd(-a, v)

            # rs := rs - conj(a)*vs
            rs.mul_iadd(-a.conjugate(), vs)

            # ||r||_2 := (r, r)
            res_sqr = r.dot(r).real

            # b := (rs, r)_{m+1} / (rs, r)_m
            b = rs.dot(r) / c

            # p := r + b*p
            p *= b
            p += r

            # ps := rs + conj(b)*ps
            ps *= b.conj()
            ps += rs

            if verbose:
                print( template.format(m, sqrt(res_sqr)) )

        if verbose:
            print( "+---------+---------------------+")

        # Convergence information
        self._info = {'niter': m, 'success': res_sqr < tol_sqr, 'res_norm': sqrt(res_sqr)}

        if recycle:
            x.copy(out=self._options["x0"])

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

    recycle : bool
        Stores a copy of the output in x0 to speed up consecutive calculations of slightly altered linear systems

    References
    ----------
    [1] A. Maister, Numerik linearer Gleichungssysteme, Springer ed. 2015.

    """
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=1000, verbose=False, recycle=False):

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
        self._options = {"x0": x0, "tol": tol, "maxiter": maxiter, "verbose": verbose, "recycle":recycle}
        self._check_options(**self._options)
        self._tmps = {key: domain.zeros() for key in ("v", "r", "p", "vr", "r0")}
        self._info = None

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert is_real(value), "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert isinstance(value, bool), "verbose must be a bool"
            elif key == 'recycle':
                assert isinstance(value, bool), "recycle must be a bool"
            else:
                raise ValueError(f"Key '{key}' not understood. See self._options for allowed keys.")

    def transpose(self, conjugate=False):
        At = self._A.transpose(conjugate=conjugate)
        solver = self._solver
        options = self._options
        return inverse(At, solver, **options)

    def solve(self, b, out=None):
        """
        Biconjugate gradient stabilized method (BCGSTAB) algorithm for solving linear system Ax=b.
        Implementation from [1], page 175.
        ToDo: Add optional preconditioner

        Parameters
        ----------
        b : psydac.linalg.basic.Vector
            Right-hand-side vector of linear system. Individual entries b[i] need
            not be accessed, but b has 'shape' attribute and provides 'copy()' and
            'dot(p)' functions (dot(p) is the vector inner product b*p ); moreover,
            scalar multiplication and sum operations are available.
        out : psydac.linalg.basic.Vector | NoneType
            The output vector, or None (optional).

        Returns
        -------
        x : psydac.linalg.basic.Vector
            Numerical solution of linear system. To check the convergence of the solver,
            use the method InverseLinearOperator.get_info().
        
        info : dict
            Dictionary containing convergence information:
              - 'niter'    = (int) number of iterations
              - 'success'  = (boolean) whether convergence criteria have been met
              - 'res_norm' = (float) 2-norm of residual vector r = A*x - b.

        References
        ----------
        [1] H. A. van der Vorst. Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the
        solution of nonsymmetric linear systems. SIAM J. Sci. Stat. Comp., 13(2):631–644, 1992.
        """

        A = self._A
        domain = self._domain
        codomain = self._codomain
        options = self._options
        x0 = options["x0"]
        tol = options["tol"]
        maxiter = options["maxiter"]
        verbose = options["verbose"]
        recycle = options["recycle"]

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
        vr = self._tmps["vr"]
        r0 = self._tmps["r0"]

        # First values
        A.dot(x, out=v)
        b.copy(out=r)
        r -= v
        #r = b - A.dot(x)
        r.copy(out=p)
        v *= 0.0
        vr *= 0.0

        r.copy(out=r0)

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

            # r := r - a*v
            r.mul_iadd(-a, v)

            # vr :=  A*r
            vr = A.dot(r, out=vr)

            # w := (r, A*r) / (A*r, A*r)
            w = r.dot(vr) / vr.dot(vr)

            # -----------------------
            # SOLUTION UPDATE
            # -----------------------
            # x := x + a*p +w*r
            x.mul_iadd(a, p)
            x.mul_iadd(w, r)
            # -----------------------

            # r := r - w*A*r
            r.mul_iadd(-w, vr)

            # ||r||_2 := (r, r)
            res_sqr = r.dot(r).real

            if res_sqr < tol_sqr:
                break

            # b := a / w * (r0, r)_{m+1} / (r0, r)_m
            b = r0.dot(r) * a / (c * w)

            # p := r + b*p- b*w*v
            p *= b
            p += r
            p.mul_iadd(-b * w, v)

            if verbose:
                print(template.format(m, sqrt(res_sqr)))

        if verbose:
            print("+---------+---------------------+")

        # Convergence information
        self._info = {'niter': m, 'success': res_sqr < tol_sqr, 'res_norm': sqrt(res_sqr)}

        if recycle:
            x.copy(out=self._options["x0"])

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

    recycle : bool
        Stores a copy of the output in x0 to speed up consecutive calculations of slightly altered linear systems

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
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=1000, verbose=False, recycle=False):

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
        self._options = {"x0":x0, "tol":tol, "maxiter":maxiter, "verbose":verbose, "recycle":recycle}
        self._check_options(**self._options)
        self._tmps = {key: domain.zeros() for key in ("res_old", "res_new", "w_new", "w_work", "w_old", "v", "y")}
        self._info = None

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert is_real(value), "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert isinstance(value, bool), "verbose must be a bool"
            elif key == 'recycle':
                assert isinstance(value, bool), "recycle must be a bool"
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

        Returns
        -------
        x : psydac.linalg.basic.Vector
            Numerical solution of linear system. To check the convergence of the solver,
            use the method InverseLinearOperator.get_info().

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
        recycle = options["recycle"]

        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)

        # Extract local storage
        v = self._tmps["v"]
        y = self._tmps["y"]
        w_new = self._tmps["w_new"]
        w_work = self._tmps["w_work"]
        w_old = self._tmps["w_old"]
        res_old = self._tmps["res_old"]
        res_new = self._tmps["res_new"]

        istop = 0
        itn   = 0
        rnorm = 0

        eps = np.finfo(b.dtype).eps

        A.dot(x, out=y)
        y -= b
        y *= -1.0
        y.copy(out=res_old)

        beta = sqrt(res_old.dot(res_old))

        # Initialize other quantities
        oldb    = 0
        dbar    = 0
        epsln   = 0
        phibar  = beta
        rhs1    = beta
        rhs2    = 0
        tnorm2  = 0
        gmax    = 0
        gmin    = np.finfo(b.dtype).max
        cs      = -1
        sn      = 0
        w_new  *= 0.0
        w_work *= 0.0
        w_old *= 0.0
        res_old.copy(out=res_new)

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
            A.dot(v, out=y)

            if itn >= 2:
                y.mul_iadd(-(beta/oldb), res_old)

            alfa = v.dot(y)
            y.mul_iadd(-(alfa/beta), res_new)

            # We put res_new in res_old and y in res_new
            res_new, res_old = res_old, res_new
            y.copy(out=res_new)

            oldb = beta
            beta = sqrt(res_new.dot(res_new))
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

            # Compute the next plane rotation Qk

            gamma  = sqrt(gbar**2 + beta**2)  # gammak
            gamma  = max(gamma, eps)
            cs     = gbar / gamma                # ck
            sn     = beta / gamma                # sk
            phi    = cs * phibar                 # phik
            phibar = sn * phibar                 # phibark+1

            # Update  x.
            denom = 1.0/gamma

            # We put w_old in w_work and w_new in w_old
            w_work, w_old = w_old, w_work
            w_new.copy(out=w_old)

            w_new *= delta
            w_new.mul_iadd(oldeps, w_work)
            w_new -= v
            w_new *= -denom
            x.mul_iadd(phi, w_new)

            # Go round again.

            gmax = max(gmax, gamma)
            gmin = min(gmin, gamma)
            z    = rhs1 / gamma
            rhs1 = rhs2 - delta*z
            rhs2 = - epsln*z

            # Estimate various norms and test for convergence.

            Anorm = sqrt(tnorm2)
            ynorm = sqrt(x.dot(x))

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

        if recycle:
            x.copy(out=self._options["x0"])

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

    recycle : bool
        Stores a copy of the output in x0 to speed up consecutive calculations of slightly altered linear systems

    Notes
    -----
    This is an adaptation of the LSMR Solver in Scipy, where the method is modified to accept Psydac data structures,
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/sparse/linalg/isolve/lsmr.py

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders,
           "LSMR: An iterative algorithm for sparse least-squares problems",
           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
           arxiv:`1006.0758`
    .. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/
    
    """
    def __init__(self, A, *, x0=None, tol=None, atol=None, btol=None, maxiter=1000, conlim=1e8, verbose=False, recycle=False):

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
                         "maxiter":maxiter, "conlim":conlim, "verbose":verbose, "recycle":recycle}
        self._check_options(**self._options)
        self._info = None
        self._successful = None
        tmps_domain = {key: domain.zeros() for key in ("u", "u_work")}
        tmps_codomain = {key: codomain.zeros() for key in ("v", "v_work", "h", "hbar")}
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
                    assert is_real(value), "tol must be a real number"
                    assert value > 0, "tol must be positive" # suppose atol/btol must also be positive numbers
            elif key == 'atol' or key == 'btol':
                if value is not None:
                    assert is_real(value), "atol/btol must be a real number"
                    assert value >= 0, "atol/btol must not be negative"
            elif key == 'maxiter':
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'conlim':
                assert is_real(value), "conlim must be a real number" # actually an integer?
                assert value > 0, "conlim must be positive" # supposedly
            elif key == 'verbose':
                assert isinstance(value, bool), "verbose must be a bool"
            elif key == 'recycle':
                assert isinstance(value, bool), "recycle must be a bool"
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

        Returns
        -------
        x : psydac.linalg.basic.Vector
            Numerical solution of linear system. To check the convergence of the solver,
            use the method InverseLinearOperator.get_info().

        Notes
        -----
        This is an adaptation of the LSMR Solver in Scipy, where the method is modified to accept Psydac data structures,
        https://github.com/scipy/scipy/blob/v1.7.1/scipy/sparse/linalg/isolve/lsmr.py

        References
        ----------
        .. [1] D. C.-L. Fong and M. A. Saunders,
            "LSMR: An iterative algorithm for sparse least-squares problems",
            SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
            arxiv:`1006.0758`
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
        recycle = options["recycle"]

        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)

        # Extract local storage
        u = self._tmps["u"]
        v = self._tmps["v"]
        h = self._tmps["h"]
        hbar = self._tmps["hbar"]
        # Not strictly needed by the LSMR, but necessary to avoid temporaries
        u_work = self._tmps["u_work"]
        v_work = self._tmps["v_work"]

        if atol is None:atol = 1e-6
        if btol is None:btol = 1e-6
        if tol is not None: 
            atol = tol
            btol = tol

        b.copy(out=u)
        normb = sqrt(b.dot(b).real)

        A.dot(x, out=u_work)
        u -= u_work
        beta = sqrt(u.dot(u).real)

        if beta > 0:
            u *= (1 / beta)
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

        # Items for use in stopping rules, normb set earlier
        istop = 0
        ctol  = 0
        if conlim > 0:ctol = 1 / conlim
        normr = beta

        # Reverse the order here from the original matlab code because

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
            A.dot(v, out=u_work)
            u += u_work
            beta = sqrt(u.dot(u).real)

            if beta > 0:
                u     *= (1 / beta)
                v     *= -beta
                At.dot(u, out=v_work)
                v     += v_work
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

            x.mul_iadd((zeta / (rho * rhobar)), hbar)

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

        if recycle:
            x.copy(out=self._options["x0"])

        return x

    def dot(self, b, out=None):
        return self.solve(b, out=out)

#===============================================================================
class GMRES(InverseLinearOperator):
    """
    A LinearOperator subclass. Objects of this class are meant to be created using :func:~`solvers.inverse`.

    The .dot (and also the .solve) function are based on the 
    generalized minimal residual algorithm for solving linear system Ax=b.
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

    recycle : bool
        Stores a copy of the output in x0 to speed up consecutive calculations of slightly altered linear systems

    References
    ----------
    [1] Y. Saad and M.H. Schultz, "GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems", SIAM J. Sci. Stat. Comput., 7:856–869, 1986.

    """
    def __init__(self, A, *, x0=None, tol=1e-6, maxiter=100, verbose=False, recycle=False):

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
        self._options = {"x0":x0, "tol":tol, "maxiter":maxiter, "verbose":verbose, "recycle":recycle}
        self._check_options(**self._options) 
        self._tmps = {key: domain.zeros() for key in ("r", "p")}

        # Initialize upper Hessenberg matrix
        self._H = np.zeros((self._options["maxiter"] + 1, self._options["maxiter"]), dtype=A.dtype)
        self._Q = []
        self._info = None

    def _check_options(self, **kwargs):
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self._codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert is_real(value), "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert isinstance(value, bool), "verbose must be a bool"
            elif key == 'recycle':
                assert isinstance(value, bool), "recycle must be a bool"
            else:
                raise ValueError(f"Key '{key}' not understood. See self._options for allowed keys.")

    def transpose(self, conjugate=False):
        At = self._A.transpose(conjugate=conjugate)
        solver = self._solver
        options = self._options
        return inverse(At, solver, **options)

    def solve(self, b, out=None):
        """
        Generalized minimal residual algorithm for solving linear system Ax=b.
        Implementation from Wikipedia.
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

        Returns
        -------
        x : psydac.linalg.basic.Vector
            Numerical solution of the linear system. To check the convergence of the solver,
            use the method InverseLinearOperator.get_info().
        
        References
        ----------
        [1] Y. Saad and M.H. Schultz, "GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems", SIAM J. Sci. Stat. Comput., 7:856–869, 1986.
        
        """

        A = self._A
        domain = self._domain
        codomain = self._codomain
        options = self._options
        x0 = options["x0"]
        tol = options["tol"]
        maxiter = options["maxiter"]
        verbose = options["verbose"]
        recycle = options["recycle"]
        
        assert isinstance(b, Vector)
        assert b.space is domain

        # First guess of solution
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is codomain

        x = x0.copy(out=out)
        
        # Extract local storage
        r = self._tmps["r"]
        p = self._tmps["p"]

        # Internal objects of GMRES
        self._H[:,:] = 0.
        beta = []
        sn = []
        cn = []

        # First values
        A.dot( x , out=r)
        r -= b

        am = sqrt(r.dot(r).real)
        beta.append(am)
        r *= - 1 / am
        
        if len(self._Q) == 0:
            self._Q.append(r)
        else:
            r.copy(out=self._Q[0])

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
            self.arnoldi(k, p)

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
        y = self.solve_triangular(self._H[:k, :k], beta[:k]) # system of upper triangular matrix

        for i in range(k):
            x.mul_iadd(y[i], self._Q[i])

        # Convergence information
        self._info = {'niter': k+1, 'success': am < tol, 'res_norm': am }
        
        if recycle:
            x.copy(out=self._options["x0"])

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

    def arnoldi(self, k, p):
        h = self._H[:k+2, k]
        self._A.dot( self._Q[k] , out=p) # Krylov vector

        for i in range(k + 1): # Modified Gram-Schmidt, keeping Hessenberg matrix
            h[i] = p.dot(self._Q[i])
            p.mul_iadd(-h[i], self._Q[i])
        
        h[k+1] = sqrt(p.dot(p).real)
        p /= h[k+1] # Normalize vector

        if len(self._Q) > k + 1:
            p.copy(out=self._Q[k+1])
        else:
            self._Q.append(p.copy())

    def apply_givens_rotation(self, k, sn, cn):
        # Apply Givens rotation to last column of H
        h = self._H[:k+2, k]

        for i in range(k):
            h_i_prev = h[i]

            h[i] *= cn[i]
            h[i] += sn[i] * h[i+1]

            h[i+1] *= cn[i]
            h[i+1] -= sn[i] * h_i_prev
        
        mod = (h[k]**2 + h[k+1]**2)**0.5
        cn.append( h[k] / mod )
        sn.append( h[k+1] / mod )

        h[k] *= cn[k]
        h[k] += sn[k] * h[k+1]
        h[k+1] = 0. # becomes triangular

    def dot(self, b, out=None):
        return self.solve(b, out=out)
    