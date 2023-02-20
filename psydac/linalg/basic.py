# coding: utf-8
#
# Copyright 2018 Yaman Güçlü, Jalal Lakhlili
# Copyright 2022 Yaman Güçlü, Said Hadjout, Julian Owezarek

from abc   import ABC, abstractmethod
import numpy as np

__all__ = ['VectorSpace', 'Vector', 'LinearOperator', 'ZeroOperator', 'IdentityOperator', 'ScaledLinearOperator',
           'SumLinearOperator', 'ComposedLinearOperator', 'PowerLinearOperator', 'InverseLinearOperator', 'LinearSolver']

#===============================================================================
class VectorSpace(ABC):
    """
    Finite-dimensional vector space V with a scalar (dot) product.

    """
    @property
    @abstractmethod
    def dimension(self):
        """
        The dimension of a vector space V is the cardinality
        (i.e. the number of vectors) of a basis of V over its base field.

        """

    @property
    @abstractmethod
    def dtype(self):
        """
        The data type of the field over which the space is built.

        See also
        --------
        https://en.wikipedia.org/wiki/Field_(mathematics)

        """

    @abstractmethod
    def zeros(self):
        """
        Get a copy of the null element of the vector space V.

        Returns
        -------
        null : Vector
            A new vector object with all components equal to zero.

        """

#    @abstractmethod
    def dot(self, a, b):
        """
        Evaluate the scalar product between two vectors of the same space.

        """

#===============================================================================
class Vector(ABC):
    """
    Element of a (normed) vector space V.

    """
    @property
    def shape(self):
        """ A tuple containing the dimension of the space. """
        return (self.space.dimension, )

    @property
    def dtype(self):
        return self.space.dtype

    def dot(self, other):
        """
        Evaluate the scalar product with another vector of the same space.

        """
        assert isinstance(other, Vector)
        assert self.space is other.space
        return self.space.dot(self, other)

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def space(self):
        """ Vector space to which this vector belongs. """

    @abstractmethod
    def toarray(self, **kwargs):
        """ Convert to Numpy 1D array. """

    @abstractmethod
    def copy(self, out=None):
        pass

    @abstractmethod
    def __neg__(self):
        pass

    @abstractmethod
    def __mul__(self, a):
        pass

    @abstractmethod
    def __rmul__(self, a):
        pass

    @abstractmethod
    def __add__(self, v):
        pass

    @abstractmethod
    def __sub__(self, v):
        pass

    @abstractmethod
    def __imul__(self, a):
        pass

    @abstractmethod
    def __iadd__(self, v):
        pass

    @abstractmethod
    def __isub__(self, v):
        pass

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def __truediv__(self, a):
        return self * (1.0 / a)

    def __itruediv__(self, a):
        self *= 1.0 / a
        return self

#===============================================================================
class LinearOperator(ABC):
    """
    Linear operator acting between two (normed) vector spaces V (domain)
    and W (codomain).

    """
    @property
    def shape(self):
        """ A tuple containing the dimension of the codomain and domain. """
        return (self.codomain.dimension, self.domain.dimension)

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def domain(self):
        """ The domain of the linear operator - an element of Vectorspace """
        pass

    @property
    @abstractmethod
    def codomain(self):
        """ The codomain of the linear operator - an element of Vectorspace """
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

    @property
    def T(self):
        return self.transpose()

    @abstractmethod
    def tosparse(self):
        pass

    @abstractmethod
    def toarray(self):
        pass

    @abstractmethod
    def dot(self, v, out=None):
        """ Apply linear operator to Vector v. Result is written to Vector out, if provided."""
        pass

    #-------------------------------------
    # Magic methods
    #-------------------------------------
    def __neg__(self):
        return ScaledLinearOperator(self._domain, self._codomain, -1.0, self)

    def __mul__(self, c):
        """
        Scales a linear operator by c by creating an object of class :ref:`ScaledLinearOperator <scaledlinearoperator>`,
        unless c = 0 or c = 1, in which case either a :ref:`ZeroOperator <zerooperator>` or self is returned.

        """
        assert np.isscalar(c)
        if c==0:
            return ZeroOperator(self._domain, self._codomain)
        elif c == 1:
            return self
        else:
            return ScaledLinearOperator(self._domain, self._codomain, c, self)

    def __rmul__(self, c):
        """ Calles :ref:`__mul__ <mul>` instead. """
        return self * c

    def __matmul__(self, B):
        """ Creates an object of class :ref:`ComposedLinearOperator <composedlinearoperator>`. """
        assert isinstance(B, (LinearOperator, Vector))
        if isinstance(B, LinearOperator):
            assert self._domain == B.codomain
            if isinstance(B, ZeroOperator):
                return ZeroOperator(B.domain, self._codomain)
            elif isinstance(B, IdentityOperator):
                return self
            else:
                return ComposedLinearOperator(B.domain, self._codomain, self, B)
        else:
            return self.dot(B)

    def __add__(self, B):
        """ Creates an object of class :ref:`SumLinearOperator <sumlinearoperator>` unless B is a :ref:`ZeroOperator <zerooperator>` in which case self is returned. """
        assert isinstance(B, LinearOperator)
        if isinstance(B, ZeroOperator):
            return self
        else:
            return SumLinearOperator(self._domain, self._codomain, self, B)

    def __sub__(self, m):
        assert isinstance(m, LinearOperator)
        if isinstance(m, ZeroOperator):
            return self
        else:
            return SumLinearOperator(self._domain, self._codomain, self, -m)

    def __pow__(self, n):
        """ Creates an object of class :ref:`PowerLinearOperator <powerlinearoperator>`. """
        return PowerLinearOperator(self._domain, self._codomain, self, n)

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def transpose(self):
        raise NotImplementedError()

    def idot(self, v, out):
        """
        Implements out += self @ v with a temporary.
        Subclasses should provide a implementation without a temporary.

        """
        assert isinstance(v, Vector)
        assert v.space == self.domain
        assert isinstance(out, Vector)
        assert out.space == self.codomain
        out += self.dot(v)

#===============================================================================
class ZeroOperator(LinearOperator):

    def __new__(cls, domain, codomain=None):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)

        from psydac.linalg.block import BlockVectorSpace, BlockLinearOperator
        if isinstance(domain, BlockVectorSpace) or isinstance(codomain, BlockVectorSpace):
            if isinstance(domain, BlockVectorSpace):
                domain_spaces = domain.spaces
            else:
                domain_spaces = (domain,)
            if isinstance(codomain, BlockVectorSpace):
                codomain_spaces = codomain.spaces
            else:
                codomain_spaces = (codomain,)
            blocks = {}
            for i, D in enumerate(domain_spaces):
                for j, C in enumerate(codomain_spaces):
                    blocks[j,i] = ZeroOperator(D,C)
            return BlockLinearOperator(domain, codomain, blocks)
        else:
            return super().__new__(cls)
    
    def __init__(self, domain, codomain):

        self._domain = domain
        self._codomain = codomain

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return None

    def copy(self):
        return ZeroOperator(self._domain, self._codomain)

    def toarray(self):
        return np.zeros(self.shape, dtype=self.dtype) 

    def tosparse(self):
        from scipy.sparse import csr_matrix
        return csr_matrix(self.shape, dtype=self.dtype)

    def transpose(self):
        return ZeroOperator(domain=self._codomain, codomain=self._domain)

    def dot(self, v, out=None):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            out *= 0
        else:
            out = self._codomain.zeros()
        return out

    def __neg__(self):
        return self

    def __add__(self, B):
        assert isinstance(B, LinearOperator)
        assert self._domain == B.domain
        assert self._codomain == B.codomain
        return B

    def __sub__(self, B):
        assert isinstance(B, LinearOperator)
        assert self._domain == B.domain
        assert self._codomain == B.codomain
        return -B

    def __mul__(self, c):
        assert np.isscalar(c)
        return self

    def __matmul__(self, B):
        assert isinstance(B, (LinearOperator, Vector))
        if isinstance(B, LinearOperator):
            assert self._domain == B.codomain
            return ZeroOperator(domain=B.domain, codomain=self._codomain)
        else:
            return self.dot(B)

#===============================================================================
class IdentityOperator(LinearOperator):

    def __new__(cls, domain, codomain=None):

        assert isinstance(domain, VectorSpace)
        if codomain:
            assert isinstance(codomain, VectorSpace)
            assert domain == codomain

        from psydac.linalg.block import BlockVectorSpace, BlockLinearOperator
        if isinstance(domain, BlockVectorSpace):
            spaces = domain.spaces
            blocks = {}
            for i, V in enumerate(spaces):
                blocks[i,i] = IdentityOperator(V)
            return BlockLinearOperator(domain, domain, blocks)
        else:
            return super().__new__(cls)

    
    def __init__(self, domain, codomain=None):

        self._domain = domain
        self._codomain = domain

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return None

    def copy(self):
        return IdentityOperator(self._domain, self._codomain)

    def toarray(self):
        return np.diag(np.ones(self._domain.dimension , dtype=self.dtype)) 

    def tosparse(self):
        from scipy.sparse import identity
        return identity(self._domain.dimension, dtype=self.dtype, format="csr")

    def transpose(self):
        """ Could return self, but by convention returns new object. """
        return IdentityOperator(self._domain, self._codomain)

    def dot(self, v, out=None):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            out *= 0
            out += v
            return out
        else:
            return v.copy()

    def __matmul__(self, B):
        assert isinstance(B, (LinearOperator, Vector))
        if isinstance(B, LinearOperator):
            assert self._domain == B.codomain
            return B
        else:
            return self.dot(B)

#===============================================================================
class ScaledLinearOperator(LinearOperator):

    def __init__(self, domain, codomain, c, A):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)
        assert np.isscalar(c)
        assert isinstance(A, LinearOperator)
        assert domain   == A.domain
        assert codomain == A.codomain

        if isinstance(A, ScaledLinearOperator):
            scalar = A.scalar * c
            operator = A.operator
        else:
            scalar = c
            operator = A

        self._operator = operator
        self._scalar   = scalar
        self._domain   = domain
        self._codomain = codomain

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def scalar(self):
        return self._scalar

    @property
    def operator(self):
        return self._operator

    @property
    def dtype(self):
        return None

    def toarray(self):
        return self._scalar*self._operator.toarray() 

    def tosparse(self):
        from scipy.sparse import csr_matrix
        return self._scalar*csr_matrix(self._operator.toarray())

    def transpose(self):
        return ScaledLinearOperator(domain=self._codomain, codomain=self._domain, c=self._scalar, A=self._operator.T)

    def __neg__(self):
        return ScaledLinearOperator(domain=self._domain, codomain=self._codomain, c=-1*self._scalar, A=self._operator)

    def dot(self, v, out=None):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            self._operator.dot(v, out = out)
            out *= self._scalar
            return out
        else:
            out = self._operator.dot(v)
            out *= self._scalar
            return out

#===============================================================================
class SumLinearOperator(LinearOperator):
    """
    A sum of linear operatos acting between the same (normed) vector spaces V (domain) and W (codomain).

    """
    def __new__(cls, domain, codomain, *args):

        if len(args) == 0:
            return ZeroOperator(domain,codomain)
        elif len(args) == 1:
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(self, domain, codomain, *args):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)
        for a in args:
            assert isinstance(a, LinearOperator)
            assert a.domain == domain
            assert a.codomain == codomain

        addends = ()
        for a in args:
            if isinstance(a, SumLinearOperator):
                addends = (*addends, *a.addends)
            else:
                addends = (*addends, a)

        addends = SumLinearOperator.simplifiy(addends)

        self._domain = domain
        self._codomain = codomain
        self._addends = addends

    @property
    def domain(self):
        """ The domain of the linear operator, element of class ``VectorSpace``. """
        return self._domain

    @property
    def codomain(self):
        """ The codomain of the linear operator, element of class ``VectorSpace``. """
        return self._codomain

    @property
    def addends(self):
        """ A tuple containing the addends of the linear operator, elements of class ``LinearOperator``. """
        return self._addends

    @property
    def dtype(self):
        """
        todo

        """
        return None

    def toarray(self):
        out = np.zeros(self.shape, dtype=self.dtype)
        for a in self._addends:
            out += a.toarray()
        return out

    def tosparse(self):
        from scipy.sparse import csr_matrix
        out = csr_matrix(self.shape, dtype=self.dtype)
        for a in self._addends:
            out += a.tosparse()
        return out

    def transpose(self):
        t_addends = ()
        for a in self._addends:
            t_addends = (*t_addends, a.T)
        return SumLinearOperator(self._codomain, self._domain, *t_addends)

    @staticmethod
    def simplifiy(addends):
        class_list = [addends[i].__class__.__name__ for i in range(len(addends))]
        unique_list = list(set(class_list))
        if len(unique_list) == 1:
            return addends
        out = ()
        for j in unique_list:
            indices = [k for k, l in enumerate(class_list) if l == j]
            if len(indices) == 1:
                out = (*out, addends[indices[0]])
            else:
                A = addends[indices[0]] + addends[indices[1]]
                for n in range(len(indices)-2):
                    A += addends[indices[n+2]]
                if isinstance(A, SumLinearOperator):
                    out = (*out, *A.addends)
                else:
                    out = (*out, A)
        return out

    def dot(self, v, out=None):
        """ Evaluates SumLinearOperator object at a vector v element of domain. """
        assert isinstance(v, Vector)
        assert v.space == self._domain
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            out *= 0
            for a in self._addends:
                a.idot(v, out)
            return out
        else:
            out = self._codomain.zeros()
            for a in self._addends:
                a.idot(v, out=out)
            return out

#===============================================================================
class ComposedLinearOperator(LinearOperator):

    def __init__(self, domain, codomain, *args):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)

        for a in args:
            assert isinstance(a, LinearOperator)
        assert args[0].codomain == codomain
        assert args[-1].domain == domain

        for i in range(len(args)-1):
            assert args[i].domain == args[i+1].codomain

        multiplicants = ()
        tmp_vectors = []
        for a in args[:-1]:
            if isinstance(a, ComposedLinearOperator):
                multiplicants = (*multiplicants, *a.multiplicants)
                tmp_vectors.extend(a.tmp_vectors)
                tmp_vectors.append(a.domain.zeros())
            else:
                multiplicants = (*multiplicants, a)
                tmp_vectors.append(a.domain.zeros())

        last = args[-1]
        if isinstance(last, ComposedLinearOperator):
            multiplicants = (*multiplicants, *last.multiplicants)
            tmp_vectors.extend(last.tmp_vectors[:-1])
        else:
            multiplicants = (*multiplicants, last)

        self._domain = domain
        self._codomain = codomain
        self._multiplicants = multiplicants
        self._tmp_vectors = tuple(tmp_vectors)

    @property
    def tmp_vectors(self):
        return self._tmp_vectors

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def multiplicants(self):
        return self._multiplicants

    @property
    def dtype(self):
        return None

    def toarray(self):
        raise NotImplementedError('toarray() is not defined for ComposedLinearOperators.')

    def tosparse(self):
        raise NotImplementedError('tosparse() is not defined for ComposedLinearOperators.')

    def transpose(self):
        t_multiplicants = ()
        for a in self._multiplicants:
            t_multiplicants = (a.T, *t_multiplicants)
        new_dom = self._codomain
        new_cod = self._domain
        assert isinstance(new_dom, VectorSpace)
        assert isinstance(new_cod, VectorSpace)
        print(*t_multiplicants)
        return ComposedLinearOperator(self._codomain, self._domain, *t_multiplicants)

    def dot(self, v, out=None):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._codomain

        x = v
        for i in range(len(self._tmp_vectors)):
            y = self._tmp_vectors[-1-i]
            A = self._multiplicants[-1-i]
            A.dot(x, out=y)
            x = y

        A = self._multiplicants[0]
        if out is not None:
            A.dot(x, out=out)
        else:
            out = A.dot(x)
        return out

#===============================================================================
class PowerLinearOperator(LinearOperator):

    def __new__(cls, domain, codomain, A, n):

        assert isinstance(n, int)
        assert n >= 0

        assert isinstance(A, LinearOperator)
        assert A.domain == domain
        assert A.codomain == codomain
        assert domain == codomain

        if n == 0:
            return IdentityOperator(domain, codomain)
        elif n == 1:
            return A
        else:
            return super().__new__(cls)

    def __init__(self, domain, codomain, A, n):

        if isinstance(A, PowerLinearOperator):
            self._operator = A.operator
            self._factorial = A.factorial*n
        else:
            self._operator = A
            self._factorial = n
        self._domain = domain
        self._codomain = codomain

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return None

    @property
    def operator(self):
        return self._operator

    @property
    def factorial(self):
        return self._factorial

    def toarray(self):
        raise NotImplementedError('toarray() is not defined for PowerLinearOperators.')

    def tosparse(self):
        raise NotImplementedError('tosparse() is not defined for PowerLinearOperators.')

    def transpose(self):
        return PowerLinearOperator(domain=self._codomain, codomain=self._domain, A=self._operator.T, n=self._factorial)

    def dot(self, v, out=None):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space == self._codomain
            for i in range(self._factorial):
                self._operator.dot(v, out=out)
                v = out.copy()
        else:
            out = v.copy()
            for i in range(self._factorial):
                out = self._operator.dot(out)
        return out

#===============================================================================
class InverseLinearOperator(LinearOperator):
    """
    Iterative solver for square linear system Ax=b, where x and b belong to (normed)
    vector space V.

    """

    @property
    def space(self):
        return self._space

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return None

    @property
    def linop(self):
        return self._A

    @property
    def options(self):
        return self._options

    def toarray(self):
        raise NotImplementedError('toarray() is not defined for InverseLinearOperators.')

    def tosparse(self):
        raise NotImplementedError('tosparse() is not defined for InverseLinearOperators.')

    def get_info(self):
        return self._info

    @abstractmethod
    def _update_options(self):
        pass

    def getoptions(self):
        for key, value in self.options.items():
            print(key, ": ", value)

    def setoptions(self, **kwargs):
        self._check_options(**kwargs)
        for key, value in kwargs.items():
            setattr(self, '_'+key, value)
        self._update_options()

    @abstractmethod
    def _check_options(self, **kwargs):
        pass

    @abstractmethod
    def transpose(self):
        pass

    @staticmethod
    def jacobi(A, b, out=None):
        """
        Jacobi preconditioner.
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
            if out is not None:
                for i, bi in enumerate(b.blocks):
                    InverseLinearOperator.jacobi(A[i,i], bi, out=out[i])
                return out
            else:
                x = [InverseLinearOperator.jacobi(A[i, i], bi) for i, bi in enumerate(b.blocks)]
                y = BlockVector(b.space, blocks=x)
                return y
        #-------------------------------------------------------------

        V = b.space
        i = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))

        if out is not None:
            b.copy(out=out)
            out[i] /= A.diagonal()
            out.update_ghost_regions()
        else:
            out = b.copy()
            out[i] /= A.diagonal()
            out.update_ghost_regions()
            return out

    @staticmethod
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

#===============================================================================
class LinearSolver(ABC):
    """
    Solver for square linear system Ax=b, where x and b belong to (normed)
    vector space V.

    """
    @property
    def shape(self):
        return (self.space.dimension, self.space.dimension)

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def space(self):
        pass

    @abstractmethod
    def solve(self, rhs, out=None, transposed=False):
        pass
