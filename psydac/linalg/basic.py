# coding: utf-8
#
# Copyright 2018 Yaman Güçlü, Jalal Lakhlili
# Copyright 2022 Yaman Güçlü, Said Hadjout, Julian Owezarek
"""
provides the fundamental classes for linear algebra operations.

"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import coo_matrix

from psydac.utilities.utils import is_real

__all__ = (
    'VectorSpace',
    'Vector',
    'LinearOperator',
    'ZeroOperator',
    'IdentityOperator',
    'ScaledLinearOperator',
    'SumLinearOperator',
    'ComposedLinearOperator',
    'PowerLinearOperator',
    'InverseLinearOperator',
    'LinearSolver'
)

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

    @abstractmethod
    def axpy(self, a, x, y):
        """
        Increment the vector y with the a-scaled vector x, i.e. y = a * x + y,
        provided that x and y belong to the same vector space V (self).
        The scalar value a may be real or complex, depending on the field of V.

        Parameters
        ----------
        a : scalar
            The scaling coefficient needed for the operation.

        x : Vector
            The vector which is not modified by this function.

        y : Vector
            The vector modified by this function (incremented by a * x).
        """

#===============================================================================
class Vector(ABC):
    """
    Element of a vector space V.

    """
    @property
    def shape(self):
        """ A tuple containing the dimension of the space. """
        return (self.space.dimension, )

    @property
    def dtype(self):
        """ The data type of the vector field V this vector belongs to. """
        return self.space.dtype

    def dot(self, v):
        """
        Evaluate the scalar product with the vector v of the same space.

        Parameters
        ----------
        v : Vector
            Vector belonging to the same space as self.

        """
        assert isinstance(v, Vector)
        assert self.space is v.space
        return self.space.dot(self, v)

    def mul_iadd(self, a, v):
        """
        Compute self += a * v, where v is another vector of the same space.

        Parameters
        ----------
        a : scalar
            Rescaling coefficient, which can be cast to the correct dtype.

        v : Vector
            Vector belonging to the same space as self.
        """
        self.space.axpy(a, v, self)

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
        """Ensure x.copy(out=x) returns x and not a new object."""
        pass

    @abstractmethod
    def __neg__(self):
        pass

    @abstractmethod
    def __mul__(self, a):
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

    @abstractmethod
    def conjugate(self, out=None):
        """Compute the complex conjugate vector.

        If the field is real (i.e. `self.dtype in (np.float32, np.float64)`) this method is equivalent to `copy`.
        If the field is complex (i.e. `self.dtype in (np.complex64, np.complex128)`) this method returns
        the complex conjugate of `self`, element-wise.

        The behavior of this function is similar to `numpy.conjugate(self, out=None)`.
        """

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def __rmul__(self, a):
        return self * a

    def __truediv__(self, a):
        return self * (1.0 / a)

    def __itruediv__(self, a):
        self *= 1.0 / a
        return self

    def conj(self, out=None):
        """Compute the complex conjugate vector.

        If the field is real (i.e. `self.dtype in (np.float32, np.float64)`) this method is equivalent to `copy`.
        If the field is complex (i.e. `self.dtype in (np.complex64, np.complex128)`) this method returns
        the complex conjugate of `self`, element-wise.

        The behavior of this function is similar to `numpy.conj(self, out=None)`.
        """
        return self.conjugate(out)

#===============================================================================
class LinearOperator(ABC):
    """
    Abstract base class for all linear operators acting between two vector spaces V (domain)
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

    @abstractmethod
    def tosparse(self):
        pass

    @abstractmethod
    def toarray(self):
        """ Convert to Numpy 2D array. """
        pass

    @abstractmethod
    def dot(self, v, out=None):
        """ Apply linear operator to Vector v. Result is written to Vector out, if provided."""
        pass

    @abstractmethod
    def transpose(self, conjugate=False):
        """
        Transpose the LinearOperator .

        If conjugate is True, return the Hermitian transpose.
        """
        pass

    # TODO: check if we should add a copy method!!!

    #-------------------------------------
    # Magic methods
    #-------------------------------------
    def __neg__(self):
        """
        Scales itself by -1 and thus returns the addititive inverse as 
        a new object of the class ScaledLinearOperator.
        
        """
        return ScaledLinearOperator(self._domain, self._codomain, -1.0, self)

    def __mul__(self, c):
        """
        Scales a linear operator by a real scalar c by creating an object of the class ScaledLinearOperator,
        unless c = 0 or c = 1, in which case either a ZeroOperator or self is returned.

        """
        assert np.isscalar(c)
        assert np.isreal(c)
        if c==0:
            return ZeroOperator(self._domain, self._codomain)
        elif c == 1:
            return self
        else:
            return ScaledLinearOperator(self._domain, self._codomain, c, self)

    def __rmul__(self, c):
        """ Calls __mul__ instead. """
        return self * c

    def __matmul__(self, B):
        """ Creates an object of the class ComposedLinearOperator. """
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
        """ Creates an object of the class SumLinearOperator unless B is a ZeroOperator in which case self is returned. """
        assert isinstance(B, LinearOperator)
        if isinstance(B, ZeroOperator):
            return self
        else:
            return SumLinearOperator(self._domain, self._codomain, self, B)

    def __sub__(self, B):
        """ Creates an object of the class SumLinearOperator unless B is a ZeroOperator in which case self is returned. """
        assert isinstance(B, LinearOperator)
        if isinstance(B, ZeroOperator):
            return self
        else:
            return SumLinearOperator(self._domain, self._codomain, self, -B)

    def __pow__(self, n):
        """ Creates an object of class :ref:`PowerLinearOperator <powerlinearoperator>`. """
        return PowerLinearOperator(self._domain, self._codomain, self, n)

    def __truediv__(self, c):
        """ Divide by scalar. """
        return self * (1.0 / c)

    def __itruediv__(self, c):
        """ Divide by scalar, in place. """
        self *= 1.0 / c
        return self

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------

    @property
    def T(self):
        """ Calls transpose method to return the transpose of self. """
        return self.transpose()

    @property
    def H(self):
        """ Calls transpose method with `conjugate=True` flag to return the Hermitian transpose of self. """
        return self.transpose(conjugate=True)

    def idot(self, v, out):
        """
        Implements out += self @ v with a temporary.
        Subclasses should provide an implementation without a temporary.

        """
        assert isinstance(v, Vector)
        assert v.space == self.domain
        assert isinstance(out, Vector)
        assert out.space == self.codomain
        out += self.dot(v)

#===============================================================================
class ZeroOperator(LinearOperator):
    """
    Zero operator mapping any vector from its domain V to the zero vector of its codomain W.
    
    """

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
        """ Returns a new ZeroOperator object acting between the same vector spaces."""
        return ZeroOperator(self._domain, self._codomain)

    def toarray(self):
        return np.zeros(self.shape, dtype=self.dtype) 

    def tosparse(self):
        from scipy.sparse import csr_matrix
        return csr_matrix(self.shape, dtype=self.dtype)

    def transpose(self, conjugate=False):
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
    """
    Identity operator acting between a vector space V and itself.
    Useful for example in custom linear operator classes together with the apply_essential_bc method to create projection operators.
    
    """

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
        """ Returns a new IdentityOperator object acting between the same vector spaces."""
        return IdentityOperator(self._domain, self._codomain)

    def toarray(self):
        return np.diag(np.ones(self._domain.dimension , dtype=self.dtype)) 

    def tosparse(self):
        from scipy.sparse import identity
        return identity(self._domain.dimension, dtype=self.dtype, format="csr")

    def transpose(self, conjugate=False):
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
    """
    A linear operator :math:`A` scalar multiplied by a real constant :math:`c`. 
    
    """

    def __init__(self, domain, codomain, c, A):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)
        assert np.isscalar(c)
        assert np.isreal(c)
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
        """ Returns the scalar value by which the operator is multiplied."""
        return self._scalar

    @property
    def operator(self):
        """ Returns the operator that is multiplied by the scalar."""
        return self._operator

    @property
    def dtype(self):
        return None

    def toarray(self):
        return self._scalar*self._operator.toarray() 

    def tosparse(self):
        from scipy.sparse import csr_matrix
        return self._scalar*csr_matrix(self._operator.toarray())

    def transpose(self, conjugate=False):
        return ScaledLinearOperator(domain=self._codomain, codomain=self._domain, c=self._scalar, A=self._operator.transpose(conjugate=conjugate))

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
    Sum :math:`\sum_{i=1}^n A_i` of linear operators :math:`A_1,\dots,A_n` acting between the same vector spaces V (domain) and W (codomain).

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

        addends = SumLinearOperator.simplify(addends)

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

    def transpose(self, conjugate=False):
        t_addends = ()
        for a in self._addends:
            t_addends = (*t_addends, a.transpose(conjugate=conjugate))
        return SumLinearOperator(self._codomain, self._domain, *t_addends)

    @staticmethod
    def simplify(addends):
        """ Simplifies a sum of linear operators by combining addends of the same class. """
        class_list  = [a.__class__ for a in addends]
        unique_list = [*{c: a for c, a in zip(class_list, addends)}]
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
    """
    Composition :math:`A_n\circ\dots\circ A_1` of two or more linear operators :math:`A_1,\dots,A_n`.
    
    """

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
            tmp_vectors.extend(last.tmp_vectors)
        else:
            multiplicants = (*multiplicants, last)

        self._domain = domain
        self._codomain = codomain
        self._multiplicants = multiplicants
        self._tmp_vectors = tuple(tmp_vectors)

    @property
    def tmp_vectors(self):
        """
        A tuple containing the storage vectors that are repeatedly being used upon calling the `dot` method.
        This avoids the creation of new vectors at each call of the `dot` method.
        
        """
        return self._tmp_vectors

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def multiplicants(self):
        """
        A tuple :math:`(A_1,\dots,A_n)` containing the multiplicants of the linear operator 
        :math:`self = A_n\circ\dots\circ A_1`.
        
        """
        return self._multiplicants

    @property
    def dtype(self):
        return None

    def toarray(self):
        raise NotImplementedError('toarray() is not defined for ComposedLinearOperators.')

    def tosparse(self):
        mats = [M.tosparse() for M in self._multiplicants]
        M = mats[0]
        for Mi in mats[1:]:
            M = M @ Mi
        return coo_matrix(M)

    def transpose(self, conjugate=False):
        t_multiplicants = ()
        for a in self._multiplicants:
            t_multiplicants = (a.transpose(conjugate=conjugate), *t_multiplicants)
        new_dom = self._codomain
        new_cod = self._domain
        assert isinstance(new_dom, VectorSpace)
        assert isinstance(new_cod, VectorSpace)
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

    def exchange_assembly_data(self):
        for op in self._multiplicants:
            op.exchange_assembly_data()

    def set_backend(self, backend):
        for op in self._multiplicants:
            op.set_backend(backend)

#===============================================================================
class PowerLinearOperator(LinearOperator):
    """
    Power :math:`A^n` of a linear operator :math:`A` for some integer :math:`n\geq 0`.
    
    """

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
        """ Returns the operator that is raised to the power. """
        return self._operator

    @property
    def factorial(self):
        """ Returns the power to which the operator is raised. """
        return self._factorial

    def toarray(self):
        raise NotImplementedError('toarray() is not defined for PowerLinearOperators.')

    def tosparse(self):
        raise NotImplementedError('tosparse() is not defined for PowerLinearOperators.')

    def transpose(self, conjugate=False):
        return PowerLinearOperator(domain=self._codomain, codomain=self._domain, A=self._operator.transpose(conjugate=conjugate), n=self._factorial)

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
    Abstract base class for the (approximate) inverse :math:`A^{-1}` of a
    square matrix :math:`A`. The result of A_inv.dot(b) is the (approximate) solution x
    of the linear system A x = b, where x and b belong to the same vector space V.

    We assume that the linear system is solved by an iterative method, which
    needs a first guess `x0` and an exit condition based on `tol` and `maxiter`.

    Concrete subclasses of this class must implement the `dot` method and take
    care of any internal storage which might be necessary.

    Parameters
    ----------
    A : psydac.linalg.basic.LinearOperator
        Left-hand-side matrix A of linear system.
        
    x0 : psydac.linalg.basic.Vector
        First guess of solution for iterative solver (optional).
        
    tol : float
        Absolute tolerance for L2-norm of residual r = A*x - b.
        
    maxiter: int
        Maximum number of iterations.
        
    verbose : bool
        If True, L2-norm of residual r is printed at each iteration.
    """

    def __init__(self, A, **kwargs):

        assert isinstance(A, LinearOperator)
        assert A.domain.dimension == A.codomain.dimension
        domain = A.codomain
        codomain = A.domain

        if kwargs['x0'] is None:
            kwargs['x0'] = codomain.zeros()

        self._A = A
        self._domain = domain
        self._codomain = codomain

        self._check_options(**kwargs)
        self._options = kwargs

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
        """
        The linear operator :math:`A` of which this object is the inverse :math:`A^{-1}`.

        The linear operator :math:`A` can be modified in place, or replaced entirely
        through the setter. A substitution should only be made in cases where
        no other options are viable, as it breaks the one-to-one map between
        the original linear operator :math:`A` (passed to the constructor) and the
        current `InverseLinearOperator` object :math:`A^{-1}`. Use with extreme care!

        """
        return self._A
    
    @linop.setter
    def linop(self, a):
        """ Set the linear operator :math:`A` of which this object is the inverse :math:`A^{-1}`. """
        assert isinstance(a, LinearOperator)
        assert a.domain is self.domain
        assert a.codomain is self.codomain
        self._A = a

    def _check_options(self, **kwargs):
        """ Check whether the options passed to the solver class are valid. """
        for key, value in kwargs.items():

            if key == 'x0':
                if value is not None:
                    assert isinstance(value, Vector), "x0 must be a Vector or None"
                    assert value.space == self.codomain, "x0 belongs to the wrong VectorSpace"
            elif key == 'tol':
                assert is_real(value), "tol must be a real number"
                assert value > 0, "tol must be positive"
            elif key == 'maxiter':
                assert isinstance(value, int), "maxiter must be an int"
                assert value > 0, "maxiter must be positive"
            elif key == 'verbose':
                assert isinstance(value, bool), "verbose must be a bool"

    def toarray(self):
        raise NotImplementedError('toarray() is not defined for InverseLinearOperators.')

    def tosparse(self):
        raise NotImplementedError('tosparse() is not defined for InverseLinearOperators.')

    def get_info(self):
        """ Returns the previous convergence information. """
        return self._info

    def get_options(self, key=None):
        """Get a copy of all the solver options, or a specific value of interest.

        Parameters
        ----------
        key : str | None
            Name of the specific option of interest (default: None).

        Returns
        -------
        dict | type(self._options['key']) | None
            If `key` is given, get the specific option of interest. If there is
            no such option, `None` is returned instead. If `key` is not given,
            get a copy of all the solver options in a dictionary.

        """
        if key is None:
            return self._options.copy()
        else:
            return self._options.get(key)

    def set_options(self, **kwargs):
        """ Set the solver options by passing keyword arguments. """
        self._check_options(**kwargs)
        self._options.update(kwargs)

    def transpose(self, conjugate=False):
        cls     = type(self)
        At      = self.linop.transpose(conjugate=conjugate)
        options = self._options
        return cls(At, **options)

#===============================================================================
class LinearSolver(ABC):
    """
    Solver for the square linear system Ax=b, where x and b belong to the same vector space V.

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
    def transpose(self):
        """Return the transpose of the LinearSolver."""
        pass

    @abstractmethod
    def solve(self, rhs, out=None):
        pass

    @property
    def T(self):
        return self.transpose()
