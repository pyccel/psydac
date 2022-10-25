# coding: utf-8
#
# Copyright 2018 Yaman Güçlü, Jalal Lakhlili

from abc import ABCMeta, abstractmethod
import numpy as np

__all__ = ['VectorSpace', 'Vector', 'LinearOperator', 'SumLinearOperator', 'ComposedLinearOperator', 'ScaledLinearOperator', 'ZeroOperator', 'IdentityOperator', ]

#===============================================================================
class VectorSpace( metaclass=ABCMeta ):
    """
    Generic vector space V.

    """

    ### axpy, ... lapack

    @property
    @abstractmethod
    def dimension( self ):
        """ The dimension of the vector space. """

    @property
    @abstractmethod
    def dtype( self ):
        """ The data type of the space elements. """

    @abstractmethod
    def zeros( self ):
        """ Get a copy of the null element of the vector space V. """

#===============================================================================
class Vector( metaclass=ABCMeta ):
    """
    Element of a (normed) vector space V.

    """
    @property
    def shape( self ):
        """ Returns a tuple containing the dimension of the ``VectorSpace`` to which the vector belongs to. """
        return (self.space.dimension, )

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def space( self ):
        """ Returns the ``Vectorspace`` to which the vector belongs. """
        pass

    @property
    @abstractmethod
    def dtype( self ):
        """ The data type of the vector. """
        pass

    @abstractmethod
    def dot( self, v ):
        """ Evaluate inner product between self and v. """
        pass

    # @abstractmethod
    # def toarray( self, **kwargs ):
    #     """ Convert to Numpy 1D array. """

    @abstractmethod
    def copy( self ):
        pass

    @abstractmethod
    def __neg__( self ):
        pass

    @abstractmethod
    def __mul__( self, a ):
        pass

    @abstractmethod
    def __rmul__( self, a ):
        pass

    @abstractmethod
    def __add__( self, v ):
        pass

    @abstractmethod
    def __sub__( self, v ):
        pass

    @abstractmethod
    def __imul__( self, a ):
        pass

    @abstractmethod
    def __iadd__( self, v ):
        pass

    @abstractmethod
    def __isub__( self, v ):
        pass

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def __truediv__( self, a ):
        return self * (1.0 / a)

    def __itruediv__( self, a ):
        self *= 1.0 / a
        return self

#===============================================================================
class LinearOperator( metaclass=ABCMeta ):
    """
    Linear operator acting between two (normed) vector spaces V (domain)
    and W (codomain).

    """
    @property
    def shape( self ):
        """ A tuple containing the dimension of the codomain and domain. """
        return (self.codomain.dimension, self.domain.dimension)

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def domain( self ):
        pass

    @property
    @abstractmethod
    def codomain( self ):
        pass

    @property
    @abstractmethod
    def dtype( self ):
        pass

    @abstractmethod
    def dot( self, v, out=None ):
        pass

    def __add__( self, B ):
        """ Creates an object of class :ref:`SumLinearOperator <sumlinearoperator>` unless B is a :ref:`ZeroOperator <zerooperator>` in which case self is returned. """
        assert isinstance(B, LinearOperator)
        if isinstance(B, ZeroOperator):
            return self
        else:
            return SumLinearOperator(self._domain, self._codomain, self, B)

    def __mul__( self, c ):
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

    def __rmul__( self, c ):
        """ Calles :ref:`__mul__ <mul>` instead. """
        return self * c

    def __matmul__( self, B ):
        """ Creates an object of class :ref:`ComposedLinearOperator <composedlinearoperator>`. """
        assert isinstance(B, LinearOperator)
        assert self._domain == B.codomain
        if isinstance(B, ZeroOperator):
            return ZeroOperator(B.domain, self._codomain)
        elif isinstance(B, IdentityOperator):
            return self
        else:
            return ComposedLinearOperator(B.domain, self._codomain, self, B)

    def __pow__( self, n ):
        """ Creates an object of class :ref:`PowerLinearOperator <powerlinearoperator>`. """
        return PowerLinearOperator(self._domain, self._codomain, self, n)

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def idot( self, v, out ):
        """
        Overwrites the vector out, element of codomain, by adding self evaluated at v.
        
        """
        assert isinstance(v, Vector)
        assert v.space == self.domain
        assert isinstance(out, Vector)
        assert out.space == self.codomain
        out += self.dot(v)

#===============================================================================
class LinearSolver( LinearOperator ):
    """
    Solver for square linear system Ax=b, where x and b belong to (normed)
    vector space V.

    """
    @property
    def shape( self ):
        return (self.space.dimension, self.space.dimension)

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def space( self ):
        pass

    @property
    @abstractmethod
    def domain( self ):
        pass

    @property
    @abstractmethod
    def codomain( self ):
        pass

    @property
    @abstractmethod
    def dtype( self ):
        pass

    @abstractmethod
    def solve( self, rhs, out=None, transposed=False ):
        pass

    def dot( self, rhs, out=None, transposed=False):
        return self.solve(rhs, out, transposed)

#===============================================================================
class SumLinearOperator( LinearOperator ):
    """
    A sum of linear operatos acting between the same (normed) vector spaces V (domain) and W (codomain).

    """
    def __new__( cls, domain, codomain, *args ):

        if len(args) == 0:
            return ZeroOperator(domain,codomain)
        elif len(args) == 1:
            return args[0]
        else:
            return super().__new__(cls)
    
    def __init__( self, domain, codomain, *args ):

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

        self._domain = domain
        self._codomain = codomain
        self._addends = addends
    
    @property
    def domain( self ):
        """ The domain of the linear operator, element of class ``VectorSpace``. """
        return self._domain

    @property
    def codomain( self ):
        """ The codomain of the linear operator, element of class ``VectorSpace``. """
        return self._codomain

    @property
    def addends( self ):
        """ A tuple containing the addends of the linear operator, elements of class ``LinearOperator``. """
        return self._addends

    @property
    def dtype( self ):
        """
        todo

        """
        return None

    def simplifiy( self, addends ):
        class_list = [addends[i].__class__.__name__ for i in range(len(addends))]
        unique_list = list(set(class_list))
        out = ZeroOperator(domain=self._domain, codomain=self._codomain)
        for i, j in enumerate(unique_list): #better?: for i in range(len(unique_list)): 
            indices = [k for k, l in enumerate(class_list) if class_list[k] == unique_list[i]] #for k in range(len(class_list))
            if len(indices) == 1:
                out += addends[indices[0]]
            else:
                A = addends[indices[0]] # might change addends[indices[0]]? try .copy / .copy() or implement ...
                for n in range(len(indices)-1):
                    A += addends[indices[n+1]]
                out += A
        return out

    def dot( self, v, simplified = False ):
        """ Evaluates SumLinearOperator object at a vector v element of domain. """
        assert isinstance(v, Vector)
        assert v.space == self._domain

        if simplified == False:
            self._addends = self.simplifiy(self._addends).addends
        elif simplified != True:
            raise ValueError('simplified expects True or False.')

        out = self._codomain.zeros()
        for a in self._addends:
            a.idot(v, out)
        return out

#===============================================================================
class ComposedLinearOperator( LinearOperator ):
    def __init__( self, domain, codomain, *args ):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)
        
        for a in args:
            assert isinstance(a, LinearOperator)
        assert args[0].codomain == codomain
        assert args[-1].domain == domain

        for i in range(len(args)-1):
            assert args[i].domain == args[i+1].codomain

        multiplicants = ()
        for a in args:
            if isinstance(a, ComposedLinearOperator):
                multiplicants = (*multiplicants, *a.multiplicants)
            else:
                multiplicants = (*multiplicants, a)

        self._domain = domain
        self._codomain = codomain
        self._multiplicants = multiplicants

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def multiplicants( self ):
        return self._multiplicants

    @property
    def dtype( self ):
        return None

    def dot( self, v ):
        assert isinstance(v,Vector)
        assert v.space == self._domain
        out = self._multiplicants[-1].dot(v)
        for i in range(1,len(self._multiplicants)):
            out = self._multiplicants[-1-i].dot(out)
        return out

#===============================================================================
class ScaledLinearOperator( LinearOperator ):
    def __init__( self, domain, codomain, c, A ):

        assert np.isscalar(c)
        assert isinstance(A, LinearOperator)
        assert domain == A.domain
        assert codomain == A.codomain

        if isinstance(A,ScaledLinearOperator):
            scalar = A.scalar*c
            operator = A.operator
        else:
            scalar = c
            operator = A

        self._operator = operator
        self._scalar = scalar
        self._domain = domain
        self._codomain = codomain

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def scalar( self ):
        return self._scalar

    @property
    def operator( self ):
        return self._operator

    @property
    def dtype( self ):
        return None

    def dot( self, v ):
        assert isinstance(v, Vector)
        assert v.space == self._domain        
        return self._operator.dot(v) * self._scalar

#===============================================================================
class PowerLinearOperator( LinearOperator ):
    def __new__( cls, domain, codomain, A, n ):

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
    
    def __init__( self, domain, codomain, A, n ):

        if isinstance(A, PowerLinearOperator):
            self._operator = A.operator
            self._factorial = A.factorial*n
        else:
            self._operator = A
            self._factorial = n
        self._domain = domain
        self._codomain = codomain

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return None

    @property
    def operator( self ):
        return self._operator

    @property
    def factorial( self ):
        return self._factorial

    def dot( self, v ):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        out = v.copy()
        for i in range(self._factorial):
            out = self._operator.dot(out)
        return out

#===============================================================================
class ZeroOperator( LinearOperator ):
    def __init__(self, domain, codomain ):

        assert isinstance(domain, VectorSpace)
        assert isinstance(codomain, VectorSpace)

        self._domain = domain
        self._codomain = codomain

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return None

    def dot( self, v ):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        return self._codomain.zeros()

    def __add__( self, B ):
        assert isinstance(B, LinearOperator)
        assert self._domain == B.domain
        assert self._codomain == B.codomain
        return B

    def __radd__( self, A ):
        return self + A

    def __mul__( self, c ):
        assert np.isscalar(c)
        return self

    def __rmul__( self, c ):
        return self * c

    def __matmul__( self, B ):
        assert isinstance(B, LinearOperator)
        assert self._domain == B.codomain
        return ZeroOperator(domain=B.domain, codomain=self._codomain)

    def __rmatmul__( self, A ):
        assert isinstance(A, LinearOperator)
        assert self._codomain == A.domain
        return ZeroOperator(domain=self._domain, codomain=A.codomain)

#===============================================================================
class IdentityOperator( LinearOperator ):
    def __init__(self, domain, codomain=None ):

        assert isinstance(domain, VectorSpace)
        if codomain:
            assert isinstance(codomain, VectorSpace)
            assert domain == codomain

        self._domain = domain
        self._codomain = domain

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def dtype( self ):
        return None

    def dot( self, v ):
        assert isinstance(v, Vector)
        assert v.space == self._domain
        return v

    def __matmul__( self, B ):
        assert isinstance(B, LinearOperator)
        assert self._domain == B.codomain
        return B

    def __rmatmul__( self, A ):
        assert isinstance(A, LinearOperator)
        assert self._codomain == A.domain
        return A