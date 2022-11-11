# coding: utf-8
#
# Copyright 2018 Yaman Güçlü, Jalal Lakhlili

from abc import ABCMeta, abstractmethod
import numpy as np
#from math import sqrt

__all__ = ['VectorSpace', 'Vector', 'LinearOperator', 
'SumLinearOperator', 'ComposedLinearOperator', 'ScaledLinearOperator', 'PowerLinearOperator', 'ZeroOperator', 'IdentityOperator', 
'InverseLinearOperator', 'LinearSolver', 
'Matrix']

#===============================================================================
class VectorSpace( metaclass=ABCMeta ):
    """
    Generic vector space V.

    """

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
        """ abstract method - Get a copy of the null element of the vector space V. """

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

    @abstractmethod
    def toarray( self, **kwargs ):
        """ Convert to Numpy 1D array. """

    @abstractmethod
    def copy( self ):
        pass

    @abstractmethod
    def __neg__( self ):
        pass

    @abstractmethod
    def __mul__( self, c ):
        pass

    @abstractmethod
    def __rmul__( self, c ):
        pass

    @abstractmethod
    def __add__( self, v ):
        pass

    @abstractmethod
    def __sub__( self, v ):
        pass

    @abstractmethod
    def __imul__( self, c ):
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
        """ The domain of the linear operator - an element of Vectorspace """
        pass

    @property
    @abstractmethod
    def codomain( self ):
        """ The codomain of the linear operator - an element of Vectorspace """
        pass

    @property
    @abstractmethod
    def dtype( self ):
        pass

    @property
    def T( self ):
        return self.transpose()

    @abstractmethod
    def dot( self, v, out=None ):
        """ abstract method - Evaluates self at v, an element of Vector """
        pass

    #-------------------------------------
    # Magic methods
    #-------------------------------------
    def __neg__(self):
        return ScaledLinearOperator(self._domain, self._codomain, self, -1.0)

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

    def __add__( self, B ):
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
    def transpose( self ):
        raise NotImplementedError()

    def inverse( self, solver, **kwargs ):
        return InverseLinearOperator(self, solver=solver, **kwargs)

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
class InverseLinearOperator( LinearOperator ):
    """
    Iterative solver for square linear system Ax=b, where x and b belong to (normed)
    vector space V.

    """
    def __new__(cls, linop, solver=None, **kwargs):
        
        #cls._check_options(options)

        if solver is not None:
            if solver == 'cg':
                from psydac.linalg2.iterative_solvers import ConjugateGradient
                obj = ConjugateGradient(linop, solver=None, **kwargs)
            elif solver == 'pcg':
                from psydac.linalg2.iterative_solvers import PConjugateGradient
                obj = PConjugateGradient(linop, solver=None, **kwargs)
            elif solver == 'bicg':
                from psydac.linalg2.iterative_solvers import BiConjugateGradient
                obj = BiConjugateGradient(linop, solver=None, **kwargs)
            else:
                raise ValueError(f"Required solver '{solver}' not understood.")

            return obj
        else:
            return super().__new__(cls)

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

    @property
    def linop( self ):
        return self._linop

    @property
    def options( self ):
        return self._options

    @abstractmethod
    def _update_options(self):
        pass

    def getoptions( self ):
        for key, value in self.options.items():
            print(key, ": ", value)

    def setoptions( self, **kwargs ):
        # check feasibility of kwargs
        for key, value in kwargs.items():
            setattr(self, '_'+key, value)
        self._update_options()

    @staticmethod
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
        from psydac.linalg2.block   import BlockMatrix, BlockVector
        from psydac.linalg2.stencil import StencilMatrix, StencilVector

        # Sanity checks
        assert isinstance(A, (StencilMatrix, BlockMatrix))
        assert isinstance(b, (StencilVector, BlockVector))
        assert A.codomain == A.domain
        assert A.codomain == b.space

        #-------------------------------------------------------------
        # Handle the case of a block linear system
        if isinstance(A, BlockMatrix):
            x = [InverseLinearOperator.jacobi(A[i, i], bi) for i, bi in enumerate(b.blocks)]
            return BlockVector(b.space, blocks=x)
        #-------------------------------------------------------------

        V = b.space
        i = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
        ii = i + (0,) * V.ndim

        x = b.copy()
        x[i] /= A[ii]
        x.update_ghost_regions()

        return x

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
        #from psydac.linalg2.stencil import StencilVector

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

    def transpose( self ):
        t_addends = ()
        for a in self._addends:
            t_addends = (*t_addends, a.T)
        return SumLinearOperator(domain=self._codomain, codomain=self._domain, *t_addends)

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

    def transpose( self ):
        t_multiplicants = ()
        for a in self._multiplicants:
            t_multiplicants = (a.T, *t_multiplicants)
        return ComposedLinearOperator(domain=self._codomain, codomain=self._domain, *t_multiplicants)

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

    def transpose( self ):
        return ScaledLinearOperator(domain=self._codomain, codomain=self._domain, c=self._scalar, A=self._operator.T)

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

    def transpose( self ):
        return PowerLinearOperator(domain=self._codomain, codomain=self._domain, A=self._operator.T, n=self._factorial)

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

    def transpose( self ):
        return ZeroOperator(domain=self._codomain, codomain=self._domain)

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

    def transpose( self ):
        """ Could return self, but by convention returns new object. """
        return IdentityOperator(domain=self._domain)

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

#===============================================================================
class Matrix( LinearOperator ):
    """
    Linear operator whose coefficients can be viewed as a 2D matrix.

    """
    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    @property
    @abstractmethod
    def domain( self ):
        """ The domain of the linear operator - an element of Vectorspace """
        pass

    @property
    @abstractmethod
    def codomain( self ):
        """ The codomain of the linear operator - an element of Vectorspace """
        pass

    @property
    @abstractmethod
    def dtype( self ):
        pass

    @property
    def T( self ):
        return self.transpose()

    def transpose( self ):
        raise NotImplementedError()

    @abstractmethod
    def dot( self, v, out=None ):
        """ abstract method - Evaluates self at v, an element of Vector """
        pass

    @abstractmethod
    def toarray( self, **kwargs ):
        """ Convert to Numpy 2D array. """

    @abstractmethod
    def tosparse( self, **kwargs ):
        """ Convert to any Scipy sparse matrix format. """

    @abstractmethod
    def copy(self):
        """ Create an identical copy of the matrix. """

    @abstractmethod
    def __neg__(self):
        """ Get the opposite matrix, i.e. a copy with negative sign. """

    @abstractmethod
    def __mul__(self, a):
        """ Multiply by scalar. """

    @abstractmethod
    def __rmul__(self, a):
        """ Multiply by scalar. """

    @abstractmethod
    def __add__(self, m):
        """ Add matrix. """

    @abstractmethod
    def __sub__(self, m):
        """ Subtract matrix. """

    @abstractmethod
    def __imul__(self, a):
        """ Multiply by scalar, in place. """

    @abstractmethod
    def __iadd__(self, m):
        """ Add matrix, in place. """

    @abstractmethod
    def __isub__(self, m):
        """ Subtract matrix, in place. """

    #-------------------------------------
    # Methods with default implementation
    #-------------------------------------
    def __truediv__(self, a):
        """ Divide by scalar. """
        return self * (1.0 / a)

    def __itruediv__(self, a):
        """ Divide by scalar, in place. """
        self *= 1.0 / a
        return self
