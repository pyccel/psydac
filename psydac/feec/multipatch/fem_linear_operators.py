# coding: utf-8

from mpi4py import MPI

from scipy.sparse import eye as sparse_id

from psydac.linalg.basic import LinearOperator
from psydac.fem.basic   import FemField

#===============================================================================
class FemLinearOperator( LinearOperator ):
    """
    Linear operators with an additional Fem layer
    """

    def __init__( self, fem_domain=None, fem_codomain=None, matrix=None):
        assert fem_domain
        self._fem_domain   = fem_domain
        if fem_codomain:
            self._fem_codomain = fem_codomain
        else:
            self._fem_codomain = fem_domain
        self._domain   = self._fem_domain.vector_space
        self._codomain = self._fem_codomain.vector_space

        self._matrix = matrix

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    @property
    def fem_domain( self ):
        return self._fem_domain

    @property
    def fem_codomain( self ):
        return self._fem_codomain

    @property
    def matrix( self ):
        return self._matrix

    @property
    def T(self):
        return self.transpose()

    # ...
    def transpose(self):
        raise NotImplementedError('Class does not provide a transpose() method')

    # ...
    def to_sparse_matrix( self ):
        if self._matrix:
            return self._matrix.tosparse()
        else:
            raise NotImplementedError('Class does not provide a get_sparse_matrix() method without a matrix')

    # ...
    def __call__( self, f ):
        if self._matrix:
            coeffs = self._matrix.dot(f.coeffs)
            return FemField(self.fem_codomain, coeffs=coeffs)
        else:
            raise NotImplementedError('Class does not provide a __call__ method without a matrix')

    # ...
    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        if self._matrix:
            f = FemField(self.fem_domain, coeffs=f_coeffs)
            return self(f).coeffs
        else:
            raise NotImplementedError('Class does not provide a dot method without a matrix')

    # ...
    def __mul__(self, c):
        return MultLinearOperator(c, self)

    # ...
    def __rmul__(self, c):
        return MultLinearOperator(c, self)

    # ...
    def __add__(self, C):
        assert isinstance(C, FemLinearOperator)
        return SumLinearOperator(C, self)

    # ...
    def __sub__(self, C):
        assert isinstance(C, FemLinearOperator)
        return SumLinearOperator(C, -self)

    # ...
    def __neg__(self):
        return MultLinearOperator(-1, self)


#==============================================================================
class ComposedLinearOperator( FemLinearOperator ):
    """
    operator L = L_1 .. L_n
    with L_i = self._operators[i-1]
    (so, the last one is applied first, like in a product)
    """
    def __init__( self, operators ):
        n = len(operators)
        assert all([isinstance(operators[i], FemLinearOperator) for i in range(n)])
        assert all([operators[i].fem_domain == operators[i+1].fem_codomain for i in range(n-1)])
        FemLinearOperator.__init__(
            self, fem_domain=operators[-1].fem_domain, fem_codomain=operators[0].fem_codomain
        )
        self._operators = operators
        self._n = n

        # matrix not defined by matrix product because it could break the Stencil Matrix structure

    def to_sparse_matrix( self ):
        mat = self._operators[-1].to_sparse_matrix()
        for i in range(2, self._n+1):
            mat = self._operators[-i].to_sparse_matrix() * mat
        return mat

    def __call__( self, f ):
        v = self._operators[-1](f)
        for i in range(2, self._n+1):
            v = self._operators[-i](v)
        return v

    def dot( self, f_coeffs, out=None ):
        v_coeffs = self._operators[-1].dot(f_coeffs)
        for i in range(2, self._n+1):
            v_coeffs = self._operators[-i].dot(v_coeffs)
        return v_coeffs


#==============================================================================
class IdLinearOperator( FemLinearOperator ):

    def __init__( self, V ):
        FemLinearOperator.__init__(self, fem_domain=V)

    def to_sparse_matrix( self ):
        return sparse_id( self.fem_domain.nbasis )

    def __call__( self, f ):
        return f

    def dot( self, f_coeffs, out=None ):
        return f_coeffs

#==============================================================================
class SumLinearOperator( FemLinearOperator ):

    def __init__( self, B, A ):
        assert isinstance(A, FemLinearOperator)
        assert isinstance(B, FemLinearOperator)
        assert B.fem_domain == A.fem_domain
        assert B.fem_codomain == A.fem_codomain
        FemLinearOperator.__init__(
            self, fem_domain=A.fem_domain, fem_codomain=A.fem_codomain
        )
        self._A = A
        self._B = B

    def to_sparse_matrix( self ):
        return self._A.to_sparse_matrix() + self._B.to_sparse_matrix()

    def __call__( self, f ):
        # fem layer
        return  self._B(f) + self._A(f)

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        return  self._B.dot(f_coeffs) + self._A.dot(f_coeffs)

#==============================================================================
class MultLinearOperator( FemLinearOperator ):

    def __init__( self, c, A ):
        assert isinstance(A, FemLinearOperator)
        FemLinearOperator.__init__(
            self, fem_domain=A.fem_domain, fem_codomain=A.fem_codomain
        )
        self._A = A
        self._c = c

    def to_sparse_matrix( self ):
        return self._c * self._A.to_sparse_matrix()

    def __call__( self, f ):
        # fem layer
        return self._c * self._A(f)

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        return self._c * self._A.dot(f_coeffs)

