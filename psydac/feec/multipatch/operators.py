
from mpi4py import MPI

import numpy as np


from sympde.topology import element_of, elements_of
from sympde.calculus import grad, dot, inner, rot, div
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.api.discretization import discretize
from psydac.linalg.basic import LinearOperator
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix
from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.direct_solvers import SparseSolver
from psydac.fem.basic   import FemField
from psydac.fem.vector import ProductFemSpace, VectorFemSpace

from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl
from psydac.feec.derivatives import Gradient_2D

from psydac.feec.derivatives import DiffOperator

#===============================================================================
class FemLinearOperator( LinearOperator ):
    """
    Linear operator, with an additional Fem layer
    """

    def __init__( self, fem_domain=None, fem_codomain=None):
        assert fem_domain
        self._fem_domain   = fem_domain
        if fem_codomain:
            self._fem_codomain = fem_codomain
        else:
            self._fem_codomain = fem_domain
        self._domain   = self._fem_domain.vector_space
        self._codomain = self._fem_codomain.vector_space

    @property
    def domain( self ):
        # if self._domain is None:
        #     return self._fem_domain.vector_space
        # else:
        return self._domain

    @property
    def codomain( self ):
        # if self._codomain is None:
        #     return self._fem_codomain.vector_space
        # else:
        return self._codomain

    @property
    def fem_domain( self ):
        return self._fem_domain

    @property
    def fem_codomain( self ):
        return self._fem_codomain

    @property
    def T(self):
        return self.transpose()

    # ...
    def transpose(self):
        raise NotImplementedError('Class does not provide a transpose() method')

    #-------------------------------------
    # Deferred methods
    #-------------------------------------
    #
    # @abstractmethod
    # def dot( self, v, out=None ):
    #     pass


#===============================================================================
class ConformingProjection( FemLinearOperator ):
    """
    Conforming projection from global broken space to conforming global space

    proj.dot(v) returns the conforming projection of v, computed by solving linear system


    """
    def __init__(self, V0h, domain_h):

        FemLinearOperator.__init__(self, fem_domain=V0h)

        # self._fem_domain   = V0h
        # self._fem_codomain = V0h
        #
        # self._domain   = self._fem_domain.vector_space
        # self._codomain = self._fem_codomain.vector_space

        V0             = V0h.symbolic_space
        domain         = V0.domain

        # domain_h = V0h.domain  # would be nice

        u, v = elements_of(V0, names='u, v')
        expr   = u*v  # dot(u,v)

        I      = domain.interfaces  # note: interfaces does not include the boundary
        expr_I = ( plus(u)-minus(u) )*( plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(I, expr_I))

        ah = discretize(a, domain_h, [V0h, V0h])

        self._A = ah.assemble()

        spaces = self._A.domain.spaces
        sp1    = spaces[0]
        sp2    = spaces[1]

        s1 = sp1.starts[I.axis]
        e1 = sp1.ends[I.axis]

        s2 = sp2.starts[I.axis]
        e2 = sp2.ends[I.axis]

        d1     = V0h.spaces[0].degree[I.axis]
        d2     = V0h.spaces[1].degree[I.axis]

        self._A[0,0][:,:,:,:] = 0
        self._A[1,1][:,:,:,:] = 0
        self._A[0,1][:,:,:,:] = 0
        self._A[1,0][:,:,:,:] = 0

        self._A[0,0][:,:,0,0]  = 1
        self._A[1,1][:,:,0,0]  = 1

        self._A[0,0][:,e1,0,0] = 1/2
        self._A[1,1][:,s2,0,0] = 1/2

        self._A[0,1][:,d1,0,-d2] = 1/2
        self._A[1,0][:,s2,0, d1] = 1/2

    # ...
    def __call__( self, f ):

        coeffs = self._A.dot(f.coeffs)
        return FemField(self.fem_codomain, coeffs=coeffs)

    # ...
    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        f = FemField(self.fem_domain, coeffs=f_coeffs)
        return self(f).coeffs

#===============================================================================
class BrokenMass_V0( FemLinearOperator ):
    """
    Broken mass matrix, seen as a LinearOperator
    """
    def __init__( self, V0h, domain_h):

        FemLinearOperator.__init__(self, fem_domain=V0h)
        # self._fem_domain   = V0h
        # self._fem_codomain = V0h
        # self._domain   = self._fem_domain.vector_space
        # self._codomain = self._fem_codomain.vector_space

        V0 = V0h.symbolic_space
        domain = V0.domain
        # domain_h = V0h.domain  # would be nice
        u, v = elements_of(V0, names='u, v')
        expr   = u*v  # dot(u,v)
        a = BilinearForm((u,v), integral(domain, expr))
        ah = discretize(a, domain_h, [V0h, V0h])
        self._M = ah.assemble() #.toarray()

    def mat(self):
        return self._M

    def __call__( self, f ):
        # Fem layer
        Mf_coeffs = self.dot(f.coeffs)
        return FemField(self.fem_domain, coeffs=Mf_coeffs)

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        return self._M.dot(f_coeffs)

#===============================================================================
class BrokenMass_V1( FemLinearOperator ):
    """
    Broken mass matrix in V1, seen as a LinearOperator
    """
    def __init__( self, V1h, domain_h):

        FemLinearOperator.__init__(self, fem_domain=V1h)

        V1 = V1h.symbolic_space
        domain = V1.domain
        # domain_h = V0h.domain  # would be nice
        # self._domain   = V1h
        # self._codomain = V1h
        u, v = elements_of(V1, names='u, v')
        expr   = dot(u,v)
        a = BilinearForm((u,v), integral(domain, expr))
        ah = discretize(a, domain_h, [V1h, V1h])
        self._M = ah.assemble() #.toarray()

    def mat(self):
        return self._M

    def __call__( self, f ):
        Mf_coeffs = self.dot(f.coeffs)
        return FemField(self.fem_domain, coeffs=Mf_coeffs)

    def dot( self, f_coeffs, out=None ):
        return self._M.dot(f_coeffs)

#==============================================================================
class ComposedLinearOperator( FemLinearOperator ):

    def __init__( self, B, A ):
        assert isinstance(A, FemLinearOperator)
        assert isinstance(B, FemLinearOperator)
        assert B.fem_domain == A.fem_codomain
        FemLinearOperator.__init__(
            self, fem_domain=A.fem_domain, fem_codomain=B.fem_codomain
        )
        self._A = A
        self._B = B

    def __call__( self, f ):
        return self._B(self._A(f))

    def dot( self, f_coeffs, out=None ):
        return self._B.dot(self._A.dot(f_coeffs))

#==============================================================================
class IdLinearOperator( FemLinearOperator ):

    def __init__( self, V ):
        FemLinearOperator.__init__(self, fem_domain=V)

    def __call__( self, f ):
        # fem layer
        return f

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
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

    def __call__( self, f ):
        # fem layer
        return self._c * self._A(f)

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        return self._c * self._A.dot(f_coeffs)

#==============================================================================
class BrokenGradient_2D(FemLinearOperator):

    def __init__(self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V0h, fem_codomain=V1h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix for i, D0i in enumerate(D0s)})

    def dot(self, u0_coeffs, out = None):
        return self._matrix.dot(u0_coeffs, out=out)

    def __call__(self, u0):
        return FemField(self.fem_codomain, coeffs = self.dot(u0.coeffs))

    def transpose(self):
        return BrokenTransposedGradient_2D(self.fem_domain, self.fem_codomain)

#==============================================================================
class BrokenTransposedGradient_2D( FemLinearOperator ):

    def __init__( self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V0h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix.T for i, D0i in enumerate(D0s)})

    def dot(self, u0_coeffs, out = None):
        return self._matrix.dot(u0_coeffs, out=out)

    def __call__(self, u0):
        return FemField(self.fem_codomain, coeffs = self.dot(u0.coeffs))

    def transpose(self):
        return BrokenGradient_2D(self.fem_codomain, self.fem_domain)

#==============================================================================
from sympy import Tuple

# def multipatch_Moments_Hcurl(f, V1h, domain_h):
def ortho_proj_Hcurl(EE, V1h, domain_h, M1):
    """
    return vector of moments of E against V1h basis
    """
    assert isinstance(EE, Tuple)
    V1 = V1h.symbolic_space
    v = element_of(V1, name='v')
    # x,y = V1.domain.coordinates
    # EE = Tuple(2*x, 2*y)
    # print("in op:", type(EE))
    l = LinearForm(v, integral(V1.domain, dot(v,EE)))
    lh = discretize(l, domain_h, V1h)
    b = lh.assemble()
    sol_coeffs, info = pcg(M1.mat(), b, pc="jacobi", tol=1e-10)

    return FemField(V1h, coeffs=sol_coeffs)

#==============================================================================
class Multipatch_Projector_Hcurl:

    """
    to apply the Hcurl projection on every patch
    """
    def __init__(self, V1h, nquads=None):

        self._P1s = [Projector_Hcurl(V, nquads=nquads) for V in V1h.spaces]
        self._V1h  = V1h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        E1s = [P(fun) for P, fun, in zip(self._P1s, funs_log)]

        E1_coeffs = BlockVector(self._V1h.vector_space, \
                blocks = [E1j.coeffs for E1j in E1s])

        return FemField(self._V1h, coeffs = E1_coeffs)

#==============================================================================
class Multipatch_Projector_H1:
    """
    to apply the H1 projection on every patch
    """
    def __init__(self, V0h):

        self._P0s = [Projector_H1(V) for V in V0h.spaces]
        self._V0h  = V0h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        u0s = [P(fun) for P, fun, in zip(self._P0s, funs_log)]

        u0_coeffs = BlockVector(self._V0h.vector_space, \
                blocks = [u0j.coeffs for u0j in u0s])

        return FemField(self._V0h, coeffs = u0_coeffs)

#==============================================================================
def get_scalar_patch_fields(u, V0h):
    return [FemField(V, coeffs=c) for V, c in zip(V0h.spaces, u.coeffs)]

def get_vector_patch_fields(E, V1h):
    return [FemField(V, coeffs=c) for V, c in zip(V1h.spaces, E.coeffs)]
