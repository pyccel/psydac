
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
from psydac.fem.vector import ProductFemSpace, VectorFemSpace, VectorFemField

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
    def __init__( self, V0h_1, V0h_2, domain_h_1, domain_h_2, V0h, domain_h, homogeneous_bc=False, verbose=False):

        FemLinearOperator.__init__(self, fem_domain=V0h)

        # self._fem_domain   = V0h
        # self._fem_codomain = V0h
        #
        # self._domain   = self._fem_domain.vector_space
        # self._codomain = self._fem_codomain.vector_space

        V0 = V0h.symbolic_space
        domain = V0.domain
        self._verbose = verbose

        # domain_h = V0h.domain  # would be nice

        u, v = elements_of(V0, names='u, v')
        expr   = u*v  # dot(u,v)

        kappa  = 10**4
        I = domain.interfaces  # note: interfaces does not include the boundary
        expr_I = kappa*( plus(u)-minus(u) )*( plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        if homogeneous_bc:
            B = domain.boundary
            expr_B = kappa*( u*v )
            a = BilinearForm((u,v), integral(domain, expr) + integral(I, expr_I) + integral(B, expr_B))
        else:
            a = BilinearForm((u,v), integral(domain, expr) + integral(I, expr_I))

        ah = discretize(a, domain_h, [V0h, V0h])    # ... or (V0h, V0h)?

        self._A = ah.assemble() #.toarray()
        self._solver = SparseSolver( self._A.tosparse() )

        V0_1 = V0h_1.symbolic_space
        V0_2 = V0h_2.symbolic_space

        v1, f1 = elements_of(V0_1, names='v1, f1')
        v2, f2 = elements_of(V0_2, names='v2, f2')

        l1 = LinearForm(v1, integral(V0_1.domain, f1*v1))
        l2 = LinearForm(v2, integral(V0_2.domain, f2*v2))

        self._lh_1 = discretize(l1, domain_h_1, V0h_1)
        self._lh_2 = discretize(l2, domain_h_2, V0h_2)

    def __call__( self, f ):
        # Fem field layer

        # f = VectorFemField(self.fem_domain, coeffs=f_coeffs)
        f1,f2 = f.fields

        b1 = self._lh_1.assemble(f1=f1)
        b2 = self._lh_2.assemble(f2=f2)
        b  = BlockVector(self.codomain, blocks=[b1, b2])
        sol_coeffs, info = pcg( self._A, b, pc="jacobi", tol=1e-6, verbose=self._verbose )

        return VectorFemField(self.fem_codomain, coeffs=sol_coeffs)

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        f = VectorFemField(self.fem_domain, coeffs=f_coeffs)
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
        return VectorFemField(self.fem_domain, coeffs=Mf_coeffs)

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
        return VectorFemField(self.fem_domain, coeffs=Mf_coeffs)

    def dot( self, f_coeffs, out=None ):
        return self._M.dot(f_coeffs)


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

class IdLinearOperator( FemLinearOperator ):

    def __init__( self, V ):
        FemLinearOperator.__init__(self, fem_domain=V)

    def __call__( self, f ):
        # fem layer
        return f

    def dot( self, f_coeffs, out=None ):
        # coeffs layer
        return f_coeffs


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



class BrokenGradient_2D( FemLinearOperator ):

    def __init__( self, V0hs, V1hs, V0h, V1h ):

        # V0hs is the list of single patch spaces
        # V0h is the multipatch space
        # same for V1
        # todo: provide only V0h and V1h (remove redundancy)

        FemLinearOperator.__init__(
            self, fem_domain=V0h, fem_codomain=V1h
        )

        # 'new' flag to try better implementations (doesn't work yet)
        self._new = False

        self._V0hs = V0hs  # needed ?
        self._V1hs = V1hs  # needed ?

        self._npatches = len(V0hs)
        assert self._npatches == len(V1hs)

        self._D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0hs, V1hs)]
        self._mats = [D0._matrix for D0 in self._D0s]

    def __call__( self, u0 ):
        # fem layer
        # u0 should be a multipatch V0 field

        if self._new:
            return VectorFemField(self.fem_codomain, coeffs = self.dot(u0.coeffs))

        else:
            grad_u0 = VectorFemField(self.fem_codomain)
            grad_u0_cs = [mat.dot(c) for mat, c in zip(self._mats, u0.coeffs)]

            for k in range(self._npatches):
                for b1, b2 in zip(grad_u0.coeffs[k]._blocks, grad_u0_cs[k]._blocks):
                    b1[:] = b2[:]

            # other option, using the gradient operators instead of the matrices:
            #     # creating a list of single patch fields with proper spaces and copying the coefficients,
            #     # because the TensorFemSpace class does not have an __eq__ magic method
            #     u0s = get_scalar_patch_fields(u0, self._V0hs)
            #     grad_u0s = [D0(u) for D0, u in zip(self._D0s, u0s)]
            #
            #     for k in range(self._npatches):
            #         for d in range(2):
            #             grad_u0.coeffs[k][d][:] = grad_u0s[k].fields[d].coeffs[:]      # patch k, component d

            grad_u0.coeffs.update_ghost_regions()

            return grad_u0

    def dot( self, u_coeffs, out=None ):
        # coeffs layer
        if self._new:
            # self._block_mat = BlockMatrix(self.domain, self.codomain, blocks = ...)
            blocks = [mat.dot(c) for mat, c in zip(self._mats, u_coeffs)]
            return BlockVector(self.codomain, blocks=blocks)
        else:

            u0 = VectorFemField(self.fem_domain, coeffs=u_coeffs)
            E_coeffs = self(u0).coeffs
            return E_coeffs

class BrokenTransposedGradient_2D( FemLinearOperator ):

    def __init__( self, V0hs, V1hs, V0h, V1h):

        # V0hs is the list of single patch spaces
        # V0h is the multipatch space
        # same for V1
        # todo: provide only V0h and V1h (remove redundancy)

        FemLinearOperator.__init__(
            self, fem_domain=V1h, fem_codomain=V0h
        )

        self._V0hs = V0hs  # needed ?
        self._V1hs = V1hs  # needed ?

        self._npatches = len(V0hs)
        assert self._npatches == len(V1hs)

        self._D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0hs, V1hs)]  # needed ?
        self._mats = [D0._matrix.T for D0 in self._D0s]

    def __call__( self, E1 ):
        # fem layer
        # return u0 with coeffs = D0_T E1.coeffs
        # E1 should be a multipatch V1 field

        # heavy but seems to be needed because of discrepancy between multipatch V1h and list of single patch operators....
        E1s_coeffs = [BlockVector(mat.domain) for mat in self._mats]

        for k in range(self._npatches):
            for E1s_c, E1_c in zip(E1s_coeffs, E1.coeffs):
                E1s_c._blocks[:] = E1_c[:]

        u0s_coeffs = [mat.dot(E1s_c) for mat, E1s_c in zip(self._mats, E1s_coeffs)]
        # back to the multi-patch field
        u0 = VectorFemField(self.fem_codomain)
        for u_c, us_c,  in zip(u0.coeffs, u0s_coeffs):
            u_c[:] = us_c[:]
        u0.coeffs.update_ghost_regions()
        return u0

    def dot( self, E_coeffs, out=None ):
        # coeffs layer
        # todo: remove the fem field layer here (but I'm not sure how)
        E1 = VectorFemField(self.fem_domain, coeffs=E_coeffs)
        E_coeffs = self(E1).coeffs
        return E_coeffs


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

    return VectorFemField(V1h, coeffs=sol_coeffs)



class Multipatch_Projector_Hcurl:

    """
    to apply the Hcurl projection on every patch
    """
    def __init__(self, V1hs, V1h, nquads=None):

        self._P1s = [Projector_Hcurl(V, nquads=nquads) for V in V1hs]
        self._V1h  = V1h   # multipatch Fem Space

    #--------------------------------------------------------------------------
    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        E1s = [P(fun) for P, fun, in zip(self._P1s, funs_log)]

        E1 = VectorFemField(self._V1h)

        for E_patch, E_c in zip(E1s, E1.coeffs):
            # set coefs for component d
            for d in [0,1]:
                E_c[d][:] = E_patch.fields[d].coeffs[:]

        # for k in range(2):
        #     for d in range(2):
        #         E1.coeffs[k][d][:] = E1s[k].fields[d].coeffs[:]   # patch k, component d
        E1.coeffs.update_ghost_regions()

        return E1


class Multipatch_Projector_H1:
    """
    to apply the H1 projection on every patch
    """
    def __init__(self, V0hs, V0h):

        self._P0s = [Projector_H1(V) for V in V0hs]
        self._V0h  = V0h   # multipatch Fem Space

    #--------------------------------------------------------------------------
    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        u0s = [P(fun) for P, fun, in zip(self._P0s, funs_log)]

        u0 = VectorFemField(self._V0h)
        for u_patch, u_c in zip(u0s, u0.coeffs):
            u_c[:] = u_patch.coeffs[:]
        u0.coeffs.update_ghost_regions()

        return u0


def get_scalar_patch_fields(u, V0hs):
    return [FemField(V, coeffs=c) for V, c in zip(V0hs, u.coeffs)]

def get_vector_patch_fields(E, V1hs):
    # Es = [VectorFemField(V, c) for V, c in zip(V1hs, E.coeffs)]  doesn't work because vector_space doesn't match
    Es = [VectorFemField(V) for V in V1hs]
    for E_patch, E_c in zip(Es, E.coeffs):
        for d in [0,1]:
            E_patch.fields[d].coeffs[:] = E_c[d][:]
    return Es


