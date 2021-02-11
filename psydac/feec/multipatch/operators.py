
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
class ConformingProjection( LinearOperator ):
    """
    Conforming projection from global broken space to conforming global space

    proj.dot(v) returns the conforming projection of v, computed by solving linear system


    """
    def __init__( self, V0h_1, V0h_2, domain_h_1, domain_h_2, V0h, domain_h):

        V0 = V0h.symbolic_space
        domain = V0.domain
        # domain_h = V0h.domain  # would be nice

        self._domain   = V0h
        self._codomain = V0h

        u, v = elements_of(V0, names='u, v')
        expr   = u*v  # dot(u,v)

        kappa  = 10**6
        I = domain.interfaces  # note: interfaces does not include the boundary
        expr_I = kappa*( plus(u)-minus(u) )*( plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(I, expr_I))  # + integral(domain.boundary, u*v))

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

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain


    def __call__( self, f ):

        f1,f2 = f.fields

        b1 = self._lh_1.assemble(f1=f1)
        b2 = self._lh_2.assemble(f2=f2)
        b  = BlockVector(self.codomain.vector_space, blocks=[b1, b2])

        # sol_coeffs, info = cg( self._A, b, tol=1e-13, verbose=True )
        sol_coeffs, info = pcg( self._A, b, pc="jacobi", tol=1e-6, verbose=True )
        #
        # sol_coeffs = self._solver.solve( b )

        return  VectorFemField(self.codomain, coeffs=sol_coeffs)

    def dot( self, f_coeffs, out=None ):

        f = VectorFemField(self.domain, coeffs=f_coeffs)

        return self(f).coeffs


#===============================================================================
class BrokenMass_V0( LinearOperator ):
    """
    Broken mass matrix, seen as a LinearOperator
    """
    def __init__( self, V0h, domain_h):

        V0 = V0h.symbolic_space
        domain = V0.domain
        # domain_h = V0h.domain  # would be nice
        self._domain   = V0h
        self._codomain = V0h
        u, v = elements_of(V0, names='u, v')
        expr   = u*v  # dot(u,v)
        a = BilinearForm((u,v), integral(domain, expr))
        ah = discretize(a, domain_h, [V0h, V0h])
        self._M = ah.assemble() #.toarray()

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    def __call__( self, f ):
        Mf_coeffs = self.dot(f.coeffs)
        return VectorFemField(self.domain, coeffs=Mf_coeffs)

    def dot( self, f_coeffs, out=None ):
        return self._M.dot(f_coeffs)


#===============================================================================
class BrokenMass_V1( LinearOperator ):
    """
    Broken mass matrix in V1, seen as a LinearOperator
    """
    def __init__( self, V1h, domain_h):

        V1 = V1h.symbolic_space
        domain = V1.domain
        # domain_h = V0h.domain  # would be nice
        self._domain   = V1h
        self._codomain = V1h
        u, v = elements_of(V1, names='u, v')
        expr   = dot(u,v)
        a = BilinearForm((u,v), integral(domain, expr))
        ah = discretize(a, domain_h, [V1h, V1h])
        self._M = ah.assemble() #.toarray()

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    def __call__( self, f ):
        Mf_coeffs = self.dot(f.coeffs)
        return VectorFemField(self.domain, coeffs=Mf_coeffs)

    def dot( self, f_coeffs, out=None ):
        return self._M.dot(f_coeffs)


class ComposedLinearOperator( LinearOperator ):

    def __init__( self, B, A ):

        assert isinstance(A, LinearOperator)
        assert isinstance(B, LinearOperator)
        assert B.domain == A.codomain

        self._domain   = A.domain
        self._codomain = B.codomain

        self._A = A
        self._B = B

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    def __call__( self, f ):

        return  self._B(self._A(f))

    def dot( self, f_coeffs, out=None ):

        return  self._B.dot(self._A.dot(f_coeffs))

class IdLinearOperator( LinearOperator ):

    def __init__( self, V ):
        self._domain   = V
        self._codomain = V

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    def __call__( self, f ):
        return f

    def dot( self, f_coeffs, out=None ):
        return f_coeffs


class SumLinearOperator( LinearOperator ):

    def __init__( self, B, A ):
        assert isinstance(A, LinearOperator)
        assert isinstance(B, LinearOperator)
        assert B.domain == A.domain
        assert B.codomain == A.codomain
        self._domain   = A.domain
        self._codomain = A.codomain
        self._A = A
        self._B = B

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    def __call__( self, f ):
        return  self._B(f) + self._A(f)

    def dot( self, f_coeffs, out=None ):
        return  self._B.dot(f_coeffs) + self._A.dot(f_coeffs)

class MultLinearOperator( LinearOperator ):

    def __init__( self, c, A ):

        assert isinstance(A, LinearOperator)
        self._domain   = A.domain
        self._codomain = A.codomain

        self._A = A
        self._c = c

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    def __call__( self, f ):

        return  self._c * self._A(f)

    def dot( self, f_coeffs, out=None ):

        return  self._c * self._A.dot(f_coeffs)



class BrokenGradient_2D( LinearOperator ):

    def __init__( self, V0hs, V1hs, V0h, V1h, as_mat=False, transpose=False ):

        # V0hs is the list of single patch spaces
        # V0h is the multipatch space
        # same for V1
        # todo: provide only V0h and V1h (remove redundancy)

        self._V0hs = V0hs
        self._V1hs = V1hs

        self._npatches = len(V0hs)
        assert self._npatches == len(V1hs)

        self._as_mat = as_mat

        self._D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0hs, V1hs)]
        if as_mat and transpose:
            self._mats = [D0._matrix.T for D0 in self._D0s]
            self._domain = V1h
            self._codomain = V0h

        else:
            self._mats = [D0._matrix for D0 in self._D0s]
            self._domain = V0h
            self._codomain = V1h

        assert self._npatches == 2
        # self.mat = BlockMatrix(V0h.vector_space, V1h.vector_space, blocks=[[self._mats[0], None],[None, self._mats[1]]])

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain

    def __call__( self, u0 ):
        # u0 should be a multipatch V0 field
        V1h = self._codomain
        grad_u0 = VectorFemField(V1h)

        if self._as_mat:
            grad_u0_cs = [mat.dot(c) for mat, c in zip(self._mats, u0.coeffs)]

            # print("type(grad_u0_cs[0]) = ", type(grad_u0_cs[0]))
            # print("type(grad_u0_cs[0]._blocks[0]) = ", type(grad_u0_cs[0]._blocks[0]))
            # print("type(grad_u0.coeffs[0]) = ", type(grad_u0.coeffs[0]))
            for k in range(self._npatches):
                for b1, b2 in zip(grad_u0.coeffs[k]._blocks, grad_u0_cs[k]._blocks):
                    # for d in range(2):
                    b1[:] = b2[:]
                # grad_u0.coeffs[k][:] = grad_u0_cs[k]._blocks
                # w = BlockVector( self._space, [b.copy() for b in self._blocks] )
                # grad_u0.coeffs[k] = grad_u0_cs[k].copy()

        else:
            # creating a list of single patch fields with proper spaces and copying the coefficients,
            # because the TensorFemSpace class does not have an __eq__ magic method
            u0s = get_scalar_patch_fields(u0, self._V0hs)
            grad_u0s = [D0(u) for D0, u in zip(self._D0s, u0s)]

            for k in range(self._npatches):
                for d in range(2):
                    grad_u0.coeffs[k][d][:] = grad_u0s[k].fields[d].coeffs[:]      # patch k, component d

        grad_u0.coeffs.update_ghost_regions()
        return grad_u0

    def dot( self, u_coeffs, out=None ):

        # if self._as_mat:
        #     f_coeffs = self._mat.dot(u_coeffs)
        # else:
        f = VectorFemField(self.domain, coeffs=u_coeffs)
        f_coeffs = self(f).coeffs

        return f_coeffs

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





#
# #====================================================================================================
# class BrokenGradient_2D(DiffOperator):
#     """
#     Broken Gradient operator in 2D multipatch domains.
#
#     Parameters
#     ----------
#     H1 : 2D TensorFemSpace
#         Domain of gradient operator.
#
#     Hcurl : 2D ProductFemSpace
#         Codomain of gradient operator.
#
#     """
#     def __init__(self, H1, Hcurl):
#
#         # assert isinstance(   H1,  TensorFemSpace); assert    H1.ldim == 2
#         # assert isinstance(Hcurl, ProductFemSpace); assert Hcurl.ldim == 2
#         #
#         # assert Hcurl.spaces[0].periodic == H1.periodic
#         # assert Hcurl.spaces[1].periodic == H1.periodic
#         #
#         # assert tuple(Hcurl.spaces[0].degree) == (H1.degree[0]-1, H1.degree[1]  )
#         # assert tuple(Hcurl.spaces[1].degree) == (H1.degree[0]  , H1.degree[1]-1)
#
#         # 1D differentiation matrices (from B-spline of degree p to M-spline of degree p-1)
#         Dx, Dy = [d_matrix(N.nbasis, N.degree, N.periodic) for N in H1.spaces]
#
#         # 1D identity matrices for B-spline coefficients
#         Ix, Iy = [IdentityMatrix(N.vector_space) for N in H1.spaces]
#
#         # Tensor-product spaces of coefficients - domain
#         B_B = H1.vector_space
#
#         # Tensor-product spaces of coefficients - codomain
#         (M_B, B_M) = Hcurl.vector_space.spaces
#
#         # Build Gradient matrix block by block
#         blocks = [[KroneckerStencilMatrix(B_B, M_B, *(Dx, Iy))],
#                   [KroneckerStencilMatrix(B_B, B_M, *(Ix, Dy))]]
#         matrix = BlockMatrix(BlockVectorSpace(H1.vector_space), Hcurl.vector_space, blocks=blocks)
#
#         # Store data in object
#         self._domain   = H1
#         self._codomain = Hcurl
#         self._matrix   = block_tostencil(matrix)
#
#     def __call__(self, u):
#
#         assert isinstance(u, FemField)
#         assert u.space == self.domain
#
#         coeffs = self.matrix.dot(u.coeffs)
#         coeffs.update_ghost_regions()
#
#         return VectorFemField(self.codomain, coeffs=coeffs)
#
