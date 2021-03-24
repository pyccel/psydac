# coding: utf-8

# Conga operators on piecewise (broken) de Rham sequences

from mpi4py import MPI

import numpy as np
from sympde.topology import Boundary, Interface
from sympde.topology import element_of, elements_of
from sympde.calculus import grad, dot, inner, rot, div
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.api.discretization import discretize
from psydac.api.essential_bc import apply_essential_bc_stencil
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix
from psydac.linalg.iterative_solvers import cg, pcg
from psydac.fem.basic   import FemField

from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl, Projector_L2
from psydac.feec.derivatives import Gradient_2D, ScalarCurl_2D

from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator


def get_patch_index_from_face(domain, face):
    domains = domain.interior.args
    if isinstance(face, Interface):
        raise NotImplementedError("This face is an interface, it has several indices -- I am a machine, I cannot choose. Help.")
    elif isinstance(face, Boundary):
        i = domains.index(face.domain)
    else:
        i = domains.index(face)
    return i


#===============================================================================
class ConformingProjection_V0( FemLinearOperator ):
    """
    Conforming projection from global broken space to conforming global space
    Defined by averaging of interface dofs
    """
    # todo (MCP, 16.03.2021):
    #   - extend to several interfaces
    #   - avoid discretizing a bilinear form
    #   - allow case without interfaces (single or multipatch)
    def __init__(self, V0h, domain_h, hom_bc=False):

        FemLinearOperator.__init__(self, fem_domain=V0h)

        V0             = V0h.symbolic_space
        domain         = V0.domain

        u, v = elements_of(V0, names='u, v')
        expr   = u*v  # dot(u,v)

        Interfaces  = domain.interfaces  # note: interfaces does not include the boundary
        expr_I = ( plus(u)-minus(u) )*( plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(Interfaces, expr_I))


        ah = discretize(a, domain_h, [V0h, V0h])

        self._A = ah.assemble()

        spaces = self._A.domain.spaces

        if isinstance(Interfaces, Interface):
            Interfaces = (Interfaces, )

        for b1 in self._A.blocks:
            for A in b1:
                if A is None:continue
                A[:,:,:,:] = 0

        indices = [slice(None,None)]*domain.dim + [0]*domain.dim

        for i in range(len(self._A.blocks)):
            self._A[i,i][tuple(indices)]  = 1

        for I in Interfaces:

            axis = I.axis
            i_minus = get_patch_index_from_face(domain, I.minus)
            i_plus  = get_patch_index_from_face(domain, I.plus )

            sp_minus = spaces[i_minus]
            sp_plus  = spaces[i_plus]

            s_minus = sp_minus.starts[axis]
            e_minus = sp_minus.ends[axis]

            s_plus = sp_plus.starts[axis]
            e_plus = sp_plus.ends[axis]

            d_minus = V0h.spaces[i_minus].degree[axis]
            d_plus  = V0h.spaces[i_plus].degree[axis]

            indices = [slice(None,None)]*domain.dim + [0]*domain.dim

            minus_ext = I.minus.ext
            plus_ext = I.plus.ext

            if minus_ext == 1:
                indices[axis] = e_minus
            else:
                indices[axis] = s_minus
            self._A[i_minus,i_minus][tuple(indices)] = 1/2

            if plus_ext == 1:
                indices[axis] = e_plus
            else:
                indices[axis] = s_plus

            self._A[i_plus,i_plus][tuple(indices)] = 1/2

            if plus_ext != minus_ext:

                if minus_ext == 1:
                    indices[axis] = d_minus
                else:
                    indices[axis] = s_minus

                if plus_ext == 1:
                    indices[domain.dim + axis] = d_plus
                else:
                    indices[domain.dim + axis] = -d_plus

                self._A[i_minus,i_plus][tuple(indices)] = 1/2

                if plus_ext == 1:
                    indices[axis] = d_plus
                else:
                    indices[axis] = s_plus

                if minus_ext == 1:
                    indices[domain.dim + axis] = d_minus
                else:
                    indices[domain.dim + axis] = -d_minus

                self._A[i_plus,i_minus][tuple(indices)] = 1/2
            else:

                if minus_ext == 1:
                    indices[axis] = d_minus
                else:
                    indices[axis] = s_minus

                indices[domain.dim + axis] = 0

                self._A[i_minus,i_plus][tuple(indices)] = 1/2

                if plus_ext == 1:
                    indices[axis] = d_plus
                else:
                    indices[axis] = s_plus

                self._A[i_plus,i_minus][tuple(indices)] = 1/2

        if hom_bc:
            for bn in domain.boundary:
                i = get_patch_index_from_face(domain, bn)
                for j in range(len(domain)):
                    if self._A[i,j] is None:continue
                    apply_essential_bc_stencil(self._A[i,j], axis=bn.axis, ext=bn.ext, order=0)

        self._matrix = self._A
#===============================================================================
class ConformingProjection_V1( FemLinearOperator ):
    """
    Conforming projection from global broken space to conforming global space

    proj.dot(v) returns the conforming projection of v, computed by solving linear system

    """
    # todo (MCP, 16.03.2021):
    #   - extend to several interfaces
    #   - avoid discretizing a bilinear form
    #   - allow case without interfaces (single or multipatch)
    def __init__(self, V1h, domain_h, hom_bc=False):

        FemLinearOperator.__init__(self, fem_domain=V1h)

        V1             = V1h.symbolic_space
        domain         = V1.domain

        u, v = elements_of(V1, names='u, v')
        expr   = dot(u,v)

        Interfaces      = domain.interfaces  # note: interfaces does not include the boundary
        expr_I = dot( plus(u)-minus(u) , plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(Interfaces, expr_I))

        ah = discretize(a, domain_h, [V1h, V1h])

        self._A = ah.assemble()

        for b1 in self._A.blocks:
            for b2 in b1:
                if b2 is None:continue
                for b3 in b2.blocks:
                    for A in b3:
                        if A is None:continue
                        A[:,:,:,:] = 0

        spaces = self._A.domain.spaces

        if isinstance(Interfaces, Interface):
            Interfaces = (Interfaces, )

        indices = [slice(None,None)]*domain.dim + [0]*domain.dim

        for i in range(len(self._A.blocks)):
            self._A[i,i][0,0][tuple(indices)]  = 1
            self._A[i,i][1,1][tuple(indices)]  = 1

        # empty list if no interfaces ?
        if Interfaces is not None:

            for I in Interfaces:

                i_minus = get_patch_index_from_face(domain, I.minus)
                i_plus  = get_patch_index_from_face(domain, I.plus )

                indices = [slice(None,None)]*domain.dim + [0]*domain.dim

                sp1    = spaces[i_minus]
                sp2    = spaces[i_plus]

                s11 = sp1.spaces[0].starts[I.axis]
                e11 = sp1.spaces[0].ends[I.axis]
                s12 = sp1.spaces[1].starts[I.axis]
                e12 = sp1.spaces[1].ends[I.axis]

                s21 = sp2.spaces[0].starts[I.axis]
                e21 = sp2.spaces[0].ends[I.axis]
                s22 = sp2.spaces[1].starts[I.axis]
                e22 = sp2.spaces[1].ends[I.axis]

                d11     = V1h.spaces[i_minus].spaces[0].degree[I.axis]
                d12     = V1h.spaces[i_minus].spaces[1].degree[I.axis]

                d21     = V1h.spaces[i_plus].spaces[0].degree[I.axis]
                d22     = V1h.spaces[i_plus].spaces[1].degree[I.axis]

                s_minus = [s11, s12]
                e_minus = [e11, e12]

                s_plus = [s21, s22]
                e_plus = [e21, e22]

                d_minus = [d11, d12]
                d_plus  = [d21, d22]

                for k in range(domain.dim):
                    if k == I.axis:continue

                    indices[I.axis] = e_minus[k]
                    self._A[i_minus,i_minus][k,k][tuple(indices)] = 1/2

                    indices[I.axis] = s_plus[k]
                    self._A[i_plus,i_plus][k,k][tuple(indices)] = 1/2

                    indices[I.axis] = d_minus[k]
                    indices[domain.dim + I.axis] = -d_plus[k]
                    self._A[i_minus,i_plus][k,k][tuple(indices)] = 1/2

                    indices[I.axis] = s_plus[k]
                    indices[domain.dim + I.axis] = d_minus[k]

                    self._A[i_plus,i_minus][k,k][tuple(indices)] = 1/2

        if hom_bc:
            for bn in domain.boundary:
                i = get_patch_index_from_face(domain, bn)
                for j in range(len(domain)):
                    if self._A[i,j] is None:continue
                    apply_essential_bc_stencil(self._A[i,j][1-bn.axis,1-bn.axis], axis=bn.axis, ext=bn.ext, order=0)

        self._matrix = self._A
        # exit()

#===============================================================================
class BrokenMass( FemLinearOperator ):
    """
    Broken mass matrix for a scalar space (seen as a LinearOperator... to be improved)
    # TODO: (MCP 10.03.2021) define them as Hodge FemLinearOperators
    # TODO: (MCP 16.03.2021) define also the inverse Hodge

    """
    def __init__( self, Vh, domain_h, is_scalar):

        FemLinearOperator.__init__(self, fem_domain=Vh)

        V = Vh.symbolic_space
        domain = V.domain
        # domain_h = V0h.domain  # would be nice
        u, v = elements_of(V, names='u, v')
        if is_scalar:
            expr   = u*v
        else:
            expr   = dot(u,v)
        a = BilinearForm((u,v), integral(domain, expr))
        ah = discretize(a, domain_h, [Vh, Vh])
        self._matrix = ah.assemble() #.toarray()


#==============================================================================
class BrokenGradient_2D(FemLinearOperator):

    def __init__(self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V0h, fem_codomain=V1h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix for i, D0i in enumerate(D0s)})

    def transpose(self):
        # todo (MCP): define as the dual differential operator
        return BrokenTransposedGradient_2D(self.fem_domain, self.fem_codomain)

#==============================================================================
class BrokenTransposedGradient_2D( FemLinearOperator ):

    def __init__( self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V0h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D0i._matrix.T for i, D0i in enumerate(D0s)})

    def transpose(self):
        # todo (MCP): discard
        return BrokenGradient_2D(self.fem_codomain, self.fem_domain)


#==============================================================================
class BrokenScalarCurl_2D(FemLinearOperator):
    def __init__(self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V2h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D1i._matrix for i, D1i in enumerate(D1s)})

    def transpose(self):
        return BrokenTransposedScalarCurl_2D(V1h=self.fem_domain, V2h=self.fem_codomain)


#==============================================================================
class BrokenTransposedScalarCurl_2D( FemLinearOperator ):

    def __init__( self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V2h, fem_codomain=V1h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._matrix = BlockMatrix(self.domain, self.codomain, \
                blocks={(i, i): D1i._matrix.T for i, D1i in enumerate(D1s)})

    def transpose(self):
        return BrokenScalarCurl_2D(V1h=self.fem_codomain, V2h=self.fem_domain)



#==============================================================================
from sympy import Tuple

# def multipatch_Moments_Hcurl(f, V1h, domain_h):
def ortho_proj_Hcurl(EE, V1h, domain_h, M1):
    """
    return orthogonal projection of E on V1h, given M1 the mass matrix
    """
    assert isinstance(EE, Tuple)
    V1 = V1h.symbolic_space
    v = element_of(V1, name='v')
    l = LinearForm(v, integral(V1.domain, dot(v,EE)))
    lh = discretize(l, domain_h, V1h)
    b = lh.assemble()
    sol_coeffs, info = pcg(M1.mat(), b, pc="jacobi", tol=1e-10)

    return FemField(V1h, coeffs=sol_coeffs)

#==============================================================================
class Multipatch_Projector_H1:
    """
    to apply the H1 projection (2D) on every patch
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
class Multipatch_Projector_Hcurl:

    """
    to apply the Hcurl projection (2D) on every patch
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
class Multipatch_Projector_L2:

    """
    to apply the L2 projection (2D) on every patch
    """
    def __init__(self, V2h, nquads=None):

        self._P2s = [Projector_L2(V, nquads=nquads) for V in V2h.spaces]
        self._V2h  = V2h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        B2s = [P(fun) for P, fun, in zip(self._P2s, funs_log)]

        B2_coeffs = BlockVector(self._V2h.vector_space, \
                blocks = [B2j.coeffs for B2j in B2s])

        return FemField(self._V2h, coeffs = B2_coeffs)
