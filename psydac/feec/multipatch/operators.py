# coding: utf-8

# Conga operators on piecewise (broken) de Rham sequences

from mpi4py import MPI

import numpy as np
from sympde.topology  import Boundary, Interface, Union
from sympde.topology  import element_of, elements_of
from sympde.calculus  import grad, dot, inner, rot, div
from sympde.calculus  import laplace, bracket, convect
from sympde.calculus  import jump, avg, Dn, minus, plus
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.api.discretization       import discretize
from psydac.api.essential_bc         import apply_essential_bc_stencil
from psydac.linalg.block             import BlockVectorSpace, BlockVector, BlockMatrix
from psydac.linalg.stencil           import StencilVector, StencilMatrix, StencilInterfaceMatrix
from psydac.linalg.iterative_solvers import cg, pcg
from psydac.fem.basic                import FemField

from psydac.feec.global_projectors               import Projector_H1, Projector_Hcurl, Projector_L2
from psydac.feec.derivatives                     import Gradient_2D, ScalarCurl_2D
from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator

def get_patch_index_from_face(domain, face):
    if domain.mapping:
        domain = domain.logical_domain
    if face.mapping:
        face = face.logical_domain

    domains = domain.interior.args
    if isinstance(face, Interface):
        raise NotImplementedError("This face is an interface, it has several indices -- I am a machine, I cannot choose. Help.")
    elif isinstance(face, Boundary):
        i = domains.index(face.domain)
    else:
        i = domains.index(face)
    return i

def get_interface_from_corners(corner1, corner2, domain):
    interface = []
    interfaces = domain.interfaces

    if not isinstance(interfaces, Union):
        interfaces = (interfaces,)

    for i in interfaces:
        if i.plus.domain in [corner1.domain, corner2.domain]:
            if i.minus.domain in [corner1.domain, corner2.domain]:
                interface.append(i)

    bd1 = corner1.boundaries
    bd2 = corner2.boundaries

    new_interface = []

    for i in interface:
        if i.minus in bd1+bd2:
            if i.plus in bd2+bd1:
                new_interface.append(i)

    if len(new_interface) == 1:
        return new_interface[0]
    if len(new_interface)>1:
        raise ValueError('found more than one interface for the corners {} and {}'.format(corner1, corner2))
    return None


def get_row_col_index(corner1, corner2, interface, axis, V1, V2):
    start = V1.vector_space.starts
    end   = V1.vector_space.ends
    degree = V2.degree
    start_end = (start, end)

    row    = [None]*len(start)
    col    = [0]*len(start)

    assert corner1.boundaries[0].axis == corner2.boundaries[0].axis

    for bd in corner1.boundaries:
        row[bd.axis] = start_end[(bd.ext+1)//2][bd.axis]

    if interface is None and corner1.domain != corner2.domain:
        bd = [i for i in corner1.boundaries if i.axis==axis][0]
        if bd.ext == 1:row[bd.axis] = degree[bd.axis]

    if interface is None:
        return row+col

    axis = interface.axis

    if interface.minus.domain == corner1.domain:
        if interface.minus.ext == -1:row[axis] = 0
        else:row[axis] = degree[axis]
    else:
        if interface.plus.ext == -1:row[axis] = 0
        else:row[axis] = degree[axis]

    if interface.minus.ext == interface.plus.ext:
        pass
    elif interface.minus.domain == corner1.domain:
        if interface.minus.ext == -1:
            col[axis] =  degree[axis]
        else:
            col[axis] =  -degree[axis]
    else:
        if interface.plus.ext == -1:
            col[axis] =  degree[axis]
        else:
            col[axis] =  -degree[axis]

    return row+col

#===============================================================================
def allocate_matrix(corners, test_space, trial_space):

    bi, bj = list(zip(*corners))
    permutation = np.arange(bi[0].domain.dim)

    flips = []
    k = 0
    while k<len(bi):
        c1 = np.array(bi[k].coordinates)
        c2 = np.array(bj[k].coordinates)[permutation]
        flips.append(np.array([-1 if d1!=d2 else 1 for d1,d2 in zip(c1, c2)]))

        if np.sum(abs(flips[0]-flips[-1])) != 0:
            prod = [f1*f2 for f1,f2 in zip(flips[0], flips[-1])]
            while -1 in prod:
                i1 = prod.index(-1)
                if -1 in prod[i1+1:]:
                    i2 = i1+1 + prod[i1+1:].index(-1)
                    prod = prod[i2+1:]
                    permutation[i1], permutation[i2] = permutation[i2], permutation[i1]
                    k = -1
                    flips = []
                else:
                    break

        k +=1

    assert all(abs(flips[0] - i).sum()==0 for i in flips)
    cs    = list(zip(*[i.coordinates for i in bi]))
    axis  = [all(i[0]==j for j in i) for i in cs].index(True)
    ext   = 1 if cs[axis][0]==1 else -1
    s     = test_space.quad_grids[axis].spans[-1 if ext==1 else 0] - test_space.degree[axis]

    mat  = StencilInterfaceMatrix(trial_space.vector_space, test_space.vector_space, s, s, axis, flip=flips[0], permutation=list(permutation))
    return mat

#===============================================================================
class ConformingProjection_V0( FemLinearOperator ):
    """
    Conforming projection from global broken space to conforming global space
    Defined by averaging of interface dofs
    """
    # todo (MCP, 16.03.2021):
    #   - avoid discretizing a bilinear form
    #   - allow case without interfaces (single or multipatch)
    def __init__(self, V0h, domain_h, hom_bc=False):

        FemLinearOperator.__init__(self, fem_domain=V0h)

        V0                      = V0h.symbolic_space
        domain                  = V0.domain
        self.symbolic_domain    = domain

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

            if plus_ext == minus_ext:
                if minus_ext == 1:
                    indices[axis] = d_minus
                else:
                    indices[axis] = s_minus

                self._A[i_minus,i_plus][tuple(indices)] = 1/2

                if plus_ext == 1:
                    indices[axis] = d_plus
                else:
                    indices[axis] = s_plus

                self._A[i_plus,i_minus][tuple(indices)] = 1/2

            else:
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

        domain = domain.logical_domain
        corner_blocks = {}
        for c in domain.corners:
            for b1 in c.corners:
                i = get_patch_index_from_face(domain, b1.domain)
                for b2 in c.corners:
                    j = get_patch_index_from_face(domain, b2.domain)
                    if (i,j) in corner_blocks:
                        corner_blocks[i,j] += [(b1, b2)]
                    else:
                        corner_blocks[i,j] = [(b1, b2)]

        for c in domain.corners:
            if len(c) == 2:continue
            for b1 in c.corners:
                i = get_patch_index_from_face(domain, b1.domain)
                for b2 in c.corners:
                    j = get_patch_index_from_face(domain, b2.domain)
                    interface = get_interface_from_corners(b1, b2, domain)
                    axis = None
                    if self._A[i,j] is None:
                        self._A[i,j] = allocate_matrix(corner_blocks[i,j], V0h.spaces[i], V0h.spaces[j])

                    if i!=j and self._A[i,j]:axis=self._A[i,j]._dim
                    index = get_row_col_index(b1, b2, interface, axis, V0h.spaces[i], V0h.spaces[j])
                    self._A[i,j][tuple(index)] = 1/len(c)

        self._matrix = self._A

        if hom_bc:
            for bn in domain.boundary:
                self.set_homogenous_bc(bn)

    def set_homogenous_bc(self, boundary, rhs=None):
        domain = self.symbolic_domain
        Vh = self.fem_domain
        if domain.mapping:
            domain = domain.logical_domain
        if boundary.mapping:
            boundary = boundary.logical_domain

        corners = domain.corners
        i = get_patch_index_from_face(domain, boundary)
        if rhs:
            apply_essential_bc_stencil(rhs[i], axis=boundary.axis, ext=boundary.ext, order=0)
        for j in range(len(domain)):
            if self._A[i,j] is None:continue
            apply_essential_bc_stencil(self._A[i,j], axis=boundary.axis, ext=boundary.ext, order=0)

        for c in corners:
            faces = [f for b in c.corners for f in b.boundaries]
            if len(c) == 2:continue
            if boundary in faces:
                for b1 in c.corners:
                    i = get_patch_index_from_face(domain, b1.domain)
                    for b2 in c.corners:
                        j = get_patch_index_from_face(domain, b2.domain)
                        interface = get_interface_from_corners(b1, b2, domain)
                        axis = None
                        if i!=j:axis = self._A[i,j].dim
                        index = get_row_col_index(b1, b2, interface, axis, Vh.spaces[i], Vh.spaces[j])
                        self._A[i,j][tuple(index)] = 0.

                        if i==j and rhs:rhs[i][tuple(index[:2])] = 0.

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
        self.symbolic_domain  = domain

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

                minus_ext = I.minus.ext
                plus_ext = I.plus.ext

                axis = I.axis
                for k in range(domain.dim):
                    if k == I.axis:continue

                    if minus_ext == 1:
                        indices[axis] = e_minus[k]
                    else:
                        indices[axis] = s_minus[k]
                    self._A[i_minus,i_minus][k,k][tuple(indices)] = 1/2

                    if plus_ext == 1:
                        indices[axis] = e_plus[k]
                    else:
                        indices[axis] = s_plus[k]

                    self._A[i_plus,i_plus][k,k][tuple(indices)] = 1/2

                    if plus_ext == minus_ext:
                        if minus_ext == 1:
                            indices[axis] = d_minus[k]
                        else:
                            indices[axis] = s_minus[k]

                        self._A[i_minus,i_plus][k,k][tuple(indices)] = 1/2*I.direction

                        if plus_ext == 1:
                            indices[axis] = d_plus[k]
                        else:
                            indices[axis] = s_plus[k]

                        self._A[i_plus,i_minus][k,k][tuple(indices)] = 1/2*I.direction

                    else:
                        if minus_ext == 1:
                            indices[axis] = d_minus[k]
                        else:
                            indices[axis] = s_minus[k]

                        if plus_ext == 1:
                            indices[domain.dim + axis] = d_plus[k]
                        else:
                            indices[domain.dim + axis] = -d_plus[k]

                        self._A[i_minus,i_plus][k,k][tuple(indices)] = 1/2*I.direction

                        if plus_ext == 1:
                            indices[axis] = d_plus[k]
                        else:
                            indices[axis] = s_plus[k]

                        if minus_ext == 1:
                            indices[domain.dim + axis] = d_minus[k]
                        else:
                            indices[domain.dim + axis] = -d_minus[k]

                        self._A[i_plus,i_minus][k,k][tuple(indices)] = 1/2*I.direction


        if hom_bc:
            for bn in domain.boundary:
                self.set_homogenous_bc(bn)

        self._matrix = self._A

    def set_homogenous_bc(self, boundary):
        domain = self.symbolic_domain
        Vh = self.fem_domain

        i = get_patch_index_from_face(domain, boundary)
        axis = boundary.axis
        ext  = boundary.ext
        for j in range(len(domain)):
            if self._A[i,j] is None:continue
            apply_essential_bc_stencil(self._A[i,j][1-axis,1-axis], axis=axis, ext=ext, order=0)

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
