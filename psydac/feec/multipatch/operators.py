# coding: utf-8

# Conga operators on piecewise (broken) de Rham sequences

from sympy import Tuple
from mpi4py import MPI
import os
import numpy as np

from scipy.sparse import save_npz, load_npz
from scipy.sparse import kron, block_diag
from scipy.sparse.linalg import inv

from sympde.topology import Boundary, Interface, Union
from sympde.topology import element_of, elements_of
from sympde.topology.space import ScalarFunction
from sympde.calculus import grad, dot, inner, rot, div
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.core.bsplines import collocation_matrix, histopolation_matrix

from psydac.api.discretization import discretize
from psydac.api.essential_bc import apply_essential_bc_stencil
from psydac.api.settings import PSYDAC_BACKENDS
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockLinearOperator
from psydac.linalg.stencil import StencilVector, StencilMatrix, StencilInterfaceMatrix
from psydac.linalg.solvers import inverse
from psydac.fem.basic import FemField


from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl, Projector_L2
from psydac.feec.derivatives import Gradient_2D, ScalarCurl_2D
from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator


def get_patch_index_from_face(domain, face):
    """ Return the patch index of subdomain/boundary

    Parameters
    ----------
    domain : <Sympde.topology.Domain>
     The Symbolic domain

    face : <Sympde.topology.BasicDomain>
     A patch or a boundary of a patch

    Returns
    -------
    i : <int>
     The index of a subdomain/boundary in the multipatch domain
    """

    if domain.mapping:
        domain = domain.logical_domain
    if face.mapping:
        face = face.logical_domain

    domains = domain.interior.args
    if isinstance(face, Interface):
        raise NotImplementedError(
            "This face is an interface, it has several indices -- I am a machine, I cannot choose. Help.")
    elif isinstance(face, Boundary):
        i = domains.index(face.domain)
    else:
        i = domains.index(face)
    return i


def get_interface_from_corners(corner1, corner2, domain):
    """ Return the interface between two corners from two different patches that correspond to a single (physical) vertex.

    Parameters
    ----------
    corner1 : <Sympde.topology.Corner>
     The first corner of the 2D interface

    corner2 : <Sympde.topology.Corner>
     The second corner of the 2D interface

    domain : <Sympde.topology.Domain>
     The Symbolic domain

    Returns
    -------
    interface: <Sympde.topology.Interface|None>
     The interface between two vertices

    """

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
        if i.minus in bd1 + bd2:
            if i.plus in bd2 + bd1:
                new_interface.append(i)

    if len(new_interface) == 1:
        return new_interface[0]
    if len(new_interface) > 1:
        raise ValueError(
            'found more than one interface for the corners {} and {}'.format(
                corner1, corner2))
    return None


def get_row_col_index(corner1, corner2, interface, axis, V1, V2):
    """ Return the row and column index of a corner in the StencilInterfaceMatrix
        for dofs of H1 type spaces

    Parameters
    ----------
    corner1 : <Sympde.topology.Corner>
     The first corner of the 2D interface

    corner2 : <Sympde.topology.Corner>
     The second corner of the 2D interface

    interface : <Sympde.topology.Interface|None>
     The interface between the two corners

    axis    : <int|None>
     Axis of the interface

    V1      : <FemSpace>
     Test Space

    V2      : <FemSpace>
     Trial Space

    Returns
    -------
    index: <list>
     The StencilInterfaceMatrix index of the corner, it has the form (i1, i2, k1, k2) in 2D,
     where (i1, i2) identifies the row and (k1, k2) the diagonal.
    """
    start = V1.vector_space.starts
    end = V1.vector_space.ends
    degree = V2.degree
    start_end = (start, end)

    row = [None] * len(start)
    col = [0] * len(start)

    assert corner1.boundaries[0].axis == corner2.boundaries[0].axis

    for bd in corner1.boundaries:
        row[bd.axis] = start_end[(bd.ext + 1) // 2][bd.axis]

    if interface is None and corner1.domain != corner2.domain:
        bd = [i for i in corner1.boundaries if i.axis == axis][0]
        if bd.ext == 1:
            row[bd.axis] = degree[bd.axis]

    if interface is None:
        return row + col

    axis = interface.axis

    if interface.minus.domain == corner1.domain:
        if interface.minus.ext == -1:
            row[axis] = 0
        else:
            row[axis] = degree[axis]
    else:
        if interface.plus.ext == -1:
            row[axis] = 0
        else:
            row[axis] = degree[axis]

    if interface.minus.ext == interface.plus.ext:
        pass
    elif interface.minus.domain == corner1.domain:
        if interface.minus.ext == -1:
            col[axis] = degree[axis]
        else:
            col[axis] = -degree[axis]
    else:
        if interface.plus.ext == -1:
            col[axis] = degree[axis]
        else:
            col[axis] = -degree[axis]

    return row + col


# ===============================================================================
def allocate_interface_matrix(corners, test_space, trial_space):
    """ Allocate the interface matrix for a vertex shared by two patches

    Parameters
    ----------
    corners: <list>
     The patch corners corresponding to the common shared vertex

    test_space: <FemSpace>
     The test space

    trial_space: <FemSpace>
     The trial space

    Returns
    -------
    mat: <StencilInterfaceMatrix>
     The interface matrix shared by two patches
    """
    bi, bj = list(zip(*corners))
    permutation = np.arange(bi[0].domain.dim)

    flips = []
    k = 0
    while k < len(bi):
        c1 = np.array(bi[k].coordinates)
        c2 = np.array(bj[k].coordinates)[permutation]
        flips.append(
            np.array([-1 if d1 != d2 else 1 for d1, d2 in zip(c1, c2)]))

        if np.sum(abs(flips[0] - flips[-1])) != 0:
            prod = [f1 * f2 for f1, f2 in zip(flips[0], flips[-1])]
            while -1 in prod:
                i1 = prod.index(-1)
                if -1 in prod[i1 + 1:]:
                    i2 = i1 + 1 + prod[i1 + 1:].index(-1)
                    prod = prod[i2 + 1:]
                    permutation[i1], permutation[i2] = permutation[i2], permutation[i1]
                    k = -1
                    flips = []
                else:
                    break

        k += 1

    assert all(abs(flips[0] - i).sum() == 0 for i in flips)
    cs = list(zip(*[i.coordinates for i in bi]))
    axis = [all(i[0] == j for j in i) for i in cs].index(True)
    ext = 1 if cs[axis][0] == 1 else -1
    s = test_space.get_assembly_grids(
    )[axis].spans[-1 if ext == 1 else 0] - test_space.degree[axis]

    mat = StencilInterfaceMatrix(
        trial_space.vector_space,
        test_space.vector_space,
        s,
        s,
        axis,
        flip=flips[0],
        permutation=list(permutation))
    return mat

# ===============================================================================
# The following operators are not compatible with the changes in the Stencil format
# and their datatype does not allow for non-matching interfaces, but they might be
# useful for future implementations
# ===============================================================================


class ConformingProjection_V0(FemLinearOperator):
    """
    Conforming projection from global broken V0 space to conforming global V0 space
    Defined by averaging of interface dofs

    Parameters
    ----------
    V0h: <FemSpace>
     The discrete space

    domain_h: <Geometry>
     The discrete domain of the projector

    hom_bc : <bool>
     Apply homogenous boundary conditions if True

    backend_language: <str>
     The backend used to accelerate the code

    storage_fn:
     filename to store/load the operator sparse matrix
    """
    # todo (MCP, 16.03.2021):
    #   - avoid discretizing a bilinear form
    #   - allow case without interfaces (single or multipatch)

    def __init__(
            self,
            V0h,
            domain_h,
            hom_bc=False,
            backend_language='python',
            storage_fn=None):

        FemLinearOperator.__init__(self, fem_domain=V0h)

        V0 = V0h.symbolic_space
        domain = V0.domain
        self.symbolic_domain = domain

        if storage_fn and os.path.exists(storage_fn):
            print(
                "[ConformingProjection_V0] loading operator sparse matrix from " +
                storage_fn)
            self._sparse_matrix = load_npz(storage_fn)

        else:
            # assemble the operator matrix
            u, v = elements_of(V0, names='u, v')
            expr = u * v  # dot(u,v)

            Interfaces = domain.interfaces  # note: interfaces does not include the boundary
            # this penalization is for an H1-conforming space
            expr_I = (plus(u) - minus(u)) * (plus(v) - minus(v))

            a = BilinearForm((u, v), integral(domain, expr) +
                             integral(Interfaces, expr_I))
            # print('[[ forcing python backend for ConformingProjection_V0]] ')
            # backend_language = 'python'
            ah = discretize(
                a, domain_h, [
                    V0h, V0h], backend=PSYDAC_BACKENDS[backend_language])

            # self._A = ah.assemble()
            self._A = ah.forms[0]._matrix

            spaces = self._A.domain.spaces

            if isinstance(Interfaces, Interface):
                Interfaces = (Interfaces, )

            for b1 in self._A.blocks:
                for A in b1:
                    if A is None:
                        continue
                    A[:, :, :, :] = 0

            indices = [slice(None, None)] * domain.dim + [0] * domain.dim

            for i in range(len(self._A.blocks)):
                self._A[i, i][tuple(indices)] = 1

            for I in Interfaces:

                axis = I.axis
                i_minus = get_patch_index_from_face(domain, I.minus)
                i_plus = get_patch_index_from_face(domain, I.plus)

                sp_minus = spaces[i_minus]
                sp_plus = spaces[i_plus]

                s_minus = sp_minus.starts[axis]
                e_minus = sp_minus.ends[axis]

                s_plus = sp_plus.starts[axis]
                e_plus = sp_plus.ends[axis]

                d_minus = V0h.spaces[i_minus].degree[axis]
                d_plus = V0h.spaces[i_plus].degree[axis]

                indices = [slice(None, None)] * domain.dim + [0] * domain.dim

                minus_ext = I.minus.ext
                plus_ext = I.plus.ext

                if minus_ext == 1:
                    indices[axis] = e_minus
                else:
                    indices[axis] = s_minus
                self._A[i_minus, i_minus][tuple(indices)] = 1 / 2

                if plus_ext == 1:
                    indices[axis] = e_plus
                else:
                    indices[axis] = s_plus

                self._A[i_plus, i_plus][tuple(indices)] = 1 / 2

                if plus_ext == minus_ext:
                    if minus_ext == 1:
                        indices[axis] = d_minus
                    else:
                        indices[axis] = s_minus

                    self._A[i_minus, i_plus][tuple(indices)] = 1 / 2

                    if plus_ext == 1:
                        indices[axis] = d_plus
                    else:
                        indices[axis] = s_plus

                    self._A[i_plus, i_minus][tuple(indices)] = 1 / 2

                else:
                    if minus_ext == 1:
                        indices[axis] = d_minus
                    else:
                        indices[axis] = s_minus

                    if plus_ext == 1:
                        indices[domain.dim + axis] = d_plus
                    else:
                        indices[domain.dim + axis] = -d_plus

                    self._A[i_minus, i_plus][tuple(indices)] = 1 / 2

                    if plus_ext == 1:
                        indices[axis] = d_plus
                    else:
                        indices[axis] = s_plus

                    if minus_ext == 1:
                        indices[domain.dim + axis] = d_minus
                    else:
                        indices[domain.dim + axis] = -d_minus

                    self._A[i_plus, i_minus][tuple(indices)] = 1 / 2

            domain = domain.logical_domain
            corner_blocks = {}
            for c in domain.corners:
                for b1 in c.corners:
                    i = get_patch_index_from_face(domain, b1.domain)
                    for b2 in c.corners:
                        j = get_patch_index_from_face(domain, b2.domain)
                        if (i, j) in corner_blocks:
                            corner_blocks[i, j] += [(b1, b2)]
                        else:
                            corner_blocks[i, j] = [(b1, b2)]

            for c in domain.corners:
                if len(c) == 2:
                    continue
                for b1 in c.corners:
                    i = get_patch_index_from_face(domain, b1.domain)
                    for b2 in c.corners:
                        j = get_patch_index_from_face(domain, b2.domain)
                        interface = get_interface_from_corners(b1, b2, domain)
                        axis = None
                        if self._A[i, j] is None:
                            self._A[i, j] = allocate_interface_matrix(
                                corner_blocks[i, j], V0h.spaces[i], V0h.spaces[j])

                        if i != j and self._A[i, j]:
                            axis = self._A[i, j]._dim
                        index = get_row_col_index(
                            b1, b2, interface, axis, V0h.spaces[i], V0h.spaces[j])
                        self._A[i, j][tuple(index)] = 1 / len(c)

            if hom_bc:
                for bn in domain.boundary:
                    self.set_homogenous_bc(bn)

            self._matrix = self._A
            self._sparse_matrix = self._matrix.tosparse()  # self._sparse_matrix

            if storage_fn:
                print(
                    "[ConformingProjection_V0] storing operator sparse matrix in " +
                    storage_fn)
                save_npz(storage_fn, self._sparse_matrix)

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
            apply_essential_bc_stencil(
                rhs[i], axis=boundary.axis, ext=boundary.ext, order=0)
        for j in range(len(domain)):
            if self._A[i, j] is None:
                continue
            apply_essential_bc_stencil(
                self._A[i, j], axis=boundary.axis, ext=boundary.ext, order=0)

        for c in corners:
            faces = [f for b in c.corners for f in b.boundaries]
            if len(c) == 2:
                continue
            if boundary in faces:
                for b1 in c.corners:
                    i = get_patch_index_from_face(domain, b1.domain)
                    for b2 in c.corners:
                        j = get_patch_index_from_face(domain, b2.domain)
                        interface = get_interface_from_corners(b1, b2, domain)
                        axis = None
                        if i != j:
                            axis = self._A[i, j].dim
                        index = get_row_col_index(
                            b1, b2, interface, axis, Vh.spaces[i], Vh.spaces[j])
                        self._A[i, j][tuple(index)] = 0.

                        if i == j and rhs:
                            rhs[i][tuple(index[:2])] = 0.

# ===============================================================================


class ConformingProjection_V1(FemLinearOperator):
    """
    Conforming projection from global broken V1 space to conforming V1 global space

    proj.dot(v) returns the conforming projection of v, computed by solving linear system

    Parameters
    ----------
    V1h: <FemSpace>
     The discrete space

    domain_h: <Geometry>
     The discrete domain of the projector

    hom_bc : <bool>
     Apply homogenous boundary conditions if True

    backend_language: <str>
     The backend used to accelerate the code

    storage_fn:
     filename to store/load the operator sparse matrix
    """
    # todo (MCP, 16.03.2021):
    #   - avoid discretizing a bilinear form
    #   - allow case without interfaces (single or multipatch)

    def __init__(
            self,
            V1h,
            domain_h,
            hom_bc=False,
            backend_language='python',
            storage_fn=None):

        FemLinearOperator.__init__(self, fem_domain=V1h)

        V1 = V1h.symbolic_space
        domain = V1.domain
        self.symbolic_domain = domain

        if storage_fn and os.path.exists(storage_fn):
            print(
                "[ConformingProjection_V1] loading operator sparse matrix from " +
                storage_fn)
            self._sparse_matrix = load_npz(storage_fn)

        else:
            # assemble the operator matrix
            u, v = elements_of(V1, names='u, v')
            expr = dot(u, v)
            #
            Interfaces = domain.interfaces  # note: interfaces does not include the boundary
            # this penalization is for an H1-conforming space
            expr_I = dot(plus(u) - minus(u), plus(v) - minus(v))

            a = BilinearForm((u, v), integral(domain, expr) +
                             integral(Interfaces, expr_I))
            # print('[[ forcing python backend for ConformingProjection_V1]] ')
            # backend_language = 'python'
            ah = discretize(
                a, domain_h, [
                    V1h, V1h], backend=PSYDAC_BACKENDS[backend_language])
            #
            # # self._A = ah.assemble()
            self._A = ah.forms[0]._matrix
            # C1 = V1h.vector_space
            # self._A = BlockLinearOperator(C1, C1)

            for b1 in self._A.blocks:
                for b2 in b1:
                    if b2 is None:
                        continue
                    for b3 in b2.blocks:
                        for A in b3:
                            if A is None:
                                continue
                            A[:, :, :, :] = 0

            spaces = self._A.domain.spaces

            if isinstance(Interfaces, Interface):
                Interfaces = (Interfaces, )

            indices = [slice(None, None)] * domain.dim + [0] * domain.dim

            for i in range(len(self._A.blocks)):
                self._A[i, i][0, 0][tuple(indices)] = 1
                self._A[i, i][1, 1][tuple(indices)] = 1

            # empty list if no interfaces ?
            if Interfaces is not None:

                for I in Interfaces:

                    i_minus = get_patch_index_from_face(domain, I.minus)
                    i_plus = get_patch_index_from_face(domain, I.plus)

                    indices = [slice(None, None)] * \
                        domain.dim + [0] * domain.dim

                    sp1 = spaces[i_minus]
                    sp2 = spaces[i_plus]

                    s11 = sp1.spaces[0].starts[I.axis]
                    e11 = sp1.spaces[0].ends[I.axis]
                    s12 = sp1.spaces[1].starts[I.axis]
                    e12 = sp1.spaces[1].ends[I.axis]

                    s21 = sp2.spaces[0].starts[I.axis]
                    e21 = sp2.spaces[0].ends[I.axis]
                    s22 = sp2.spaces[1].starts[I.axis]
                    e22 = sp2.spaces[1].ends[I.axis]

                    d11 = V1h.spaces[i_minus].spaces[0].degree[I.axis]
                    d12 = V1h.spaces[i_minus].spaces[1].degree[I.axis]

                    d21 = V1h.spaces[i_plus].spaces[0].degree[I.axis]
                    d22 = V1h.spaces[i_plus].spaces[1].degree[I.axis]

                    s_minus = [s11, s12]
                    e_minus = [e11, e12]

                    s_plus = [s21, s22]
                    e_plus = [e21, e22]

                    d_minus = [d11, d12]
                    d_plus = [d21, d22]

                    minus_ext = I.minus.ext
                    plus_ext = I.plus.ext

                    axis = I.axis
                    for k in range(domain.dim):
                        if k == I.axis:
                            continue

                        if minus_ext == 1:
                            indices[axis] = e_minus[k]
                        else:
                            indices[axis] = s_minus[k]
                        self._A[i_minus, i_minus][k, k][tuple(indices)] = 1 / 2

                        if plus_ext == 1:
                            indices[axis] = e_plus[k]
                        else:
                            indices[axis] = s_plus[k]

                        self._A[i_plus, i_plus][k, k][tuple(indices)] = 1 / 2

                        if plus_ext == minus_ext:
                            if minus_ext == 1:
                                indices[axis] = d_minus[k]
                            else:
                                indices[axis] = s_minus[k]

                            self._A[i_minus, i_plus][k, k][tuple(
                                indices)] = 1 / 2 * I.direction

                            if plus_ext == 1:
                                indices[axis] = d_plus[k]
                            else:
                                indices[axis] = s_plus[k]

                            self._A[i_plus, i_minus][k, k][tuple(
                                indices)] = 1 / 2 * I.direction

                        else:
                            if minus_ext == 1:
                                indices[axis] = d_minus[k]
                            else:
                                indices[axis] = s_minus[k]

                            if plus_ext == 1:
                                indices[domain.dim + axis] = d_plus[k]
                            else:
                                indices[domain.dim + axis] = -d_plus[k]

                            self._A[i_minus, i_plus][k, k][tuple(
                                indices)] = 1 / 2 * I.direction

                            if plus_ext == 1:
                                indices[axis] = d_plus[k]
                            else:
                                indices[axis] = s_plus[k]

                            if minus_ext == 1:
                                indices[domain.dim + axis] = d_minus[k]
                            else:
                                indices[domain.dim + axis] = -d_minus[k]

                            self._A[i_plus, i_minus][k, k][tuple(
                                indices)] = 1 / 2 * I.direction

            if hom_bc:
                for bn in domain.boundary:
                    self.set_homogenous_bc(bn)

            self._matrix = self._A
            self._sparse_matrix = self._matrix.tosparse()

            if storage_fn:
                print(
                    "[ConformingProjection_V1] storing operator sparse matrix in " +
                    storage_fn)
                save_npz(storage_fn, self._sparse_matrix)

    def set_homogenous_bc(self, boundary):
        domain = self.symbolic_domain
        Vh = self.fem_domain

        i = get_patch_index_from_face(domain, boundary)
        axis = boundary.axis
        ext = boundary.ext
        for j in range(len(domain)):
            if self._A[i, j] is None:
                continue
            apply_essential_bc_stencil(
                self._A[i, j][1 - axis, 1 - axis], axis=axis, ext=ext, order=0)


# ===============================================================================
def get_K0_and_K0_inv(V0h, uniform_patches=False):
    """
    Compute the change of basis matrices K0 and K0^{-1} in V0h.

    With
    K0_ij = sigma^0_i(B_j) = B_jx(n_ix) * B_jy(n_iy)
    where sigma_i is the geometric (interpolation) dof
    and B_j is the tensor-product B-spline
    """
    if uniform_patches:
        print(' [[WARNING -- hack in get_K0_and_K0_inv: using copies of 1st-patch matrices in every patch ]] ')

    V0 = V0h.symbolic_space   # VOh is ProductFemSpace
    domain = V0.domain
    K0_blocks = []
    K0_inv_blocks = []
    for k, D in enumerate(domain.interior):
        if uniform_patches and k > 0:
            K0_k = K0_blocks[0].copy()
            K0_inv_k = K0_inv_blocks[0].copy()

        else:
            V0_k = V0h.spaces[k]  # fem space on patch k: (TensorFemSpace)
            K0_k_factors = [None, None]
            for d in [0, 1]:
                # 1d fem space alond dim d (SplineSpace)
                V0_kd = V0_k.spaces[d]
                K0_k_factors[d] = collocation_matrix(
                    knots=V0_kd.knots,
                    degree=V0_kd.degree,
                    periodic=V0_kd.periodic,
                    normalization=V0_kd.basis,
                    xgrid=V0_kd.greville
                )
            K0_k = kron(*K0_k_factors)
            K0_k.eliminate_zeros()
            K0_inv_k = inv(K0_k.tocsc())
            K0_inv_k.eliminate_zeros()

        K0_blocks.append(K0_k)
        K0_inv_blocks.append(K0_inv_k)
    K0 = block_diag(K0_blocks)
    K0_inv = block_diag(K0_inv_blocks)
    return K0, K0_inv


# ===============================================================================
def get_K1_and_K1_inv(V1h, uniform_patches=False):
    """
    Compute the change of basis matrices K1 and K1^{-1} in Hcurl space V1h.

    With
    K1_ij = sigma^1_i(B_j) = int_{e_ix}(M_jx) * B_jy(n_iy)
    if i = horizontal edge [e_ix, n_iy] and j = (M_jx o B_jy)  x-oriented MoB spline
    or
    = B_jx(n_ix) * int_{e_iy}(M_jy)
    if i = vertical edge [n_ix, e_iy]  and  j = (B_jx o M_jy)  y-oriented BoM spline
    (above, 'o' denotes tensor-product for functions)
    """
    if uniform_patches:
        print(' [[WARNING -- hack in get_K1_and_K1_inv: using copies of 1st-patch matrices in every patch ]] ')

    V1 = V1h.symbolic_space   # V1h is ProductFemSpace
    domain = V1.domain
    K1_blocks = []
    K1_inv_blocks = []
    for k, D in enumerate(domain.interior):
        if uniform_patches and k > 0:
            K1_k = K1_blocks[0].copy()
            K1_inv_k = K1_inv_blocks[0].copy()

        else:
            # fem space on patch k: (ProductFemSpace (of TensorFemSpace (s))
            V1_k = V1h.spaces[k]
            K1_k_blocks = []
            for c in [0, 1]:    # dim of component
                # fem space for comp. dc (TensorFemSpace)
                V1_kc = V1_k.spaces[c]
                K1_kc_factors = [None, None]
                for d in [0, 1]:    # dim of variable
                    # 1d fem space for comp c alond dim d (SplineSpace)
                    V1_kcd = V1_kc.spaces[d]
                    if c == d:
                        K1_kc_factors[d] = histopolation_matrix(
                            knots=V1_kcd.knots,
                            degree=V1_kcd.degree,
                            periodic=V1_kcd.periodic,
                            normalization=V1_kcd.basis,
                            xgrid=V1_kcd.ext_greville
                        )
                    else:
                        K1_kc_factors[d] = collocation_matrix(
                            knots=V1_kcd.knots,
                            degree=V1_kcd.degree,
                            periodic=V1_kcd.periodic,
                            normalization=V1_kcd.basis,
                            xgrid=V1_kcd.greville
                        )
                K1_kc = kron(*K1_kc_factors)
                K1_kc.eliminate_zeros()
                K1_k_blocks.append(K1_kc)
            K1_k = block_diag(K1_k_blocks)
            K1_k.eliminate_zeros()
            K1_inv_k = inv(K1_k.tocsc())
            K1_inv_k.eliminate_zeros()

        K1_blocks.append(K1_k)
        K1_inv_blocks.append(K1_inv_k)

    K1 = block_diag(K1_blocks)
    K1_inv = block_diag(K1_inv_blocks)
    return K1, K1_inv


# #===============================================================================
# def get_M_and_M_inv(Vh, subdomains_h, is_scalar, backend_language='python'):
#     """
#     compute the mass matrix M and M^{-1} in multipatch space Vh
#     DOES NOT WORK -- SHOULD WE HAVE THE POSSIBILITY OF DOING THAT ?
#     """
#     from pprint import pprint
#
#     V = Vh.symbolic_space   # VOh is ProductFemSpace
#     domain = V.domain
#     M_blocks = []
#     M_inv_blocks = []
#
#     # print('type(domain_h) = ', type(domain_h))
#     #
#     # print('type(domain_h._patches) = ', type(domain_h._patches))
#     # print('len(domain_h._patches) = ', len(domain_h._patches))
#     #
#     # mappings = domain_h.mappings
#     # print('type(mappings) = ', type(mappings))
#     # print('len(mappings) = ', len(mappings))
#     #
#     # mappings_list = list(mappings.values())
#     # print('len(mappings_list) = ', len(mappings_list))
#     #
#     # print('type(mappings_list[0]) = ', type(mappings_list[0]))
#
#     for k, Dh_k in enumerate(subdomains_h):
#
#         print('k = ', k)
#         print('type(Dh_k) = ', type(Dh_k))
#         # print('Dh = ', Dh)
#         D_k = domain.interior[k]
#
#     # exit()
#
#     # for k, D in enumerate(domain.interior):
#
#         V_k = V.spaces[k]
#         Vh_k = Vh.spaces[k]
#
#         # print(type(domain_h))
#         #
#         # pprint(dir(domain_h))
#         #
#         #
#         # print(len(domain_h._patches))
#         # exit()
#         # Dh_k = domain_h.spaces[k]  # fem space on patch k: (TensorFemSpace)
#         u, v = elements_of(V_k, names='u, v')
#         if is_scalar:
#             expr   = u*v
#         else:
#             expr   = dot(u,v)
#         a_k = BilinearForm((u,v), integral(D_k, expr))
#         a_kh = discretize(a_k, Dh_k, [Vh_k, Vh_k], backend=PSYDAC_BACKENDS[backend_language])   # 'pyccel-gcc'])
#
#         M_k = a_kh.assemble().toarray()
#         M_k.eliminate_zeros()
#         M_inv_k = inv(M_k.tocsc())
#         M_inv_k.eliminate_zeros()
#
#         M_blocks.append(M_k)
#         M_inv_blocks.append(M_inv_k)
#     M = block_diag(M_blocks)
#     M_inv = block_diag(M_inv_blocks)
#     return M, M_inv

# ===============================================================================
class HodgeOperator(FemLinearOperator):
    """
    Change of basis operator: dual basis -> primal basis

        self._matrix: matrix of the primal Hodge = this is the mass matrix !
        self.dual_Hodge_matrix: this is the INVERSE mass matrix

    Parameters
    ----------
    Vh: <FemSpace>
     The discrete space

    domain_h: <Geometry>
     The discrete domain of the projector

    metric : <str>
     the metric of the de Rham complex

    backend_language: <str>
     The backend used to accelerate the code

    load_dir: <str>
     storage files for the primal and dual Hodge sparse matrice

    load_space_index: <str>
      the space index in the derham sequence

    Notes
    -----
     Either we use a storage, or these matrices are only computed on demand
     # todo: we compute the sparse matrix when to_sparse_matrix is called -- but never the stencil matrix (should be fixed...)
     We only support the identity metric, this implies that the dual Hodge is the inverse of the primal one.
     # todo: allow for non-identity metrics
    """

    def __init__(
            self,
            Vh,
            domain_h,
            metric='identity',
            backend_language='python',
            load_dir=None,
            load_space_index=''):

        FemLinearOperator.__init__(self, fem_domain=Vh)
        self._domain_h = domain_h
        self._backend_language = backend_language
        self._dual_Hodge_sparse_matrix = None

        assert metric == 'identity'
        self._metric = metric

        if load_dir and isinstance(load_dir, str):
            if not os.path.exists(load_dir):
                os.makedirs(load_dir)
            assert str(load_space_index) in ['0', '1', '2', '3']
            primal_Hodge_storage_fn = load_dir + \
                '/H{}_m.npz'.format(load_space_index)
            dual_Hodge_storage_fn = load_dir + \
                '/dH{}_m.npz'.format(load_space_index)

            primal_Hodge_is_stored = os.path.exists(primal_Hodge_storage_fn)
            dual_Hodge_is_stored = os.path.exists(dual_Hodge_storage_fn)
            if dual_Hodge_is_stored:
                assert primal_Hodge_is_stored
                print(
                    " ...            loading dual Hodge sparse matrix from " +
                    dual_Hodge_storage_fn)
                self._dual_Hodge_sparse_matrix = load_npz(
                    dual_Hodge_storage_fn)
                print(
                    "[HodgeOperator] loading primal Hodge sparse matrix from " +
                    primal_Hodge_storage_fn)
                self._sparse_matrix = load_npz(primal_Hodge_storage_fn)
            else:
                assert not primal_Hodge_is_stored
                print(
                    "[HodgeOperator] assembling both sparse matrices for storage...")
                self.assemble_primal_Hodge_matrix()
                print(
                    "[HodgeOperator] storing primal Hodge sparse matrix in " +
                    primal_Hodge_storage_fn)
                save_npz(primal_Hodge_storage_fn, self._sparse_matrix)
                self.assemble_dual_Hodge_matrix()
                print(
                    "[HodgeOperator] storing dual Hodge sparse matrix in " +
                    dual_Hodge_storage_fn)
                save_npz(dual_Hodge_storage_fn, self._dual_Hodge_sparse_matrix)
        else:
            # matrices are not stored, we will probably compute them later
            pass

    def copy(self):
        raise NotImplementedError

    def to_sparse_matrix(self):
        """
        the Hodge matrix is the patch-wise multi-patch mass matrix
        it is not stored by default but assembled on demand
        """

        if (self._sparse_matrix is not None) or (self._matrix is not None):
            return FemLinearOperator.to_sparse_matrix(self)

        self.assemble_primal_Hodge_matrix()

        return self._sparse_matrix

    def assemble_primal_Hodge_matrix(self):
        """
        the Hodge matrix is the patch-wise multi-patch mass matrix
        it is not stored by default but assembled on demand
        """

        if self._matrix is None:
            Vh = self.fem_domain
            assert Vh == self.fem_codomain

            V = Vh.symbolic_space
            domain = V.domain
            # domain_h = V0h.domain:  would be nice...
            u, v = elements_of(V, names='u, v')

            if isinstance(u, ScalarFunction):
                expr = u * v
            else:
                expr = dot(u, v)

            a = BilinearForm((u, v), integral(domain, expr))
            ah = discretize(a, self._domain_h, [
                            Vh, Vh], backend=PSYDAC_BACKENDS[self._backend_language])

            self._matrix = ah.assemble()  # Mass matrix in stencil format
            self._sparse_matrix = self._matrix.tosparse()

    def get_dual_Hodge_sparse_matrix(self):
        if self._dual_Hodge_sparse_matrix is None:
            self.assemble_dual_Hodge_matrix()

        return self._dual_Hodge_sparse_matrix

    def assemble_dual_Hodge_matrix(self):
        """
        the dual Hodge matrix is the patch-wise inverse of the multi-patch mass matrix
        it is not stored by default but computed on demand, by local (patch-wise) inversion of the mass matrix
        """

        if self._dual_Hodge_sparse_matrix is None:
            if not self._matrix:
                self.assemble_primal_Hodge_matrix()

            M = self._matrix  # mass matrix of the (primal) basis
            nrows = M.n_block_rows
            ncols = M.n_block_cols

            inv_M_blocks = []
            for i in range(nrows):
                Mii = M[i, i].tosparse()
                inv_Mii = inv(Mii.tocsc())
                inv_Mii.eliminate_zeros()
                inv_M_blocks.append(inv_Mii)

            inv_M = block_diag(inv_M_blocks)
            self._dual_Hodge_sparse_matrix = inv_M

# ==============================================================================


class BrokenGradient_2D(FemLinearOperator):

    def __init__(self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V0h, fem_codomain=V1h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockLinearOperator(self.domain, self.codomain, blocks={
                                           (i, i): D0i._matrix for i, D0i in enumerate(D0s)})

    def transpose(self, conjugate=False):
        # todo (MCP): define as the dual differential operator
        return BrokenTransposedGradient_2D(self.fem_domain, self.fem_codomain)
    
    def copy(self):
        return BrokenGradient_2D(self.fem_domain, self.fem_codomain)

# ==============================================================================


class BrokenTransposedGradient_2D(FemLinearOperator):

    def __init__(self, V0h, V1h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V0h)

        D0s = [Gradient_2D(V0, V1) for V0, V1 in zip(V0h.spaces, V1h.spaces)]

        self._matrix = BlockLinearOperator(self.domain, self.codomain, blocks={
                                           (i, i): D0i._matrix.T for i, D0i in enumerate(D0s)})

    def transpose(self, conjugate=False):
        # todo (MCP): discard
        return BrokenGradient_2D(self.fem_codomain, self.fem_domain)
    
    def copy(self):
        return BrokenTransposedGradient_2D(self.fem_domain, self.fem_codomain)


# ==============================================================================
class BrokenScalarCurl_2D(FemLinearOperator):
    def __init__(self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V2h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._matrix = BlockLinearOperator(self.domain, self.codomain, blocks={
                                           (i, i): D1i._matrix for i, D1i in enumerate(D1s)})

    def transpose(self, conjugate=False):
        return BrokenTransposedScalarCurl_2D(
            V1h=self.fem_domain, V2h=self.fem_codomain)
    
    def copy(self):
        return BrokenScalarCurl_2D(self.fem_domain, self.fem_codomain)


# ==============================================================================
class BrokenTransposedScalarCurl_2D(FemLinearOperator):

    def __init__(self, V1h, V2h):

        FemLinearOperator.__init__(self, fem_domain=V2h, fem_codomain=V1h)

        D1s = [ScalarCurl_2D(V1, V2) for V1, V2 in zip(V1h.spaces, V2h.spaces)]

        self._matrix = BlockLinearOperator(self.domain, self.codomain, blocks={
                                           (i, i): D1i._matrix.T for i, D1i in enumerate(D1s)})

    def transpose(self, conjugate=False):
        return BrokenScalarCurl_2D(V1h=self.fem_codomain, V2h=self.fem_domain)
    
    def copy(self):
        return BrokenTransposedScalarCurl_2D(self.fem_domain, self.fem_codomain)


# ==============================================================================

# def multipatch_Moments_Hcurl(f, V1h, domain_h):

def ortho_proj_Hcurl(EE, V1h, domain_h, M1, backend_language='python'):
    """
    return orthogonal projection of E on V1h, given M1 the mass matrix
    """
    assert isinstance(EE, Tuple)
    V1 = V1h.symbolic_space
    v = element_of(V1, name='v')
    l = LinearForm(v, integral(V1.domain, dot(v, EE)))
    lh = discretize(
        l,
        domain_h,
        V1h,
        backend=PSYDAC_BACKENDS[backend_language])
    b = lh.assemble()
    M1_inv = inverse(M1.mat(), 'pcg', pc='jacobi', tol=1e-10)
    sol_coeffs = M1_inv @ b

    return FemField(V1h, coeffs=sol_coeffs)

# ==============================================================================


class Multipatch_Projector_H1:
    """
    to apply the H1 projection (2D) on every patch
    """

    def __init__(self, V0h):

        self._P0s = [Projector_H1(V) for V in V0h.spaces]
        self._V0h = V0h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        u0s = [P(fun) for P, fun, in zip(self._P0s, funs_log)]

        u0_coeffs = BlockVector(self._V0h.vector_space,
                                blocks=[u0j.coeffs for u0j in u0s])

        return FemField(self._V0h, coeffs=u0_coeffs)

# ==============================================================================


class Multipatch_Projector_Hcurl:

    """
    to apply the Hcurl projection (2D) on every patch
    """

    def __init__(self, V1h, nquads=None):

        self._P1s = [Projector_Hcurl(V, nquads=nquads) for V in V1h.spaces]
        self._V1h = V1h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        E1s = [P(fun) for P, fun, in zip(self._P1s, funs_log)]

        E1_coeffs = BlockVector(self._V1h.vector_space,
                                blocks=[E1j.coeffs for E1j in E1s])

        return FemField(self._V1h, coeffs=E1_coeffs)

# ==============================================================================


class Multipatch_Projector_L2:

    """
    to apply the L2 projection (2D) on every patch
    """

    def __init__(self, V2h, nquads=None):

        self._P2s = [Projector_L2(V, nquads=nquads) for V in V2h.spaces]
        self._V2h = V2h   # multipatch Fem Space

    def __call__(self, funs_log):
        """
        project a list of functions given in the logical domain
        """
        B2s = [P(fun) for P, fun, in zip(self._P2s, funs_log)]

        B2_coeffs = BlockVector(self._V2h.vector_space,
                                blocks=[B2j.coeffs for B2j in B2s])

        return FemField(self._V2h, coeffs=B2_coeffs)
