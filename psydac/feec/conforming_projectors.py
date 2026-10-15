#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

# Conga operators on piecewise (broken) de Rham sequences
import os
import numpy as np

from scipy.sparse   import eye as sparse_eye
from scipy.sparse   import csr_matrix
from scipy.special  import comb

from sympde.topology import Boundary, Interface

from psydac.core.bsplines           import quadrature_grid, basis_ders_on_quad_grid, find_spans, elements_spans, cell_index, basis_ders_on_irregular_grid
from psydac.fem.basic               import FemLinearOperator
from psydac.fem.splines             import SplineSpace
from psydac.utilities.quadratures   import gauss_legendre
from psydac.linalg.sparse           import SparseMatrixLinearOperator

__all__ = (
    'ConformingProjectionV0',
    'ConformingProjectionV1',
)

def get_patch_index_from_face(domain, face):
    """
    Return the patch index of subdomain/boundary

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


class Local2GlobalIndexMap:
    def __init__(self, ndim, n_patches, n_components):
        self._shapes = [None] * n_patches
        self._ndofs = [None] * n_patches
        self._ndim = ndim
        self._n_patches = n_patches
        self._n_components = n_components

    def set_patch_shapes(self, patch_index, *shapes):
        assert len(shapes) == self._n_components
        assert all(len(s) == self._ndim for s in shapes)
        self._shapes[patch_index] = shapes
        self._ndofs[patch_index] = sum(np.prod(s) for s in shapes)

    def get_index(self, k, d, cartesian_index):
        """ Return a global scalar index.

            Parameters
            ----------
            k : int
             The patch index.

            d : int
              The component of a scalar field in the system of equations.

            cartesian_index: tuple[int]
              Multi index [i1, i2, i3 ...]

            Returns
            -------
            I : int
             The global scalar index.
        """
        sizes = [np.prod(s) for s in self._shapes[k][:d]]
        Ipc = np.ravel_multi_index(
            cartesian_index, dims=self._shapes[k][d], order='C')
        Ip = sum(sizes) + Ipc
        I = sum(self._ndofs[:k]) + Ip
        return I


def knots_to_insert(coarse_grid, fine_grid, tol=1e-14):
    """knot insertion for refinement of a 1d spline space."""
    intersection = coarse_grid[(
        np.abs(fine_grid[:, None] - coarse_grid) < tol).any(0)]
    assert abs(intersection - coarse_grid).max() < tol
    T = fine_grid[~(np.abs(coarse_grid[:, None] - fine_grid) < tol).any(0)]
    return T


def get_corners(domain, boundary_only):
    """
    Given the domain, extract the vertices on their respective domains with local coordinates.

    Parameters
    ----------
    domain: <Geometry>
     The discrete domain of the projector

    boundary_only : <bool>
     Only return vertices that lie on a boundary

    """
    cos = domain.corners
    patches = domain.interior.args
    bd = domain.boundary

    corner_data = dict()

    if boundary_only:
        for co in cos:

            corner_data[co] = dict()
            c = False
            for cb in co.corners:
                axis = set()
                # check if corner boundary is part of the domain boundary
                for cbbd in cb.args:
                    if bd.has(cbbd):
                        c = True

                p_ind = patches.index(cb.domain)
                c_coord = cb.coordinates
                corner_data[co][p_ind] = c_coord

            if not c:
                corner_data.pop(co)

    else:
        for co in cos:
            corner_data[co] = dict()
            for cb in co.corners:
                p_ind = patches.index(cb.domain)
                c_coord = cb.coordinates
                corner_data[co][p_ind] = c_coord

    return corner_data


def construct_extension_operator_1D(domain, codomain):
    """
    Compute the matrix of the extension operator on the interface.

    Parameters
    ----------
    domain  :     1d spline space on the interface (coarse grid)
    codomain    :   1d spline space on the interface (fine grid)
    """

    from psydac.core.bsplines import hrefinement_matrix
    ops = []

    assert domain.ncells <= codomain.ncells

    Ts = knots_to_insert(domain.breaks, codomain.breaks)
    P = hrefinement_matrix(Ts, domain.degree, domain.knots)

    if domain.basis == 'M':
        assert codomain.basis == 'M'
        P = np.diag(
            1 / codomain._scaling_array) @ P @ np.diag(domain._scaling_array)

    return csr_matrix(P)


def construct_restriction_operator_1D(
        coarse_space_1d, fine_space_1d, E, p_moments=-1):
    """
    Compute the matrix of the (moment preserving) restriction operator on the interface.

    Parameters
    ----------
    coarse_space_1d  :     1d spline space on the interface (coarse grid)
    fine_space_1d    :     1d spline space on the interface (fine grid)
    E                :     Extension matrix
    p_moments        :     Amount of moments to be preserved
    """
    n_c = coarse_space_1d.nbasis
    n_f = fine_space_1d.nbasis
    R = np.zeros((n_c, n_f))

    if coarse_space_1d.basis == 'B':

        #map V^+ to V^+_0
        T = np.zeros((n_f, n_f))
        for i in range(n_f):
            for j in range(n_f):
                T[i, j] = int(i == j) - E[i, 0] * int(0 == j) - E[i, -1] * int(n_f - 1 == j)

        cf_mass_mat = calculate_mixed_mass_matrix(coarse_space_1d, fine_space_1d).transpose()
        c_mass_mat = calculate_mass_matrix(coarse_space_1d)

        if p_moments > 0:
            # L^2 projection from V^+_0 to V^-
            R[:, 1:-1] = np.linalg.solve(c_mass_mat, cf_mass_mat[:, 1:-1])
            gamma = get_1d_moment_correction(coarse_space_1d, p_moments=p_moments)
            n = len(gamma)  

            # maps V^- to V^+_0 in a moment preserving way
            T2 = np.eye(n_c)
            T2[0, 0] = T2[-1, -1] = 0
            T2[1:n+1, 0] += gamma
            T2[-(n+1):-1, -1] += gamma[::-1]

            # maps V^+ to V^- in a moment preserving way
            R = T2 @ R @ T        

        else: 
            R[1:-1, 1:-1] = np.linalg.solve(c_mass_mat[1:-1, 1:-1], cf_mass_mat[1:-1, 1:-1])
            R = R @ T
    
        # add the degrees of freedom of T back
        R[0, 0] += 1
        R[-1, -1] += 1

    else:

        cf_mass_mat = calculate_mixed_mass_matrix(coarse_space_1d, fine_space_1d).transpose()
        c_mass_mat = calculate_mass_matrix(coarse_space_1d)

        # The pure L^2 projection is already moment preserving
        R = np.linalg.solve(c_mass_mat, cf_mass_mat)

    return R


def get_extension_restriction(coarse_space_1d, fine_space_1d, p_moments=-1):
    """
    Calculate the extension and restriction matrices for refining along an interface.

    Parameters
    ----------

    coarse_space_1d : SplineSpace
        Spline space of the coarse space.

    fine_space_1d : SplineSpace
        Spline space of the fine space.

    p_moments : {int}
        Amount of moments to be preserved.

    Returns
    -------
    E_1D : numpy array
        Extension matrix.

    R_1D : numpy array
        Restriction matrix.

    ER_1D : numpy array
        Extension-restriction matrix.
    """
    matching_interfaces = (coarse_space_1d.ncells == fine_space_1d.ncells)
    assert (coarse_space_1d.degree == fine_space_1d.degree) 
    assert (coarse_space_1d.basis == fine_space_1d.basis)
    spl_type = coarse_space_1d.basis

    if not matching_interfaces:
        grid = np.linspace(fine_space_1d.breaks[0], fine_space_1d.breaks[-1], coarse_space_1d.ncells + 1)
        coarse_space_1d_k_plus = SplineSpace(
            degree=fine_space_1d.degree,
            grid=grid,
            basis=fine_space_1d.basis)

        E_1D = construct_extension_operator_1D(
            domain=coarse_space_1d_k_plus, codomain=fine_space_1d)

        
        R_1D = construct_restriction_operator_1D(
            coarse_space_1d_k_plus, fine_space_1d, E_1D, p_moments)
        ER_1D = E_1D @ R_1D

        assert np.allclose(R_1D @ E_1D, np.eye(coarse_space_1d.nbasis), 1e-12, 1e-12)

    else:
        ER_1D = R_1D = E_1D = sparse_eye(
            fine_space_1d.nbasis, format="lil")

    return E_1D, R_1D, ER_1D


# Didn't find this utility in the code base.
def calculate_mass_matrix(space_1d):
    """
    Calculate the mass-matrix of a 1d spline-space.

    Parameters
    ----------

    space_1d : SplineSpace
        Spline space of the fine space.

    Returns
    -------

    Mass_mat : numpy array
        Mass matrix.
    """
    Nel = space_1d.ncells
    deg = space_1d.degree
    knots = space_1d.knots
    spl_type = space_1d.basis

    u, w = gauss_legendre(deg + 1)

    nquad = len(w)
    quad_x, quad_w = quadrature_grid(space_1d.breaks, u, w)

    basis = basis_ders_on_quad_grid(knots, deg, quad_x, 0, spl_type)
    spans = elements_spans(knots, deg)

    Mass_mat = np.zeros((space_1d.nbasis, space_1d.nbasis))

    for ie1 in range(Nel):  # loop on cells
        for il1 in range(deg + 1):  # loops on basis function in each cell
            for il2 in range(deg + 1):  # loops on basis function in each cell
                val = 0.

                for q1 in range(nquad):  # loops on quadrature points
                    v0 = basis[ie1, il1, 0, q1]
                    w0 = basis[ie1, il2, 0, q1]
                    val += quad_w[ie1, q1] * v0 * w0

                locind1 = il1 + spans[ie1] - deg
                locind2 = il2 + spans[ie1] - deg
                Mass_mat[locind1, locind2] += val

    return Mass_mat


# Didn't find this utility in the code base.
def calculate_mixed_mass_matrix(domain_space, codomain_space):
    """
    Calculate the mixed mass-matrix of two 1d spline-spaces on the same domain.

    Parameters
    ----------

    domain_space : SplineSpace
        Spline space of the domain space.

    codomain_space : SplineSpace
        Spline space of the codomain space.

    Returns
    -------

    Mass_mat : numpy array
        Mass matrix.
    """
    if domain_space.nbasis > codomain_space.nbasis:
        coarse_space = codomain_space
        fine_space = domain_space
    else:
        coarse_space = domain_space
        fine_space = codomain_space

    deg = coarse_space.degree
    knots = coarse_space.knots
    spl_type = coarse_space.basis
    breaks = coarse_space.breaks

    fdeg = fine_space.degree
    fknots = fine_space.knots
    fbreaks = fine_space.breaks
    fspl_type = fine_space.basis
    fNel = fine_space.ncells

    assert spl_type == fspl_type
    assert deg == fdeg
    assert ((knots[0] == fknots[0]) and (knots[-1] == fknots[-1]))

    u, w = gauss_legendre(deg + 1)

    nquad = len(w)
    quad_x, quad_w = quadrature_grid(fbreaks, u, w)

    fine_basis = basis_ders_on_quad_grid(fknots, fdeg, quad_x, 0, spl_type)
    coarse_basis = [
        basis_ders_on_irregular_grid(
            knots, deg, q, cell_index(breaks, q), 0, spl_type) for q in quad_x]

    fine_spans = elements_spans(fknots, deg)
    coarse_spans = [find_spans(knots, deg, q[0])[0] for q in quad_x]

    Mass_mat = np.zeros((fine_space.nbasis, coarse_space.nbasis))

    for ie1 in range(fNel):  # loop on cells
        for il1 in range(deg + 1):  # loops on basis function in each cell
            for il2 in range(deg + 1):  # loops on basis function in each cell
                val = 0.

                for q1 in range(nquad):  # loops on quadrature points
                    v0 = fine_basis[ie1, il1, 0, q1]
                    w0 = coarse_basis[ie1][q1, il2, 0]
                    val += quad_w[ie1, q1] * v0 * w0

                locind1 = il1 + fine_spans[ie1] - deg
                locind2 = il2 + coarse_spans[ie1] - deg
                Mass_mat[locind1, locind2] += val

    return Mass_mat


def calculate_poly_basis_integral(space_1d, p_moments=-1):
    """
    Calculate the "mixed mass-matrix" of a 1d spline-space with polynomials.

    Parameters
    ----------

    space_1d : SplineSpace
        Spline space of the fine space.

    p_moments : Int
        Amount of moments to be preserved.

    Returns
    -------

    Mass_mat : numpy array
        Mass matrix.
    """

    Nel = space_1d.ncells
    deg = space_1d.degree
    knots = space_1d.knots
    spl_type = space_1d.basis
    breaks = space_1d.breaks
    enddom = breaks[-1]
    begdom = breaks[0]
    denom = enddom - begdom
    order = max(p_moments + 1, deg + 1)
    u, w = gauss_legendre(order)

    nquad = len(w)
    quad_x, quad_w = quadrature_grid(space_1d.breaks, u, w)

    coarse_basis = basis_ders_on_quad_grid(knots, deg, quad_x, 0, spl_type)
    spans = elements_spans(knots, deg)

    Mass_mat = np.zeros((p_moments + 1, space_1d.nbasis))

    for ie1 in range(Nel):  # loop on cells
        for pol in range(p_moments + 1):  # loops on basis function in each cell
            for il2 in range(deg + 1):  # loops on basis function in each cell
                val = 0.

                for q1 in range(nquad):  # loops on quadrature points
                    v0 = coarse_basis[ie1, il2, 0, q1]
                    x = quad_x[ie1, q1]
                    # val += quad_w[ie1, q1] * v0 * ((enddom-x)/denom)**pol
                    val += quad_w[ie1, q1] * v0 * \
                        comb(p_moments, pol) * ((enddom - x) / denom)**(p_moments - pol) * ((x - begdom) / denom)**pol
                locind2 = il2 + spans[ie1] - deg
                Mass_mat[pol, locind2] += val

    return Mass_mat


def get_1d_moment_correction(space_1d, p_moments=-1):
    """
    Calculate the coefficients for the one-dimensional moment correction.

    Parameters
    ----------
    patch_space : SplineSpace
        1d spline space.

    p_moments  : int
        Number of moments to be preserved.

    Returns
    -------
    gamma : array
        Moment correction coefficients without the conformity factor.
    """

    if p_moments < 0:
        return []

    if space_1d.ncells <= p_moments + 1:
        p_moments = space_1d.ncells - 2
        print(f"The prescribed degree of preserved moments was too high, given the number of cells in the patch. It has been reduced to degree {p_moments}.")
    
    if p_moments >= 0:
        # to preserve moments of degree p we need 1+p conforming basis functions in the patch (the "interior" ones)
        # and for the given regularity constraint, there are
        # local_shape[conf_axis]-2*(1+reg) such conforming functions
        p_max = space_1d.nbasis - 3
        if p_max < p_moments:
            print(" ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            print(" **         WARNING -- WARNING -- WARNING ")
            print(f" ** conf. projection imposing C0 smoothness on scalar space along this axis :")
            print(f" ** there are not enough dofs in a patch to preserve moments of degree {p_moments} !")
            print(f" ** Only able to preserve up to degree --> {p_max} <-- ")
            print(" ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            p_moments = p_max

        Mass_mat = calculate_poly_basis_integral(space_1d, p_moments)
        gamma = np.linalg.solve(Mass_mat[:, 1:p_moments + 2], Mass_mat[:, 0])

    return gamma


#==============================================================================
# Multipatch conforming projectors
#==============================================================================
def construct_h1_conforming_projection(Vh, reg_orders=0, p_moments=-1, hom_bc=False):
    """
    Construct the conforming projection for a scalar space for a given regularity (0 continuous, -1 discontinuous).

    Parameters
    ----------
    Vh : MultipatchFemSpace
        Finite Element Space coming from the discrete de Rham sequence.

    reg_orders :  (int)
        Regularity in each space direction -1 or 0.

    p_moments : (int)
        Number of moments to be preserved.

    hom_bc : (bool)
        Homogeneous boundary conditions.

    Returns
    -------
    cP : scipy.sparse.csr_array
        Conforming projection as a sparse matrix.
    """

    dim_tot = Vh.nbasis

    # fully discontinuous space
    if reg_orders < 0:
        return sparse_eye(dim_tot, format="lil")

    # moment corrections perpendicular to interfaces
    # assume same moments everywhere
    gamma = get_1d_moment_correction(Vh.spaces[0].spaces[0], p_moments=p_moments)
    p_moments = len(gamma)-1

    domain = Vh.symbolic_space.domain
    ndim = 2
    n_components = 1
    n_patches = len(domain)

    l2g = Local2GlobalIndexMap(ndim, len(domain), n_components)
    for k in range(n_patches):
        Vk = Vh.spaces[k]
        # T is a TensorFemSpace and S is a 1D SplineSpace
        shapes = [S.nbasis for S in Vk.spaces]
        l2g.set_patch_shapes(k, shapes)

    # P vertex
    # vertex correction matrix
    Proj_vertex = sparse_eye(dim_tot, format="lil") 

    corner_indices = set()
    corners = get_corners(domain, False)

    def get_vertex_index_from_patch(patch, coords):
        nbasis0 = Vh.spaces[patch].spaces[coords[0]].nbasis - 1
        nbasis1 = Vh.spaces[patch].spaces[coords[1]].nbasis - 1

        # patch local index
        multi_index = [None] * ndim
        multi_index[0] = 0 if coords[0] == 0 else nbasis0
        multi_index[1] = 0 if coords[1] == 0 else nbasis1

        # global index
        return l2g.get_index(patch, 0, multi_index)

    def vertex_moment_indices(axis, coords, patch, p_moments):
        if coords[axis] == 0:
            return range(1, p_moments + 2)
        else:
            return range(Vh.spaces[patch].spaces[coords[axis]].nbasis - 1 - 1,
                         Vh.spaces[patch].spaces[coords[axis]].nbasis - 1 - p_moments - 2, -1)

    # loop over all vertices
    for (bd, co) in corners.items():
        # len(co)=#v is the number of adjacent patches at a vertex
        corr = len(co)

        for patch1 in co:
            # local vertex coordinates in patch1
            coords1 = co[patch1]
            # global index
            ig = get_vertex_index_from_patch(patch1, coords1)

            corner_indices.add(ig)

            for patch2 in co:
                # local vertex coordinates in patch2
                coords2 = co[patch2]

                # global index
                jg = get_vertex_index_from_patch(patch2, coords2)

                # conformity constraint
                Proj_vertex[jg, ig] = 1 / corr

                if patch1 == patch2:
                    continue

                if p_moments == -1:
                    continue

                # moment corrections from patch1 to patch2
                axis = 0
                d = 1
                multi_index_p = [None] * ndim

                d_moment_index = vertex_moment_indices(
                    d, coords2, patch2, p_moments)
                axis_moment_index = vertex_moment_indices(
                    axis, coords2, patch2, p_moments)

                for pd in range(0, p_moments + 1):
                    multi_index_p[d] = d_moment_index[pd]

                    for p in range(0, p_moments + 1):
                        multi_index_p[axis] = axis_moment_index[p]

                        pg = l2g.get_index(patch2, 0, multi_index_p)
                        Proj_vertex[pg, ig] += - 1 / \
                            corr * gamma[p] * gamma[pd]

            if p_moments == -1:
                continue

            # moment corrections from patch1 to patch1
            axis = 0
            d = 1
            multi_index_p = [None] * ndim

            d_moment_index = vertex_moment_indices(
                d, coords1, patch1, p_moments)
            axis_moment_index = vertex_moment_indices(
                axis, coords1, patch1, p_moments)

            for pd in range(0, p_moments + 1):
                multi_index_p[d] = d_moment_index[pd]

                for p in range(0, p_moments + 1):
                    multi_index_p[axis] = axis_moment_index[p]

                    pg = l2g.get_index(patch1, 0, multi_index_p)
                    Proj_vertex[pg, ig] += (1 - 1 / corr) * \
                        gamma[p] * gamma[pd]

    # boundary conditions
    corners = get_corners(domain, True)
    if hom_bc:
        for (bd, co) in corners.items():
            for patch1 in co:

                # local vertex coordinates in patch2
                coords1 = co[patch1]

                # global index
                ig = get_vertex_index_from_patch(patch1, coords1)

                for patch2 in co:

                    # local vertex coordinates in patch2
                    coords2 = co[patch2]

                    # global index
                    jg = get_vertex_index_from_patch(patch2, coords2)

                    # conformity constraint
                    Proj_vertex[jg, ig] = 0

                    if patch1 == patch2:
                        continue

                    if p_moments == -1:
                        continue

                    # moment corrections from patch1 to patch2
                    axis = 0
                    d = 1
                    multi_index_p = [None] * ndim

                    d_moment_index = vertex_moment_indices(
                        d, coords2, patch2, p_moments)
                    axis_moment_index = vertex_moment_indices(
                        axis, coords2, patch2, p_moments)

                    for pd in range(0, p_moments + 1):
                        multi_index_p[d] = d_moment_index[pd]

                        for p in range(0, p_moments + 1):
                            multi_index_p[axis] = axis_moment_index[p]

                            pg = l2g.get_index(patch2, 0, multi_index_p)
                            Proj_vertex[pg, ig] = 0

                if p_moments == -1:
                    continue

                # moment corrections from patch1 to patch1
                axis = 0
                d = 1
                multi_index_p = [None] * ndim

                d_moment_index = vertex_moment_indices(
                    d, coords1, patch1, p_moments)
                axis_moment_index = vertex_moment_indices(
                    axis, coords1, patch1, p_moments)

                for pd in range(0, p_moments + 1):
                    multi_index_p[d] = d_moment_index[pd]

                    for p in range(0, p_moments + 1):
                        multi_index_p[axis] = axis_moment_index[p]

                        pg = l2g.get_index(patch1, 0, multi_index_p)
                        Proj_vertex[pg, ig] = gamma[p] * gamma[pd]

    # P edge
    # edge correction matrix
    Proj_edge = sparse_eye(dim_tot, format="lil")

    Interfaces = domain.interfaces
    if isinstance(Interfaces, Interface):
        Interfaces = (Interfaces, )

    def get_edge_index(j, axis, ext, space, k):
        multi_index = [None] * ndim
        multi_index[axis] = 0 if ext == - 1 else space.spaces[axis].nbasis - 1
        multi_index[1 - axis] = j
        return l2g.get_index(k, 0, multi_index)

    def edge_moment_index(p, i, axis, ext, space, k):
        multi_index = [None] * ndim
        multi_index[1 - axis] = i
        multi_index[axis] = p + 1 if ext == - \
            1 else space.spaces[axis].nbasis - 1 - p - 1
        return l2g.get_index(k, 0, multi_index)

    def get_mu_plus(j, fine_space):
        mu_plus = np.zeros(fine_space.nbasis)
        for p in range(p_moments + 1):
            if j == 0:
                mu_plus[p + 1] = gamma[p]
            else:
                mu_plus[j - (p + 1)] = gamma[p]
        return mu_plus

    def get_mu_minus(j, coarse_space, fine_space, R):
        mu_plus = np.zeros(fine_space.nbasis)
        mu_minus = np.zeros(coarse_space.nbasis)

        if j == 0:
            mu_minus[0] = 1
            for p in range(p_moments + 1):
                mu_plus[p + 1] = gamma[p]
        else:
            mu_minus[-1] = 1
            for p in range(p_moments + 1):
                mu_plus[-1 - (p + 1)] = gamma[p]

        for m in range(coarse_space.nbasis):
            for l in range(fine_space.nbasis):
                mu_minus[m] += R[m, l] * mu_plus[l]

            if j == 0:
                mu_minus[m] -= R[m, 0]
            else:
                mu_minus[m] -= R[m, -1]

        return mu_minus

    # loop over all interfaces
    for I in Interfaces:
        axis = I.axis
        direction = I.ornt
        # for now assume the interfaces are along the same direction
        assert direction == 1
        k_minus = get_patch_index_from_face(domain, I.minus)
        k_plus = get_patch_index_from_face(domain, I.plus)

        I_minus_ncells = Vh.spaces[k_minus].ncells
        I_plus_ncells = Vh.spaces[k_plus].ncells

        # logical directions normal to interface
        if I_minus_ncells <= I_plus_ncells:
            k_fine, k_coarse = k_plus, k_minus
            fine_axis, coarse_axis = I.plus.axis, I.minus.axis
            fine_ext, coarse_ext = I.plus.ext, I.minus.ext

        else:
            k_fine, k_coarse = k_minus, k_plus
            fine_axis, coarse_axis = I.minus.axis, I.plus.axis
            fine_ext, coarse_ext = I.minus.ext, I.plus.ext

        # logical directions along the interface
        d_fine = 1 - fine_axis
        d_coarse = 1 - coarse_axis

        space_fine = Vh.spaces[k_fine]
        space_coarse = Vh.spaces[k_coarse]

        coarse_space_1d = space_coarse.spaces[d_coarse]
        fine_space_1d = space_fine.spaces[d_fine]
        E_1D, R_1D, ER_1D = get_extension_restriction(
            coarse_space_1d, fine_space_1d, p_moments=p_moments)

        # Projecting coarse basis functions
        for j in range(coarse_space_1d.nbasis):
            jg = get_edge_index(
                j,
                coarse_axis,
                coarse_ext,
                space_coarse,
                k_coarse)

            if (not corner_indices.issuperset({jg})):

                Proj_edge[jg, jg] = 1 / 2

                for p in range(p_moments + 1):
                    pg = edge_moment_index(
                        p, j, coarse_axis, coarse_ext, space_coarse, k_coarse)
                    Proj_edge[pg, jg] += 1 / 2 * gamma[p]

                for i in range(fine_space_1d.nbasis):
                    ig = get_edge_index(
                        i, fine_axis, fine_ext, space_fine, k_fine)
                    Proj_edge[ig, jg] = 1 / 2 * E_1D[i, j]

                    for p in range(p_moments + 1):
                        pg = edge_moment_index(
                            p, i, fine_axis, fine_ext, space_fine, k_fine)
                        Proj_edge[pg, jg] += -1 / 2 * gamma[p] * E_1D[i, j]
            else:
                mu_minus = get_mu_minus(
                    j, coarse_space_1d, fine_space_1d, R_1D)

                for p in range(p_moments + 1):
                    for m in range(coarse_space_1d.nbasis):
                        pg = edge_moment_index(
                            p, m, coarse_axis, coarse_ext, space_coarse, k_coarse)
                        Proj_edge[pg, jg] += 1 / 2 * gamma[p] * mu_minus[m]

                for i in range(1, fine_space_1d.nbasis - 1):
                    ig = get_edge_index(
                        i, fine_axis, fine_ext, space_fine, k_fine)
                    Proj_edge[ig, jg] = 1 / 2 * E_1D[i, j]

                    for p in range(p_moments + 1):
                        pg = edge_moment_index(
                            p, i, fine_axis, fine_ext, space_fine, k_fine)
                        for m in range(coarse_space_1d.nbasis):
                            Proj_edge[pg, jg] += -1 / 2 * \
                                gamma[p] * E_1D[i, m] * mu_minus[m]

        # Projecting fine basis functions
        for j in range(fine_space_1d.nbasis):
            jg = get_edge_index(j, fine_axis, fine_ext, space_fine, k_fine)

            if (not corner_indices.issuperset({jg})):
                for i in range(fine_space_1d.nbasis):
                    ig = get_edge_index(
                        i, fine_axis, fine_ext, space_fine, k_fine)
                    Proj_edge[ig, jg] = 1 / 2 * ER_1D[i, j]

                    for p in range(p_moments + 1):
                        pg = edge_moment_index(
                            p, i, fine_axis, fine_ext, space_fine, k_fine)
                        Proj_edge[pg, jg] += 1 / 2 * gamma[p] * ER_1D[i, j]

                for i in range(coarse_space_1d.nbasis):
                    ig = get_edge_index(
                        i, coarse_axis, coarse_ext, space_coarse, k_coarse)
                    Proj_edge[ig, jg] = 1 / 2 * R_1D[i, j]

                    for p in range(p_moments + 1):
                        pg = edge_moment_index(
                            p, i, coarse_axis, coarse_ext, space_coarse, k_coarse)
                        Proj_edge[pg, jg] += - 1 / 2 * gamma[p] * R_1D[i, j]
            else:
                mu_plus = get_mu_plus(j, fine_space_1d)

                for i in range(1, fine_space_1d.nbasis - 1):
                    ig = get_edge_index(
                        i, fine_axis, fine_ext, space_fine, k_fine)
                    Proj_edge[ig, jg] = 1 / 2 * ER_1D[i, j]

                    for p in range(p_moments + 1):
                        pg = edge_moment_index(
                            p, i, fine_axis, fine_ext, space_fine, k_fine)

                        for m in range(fine_space_1d.nbasis):
                            Proj_edge[pg, jg] += 1 / 2 * \
                                gamma[p] * ER_1D[i, m] * mu_plus[m]

                for i in range(1, coarse_space_1d.nbasis - 1):
                    ig = get_edge_index(
                        i, coarse_axis, coarse_ext, space_coarse, k_coarse)
                    Proj_edge[ig, jg] = 1 / 2 * R_1D[i, j]

                    for p in range(p_moments + 1):
                        pg = edge_moment_index(
                            p, i, coarse_axis, coarse_ext, space_coarse, k_coarse)

                        for m in range(fine_space_1d.nbasis):
                            Proj_edge[pg, jg] += - 1 / 2 * \
                                gamma[p] * R_1D[i, m] * mu_plus[m]

    # boundary condition
    if hom_bc:
        for bn in domain.boundary:
            k = get_patch_index_from_face(domain, bn)
            space_k = Vh.spaces[k]
            axis = bn.axis

            d = 1 - axis
            ext = bn.ext
            space_k_1d = space_k.spaces[d]

            for i in range(0, space_k_1d.nbasis):
                ig = get_edge_index(i, axis, ext, space_k, k)
                Proj_edge[ig, ig] = 0

                if (i != 0 and i != space_k_1d.nbasis - 1):
                    for p in range(p_moments + 1):

                        pg = edge_moment_index(p, i, axis, ext, space_k, k)
                        Proj_edge[pg, ig] = gamma[p]
                else:
                    #if corner_indices.issuperset({ig}):
                    mu_minus = get_mu_minus(
                        i, space_k_1d, space_k_1d, np.eye(
                            space_k_1d.nbasis))

                    for p in range(p_moments + 1):
                        for m in range(space_k_1d.nbasis):
                            pg = edge_moment_index(
                                p, m, axis, ext, space_k, k)
                            Proj_edge[pg, ig] = gamma[p] * mu_minus[m]

                    if not corner_indices.issuperset({ig}):
                        corner_indices.add(ig)
                        multi_index = [None] * ndim

                        for p in range(p_moments + 1):
                            multi_index[axis] = p + 1 if ext == - \
                                1 else space_k.spaces[axis].nbasis - 1 - p - 1
                            for pd in range(p_moments + 1):
                                multi_index[1 - axis] = pd + \
                                    1 if i == 0 else space_k.spaces[1 - axis].nbasis - 1 - pd - 1
                                pg = l2g.get_index(k, 0, multi_index)
                                Proj_edge[pg, ig] = gamma[p] * gamma[pd]

    return Proj_edge @ Proj_vertex


def construct_hcurl_conforming_projection(Vh, reg_orders=0, p_moments=-1, hom_bc=False):
    """
    Construct the conforming projection for a vector Hcurl space for a given regularity (0 continuous, -1 discontinuous).

    Parameters
    ----------
    Vh : MultipatchFemSpace
        Finite Element Space coming from the discrete de Rham sequence.

    reg_orders :  (int)
        Regularity in each space direction -1 or 0.

    p_moments : (int)
        Number of polynomial moments to be preserved.

    hom_bc : (bool)
        Tangential homogeneous boundary conditions.

    Returns
    -------
    cP : scipy.sparse.csr_array
        Conforming projection as a sparse matrix.
    """

    dim_tot = Vh.nbasis

    # fully discontinuous space
    if reg_orders < 0:
        return sparse_eye(dim_tot, format="lil")

    # moment corrections perpendicular to interfaces
    # should be in the V^0 spaces
    gamma = [get_1d_moment_correction(Vh.spaces[0].spaces[1 - d].spaces[d], p_moments=p_moments) for d in range(2)]
    p_moments = min([len(g) for g in gamma])-1

    domain = Vh.symbolic_space.domain
    ndim = 2
    n_components = 2
    n_patches = len(domain)

    l2g = Local2GlobalIndexMap(ndim, len(domain), n_components)
    for k in range(n_patches):
        Vk = Vh.spaces[k]
        # T is a TensorFemSpace and S is a 1D SplineSpace
        shapes = [[S.nbasis for S in T.spaces] for T in Vk.spaces]
        l2g.set_patch_shapes(k, *shapes)

    # P edge
    # edge correction matrix
    Proj_edge = sparse_eye(dim_tot, format="lil")

    Interfaces = domain.interfaces
    if isinstance(Interfaces, Interface):
        Interfaces = (Interfaces, )

    def get_edge_index(j, axis, ext, space, k):
        multi_index = [None] * ndim
        multi_index[axis] = 0 if ext == - \
            1 else space.spaces[1 - axis].spaces[axis].nbasis - 1
        multi_index[1 - axis] = j
        return l2g.get_index(k, 1 - axis, multi_index)

    def edge_moment_index(p, i, axis, ext, space, k):
        multi_index = [None] * ndim
        multi_index[1 - axis] = i
        multi_index[axis] = p + 1 if ext == - \
            1 else space.spaces[1 - axis].spaces[axis].nbasis - 1 - p - 1
        return l2g.get_index(k, 1 - axis, multi_index)
    
    # loop over all interfaces
    for I in Interfaces:
        direction = I.ornt
        # for now assume the interfaces are along the same direction
        assert direction == 1
        k_minus = get_patch_index_from_face(domain, I.minus)
        k_plus = get_patch_index_from_face(domain, I.plus)

        # logical directions normal to interface
        minus_axis, plus_axis = I.minus.axis, I.plus.axis
        # logical directions along the interface
        d_minus, d_plus = 1 - minus_axis, 1 - plus_axis
        I_minus_ncells = Vh.spaces[k_minus].spaces[d_minus].ncells[d_minus]
        I_plus_ncells = Vh.spaces[k_plus].spaces[d_plus].ncells[d_plus]

        # logical directions normal to interface
        if I_minus_ncells <= I_plus_ncells:
            k_fine, k_coarse = k_plus, k_minus
            fine_axis, coarse_axis = I.plus.axis, I.minus.axis
            fine_ext, coarse_ext = I.plus.ext, I.minus.ext

        else:
            k_fine, k_coarse = k_minus, k_plus
            fine_axis, coarse_axis = I.minus.axis, I.plus.axis
            fine_ext, coarse_ext = I.minus.ext, I.plus.ext

        # logical directions along the interface
        d_fine = 1 - fine_axis
        d_coarse = 1 - coarse_axis

        space_fine = Vh.spaces[k_fine]
        space_coarse = Vh.spaces[k_coarse]

        coarse_space_1d = space_coarse.spaces[d_coarse].spaces[d_coarse]
        fine_space_1d = space_fine.spaces[d_fine].spaces[d_fine]
        E_1D, R_1D, ER_1D = get_extension_restriction(
            coarse_space_1d, fine_space_1d, p_moments=p_moments)

        # Projecting coarse basis functions
        for j in range(coarse_space_1d.nbasis):
            jg = get_edge_index(
                j,
                coarse_axis,
                coarse_ext,
                space_coarse,
                k_coarse)

            Proj_edge[jg, jg] = 1 / 2

            for p in range(p_moments + 1):
                pg = edge_moment_index(
                    p, j, coarse_axis, coarse_ext, space_coarse, k_coarse)
                Proj_edge[pg, jg] += 1 / 2 * gamma[d_coarse][p]

            for i in range(fine_space_1d.nbasis):
                ig = get_edge_index(i, fine_axis, fine_ext, space_fine, k_fine)
                Proj_edge[ig, jg] = 1 / 2 * E_1D[i, j]

                for p in range(p_moments + 1):
                    pg = edge_moment_index(
                        p, i, fine_axis, fine_ext, space_fine, k_fine)
                    Proj_edge[pg, jg] += -1 / 2 * gamma[d_fine][p] * E_1D[i, j]

        # Projecting fine basis functions
        for j in range(fine_space_1d.nbasis):
            jg = get_edge_index(j, fine_axis, fine_ext, space_fine, k_fine)

            for i in range(fine_space_1d.nbasis):
                ig = get_edge_index(i, fine_axis, fine_ext, space_fine, k_fine)
                Proj_edge[ig, jg] = 1 / 2 * ER_1D[i, j]

                for p in range(p_moments + 1):
                    pg = edge_moment_index(
                        p, i, fine_axis, fine_ext, space_fine, k_fine)
                    Proj_edge[pg, jg] += 1 / 2 * gamma[d_fine][p] * ER_1D[i, j]

            for i in range(coarse_space_1d.nbasis):
                ig = get_edge_index(
                    i, coarse_axis, coarse_ext, space_coarse, k_coarse)
                Proj_edge[ig, jg] = 1 / 2 * R_1D[i, j]

                for p in range(p_moments + 1):
                    pg = edge_moment_index(
                        p, i, coarse_axis, coarse_ext, space_coarse, k_coarse)
                    Proj_edge[pg, jg] += - 1 / 2 * \
                        gamma[d_coarse][p] * R_1D[i, j]

    # boundary condition
    for bn in domain.boundary:
        k = get_patch_index_from_face(domain, bn)
        space_k = Vh.spaces[k]
        axis = bn.axis

        if not hom_bc:
            continue

        d = 1 - axis
        ext = bn.ext
        space_k_1d = space_k.spaces[d].spaces[d]

        for i in range(0, space_k_1d.nbasis):
            ig = get_edge_index(i, axis, ext, space_k, k)
            Proj_edge[ig, ig] = 0

            for p in range(p_moments + 1):

                pg = edge_moment_index(p, i, axis, ext, space_k, k)
                Proj_edge[pg, ig] = gamma[d][p]

    return Proj_edge

#==============================================================================
# Singlepatch conforming projectors
#==============================================================================
def construct_h1_singlepatch_conforming_projection(Vh, reg_orders=0, p_moments=-1, hom_bc=False):
    """
    Construct the conforming projection for a scalar space for a given regularity (0 continuous, -1 discontinuous).

    Parameters
    ----------
    Vh : MultipatchFemSpace
        Finite Element Space coming from the discrete de Rham sequence.

    reg_orders :  (int)
        Regularity in each space direction -1 or 0.

    p_moments : (int)
        Number of moments to be preserved.

    hom_bc : (bool)
        Homogeneous boundary conditions.

    Returns
    -------
    cP : scipy.sparse.csr_array
        Conforming projection as a sparse matrix.
    """

    dim_tot = Vh.nbasis

    # fully discontinuous space
    if reg_orders < 0 or not hom_bc:
        return sparse_eye(dim_tot, format="lil")

    # moment corrections perpendicular to interfaces
    # assume same moments everywhere
    gamma = get_1d_moment_correction(Vh.spaces[0], p_moments=p_moments)
    p_moments = len(gamma)-1

    domain = Vh.symbolic_space.domain
    ndim = 2
    n_components = 1
    n_patches = len(domain)

    l2g = Local2GlobalIndexMap(ndim, len(domain), n_components)
        # T is a TensorFemSpace and S is a 1D SplineSpace
    shapes = [S.nbasis for S in Vh.spaces]
    l2g.set_patch_shapes(0, shapes)

    # P vertex
    # vertex correction matrix
    Proj_vertex = sparse_eye(dim_tot, format="lil") 


    def get_vertex_index(coords):
        """
            Calculate the global index of the vertex basis function
            from the geometric coordinates of a vertex in the domain.
        """
        nbasis0 = Vh.spaces[0].nbasis - 1
        nbasis1 = Vh.spaces[1].nbasis - 1

        # patch local index
        multi_index = [None] * ndim
        multi_index[0] = 0 if coords[0] == 0 else nbasis0
        multi_index[1] = 0 if coords[1] == 0 else nbasis1

        # global index
        return l2g.get_index(0, 0, multi_index)

    def vertex_moment_indices(axis, coords, p_moments):
        """
            Calculate the global indices of the basis functions
            adjacent to the vertex basis function along axis
            from the geometric coordinates of a vertex in the domain.
        """
        if coords[axis] == 0:
            return range(1, p_moments + 2)
        else:
            return range(Vh.spaces[coords[axis]].nbasis - 1 - 1,
                         Vh.spaces[coords[axis]].nbasis - 1 - p_moments - 2, -1)

    # boundary conditions

    for  co in [(0,0), (1,0), (0,1), (1,1)]:

        if all(Vh.periodic):
            break

        # global index
        ig = get_vertex_index(co)

        # conformity constraint
        Proj_vertex[ig, ig] = 0


        if p_moments == -1:
            continue

        # moment corrections from patch1 to patch1
        axis = 0
        d = 1
        multi_index_p = [None] * ndim

        d_moment_index = vertex_moment_indices(d, co, p_moments)
        axis_moment_index = vertex_moment_indices(axis, co, p_moments)

        for pd in range(0, p_moments + 1):
            multi_index_p[d] = d_moment_index[pd]

            for p in range(0, p_moments + 1):
                multi_index_p[axis] = axis_moment_index[p]

                pg = l2g.get_index(0, 0, multi_index_p)
                Proj_vertex[pg, ig] = gamma[p] * gamma[pd]

    # P edge
    # edge correction matrix
    Proj_edge = sparse_eye(dim_tot, format="lil")

    def get_edge_index(j, axis, ext):
        multi_index = [None] * ndim
        multi_index[axis] = 0 if ext == - 1 else Vh.spaces[axis].nbasis - 1
        multi_index[1 - axis] = j
        return l2g.get_index(0, 0, multi_index)

    def edge_moment_index(p, i, axis, ext):
        multi_index = [None] * ndim
        multi_index[1 - axis] = i
        multi_index[axis] = p + 1 if ext == -1 else Vh.spaces[axis].nbasis - 1 - p - 1
        return l2g.get_index(0, 0, multi_index)


    def get_mu_minus(j, coarse_space, fine_space, R):
        mu_plus = np.zeros(fine_space.nbasis)
        mu_minus = np.zeros(coarse_space.nbasis)

        if j == 0:
            mu_minus[0] = 1
            for p in range(p_moments + 1):
                mu_plus[p + 1] = gamma[p]
        else:
            mu_minus[-1] = 1
            for p in range(p_moments + 1):
                mu_plus[-1 - (p + 1)] = gamma[p]

        for m in range(coarse_space.nbasis):
            for l in range(fine_space.nbasis):
                mu_minus[m] += R[m, l] * mu_plus[l]

            if j == 0:
                mu_minus[m] -= R[m, 0]
            else:
                mu_minus[m] -= R[m, -1]

        return mu_minus


    # boundary condition
    for bn in domain.boundary:

        if Vh.periodic[bn.axis]:
            continue

        space_k = Vh
        axis = bn.axis

        d = 1 - axis
        ext = bn.ext
        space_k_1d = space_k.spaces[d]

        for i in range(0, space_k_1d.nbasis):
            ig = get_edge_index(i, axis, ext)
            Proj_edge[ig, ig] = 0

            if (i != 0 and i != space_k_1d.nbasis - 1):
                for p in range(p_moments + 1):

                    pg = edge_moment_index(p, i, axis, ext)
                    Proj_edge[pg, ig] = gamma[p]
            else:
                #if corner_indices.issuperset({ig}):
                mu_minus = get_mu_minus(
                    i, space_k_1d, space_k_1d, np.eye(
                        space_k_1d.nbasis))

                for p in range(p_moments + 1):
                    for m in range(space_k_1d.nbasis):
                        pg = edge_moment_index(
                            p, m, axis, ext)
                        Proj_edge[pg, ig] = gamma[p] * mu_minus[m]


    return Proj_edge @ Proj_vertex


def construct_hcurl_singlepatch_conforming_projection(Vh, reg_orders=0, p_moments=-1, hom_bc=False):
    """
    Construct the conforming projection for a single patch vector Hcurl space for a given regularity (0 continuous, -1 discontinuous).

    Parameters
    ----------
    Vh : MultipatchFemSpace
        Finite Element Space coming from the discrete de Rham sequence.

    reg_orders :  (int)
        Regularity in each space direction -1 or 0.

    p_moments : (int)
        Number of polynomial moments to be preserved.

    hom_bc : (bool)
        Tangential homogeneous boundary conditions.

    Returns
    -------
    cP : scipy.sparse.csr_array
        Conforming projection as a sparse matrix.
    """

    dim_tot = Vh.nbasis

    # fully discontinuous space
    if reg_orders < 0 or not hom_bc:
        return sparse_eye(dim_tot, format="lil")

    # moment corrections perpendicular to interfaces
    # should be in the V^0 spaces

    gamma = [get_1d_moment_correction(Vh.spaces[1 - d].spaces[d], p_moments=p_moments) for d in range(2)]
    p_moments = min([len(g) for g in gamma])-1

    domain = Vh.symbolic_space.domain
    ndim = 2
    n_components = 2
    n_patches = len(domain)

    l2g = Local2GlobalIndexMap(ndim, len(domain), n_components)
    # T is a TensorFemSpace and S is a 1D SplineSpace
    shapes = [[S.nbasis for S in T.spaces] for T in Vh.spaces]
    l2g.set_patch_shapes(0, *shapes)

    # P edge
    # edge correction matrix
    Proj_edge = sparse_eye(dim_tot, format="lil")

    def get_edge_index(j, axis, ext):
        multi_index = [None] * ndim
        multi_index[axis] = 0 if ext == -1 else Vh.spaces[1 - axis].spaces[axis].nbasis - 1
        multi_index[1 - axis] = j
        return l2g.get_index(0, 1 - axis, multi_index)

    def edge_moment_index(p, i, axis, ext):
        multi_index = [None] * ndim
        multi_index[1 - axis] = i
        multi_index[axis] = p + 1 if ext == -1 else Vh.spaces[1 - axis].spaces[axis].nbasis - 1 - p - 1
        return l2g.get_index(0, 1 - axis, multi_index)


    # boundary condition
    for bn in domain.boundary:

        if Vh.periodic[bn.axis]:
            continue

        axis = bn.axis
        d = 1 - axis
        ext = bn.ext
        space_1d = Vh.spaces[d].spaces[d]

        for i in range(0, space_1d.nbasis):
            ig = get_edge_index(i, axis, ext)
            Proj_edge[ig, ig] = 0

            for p in range(p_moments + 1):

                pg = edge_moment_index(p, i, axis, ext)
                Proj_edge[pg, ig] = gamma[d][p]

    return Proj_edge


# ===============================================================================

class ConformingProjectionV0(FemLinearOperator):
    """
    Conforming projection from global broken V0 space to conforming global V0 space
    Defined by averaging of interface (including vertex) dofs 
    and adding moment correction terms

    Parameters
    ----------
    V0h: <FemSpace>
     The discrete space
    
    mom_pres: <bool>
        If True, preserve polynomial moments of maximal order in the projection.

    p_moments: <int>
        Number of polynomial moments to be preserved in the projection.
        (Gets overwritten if the parameter mom_pres equals True)

    hom_bc : <bool>
     Apply homogenous boundary conditions if True
    """
    def __init__(
            self,
            V0h,
            mom_pres=False,
            p_moments=-1,
            hom_bc=False):
        
        if mom_pres:
            if V0h.is_multipatch:
                p_moments = max(p_moments, max(V0h.degree[0]))
            else:
                p_moments = max(p_moments, max(V0h.degree))

        FemLinearOperator.__init__(self, fem_domain=V0h, fem_codomain=V0h)
        
        if V0h.is_multipatch:
            sparse_matrix = construct_h1_conforming_projection(V0h, reg_orders=0, p_moments=p_moments, hom_bc=hom_bc)
        else:
            sparse_matrix = construct_h1_singlepatch_conforming_projection(V0h, reg_orders=0, p_moments=p_moments, hom_bc=hom_bc)

        self._linop = SparseMatrixLinearOperator(self.linop_domain, self.linop_codomain, sparse_matrix.tocsr())


class ConformingProjectionV1(FemLinearOperator):
    """
    Conforming projection from global broken V1 space to conforming V1 global space
    Defined by averaging of (only) interface dofs 
    and adding moment correction terms

    Parameters
    ----------
    V1h: <FemSpace>
     The discrete space

    mom_pres: <bool>
        If True, preserve polynomial moments of maximal order in the projection.

    p_moments: <int>
        Number of polynomial moments to be preserved in the projection.
        (Gets overwritten if the parameter mom_pres equals True)

    hom_bc : <bool>
     Apply homogenous boundary conditions if True
    """
    def __init__(
            self,
            V1h,
            mom_pres=False,
            p_moments=-1,
            hom_bc=False):

        if mom_pres:
            if V1h.is_multipatch:
                p_moments = max(p_moments, max(V1h.spaces[0].degree[0]))
            else:
                p_moments = max(p_moments, max(V1h.degree[0]))

        FemLinearOperator.__init__(self, fem_domain=V1h, fem_codomain=V1h)

        if V1h.is_multipatch:
            sparse_matrix = construct_hcurl_conforming_projection(V1h, reg_orders=0, p_moments=p_moments, hom_bc=hom_bc)
        else:
            sparse_matrix = construct_hcurl_singlepatch_conforming_projection(V1h, reg_orders=0, p_moments=p_moments, hom_bc=hom_bc)
        
        self._linop = SparseMatrixLinearOperator(self.linop_domain, self.linop_codomain, sparse_matrix.tocsr())
