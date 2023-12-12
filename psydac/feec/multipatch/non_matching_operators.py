import os
import numpy as np
from scipy.sparse import eye as sparse_eye
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv, norm

from sympde.topology import Derham, Square
from sympde.topology import IdentityMapping
from sympde.topology import Boundary, Interface, Union
from scipy.sparse.linalg                import norm as sp_norm

from psydac.feec.multipatch.utilities import time_count
from psydac.linalg.utilities import array_to_psydac
from psydac.feec.multipatch.api import discretize
from psydac.api.settings import PSYDAC_BACKENDS
from psydac.fem.splines import SplineSpace

from psydac.fem.basic import FemField
from psydac.feec.multipatch.plotting_utilities import plot_field

from sympde.topology import IdentityMapping, PolarMapping
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain, create_domain

from psydac.utilities.quadratures import gauss_legendre
from psydac.core.bsplines import breakpoints, quadrature_grid, basis_ders_on_quad_grid, find_spans, elements_spans
from copy import deepcopy


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


class Local2GlobalIndexMap:
    def __init__(self, ndim, n_patches, n_components):
        #        A[patch_index][component_index][i1,i2]
        self._shapes = [None]*n_patches
        self._ndofs = [None]*n_patches
        self._ndim = ndim
        self._n_patches = n_patches
        self._n_components = n_components

    def set_patch_shapes(self, patch_index, *shapes):
        assert len(shapes) == self._n_components
        assert all(len(s) == self._ndim for s in shapes)
        self._shapes[patch_index] = shapes
        self._ndofs[patch_index] = sum(np.product(s) for s in shapes)

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
        sizes = [np.product(s) for s in self._shapes[k][:d]]
        Ipc = np.ravel_multi_index(
            cartesian_index, dims=self._shapes[k][d], order='C')
        Ip = sum(sizes) + Ipc
        I = sum(self._ndofs[:k]) + Ip
        return I


def knots_to_insert(coarse_grid, fine_grid, tol=1e-14):
    #    assert len(coarse_grid)*2-2 == len(fine_grid)-1
    intersection = coarse_grid[(
        np.abs(fine_grid[:, None] - coarse_grid) < tol).any(0)]
    assert abs(intersection-coarse_grid).max() < tol
    T = fine_grid[~(np.abs(coarse_grid[:, None] - fine_grid) < tol).any(0)]
    return T


def construct_extension_operator_1D(domain, codomain):
    """

    compute the matrix of the extension operator on the interface space (1D space if global space is 2D)

    domain:     1d spline space on the interface (coarse grid)
    codomain:   1d spline space on the interface (fine grid)
    """
    #from psydac.core.interface import matrix_multi_stages
    from psydac.core.bsplines import hrefinement_matrix
    ops = []

    assert domain.ncells <= codomain.ncells

    Ts = knots_to_insert(domain.breaks, codomain.breaks)
    #P = matrix_multi_stages(Ts, domain.nbasis, domain.degree, domain.knots)
    P = hrefinement_matrix(Ts, domain.degree, domain.knots)
    if domain.basis == 'M':
        assert codomain.basis == 'M'
        P = np.diag(
            1/codomain._scaling_array) @ P @ np.diag(domain._scaling_array)

    return csr_matrix(P)  # kronecker of 1 term...

# Legacy code
# def construct_V0_conforming_projection(V0h, hom_bc=None):
#     dim_tot = V0h.nbasis
#     domain = V0h.symbolic_space.domain
#     ndim = 2
#     n_components = 1
#     n_patches = len(domain)

#     l2g = Local2GlobalIndexMap(ndim, len(domain), n_components)
#     for k in range(n_patches):
#         Vk = V0h.spaces[k]
#         # T is a TensorFemSpace and S is a 1D SplineSpace
#         shapes = [S.nbasis for S in Vk.spaces] 
#         l2g.set_patch_shapes(k, shapes)

#     Proj = sparse_eye(dim_tot, format="lil")
#     Proj_vertex = sparse_eye(dim_tot, format="lil")

#     Interfaces = domain.interfaces
#     if isinstance(Interfaces, Interface):
#         Interfaces = (Interfaces, )

#     corner_indices = set()
#     stored_indices = []
#     corners = get_corners(domain, False)
#     for (bd,co) in corners.items():

#         c = 0
#         indices = set()
#         for patch in co:
#             c += 1
#             multi_index_i = [None]*ndim

#             nbasis0 = V0h.spaces[patch].spaces[co[patch][0]].nbasis-1
#             nbasis1 = V0h.spaces[patch].spaces[co[patch][1]].nbasis-1

#             multi_index_i[0] = 0 if co[patch][0] == 0 else nbasis0
#             multi_index_i[1] = 0 if co[patch][1] == 0 else nbasis1
#             ig = l2g.get_index(patch, 0, multi_index_i)
#             indices.add(ig)


#             corner_indices.add(ig)
        
#         stored_indices.append(indices)
#         for j in indices: 
#             for i in indices:
#                 Proj_vertex[j,i] = 1/c

#     # First make all interfaces conforming
#     # We also touch the vertices here, but change them later again
#     for I in Interfaces:
        
#         axis = I.axis
#         direction = I.ornt

#         k_minus = get_patch_index_from_face(domain, I.minus)
#         k_plus = get_patch_index_from_face(domain, I.plus)
#         # logical directions normal to interface
#         minus_axis, plus_axis = I.minus.axis, I.plus.axis
#         # logical directions along the interface

#         #d_minus, d_plus = 1-minus_axis, 1-plus_axis
#         I_minus_ncells = V0h.spaces[k_minus].ncells
#         I_plus_ncells = V0h.spaces[k_plus].ncells

#         matching_interfaces = (I_minus_ncells == I_plus_ncells)

#         if I_minus_ncells <= I_plus_ncells:
#             k_fine, k_coarse = k_plus, k_minus
#             fine_axis, coarse_axis = I.plus.axis, I.minus.axis
#             fine_ext,  coarse_ext = I.plus.ext,  I.minus.ext

#         else:
#             k_fine, k_coarse = k_minus, k_plus
#             fine_axis, coarse_axis = I.minus.axis, I.plus.axis
#             fine_ext,  coarse_ext = I.minus.ext, I.plus.ext

#         d_fine = 1-fine_axis
#         d_coarse = 1-coarse_axis

#         space_fine = V0h.spaces[k_fine]
#         space_coarse = V0h.spaces[k_coarse]


#         coarse_space_1d = space_coarse.spaces[d_coarse]

#         fine_space_1d = space_fine.spaces[d_fine]
#         grid = np.linspace(
#             fine_space_1d.breaks[0], fine_space_1d.breaks[-1], coarse_space_1d.ncells+1)
#         coarse_space_1d_k_plus = SplineSpace(
#             degree=fine_space_1d.degree, grid=grid, basis=fine_space_1d.basis)

#         if not matching_interfaces:
#             E_1D = construct_extension_operator_1D(
#                 domain=coarse_space_1d_k_plus, codomain=fine_space_1d)
        
#             product = (E_1D.T) @ E_1D
#             R_1D = inv(product.tocsc()) @ E_1D.T
#             ER_1D = E_1D @ R_1D
#         else:
#             ER_1D = R_1D = E_1D = sparse_eye(
#                 fine_space_1d.nbasis, format="lil")

#         # P_k_minus_k_minus
#         multi_index = [None]*ndim
#         multi_index[coarse_axis] = 0 if coarse_ext == - \
#             1 else space_coarse.spaces[coarse_axis].nbasis-1
#         for i in range(coarse_space_1d.nbasis):
#             multi_index[d_coarse] = i
#             ig = l2g.get_index(k_coarse, 0, multi_index)
#             if not corner_indices.issuperset({ig}):
#                 Proj[ig, ig] = 0.5

#         # P_k_plus_k_plus
#         multi_index_i = [None]*ndim
#         multi_index_j = [None]*ndim
#         multi_index_i[fine_axis] = 0 if fine_ext == - \
#             1 else space_fine.spaces[fine_axis].nbasis-1
#         multi_index_j[fine_axis] = 0 if fine_ext == - \
#             1 else space_fine.spaces[fine_axis].nbasis-1

#         for i in range(fine_space_1d.nbasis):
#             multi_index_i[d_fine] = i
#             ig = l2g.get_index(k_fine, 0, multi_index_i)
#             for j in range(fine_space_1d.nbasis):
#                 multi_index_j[d_fine] = j
#                 jg = l2g.get_index(k_fine, 0, multi_index_j)
#                 if not corner_indices.issuperset({ig}):
#                     Proj[ig, jg] = 0.5*ER_1D[i, j]

#         # P_k_plus_k_minus
#         multi_index_i = [None]*ndim
#         multi_index_j = [None]*ndim
#         multi_index_i[fine_axis] = 0 if fine_ext == - \
#             1 else space_fine   .spaces[fine_axis]  .nbasis-1
#         multi_index_j[coarse_axis] = 0 if coarse_ext == - \
#             1 else space_coarse.spaces[coarse_axis].nbasis-1

#         for i in range(fine_space_1d.nbasis):
#             multi_index_i[d_fine] = i
#             ig = l2g.get_index(k_fine, 0, multi_index_i)
#             for j in range(coarse_space_1d.nbasis):
#                 multi_index_j[d_coarse] = j if direction == 1 else coarse_space_1d.nbasis-j-1
#                 jg = l2g.get_index(k_coarse, 0, multi_index_j)
#                 if not corner_indices.issuperset({ig}):
#                     Proj[ig, jg] = 0.5*E_1D[i, j]*direction

#         # P_k_minus_k_plus
#         multi_index_i = [None]*ndim
#         multi_index_j = [None]*ndim
#         multi_index_i[coarse_axis] = 0 if coarse_ext == - \
#             1 else space_coarse.spaces[coarse_axis].nbasis-1
#         multi_index_j[fine_axis] = 0 if fine_ext == - \
#             1 else space_fine .spaces[fine_axis]  .nbasis-1

#         for i in range(coarse_space_1d.nbasis):
#             multi_index_i[d_coarse] = i
#             ig = l2g.get_index(k_coarse, 0, multi_index_i)
#             for j in range(fine_space_1d.nbasis):
#                 multi_index_j[d_fine] = j if direction == 1 else fine_space_1d.nbasis-j-1
#                 jg = l2g.get_index(k_fine, 0, multi_index_j)
#                 if not corner_indices.issuperset({ig}):
#                     Proj[ig, jg] = 0.5*R_1D[i, j]*direction


#     if hom_bc:
#         bd_co_indices = set()
#         for bn in domain.boundary:
#             k = get_patch_index_from_face(domain, bn)
#             space_k = V0h.spaces[k]
#             axis = bn.axis
#             d = 1-axis
#             ext = bn.ext
#             space_k_1d = space_k.spaces[d]  # t
#             multi_index_i = [None]*ndim
#             multi_index_i[axis] = 0 if ext == - \
#                 1 else space_k.spaces[axis].nbasis-1

#             for i in range(space_k_1d.nbasis):
#                 multi_index_i[d] = i
#                 ig = l2g.get_index(k, 0, multi_index_i)
#                 bd_co_indices.add(ig)
#                 Proj[ig, ig] = 0

#         # properly ensure vertex continuity
#         for ig in bd_co_indices:
#             for jg in bd_co_indices:
#                 Proj_vertex[ig, jg] = 0
                    

#     return Proj @ Proj_vertex

# def construct_V1_conforming_projection(V1h, hom_bc=None):
#     dim_tot = V1h.nbasis
#     domain = V1h.symbolic_space.domain
#     ndim = 2
#     n_components = 2
#     n_patches = len(domain)

#     l2g = Local2GlobalIndexMap(ndim, len(domain), n_components)
#     for k in range(n_patches):
#         Vk = V1h.spaces[k]
#         # T is a TensorFemSpace and S is a 1D SplineSpace
#         shapes = [[S.nbasis for S in T.spaces] for T in Vk.spaces]
#         l2g.set_patch_shapes(k, *shapes)

#     Proj = sparse_eye(dim_tot, format="lil")

#     Interfaces = domain.interfaces
#     if isinstance(Interfaces, Interface):
#         Interfaces = (Interfaces, )

#     for I in Interfaces:
#         axis = I.axis
#         direction = I.ornt

#         k_minus = get_patch_index_from_face(domain, I.minus)
#         k_plus = get_patch_index_from_face(domain, I.plus)
#         # logical directions normal to interface
#         minus_axis, plus_axis = I.minus.axis, I.plus.axis
#         # logical directions along the interface
#         d_minus, d_plus = 1-minus_axis, 1-plus_axis
#         I_minus_ncells = V1h.spaces[k_minus].spaces[d_minus].ncells[d_minus]
#         I_plus_ncells = V1h.spaces[k_plus] .spaces[d_plus] .ncells[d_plus]

#         matching_interfaces = (I_minus_ncells == I_plus_ncells)

#         if I_minus_ncells <= I_plus_ncells:
#             k_fine, k_coarse = k_plus, k_minus
#             fine_axis, coarse_axis = I.plus.axis, I.minus.axis
#             fine_ext,  coarse_ext = I.plus.ext,  I.minus.ext

#         else:
#             k_fine, k_coarse = k_minus, k_plus
#             fine_axis, coarse_axis = I.minus.axis, I.plus.axis
#             fine_ext,  coarse_ext = I.minus.ext, I.plus.ext

#         d_fine = 1-fine_axis
#         d_coarse = 1-coarse_axis

#         space_fine = V1h.spaces[k_fine]
#         space_coarse = V1h.spaces[k_coarse]

#         #print("coarse = \n", space_coarse.spaces[d_coarse])
#         #print("coarse 2 = \n", space_coarse.spaces[d_coarse].spaces[d_coarse])
#         # todo: merge with first test above
#         coarse_space_1d = space_coarse.spaces[d_coarse].spaces[d_coarse]

#         #print("fine = \n", space_fine.spaces[d_fine])
#         #print("fine 2 = \n", space_fine.spaces[d_fine].spaces[d_fine])

#         fine_space_1d = space_fine.spaces[d_fine].spaces[d_fine]
#         grid = np.linspace(
#             fine_space_1d.breaks[0], fine_space_1d.breaks[-1], coarse_space_1d.ncells+1)
#         coarse_space_1d_k_plus = SplineSpace(
#             degree=fine_space_1d.degree, grid=grid, basis=fine_space_1d.basis)

#         if not matching_interfaces:
#             E_1D = construct_extension_operator_1D(
#                 domain=coarse_space_1d_k_plus, codomain=fine_space_1d)
#             product = (E_1D.T) @ E_1D
#             R_1D = inv(product.tocsc()) @ E_1D.T
#             ER_1D = E_1D @ R_1D
#         else:
#             ER_1D = R_1D = E_1D = sparse_eye(
#                 fine_space_1d.nbasis, format="lil")

#         # P_k_minus_k_minus
#         multi_index = [None]*ndim
#         multi_index[coarse_axis] = 0 if coarse_ext == - \
#             1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1
#         for i in range(coarse_space_1d.nbasis):
#             multi_index[d_coarse] = i
#             ig = l2g.get_index(k_coarse, d_coarse, multi_index)
#             Proj[ig, ig] = 0.5

#         # P_k_plus_k_plus
#         multi_index_i = [None]*ndim
#         multi_index_j = [None]*ndim
#         multi_index_i[fine_axis] = 0 if fine_ext == - \
#             1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1
#         multi_index_j[fine_axis] = 0 if fine_ext == - \
#             1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1

#         for i in range(fine_space_1d.nbasis):
#             multi_index_i[d_fine] = i
#             ig = l2g.get_index(k_fine, d_fine, multi_index_i)
#             for j in range(fine_space_1d.nbasis):
#                 multi_index_j[d_fine] = j
#                 jg = l2g.get_index(k_fine, d_fine, multi_index_j)
#                 Proj[ig, jg] = 0.5*ER_1D[i, j]

#         # P_k_plus_k_minus
#         multi_index_i = [None]*ndim
#         multi_index_j = [None]*ndim
#         multi_index_i[fine_axis] = 0 if fine_ext == - \
#             1 else space_fine  .spaces[d_fine]  .spaces[fine_axis]  .nbasis-1
#         multi_index_j[coarse_axis] = 0 if coarse_ext == - \
#             1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1

#         for i in range(fine_space_1d.nbasis):
#             multi_index_i[d_fine] = i
#             ig = l2g.get_index(k_fine, d_fine, multi_index_i)
#             for j in range(coarse_space_1d.nbasis):
#                 multi_index_j[d_coarse] = j if direction == 1 else coarse_space_1d.nbasis-j-1
#                 jg = l2g.get_index(k_coarse, d_coarse, multi_index_j)
#                 Proj[ig, jg] = 0.5*E_1D[i, j]*direction

#         # P_k_minus_k_plus
#         multi_index_i = [None]*ndim
#         multi_index_j = [None]*ndim
#         multi_index_i[coarse_axis] = 0 if coarse_ext == - \
#             1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1
#         multi_index_j[fine_axis] = 0 if fine_ext == - \
#             1 else space_fine  .spaces[d_fine]  .spaces[fine_axis]  .nbasis-1

#         for i in range(coarse_space_1d.nbasis):
#             multi_index_i[d_coarse] = i
#             ig = l2g.get_index(k_coarse, d_coarse, multi_index_i)
#             for j in range(fine_space_1d.nbasis):
#                 multi_index_j[d_fine] = j if direction == 1 else fine_space_1d.nbasis-j-1
#                 jg = l2g.get_index(k_fine, d_fine, multi_index_j)
#                 Proj[ig, jg] = 0.5*R_1D[i, j]*direction

#     if hom_bc:
#         for bn in domain.boundary:
#             k = get_patch_index_from_face(domain, bn)
#             space_k = V1h.spaces[k]
#             axis = bn.axis
#             d = 1-axis
#             ext = bn.ext
#             space_k_1d = space_k.spaces[d].spaces[d]  # t
#             multi_index_i = [None]*ndim
#             multi_index_i[axis] = 0 if ext == - \
#                 1 else space_k.spaces[d].spaces[axis].nbasis-1

#             for i in range(space_k_1d.nbasis):
#                 multi_index_i[d] = i
#                 ig = l2g.get_index(k, d, multi_index_i)
#                 Proj[ig, ig] = 0

#     return Proj


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

    # corner_data[corner] = (patch_ind => local coordinates)
    corner_data = dict()

    if boundary_only:
        for co in cos:
            
            corner_data[co] = dict()
            c = 0
            for cb in co.corners:
                axis = set()
                #check if corner boundary is part of the domain boundary
                for cbbd in cb.args: 
                    if bd.has(cbbd): 
                        axis.add(cbbd.axis)
                        c += 1

                p_ind = patches.index(cb.domain)
                c_coord = cb.coordinates
                corner_data[co][p_ind] = (c_coord, axis)
            
            if c == 0: corner_data.pop(co)

    else:
        for co in cos:
            corner_data[co] = dict()

            for cb in co.corners:
                p_ind = patches.index(cb.domain)
                c_coord = cb.coordinates
                corner_data[co][p_ind] = c_coord 

    return corner_data


def construct_scalar_conforming_projection(Vh, reg_orders=(0,0), p_moments=(-1,-1), nquads=None, hom_bc=(False, False)):
    #construct conforming projection for a 2-dimensional scalar space 

    dim_tot = Vh.nbasis


    # fully discontinuous space
    if reg_orders[0] < 0 and reg_orders[1] < 0:
        return sparse_eye(dim_tot, format="lil")

    
    # moment corrections perpendicular to interfaces
    # a_sm, a_nb, b_sm, b_nb, Correct_coef_bnd, cc_0_ax
    cor_x = get_scalar_moment_correction(Vh.spaces[0], 0, reg_orders[0], p_moments[0], nquads, hom_bc[0])
    cor_y = get_scalar_moment_correction(Vh.spaces[0], 1, reg_orders[1], p_moments[1], nquads, hom_bc[1])
    corrections = [cor_x, cor_y]
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

    
    # vertex correction matrix
    Proj_vertex = sparse_eye(dim_tot, format="lil")

    # edge correction matrix
    Proj_edge = sparse_eye(dim_tot, format="lil")

    Interfaces = domain.interfaces
    if isinstance(Interfaces, Interface):
        Interfaces = (Interfaces, )

    corner_indices = set()
    corners = get_corners(domain, False)


    #loop over all vertices
    for (bd,co) in corners.items():

        # len(co) is the number of adjacent patches at a vertex
        corr = len(co)
        for patch1 in co:

            #local vertex coordinates in patch1
            coords1 = co[patch1]
            nbasis01 = Vh.spaces[patch1].spaces[coords1[0]].nbasis-1
            nbasis11 = Vh.spaces[patch1].spaces[coords1[1]].nbasis-1

            #patch local index
            multi_index_i = [None]*ndim
            multi_index_i[0] = 0 if coords1[0] == 0 else nbasis01
            multi_index_i[1] = 0 if coords1[1] == 0 else nbasis11

            #global index
            ig = l2g.get_index(patch1, 0, multi_index_i)
            corner_indices.add(ig)

            for patch2 in co:
                
                # local vertex coordinates in patch2
                coords2 = co[patch2]
                nbasis02 = Vh.spaces[patch2].spaces[coords2[0]].nbasis-1
                nbasis12 = Vh.spaces[patch2].spaces[coords2[1]].nbasis-1

                #patch local index
                multi_index_j = [None]*ndim
                multi_index_j[0] = 0 if coords2[0] == 0 else nbasis02
                multi_index_j[1] = 0 if coords2[1] == 0 else nbasis12

                #global index
                jg = l2g.get_index(patch2, 0, multi_index_j)

                #conformity constraint
                Proj_vertex[jg,ig] = 1/corr

                if patch1 == patch2: continue

                if (p_moments[0] == -1 and p_moments[1] == -1): continue 

                #moment corrections from patch1 to patch2
                axis = 0 
                d = 1 
                multi_index_p = [None]*ndim
                for pd in range(0, max(1, p_moments[d]+1)):
                    p_indd = pd+0+1
                    multi_index_p[d] = p_indd if coords2[d] == 0 else Vh.spaces[patch2].spaces[coords2[d]].nbasis-1-p_indd

                    for p in range(0, max(1,p_moments[axis]+1)):

                        p_ind = p+0+1 # 0 = regularity
                        multi_index_p[axis] = p_ind if coords2[axis] == 0 else Vh.spaces[patch2].spaces[coords2[axis]].nbasis-1-p_ind
                        pg = l2g.get_index(patch2, 0, multi_index_p)
                        Proj_vertex[pg, ig] += - 1/corr * corrections[axis][5][p] * corrections[d][5][pd]

            if (p_moments[0] == -1 and p_moments[1]) == -1: continue 

            #moment corrections from patch1 to patch1
            axis = 0
            d = 1 
            multi_index_p = [None]*ndim
            for pd in range(0, max(1, p_moments[d]+1)):
                p_indd = pd+0+1
                multi_index_p[d] = p_indd if coords1[d] == 0 else Vh.spaces[patch1].spaces[coords1[d]].nbasis-1-p_indd
                for p in range(0, max(1, p_moments[axis]+1)):

                    p_ind = p+0+1 # 0 = regularity
                    multi_index_p[axis] = p_ind if coords1[axis] == 0 else Vh.spaces[patch1].spaces[coords1[axis]].nbasis-1-p_ind
                    pg = l2g.get_index(patch1, 0, multi_index_p)
                    Proj_vertex[pg,ig] += (1-1/corr) * corrections[axis][5][p] * corrections[d][5][pd]
                    

    # loop over all interfaces
    for I in Interfaces:
        
        axis = I.axis
        direction = I.ornt

        k_minus = get_patch_index_from_face(domain, I.minus)
        k_plus = get_patch_index_from_face(domain, I.plus)

        I_minus_ncells = Vh.spaces[k_minus].ncells
        I_plus_ncells = Vh.spaces[k_plus].ncells

        matching_interfaces = (I_minus_ncells == I_plus_ncells)

        # logical directions normal to interface
        if I_minus_ncells <= I_plus_ncells:
            k_fine, k_coarse = k_plus, k_minus
            fine_axis, coarse_axis = I.plus.axis, I.minus.axis
            fine_ext,  coarse_ext = I.plus.ext,  I.minus.ext

        else:
            k_fine, k_coarse = k_minus, k_plus
            fine_axis, coarse_axis = I.minus.axis, I.plus.axis
            fine_ext,  coarse_ext = I.minus.ext, I.plus.ext

        # logical directions along the interface
        d_fine = 1-fine_axis
        d_coarse = 1-coarse_axis

        space_fine = Vh.spaces[k_fine]
        space_coarse = Vh.spaces[k_coarse]


        coarse_space_1d = space_coarse.spaces[d_coarse]
        fine_space_1d = space_fine.spaces[d_fine]

        E_1D, R_1D, ER_1D = get_moment_pres_scalar_extension_restriction(matching_interfaces, coarse_space_1d, fine_space_1d, 'B')

        # P_k_minus_k_minus
        multi_index = [None]*ndim
        multi_index_m = [None]*ndim
        multi_index[coarse_axis] = 0 if coarse_ext == - 1 else space_coarse.spaces[coarse_axis].nbasis-1


        for i in range(coarse_space_1d.nbasis):
            multi_index[d_coarse] = i
            multi_index_m[d_coarse] = i
            ig = l2g.get_index(k_coarse, 0, multi_index)

            if not corner_indices.issuperset({ig}):
                Proj_edge[ig, ig] = corrections[coarse_axis][0][0] 

                for p in range(0, p_moments[coarse_axis]+1):

                    p_ind = p+0+1 # 0 = regularity
                    multi_index_m[coarse_axis] = p_ind if coarse_ext == - 1 else space_coarse.spaces[coarse_axis].nbasis-1-p_ind
                    mg = l2g.get_index(k_coarse, 0, multi_index_m)

                    Proj_edge[mg, ig] += corrections[coarse_axis][0][p_ind]
                        
        # P_k_plus_k_plus
        multi_index_i = [None]*ndim
        multi_index_j = [None]*ndim
        multi_index_p = [None]*ndim

        multi_index_i[fine_axis] = 0 if fine_ext == - 1 else space_fine.spaces[fine_axis].nbasis-1
        multi_index_j[fine_axis] = 0 if fine_ext == - 1 else space_fine.spaces[fine_axis].nbasis-1

        for i in range(fine_space_1d.nbasis):
            multi_index_i[d_fine] = i
            ig = l2g.get_index(k_fine, 0, multi_index_i)
            
            multi_index_p[d_fine] = i

            for j in range(fine_space_1d.nbasis):
                multi_index_j[d_fine] = j
                jg = l2g.get_index(k_fine, 0, multi_index_j)

                if not corner_indices.issuperset({ig}):
                    Proj_edge[ig, jg] = corrections[fine_axis][0][0] * ER_1D[i,j]

                    for p in range(0, p_moments[fine_axis]+1):

                        p_ind = p+0+1 # 0 = regularity
                        multi_index_p[fine_axis] = p_ind if fine_ext == - 1 else space_fine.spaces[fine_axis].nbasis-1-p_ind
                        pg = l2g.get_index(k_fine, 0, multi_index_p)

                        Proj_edge[pg, jg] += corrections[fine_axis][0][p_ind] * ER_1D[i, j]

        # P_k_plus_k_minus
        multi_index_i = [None]*ndim
        multi_index_j = [None]*ndim
        multi_index_p = [None]*ndim

        multi_index_i[fine_axis] = 0 if fine_ext == -1 else space_fine   .spaces[fine_axis]  .nbasis-1
        multi_index_j[coarse_axis] = 0 if coarse_ext == -1 else space_coarse.spaces[coarse_axis].nbasis-1

        for i in range(fine_space_1d.nbasis):
            multi_index_i[d_fine] = i
            multi_index_p[d_fine] = i
            ig = l2g.get_index(k_fine, 0, multi_index_i)

            for j in range(coarse_space_1d.nbasis):
                multi_index_j[d_coarse] = j if direction == 1 else coarse_space_1d.nbasis-j-1
                jg = l2g.get_index(k_coarse, 0, multi_index_j)

                if not corner_indices.issuperset({ig}):
                    Proj_edge[ig, jg] = corrections[coarse_axis][1][0] *E_1D[i,j]*direction

                    for p in range(0, p_moments[fine_axis]+1):

                        p_ind = p+0+1 # 0 = regularity
                        multi_index_p[fine_axis] = p_ind if fine_ext == - 1 else space_fine.spaces[fine_axis].nbasis-1-p_ind
                        pg = l2g.get_index(k_fine, 0, multi_index_p)
                        
                        Proj_edge[pg, jg] += corrections[fine_axis][1][p_ind] *E_1D[i, j]*direction

        # P_k_minus_k_plus
        multi_index_i = [None]*ndim
        multi_index_j = [None]*ndim
        multi_index_p = [None]*ndim

        multi_index_i[coarse_axis] = 0 if coarse_ext == -1 else space_coarse.spaces[coarse_axis].nbasis-1
        multi_index_j[fine_axis] = 0 if fine_ext == -1 else space_fine .spaces[fine_axis]  .nbasis-1

        for i in range(coarse_space_1d.nbasis):
            multi_index_i[d_coarse] = i
            multi_index_p[d_coarse] = i
            ig = l2g.get_index(k_coarse, 0, multi_index_i)

            for j in range(fine_space_1d.nbasis):
                multi_index_j[d_fine] = j if direction == 1 else fine_space_1d.nbasis-j-1
                jg = l2g.get_index(k_fine, 0, multi_index_j)

                if not corner_indices.issuperset({ig}):
                    Proj_edge[ig, jg] = corrections[fine_axis][1][0] *R_1D[i,j]*direction

                    for p in range(0, p_moments[coarse_axis]+1):

                        p_ind = p+0+1 # 0 = regularity
                        multi_index_p[coarse_axis] = p_ind if coarse_ext == - 1 else space_coarse.spaces[coarse_axis].nbasis-1-p_ind
                        pg = l2g.get_index(k_coarse, 0, multi_index_p)

                        Proj_edge[pg, jg] += corrections[coarse_axis][1][p_ind] *R_1D[i, j]*direction

    # boundary conditions

    # interface correction
    bd_co_indices = set()
    for bn in domain.boundary:
        k = get_patch_index_from_face(domain, bn)
        space_k = Vh.spaces[k]
        axis = bn.axis
        if not hom_bc[axis]:
            continue 

        d = 1-axis
        ext = bn.ext
        space_k_1d = space_k.spaces[d]  # t
        multi_index_i = [None]*ndim
        multi_index_i[axis] = 0 if ext == - \
            1 else space_k.spaces[axis].nbasis-1

        multi_index_p = [None]*ndim
        multi_index_p[axis] = 0 if ext == - \
            1 else space_k.spaces[axis].nbasis-1

        for i in range(0, space_k_1d.nbasis):
            multi_index_i[d] = i
            ig = l2g.get_index(k, 0, multi_index_i)
            bd_co_indices.add(ig)
            Proj_edge[ig, ig] = 0

            multi_index_p[d] = i

            # interface correction
            if (i != 0 and i != space_k_1d.nbasis-1):
                for p in range(0, p_moments[axis]+1):

                    p_ind = p+0+1 # 0 = regularity
                    multi_index_p[axis] = p_ind if ext == - 1 else space_k.spaces[axis].nbasis-1-p_ind
                    pg = l2g.get_index(k, 0, multi_index_p)
                    #a_sm, a_nb, b_sm, b_nb, Correct_coef_bnd
                    Proj_edge[pg, ig] = corrections[axis][4][p] #* corrections[d][4][p]

                
    # vertex corrections 
    corners = get_corners(domain, True)
    for (bd,co) in corners.items():
        
        # len(co) is the number of adjacent patches at a vertex
        corr = len(co)
        for patch1 in co:
            c = 0
            if hom_bc[0]: 
                if 0 in co[patch1][1]: c += 1 
            if hom_bc[1]: 
                if 1 in co[patch1][1]: c+=1
            if c == 0: break

            #local vertex coordinates in patch1
            coords1 = co[patch1][0]
            nbasis01 = Vh.spaces[patch1].spaces[coords1[0]].nbasis-1
            nbasis11 = Vh.spaces[patch1].spaces[coords1[1]].nbasis-1

            #patch local index
            multi_index_i = [None]*ndim
            multi_index_i[0] = 0 if coords1[0] == 0 else nbasis01
            multi_index_i[1] = 0 if coords1[1] == 0 else nbasis11

            #global index
            ig = l2g.get_index(patch1, 0, multi_index_i)
            corner_indices.add(ig)

            for patch2 in co:
                
                # local vertex coordinates in patch2
                coords2 = co[patch2][0]
                nbasis02 = Vh.spaces[patch2].spaces[coords2[0]].nbasis-1
                nbasis12 = Vh.spaces[patch2].spaces[coords2[1]].nbasis-1

                #patch local index
                multi_index_j = [None]*ndim
                multi_index_j[0] = 0 if coords2[0] == 0 else nbasis02
                multi_index_j[1] = 0 if coords2[1] == 0 else nbasis12

                #global index
                jg = l2g.get_index(patch2, 0, multi_index_j)

                #conformity constraint
                Proj_vertex[jg,ig] = 0

                if patch1 == patch2: continue

                if (p_moments[0] == -1 and p_moments[1] == -1): continue 

                #moment corrections from patch1 to patch2
                axis = 0 
                d = 1 
                multi_index_p = [None]*ndim
                for pd in range(0, max(1, p_moments[d]+1)):
                    p_indd = pd+0+1
                    multi_index_p[d] = p_indd if coords2[d] == 0 else Vh.spaces[patch2].spaces[coords2[d]].nbasis-1-p_indd

                    for p in range(0, max(1,p_moments[axis]+1)):

                        p_ind = p+0+1 # 0 = regularity
                        multi_index_p[axis] = p_ind if coords2[axis] == 0 else Vh.spaces[patch2].spaces[coords2[axis]].nbasis-1-p_ind
                        pg = l2g.get_index(patch2, 0, multi_index_p)
                        Proj_vertex[pg, ig] = 0

            if (p_moments[0] == -1 and p_moments[1]) == -1: continue 

            #moment corrections from patch1 to patch1
            axis = 0
            d = 1 
            multi_index_p = [None]*ndim
            for pd in range(0, max(1, p_moments[d]+1)):
                p_indd = pd+0+1
                multi_index_p[d] = p_indd if coords1[d] == 0 else Vh.spaces[patch1].spaces[coords1[d]].nbasis-1-p_indd
                for p in range(0, max(1, p_moments[axis]+1)):

                    p_ind = p+0+1 # 0 = regularity
                    multi_index_p[axis] = p_ind if coords1[axis] == 0 else Vh.spaces[patch1].spaces[coords1[axis]].nbasis-1-p_ind
                    pg = l2g.get_index(patch1, 0, multi_index_p)
                    Proj_vertex[pg,ig] = corrections[axis][5][p] * corrections[d][5][pd]

    return Proj_edge @ Proj_vertex

def construct_vector_conforming_projection(Vh, reg_orders= (0,0), p_moments=(-1,-1), nquads=None, hom_bc=(False, False)):
    dim_tot = Vh.nbasis

    # fully discontinuous space
    if reg_orders[0] < 0 and reg_orders[1] < 0:
        return sparse_eye(dim_tot, format="lil")

    #moment corrections
    corrections_0 =  get_vector_moment_correction(Vh.spaces[0], 0, 0, reg=reg_orders[0], p_moments=p_moments[0], nquads=nquads, hom_bc=hom_bc[0])
    corrections_1 = get_vector_moment_correction(Vh.spaces[0], 0, 1, reg=reg_orders[1], p_moments=p_moments[1], nquads=nquads, hom_bc=hom_bc[1])

    corrections_00 =  get_vector_moment_correction(Vh.spaces[0], 1, 0, reg=reg_orders[0], p_moments=p_moments[0], nquads=nquads, hom_bc=hom_bc[0])
    corrections_11 = get_vector_moment_correction(Vh.spaces[0], 1, 1, reg=reg_orders[1], p_moments=p_moments[1], nquads=nquads, hom_bc=hom_bc[1])

    corrections = [[corrections_0, corrections_1], [corrections_00, corrections_11]]

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

    Proj = sparse_eye(dim_tot, format="lil")

    Interfaces = domain.interfaces
    if isinstance(Interfaces, Interface):
        Interfaces = (Interfaces, )

    for I in Interfaces:
        axis = I.axis
        direction = I.ornt

        k_minus = get_patch_index_from_face(domain, I.minus)
        k_plus = get_patch_index_from_face(domain, I.plus)
        # logical directions normal to interface
        minus_axis, plus_axis = I.minus.axis, I.plus.axis
        # logical directions along the interface
        d_minus, d_plus = 1-minus_axis, 1-plus_axis
        I_minus_ncells = Vh.spaces[k_minus].spaces[d_minus].ncells[d_minus]
        I_plus_ncells = Vh.spaces[k_plus] .spaces[d_plus] .ncells[d_plus]

        matching_interfaces = (I_minus_ncells == I_plus_ncells)

        if I_minus_ncells <= I_plus_ncells:
            k_fine, k_coarse = k_plus, k_minus
            fine_axis, coarse_axis = I.plus.axis, I.minus.axis
            fine_ext,  coarse_ext = I.plus.ext,  I.minus.ext

        else:
            k_fine, k_coarse = k_minus, k_plus
            fine_axis, coarse_axis = I.minus.axis, I.plus.axis
            fine_ext,  coarse_ext = I.minus.ext, I.plus.ext

        d_fine = 1-fine_axis
        d_coarse = 1-coarse_axis

        space_fine = Vh.spaces[k_fine]
        space_coarse = Vh.spaces[k_coarse]

        coarse_space_1d = space_coarse.spaces[d_coarse].spaces[d_coarse]
        fine_space_1d = space_fine.spaces[d_fine].spaces[d_fine]

        E_1D, R_1D, ER_1D = get_moment_pres_scalar_extension_restriction(matching_interfaces, coarse_space_1d, fine_space_1d, 'M') 

        # P_k_minus_k_minus
        multi_index = [None]*ndim
        multi_index_m = [None]*ndim
        multi_index[coarse_axis] = 0 if coarse_ext == - \
            1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1
            
        for i in range(coarse_space_1d.nbasis):
            multi_index[d_coarse] = i
            multi_index_m[d_coarse] = i
            ig = l2g.get_index(k_coarse, d_coarse, multi_index)
            Proj[ig, ig] = corrections[d_coarse][coarse_axis][0][0] 

            for p in range(0, p_moments[coarse_axis]+1):

                p_ind = p+0+1 # 0 = regularity
                multi_index_m[coarse_axis] = p_ind if coarse_ext == - 1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1-p_ind
                mg = l2g.get_index(k_coarse, d_coarse, multi_index_m)

                Proj[mg, ig] = corrections[d_coarse][coarse_axis][0][p_ind]

        # P_k_plus_k_plus
        multi_index_i = [None]*ndim
        multi_index_j = [None]*ndim
        multi_index_p = [None]*ndim
        multi_index_i[fine_axis] = 0 if fine_ext == - \
            1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1
        multi_index_j[fine_axis] = 0 if fine_ext == - \
            1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1

        for i in range(fine_space_1d.nbasis):
            multi_index_i[d_fine] = i
            multi_index_p[d_fine] = i
            ig = l2g.get_index(k_fine, d_fine, multi_index_i)

            for j in range(fine_space_1d.nbasis):
                multi_index_j[d_fine] = j
                jg = l2g.get_index(k_fine, d_fine, multi_index_j)
                Proj[ig, jg] = corrections[d_fine][fine_axis][0][0] * ER_1D[i, j]

                for p in range(0, p_moments[fine_axis]+1):

                    p_ind = p+0+1 # 0 = regularity
                    multi_index_p[fine_axis] = p_ind if fine_ext == - 1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1-p_ind
                    pg = l2g.get_index(k_fine, d_fine, multi_index_p)

                    Proj[pg, jg] = corrections[d_fine][fine_axis][0][p_ind] * ER_1D[i, j]

        # P_k_plus_k_minus
        multi_index_i = [None]*ndim
        multi_index_j = [None]*ndim
        multi_index_p = [None]*ndim
        multi_index_i[fine_axis] = 0 if fine_ext == - \
            1 else space_fine  .spaces[d_fine]  .spaces[fine_axis]  .nbasis-1
        multi_index_j[coarse_axis] = 0 if coarse_ext == - \
            1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1

        for i in range(fine_space_1d.nbasis):
            multi_index_i[d_fine] = i
            multi_index_p[d_fine] = i
            ig = l2g.get_index(k_fine, d_fine, multi_index_i)

            for j in range(coarse_space_1d.nbasis):
                multi_index_j[d_coarse] = j if direction == 1 else coarse_space_1d.nbasis-j-1
                jg = l2g.get_index(k_coarse, d_coarse, multi_index_j)
                Proj[ig, jg] = corrections[d_fine][fine_axis][1][0] *E_1D[i, j]*direction

                for p in range(0, p_moments[fine_axis]+1):

                    p_ind = p+0+1 # 0 = regularity
                    multi_index_p[fine_axis] = p_ind if fine_ext == - 1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1-p_ind
                    pg = l2g.get_index(k_fine, d_fine, multi_index_p)

                    Proj[pg, jg] = corrections[d_fine][fine_axis][1][p_ind] *E_1D[i, j]*direction

        # P_k_minus_k_plus
        multi_index_i = [None]*ndim
        multi_index_j = [None]*ndim
        multi_index_p = [None]*ndim
        multi_index_i[coarse_axis] = 0 if coarse_ext == - \
            1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1
        multi_index_j[fine_axis] = 0 if fine_ext == - \
            1 else space_fine  .spaces[d_fine]  .spaces[fine_axis]  .nbasis-1

        for i in range(coarse_space_1d.nbasis):
            multi_index_i[d_coarse] = i
            multi_index_p[d_coarse] = i
            ig = l2g.get_index(k_coarse, d_coarse, multi_index_i)
            for j in range(fine_space_1d.nbasis):
                multi_index_j[d_fine] = j if direction == 1 else fine_space_1d.nbasis-j-1
                jg = l2g.get_index(k_fine, d_fine, multi_index_j)
                Proj[ig, jg] = corrections[d_coarse][coarse_axis][1][0] *R_1D[i, j]*direction
                
                for p in range(0, p_moments[coarse_axis]+1):

                    p_ind = p+0+1 # 0 = regularity
                    multi_index_p[coarse_axis] = p_ind if coarse_ext == - 1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1-p_ind
                    pg = l2g.get_index(k_coarse, d_coarse, multi_index_p)

                    Proj[pg, jg] = corrections[d_coarse][coarse_axis][1][p_ind] *R_1D[i, j]*direction

    #if hom_bc:
    for bn in domain.boundary:
        k = get_patch_index_from_face(domain, bn)
        space_k = Vh.spaces[k]
        axis = bn.axis
        d = 1-axis
        ext = bn.ext

        if not hom_bc[axis]:
            continue 

        space_k_1d = space_k.spaces[d].spaces[d]  # t
        multi_index_i = [None]*ndim
        multi_index_i[axis] = 0 if ext == - \
            1 else space_k.spaces[d].spaces[axis].nbasis-1
        multi_index_p = [None]*ndim

        for i in range(space_k_1d.nbasis):
            multi_index_i[d] = i
            multi_index_p[d] = i
            ig = l2g.get_index(k, d, multi_index_i)
            Proj[ig, ig] = 0

            for p in range(0, p_moments[axis]+1):

                p_ind = p+0+1 # 0 = regularity
                multi_index_p[axis] = p_ind if ext == - 1 else space_k.spaces[d].spaces[axis].nbasis-1-p_ind
                pg = l2g.get_index(k, d, multi_index_p)
                #a_sm, a_nb, b_sm, b_nb, Correct_coef_bnd

                Proj[pg, ig] = corrections[d][axis][4][p]

    return Proj


def get_scalar_moment_correction(patch_space, conf_axis, reg=0, p_moments=-1, nquads=None, hom_bc=False):

    proj_op = 0
    #patch_space = Vh.spaces[0]
    local_shape = [patch_space.spaces[0].nbasis,patch_space.spaces[1].nbasis]
    Nel    = patch_space.ncells                     # number of elements
    degree = patch_space.degree
    breakpoints_xy = [breakpoints(patch_space.knots[axis],degree[axis]) for axis in range(2)]

    if nquads is None:
        # default: Gauss-Legendre quadratures should be exact for polynomials of deg ≤ 2*degree
        nquads = [ degree[axis]+1 for axis in range(2)]

    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]

    grid = [np.array([deepcopy((0.5*(u[axis]+1)*(breakpoints_xy[axis][i+1]-breakpoints_xy[axis][i])+breakpoints_xy[axis][i])) 
                      for i in range(Nel[axis])])
            for axis in range(2)]
    _, basis, span, _ = patch_space.preprocess_regular_tensor_grid(grid,der=1)  # todo: why not der=0 ?

    span = [deepcopy(span[k] + patch_space.vector_space.starts[k] - patch_space.vector_space.shifts[k] * patch_space.vector_space.pads[k]) for k in range(2)]
    p_axis = degree[conf_axis]
    enddom = breakpoints_xy[conf_axis][-1]
    begdom = breakpoints_xy[conf_axis][0]
    denom = enddom-begdom

    a_sm = np.zeros(p_moments+2+reg)   # coefs of P B0 on same patch
    a_nb = np.zeros(p_moments+2+reg)   # coefs of P B0 on neighbor patch
    b_sm = np.zeros(p_moments+3)   # coefs of P B1 on same patch
    b_nb = np.zeros(p_moments+3)   # coefs of P B1 on neighbor patch
    Correct_coef_bnd = np.zeros(p_moments+1)
    Correct_coef_0 = np.zeros(p_moments+2+reg)

    if reg >= 0:
        # projection coefs:
        a_sm[0] = 1/2
        a_nb[0] = a_sm[0]
    if reg == 1:

        if proj_op == 0:
            # new slope is average of old ones
            a_sm[1] = 0  
        elif proj_op == 1:
            # new slope is average of old ones after averaging of interface coef
            a_sm[1] = 1/2
        elif proj_op == 2:
            # new slope is average of reconstructed ones using local values and slopes
            a_sm[1] = 1/(2*p_axis)
        else:
            # just to try something else
            a_sm[1] = proj_op/2

        a_nb[1] = 2*a_sm[0] - a_sm[1]
        b_sm[0] = 0
        b_sm[1] = 1/2
        b_nb[0] = b_sm[0]
        b_nb[1] = 2*b_sm[0] - b_sm[1]
    
    if p_moments >= 0:
        # to preserve moments of degree p we need 1+p conforming basis functions in the patch (the "interior" ones)
        # and for the given regularity constraint, there are local_shape[conf_axis]-2*(1+reg) such conforming functions 
        p_max = local_shape[conf_axis]-2*(1+reg) - 1
        if p_max < p_moments:
            print( " ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            print( " **         WARNING -- WARNING -- WARNING ")
            print(f" ** conf. projection imposing C{reg} smoothness on scalar space along axis {conf_axis}:")            
            print(f" ** there are not enough dofs in a patch to preserve moments of degree {p_moments} !")
            print(f" ** Only able to preserve up to degree --> {p_max} <-- ")
            print( " ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            p_moments = p_max

        # computing the contribution to every moment of the differents basis function
        # for simplicity we assemble the full matrix with all basis functions (ok if patches not too large)
        Mass_mat = np.zeros((p_moments+1,local_shape[conf_axis]))    
        for poldeg in range(p_moments+1):
            for ie1 in range(Nel[conf_axis]):   #loop on cells
                for il1 in range(p_axis+1): #loops on basis function in each cell
                    val=0.
                    for q1 in range(nquads[conf_axis]): #loops on quadrature points
                        v0 = basis[conf_axis][ie1,il1,0,q1]
                        x  = grid[conf_axis][ie1,q1]
                        val += w[conf_axis][q1]*v0*((enddom-x)/denom)**poldeg
                    locind=span[conf_axis][ie1]-p_axis+il1
                    Mass_mat[poldeg,locind]+=val
        Rhs_0 = Mass_mat[:,0]

        if reg == 0:
            Mat_to_inv = Mass_mat[:,1:p_moments+2]
        else:
            Mat_to_inv = Mass_mat[:,2:p_moments+3]

        Correct_coef_0 = np.linalg.solve(Mat_to_inv,Rhs_0)    
        cc_0_ax = Correct_coef_0
        
        if reg == 1:
            Rhs_1 = Mass_mat[:,1]
            Correct_coef_1 = np.linalg.solve(Mat_to_inv,Rhs_1)    
            cc_1_ax = Correct_coef_1
        
        if hom_bc:
            # homogeneous bc is on the point value: no constraint on the derivatives
            # so only the projection of B0 (to 0) has to be corrected
            Mat_to_inv_bnd = Mass_mat[:,1:p_moments+2]
            Correct_coef_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)


        for p in range(0,p_moments+1):
            # correction for moment preserving : 
            # we use the first p_moments+1 conforming ("interior") functions to preserve the p+1 moments
            # modified by the C0 or C1 enforcement
            if reg == 0:
                a_sm[p+1] = (1-a_sm[0]) * cc_0_ax[p]
                # proj constraint:
                a_nb[p+1] = -a_sm[p+1]
            
            else:
                a_sm[p+2] = (1-a_sm[0]) * cc_0_ax[p]     -a_sm[1]  * cc_1_ax[p]
                b_sm[p+2] =   -b_sm[0]  * cc_0_ax[p] + (1-b_sm[1]) * cc_1_ax[p]
                
                # proj constraint:
                b_nb[p+2] = b_sm[p+2]
                a_nb[p+2] = -(a_sm[p+2] + 2*b_sm[p+2])
    return a_sm, a_nb, b_sm, b_nb, Correct_coef_bnd, Correct_coef_0

def get_vector_moment_correction(patch_space, conf_comp, conf_axis, reg=([0,0], [0,0]), p_moments=([-1,-1], [-1,-1]), nquads=None, hom_bc=([False, False],[False, False])):

    proj_op = 0
    local_shape = [[patch_space.spaces[comp].spaces[axis].nbasis 
                    for axis in range(2)] for comp in range(2)]
    Nel    = patch_space.ncells                     # number of elements
    patch_space_x, patch_space_y = [patch_space.spaces[comp] for comp in range(2)]
    degree = patch_space.degree 
    p_comp_axis = degree[conf_comp][conf_axis]

    breaks_comp_axis = [[breakpoints(patch_space.spaces[comp].knots[axis],degree[comp][axis])
                              for axis in range(2)] for comp in range(2)]
    if nquads is None:
    # default: Gauss-Legendre quadratures should be exact for polynomials of deg ≤ 2*degree
        nquads = [ degree[0][k]+1 for k in range(2)]
    #Creating vector of weights for moments preserving
    uw = [gauss_legendre( k-1 ) for k in nquads]
    u = [u[::-1] for u,w in uw]
    w = [w[::-1] for u,w in uw]
    
    grid = [np.array([deepcopy((0.5*(u[axis]+1)*(breaks_comp_axis[0][axis][i+1]-breaks_comp_axis[0][axis][i])+breaks_comp_axis[0][axis][i])) 
                      for i in range(Nel[axis])])
            for axis in range(2)]

    _, basis_x, span_x, _ = patch_space_x.preprocess_regular_tensor_grid(grid,der=0)
    _, basis_y, span_y, _ = patch_space_y.preprocess_regular_tensor_grid(grid,der=0)
    span_x = [deepcopy(span_x[k] + patch_space_x.vector_space.starts[k] - patch_space_x.vector_space.shifts[k] * patch_space_x.vector_space.pads[k]) for k in range(2)]
    span_y = [deepcopy(span_y[k] + patch_space_y.vector_space.starts[k] - patch_space_y.vector_space.shifts[k] * patch_space_y.vector_space.pads[k]) for k in range(2)]
    basis = [basis_x, basis_y]
    span = [span_x, span_y]
    enddom = breaks_comp_axis[0][0][-1]
    begdom = breaks_comp_axis[0][0][0]
    denom = enddom-begdom

    # projection coefficients
    
    a_sm = np.zeros(p_moments+2+reg)   # coefs of P B0 on same patch
    a_nb = np.zeros(p_moments+2+reg)   # coefs of P B0 on neighbor patch
    b_sm = np.zeros(p_moments+3)   # coefs of P B1 on same patch
    b_nb = np.zeros(p_moments+3)   # coefs of P B1 on neighbor patch
    Correct_coef_bnd = np.zeros(p_moments+1)
    Correct_coef_0 = np.zeros(p_moments+2+reg)
    a_sm[0] = 1/2
    a_nb[0] = a_sm[0]

    if reg == 1:
        b_sm = np.zeros(p_moments+3)   # coefs of P B1 on same patch
        b_nb = np.zeros(p_moments+3)   # coefs of P B1 on neighbor patch
        if proj_op == 0:
            # new slope is average of old ones
            a_sm[1] = 0  
        elif proj_op == 1:
            # new slope is average of old ones after averaging of interface coef
            a_sm[1] = 1/2
        elif proj_op == 2:
            # new slope is average of reconstructed ones using local values and slopes
            a_sm[1] = 1/(2*p_comp_axis)
        else:
            # just to try something else
            a_sm[1] = proj_op/2

        a_nb[1] = 2*a_sm[0] - a_sm[1]
        b_sm[0] = 0
        b_sm[1] = 1/2
        b_nb[0] = b_sm[0]
        b_nb[1] = 2*b_sm[0] - b_sm[1]
    
    if p_moments >= 0:
        # to preserve moments of degree p we need 1+p conforming basis functions in the patch (the "interior" ones)
        # and for the given regularity constraint, there are local_shape[conf_comp][conf_axis]-2*(1+reg) such conforming functions 
        p_max = local_shape[conf_comp][conf_axis]-2*(1+reg) - 1
        if p_max < p_moments:
            print( " ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            print( " **         WARNING -- WARNING -- WARNING ")
            print(f" ** conf. projection imposing C{reg} smoothness on component {conf_comp} along axis {conf_axis}:")            
            print(f" ** there are not enough dofs in a patch to preserve moments of degree {p_moments} !")
            print(f" ** Only able to preserve up to degree --> {p_max} <-- ")
            print( " ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **")
            p_moments = p_max

        # computing the contribution to every moment of the differents basis function
        # for simplicity we assemble the full matrix with all basis functions (ok if patches not too large)
        Mass_mat = np.zeros((p_moments+1,local_shape[conf_comp][conf_axis]))    
        for poldeg in range(p_moments+1):
            for ie1 in range(Nel[conf_axis]):   #loop on cells
                # cell_size = breaks_comp_axis[conf_comp][conf_axis][ie1+1]-breakpoints_x_y[ie1]  # todo: try without (probably not needed
                for il1 in range(p_comp_axis+1): #loops on basis function in each cell
                    val=0.
                    for q1 in range(nquads[conf_axis]): #loops on quadrature points
                        v0 = basis[conf_comp][conf_axis][ie1,il1,0,q1]
                        xd = grid[conf_axis][ie1,q1]
                        val += w[conf_axis][q1]*v0*((enddom-xd)/denom)**poldeg
                    locind=span[conf_comp][conf_axis][ie1]-p_comp_axis+il1
                    Mass_mat[poldeg,locind]+=val
        Rhs_0 = Mass_mat[:,0]

        if reg == 0:
            Mat_to_inv = Mass_mat[:,1:p_moments+2]
        else:
            Mat_to_inv = Mass_mat[:,2:p_moments+3]
        Correct_coef_0 = np.linalg.solve(Mat_to_inv,Rhs_0)    
        cc_0_ax = Correct_coef_0
        
        if reg == 1:
            Rhs_1 = Mass_mat[:,1]
            Correct_coef_1 = np.linalg.solve(Mat_to_inv,Rhs_1)    
            cc_1_ax = Correct_coef_1

        if hom_bc:
            # homogeneous bc is on the point value: no constraint on the derivatives
            # so only the projection of B0 (to 0) has to be corrected
            Mat_to_inv_bnd = Mass_mat[:,1:p_moments+2]
            Correct_coef_bnd = np.linalg.solve(Mat_to_inv_bnd,Rhs_0)

        for p in range(0,p_moments+1):
            # correction for moment preserving : 
            # we use the first p_moments+1 conforming ("interior") functions to preserve the p+1 moments
            # modified by the C0 or C1 enforcement
            if reg == 0:
                a_sm[p+1] = (1-a_sm[0]) * cc_0_ax[p]
                # proj constraint:
                a_nb[p+1] = -a_sm[p+1]
            
            else:
                a_sm[p+2] = (1-a_sm[0]) * cc_0_ax[p]     -a_sm[1]  * cc_1_ax[p]
                b_sm[p+2] =   -b_sm[0]  * cc_0_ax[p] + (1-b_sm[1]) * cc_1_ax[p]
                
                # proj constraint:
                b_nb[p+2] = b_sm[p+2]
                a_nb[p+2] = -(a_sm[p+2] + 2*b_sm[p+2])

    return a_sm, a_nb, b_sm, b_nb, Correct_coef_bnd, Correct_coef_0

def get_moment_pres_scalar_extension_restriction(matching_interfaces, coarse_space_1d, fine_space_1d, spl_type):
    grid = np.linspace(fine_space_1d.breaks[0], fine_space_1d.breaks[-1], coarse_space_1d.ncells+1)
    coarse_space_1d_k_plus = SplineSpace(degree=fine_space_1d.degree, grid=grid, basis=fine_space_1d.basis)

    if not matching_interfaces:
        E_1D = construct_extension_operator_1D(
            domain=coarse_space_1d_k_plus, codomain=fine_space_1d)
      
        # Calculate the mass matrices
        M_coarse = calculate_mass_matrix(coarse_space_1d, spl_type)
        M_fine = calculate_mass_matrix(fine_space_1d, spl_type)

        if spl_type == 'B':
            M_coarse[:, 0] *= 1e13
            M_coarse[:, -1] *= 1e13

        M_coarse_inv = np.linalg.inv(M_coarse)
        R_1D = M_coarse_inv @ E_1D.T @ M_fine
        
        if spl_type == 'B':
            R_1D[0,0] = R_1D[-1,-1] = 1

        ER_1D = E_1D @ R_1D
        
       # id_err = np.linalg.norm(R_1D @ E_1D - sparse_eye( coarse_space_1d.nbasis, format="lil"))
    else:
        ER_1D = R_1D = E_1D = sparse_eye(
            fine_space_1d.nbasis, format="lil")

    return E_1D, R_1D, ER_1D

def calculate_mass_matrix(space_1d, spl_type):
    Nel = space_1d.ncells
    deg = space_1d.degree
    knots = space_1d.knots

    u, w = gauss_legendre(deg ) 
    # invert order
    u = u[::-1]
    w = w[::-1]

    nquad = len(w)
    quad_x, quad_w = quadrature_grid(space_1d.breaks, u, w)

    coarse_basis = basis_ders_on_quad_grid(knots, deg, quad_x, 0, spl_type)
    spans = elements_spans(knots, deg)

    Mass_mat = np.zeros((space_1d.nbasis,space_1d.nbasis))    

    for ie1 in range(Nel):   #loop on cells
        for il1 in range(deg+1): #loops on basis function in each cell
            for il2 in range(deg+1): #loops on basis function in each cell
                val=0.

                for q1 in range(nquad): #loops on quadrature points
                    v0 = coarse_basis[ie1,il1,0,q1]
                    w0 = coarse_basis[ie1,il2,0,q1]
                    val += quad_w[ie1, q1] * v0 * w0

                locind1 = il1 + spans[ie1] - deg
                locind2 = il2 + spans[ie1] - deg
                Mass_mat[locind1,locind2] += val

    return Mass_mat


# if __name__ == '__main__':
 
#     nc = 5
#     deg = 3
#     nonconforming = True
#     plot_dir = 'run_plots_nc={}_deg={}'.format(nc, deg)

#     if plot_dir is not None and not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)

#     ncells = [nc, nc]
#     degree = [deg, deg]
#     reg_orders=[0,0]
#     p_moments=[3,3]

#     nquads=None
#     hom_bc=[False, False]
#     print(' .. multi-patch domain...')

#     #domain_name = 'square_6'
#     #domain_name = '2patch_nc_mapped'
#     domain_name = '2patch_nc'
#     #domain_name = "curved_L_shape"

#     if domain_name == '2patch_nc_mapped':

#         A = Square('A', bounds1=(0.5, 1), bounds2=(0,       np.pi/2))
#         B = Square('B', bounds1=(0.5, 1), bounds2=(np.pi/2, np.pi))
#         M1 = PolarMapping('M1', 2, c1=0, c2=0, rmin=0., rmax=1.)
#         M2 = PolarMapping('M2', 2, c1=0, c2=0, rmin=0., rmax=1.)
#         A = M1(A)
#         B = M2(B)

#         domain = create_domain([A, B], [[A.get_boundary(axis=1, ext=1), B.get_boundary(axis=1, ext=-1), 1]], name='domain')

#     elif domain_name == '2patch_nc':

#         A = Square('A', bounds1=(0, 0.5), bounds2=(0, 1))
#         B = Square('B', bounds1=(0.5, 1.), bounds2=(0, 1))
#         M1 = IdentityMapping('M1', dim=2)
#         M2 = IdentityMapping('M2', dim=2)
#         A = M1(A)
#         B = M2(B)

#         domain = create_domain([A, B], [[A.get_boundary(axis=0, ext=1), B.get_boundary(axis=0, ext=-1), 1]], name='domain')
#     elif domain_name == '4patch_nc':

#         A = Square('A', bounds1=(0, 0.5), bounds2=(0, 0.5))
#         B = Square('B', bounds1=(0.5, 1.), bounds2=(0, 0.5))
#         C = Square('C', bounds1=(0, 0.5), bounds2=(0.5, 1))
#         D = Square('D', bounds1=(0.5, 1.), bounds2=(0.5, 1))
#         M1 = IdentityMapping('M1', dim=2)
#         M2 = IdentityMapping('M2', dim=2)
#         M3 = IdentityMapping('M3', dim=2)
#         M4 = IdentityMapping('M4', dim=2)
#         A = M1(A)
#         B = M2(B)
#         C = M3(C)
#         D = M4(D)

#         domain = create_domain([A, B, C, D], [[A.get_boundary(axis=0, ext=1), B.get_boundary(axis=0, ext=-1), 1], 
#                                             [A.get_boundary(axis=1, ext=1), C.get_boundary(axis=1, ext=-1), 1],
#                                             [C.get_boundary(axis=0, ext=1), D.get_boundary(axis=0, ext=-1), 1],
#                                             [B.get_boundary(axis=1, ext=1), D.get_boundary(axis=1, ext=-1), 1] ], name='domain')
#     else:
#         domain = build_multipatch_domain(domain_name=domain_name)


#     n_patches = len(domain)

#     def levelof(k):
#         # some random refinement level (1 or 2 here)
#         return 1+((2*k) % 3) % 2
#     if nonconforming:
#         if len(domain) == 1:
#             ncells_h = {
#                 'M1(A)': [nc, nc],
#             }

#         elif len(domain) == 2:
#             ncells_h = {
#                 'M1(A)': [nc, nc],
#                 'M2(B)': [2*nc, 2*nc],
#             }

#         else:
#             ncells_h = {}
#             for k, D in enumerate(domain.interior):
#                 print(k, D.name)
#                 ncells_h[D.name] = [2**k *nc, 2**k * nc ]
#     else:
#         ncells_h = {}
#         for k, D in enumerate(domain.interior):
#             ncells_h[D.name] = [nc, nc]

#     print('ncells_h = ', ncells_h)
#     backend_language = 'python'

#     t_stamp = time_count()
#     print(' .. derham sequence...')
#     derham = Derham(domain, ["H1", "Hcurl", "L2"])

#     t_stamp = time_count(t_stamp)
#     print(' .. discrete domain...')

#     domain_h = discretize(domain, ncells=ncells_h)   # Vh space
#     derham_h = discretize(derham, domain_h, degree=degree)
#     V0h = derham_h.V0
#     V1h = derham_h.V1

#    # test_extension_restriction(V1h, domain)

  
#     #cP1_m_old = construct_V1_conforming_projection(V1h, True)
#    # cP0_m_old = construct_V0_conforming_projection(V0h,hom_bc[0])
#     cP0_m = construct_scalar_conforming_projection(V0h, reg_orders, p_moments, nquads, hom_bc)
#     cP1_m = construct_vector_conforming_projection(V1h, reg_orders, p_moments, nquads, hom_bc)

#     #print("Error:")
#     #print( norm(cP1_m - conf_cP1_m) )
#     np.set_printoptions(linewidth=100000, precision=2,
#                         threshold=100000, suppress=True)
#     #print(cP0_m.toarray())
     
#     # apply cP1 on some discontinuous G

#     # G_sol_log = [[lambda xi1, xi2, ii=i : ii+xi1+xi2**2 for d in [0,1]] for i in range(len(domain))]
#     # G_sol_log = [[lambda xi1, xi2, kk=k : levelof(kk)-1  for d in [0,1]] for k in range(len(domain))]
#     G_sol_log = [[lambda xi1, xi2, kk=k: kk for d in [0, 1]]
#                   for k in range(len(domain))]
#     #G_sol_log = [[lambda xi1, xi2, kk=k: np.cos(xi1)*np.sin(xi2) for d in [0, 1]]
#     #              for k in range(len(domain))]
#     P0, P1, P2 = derham_h.projectors()

#     G1h = P1(G_sol_log)
#     G1h_coeffs = G1h.coeffs.toarray()

#     #G1h_coeffs = np.zeros(G1h_coeffs.size)
#     #183, 182, 184
#     #G1h_coeffs[27] = 1

#     plot_field(numpy_coeffs=G1h_coeffs, Vh=V1h, space_kind='hcurl',
#               plot_type='components',
#               domain=domain, title='G1h', cmap='viridis',
#               filename=plot_dir+'/G.png')

   

#     G1h_conf_coeffs = cP1_m @ G1h_coeffs
    
#     plot_field(numpy_coeffs=G1h_conf_coeffs, Vh=V1h, space_kind='hcurl',
#                plot_type='components',
#               domain=domain, title='PG', cmap='viridis',
#                filename=plot_dir+'/PG.png')



#     #G0_sol_log = [[lambda xi1, xi2, kk=k: kk for d in [0]]
#     #             for k in range(len(domain))]
#     G0_sol_log = [[lambda xi1, xi2, kk=k:kk  for d in [0]]
#                     for k in range(len(domain))]
#     #G0_sol_log = [[lambda xi1, xi2, kk=k: np.cos(xi1)*np.sin(xi2) for d in [0]]
#     #            for k in range(len(domain))]
#     G0h = P0(G0_sol_log)
#     G0h_coeffs = G0h.coeffs.toarray()

#     #G0h_coeffs = np.zeros(G0h_coeffs.size)
#     #183, 182, 184
#     #conforming
#     # 30 - 24
#     # 28 - 23
#     #nc = 4, co: 59, co_ed:45, fi_ed:54
#     #G0h_coeffs[54] = 1
#     #G0h_coeffs[23] = 1

#     plot_field(numpy_coeffs=G0h_coeffs, Vh=V0h, space_kind='h1',
#               domain=domain, title='G0h', cmap='viridis',
#               filename=plot_dir+'/G0.png')

#     G0h_conf_coeffs = (cP0_m@cP0_m-cP0_m) @ G0h_coeffs

#     plot_field(numpy_coeffs=G0h_conf_coeffs, Vh=V0h, space_kind='h1',
#               domain=domain, title='PG0', cmap='viridis',
#               filename=plot_dir+'/PG0.png')

#     plot_field(numpy_coeffs=cP0_m @ G0h_coeffs, Vh=V0h, space_kind='h1',
#               domain=domain, title='PG00', cmap='viridis',
#               filename=plot_dir+'/PG00.png')

#     if not nonconforming:
#         cP0_martin = conf_proj_scalar_space(V0h, reg_orders, p_moments, nquads, hom_bc)

#         G0h_conf_coeffs_martin = cP0_martin @ G0h_coeffs
#         #plot_field(numpy_coeffs=G0h_conf_coeffs_martin, Vh=V0h, space_kind='h1',
#         #        domain=domain, title='PG0_martin', cmap='viridis',
#         #        filename=plot_dir+'/PG0_martin.png')

#         import numpy as np
#         import matplotlib.pyplot as plt
#         reg = 0
#         reg_orders  = [[reg-1, reg   ], [reg,    reg-1]]
#         hom_bc_list = [[False, hom_bc[1]], [hom_bc[0], False]]
#         deg_moments = [p_moments,p_moments]
#         V1h = derham_h.V1
#         V1 = V1h.symbolic_space
#         cP1_martin = conf_proj_vector_space(V1h, reg_orders=reg_orders, deg_moments=deg_moments, nquads=None, hom_bc_list=hom_bc_list)

#         #cP0_martin, cP1_martin, cP2_martin = conf_projectors_scipy(derham_h, single_space=None, reg=0, mom_pres=True, nquads=None, hom_bc=False)
        
#         G1h_conf_martin = cP1_martin @ G1h_coeffs

#        # plot_field(numpy_coeffs=G1h_conf_martin, Vh=V1h, space_kind='hcurl',
#        #             plot_type='components',
#        #         domain=domain, title='PG_martin', cmap='viridis',
#        #             filename=plot_dir+'/PG_martin.png')


#         plt.matshow((cP1_m - cP1_martin).toarray())
#         plt.colorbar()
#         print(sp_norm(cP1_m - cP1_martin))
#         #plt.matshow((cP0_m).toarray())

#         #plt.matshow((cP0_martin).toarray())
#         #plt.show()

#         #print( np.sum(cP0_m - cP0_martin))
#        # print( cP0_m - cP0_martin)

#     print(sp_norm(cP0_m-    cP0_m @ cP0_m))
#     print(sp_norm(cP1_m-    cP1_m @ cP1_m))

