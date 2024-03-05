import os
import numpy as np
from scipy.sparse import eye as sparse_eye
from scipy.sparse import csr_matrix
from sympde.topology import Boundary, Interface
from psydac.fem.splines import SplineSpace
from psydac.utilities.quadratures import gauss_legendre
from psydac.core.bsplines import breakpoints, quadrature_grid, basis_ders_on_quad_grid, find_spans, elements_spans
from copy import deepcopy


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
        self._shapes = [None]*n_patches
        self._ndofs = [None]*n_patches
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

    intersection = coarse_grid[(
        np.abs(fine_grid[:, None] - coarse_grid) < tol).any(0)]
    assert abs(intersection-coarse_grid).max() < tol
    T = fine_grid[~(np.abs(coarse_grid[:, None] - fine_grid) < tol).any(0)]
    return T


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
            1/codomain._scaling_array) @ P @ np.diag(domain._scaling_array)

    return csr_matrix(P)  

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
    """ Construct the conforming projection for a scalar space for a given regularity (0 continuous, -1 discontinuous). 
        The conservation of p-moments only works for a matching TensorFemSpace. 

    Parameters
    ----------
    Vh : TensorFemSpace
        Finite Element Space coming from the discrete de Rham sequence.

    reg_orders : tuple-like (int)
        Regularity in each space direction -1 or 0. 

    p_moments : tuple-like (int)
        Number of moments to be preserved. 

    nquads  : int | None
        Number of quadrature points. 

    hom_bc : tuple-like (bool)
        Homogeneous boundary conditions. 

    Returns
    -------
    cP : scipy.sparse.csr_array
        Conforming projection as a sparse matrix.
    """

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
    """ Construct the conforming projection for a scalar space for a given regularity (0 continuous, -1 discontinuous). 
        The conservation of p-moments only works for a matching VectorFemSpace. 

    Parameters
    ----------
    Vh : VectorFemSpace
        Finite Element Space coming from the discrete de Rham sequence.

    reg_orders : tuple-like (int)
        Regularity in each space direction -1 or 0. 

    p_moments : tuple-like (int)
        Number of moments to be preserved. 

    nquads  : int | None
        Number of quadrature points. 

    hom_bc : tuple-like (bool)
        Homogeneous boundary conditions. 

    Returns
    -------
    cP : scipy.sparse.csr_array
        Conforming projection as a sparse matrix.
    """
    
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
    """ 
    Calculate the coefficients for the one-dimensional moment correction.

    Parameters
    ----------
    patch_space : TensorFemSpace
        Finite Element Space of an adjacent patch.

    conf_axis : {0, 1}
        Coefficients for which axis. 

    reg : {-1, 0}
        Regularity -1 or 0. 

    p_moments  : int 
        Number of moments to be preserved. 

    nquads  : int | None
        Number of quadrature points. 

    hom_bc : tuple-like (bool)
        Homogeneous boundary conditions. 

    Returns
    -------
    coeffs : list of arrays
        Collection of the different coefficients. 
    """
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
    """ 
    Calculate the coefficients for the vector-valued moment correction.

    Parameters
    ----------
    patch_space : VectorFemSpace
        Finite Element Space of an adjacent patch.

    conf_comp : {0, 1}
        Coefficients for which vector component. 

    conf_axis : {0, 1}
        Coefficients for which axis. 

    reg : tuple-like 
        Regularity -1 or 0. 

    p_moments  : tuple-like  
        Number of moments to be preserved. 

    nquads  : int | None
        Number of quadrature points. 

    hom_bc : tuple-like (bool)
        Homogeneous boundary conditions. 

    Returns
    -------
    coeffs : list of arrays
        Collection of the different coefficients. 
    """
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
    """ 
    Calculate the extension and restriction matrices for refining along an interface.

    Parameters
    ----------
    matching_interfaces : bool
        Do both patches have the same number of cells? 

    coarse_space_1d : SplineSpace
        Spline space of the coarse space. 

    fine_space_1d : SplineSpace
        Spline space of the fine space. 

    spl_type : {'B', 'M'}
        Spline type.  

    Returns
    -------
    E_1D : numpy array
        Extension matrix. 

    R_1D : numpy array
        Restriction matrix. 

    ER_1D : numpy array
        Extension-restriction matrix.
    """
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
        
    else:
        ER_1D = R_1D = E_1D = sparse_eye(
            fine_space_1d.nbasis, format="lil")

    return E_1D, R_1D, ER_1D

# Didn't find this utility in the code base.
def calculate_mass_matrix(space_1d, spl_type):
    """ 
    Calculate the mass-matrix of a 1d spline-space.

    Parameters
    ----------

    space_1d : SplineSpace
        Spline space of the fine space. 

    spl_type : {'B', 'M'}
        Spline type.  

    Returns
    -------

    Mass_mat : numpy array
        Mass matrix.
    """
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