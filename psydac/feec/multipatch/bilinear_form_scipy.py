import os
import numpy as np
from scipy.sparse import eye as sparse_eye
from scipy.sparse import csr_matrix, lil_matrix, kron
from scipy.sparse.linalg import inv

from sympde.topology import Derham, Square
from sympde.topology import IdentityMapping
from sympde.topology import Boundary, Interface, Union

from psydac.feec.multipatch.utilities import time_count
# from psydac.linalg.utilities import array_to_stencil
from psydac.feec.multipatch.api import discretize
from psydac.api.settings import PSYDAC_BACKENDS
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor  import TensorFemSpace
from psydac.core.bsplines import quadrature_grid, basis_ders_on_quad_grid
from psydac.core.bsplines import elements_spans
from psydac.utilities.quadratures import gauss_legendre

from psydac.fem.basic import FemField
from psydac.feec.multipatch.plotting_utilities import plot_field

from sympde.topology import IdentityMapping, PolarMapping
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

### copied from devel_conga_non_conf branch:

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

class Local2GlobalIndexMapv0:
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

        Ipc = np.ravel_multi_index(
            cartesian_index, dims=self._shapes[k][0], order='C')
        I = sum(self._ndofs[:k]) + Ipc
        return I



## mass matrix in V1 first:

def construct_V1_mass_matrix(V1h, storage_fn=None):

    print("construct_V1_mass_matrix ...")
    dim_tot = V1h.nbasis
    domain = V1h.symbolic_space.domain
    ndim = 2
    n_components = 2
    n_patches = len(domain)

    M = lil_matrix((dim_tot, dim_tot))

    l2g = Local2GlobalIndexMap(ndim, len(domain), n_components)
    for k in range(n_patches):
        Vk = V1h.spaces[k]
        # T is a TensorFemSpace and S is a 1D SplineSpace
        shapes = [[S.nbasis for S in T.spaces] for T in Vk.spaces]
        l2g.set_patch_shapes(k, *shapes)

        for dim in range(ndim):
            # compute products Lambda_{dim,i} . Lambda_{dim,j}

            Vk_dim = Vk.spaces[dim]  
            assert isinstance(Vk_dim, TensorFemSpace)             

            M_1D = [None]*ndim
            multi_index_i = [None]*ndim
            multi_index_j = [None]*ndim

            for axis in range(ndim):
                space_axis = Vk_dim.spaces[axis]
                M_1D_axis = lil_matrix((space_axis.nbasis, space_axis.nbasis))

                knots  = space_axis.knots
                degree = space_axis.degree
                
                # quad_grid = Vk_dim.quad_grids[axis]        ##   use this ??
                                
                grid      = space_axis.breaks          # breakpoints, should be the same for both spaces !
                
                # Gauss-legendre quadrature rule
                u, w = gauss_legendre( degree )  # check degree ?..

                # invert order  ( why?)
                u = u[::-1]
                w = w[::-1]
                
                # Lists of quadrature coordinates and weights on each element
                quad_x, quad_w = quadrature_grid(grid, u, w)

                # normalization = 'M' if axis == dim else 'B'
                # print(space_axis.basis)
                # exit()
                quad_basis = basis_ders_on_quad_grid(knots, degree, quad_x, nders=0, normalization=space_axis.basis)
                # mass matrix: same basis for both spaces....

                # sg = quad_grid.local_element_start
                # eg = quad_grid.local_element_end   

                # loop over elements and local basis functions
                # for ie in range(sg,eg+1):
                for ie, span in enumerate(elements_spans(knots, degree)):
                    for i_loc in range(degree+1):
                        i_glob = span-degree + i_loc
                        for j_loc in range(degree+1):        
                            j_glob = span-degree + j_loc

                            M_1D_axis[i_glob,j_glob] += np.dot(quad_basis[ie,i_loc,0,:] * quad_basis[ie,j_loc,0,:], quad_w[ie,:])
                
                M_1D[axis] = M_1D_axis

            for i0 in range(Vk_dim.spaces[0].nbasis):
                for i1 in range(Vk_dim.spaces[1].nbasis):
                    multi_index_i[0] = i0
                    multi_index_i[1] = i1
                    ig = l2g.get_index(k, dim, multi_index_i)  
                            
                    # note: we could localize the j loop (useful for large patches)
                    for j0 in range(Vk_dim.spaces[0].nbasis):
                        for j1 in range(Vk_dim.spaces[1].nbasis):
                            multi_index_j[0] = j0
                            multi_index_j[1] = j1
                            jg = l2g.get_index(k, dim, multi_index_j)  

                            M[ig,jg] = M_1D[0][i0,j0] * M_1D[1][i1,j1]
            
    return M
            
    

def construct_pairing_matrix(Vh, Wh, storage_fn=None):

    """
    compute the matrix in scipy format:

    M = (<Lambda^V_i, Lambda^W_j>)_{i,j} 

    """
    
    print("construct_pairing_matrix ...")
    
    domain = Vh.symbolic_space.domain
    ndim = 2            # dimensions of the logical domain
    n_components = 2    # dimension of the functions value
    n_patches = len(domain)

    M = lil_matrix((Vh.nbasis, Wh.nbasis))

    l2g_V = Local2GlobalIndexMap(ndim, n_patches, n_components)
    l2g_W = Local2GlobalIndexMap(ndim, n_patches, n_components)
    for k in range(n_patches):
        Vk = Vh.spaces[k]
        Wk = Wh.spaces[k]
        # T is a TensorFemSpace and S is a 1D SplineSpace
        V_shapes = [[S.nbasis for S in T.spaces] for T in Vk.spaces]
        W_shapes = [[S.nbasis for S in T.spaces] for T in Wk.spaces]
        l2g_V.set_patch_shapes(k, *V_shapes)
        l2g_W.set_patch_shapes(k, *W_shapes)

        for d in range(n_components):
            # compute products Lambda_V_{d,i} . Lambda_W_{d,j}
            # -- hard-coded assumption: Lambda_V_{c,i} . Lambda_W_{d,j} = 0 for c≠d

            Vk_d = Vk.spaces[d]
            Wk_d = Wk.spaces[d]
            assert isinstance(Vk_d, TensorFemSpace)             

            M_1D = [None]*ndim
            multi_index_i = [None]*ndim
            multi_index_j = [None]*ndim

            for axis in range(ndim):
                V_space_axis = Vk_d.spaces[axis]
                W_space_axis = Wk_d.spaces[axis]
                
                M_1D_axis = lil_matrix((V_space_axis.nbasis, W_space_axis.nbasis))

                V_degree = V_space_axis.degree
                W_degree = W_space_axis.degree
                
                V_knots  = V_space_axis.knots 
                W_knots  = W_space_axis.knots 
                
                grid     = V_space_axis.breaks
                assert all(grid  == W_space_axis.breaks)
                
                # quad_grid = Vk_dim.quad_grids[axis]        ##   use this ?
                             
                # Gauss-legendre quadrature rule
                u, w = gauss_legendre( max(V_degree, W_degree) )  # degree high enough ?

                # invert order  ( why?)
                u = u[::-1]
                w = w[::-1]
                
                # Lists of quadrature coordinates and weights on each element
                quad_x, quad_w = quadrature_grid(grid, u, w)

                V_quad_basis = basis_ders_on_quad_grid(V_knots, V_degree, quad_x, nders=0, normalization=V_space_axis.basis)
                W_quad_basis = basis_ders_on_quad_grid(W_knots, W_degree, quad_x, nders=0, normalization=W_space_axis.basis)

                # loop over elements and local basis functions
                # for ie in range(sg,eg+1):
                for ie, (V_span, W_span) in enumerate(zip(elements_spans(V_knots, V_degree), elements_spans(W_knots, W_degree))):
                    for i_loc in range(V_degree+1):
                        i_glob = V_span-V_degree + i_loc
                        for j_loc in range(W_degree+1):        
                            j_glob = W_span-W_degree + j_loc

                            M_1D_axis[i_glob,j_glob] += np.dot(V_quad_basis[ie,i_loc,0,:] * W_quad_basis[ie,j_loc,0,:], quad_w[ie,:])
                
                M_1D[axis] = M_1D_axis

            for i0 in range(Vk_d.spaces[0].nbasis):
                for i1 in range(Vk_d.spaces[1].nbasis):
                    multi_index_i[0] = i0
                    multi_index_i[1] = i1
                    ig = l2g_V.get_index(k, d, multi_index_i)  
                            
                    # note: we could localize the j loop (useful for large patches)
                    for j0 in range(Wk_d.spaces[0].nbasis):
                        for j1 in range(Wk_d.spaces[1].nbasis):
                            multi_index_j[0] = j0
                            multi_index_j[1] = j1
                            jg = l2g_W.get_index(k, d, multi_index_j)  

                            M[ig,jg] = M_1D[0][i0,j0] * M_1D[1][i1,j1]
            
    return M
            
    












    # Interfaces = domain.interfaces
    # if isinstance(Interfaces, Interface):
    #     Interfaces = (Interfaces, )
    # print(Interfaces)
    # for I in Interfaces:
    #     axis = I.axis
    #     direction = I.direction

    #     k_minus = get_patch_index_from_face(domain, I.minus)
    #     k_plus = get_patch_index_from_face(domain, I.plus)
    #     # logical directions normal to interface
    #     minus_axis, plus_axis = I.minus.axis, I.plus.axis
    #     # logical directions along the interface
    #     d_minus, d_plus = 1-minus_axis, 1-plus_axis
    #     I_minus_ncells = V1h.spaces[k_minus].spaces[d_minus].ncells[d_minus]
    #     I_plus_ncells = V1h.spaces[k_plus] .spaces[d_plus] .ncells[d_plus]

    #     matching_interfaces = (I_minus_ncells == I_plus_ncells)

    #     if I_minus_ncells <= I_plus_ncells:
    #         k_fine, k_coarse = k_plus, k_minus
    #         fine_axis, coarse_axis = I.plus.axis, I.minus.axis
    #         fine_ext,  coarse_ext = I.plus.ext,  I.minus.ext

    #     else:
    #         k_fine, k_coarse = k_minus, k_plus
    #         fine_axis, coarse_axis = I.minus.axis, I.plus.axis
    #         fine_ext,  coarse_ext = I.minus.ext, I.plus.ext

    #     d_fine = 1-fine_axis
    #     d_coarse = 1-coarse_axis

    #     space_fine = V1h.spaces[k_fine]
    #     space_coarse = V1h.spaces[k_coarse]

    #     #print("coarse = \n", space_coarse.spaces[d_coarse])
    #     #print("coarse 2 = \n", space_coarse.spaces[d_coarse].spaces[d_coarse])
    #     # todo: merge with first test above
    #     coarse_space_1d = space_coarse.spaces[d_coarse].spaces[d_coarse]

    #     #print("fine = \n", space_fine.spaces[d_fine])
    #     #print("fine 2 = \n", space_fine.spaces[d_fine].spaces[d_fine])

    #     fine_space_1d = space_fine.spaces[d_fine].spaces[d_fine]
    #     grid = np.linspace(
    #         fine_space_1d.breaks[0], fine_space_1d.breaks[-1], coarse_space_1d.ncells+1)
    #     coarse_space_1d_k_plus = SplineSpace(
    #         degree=fine_space_1d.degree, grid=grid, basis=fine_space_1d.basis)

    #     if not matching_interfaces:
    #         E_1D = construct_extension_operator_1D(
    #             domain=coarse_space_1d_k_plus, codomain=fine_space_1d)
    #         product = (E_1D.T) @ E_1D
    #         R_1D = inv(product.tocsc()) @ E_1D.T
    #         ER_1D = E_1D @ R_1D
    #     else:
    #         ER_1D = R_1D = E_1D = sparse_eye(
    #             fine_space_1d.nbasis, format="lil")

    #     # P_k_minus_k_minus
    #     multi_index = [None]*ndim
    #     multi_index[coarse_axis] = 0 if coarse_ext == - \
    #         1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1
    #     for i in range(coarse_space_1d.nbasis):
    #         multi_index[d_coarse] = i
    #         ig = l2g.get_index(k_coarse, d_coarse, multi_index)
    #         Proj[ig, ig] = 0.5

    #     # P_k_plus_k_plus
    #     multi_index_i = [None]*ndim
    #     multi_index_j = [None]*ndim
    #     multi_index_i[fine_axis] = 0 if fine_ext == - \
    #         1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1
    #     multi_index_j[fine_axis] = 0 if fine_ext == - \
    #         1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1

    #     for i in range(fine_space_1d.nbasis):
    #         multi_index_i[d_fine] = i
    #         ig = l2g.get_index(k_fine, d_fine, multi_index_i)
    #         for j in range(fine_space_1d.nbasis):
    #             multi_index_j[d_fine] = j
    #             jg = l2g.get_index(k_fine, d_fine, multi_index_j)
    #             Proj[ig, jg] = 0.5*ER_1D[i, j]

    #     # P_k_plus_k_minus
    #     multi_index_i = [None]*ndim
    #     multi_index_j = [None]*ndim
    #     multi_index_i[fine_axis] = 0 if fine_ext == - \
    #         1 else space_fine  .spaces[d_fine]  .spaces[fine_axis]  .nbasis-1
    #     multi_index_j[coarse_axis] = 0 if coarse_ext == - \
    #         1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1

    #     for i in range(fine_space_1d.nbasis):
    #         multi_index_i[d_fine] = i
    #         ig = l2g.get_index(k_fine, d_fine, multi_index_i)
    #         for j in range(coarse_space_1d.nbasis):
    #             multi_index_j[d_coarse] = j if direction == 1 else coarse_space_1d.nbasis-j-1
    #             jg = l2g.get_index(k_coarse, d_coarse, multi_index_j)
    #             Proj[ig, jg] = 0.5*E_1D[i, j]*direction

    #     # P_k_minus_k_plus
    #     multi_index_i = [None]*ndim
    #     multi_index_j = [None]*ndim
    #     multi_index_i[coarse_axis] = 0 if coarse_ext == - \
    #         1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1
    #     multi_index_j[fine_axis] = 0 if fine_ext == - \
    #         1 else space_fine  .spaces[d_fine]  .spaces[fine_axis]  .nbasis-1

    #     for i in range(coarse_space_1d.nbasis):
    #         multi_index_i[d_coarse] = i
    #         ig = l2g.get_index(k_coarse, d_coarse, multi_index_i)
    #         for j in range(fine_space_1d.nbasis):
    #             multi_index_j[d_fine] = j if direction == 1 else fine_space_1d.nbasis-j-1
    #             jg = l2g.get_index(k_fine, d_fine, multi_index_j)
    #             Proj[ig, jg] = 0.5*R_1D[i, j]*direction

    # if hom_bc:
    #     for bn in domain.boundary:
    #         k = get_patch_index_from_face(domain, bn)
    #         space_k = V1h.spaces[k]
    #         axis = bn.axis
    #         d = 1-axis
    #         ext = bn.ext
    #         space_k_1d = space_k.spaces[d].spaces[d]  # t
    #         multi_index_i = [None]*ndim
    #         multi_index_i[axis] = 0 if ext == - \
    #             1 else space_k.spaces[d].spaces[axis].nbasis-1

    #         for i in range(space_k_1d.nbasis):
    #             multi_index_i[d] = i
    #             ig = l2g.get_index(k, d, multi_index_i)
    #             Proj[ig, ig] = 0

    # return Proj


