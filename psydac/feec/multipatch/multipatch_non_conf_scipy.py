#def unravel_single_patch_indices(cartesian_index, *shapes, component_index=0):
#    """ Convert the local multi index to the global index in the flattened array

#    Parameters
#    ----------
#    multi_index : <tuple|list>
#     The multidimentional index is of the form [patch_index, component_index, array_index]
#     or [patch_index, array_index] if the number of components is one

#    n_patches: int
#     The total number of patches

#    single_patch_shapes: a list of tuples or a tuple
#     It contains the shapes of the multidimentional arrays in a single patch

#    Returns
#    -------
#     I : int
#      The global index in the flattened array.

#    Examples
#    --------
#    loca2global([0,0,1],2,[100,100])
#    >>> 10000

#    loca2global([0,1,0,1],2,[[80,80],[100,100]])
#    >>> 6401
#    """
#    import numpy as np

#    sizes = [np.product(s) for s in shapes[:component_index]]
#    Ipc = np.ravel_multi_index(cartesian_index, dims=shapes[component_index], order='C')
#    return sum(sizes) + Ipc

import os
import numpy as np
from scipy.sparse import eye as sparse_eye
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

from sympde.topology  import Derham, Square
from sympde.topology  import IdentityMapping
from sympde.topology  import Boundary, Interface, Union

from psydac.feec.multipatch.utilities import time_count
from psydac.linalg.utilities          import array_to_stencil
from psydac.feec.multipatch.api       import discretize
from psydac.api.settings              import PSYDAC_BACKENDS
from psydac.fem.splines               import SplineSpace

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
        raise NotImplementedError("This face is an interface, it has several indices -- I am a machine, I cannot choose. Help.")
    elif isinstance(face, Boundary):
        i = domains.index(face.domain)
    else:
        i = domains.index(face)
    return i

class Local2GlobalIndexMap:
    def __init__(self, ndim, n_patches, n_components):
#        A[patch_index][component_index][i1,i2]
        self._shapes = [None]*n_patches
        self._ndofs  = [None]*n_patches
        self._ndim   = ndim
        self._n_patches    = n_patches
        self._n_components = n_components

    def set_patch_shapes(self, patch_index, *shapes):
        assert len(shapes) == self._n_components
        assert all(len(s) == self._ndim for s in shapes)
        self._shapes[patch_index] = shapes
        self._ndofs[patch_index]  = sum(np.product(s) for s in shapes)

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
        Ipc = np.ravel_multi_index(cartesian_index, dims=self._shapes[k][d], order='C')
        Ip  = sum(sizes) + Ipc
        I   = sum(self._ndofs[:k]) + Ip
        return I

def knots_to_insert(coarse_grid, fine_grid, tol=1e-14):
#    assert len(coarse_grid)*2-2 == len(fine_grid)-1
    intersection = coarse_grid[(np.abs(fine_grid[:,None] - coarse_grid) < tol).any(0)]
    assert abs(intersection-coarse_grid).max()<tol
    T = fine_grid[~(np.abs(coarse_grid[:,None] - fine_grid) < tol).any(0)]
    return T

def construct_extension_operator_1D(domain, codomain):
    """

    compute the matrix of the extension operator on the interface space (1D space if global space is 2D)
    
    domain:     1d spline space on the interface (coarse grid)
    codomain:   1d spline space on the interface (fine grid)
    """
    from psydac.core.interface import matrix_multi_stages
    
    ops = []


    assert domain.ncells < codomain.ncells

    Ts = knots_to_insert(domain.breaks, codomain.breaks)
    P  = matrix_multi_stages(Ts, domain.nbasis , domain.degree, domain.knots)
    if domain.basis == 'M':
        assert codomain.basis == 'M'
        P = np.diag(1/codomain._scaling_array) @ P @ np.diag(domain._scaling_array)

    return csr_matrix(P) # kronecker of 1 term...

def construct_V1_conforming_projection(V1h, domain_h, hom_bc=False, storage_fn=None):
    dim_tot      = V1h.nbasis
    domain       = V1h.symbolic_space.domain
    ndim         = 2
    n_components = 2
    n_patches    = len(domain)

    l2g = Local2GlobalIndexMap(ndim, len(domain), n_components)
    for k in range(n_patches):
        Vk = V1h.spaces[k]
        # T is a TensorFemSpace and S is a 1D SplineSpace
        shapes = [[S.nbasis for S in T.spaces] for T in Vk.spaces]
        l2g.set_patch_shapes(k, *shapes)

    Proj     = sparse_eye(dim_tot,format="lil")

    Interfaces  = domain.interfaces
    if isinstance(Interfaces, Interface):
        Interfaces = (Interfaces, )

    for I in Interfaces:
        axis      = I.axis
        direction = I.direction

        k_fine    = get_patch_index_from_face(domain, I.minus)
        k_coarse  = get_patch_index_from_face(domain, I.plus )

        # This is the component normal to the interface
        fine_axis, coarse_axis   = I.minus.axis, I.plus.axis
        fine_ext, coarse_ext   = I.minus.ext, I.plus.ext

        d_coarse = 1-coarse_axis   # direction of the components on interface
        d_fine   = 1-fine_axis

        if V1h.spaces[k_fine].spaces[d_fine].ncells[d_fine] < V1h.spaces[k_coarse].spaces[d_coarse].ncells[d_coarse]:
            k_fine, k_coarse       = k_coarse, k_fine
            fine_axis, coarse_axis = I.plus.axis, I.minus.axis
            fine_ext, coarse_ext   = I.plus.ext, I.minus.ext

            d_coarse = 1-coarse_axis   # direction of the components on interface
            d_fine   = 1-fine_axis

        space_fine   = V1h.spaces[k_fine]
        space_coarse = V1h.spaces[k_coarse]

        coarse_space_1d = space_coarse.spaces[d_coarse].spaces[d_coarse]
        fine_space_1d   = space_fine.spaces[d_fine].spaces[d_fine]
        grid = np.linspace(fine_space_1d.breaks[0], fine_space_1d.breaks[-1], coarse_space_1d.ncells+1)
        coarse_space_1d_k_plus  = SplineSpace(degree=fine_space_1d.degree, grid=grid, basis=fine_space_1d.basis)

        E_1D    = construct_extension_operator_1D(domain=coarse_space_1d_k_plus, codomain=fine_space_1d)

        product = (E_1D.T) @ E_1D
        R_1D    = inv(product.tocsc()) @ E_1D.T
        ER_1D   = E_1D @ R_1D

        # P_k_minus_k_minus
        multi_index = [None]*ndim
        multi_index[coarse_axis] = 0 if coarse_ext == -1 else space_coarse.spaces.spaces[d_coarse][coarse_axis].nbasis-1
        for i in range(coarse_space_1d.nbasis):
            multi_index[d_coarse]  = i
            ig = l2g.get_index(k_coarse, d_coarse, multi_index)
            Proj[ig, ig] = 0.5

        # P_k_plus_k_plus
        multi_index_i = [None]*ndim
        multi_index_j = [None]*ndim
        multi_index_i[fine_axis] = 0 if fine_ext == -1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1
        multi_index_j[fine_axis] = 0 if fine_ext == -1 else space_fine.spaces[d_fine].spaces[fine_axis].nbasis-1

        for i in range(fine_space_1d.nbasis):
            multi_index_i[d_fine] = i
            ig   = l2g.get_index(k_fine, d_fine, multi_index_i)
            for j in range(fine_space_1d.nbasis):
                multi_index_j[d_fine] = j
                jg           = l2g.get_index(k_fine, d_fine, multi_index_j)
                Proj[ig, jg] = 0.5*ER_1D[i, j]

        # P_k_plus_k_minus
        multi_index_i = [None]*ndim
        multi_index_j = [None]*ndim
        multi_index_i[fine_axis]   = 0 if fine_ext   == -1 else space_fine  .spaces[d_fine]  .spaces[fine_axis]  .nbasis-1
        multi_index_j[coarse_axis] = 0 if coarse_ext == -1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1

        for i in range(fine_space_1d.nbasis):
            multi_index_i[d_fine] = i
            ig          = l2g.get_index(k_fine, d_fine, multi_index_i)
            for j in range(coarse_space_1d.nbasis):
                multi_index_j[d_coarse] = j if direction == 1 else coarse_space_1d.nbasis-j-1
                jg           = l2g.get_index(k_coarse, d_coarse, multi_index_j)
                Proj[ig, jg] = 0.5*E_1D[i, j]*direction

        # P_k_minus_k_plus
        multi_index_i = [None]*ndim
        multi_index_j = [None]*ndim
        multi_index_i[coarse_axis] = 0 if coarse_ext == -1 else space_coarse.spaces[d_coarse].spaces[coarse_axis].nbasis-1
        multi_index_j[fine_axis]   = 0 if fine_ext   == -1 else space_fine  .spaces[d_fine]  .spaces[fine_axis]  .nbasis-1

        for i in range(coarse_space_1d.nbasis):
            multi_index_i[d_coarse] = i
            ig          = l2g.get_index(k_coarse, d_coarse, multi_index_i)
            for j in range(fine_space_1d.nbasis):
                multi_index_j[d_fine] = j if direction == 1 else fine_space_1d.nbasis-j-1
                jg           = l2g.get_index(k_fine, d_fine, multi_index_j)
                Proj[ig, jg] = 0.5*R_1D[i, j]*direction

    return Proj

if __name__ == '__main__':

    nc  = 2
    deg = 2
    plot_dir = 'run_plots_nc={}_deg={}'.format(nc,deg)

    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    ncells = [nc, nc]
    degree = [deg,deg]

    print(' .. multi-patch domain...')
    # domain = build_multipatch_domain(domain_name='two_patch_nc')

    A  = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B  = Square('B',bounds1=(0.5, 1.), bounds2=(0, 1))
    M1 = IdentityMapping('M1', dim=2)
    M2 = IdentityMapping('M2', dim=2)
    A  = M1(A)
    B  = M2(B)

    domain = A.join(B, name = 'domain',
                bnd_minus = A.get_boundary(axis=0, ext=1),
                bnd_plus  = B.get_boundary(axis=0, ext=-1),
                direction=1)
    
    # nc = 2
    ncells_c = {
        'M1(A)':[nc, nc],
        'M2(B)':[nc, nc],
    }
    ncells_f = {
        'M1(A)':[2*nc, 2*nc],
        'M2(B)':[2*nc, 2*nc],
    }
    ncells_h = {
        'M1(A)':[2*nc, 2*nc],
        'M2(B)':[nc, nc],
    }

    backend_language = 'python'

    t_stamp = time_count()
    print(' .. derham sequence...')
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. discrete domain...')

    domain_h = discretize(domain, ncells=ncells_h)   # Vh space
    derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    V1h = derham_h.V1

    CP1 = construct_V1_conforming_projection(V1h, domain_h)

    np.set_printoptions(linewidth=100000, precision=2, threshold=100000, suppress=True)
    print(CP1.toarray())
