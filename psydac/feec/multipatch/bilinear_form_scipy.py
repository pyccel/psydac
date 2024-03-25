import numpy as np
from scipy.sparse import eye as sparse_eye
from scipy.sparse import csr_matrix, lil_matrix, kron, block_diag
from scipy.sparse.linalg import inv as spla_inv

from sympde.topology  import element_of
from sympde.calculus  import dot
from sympde.topology.space  import ScalarFunction

from psydac.fem.tensor  import TensorFemSpace, FemSpace
from psydac.fem.vector  import VectorFemSpace
from psydac.core.bsplines import quadrature_grid, basis_ders_on_quad_grid
from psydac.core.bsplines import elements_spans
from psydac.utilities.quadratures import gauss_legendre
from sympde.expr.expr import BilinearForm
from sympde.expr.expr import integral
from psydac.api.discretization       import discretize
from psydac.api.settings             import PSYDAC_BACKENDS

### copied from devel_conga_non_conf branch:

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

def block_diag_inv(M):
    nrows = M.n_block_rows
    assert nrows == M.n_block_cols

    inv_M_blocks = []
    for i in range(nrows):
        Mii = M[i,i].tosparse()
        inv_Mii = spla_inv(Mii.tocsc())
        inv_Mii.eliminate_zeros()
        inv_M_blocks.append(inv_Mii)

    return block_diag(inv_M_blocks)


def construct_pairing_matrix(Vh, Wh, domain_h, storage_fn=None):
    """
    compute a pairing (coupling) in scipy format:

    K = (<Lambda^V_i, Lambda^W_j>)_{i,j} 

    on the logical spaces: no mappings involved here.
    
    Note: 
        if Vh == Wh then this is the logical mass matrix 
        (should then be the same matrix as the 
        Psydac mass matrix converted in scipy format)

    """
    
    print("construct_pairing_matrix ...")
    
    ndim = 2            # dimensions of the logical domain

    assert isinstance(Vh, FemSpace)
    assert isinstance(Wh, FemSpace)
    domain = Vh.symbolic_space.domain
    n_patches = len(domain)
    assert n_patches == len(Vh.spaces)
    assert n_patches == len(Wh.spaces)
    # print("type(Vh.spaces[0]) = ", type(Vh.spaces[0]))
    
    if isinstance(Vh.spaces[0], TensorFemSpace):
        # Vh is a scalar-valued space
        n_components = 1
        if not isinstance(Wh.spaces[0], TensorFemSpace):
            raise TypeError("Vh seems to be scalar-valued but Wh is not")
    else:
        assert isinstance(Vh.spaces[0], VectorFemSpace)
        n_components = len(Vh.spaces[0].spaces) # dimension of the functions value
        if not isinstance(Wh.spaces[0], VectorFemSpace):
            raise TypeError("Vh seems to be vector-valued but Wh is not")
        if not n_components == len(Wh.spaces[0].spaces):
            raise TypeError("Vh and Wh don't have the same number of components")
    
        assert isinstance(Vh.spaces[0].spaces[0], TensorFemSpace)
        assert isinstance(Vh.spaces[0].spaces[0], TensorFemSpace)

    V = Vh.symbolic_space
    W = Wh.symbolic_space

    # domain_h = V0h.domain:  would be nice...
    u = element_of(W, name='ululu')
    v = element_of(V, name='vlvlv')

    if isinstance(u, ScalarFunction):
        expr   = u*v
    else:
        expr   = dot(u,v)
    a = BilinearForm((u,v), integral(domain_h.domain, expr))
    ah = discretize(a, domain_h, [Wh, Vh], backend=PSYDAC_BACKENDS['python'])
    K1 = ah.assemble().tosparse().tolil()

    K = lil_matrix((Vh.nbasis, Wh.nbasis))

    l2g_V = Local2GlobalIndexMap(ndim, n_patches, n_components)
    l2g_W = Local2GlobalIndexMap(ndim, n_patches, n_components)
    for k in range(n_patches):
        Vk = Vh.spaces[k]
        Wk = Wh.spaces[k]
        # T is a TensorFemSpace and S is a 1D SplineSpace
        if n_components > 1:
            Vk_scalar_spaces = Vk.spaces
            Wk_scalar_spaces = Wk.spaces

        else:
            Vk_scalar_spaces = [Vk]
            Wk_scalar_spaces = [Wk]

        V_shapes = [[S.nbasis for S in T.spaces] for T in Vk_scalar_spaces]
        W_shapes = [[S.nbasis for S in T.spaces] for T in Wk_scalar_spaces]

        l2g_V.set_patch_shapes(k, *V_shapes)
        l2g_W.set_patch_shapes(k, *W_shapes)

        for d in range(n_components):
            # compute products Lambda_V_{d,i} . Lambda_W_{d,j}
            # -- hard-coded assumption: Lambda_V_{c,i} . Lambda_W_{d,j} = 0 for c≠d

            Vk_d = Vk_scalar_spaces[d]
            Wk_d = Wk_scalar_spaces[d]
            assert isinstance(Vk_d, TensorFemSpace)             

            K_1D = [None]*ndim
            multi_index_i = [None]*ndim
            multi_index_j = [None]*ndim

            for axis in range(ndim):
                V_space_axis = Vk_d.spaces[axis]
                W_space_axis = Wk_d.spaces[axis]
                
                K_1D_axis = lil_matrix((V_space_axis.nbasis, W_space_axis.nbasis))

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

                            K_1D_axis[i_glob,j_glob] += np.dot(V_quad_basis[ie,i_loc,0,:] * W_quad_basis[ie,j_loc,0,:], quad_w[ie,:])
                
                K_1D[axis] = K_1D_axis

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

                            K[ig,jg] = K_1D[0][i0,j0] * K_1D[1][i1,j1]

    print(K[0,0], K1[0,0])
    print(K[1,3], K1[1,3])
    print(K[3,3], K1[3,3])   
    print(K[25,27], K1[25,27])  
    #print(K-K1)
    exit()
    return K
            
