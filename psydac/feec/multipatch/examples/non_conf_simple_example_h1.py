import os
# from typing import Mapping
import numpy as np

from scipy.sparse import coo_matrix, bmat
from scipy.sparse.linalg import inv as sp_inv

from sympde.topology      import Square    
from sympde.topology      import IdentityMapping
from psydac.fem.vector import ProductFemSpace

# from psydac.api.discretization import discretize #  ???
from psydac.feec.multipatch.api import discretize
from psydac.api.settings   import PSYDAC_BACKENDS
from psydac.feec.multipatch.plotting_utilities          import plot_field
from sympde.topology  import Derham

from psydac.feec.multipatch.utilities                   import time_count
from psydac.linalg.block import BlockVector
from psydac.linalg.utilities                            import array_to_stencil
from psydac.fem.basic                                   import FemField

def run_simple_2patch_example(nc=2, deg=2):

    plot_dir = 'run_plots_nc={}_deg={}'.format(nc,deg)

    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    ncells = [nc, nc]
    degree = [deg,deg]

    print(' .. multi-patch domain...')
    # domain = build_multipatch_domain(domain_name='two_patch_nc')

    A = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, 1))
    M1 = IdentityMapping('M1', dim=2)
    M2 = IdentityMapping('M2', dim=2)
    A = M1(A)
    B = M2(B)

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

    # mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    # mappings_list = list(mappings.values())

    # for diagnosttics
    # diag_grid = DiagGrid(mappings=mappings, N_diag=100)
    backend_language = 'python'

    t_stamp = time_count()
    print(' .. derham sequence...')
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. discrete domain...')
    # domain_h = discretize(domain, ncells=ncells_h)   # Vh space
    # domain_hc = discretize(domain, ncells=[nc,nc])  # coarse Vh space
    domain_hc = discretize(domain, ncells=ncells_c)  # coarse Vh space
    domain_hf = discretize(domain, ncells=ncells_f)  # fine Vh space

    t_stamp = time_count(t_stamp)
    print(' .. discrete derham sequence...')
    # derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])
    derham_hc = discretize(derham, domain_hc, degree=degree, backend=PSYDAC_BACKENDS[backend_language])
    derham_hf = discretize(derham, domain_hf, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    t_stamp = time_count(t_stamp)
    print(' .. conforming projection operators...')
    # print(' .. commuting projection operators...')
    # nquads = [4*(d + 1) for d in degree]
    # P0, P1, P2 = derham_h.projectors(nquads=nquads)

    cP0_c = derham_hc.conforming_projection(space='V0', hom_bc=False, backend_language=backend_language)
    cP0_f = derham_hf.conforming_projection(space='V0', hom_bc=False, backend_language=backend_language)

    t_stamp = time_count(t_stamp)
    print(' .. spaces...')

    V0h_c = derham_hc.V0
    V0h_f = derham_hf.V0
    V0h_h = ProductFemSpace(V0h_f.spaces[0],V0h_c.spaces[1])  # fine space on patch 0, coarse on patch 1

    c2f_patch1 = construct_projection_operator(domain=V0h_c.spaces[1], codomain=V0h_f.spaces[1])

    cf2_t = c2f_patch1.transpose()
    product = cf2_t @ c2f_patch1

    inv_prod = sp_inv(product.tocsc())
    f2c_patch1 = inv_prod @ cf2_t

    E1 = c2f_patch1
    E1_star = f2c_patch1

    # f2c_patch0 = c2f_patch0.transpose()


    # cP0 = BlockMatrix(domain=V0h_h.vector_space, self.codomain=V0h_h.vector_space)
    # cP0[0,0] = c2f_patch1,  ...

    # numpy:
    cP0_c_00 = cP0_c.matrix[0,0].tosparse()
    cP0_c_10 = cP0_c.matrix[1,0].tosparse()
    cP0_c_01 = cP0_c.matrix[0,1].tosparse()
    cP0_c_11 = cP0_c.matrix[1,1].tosparse()

    cP0_f_00 = cP0_f.matrix[0,0].tosparse()
    cP0_f_10 = cP0_f.matrix[1,0].tosparse()
    cP0_f_01 = cP0_f.matrix[0,1].tosparse()
    cP0_f_11 = cP0_f.matrix[1,1].tosparse()

    # same for diff operators
    # ...

    print(E1.shape)
    print(cP0_c_00.shape)
    print(V0h_f.nbasis)

    cP0_m = bmat([
        [E1 @ cP0_c_00 @ E1_star,      E1 @ cP0_c_01],
        [     cP0_c_10 @ E1_star,           cP0_c_11]
    ])

    cP0_m = bmat([
         [       cP0_f_00 ,        cP0_f_01 @ E1],
#         [E1.T @ cP0_f_10 , E1.T @ cP0_f_11 @ E1]
         [E1.T @ cP0_f_10 ,        cP0_c_11]
#         [E1_star @ cP0_f_10 , cP0_c_11]
     ])

#    cP0_m = bmat([
#         [       cP0_f_00 ,           cP0_f_01 @ E1],
#         [E1.T @ cP0_f_10 ,    E1.T @ cP0_f_11 @ E1]
#     ])

    print(cP0_m.shape)

    G_sol_log = [lambda xi1, xi2, ii=i : ii+xi1+xi2**2 for i in range(len(domain))]   

    P0c, P1c, P2c = derham_hc.projectors()
    P0f, P1f, P2f = derham_hf.projectors()

    G0c   = P0c(G_sol_log)
    G0f   = P0f(G_sol_log)
    
    ## test c2f matrix

    # plot G_0 and G_1
    plot_field(numpy_coeffs=G0c.coeffs.toarray(), Vh=V0h_c, space_kind='h1',
            domain=domain, title='Gc', cmap='viridis',
            filename=plot_dir+'/Gc.png')

    # plot EG_0 and G_1
    G0c_patch0_coeffs = G0c.coeffs[0].toarray()
    G0c_patch1_coeffs = G0c.coeffs[1].toarray()

#    EG0_patch1_coeffs = E1 @ G0c_patch0_coeffs

#    G0h_coeffs = np.block([EG0_patch1_coeffs, G0c_patch1_coeffs])
#    
#    plot_field(numpy_coeffs=G0h_coeffs, Vh=V0h_h, space_kind='h1',
#            domain=domain, title='EGc', cmap='viridis',
#            filename=plot_dir+'/EGc.png')


    ## apply conforming P on hybrid space

    G0f_patch0_coeffs = G0f.coeffs[0].toarray()
    G0c_patch1_coeffs = G0c.coeffs[1].toarray()

    print('G0f_patch0_coeffs', G0f_patch0_coeffs)
    print('------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ')
    print('G0c_patch1_coeffs', G0c_patch1_coeffs)

    G0h_coeffs = np.block([G0f_patch0_coeffs,G0c_patch1_coeffs])
    G0h = FemField(V0h_h, coeffs=array_to_stencil(G0h_coeffs, V0h_h.vector_space))
    
    plot_field(numpy_coeffs=G0h_coeffs, Vh=V0h_h, space_kind='h1',
            domain=domain, title='G', cmap='viridis',
            filename=plot_dir+'/G.png')

    PG0h_coeffs = cP0_m @ G0h_coeffs
    PG0h        = FemField(V0h_h, coeffs=array_to_stencil(PG0h_coeffs, V0h_h.vector_space))
    plot_field(fem_field=PG0h, Vh=V0h_h, space_kind='h1',
            domain=domain, title='PGh', cmap='viridis',
            filename=plot_dir+'/PGh.png')


    PG0c = cP0_c(G0c)

    plot_field(fem_field=PG0c, Vh=V0h_c, space_kind='h1',
            domain=domain, title='PGc', cmap='viridis',
            filename=plot_dir+'/PGc.png')

    PG0f = cP0_f(G0f)

    plot_field(fem_field=PG0f, Vh=V0h_f, space_kind='h1',
            domain=domain, title='PGf', cmap='viridis',
            filename=plot_dir+'/PGf.png')

    # G0c  = Pconf_1(G0)  # should be curl-conforming




def knots_to_insert(coarse_grid, fine_grid, tol=1e-14):
#    assert len(coarse_grid)*2-2 == len(fine_grid)-1
    intersection = coarse_grid[(np.abs(fine_grid[:,None] - coarse_grid) < tol).any(0)]
    assert abs(intersection-coarse_grid).max()<tol
    T = fine_grid[~(np.abs(coarse_grid[:,None] - fine_grid) < tol).any(0)]
    return T

def construct_projection_operator(domain, codomain):
    from psydac.core.interface import matrix_multi_stages
    from scipy.sparse          import csr_matrix

    ops = []
    for d,c in zip(domain.spaces, codomain.spaces):
        if d.ncells>c.ncells:
            Ts = knots_to_insert(c.breaks, d.breaks)
            P  = matrix_multi_stages(Ts, c.nbasis , c.degree, c.knots)
            ops.append(P.T)
            raise NotImplementedError

        elif d.ncells<c.ncells:
            Ts = knots_to_insert(d.breaks, c.breaks)
            P  = matrix_multi_stages(Ts, d.nbasis , d.degree, d.knots)
            if d.basis == 'M':
                assert c.basis == 'M'
                P = np.diag(1/c._scaling_array) @ P @ np.diag(d._scaling_array)
            ops.append(P)
        else:
            P   = np.eye(d.nbasis) #IdentityStencilMatrix(StencilVectorSpace([d.nbasis], [d.degree], [d.periodic]))
            ops.append(P.toarray())

    # return KroneckerDenseMatrix(domain.vector_space, codomain.vector_space, *ops)
    return csr_matrix(np.kron(*ops))


if __name__ == '__main__':
    
    run_simple_2patch_example(nc=14, deg=2)
