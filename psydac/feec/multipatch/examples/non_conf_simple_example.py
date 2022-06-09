import numpy as np

from scipy.sparse import coo_matrix, bmat

from sympde.topology      import Square    
from psydac.fem.vector import ProductFemSpace

from psydac.api.discretization import discretize #  ???
# from psydac.feec.multipatch.api import discretize
from psydac.api.settings   import PSYDAC_BACKENDS

from sympde.topology  import Derham

from psydac.feec.multipatch.utilities                   import time_count
from psydac.linalg.utilities                            import array_to_stencil
from psydac.fem.basic                                   import FemField

def run_simple_2patch_example(nc=2, deg=2):

    ncells = [nc, nc]
    degree = [deg,deg]

    print(' .. multi-patch domain...')
    # domain = build_multipatch_domain(domain_name='two_patch_nc')

    A = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, 1))

    domain = A.join(B, name = 'domain',
                bnd_minus = A.get_boundary(axis=0, ext=1),
                bnd_plus  = B.get_boundary(axis=0, ext=-1))

    nc = 2
    ncells_c = {
        'A':[nc, nc],
        'B':[nc, nc],
    }

    ncells_f = {
        'A':[2*nc, 2*nc],
        'B':[2*nc, 2*nc],
    }
    ncells_h = {
        'A':[2*nc, 2*nc],
        'B':[nc, nc],
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
    domain_hc = discretize(domain, ncells=[nc,nc])  # coarse Vh space
    # domain_hc = discretize(domain, ncells=ncells)  # coarse Vh space
    domain_hf = discretize(domain, ncells=ncells_f)  # fine Vh space

    t_stamp = time_count(t_stamp)
    print(' .. discrete derham sequence...')
    # derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])
    derham_hc = discretize(derham, domain_hc, degree=degree, backend=PSYDAC_BACKENDS[backend_language])
    derham_hf = discretize(derham, domain_hf, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    # t_stamp = time_count(t_stamp)
    # print(' .. commuting projection operators...')
    # nquads = [4*(d + 1) for d in degree]
    # P0, P1, P2 = derham_h.projectors(nquads=nquads)

    cP1_c = derham_hc.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language)
    cP1_f = derham_hf.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language)

    V1h_c = derham_hc.V1
    V1h_f = derham_hf.V1
    V1h_h = ProductFemSpace(V1h_f.spaces[0],V1h_c.spaces[1])  # fine space on patch 0, coarse on patch 1

    # matrix of coarse to fine change of basis (for patch 0)
    c2f_patch0 = construct_projection_operator(domain=V1h_c.spaces[0],codomain=V1h_f.spaces[0])

    f2c_patch0 = c2f_patch0.transpose()

    # cP1 = BlockMatrix(domain=V1h_h.vector_space, self.codomain=V1h_h.vector_space)
    # cP1[0,0] = c2f_patch1,  ...

    # numpy:
    cP1_c_00 = cP1_c[0,0].tosparse()
    cP1_c_10 = cP1_c[1,0].tosparse()
    cP1_c_01 = cP1_c[0,1].tosparse()
    cP1_c_11 = cP1_c[1,1].tosparse()

    cP1_f_00 = cP1_f[0,0].tosparse()
    cP1_f_10 = cP1_f[1,0].tosparse()
    cP1_f_01 = cP1_f[0,1].tosparse()
    cP1_f_11 = cP1_f[1,1].tosparse()

    # same for diff operators
    # ...

    print(c2f_patch0.shape)
    print(cP1_c_00.shape)
    print(V1h_f.nbasis)

    exit(   )

    cP1_m = np.block([
        [c2f_patch0 * cP1_c_00 * f2c_patch0, c2f_patch0 * cP1_c_01],
        [             cP1_c_10 * f2c_patch0,              cP1_c_11]
    ])



def knots_to_insert(coarse_grid, fine_grid, tol=1e-14):
#    assert len(coarse_grid)*2-2 == len(fine_grid)-1
    intersection = coarse_grid[(np.abs(fine_grid[:,None] - coarse_grid) < tol).any(0)]
    assert abs(intersection-coarse_grid).max()<tol
    T = fine_grid[~(np.abs(coarse_grid[:,None] - fine_grid) < tol).any(0)]
    return T

def construct_projection_operator(domain, codomain):
    from psydac.core.interface import matrix_multi_stages

    ops = []
    for d,c in zip(domain.spaces, codomain.spaces):
        if d.ncells>c.ncells:
            Ts = knots_to_insert(c.breaks, d.breaks)
            P  = matrix_multi_stages(Ts, c.nbasis , c.degree, c.knots)
            ops.append(P.T)
        elif d.ncells<c.ncells:
            Ts = knots_to_insert(d.breaks, c.breaks)
            P  = matrix_multi_stages(Ts, d.nbasis , d.degree, d.knots)
            ops.append(P)
        else:
            P   = np.eye(d.nbasis) #IdentityStencilMatrix(StencilVectorSpace([d.nbasis], [d.degree], [d.periodic]))
            ops.append(P.toarray())

    # return KroneckerDenseMatrix(domain.vector_space, codomain.vector_space, *ops)
    return np.kron(*ops)


if __name__ == '__main__':
    
    run_simple_2patch_example(nc=2, deg=2)
