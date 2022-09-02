import numpy as np

from psydac.linalg.kron     import KroneckerDenseMatrix
from psydac.core.bsplines   import hrefinement_matrix
from psydac.linalg.identity import IdentityStencilMatrix
from psydac.linalg.stencil  import StencilVectorSpace

def knots_to_insert(coarse_grid, fine_grid, tol=1e-14):
#    assert len(coarse_grid)*2-2 == len(fine_grid)-1
    intersection = coarse_grid[(np.abs(fine_grid[:,None] - coarse_grid) < tol).any(0)]
    assert abs(intersection-coarse_grid).max()<tol
    T = fine_grid[~(np.abs(coarse_grid[:,None] - fine_grid) < tol).any(0)]
    return T

def construct_projection_operator(domain, codomain):
    ops = []
    for d,c in zip(domain.spaces, codomain.spaces):
        if d.ncells>c.ncells:
            Ts = knots_to_insert(c.breaks, d.breaks)
            P  = hrefinement_matrix(Ts, c.degree, c.knots)

            if d.basis == 'M':
                assert c.basis == 'M'
                P = np.diag(1/d._scaling_array) @ P @ np.diag(c._scaling_array)

            ops.append(P.T)
        elif d.ncells<c.ncells:
            Ts = knots_to_insert(d.breaks, c.breaks)
            P  = hrefinement_matrix(Ts , d.degree, d.knots)

            if d.basis == 'M':
                assert c.basis == 'M'
                P = np.diag(1/c._scaling_array) @ P @ np.diag(d._scaling_array)

            ops.append(P)
        else:
            ops.append(np.eye(d.nbasis))

    return KroneckerDenseMatrix(domain.vector_space, codomain.vector_space, *ops)

