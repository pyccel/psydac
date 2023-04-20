import numpy as np

from psydac.linalg.kron     import KroneckerDenseMatrix
from psydac.core.bsplines   import hrefinement_matrix
from psydac.linalg.identity import IdentityStencilMatrix
from psydac.linalg.stencil  import StencilVectorSpace

def knots_to_insert(coarse_grid, fine_grid, tol=1e-14):
    """ Compute the point difference between the fine grid and coarse grid."""
#    assert len(coarse_grid)*2-2 == len(fine_grid)-1
    indices1 =  (np.abs(fine_grid  [:,None] - coarse_grid) < tol).any(0)
    indices2 = ~(np.abs(coarse_grid[:,None] - fine_grid  ) < tol).any(0)

    intersection = coarse_grid[indices1]
    T            = fine_grid[indices2]

    assert abs(intersection-coarse_grid).max()<tol
    return T

def knot_insertion_projection_operator(domain, codomain):
    """ Compute the projection operator based on the knot insertion technique from the domain to the codomain.
        We assume that either domain is a subspace of the codomain or vice versa.
    """
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

