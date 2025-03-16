import numpy as np

from psydac.linalg.kron     import KroneckerDenseMatrix
from psydac.core.bsplines   import hrefinement_matrix
from psydac.linalg.stencil  import StencilVectorSpace

from psydac.fem.basic       import FemSpace

__all__ = ('knots_to_insert', 'knot_insertion_projection_operator', 'get_moments_of_function')

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
    """
    Compute the projection operator based on the knot insertion technique.

    Return a linear operator which projects an element of the domain to an
    element of the codomain. Domain and codomain are scalar spline spaces over
    a cuboid, built as the tensor product of 1D spline spaces. In particular,
    domain and codomain have the same multi-degree (p1, p2, ...).

    This function returns a LinearOperator K working at the level of the
    spline coefficients, which are represented by StencilVector objects.

    Thanks to the tensor-product structure of the spline spaces, the projection
    operator is the Kronecker product of 1D projection operators K[i] operating
    between 1D spaces. Each 1D operators is represented by a dense matrix:

        K = K[0] x K[1] x ...

    For each dimension i the 1D grids defined by the breakpoints of the two
    spaces are assumed to be identical, or one nested into the other. Let nd[i]
    and nc[i] be the number of cells along dimension i for domain and codomain,
    respectively. We then have three different cases:

    1. nd[i] == nc[i]:
       The two 1D grids are assumed identical, and K[i] is the identity matrix.

    2. nd[i] < nc[i]:
       The 1D grid of the domain is assumed nested into the 1D grid of the
       codomain, hence the 1D spline space of the domain is a subspace of the
       1D spline space of the codomain. In this case we build K[i] using the
       knot insertion algorithm.

    3. nd[i] > nc[i]:
       The 1D grid of the codomain is assumed nested into the 1D grid of the
       domain, hence the 1D spline space of the codomain is a subspace of the
       1D spline space of the domain. In this case we build K[i] as the
       transpose of the matrix obtained using the knot insertion algorithm from
       the codomain to the domain.

    Parameters
    ----------
    domain : TensorFemSpace
        Domain of the projection operator.

    codomain : TensorFemSpace
        Codomain of the projection operator.

    Returns
    -------
    KroneckerDenseMatrix
        Matrix representation of the projection operator. This is a
        LinearOperator acting on the spline coefficients.

    """
    ops = []
    for d, c in zip(domain.spaces, codomain.spaces):

        if d.ncells > c.ncells:
            Ts = knots_to_insert(c.breaks, d.breaks)
            P  = hrefinement_matrix(Ts, c.degree, c.knots)

            if d.basis == 'M':
                assert c.basis == 'M'
                P = np.diag(1 / d._scaling_array) @ P @ np.diag(c._scaling_array)

            ops.append(P.T)

        elif d.ncells < c.ncells:
            Ts = knots_to_insert(d.breaks, c.breaks)
            P  = hrefinement_matrix(Ts, d.degree, d.knots)

            if d.basis == 'M':
                assert c.basis == 'M'
                P = np.diag(1 / c._scaling_array) @ P @ np.diag(d._scaling_array)

            ops.append(P)

        else:
            ops.append(np.eye(d.nbasis))

    return KroneckerDenseMatrix(domain.vector_space, codomain.vector_space, *ops)

def get_moments_of_function(f, Vh, backend_language="python", return_format='stencil_array'):
    """
    return the integrals of some analytical f (given as symbolic expression) against the basis functions of some space Vh, i.e. the values    
        tilde_sigma_i(f) = < Lambda_i, f >_{L2}   for i = 1 .. dim(Vh)
    
    note: 
     - the values tilde_sigma_i(f) correspond to the dofs of f in the dual basis of Vh
     - the coefficients c(f) of the L2 projection of f in Vh satisfy M @ c(f) = tilde_sigma(f)
       where M is the mass matrix of Vh, with the basis Lambda.

    Parameters
    ----------
    Vh : FemSpace

    f : <sympy.Expr>

    backend_language: <str>
        The backend used to accelerate the code

    return_format: <str>
        The format of the dofs, can be 'stencil_array' or 'numpy_array'

    Returns
    -------
    tilde_f: <Vector|ndarray>
        The dual dofs of f
    """
    assert isinstance(Vh, FemSpace)

    V  = Vh.symbolic_space
    v  = element_of(V, name='v')

    if isinstance(v, ScalarFunction):
        expr   = f*v
    else:
        expr   = dot(f,v)

    l        = LinearForm(v, integral( V.domain, expr))
    lh       = discretize(l, self._domain_h, Vh, backend=PSYDAC_BACKENDS[backend_language])
    tilde_f  = lh.assemble()

    if return_format == 'numpy_array':
        return tilde_f.toarray()
    else:
        return tilde_f
