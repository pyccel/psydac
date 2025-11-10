import numpy as np

from sympde.topology            import element_of
from sympde.topology.space      import ScalarFunction
from sympde.topology.mapping    import Mapping
from sympde.calculus            import dot
from sympde.expr.expr           import LinearForm, integral

from psydac.api.settings        import PSYDAC_BACKENDS

from psydac.linalg.kron         import KroneckerDenseMatrix
from psydac.core.bsplines       import hrefinement_matrix
from psydac.linalg.stencil      import StencilVectorSpace
from psydac.fem.basic           import FemSpace

__all__ = ('knots_to_insert', 'knot_insertion_projection_operator')

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

    return KroneckerDenseMatrix(domain.coeff_space, codomain.coeff_space, *ops)


def get_dual_dofs(Vh, f, domain_h, backend_language="python", return_format='stencil_array'):
    """
    return the dual dofs tilde_sigma_i(f) = < Lambda_i, f >_{L2} i = 1, .. dim(Vh)) of a given function f, as a stencil array or numpy array

    Parameters
    ----------
    Vh : FemSpace
        The discrete space for the dual dofs

    f : <sympy.Expr>
        The function used for evaluation

    domain_h : 
        The discrete domain corresponding to Vh

    backend_language: <str>
        The backend used to accelerate the code

    return_format: <str>
        The format of the dofs, can be 'stencil_array' or 'numpy_array'

    Returns
    -------
    tilde_f: <Vector|ndarray>
        The dual dofs
    """

    from psydac.api.discretization  import discretize 

    assert isinstance(Vh, FemSpace)

    V  = Vh.symbolic_space
    v  = element_of(V, name='v')

    if Vh.is_vector_valued: 
        expr   = dot(f,v)
    else:
        expr   = f*v

    l        = LinearForm(v, integral( V.domain, expr))
    lh       = discretize(l, domain_h, Vh, backend=PSYDAC_BACKENDS[backend_language])
    tilde_f  = lh.assemble()

    if return_format == 'numpy_array':
        return tilde_f.toarray()
    else:
        return tilde_f
