import numpy as np

from sympde.topology            import element_of
from sympde.calculus            import dot
from sympde.expr                import LinearForm, integral, EssentialBC
from sympde.topology.datatype   import SpaceType

from psydac.api.essential_bc    import apply_essential_bc
from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.core.bsplines       import hrefinement_matrix
from psydac.fem.basic           import FemSpace
from psydac.linalg.basic        import LinearOperator, Vector
from psydac.linalg.kron         import KroneckerDenseMatrix
from psydac.linalg.stencil      import StencilVectorSpace, StencilVector

__all__ = ('knots_to_insert', 'knot_insertion_projection_operator', 'get_dual_dofs',
           'DirichletBoundaryProjector', 'DirichletMultipatchBoundaryProjector')

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

#===============================================================================
class DirichletBoundaryProjector(LinearOperator):

    def __init__(self, fem_space, bcs=None, space_kind=None):

        assert isinstance(fem_space, FemSpace)

        coeff_space    = fem_space.coeff_space
        self._domain   = coeff_space
        self._codomain = coeff_space

        if bcs is not None:
            self._bcs = bcs
        else:
            self._bcs = self._get_bcs(fem_space, space_kind=space_kind)

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._domain
    
    @property
    def dtype(self):
        return None
    
    @property
    def bcs(self):
        return self._bcs
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        raise NotImplementedError
    
    def transpose(self, conjugate=False):
        return self
    
    def _get_bcs(self, fem_space, space_kind=None):
        """Returns the correct Dirichlet boundary conditions for the passed fem_space."""
        space    = fem_space.symbolic_space
        periodic = fem_space.periodic

        space_kind_str = space.kind.name
        if space_kind is not None:
            # Check whether kind is a valid input
            if isinstance(space_kind, str):
                kind_str = space_kind.lower()
                assert(kind_str in ['h1', 'hcurl', 'hdiv', 'l2', 'undefined'])
            elif isinstance(space_kind, SpaceType):
                kind_str = space_kind.name
            else:
                raise TypeError(f'Expecting space_kind {space_kind} to be a str or of SpaceType')
            
            # If fem_space has a kind, it must be compatible with kind
            if space_kind_str != 'undefined':
                assert space_kind_str == kind_str, f'fem_space and space_kind are not compatible.'
            else:
                # If space_kind_str = 'undefined': Update the variable using kind
                space_kind_str = kind_str


        kind = space_kind_str
        dim  = space.domain.dim
        
        if kind == 'l2':
            return None
        
        u = element_of(space, name="u")
        ebcs = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]

        if kind == "h1":
            bcs = [ebcs[0], ebcs[1]] if periodic[0] == False else []
            if dim >= 2:
                bcs += [ebcs[2], ebcs[3]] if periodic[1] == False else []
            if dim == 3:
                bcs += [ebcs[4], ebcs[5]] if periodic[2] == False else []

        elif kind == 'hcurl':
            assert dim in (2, 3)
            bcs_x = [ebcs[2], ebcs[3]] if periodic[1] == False else []
            if dim == 3:
                bcs_x += [ebcs[4], ebcs[5]] if periodic[2] == False else []
            bcs_y = [ebcs[0], ebcs[1]] if periodic[0] == False else []
            if dim == 3:
                bcs_y += [ebcs[4], ebcs[5]] if periodic[2] == False else []
            if dim == 3:
                bcs_z = [ebcs[0], ebcs[1]] if periodic[0] == False else []
                bcs_z += [ebcs[2], ebcs[3]] if periodic[1] == False else []
            bcs = [bcs_x, bcs_y]
            if dim == 3:
                bcs.append(bcs_z)

        elif kind == 'hdiv':
            assert dim in (2, 3)
            bcs_x = [ebcs[0], ebcs[1]] if periodic[0] == False else []
            bcs_y = [ebcs[2], ebcs[3]] if periodic[1] == False else []
            if dim == 3:
                bcs_z = [ebcs[4], ebcs[5]] if periodic[2] == False else []
            bcs = [bcs_x, bcs_y]
            if dim == 3:
                bcs.append(bcs_z)

        else:
            raise ValueError(f'{kind} must be either "h1", "hcurl" or "hdiv"')
        
        return bcs

    def dot(self, v, out=None):
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)
        if isinstance(self.domain, StencilVectorSpace):
            apply_essential_bc(out, *self._bcs)
        else:
            for block, block_bcs in zip(out, self._bcs):
                apply_essential_bc(block, *block_bcs)

        return out

#===============================================================================
class DirichletMultipatchBoundaryProjector(LinearOperator):

    def __init__(self, fem_space, bcs=None, space_kind=None):

        assert isinstance(fem_space, FemSpace)
        assert fem_space.is_multipatch

        coeff_space    = fem_space.coeff_space
        self._domain   = coeff_space
        self._codomain = coeff_space

        if bcs is not None:
            self._bcs = bcs
        else:
            self._bcs = self._get_bcs(fem_space, space_kind=space_kind)

    @property
    def domain(self):
        return self._domain
    
    @property
    def codomain(self):
        return self._domain
    
    @property
    def dtype(self):
        return None
    
    @property
    def bcs(self):
        return self._bcs
    
    def tosparse(self):
        raise NotImplementedError
    
    def toarray(self):
        raise NotImplementedError
    
    def transpose(self, conjugate=False):
        return self
    
    def _get_bcs(self, fem_space, space_kind=None):
        """Returns the correct Dirichlet boundary conditions for the passed fem_space."""
        space    = fem_space.symbolic_space
        periodic = fem_space.periodic

        space_kind_str = space.kind.name
        if space_kind is not None:
            # Check whether kind is a valid input
            if isinstance(space_kind, str):
                kind_str = space_kind.lower()
                assert(kind_str in ['h1', 'hcurl', 'hdiv', 'l2', 'undefined'])
            elif isinstance(space_kind, SpaceType):
                kind_str = space_kind.name
            else:
                raise TypeError(f'Expecting space_kind {space_kind} to be a str or of SpaceType')
            
            # If fem_space has a kind, it must be compatible with kind
            if space_kind_str != 'undefined':
                assert space_kind_str == kind_str, f'fem_space and space_kind are not compatible.'
            else:
                # If space_kind_str = 'undefined': Update the variable using kind
                space_kind_str = kind_str

        kind = space_kind_str
        dim  = space.domain.dim
        assert dim==2 
        
        if kind == 'l2':
            return None
        
        u = element_of(space, name="u")

        if kind == "h1":
            bcs = [EssentialBC(u, 0, side, position=0) for side in space.domain.boundary]


        elif kind == 'hcurl':
            bcs_x = []
            bcs_y = []

            for bn in space.domain.boundary:
                if bn.axis == 0:
                    bcs_y.append(EssentialBC(u, 0, bn, position=0))
                elif bn.axis == 1:
                    bcs_x.append(EssentialBC(u, 0, bn, position=0))

            bcs = [bcs_x, bcs_y]


        elif kind == 'hdiv':
            bcs_x = []
            bcs_y = []

            for bn in space.domain.boundary:
                if bn.axis == 1:
                    bcs_y.append(EssentialBC(u, 0, bn, position=0))
                elif bn.axis == 0:
                    bcs_x.append(EssentialBC(u, 0, bn, position=0))

            bcs = [bcs_x, bcs_y]

        else:
            raise ValueError(f'{kind} must be either "h1", "hcurl" or "hdiv"')
        
        return bcs

    def dot(self, v, out=None):
        if out is not None:
            assert isinstance(out, Vector)
            assert out.space is self.codomain
        else:
            out = self.codomain.zeros()

        v.copy(out=out)

        # apply bc on each patch
        for p in out.blocks:

            if isinstance(p, StencilVector):
                apply_essential_bc(p, *self._bcs)
            else:
                for block, block_bcs in zip(p, self._bcs):
                    apply_essential_bc(block, *block_bcs)

        return out
