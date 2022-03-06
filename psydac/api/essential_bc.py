# coding: utf-8
from sympde.expr.equation  import EssentialBC

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.stencil import StencilInterfaceMatrix, ProductLinearOperator
from psydac.linalg.block   import BlockVector, BlockMatrix
from psydac.linalg.kron    import KroneckerDenseMatrix

__all__ = ('apply_essential_bc',)

#==============================================================================
def apply_essential_bc(a, *bcs, **kwargs):

    if isinstance(a, (StencilVector, StencilMatrix, StencilInterfaceMatrix)):
        for bc in bcs:
            check_boundary_type(bc)
            apply_essential_bc_stencil(a,
                axis  = bc.boundary.axis,
                ext   = bc.boundary.ext,
                order = bc.order,
                **kwargs
            )

    elif isinstance(a, ProductLinearOperator):
        apply_essential_bc(a.operators[0], *bcs, **kwargs)

    elif isinstance(a, KroneckerDenseMatrix):
        for bc in bcs:
            check_boundary_type(bc)
            apply_essential_bc_kronnecker_dense_matrix(a,
                axis  = bc.boundary.axis,
                ext   = bc.boundary.ext,
                order = bc.order,
                **kwargs
            )
    elif isinstance(a, BlockVector):
        for bc in bcs:
            check_boundary_type(bc)
            apply_essential_bc_BlockVector(a, bc)

    elif isinstance(a, BlockMatrix):
        for bc in bcs:
            check_boundary_type(bc)
            apply_essential_bc_BlockMatrix(a, bc, **kwargs)

    else:
        raise TypeError('Cannot apply essential BCs to object of type {}'\
                .format(type(a)))

#==============================================================================
def check_boundary_type(bc):
    if not isinstance(bc, EssentialBC):
        raise TypeError('Essential boundary condition must be of type '\
                'EssentialBC from sympde.expr.equation, got {} instead'\
                .format(type(bc)))
#==============================================================================
def apply_essential_bc_kronnecker_dense_matrix(a, *, axis, ext, order, identity=False):
    mats = a.mats
    p = a.codomain.pads[axis]

    if ext == 1:
        mats[axis][-p-1] = 0.
    elif ext == -1:
        mats[axis][p] = 0.

#==============================================================================
def apply_essential_bc_stencil(a, *, axis, ext, order, identity=False):
    """ This function applies the homogeneous boundary condition to the Stencil objects,
        by setting the boundary degrees of freedom to zero in the StencilVector,
        and the corresponding rows in the StencilMatrix/StencilInterfaceMatrix to zeros.
        If the identity keyword argument is set to True, the boundary diagonal terms are set to 1.

    Parameters
    ----------
    a : StencilVector, StencilMatrix or StencilInterfaceMatrix
        The Matrix or the Vector that will be modified.

    axis : int
        Axis of the boundary.

    ext : int
        Extremity of the boundary, it takes the value of -1 or 1.

    order : int
        All function derivatives up to `order` are set to zero
        on the specified boundary. `order >= 0` is required.

    identity : bool
        If true, the diagonal terms corresponding to boundary coefficients are set to 1.
    """

    if isinstance(a, StencilVector):
        V = a.space
        n = V.ndim
    elif isinstance(a, StencilMatrix):
        V = a.codomain
        n = V.ndim * 2
    elif isinstance(a, StencilInterfaceMatrix):
        V = a.codomain
        n = V.ndim * 2
        if axis == a._dim:
            return
    else:
        raise TypeError('Cannot apply essential BC to object {} of type {}'\
                .format(a, type(a)))

    if axis not in range(V.ndim):
        raise ValueError('Cannot apply essential BC along axis x{} in {}D'\
                .format(axis + 1, V.ndim))

    if ext not in (-1, 1):
        raise ValueError("Argument 'ext' can only be -1 or 1, got {} instead"\
                .format(ext))

    if not isinstance(order, int) or order < 0:
        raise ValueError("Argument 'order' must be a non-negative integer, got "
                "{} instead".format(order))

    if V.periods[axis]:
        raise ValueError('Cannot apply essential BC along periodic direction '\
                'x{}'.format(axis + 1))

    if ext == -1 and V.starts[axis] == 0:
        s = V.starts[axis]
        index = [(s + order if j == axis else slice(None)) for j in range(n)]
        a[tuple(index)] = 0.0
        if isinstance(a, StencilMatrix) and identity:
            a[tuple(index[:n//2])+(0,)*(n//2)] = 1.

    elif ext == 1 and V.ends[axis] == V.npts[axis] - 1:
        e = V.ends[axis]
        index = [(e - order if j == axis else slice(None)) for j in range(n)]
        a[tuple(index)] = 0.0
        if isinstance(a, StencilMatrix) and identity:
            a[tuple(index[:n//2])+(0,)*(n//2)] = 1.
    else:
        pass

#==============================================================================
def apply_essential_bc_BlockMatrix(a, bc, identity=False):
    """ Apply homogeneous dirichlet boundary conditions in nD """

    assert isinstance(a, BlockMatrix)
    keys = list(a._blocks.keys())

    if bc.index_component is not None:
        for i_loc in bc.index_component:
            i = bc.position + i_loc
            js = [ij[1] for ij in keys if ij[0] == i]
            for j in js:
                apply_essential_bc(a[i, j], bc, identity=(identity and i==j))

    elif bc.position is not None and not bc.variable.space.is_broken:
        i = bc.position
        js = [ij[1] for ij in keys if ij[0] == i]
        for j in js:
            apply_essential_bc(a[i, j], bc, identity=(identity and i==j))
    else:
        var = bc.variable
        space = var.space
        if space.is_broken:
            domains = space.domain.interior.args
            bd = bc.boundary.domain
            i  = domains.index(bd)
            js = [ij[1] for ij in keys if ij[0] == i]
            for j in js:
                apply_essential_bc(a[i, j], bc, identity=(identity and i==j))

#==============================================================================
def apply_essential_bc_BlockVector(a, bc):
    """ Apply homogeneous dirichlet boundary conditions in nD """

    assert isinstance(a, BlockVector)

    if bc.index_component:
        for i_loc in bc.index_component:
            i = bc.position + i_loc
            apply_essential_bc(a[i], bc)
    else:
        var = bc.variable
        space = var.space
        if space.is_broken:
            domains = space.domain.interior.args
            bd = bc.boundary.domain
            i  = domains.index(bd)
            apply_essential_bc(a[i], bc)
