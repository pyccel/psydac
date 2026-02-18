#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympde.expr.equation  import EssentialBC

from psydac.linalg.basic   import ComposedLinearOperator
from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.stencil import StencilInterfaceMatrix
from psydac.linalg.kron    import KroneckerDenseMatrix
from psydac.linalg.block   import BlockVector, BlockLinearOperator

__all__ = ('apply_essential_bc', 'check_boundary_type', 'apply_essential_bc_kronecker_dense_matrix', 'apply_essential_bc_stencil', 
           'apply_essential_bc_BlockLinearOperator', 'apply_essential_bc_BlockVector')
#==============================================================================
def apply_essential_bc(a, *bcs, **kwargs):

    if isinstance(a, (StencilVector, StencilMatrix, StencilInterfaceMatrix)):
        kwargs.pop('is_broken', None)
        for bc in bcs:
            check_boundary_type(bc)
            apply_essential_bc_stencil(a,
                axis  = bc.boundary.axis,
                ext   = bc.boundary.ext,
                order = bc.order,
                **kwargs
            )

    elif isinstance(a, ComposedLinearOperator):
        apply_essential_bc(a.multiplicants[0], *bcs, **kwargs)

    elif isinstance(a, KroneckerDenseMatrix):
        for bc in bcs:
            check_boundary_type(bc)
            apply_essential_bc_kronecker_dense_matrix(a,
                axis  = bc.boundary.axis,
                ext   = bc.boundary.ext,
                order = bc.order,
                **kwargs
            )
    elif isinstance(a, BlockVector):
        is_broken=kwargs.pop('is_broken', True)
        for bc in bcs:
            check_boundary_type(bc)
            apply_essential_bc_BlockVector(a, bc, is_broken=is_broken)

    elif isinstance(a, BlockLinearOperator):
        for bc in bcs:
            check_boundary_type(bc)
            apply_essential_bc_BlockLinearOperator(a, bc, **kwargs)

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
def apply_essential_bc_kronecker_dense_matrix(a, *, axis, ext, order, identity=False):
    """ This function applies the homogeneous boundary condition to the Kronecker product matrix objects,
        If the identity keyword argument is set to True, the boundary diagonal terms are set to 1.

    Parameters
    ----------
    a : KroneckerDenseMatrix
        The matrix to be modified.

    axis : int
        Axis of the boundary, i.e. the index of the coordinate which remains constant.

    ext : int
        Extremity of the boundary, it takes the value of -1 or 1.

    order : int
        All function derivatives up to `order` are set to zero
        on the specified boundary. `order >= 0` is required.

    identity : bool
        If true, the diagonal terms corresponding to boundary coefficients are set to 1.
    """

    mats = a.mats
    p = a.codomain.pads[axis]

    if ext == 1:
        mats[axis][-p-1] = 0.
    elif ext == -1:
        mats[axis][p] = 0.

    if identity and ext == 1:
        mats[axis][-p-1,mats[axis].shape[0]-2*p-1] = 1
    elif identity and ext == -1:
        mats[axis][p][0] = 1

#==============================================================================
def apply_essential_bc_stencil(a, *, axis, ext, order, identity=False):
    """ This function applies the homogeneous boundary condition to the Stencil objects,
        by setting the boundary degrees of freedom to zero in the StencilVector,
        and the corresponding rows in the StencilMatrix/StencilInterfaceMatrix to zeros.
        If the identity keyword argument is set to True, the boundary diagonal terms are set to 1.

    Parameters
    ----------
    a : StencilVector, StencilMatrix or StencilInterfaceMatrix
        The matrix or the Vector to be modified.

    axis : int
        Axis of the boundary, i.e. the index of the coordinate which remains constant.

    ext : int
        Extremity of the boundary, it takes the value of -1 or 1.

    order : int
        All function derivatives up to `order` are set to zero
        on the specified boundary. `order >= 0` is required.

    identity : bool
        If True, the diagonal terms corresponding to boundary coefficients are set to 1.
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

        if axis == a.codomain_axis:
            return
    else:
        raise TypeError('Cannot apply essential BC to object {} of type {}'\
                .format(a, type(a)))

    if V.parallel and V.cart.is_comm_null:
        return

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
def apply_essential_bc_BlockLinearOperator(a, bc, *, identity=False, is_broken=True):
    """
    Apply homogeneous dirichlet boundary conditions in nD.
    is_broken is used to identify if we are in a multipatch setting, where we assume
    that the domain and codomain of each block of the BlockLinearOperator corresponds to a single patch.

    Parameters
    ----------
    a : BlockLinearOperator
        The BlockLinearOperator to be modified.
 
    bc: Sympde.expr.equation.BasicBoundaryCondition
        The boundary condition type that will be applied to a.

    is_broken: bool
        Set to True if we are in a multipatch setting and False otherwise.
    """

    assert isinstance(a, BlockLinearOperator)
    keys = a.nonzero_block_indices

    is_broken = bc.variable.space.is_broken and is_broken
    if bc.index_component and not is_broken:
        for i_loc in bc.index_component:
            i = bc.position + i_loc
            js = [ij[1] for ij in keys if ij[0] == i]
            for j in js:
                apply_essential_bc(a[i, j], bc, identity=(identity and i==j))

    elif bc.position is not None and not is_broken:
        i = bc.position
        js = [ij[1] for ij in keys if ij[0] == i]
        for j in js:
            apply_essential_bc(a[i, j], bc, identity=(identity and i==j))
    elif is_broken:
        space = bc.variable.space
        domains = space.domain.interior.args
        assert len(a.blocks) == len(domains)
        bd = bc.boundary.domain
        i  = domains.index(bd)
        js = [ij[1] for ij in keys if ij[0] == i]
        for j in js:
            apply_essential_bc(a[i, j], bc, identity=(identity and i==j), is_broken=False)

#==============================================================================
def apply_essential_bc_BlockVector(a, bc, *, is_broken=True):
    """ Apply homogeneous dirichlet boundary conditions in nD.
        is_broken is used to identify if we are in a multipatch setting, where we assume
        each block of the BlockVector corresponds to a different patch.

    Parameters
    ----------
    a : BlockVector
        The BlockVector to be modified.
 
    bc: Sympde.expr.equation.BasicBoundaryCondition
        The boundary condition type that will be applied to a.

    is_broken: bool
        Set to True if we are in a multipatch setting and False otherwise. 
    """

    assert isinstance(a, BlockVector)

    is_broken = bc.variable.space.is_broken and is_broken
    if bc.index_component and not is_broken:
        for i_loc in bc.index_component:
            i = bc.position + i_loc
            apply_essential_bc(a[i], bc)
    elif bc.position is not None and not is_broken:
        i = bc.position
        apply_essential_bc(a[i], bc)
    elif is_broken:
        space = bc.variable.space
        domains = space.domain.interior.args
        bd = bc.boundary.domain
        assert len(a.blocks) == len(domains)
        i  = domains.index(bd)
        apply_essential_bc(a[i], bc, is_broken=False)
