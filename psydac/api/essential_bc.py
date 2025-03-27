# coding: utf-8
from psydac.linalg.basic   import ComposedLinearOperator
from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.stencil import StencilInterfaceMatrix
from psydac.linalg.kron    import KroneckerDenseMatrix
from psydac.linalg.block   import BlockVector, BlockLinearOperator


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

