# coding: utf-8

# TODO remove V from apply_dirichlet_bc functions => get info from vector/matrix

from sympde.topology import Boundary as sym_Boundary

from spl.linalg.stencil import StencilVector, StencilMatrix
from spl.linalg.block   import BlockVector, BlockMatrix


#==============================================================================
def apply_homogeneous_dirichlet_bc_1d_StencilVector(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 1D """

    # assumes a 1D spline space
    # add asserts on the space if it is periodic

    # get order
    order = bc.order

    if bc.boundary.ext == -1:
        a[0+order] = 0.

    if bc.boundary.ext == 1:
        a[V.nbasis-1-order] = 0.

#==============================================================================
def apply_homogeneous_dirichlet_bc_1d_StencilMatrix(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 1D """

    # assumes a 1D spline space
    # add asserts on the space if it is periodic

    # get order
    order = bc.order

    if bc.boundary.ext == -1:
        a[ 0+order,:] = 0.

    if bc.boundary.ext == 1:
        a[-1-order,:] = 0.

#==============================================================================
def apply_homogeneous_dirichlet_bc_2d_StencilVector(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 2D """

    # assumes a 2D Tensor space
    # add asserts on the space if it is periodic

    V1,V2 = V.spaces

    s1, s2 = a.space.starts
    e1, e2 = a.space.ends

    # get order
    order = bc.order

    if bc.boundary.axis == 0:
        # left  bc.boundary.at x=0.
        if s1 == 0 and bc.boundary.ext == -1:
            a[0+order,:] = 0.

        # right bc.boundary.at x=1.
        if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
            a [e1-order,:] = 0.

    if bc.boundary.axis == 1:
        # lower bc.boundary.at y=0.
        if s2 == 0 and bc.boundary.ext == -1:
            a [:,0+order] = 0.

        # upper bc.boundary.at y=1.
        if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
            a [:,e2-order] = 0.

#==============================================================================
def apply_homogeneous_dirichlet_bc_2d_StencilMatrix(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 2D """

    # assumes a 2D Tensor space
    # add asserts on the space if it is periodic

    V1,V2 = V.spaces

    s1, s2 = a.domain.starts
    e1, e2 = a.domain.ends

    # get order
    order = bc.order

    if bc.boundary.axis == 0:
        # left  bc.boundary.at x=0.
        if s1 == 0 and bc.boundary.ext == -1:
            a[0+order,:,:,:] = 0.

        # right bc.boundary.at x=1.
        if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
            a[e1-order,:,:,:] = 0.

    if bc.boundary.axis == 1:
        # lower bc.boundary.at y=0.
        if s2 == 0 and bc.boundary.ext == -1:
            a[:,0+order,:,:] = 0.

        # upper bc.boundary.at y=1.
        if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
            a[:,e2-order,:,:] = 0.



#==============================================================================
def apply_homogeneous_dirichlet_bc_3d_StencilVector(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 3D """

    # assumes a 3D Tensor space
    # add asserts on the space if it is periodic

    V1,V2,V3 = V.spaces

    s1, s2, s3 = a.space.starts
    e1, e2, e3 = a.space.ends

    # get order
    order = bc.order

    if bc.boundary.axis == 0:
        # left  bc at x=0.
        if s1 == 0 and bc.boundary.ext == -1:
            a[0+order,:,:] = 0.

        # right bc at x=1.
        if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
            a [e1-order,:,:] = 0.

    if bc.boundary.axis == 1:
        # lower bc at y=0.
        if s2 == 0 and bc.boundary.ext == -1:
            a [:,0+order,:] = 0.

        # upper bc at y=1.
        if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
            a [:,e2-order,:] = 0.

    if bc.boundary.axis == 2:
        # lower bc at z=0.
        if s3 == 0 and bc.boundary.ext == -1:
            a [:,:,0+order] = 0.

        # upper bc at z=1.
        if e3 == V3.nbasis-1 and bc.boundary.ext == 1:
            a [:,:,e3-order] = 0.

#==============================================================================
def apply_homogeneous_dirichlet_bc_3d_StencilMatrix(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 3D """

    # assumes a 3D Tensor space
    # add asserts on the space if it is periodic

    V1,V2,V3 = V.spaces

    s1, s2, s3 = a.domain.starts
    e1, e2, e3 = a.domain.ends

    # get order
    order = bc.order

    if bc.boundary.axis == 0:
        # left  bc at x=0.
        if s1 == 0 and bc.boundary.ext == -1:
            a[0+order,:,:,:,:,:] = 0.

        # right bc at x=1.
        if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
            a[e1-order,:,:,:,:,:] = 0.

    if bc.boundary.axis == 1:
        # lower bc at y=0.
        if s2 == 0 and bc.boundary.ext == -1:
            a[:,0+order,:,:,:,:] = 0.

        # upper bc at y=1.
        if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
            a[:,e2-order,:,:,:,:] = 0.

    if bc.boundary.axis == 2:
        # lower bc at z=0.
        if s3 == 0 and bc.boundary.ext == -1:
            a[:,:,0+order,:,:,:] = 0.

        # upper bc at z=1.
        if e3 == V3.nbasis-1 and bc.boundary.ext == 1:
            a[:,:,e3-order,:,:,:] = 0.


#==============================================================================
# V is a ProductFemSpace here
# TODO must use bc.position
def apply_homogeneous_dirichlet_bc_BlockMatrix(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in nD """

    for ij, M in a._blocks.items():
        i_row, i_col = ij
        # TODO must use col space too
        W = V.spaces[i_row]
        apply_homogeneous_dirichlet_bc(W, bc, M, order=order)


#==============================================================================
# V is a ProductFemSpace here
# TODO must use bc.position
def apply_homogeneous_dirichlet_bc_BlockVector(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in nD """

    n_blocks = a.n_blocks
    for i in range(0, n_blocks):
        M = a[i]
        # TODO must use col space too
        W = V.spaces[i]
        apply_homogeneous_dirichlet_bc(W, bc, M, order=order)


#==============================================================================
# TODO must pass two spaces for a matrix
def apply_essential_bc(V, bc, *args, **kwargs):

    if not isinstance(bc, (tuple, list)):
        bc = [bc]

    _avail_classes = [StencilVector, StencilMatrix,
                      BlockVector, BlockMatrix]
    for a in args:
        classes = type(a).__mro__
        classes = set(classes) & set(_avail_classes)
        classes = list(classes)
        if not classes:
            raise TypeError('> wrong argument type {}'.format(type(a)))

        cls = classes[0]

        if not isinstance(a, (BlockMatrix, BlockVector)):
            pattern = 'apply_homogeneous_dirichlet_bc_{dim}d_{name}'
            apply_bc = pattern.format( dim = V.ldim, name = cls.__name__ )

        else:
            pattern = 'apply_homogeneous_dirichlet_bc_{name}'
            apply_bc = pattern.format( name = cls.__name__ )

        apply_bc = eval(apply_bc)
        for b in bc:
            apply_bc(V, b, a, **kwargs)
