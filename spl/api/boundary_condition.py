# coding: utf-8

# TODO remove V from apply_dirichlet_bc functions => get info from vector/matrix

from sympde.core import Boundary as sym_Boundary

from spl.linalg.stencil import StencilVector, StencilMatrix
from spl.linalg.block   import BlockVector, BlockMatrix

class DiscreteBoundary(object):
    _expr = None
    _axis = None
    _ext  = None

    def __init__(self, expr, axis=None, ext=None):
        if not isinstance(expr, sym_Boundary):
            raise TypeError('> Expecting a Boundary object')

        if not(axis) and not(ext):
            msg = '> for the moment, both axis and ext must be given'
            raise NotImplementedError(msg)

        self._expr = expr
        self._axis = axis
        self._ext = ext

    @property
    def expr(self):
        return self._expr

    @property
    def axis(self):
        return self._axis

    @property
    def ext(self):
        return self._ext

    # TODO how to improve? use TotalBoundary?
    def __neg__(self):
        return DiscreteComplementBoundary(self)

    def __add__(self, other):
        if isinstance(other, DiscreteComplementBoundary):
            raise TypeError('> Cannot add complement of boundary')

        return DiscreteUnionBoundary(self, other)

class DiscreteUnionBoundary(DiscreteBoundary):

    def __init__(self, *boundaries):
        # ...
        if isinstance(boundaries, DiscreteBoundary):
            boundaries = [boundaries]

        elif not isinstance(boundaries, (list, tuple)):
            raise TypeError('> Wrong type for boundaries')
        # ...

        self._boundaries = boundaries

        dim = boundaries[0].expr.domain.dim

        bnd_axis_ext = [(i.axis, i.ext) for i in boundaries]

        self._axis = [i[0] for i in bnd_axis_ext]
        self._ext  = [i[1] for i in bnd_axis_ext]

    @property
    def boundaries(self):
        return self._boundaries

class DiscreteComplementBoundary(DiscreteBoundary):

    def __init__(self, *boundaries):
        # ...
        if isinstance(boundaries, DiscreteBoundary):
            boundaries = [boundaries]

        elif not isinstance(boundaries, (list, tuple)):
            raise TypeError('> Wrong type for boundaries')
        # ...

        # ...
        new = []
        for bnd in boundaries:
            if isinstance(bnd, DiscreteUnionBoundary):
                new += bnd.boundaries
        if new:
            boundaries = new
        # ...

        self._boundaries = boundaries

        dim = boundaries[0].expr.domain.dim

        all_axis_ext = [(axis, ext) for axis in range(0, dim) for ext in [-1, 1]]
        bnd_axis_ext = [(i.axis, i.ext) for i in boundaries]
        cmp_axis_ext = set(all_axis_ext) - set(bnd_axis_ext)

        self._axis = [i[0] for i in cmp_axis_ext]
        self._ext  = [i[1] for i in cmp_axis_ext]

    @property
    def boundaries(self):
        return self._boundaries


class DiscreteBoundaryCondition(object):

    def __init__(self, boundary, value=None):
        self._boundary = boundary
        self._value = value

    @property
    def boundary(self):
        return self._boundary

    @property
    def value(self):
        return self._value

    def duplicate(self, B):
        return DiscreteBoundaryCondition(B, self.value)

class DiscreteDirichletBC(DiscreteBoundaryCondition):

    def duplicate(self, B):
        return DiscreteDirichletBC(B)


def apply_homogeneous_dirichlet_bc_1d_StencilVector(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 1D """

    # assumes a 1D spline space
    # add asserts on the space if it is periodic

    if bc.boundary.ext == -1:
        a[0] = 0.

    if bc.boundary.ext == 1:
        a[V.nbasis-1] = 0.

def apply_homogeneous_dirichlet_bc_1d_StencilMatrix(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 1D """

    # assumes a 1D spline space
    # add asserts on the space if it is periodic

    if bc.boundary.ext == -1:
        a[ 0,:] = 0.

    if bc.boundary.ext == 1:
        a[-1,:] = 0.

def apply_homogeneous_dirichlet_bc_2d_StencilVector(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 2D """

    # assumes a 2D Tensor space
    # add asserts on the space if it is periodic

    V1,V2 = V.spaces

    s1, s2 = a.space.starts
    e1, e2 = a.space.ends

    if bc.boundary.axis == 0:
        # left  bc.boundary.at x=0.
        if s1 == 0 and bc.boundary.ext == -1:
            a[0,:] = 0.

        # right bc.boundary.at x=1.
        if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
            a [e1,:] = 0.

    if bc.boundary.axis == 1:
        # lower bc.boundary.at y=0.
        if s2 == 0 and bc.boundary.ext == -1:
            a [:,0] = 0.

        # upper bc.boundary.at y=1.
        if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
            a [:,e2] = 0.

def apply_homogeneous_dirichlet_bc_2d_StencilMatrix(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 2D """

    # assumes a 2D Tensor space
    # add asserts on the space if it is periodic

    V1,V2 = V.spaces

    s1, s2 = a.domain.starts
    e1, e2 = a.domain.ends

    if bc.boundary.axis == 0:
        # left  bc.boundary.at x=0.
        if s1 == 0 and bc.boundary.ext == -1:
            a[0,:,:,:] = 0.

        # right bc.boundary.at x=1.
        if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
            a[e1,:,:,:] = 0.

    if bc.boundary.axis == 1:
        # lower bc.boundary.at y=0.
        if s2 == 0 and bc.boundary.ext == -1:
            a[:,0,:,:] = 0.

        # upper bc.boundary.at y=1.
        if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
            a[:,e2,:,:] = 0.


# V is a ProductFemSpace here
def apply_homogeneous_dirichlet_bc_2d_BlockMatrix(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 2D """

    for ij, M in a._blocks.items():
        i_row, i_col = ij
        # TODO must use col space too
        W = V.spaces[i_row]
        apply_homogeneous_dirichlet_bc(W, bc, M)


# V is a ProductFemSpace here
def apply_homogeneous_dirichlet_bc_2d_BlockVector(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 2D """

    n_blocks = a.n_blocks
    for i in range(0, n_blocks):
        M = a[i]
        # TODO must use col space too
        W = V.spaces[i]
        apply_homogeneous_dirichlet_bc(W, bc, M)


def apply_homogeneous_dirichlet_bc_3d_StencilVector(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 3D """

    # assumes a 3D Tensor space
    # add asserts on the space if it is periodic

    V1,V2,V3 = V.spaces

    s1, s2, s3 = a.space.starts
    e1, e2, e3 = a.space.ends

    if bc.boundary.axis == 0:
        # left  bc at x=0.
        if s1 == 0 and bc.boundary.ext == -1:
            a[0,:,:] = 0.

        # right bc at x=1.
        if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
            a [e1,:,:] = 0.

    if bc.boundary.axis == 1:
        # lower bc at y=0.
        if s2 == 0 and bc.boundary.ext == -1:
            a [:,0,:] = 0.

        # upper bc at y=1.
        if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
            a [:,e2,:] = 0.

    if bc.boundary.axis == 2:
        # lower bc at z=0.
        if s3 == 0 and bc.boundary.ext == -1:
            a [:,:,0] = 0.

        # upper bc at z=1.
        if e3 == V3.nbasis-1 and bc.boundary.ext == 1:
            a [:,:,e3] = 0.

def apply_homogeneous_dirichlet_bc_3d_StencilMatrix(V, bc, a):
    """ Apply homogeneous dirichlet boundary conditions in 3D """

    # assumes a 3D Tensor space
    # add asserts on the space if it is periodic

    V1,V2,V3 = V.spaces

    s1, s2, s3 = a.domain.starts
    e1, e2, e3 = a.domain.ends

    if bc.boundary.axis == 0:
        # left  bc at x=0.
        if s1 == 0 and bc.boundary.ext == -1:
            a[0,:,:,:,:,:] = 0.

        # right bc at x=1.
        if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
            a[e1,:,:,:,:,:] = 0.

    if bc.boundary.axis == 1:
        # lower bc at y=0.
        if s2 == 0 and bc.boundary.ext == -1:
            a[:,0,:,:,:,:] = 0.

        # upper bc at y=1.
        if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
            a[:,e2,:,:,:,:] = 0.

    if bc.boundary.axis == 2:
        # lower bc at z=0.
        if s3 == 0 and bc.boundary.ext == -1:
            a[:,:,0,:,:,:] = 0.

        # upper bc at z=1.
        if e3 == V3.nbasis-1 and bc.boundary.ext == 1:
            a[:,:,e3,:,:,:] = 0.


# TODO must pass two spaces for a matrix
def apply_homogeneous_dirichlet_bc(V, bc, *args):

    _avail_classes = [StencilVector, StencilMatrix,
                      BlockVector, BlockMatrix]
    for a in args:
        classes = type(a).__mro__
        classes = set(classes) & set(_avail_classes)
        classes = list(classes)
        if not classes:
            raise TypeError('> wrong argument type {}'.format(type(a)))

        cls = classes[0]

        pattern = 'apply_homogeneous_dirichlet_bc_{dim}d_{name}'
        apply_bc = pattern.format( dim = V.ldim, name = cls.__name__ )
        apply_bc = eval(apply_bc)
        apply_bc(V, bc, a)
