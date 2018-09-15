# coding: utf-8

# TODO remove V from apply_dirichlet_bc functions => get info from vector/matrix

from sympde.core import Boundary as sym_Boundary

from spl.linalg.stencil import StencilVector, StencilMatrix

class DiscreteBoundary(object):
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

class DiscreteDirichletBC(DiscreteBoundaryCondition):
    pass


# TODO set bc
def apply_homogeneous_dirichlet_bc_1d(V, bc, *args):
    """ Apply homogeneous dirichlet boundary conditions in 1D """

    # assumes a 1D spline space
    # add asserts on the space if it is periodic

    for a in args:
        if isinstance(a, StencilVector):
            if bc.boundary.ext == -1:
                a[0] = 0.

            if bc.boundary.ext == 1:
                a[V.nbasis-1] = 0.

        elif isinstance(a, StencilMatrix):
            if bc.boundary.ext == -1:
                a[ 0,:] = 0.

            if bc.boundary.ext == 1:
                a[-1,:] = 0.

        else:
            TypeError('> Expecting StencilVector or StencilMatrix')

def apply_homogeneous_dirichlet_bc_2d(V, bc, *args):
    """ Apply homogeneous dirichlet boundary conditions in 2D """

    # assumes a 2D Tensor space
    # add asserts on the space if it is periodic

    V1,V2 = V.spaces

    for a in args:
        if isinstance(a, StencilVector):
            s1, s2 = a.space.starts
            e1, e2 = a.space.ends

        elif isinstance(a, StencilMatrix):
            s1, s2 = a.domain.starts
            e1, e2 = a.domain.ends

        if bc.boundary.axis == 0:
            if isinstance(a, StencilVector):
                # left  bc.boundary.at x=0.
                if s1 == 0 and bc.boundary.ext == -1:
                    a[0,:] = 0.

                # right bc.boundary.at x=1.
                if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
                    a [e1,:] = 0.

            elif isinstance(a, StencilMatrix):
                # left  bc.boundary.at x=0.
                if s1 == 0 and bc.boundary.ext == -1:
                    a[0,:,:,:] = 0.

                # right bc.boundary.at x=1.
                if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
                    a[e1,:,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')

        if bc.boundary.axis == 1:
            if isinstance(a, StencilVector):
                # lower bc.boundary.at y=0.
                if s2 == 0 and bc.boundary.ext == -1:
                    a [:,0] = 0.

                # upper bc.boundary.at y=1.
                if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
                    a [:,e2] = 0.

            elif isinstance(a, StencilMatrix):
                # lower bc.boundary.at y=0.
                if s2 == 0 and bc.boundary.ext == -1:
                    a[:,0,:,:] = 0.
                # upper bc.boundary.at y=1.
                if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
                    a[:,e2,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')

# TODO set bc
def apply_homogeneous_dirichlet_bc_3d(V, bc, *args):
    """ Apply homogeneous dirichlet boundary conditions in 3D """

    # assumes a 3D Tensor space
    # add asserts on the space if it is periodic

    V1,V2,V3 = V.spaces

    for a in args:
        if isinstance(a, StencilVector):
            s1, s2, s3 = a.space.starts
            e1, e2, e3 = a.space.ends

        elif isinstance(a, StencilMatrix):
            s1, s2, s3 = a.domain.starts
            e1, e2, e3 = a.domain.ends

        if bc.boundary.axis == 0:
            if isinstance(a, StencilVector):
                # left  bc at x=0.
                if s1 == 0 and bc.boundary.ext == -1:
                    a[0,:,:] = 0.

                # right bc at x=1.
                if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
                    a [e1,:,:] = 0.

            elif isinstance(a, StencilMatrix):
                # left  bc at x=0.
                if s1 == 0 and bc.boundary.ext == -1:
                    a[0,:,:,:,:,:] = 0.

                # right bc at x=1.
                if e1 == V1.nbasis-1 and bc.boundary.ext == 1:
                    a[e1,:,:,:,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')

        if bc.boundary.axis == 1:
            if isinstance(a, StencilVector):
                # lower bc at y=0.
                if s2 == 0 and bc.boundary.ext == -1:
                    a [:,0,:] = 0.

                # upper bc at y=1.
                if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
                    a [:,e2,:] = 0.

            elif isinstance(a, StencilMatrix):
                # lower bc at y=0.
                if s2 == 0 and bc.boundary.ext == -1:
                    a[:,0,:,:,:,:] = 0.

                # upper bc at y=1.
                if e2 == V2.nbasis-1 and bc.boundary.ext == 1:
                    a[:,e2,:,:,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')

        if bc.boundary.axis == 2:
            if isinstance(a, StencilVector):
                # lower bc at z=0.
                if s3 == 0 and bc.boundary.ext == -1:
                    a [:,:,0] = 0.

                # upper bc at z=1.
                if e3 == V3.nbasis-1 and bc.boundary.ext == 1:
                    a [:,:,e3] = 0.

            elif isinstance(a, StencilMatrix):
                # lower bc at z=0.
                if s3 == 0 and bc.boundary.ext == -1:
                    a[:,:,0,:,:,:] = 0.

                # upper bc at z=1.
                if e3 == V3.nbasis-1 and bc.boundary.ext == 1:
                    a[:,:,e3,:,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')


def apply_homogeneous_dirichlet_bc(V, bc, *args):
    if V.ldim == 1:
        apply_homogeneous_dirichlet_bc_1d(V, bc, *args)

    elif V.ldim == 2:
        apply_homogeneous_dirichlet_bc_2d(V, bc, *args)

    elif V.ldim == 3:
        apply_homogeneous_dirichlet_bc_3d(V, bc, *args)
