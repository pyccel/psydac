from spl.linalg.stencil import StencilVector, StencilMatrix

def apply_homogeneous_dirichlet_bc_1d(V, *args):
    """ Apply homogeneous dirichlet boundary conditions in 1D """

    # assumes a 1D spline space
    # add asserts on the space if it is periodic

    for a in args:
        if isinstance(a, StencilVector):
            a[0] = 0.
            a[V.nbasis-1] = 0.
        elif isinstance(a, StencilMatrix):
            a[ 0,:] = 0.
            a[-1,:] = 0.
        else:
            TypeError('> Expecting StencilVector or StencilMatrix')

def apply_homogeneous_dirichlet_bc_2d(V, *args):
    """ Apply homogeneous dirichlet boundary conditions in 2D """

    # assumes a 2D Tensor space
    # add asserts on the space if it is periodic

    V1,V2 = V.spaces

    s1, s2 = V.vector_space.starts
    e1, e2 = V.vector_space.ends

    for a in args:
        if not V1.periodic:
            if isinstance(a, StencilVector):
                # left  bc at x=0.
                if s1 == 0:
                    a[0,:] = 0.

                # right bc at x=1.
                if e1 == V1.nbasis-1:
                    a [e1,:] = 0.

            elif isinstance(a, StencilMatrix):
                # left  bc at x=0.
                if s1 == 0:
                    a[0,:,:,:] = 0.

                # right bc at x=1.
                if e1 == V1.nbasis-1:
                    a[e1,:,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')

        if not V2.periodic:
            if isinstance(a, StencilVector):
                # lower bc at y=0.
                if s2 == 0:
                    a [:,0] = 0.

                # upper bc at y=1.
                if e2 == V2.nbasis-1:
                    a [:,e2] = 0.

            elif isinstance(a, StencilMatrix):
                # lower bc at y=0.
                if s2 == 0:
                    a[:,0,:,:] = 0.
                # upper bc at y=1.
                if e2 == V2.nbasis-1:
                    a[:,e2,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')

def apply_homogeneous_dirichlet_bc_3d(V, *args):
    """ Apply homogeneous dirichlet boundary conditions in 3D """

    # assumes a 3D Tensor space
    # add asserts on the space if it is periodic

    V1,V2,V3 = V.spaces

    s1, s2, s3 = V.vector_space.starts
    e1, e2, e3 = V.vector_space.ends

    for a in args:
        if not V1.periodic:
            if isinstance(a, StencilVector):
                # left  bc at x=0.
                if s1 == 0:
                    a[0,:,:] = 0.

                # right bc at x=1.
                if e1 == V1.nbasis-1:
                    a [e1,:,:] = 0.

            elif isinstance(a, StencilMatrix):
                # left  bc at x=0.
                if s1 == 0:
                    a[0,:,:,:,:,:] = 0.

                # right bc at x=1.
                if e1 == V1.nbasis-1:
                    a[e1,:,:,:,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')

        if not V2.periodic:
            if isinstance(a, StencilVector):
                # lower bc at y=0.
                if s2 == 0:
                    a [:,0,:] = 0.

                # upper bc at y=1.
                if e2 == V2.nbasis-1:
                    a [:,e2,:] = 0.

            elif isinstance(a, StencilMatrix):
                # lower bc at y=0.
                if s2 == 0:
                    a[:,0,:,:,:,:] = 0.
                # upper bc at y=1.
                if e2 == V2.nbasis-1:
                    a[:,e2,:,:,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')

        if not V3.periodic:
            if isinstance(a, StencilVector):
                # lower bc at z=0.
                if s3 == 0:
                    a [:,:,0] = 0.

                # upper bc at z=1.
                if e3 == V3.nbasis-1:
                    a [:,:,e3] = 0.

            elif isinstance(a, StencilMatrix):
                # lower bc at z=0.
                if s3 == 0:
                    a[:,:,0,:,:,:] = 0.
                # upper bc at z=1.
                if e3 == V3.nbasis-1:
                    a[:,:,e3,:,:,:] = 0.

            else:
                TypeError('> Expecting StencilVector or StencilMatrix')


def apply_homogeneous_dirichlet_bc(V, *args):
    if V.ldim == 1:
        apply_homogeneous_dirichlet_bc_1d(V, *args)

    elif V.ldim == 2:
        apply_homogeneous_dirichlet_bc_2d(V, *args)

    elif V.ldim == 3:
        apply_homogeneous_dirichlet_bc_3d(V, *args)
