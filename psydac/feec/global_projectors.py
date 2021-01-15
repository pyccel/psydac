# -*- coding: UTF-8 -*-

from psydac.linalg.utilities      import array_to_stencil
from psydac.linalg.kron           import kronecker_solve
from psydac.linalg.stencil        import StencilVector
from psydac.linalg.block          import BlockVector
from psydac.core.bsplines         import quadrature_grid
from psydac.utilities.quadratures import gauss_legendre
from psydac.fem.basic             import FemField
from psydac.fem.vector            import VectorFemField

class Projector_H1:

    def __init__(self, H1):

        # Quadrature grids in cells defined by consecutive Greville points

        points = [V.greville for V in H1.spaces]

        H1.init_interpolation()

        # Collocation matrices for N-splines in each direction
        self.N   = [V._interpolator for V in H1.spaces]
        n_basis  = [V.nbasis for V in H1.spaces]
        self.rhs = StencilVector(H1.vector_space)
        slices   = tuple(slice(p,-p) for p in H1.degree)

        self.space = H1
        self.args  = (*n_basis, *points, self.rhs._data[slices])

        if len(self.N) == 1:
            self.func = evaluate_dofs_1d_0form
        elif len(self.N) == 2:
            self.func = evaluate_dofs_2d_0form
        elif len(self.N) == 3:
            self.func = evaluate_dofs_3d_0form
        else:
            raise ValueError('H1 projector of dimension {} not available'.format(str(len(self.N))))

    # ======================================
    def __call__(self, fun):
        r'''
        Projection on the space V0 via interpolation.

        Parameters
        ----------
        fun : callable
            fun(x) \in R is the 0-form to be projected.

        Returns
        -------
        coeffs : 1D array_like
            Finite element coefficients obtained by projection.
        '''

        # build the rhs
        self.func(*self.args, fun)

        if len(self.N)==1:
            rhs = self.rhs.toarray()
            coeffs = array_to_stencil(self.N[0].solve(rhs), self.space.vector_space)
            coeffs.update_ghost_regions()
        else:
            coeffs = kronecker_solve(solvers = self.N, rhs = self.rhs)

        return FemField(self.space, coeffs=coeffs)

class Projector_Hcurl:

    def __init__(self, Hcurl, n_quads=None):

        if n_quads:
            uw = [gauss_legendre( k-1 ) for k in n_quads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_grids[i].quad_rule_x,V.quad_grids[i].quad_rule_w) for i,V in enumerate(Hcurl.spaces)]

        dim = len(n_quads)

        self.space  = Hcurl
        self.rhs    = BlockVector(Hcurl.vector_space)
        self.dim    = dim
        self.mats   = [None]*dim

        for V in Hcurl.spaces:
            V.init_interpolation()
            V.init_histopolation()

        if dim == 3:

            # 1D spline spaces (B-splines of degree p and M-splines of degree p-1)
            Ns = [Hcurl.spaces[1].spaces[0], Hcurl.spaces[0].spaces[1], Hcurl.spaces[0].spaces[2]]
            Ds = [Hcurl.spaces[0].spaces[0], Hcurl.spaces[1].spaces[1], Hcurl.spaces[2].spaces[2]]

            # Package 1D interpolators and 1D histopolators for 3D Kronecker solver
            self.mats[0] = [Ds[0]._histopolator, Ns[1]._interpolator, Ns[2]._interpolator]
            self.mats[1] = [Ns[0]._interpolator, Ds[1]._histopolator, Ns[2]._interpolator]
            self.mats[2] = [Ns[0]._interpolator, Ns[1]._interpolator, Ds[2]._histopolator]

            # Interpolation points
            intp_x = [V.greville for V in Ns]

            # Quadrature points and weights
            quads = [quadrature_grid(V.histopolation_grid, u, w) for V,(u,w) in zip(Ds, uw)]
            quad_x, quad_w = list(zip(*quads))

            # Arrays of degrees of freedom (to be computed) as slices of RHS vector
            slices = tuple(slice(p, -p) for p in Hcurl.spaces[0].vector_space.pads)
            dofs   = [x._data[slices] for x in self.rhs]

            # Store data in object
            self.args = (*intp_x, *quad_x, *quad_w, *dofs)
            self.func = evaluate_dofs_3d_1form
            self.Ns = Ns
            self.Ds = Ds

        elif dim == 2:
            # 1D spline spaces (B-splines of degree p and M-splines of degree p-1)
            Ns = [Hcurl.spaces[1].spaces[0], Hcurl.spaces[0].spaces[1]]
            Ds = [Hcurl.spaces[0].spaces[0], Hcurl.spaces[1].spaces[1]]

            # Package 1D interpolators and 1D histopolators for 2D Kronecker solver
            self.mats[0] = [Ds[0]._histopolator, Ns[1]._interpolator]
            self.mats[1] = [Ns[0]._interpolator, Ds[1]._histopolator]

            # Interpolation points
            intp_x = [V.greville for V in Ns]

            # Quadrature points and weights
            quads = [quadrature_grid(V.histopolation_grid, u, w) for V,(u,w) in zip(Ds, uw)]
            quad_x, quad_w = list(zip(*quads))

            # Arrays of degrees of freedom (to be computed) as slices of RHS vector
            slices = tuple(slice(p, -p) for p in Hcurl.spaces[0].vector_space.pads)
            dofs   = [x._data[slices] for x in self.rhs]

            # Store data in object
            self.args = (*intp_x, *quad_x, *quad_w, *dofs)
            self.func = evaluate_dofs_2d_1form_hcurl
            self.Ns = Ns
            self.Ds = Ds
        else:
            raise NotImplementedError('only 3d and 2d are available')

    # ======================================
    def __call__(self, fun):

        # build the rhs
        self.func(*self.args, *fun)

        self.rhs.update_ghost_regions()

        coeffs    = BlockVector(self.space.vector_space)
        for i in range(self.dim):
            coeffs[i] = kronecker_solve(solvers = self.mats[i], rhs = self.rhs[i])

        coeffs.update_ghost_regions()
        return VectorFemField(self.space, coeffs=coeffs)

class Projector_Hdiv:

    def __init__(self, Hdiv, n_quads=None):

        if n_quads:
            uw = [gauss_legendre( k-1 ) for k in n_quads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_grids[i].quad_rule_x,V.quad_grids[i].quad_rule_w) for i,V in enumerate(Hdiv.spaces)]

        dim = len(n_quads)

        self.space  = Hdiv
        self.rhs    = BlockVector(Hdiv.vector_space)
        self.dim    = dim
        self.mats   = [None]*dim


        for V in Hdiv.spaces:
            V.init_interpolation()
            V.init_histopolation()

        if dim == 3:

            # 1D spline spaces (B-splines of degree p and M-splines of degree p-1)
            Ns = [Hdiv.spaces[0].spaces[0], Hdiv.spaces[1].spaces[1], Hdiv.spaces[2].spaces[2]]
            Ds = [Hdiv.spaces[1].spaces[0], Hdiv.spaces[0].spaces[1], Hdiv.spaces[0].spaces[2]]

            # Package 1D interpolators and 1D histopolators for 3D Kronecker solver
            self.mats[0] = [Ns[0]._interpolator, Ds[1]._histopolator, Ds[2]._histopolator]
            self.mats[1] = [Ds[0]._histopolator, Ns[1]._interpolator, Ds[2]._histopolator]
            self.mats[2] = [Ds[0]._histopolator, Ds[1]._histopolator, Ns[2]._interpolator]

            # Interpolation points
            intp_x = [V.greville for V in Ns]

            # Quadrature points and weights
            quads  = [quadrature_grid(V.histopolation_grid, u, w) for V,(u,w) in zip(Ds, uw)]
            quad_x, quad_w = list(zip(*quads))

            # Arrays of degrees of freedom (to be computed) as slices of RHS vector
            slices = tuple(slice(p,-p) for p in Hdiv.spaces[0].vector_space.pads)
            dofs   = [x._data[slices] for x in self.rhs]

            # Store data in object
            self.args = (*intp_x, *quad_x, *quad_w, *dofs)
            self.func = evaluate_dofs_3d_2form
            self.Ns = Ns
            self.Ds = Ds

        elif dim == 2:
            # 1D spline spaces (B-splines of degree p and M-splines of degree p-1)
            Ns = [Hdiv.spaces[0].spaces[0], Hdiv.spaces[1].spaces[1]]
            Ds = [Hdiv.spaces[1].spaces[0], Hdiv.spaces[0].spaces[1]]

            # Package 1D interpolators and 1D histopolators for 2D Kronecker solver
            self.mats[0] = [Ns[0]._interpolator, Ds[1]._histopolator]
            self.mats[1] = [Ds[0]._histopolator, Ns[1]._interpolator]

            # Interpolation points
            intp_x = [V.greville for V in Ns]

            # Quadrature points and weights
            quads  = [quadrature_grid(V.histopolation_grid, u, w) for V,(u,w) in zip(Ds, uw)]
            quad_x, quad_w = list(zip(*quads))

            # Arrays of degrees of freedom (to be computed) as slices of RHS vector
            slices = tuple(slice(p,-p) for p in Hdiv.spaces[0].vector_space.pads)
            dofs   = [x._data[slices] for x in self.rhs]

            # Store data in object
            self.args = (*intp_x, *quad_x, *quad_w, *dofs)
            self.func = evaluate_dofs_2d_1form_hdiv
            self.Ns = Ns
            self.Ds = Ds
        else:
            raise NotImplementedError('only 3d is available')

    # ======================================
    def __call__(self, fun):

        # build the rhs
        self.func(*self.args, *fun)

        self.rhs.update_ghost_regions()

        coeffs    = BlockVector(self.space.vector_space)

        for i in range(self.dim):
            coeffs[i] = kronecker_solve(solvers = self.mats[i], rhs = self.rhs[i])

        coeffs.update_ghost_regions()
        return VectorFemField(self.space, coeffs=coeffs)

class Projector_L2:

    def __init__(self, L2, quads=None):

        # Quadrature grids in cells defined by consecutive Greville points
        if quads:
            uw = [gauss_legendre( k-1 ) for k in quads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_rule_x,V.quad_rule_w) for V in L2.quad_grids]

        quads = [quadrature_grid(V.histopolation_grid, u, w) for V,(u,w) in zip(L2.spaces, uw)]
        points, weights = list(zip(*quads))

        L2.init_histopolation()

        # Histopolation matrices for D-splines in each direction
        self.D = [V._histopolator for V in L2.spaces]

        self.space = L2
        self.rhs   = StencilVector(L2.vector_space)
        slices     = tuple(slice(p+1,-p-1) for p in L2.degree)

        if len(self.D) == 1:
            self.func = evaluate_dofs_1d_1form
        elif len(self.D) == 2:
            self.func = evaluate_dofs_2d_2form
        elif len(self.D) == 3:
            self.func = evaluate_dofs_3d_3form
        else:
            raise ValueError('H1 projector of dimension {} not available'.format(str(len(self.N))))

        self.args  = (*points, *weights, self.rhs._data[slices])

    def __call__(self, fun):
        r'''
        Projection on the space V1 via histopolation.

        Parameters
        ----------
        fun : callable
            fun(x) \in R is the 1-form to be projected.

        Returns
        -------
        coeffs : Vector
            Finite element coefficients obtained by projection.
        '''

        # build the rhs
        self.func(*self.args, fun)

        if len(self.D) == 1:
            rhs = self.rhs.toarray()
            coeffs = array_to_stencil(self.D[0].solve(rhs), self.space.vector_space)
            coeffs.update_ghost_regions()
        else:
            coeffs = kronecker_solve(solvers = self.D, rhs = self.rhs)

        return FemField(self.space, coeffs=coeffs)

#==============================================================================
# 1D DEGREES OF FREEDOM
#==============================================================================

# TODO: cleanup
def evaluate_dofs_1d_0form(n1, points_1, F, f):
    for i1 in range(n1):
        F[i1] = f(points_1[i1])
        
#------------------------------------------------------------------------------
def evaluate_dofs_1d_1form(
        quad_x1, # quadrature points
        quad_w1, # quadrature weights
        F,       # array of degrees of freedom (intent out)
        f        # input scalar function (callable)
        ):

    k1 = quad_x1.shape[1]

    n1, = F.shape
    for i1 in range(n1):
        F[i1] = 0.0
        for g1 in range(k1):
            F[i1] += quad_w1[i1, g1] * f(quad_x1[i1, g1])

#==============================================================================
# 2D DEGREES OF FREEDOM
#==============================================================================

# TODO: cleanup
def evaluate_dofs_2d_0form(n1, n2, points_1, points_2, F, f):
    for i1 in range(n1):
        for i2 in range(n2):
            F[i1,i2] = f(points_1[i1], points_2[i2])

#------------------------------------------------------------------------------
def evaluate_dofs_2d_1form_hcurl(
        intp_x1, intp_x2, # interpolation points
        quad_x1, quad_x2, # quadrature points
        quad_w1, quad_w2, # quadrature weights
        F1, F2,           # arrays of degrees of freedom (intent out)
        f1, f2            # input scalar functions (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F1[i1, i2] = 0.0
            for g1 in range(k1):
                F1[i1, i2] += quad_w1[i1, g1] * f1(quad_x1[i1, g1], intp_x2[i2])

    n1, n2 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F2[i1, i2] = 0.0
            for g2 in range(k2):
                F2[i1, i2] += quad_w2[i2, g2] * f2(intp_x1[i1], quad_x2[i2, g2])

#------------------------------------------------------------------------------
def evaluate_dofs_2d_1form_hdiv(
        intp_x1, intp_x2, # interpolation points
        quad_x1, quad_x2, # quadrature points
        quad_w1, quad_w2, # quadrature weights
        F1, F2,           # arrays of degrees of freedom (intent out)
        f1, f2            # input scalar functions (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F1[i1, i2] = 0.0
            for g2 in range(k2):
                F1[i1, i2] += quad_w2[i2, g2] * f1(intp_x1[i1], quad_x2[i2, g2])

    n1, n2 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F2[i1, i2] = 0.0
            for g1 in range(k1):
                F2[i1, i2] += quad_w1[i1, g1] * f2(quad_x1[i1, g1], intp_x2[i2])

#------------------------------------------------------------------------------
def evaluate_dofs_2d_2form(
        quad_x1, quad_x2, # quadrature points
        quad_w1, quad_w2, # quadrature weights
        F,                # array of degrees of freedom (intent out)
        f,                # input scalar function (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F.shape
    for i1 in range(n1):
        for i2 in range(n2):
            F[i1, i2] = 0.0
            for g1 in range(k1):
                for g2 in range(k2):
                    F[i1, i2] += quad_w1[i1, g1] * quad_w2[i2, g2] * \
                            f(quad_x1[i1, g1], quad_x2[i2, g2])

#==============================================================================
# 3D DEGREES OF FREEDOM
#==============================================================================

# TODO: cleanup
def evaluate_dofs_3d_0form(n1, n2, n3, points_1, points_2, points_3, F, f):
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F[i1, i2, i3] = f(points_1[i1], points_2[i2], points_3[i3])

#------------------------------------------------------------------------------
def evaluate_dofs_3d_1form(
        intp_x1, intp_x2, intp_x3, # interpolation points
        quad_x1, quad_x2, quad_x3, # quadrature points
        quad_w1, quad_w2, quad_w3, # quadrature weights
        F1, F2, F3,                # arrays of degrees of freedom (intent out)
        f1, f2, f3                 # input scalar functions (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]
    k3 = quad_x3.shape[1]

    n1, n2, n3 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F1[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    F1[i1, i2, i3] += quad_w1[i1, g1] * \
                            f1(quad_x1[i1, g1], intp_x2[i2], intp_x3[i3])

    n1, n2, n3 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F2[i1, i2, i3] = 0.0
                for g2 in range(k2):
                    F2[i1, i2, i3] += quad_w2[i2, g2] * \
                            f2(intp_x1[i1], quad_x2[i2, g2], intp_x3[i3])

    n1, n2, n3 = F3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F3[i1, i2, i3] = 0.0
                for g3 in range(k3):
                    F3[i1, i2, i3] += quad_w3[i3, g3] * \
                            f3(intp_x1[i1], intp_x2[i2], quad_x3[i3, g3])

#------------------------------------------------------------------------------
def evaluate_dofs_3d_2form(
        intp_x1, intp_x2, intp_x3, # interpolation points
        quad_x1, quad_x2, quad_x3, # quadrature points
        quad_w1, quad_w2, quad_w3, # quadrature weights
        F1, F2, F3,                # arrays of degrees of freedom (intent out)
        f1, f2, f3                 # input scalar functions (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]
    k3 = quad_x3.shape[1]

    n1, n2, n3 = F1.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F1[i1, i2, i3] = 0.0
                for g2 in range(k2):
                    for g3 in range(k3):
                        F1[i1, i2, i3] += quad_w2[i2, g2] * quad_w3[i3, g3] * \
                            f1(intp_x1[i1], quad_x2[i2, g2], quad_x3[i3, g3])

    n1, n2, n3 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F2[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g3 in range(k3):
                        F2[i1, i2, i3] += quad_w1[i1, g1] * quad_w3[i3, g3] * \
                            f2(quad_x1[i1, g1], intp_x2[i2], quad_x3[i3, g3])

    n1, n2, n3 = F3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F3[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g2 in range(k2):
                        F3[i1, i2, i3] += quad_w1[i1, g1] * quad_w2[i2, g2] * \
                            f3(quad_x1[i1, g1], quad_x2[i2, g2], intp_x3[i3])

#------------------------------------------------------------------------------
def evaluate_dofs_3d_3form(
        quad_x1, quad_x2, quad_x3, # quadrature points
        quad_w1, quad_w2, quad_w3, # quadrature weights
        F,                         # array of degrees of freedom (intent out)
        f,                         # input scalar function (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]
    k3 = quad_x3.shape[1]

    n1, n2, n3 = F.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F[i1, i2, i3] = 0.0
                for g1 in range(k1):
                    for g2 in range(k2):
                        for g3 in range(k3):
                            F[i1, i2, i3] += \
                                    quad_w1[i1, g1] * quad_w2[i2, g2] * quad_w3[i3, g3] * \
                                    f(quad_x1[i1, g1], quad_x2[i2, g2], quad_x3[i3, g3])
