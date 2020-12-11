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
            self.func = evaluate_dof_0form_1d
        elif len(self.N) == 2:
            self.func = evaluate_dof_0form_2d
        elif len(self.N) == 3:
            self.func = evaluate_dof_0form_3d
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
            self.func = evaluate_dof_1form_3d
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
            self.func = evaluate_dof_1form_2d
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
            self.func = evaluate_dof_2form_3d
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
            self.func = evaluate_dof_2form_2d
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
            self.func = evaluate_dof_3form_1d
        elif len(self.D) == 2:
            self.func = evaluate_dof_3form_2d
        elif len(self.D) == 3:
            self.func = evaluate_dof_3form_3d
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
def evaluate_dof_0form_1d(n1, points_1, F, f):
    for i1 in range(n1):
        F[i1] = f(points_1[i1])
        
def evaluate_dof_0form_2d(n1, n2, points_1, points_2, F, f):
    for i1 in range(n1):
        for i2 in range(n2):
            F[i1,i2] = f(points_1[i1], points_2[i2])

def evaluate_dof_0form_3d(n1, n2, n3, points_1, points_2, points_3, F, f):
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F[i1, i2, i3] = f(points_1[i1], points_2[i2], points_3[i3])
               
#==============================================================================
def evaluate_dof_1form_2d(
        intp_x1, intp_x2, # interpolation points
        quad_x1, quad_x2, # quadrature points
        quad_w1, quad_w2, # quadrature weights
        F1, F2,           # arrays of degrees of freedom (intent out)
        f1, f2            # input scalar functions (callable)
        ):

    k1 = quad_x1.shape[1]
    k2 = quad_x2.shape[1]

    n1, n2 = F1.shape
    for i2 in range(n1):
        for i1 in range(n2):
            for g1 in range(k1):
                F1[i1, i2] += quad_w1[i1, g1]*f1(quad_x1[i1, g1], intp_x2[i2])

    n1, n2 = F2.shape               
    for i1 in range(n1):
        for i2 in range(n2):
            for g2 in range(k2):
                F2[i1, i2] += quad_w2[i2, g2]*f2(intp_x1[i1], quad_x2[i2, g2])

def evaluate_dof_1form_3d(
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
                    F1[i1, i2, i3] += quad_w1[i1, g1] * f1(quad_x1[i1, g1], intp_x2[i2], intp_x3[i3])

    n1, n2, n3 = F2.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F2[i1, i2, i3] = 0.0
                for g2 in range(k2):
                    F2[i1, i2, i3] += quad_w2[i2, g2] * f2(intp_x1[i1], quad_x2[i2, g2], intp_x3[i3])

    n1, n2, n3 = F3.shape
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                F3[i1, i2, i3] = 0.0
                for g3 in range(k3):
                    F3[i1, i2, i3] += quad_w3[i3, g3] * f3(intp_x1[i1], intp_x2[i2], quad_x3[i3, g3])

#==============================================================================
def evaluate_dof_2form_2d(     
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
            for g2 in range(k2):
                F1[i1, i2] += quad_w2[i2, g2]*f1(intp_x1[i1],quad_x2[i2, g2])

    n1, n2 = F2.shape
    for i2 in range(n1):
        for i1 in range(n2):          
            for g1 in range(k1):
                F2[i1, i2] += quad_w1[i1, g1]*f2(quad_x1[i1, g1],intp_x2[i2])
                

def evaluate_dof_2form_3d(
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

#==============================================================================
def evaluate_dof_3form_1d(points, weights, F, fun):
    """Integrates the function f over the quadrature grid
    defined by (points,weights) in 1d.

    points: np.array
        a multi-dimensional array describing the quadrature points mapped onto
        the grid. it must be constructed using construct_quadrature_grid

    weights: np.array
        a multi-dimensional array describing the quadrature weights (scaled) mapped onto
        the grid. it must be constructed using construct_quadrature_grid

    Examples

    >>> from psydac.core.interface import make_open_knots
    >>> from psydac.core.interface import construct_grid_from_knots
    >>> from psydac.core.interface import construct_quadrature_grid
    >>> from psydac.core.interface import compute_greville
    >>> from psydac.utilities.quadratures import gauss_legendre

    >>> n_elements = 8
    >>> p = 2                    # spline degree
    >>> n = n_elements + p - 1   # number of control points
    >>> T = make_open_knots(p, n)
    >>> grid = compute_greville(p, n, T)
    >>> u, w = gauss_legendre(p)  # gauss-legendre quadrature rule
    >>> k = len(u)
    >>> ne = len(grid) - 1        # number of elements
    >>> points, weights = construct_quadrature_grid(ne, k, u, w, grid)
    >>> f = lambda u: u*(1.-u)
    >>> f_int = integrate(points, weights, f)
    >>> f_int
    [0.00242954 0.01724976 0.02891156 0.03474247 0.03474247 0.02891156
     0.01724976 0.00242954]
    n = points.shape[0]
    k = points.shape[1]
    """
    n1 = points.shape[0]
    k1 = points.shape[1]

    for ie1 in range(n1):
        for g1 in range(k1):
            F[ie1] += weights[ie1, g1]*fun(points[ie1, g1])

def evaluate_dof_3form_2d(points_1, points_2, weights_1, weights_2, F, fun):

    """Integrates the function f over the quadrature grid
    defined by (points,weights) in 2d.

    points: list, tuple
        list of quadrature points, as they should be passed for `integrate`

    weights: list, tuple
        list of quadrature weights, as they should be passed for `integrate`

    Examples

    """

    n1 = points_1.shape[0]
    n2 = points_2.shape[0]

    k1 = points_1.shape[1]
    k2 = points_2.shape[1]

    for ie1 in range(n1):
        for ie2 in range(n2):
            for g1 in range(k1):
                for g2 in range(k2):
                    F[ie1, ie2] += weights_1[ie1, g1]*weights_2[ie2, g2]*fun(points_1[ie1, g1], points_2[ie2, g2])


def evaluate_dof_3form_3d(points_1, points_2, points_3,  weights_1, weights_2, weights_3, F, fun):

    n1 = points_1.shape[0]
    n2 = points_2.shape[0]
    n3 = points_3.shape[0]

    k1 = points_1.shape[1]
    k2 = points_2.shape[1]
    k3 = points_3.shape[1]

    for ie1 in range(n1):
        for ie2 in range(n2):
            for ie3 in range(n3):
                for g1 in range(k1):
                    for g2 in range(k2):
                        for g3 in range(k3):
                            F[ie1, ie2, ie3] += weights_1[ie1, g1]*weights_2[ie2, g2]*weights_3[ie3, g3]\
                                                       *fun(points_1[ie1, g1], points_2[ie2, g2], points_3[ie3, g3])
