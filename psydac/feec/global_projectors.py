# -*- coding: UTF-8 -*-

from psydac.linalg.utilities      import array_to_stencil
from psydac.linalg.kron           import KroneckerLinearSolver
from psydac.linalg.stencil        import StencilVector
from psydac.linalg.block          import BlockDiagonalSolver, BlockVector
from psydac.core.bsplines         import quadrature_grid
from psydac.utilities.quadratures import gauss_legendre
from psydac.fem.basic             import FemField

#==============================================================================
class Projector_H1:
    """
    Projector from H1 to an H1-conforming finite element space (i.e. a finite
    dimensional subspace of H1) constructed with tensor-product B-splines in 1,
    2 or 3 dimensions.

    This is a global projector based on interpolation over a tensor-product
    grid in the logical domain. The interpolation grid is the tensor product of
    the 1D splines' Greville points along each direction.

    Parameters
    ----------
    H1 : SplineSpace or TensorFemSpace
        H1-conforming finite element space, codomain of the projection operator
    """
    def __init__(self, H1):

        # Number of dimensions
        dim = H1.ldim

        # Collocation matrices for B-splines in each direction
        H1.init_interpolation()
        N = [V._interpolator for V in H1.spaces]

        # Empty vector to store right-hand side of linear system
        rhs = StencilVector(H1.vector_space)

        # Construct arguments for computing degrees of freedom
        n_basis = [V.nbasis for V in H1.spaces]
        intp_x  = [V.greville for V in H1.spaces]
        slices  = tuple(slice(p,-p) for p in H1.degree)
        args    = (*n_basis, *intp_x, rhs._data[slices])

        # Select correct function for computing degrees of freedom
        if   dim == 1:  func = evaluate_dofs_1d_0form
        elif dim == 2:  func = evaluate_dofs_2d_0form
        elif dim == 3:  func = evaluate_dofs_3d_0form
        else:
            raise ValueError('H1 projector of dimension {} not available'.format(dim))

        # Store attributes in object
        self.space  = H1
        self.N      = N
        self.func   = func
        self.args   = args
        self.rhs    = rhs
        self.solver = KroneckerLinearSolver(H1.vector_space, self.N)

    #--------------------------------------------------------------------------
    def __call__(self, fun):
        r"""
        Project scalar function onto the H1-conforming finite element space.
        This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : callable
            Real-valued scalar function to be projected, with arguments the
            coordinates (x_1, ..., x_N) of a point in the logical domain. This
            corresponds to the coefficient of a 0-form.

            $fun : \hat{\Omega} \mapsto \mathbb{R}$.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H1-conforming finite
            element space). This is also a real-valued scalar function in the
            logical domain.
        """
        # build the rhs
        self.func(*self.args, fun)

        coeffs = self.solver.solve(self.rhs)

        return FemField(self.space, coeffs=coeffs)

#==============================================================================
class Projector_Hcurl:
    """
    Projector from H(curl) to an H(curl)-conforming finite element space, i.e.
    a finite dimensional subspace of H(curl), constructed with tensor-product
    B- and M-splines in 2 or 3 dimensions.

    This is a global projector constructed over a tensor-product grid in the
    logical domain. The vertices of this grid are obtained as the tensor
    product of the 1D splines' Greville points along each direction.

    The H(curl) projector matches the "geometric" degrees of freedom of
    discrete 1-forms, which are the line integrals of a vector field along cell
    edges. To achieve this, each component of the vector field is projected
    independently, by combining 1D histopolation along the direction of the
    edges with 1D interpolation along the other directions.

    Parameters
    ----------
    Hcurl : ProductFemSpace
        H(curl)-conforming finite element space, codomain of the projection
        operator.

    nquads : list(int) | tuple(int)
        Number of quadrature points along each direction, to be used in Gauss
        quadrature rule for computing the (approximated) degrees of freedom.
    """
    def __init__(self, Hcurl, nquads=None):

        dim = Hcurl.n_components

        if nquads:
            assert len(nquads) == dim
            uw = [gauss_legendre( k-1 ) for k in nquads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_grids[i].quad_rule_x,V.quad_grids[i].quad_rule_w) for i,V in enumerate(Hcurl.spaces)]

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
            raise NotImplementedError('Hcurl projector is only available in 2D or 3D.')

        solverblocks =  [KroneckerLinearSolver(block.vector_space, self.mats[i]) for i, block in enumerate(Hcurl.spaces)]
        self.solver = BlockDiagonalSolver(Hcurl.vector_space, blocks=solverblocks)

    #--------------------------------------------------------------------------
    def __call__(self, fun):
        r"""
        Project vector function onto the H(curl)-conforming finite element
        space. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : list/tuple of callables
            Scalar components of the real-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain. These correspond to the coefficients
            of a 1-form in the canonical basis (dx_1, ..., dx_N).

            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H(curl)-conforming
            finite element space). This is also a real-valued vector function
            in the logical domain.
        """
        # build the rhs
        self.func(*self.args, *fun)

        coeffs = self.solver.solve(self.rhs)
        
        return FemField(self.space, coeffs=coeffs)

#==============================================================================
class Projector_Hdiv:
    """
    Projector from H(div) to an H(div)-conforming finite element space, i.e. a
    finite dimensional subspace of H(div), constructed with tensor-product
    B- and M-splines in 2 or 3 dimensions.

    This is a global projector constructed over a tensor-product grid in the
    logical domain. The vertices of this grid are obtained as the tensor
    product of the 1D splines' Greville points along each direction.

    The H(div) projector matches the "geometric" degrees of freedom of discrete
    (N-1)-forms in N dimensions, which are the integrated flux of a vector
    field through cell faces (in 3D) or cell edges (in 2D).

    To achieve this, each component of the vector field is projected
    independently, by combining histopolation along the direction(s) tangential
    to the face (in 3D) or edge (in 2D), with 1D interpolation along the normal
    direction.

    Parameters
    ----------
    Hdiv : ProductFemSpace
        H(div)-conforming finite element space, codomain of the projection
        operator.

    nquads : list(int) | tuple(int)
        Number of quadrature points along each direction, to be used in Gauss
        quadrature rule for computing the (approximated) degrees of freedom.
    """
    def __init__(self, Hdiv, nquads=None):

        dim = Hdiv.n_components

        if nquads:
            assert len(nquads) == dim
            uw = [gauss_legendre( k-1 ) for k in nquads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_grids[i].quad_rule_x,V.quad_grids[i].quad_rule_w) for i,V in enumerate(Hdiv.spaces)]

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
            raise NotImplementedError('Hdiv projector is only available in 2D or 3D.')

        solverblocks =  [KroneckerLinearSolver(block.vector_space, self.mats[i]) for i, block in enumerate(Hdiv.spaces)]
        self.solver = BlockDiagonalSolver(Hdiv.vector_space, blocks=solverblocks)

    #--------------------------------------------------------------------------
    def __call__(self, fun):
        r"""
        Project vector function onto the H(div)-conforming finite element
        space. This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : list/tuples of callable
            Scalar components of the real-valued vector function to be
            projected, with arguments the coordinates (x_1, ..., x_N) of a
            point in the logical domain. In 3D these correspond to the
            coefficients of a 2-form in the canonical basis (dx_1 ∧ dx_2,
            dx_2 ∧ dx_3, dx_3 ∧ dx_1).

            $fun_i : \hat{\Omega} \mapsto \mathbb{R}$ with i = 1, ..., N.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the H(div)-conforming
            finite element space). This is also a real-valued vector function
            in the logical domain.
        """
        # build the rhs
        self.func(*self.args, *fun)

        coeffs = self.solver.solve(self.rhs)

        return FemField(self.space, coeffs=coeffs)

#==============================================================================
class Projector_L2:
    """
    Projector from L2 to an L2-conforming finite element space (i.e. a finite
    dimensional subspace of L2) constructed with tensor-product M-splines in 1,
    2 or 3 dimensions.

    This is a global projector constructed over a tensor-product grid in the
    logical domain. The vertices of this grid are obtained as the tensor
    product of the 1D splines' Greville points along each direction.

    The L2 projector matches the "geometric" degrees of freedom of discrete
    N-forms in N dimensions, which are line/surface/volume integrals of a
    scalar field over an edge/face/cell in 1/2/3 dimension(s). To this end
    histopolation is used along each direction.

    Parameters
    ----------
    L2 : SplineSpace
        L2-conforming finite element space, codomain of the projection operator

    nquads : list(int) | tuple(int)
        Number of quadrature points along each direction, to be used in Gauss
        quadrature rule for computing the (approximated) degrees of freedom.
    """
    def __init__(self, L2, nquads=None):

        # Quadrature grids in cells defined by consecutive Greville points
        if nquads:
            uw = [gauss_legendre( k-1 ) for k in nquads]
            uw = [(u[::-1], w[::-1]) for u,w in uw]
        else:
            uw = [(V.quad_rule_x,V.quad_rule_w) for V in L2.quad_grids]

        quads = [quadrature_grid(V.histopolation_grid, u, w) for V,(u,w) in zip(L2.spaces, uw)]
        quad_x, quad_w = list(zip(*quads))

        L2.init_histopolation()

        # Histopolation matrices for D-splines in each direction
        self.D = [V._histopolator for V in L2.spaces]

        self.space = L2
        self.rhs   = StencilVector(L2.vector_space)
        slices     = tuple(slice(p+1,-p-1) for p in L2.degree)

        if   len(self.D) == 1:  self.func = evaluate_dofs_1d_1form
        elif len(self.D) == 2:  self.func = evaluate_dofs_2d_2form
        elif len(self.D) == 3:  self.func = evaluate_dofs_3d_3form
        else:
            raise ValueError('L2 projector of dimension {} not available'.format(str(len(self.D))))

        self.args  = (*quad_x, *quad_w, self.rhs._data[slices])
        self.solver = KroneckerLinearSolver(L2.vector_space, self.D)

    #--------------------------------------------------------------------------
    def __call__(self, fun):
        r"""
        Project scalar function onto the L2-conforming finite element space.
        This happens in the logical domain $\hat{\Omega}$.

        Parameters
        ----------
        fun : callable
            Real-valued scalar function to be projected, with arguments the
            coordinates (x_1, ..., x_N) of a point in the logical domain. This
            corresponds to the coefficient of an N-form in N dimensions, in
            the canonical basis dx_1 ∧ ... ∧ dx_N.

            $fun : \hat{\Omega} \mapsto \mathbb{R}$.

        Returns
        -------
        field : FemField
            Field obtained by projection (element of the L2-conforming finite
            element space). This is also a real-valued scalar function in the
            logical domain.
        """
        # build the rhs
        self.func(*self.args, fun)

        coeffs = self.solver.solve(self.rhs)

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
