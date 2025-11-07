# coding: utf-8
import sympy
from mpi4py import MPI
from time import time, sleep
# import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.pyplot as plt
from sympy import sqrt, sin, cos, pi

from sympde.topology.analytical_mapping import PolarMapping, TargetMapping, CzarnyMapping
from sympde.topology.domain import Square, Domain
from sympde.topology import ScalarFunctionSpace, elements_of
from sympde.expr import BilinearForm, LinearForm, integral, Norm, Functional
from sympde.calculus import dot, grad, laplace
from sympde.topology.mapping import Mapping
from sympde.expr.evaluation import LogicalExpr

from psydac.api.discretization import discretize
from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.basic import LinearOperator
from psydac.linalg.solvers import inverse
from psydac.fem.splines import SplineSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.basic import FemField
from psydac.mapping.discrete import SplineMapping
from psydac.utilities.utils import refine_array_1d
from psydac.cad.geometry import Geometry
from psydac.ddm.cart import DomainDecomposition

from psydac.polar.c1_projections import C1Projector

from psydac.api.settings import PSYDAC_BACKENDS

from psydac.feec.polar.conga_projections import C0PolarProjection_V0

# backend = PSYDAC_BACKENDS['numba']
backend = PSYDAC_BACKENDS['python']


# ==============================================================================
class Laplacian:

    def __init__(self, mapping):
        assert isinstance(mapping, Mapping)

        self._eta = mapping.logical_coordinates
        self._metric = mapping.metric_expr
        self._metric_det = mapping.metric_det_expr

    # ...
    def __call__(self, phi):
        from sympy import sqrt, Matrix

        u = self._eta
        G = self._metric
        sqrt_g = sqrt(self._metric_det)

        # Store column vector of partial derivatives of phi w.r.t. uj
        dphi_du = Matrix([phi.diff(uj) for uj in u])

        # Compute gradient of phi in tangent basis: A = G^(-1) dphi_du
        A = G.LUsolve(dphi_du)

        # Compute Laplacian of phi using formula for divergence of vector A
        lapl = sum((sqrt_g * Ai).diff(ui) for ui, Ai in zip(u, A)) / sqrt_g

        return lapl


# ============================= EXACT SOLUTION ================================#
class Poisson2D:
    r"""
    Exact solution to the 2D Poisson equation with Dirichlet boundary
    conditions, to be employed for the method of manufactured solutions.

    :code
    $(\partial^2_{xx} + \partial^2_{yy}) \phi(x,y) = -\rho(x,y)$

    """

    def __init__(self, domain, mapping, phi, rho):
        assert isinstance(mapping, Mapping)

        self._domain = domain
        self._mapping = mapping
        self._phi = phi
        self._rho = rho


        s, t = mapping.logical_coordinates
        self._phi_callable = sympy.lambdify([s, t], phi)
        self._rho_callable = sympy.lambdify([s, t], rho)

        # x, y  = mapping.coordinates
        # self._phi_callable = sympy.lambdify([x, y], phi)
        # # self._rho_callable = sympy.lambdify([x, y], rho)

    # ...
    @staticmethod
    def disk_domain(R, shift_D):
        r"""
        Solve Poisson's equation on a disk of radius R centered at (x,y) = (0, 0),
        with logical coordinates (s, theta):

        - The radial coordinate s belongs to the interval [0, R];
        - The angular coordinate theta belongs to the interval [0, 2 * pi).

        : code
        $\phi(x,y) = sin(3.5 \pi (R^2 - x^2 - y^2)/R^2)$.
        """
        domain = ((0, R), (0, 2 * np.pi))
        mapping = TargetMapping('TM', c1=shift_D * R * R, c2=0, k=0, D=shift_D)

        # physical field (cf use of physical ref solution in Maxwell case)
        params = dict(c1=0, c2=0, k=0, D=shift_D)
        k = params['k']
        D = params['D']
        kx = 2 * pi / (R * (1 - k + D))
        ky = 2 * pi / (R * (1 + k))
        x, y = sympy.symbols('x, y')
        phi = (1 - ((x * x + y * y) / (R * R)) ** 4) * sin(kx * x) * cos(ky * y)
        # phi = (1 - ((X * X + Y * Y) / (R * R)) ** 4)
        rho = - phi.diff(x, x) - phi.diff(y, y)
        obj = Poisson2D(domain, mapping, phi, rho)
        obj.coordinates = (x, y)

        return obj

    # ...
    @staticmethod
    def target_domain():
        r"""
        Solve Poisson's equation on a polar domain, with logical coordinates (s, theta):

        - The radial coordinate s belongs to the interval [0, 1];
        - The angular coordinate theta belongs to the interval [0, 2 * pi).

        The shape of the domain is set by parameter k in [0, 1): for k = 0 we have
        a disk and for k --> 1 the disk is horizontally squeezed.

        The parameter D in (-(1 - k)/2, (1 - k)/2) moves horizontally the pole within
        the shape.

        The parameter c1, c2 are just shifts on the plane of the domain.

        : code
        $\phi(x,y) = (1 - s^8)\sin(k_x(x - 0.5))\cos(k_y y)$.

        """

        domain = ((0, 1), (0, 2 * np.pi))
        params = dict(c1=0, c2=0, k=0.3, D=0.2)
        mapping = TargetMapping('F', **params)

        from sympy import sin, cos, pi, sqrt
        # from sympy.abc import x, y

        lapl = Laplacian(mapping)
        s, t = mapping.logical_coordinates
        x, y = mapping.expressions

        # Manufactured solution in logical coordinates
        k = params['k']
        D = params['D']
        kx = 2 * pi / (1 - k + D)
        ky = 2 * pi / (1 + k)

        phi = (1 - s ** 8) * sin(kx * (x - 0.5)) * cos(ky * y)
        rho = - lapl(phi)

        # c1    = params['c1']
        # c2    = params['c2']
        # y_tilde = (y - c2)/(1 + k)
        # x_tilde = (x - c1)/(1 - k)
        # D_tilde = (2 * D)/(1 - k)
        # r = (x_tilde**2 + y_tilde**2)
        # R = sqrt((2 * r)/(1 - D_tilde * x_tilde + sqrt((1 - D_tilde)**2 - D_tilde**2 * r)))
        # phi = (1 - R**8) * sin(kx * (x - 0.5)) * cos(ky * y)
        # rho = - laplace(phi)

        return Poisson2D(domain, mapping, phi, rho)

    # ...
    @staticmethod
    def czarny_domain():
        r"""
        Solve Poisson's equation on a czarny domain, with logical coordinates (s, theta):

        - The radial coordinate s belongs to the interval [0, 1];
        - The angular coordinate theta belongs to the interval [0, 2 * pi).

        : code
        $\phi(x,y) = (1 - s^8)\sin(\pi x)\cos(\pi y)$.

        """

        domain = ((0, 1), (0, 2 * np.pi))
        params = dict(c1=0, c2=0, eps=0.2, b=1.4)
        mapping = CzarnyMapping('F', **params)

        from sympy import sin, cos, pi

        lapl = Laplacian(mapping)
        s, t = mapping.logical_coordinates
        x, y = mapping.expressions

        # Manufactured solution in logical coordinates
        phi = (1 - s ** 8) * sin(pi * x) * cos(pi * y)
        rho = - lapl(phi)

        return Poisson2D(domain, mapping, phi, rho)

    # ...
    @property
    def domain(self):
        return self._domain

    @property
    def mapping(self):
        return self._mapping

    @property
    def phi(self):
        return self._phi

    @property
    def rho(self):
        return self._rho

    @property
    def phi_callable(self):
        return self._phi_callable

    @property
    def rho_callable(self):
        return self._rho_callable


# ====================== CONGA (PENALIZED) POISSON ============================#

class CongaLaplacian(LinearOperator):

    def __init__(self, S, M, P, alpha):
        assert isinstance(S, StencilMatrix)
        assert isinstance(M, StencilMatrix)
        assert isinstance(P, (C0PolarProjection_V0, ))

        W0 = P.W0.coeff_space

        assert S.domain is S.codomain is W0
        assert M.domain is M.codomain is W0

        self.S = S
        self.M = M
        self.P = P
        self.alpha = alpha
        self.W0 = W0

    def dot(self, x, out=None):
        if out is None:
            y = self.M._domain.zeros()
        else:
            assert isinstance(out, StencilVector)
            assert out.space is self.M._domain
            y = out

        y1 = self.P.T.dot(self.S.dot(self.P.dot(x)))
        y2 = x - self.P.dot(x)
        y2 = self.M.dot(y2)
        y2 = self.alpha * (y2 - self.P.T.dot(y2))
        # y = y1 + y2
        y1.copy(out=y)
        y += y2

        y.update_ghost_regions()
        return y

    def transpose(self):
        return CongaLaplacian(self.S.T, self.M.T, self.P.T, self.alpha)

    def tosparse(self):
        S = self.S.tosparse()
        M = self.M.tosparse()
        P = self.P.tosparse()
        alpha = self.alpha
        n = self.W0.dimension

        from scipy.sparse import eye
        I = eye(n)

        A = alpha * (I - P).T @ M @ (I - P) + P.T @ S @ P

        return A

    def toarray(self):
        return self.tosparse().toarray()

    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self):
        return (self.W0.dimension, self.W0.dimension)

    @property
    def domain(self):
        return self.W0

    @property
    def codomain(self):
        return self.W0

    @property
    def dtype(self):
        return float


###############################################################################

def run_poisson_2d(*, test_case, ncells, degree,
                   shift_D, R, use_spline_mapping, smooth_method,
                   cgtol, cgiter, alphaCONGA, study='poisson', verbose=False):
    timing = {}
    timing['assembly'] = 0.0
    timing['projection'] = 0.0
    timing['solution'] = 0.0
    timing['diagnostics'] = 0.0
    timing['export'] = 0.0

    assert study == 'poisson'  # for now

    # Method of manufactured solution
    if test_case == 'disk':
        model = Poisson2D.disk_domain(R=R, shift_D=shift_D)
    elif test_case == 'target':
        model = Poisson2D.target_domain()
    elif test_case == 'czarny':
        model = Poisson2D.czarny_domain()
    else:
        raise ValueError("Only available test-cases are 'disk', 'target' and 'czarny'")

    if smooth_method not in ('polar-std', 'polar-spec', 'C0conga', 'C1conga', 'None'):
        raise ValueError(
            "Only available options for pole smoothness are 'polar-spec', 'polar-std', 'C0conga', 'C1conga', 'None'")

    if smooth_method == 'polar-spec' and (not use_spline_mapping):
        print('WARNING: C1 conforming discretization only available for spline mappings')
        print('The domain will be approximated in the 0-forms spline space.')
        print()
        use_spline_mapping = True

    # Communicator, size, rank
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    # Number of elements and spline degree
    ne1, ne2 = ncells
    p1, p2 = degree

    # ==================== SPLINE SPACE FOR SPLINE MAPPINGS =======================#

    # Create uniform grid
    grid_1 = np.linspace(*model.domain[0], num=ne1 + 1)
    grid_2 = np.linspace(*model.domain[1], num=ne2 + 1)

    # Create 1D finite element spaces
    V1 = SplineSpace(p1, grid=grid_1, periodic=False)
    V2 = SplineSpace(p2, grid=grid_2, periodic=True)

    # Create 2D tensor product finite element space
    domain_decomposition = DomainDecomposition(ncells, [False, True] , comm = mpi_comm)
    V = TensorFemSpace(domain_decomposition, V1, V2)

    s1, s2 = V.coeff_space.starts
    e1, e2 = V.coeff_space.ends

    # ==================== MAPPING & PHYSICAL DOMAIN ==============================#
    logical_domain = Square('Omega', bounds1=model.domain[0],
                            bounds2=model.domain[1])
    if use_spline_mapping:
        # Create spline mapping by interpolation of analytical mapping
        map_analytic = model.mapping.get_callable_mapping()
        map_discrete = SplineMapping.from_mapping(V, map_analytic)
        # Create symbolic mapping with callable mapping as spline
        mapping = Mapping('M', dim=2)
        mapping.set_callable_mapping(map_discrete)
        # In order to create a sympde.Domain object from this mapping we have
        # to create first a HDF5 file and then load as sympde.Domain.fromfile
        t0 = time()
        geometry = Geometry.from_discrete_mapping(map_discrete, comm=mpi_comm)
        geometry.export('geo.h5')
        t1 = time()
        timing['export'] += t1 - t0
        domain = Domain.from_file('geo.h5')

        #check_regular_ring_map(map_discrete)

    else:
        # Only symbolic mapping is necessary
        mapping = model.mapping
        domain = mapping(logical_domain)

    rp_str = f'{ncells[0]}_{ncells[1]}_p={degree[0]}_t={test_case}_D{shift_D}_m={smooth_method}'
    if use_spline_mapping:
        rp_str += '_sm'
    else:
        rp_str += '_pm'  # WARNING: check that polar_mapping == True ?

    # ========================== SYMBOLIC DEFINITION ==============================#

    # Equations
    V0 = ScalarFunctionSpace('V0', domain)
    u0, v0 = elements_of(V0, names='u0, v0')
    aM = BilinearForm((u0, v0), integral(domain, u0 * v0))
    aS = BilinearForm((u0, v0), integral(domain, dot(grad(u0), grad(v0))))
    # model.rho is in logical coordinates instead of physical but it works anyways
    # but needs to comment "Check linearity" in LinearForm._init_
    # rhs = LinearForm(v0, integral(domain, model.rho * v0))

    from sympy.abc import x, y  # try

    # f = 2 * 7 * pi * cos(7 * pi / 2 * (1 - x ** 2 - y ** 2)) + (7 * pi) ** 2 * (x ** 2 + y ** 2) * sin(
    #     7 * pi / 2 * (1 - x ** 2 - y ** 2))
    # f = x+y

    # f = model.rho
    X, Y = model.coordinates
    x, y = model.mapping.expressions
    f = model.rho.subs({X: x, Y: y})

    rhs = LinearForm(v0, integral(domain, f * v0))

    err_diff = model.phi - u0

    # NOTE (mcp, dec 2024): norms and errors now computed only for discrete solutions (using M and S matrices)
    errL2 = Norm(err_diff, domain, kind='L2')
    errH1 = Norm(err_diff, domain, kind='H1')

    u0L2norm = Norm(u0, domain, kind='L2')
    u0H1norm = Norm(u0, domain, kind='H1')

    # L2norm = Norm(err_diff*sqrt(mapping.jacobian.det()), logical_domain, kind = 'L2')
    # H1norm = Norm(mapping.jacobian.T**(-1) * grad(err_diff)
    #               * sqrt(mapping.jacobian.det()), logical_domain, kind = 'L2')

    # ============================= DISCRETIZATION ================================#
    if use_spline_mapping:
        domain_h = discretize(domain, filename='geo.h5', comm = mpi_comm)
        V0_h = discretize(V0, domain_h)
        F = list(domain_h.mappings.values()).pop()
    else:
        domain_h = discretize(domain, ncells=ncells, periodic=[False, True], comm = mpi_comm)
        V0_h = discretize(V0, domain_h, degree=degree)
        F = mapping.get_callable_mapping()

    aM_h = discretize(aM, domain_h, (V0_h, V0_h), backend=backend)
    aS_h = discretize(aS, domain_h, (V0_h, V0_h), backend=backend)
    rhs_h = discretize(rhs, domain_h, V0_h, backend=backend)


    errL2_h = discretize(errL2, domain_h, V0_h, backend=backend)
    errH1_h = discretize(errH1, domain_h, V0_h, backend=backend)

    u0L2norm_h = discretize(u0L2norm, domain_h, V0_h, backend=backend)
    u0H1norm_h = discretize(u0H1norm, domain_h, V0_h, backend=backend)

    M = aM_h.assemble()
    S = aS_h.assemble()
    b = rhs_h.assemble()

    S.update_ghost_regions()
    b.update_ghost_regions()
    M.update_ghost_regions()

    # =================== PROJECT THE EXACT SOLUTION  =========================#

    from psydac.feec.pull_push import pull_2d_h1
    from sympy import lambdify
    from psydac.feec.global_geometric_projectors import GlobalGeometricProjectorH1

    Pi0 = GlobalGeometricProjectorH1(V0_h)
    phi_symref = model.phi
    phi_calref = lambdify(domain.coordinates, phi_symref)
    phi_callog = pull_2d_h1(phi_calref, F)
    phi_ref = Pi0(phi_callog)
    phi_ref.coeffs.update_ghost_regions()


    # ========================= HANDLING THE SINGULARITY ===========================#

    # If required by user, create C1 projector and then restrict
    # stiffness/mass matrices and right-hand-side vector to C1 space
    t0 = time()
    if smooth_method == 'polar-spec':
        proj = C1Projector(F)
        Sp = proj.change_matrix_basis(S)
        bp = proj.change_rhs_basis(b)
        alpha = 'None'
    if smooth_method == 'polar-std':
        # Build standard polar map from control points of standard polar map
        n1, n2 = [W.nbasis for W in V0_h.spaces]
        rho = np.array([i1 / (n1 - 1) for i1 in range(n1)])
        theta = np.array([i2 * 2 * np.pi / n2 for i2 in range(n2)])
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        cp = np.zeros((n1, n2, 2))
        for i1 in range(n1):
            for i2 in range(n2):
                cp[i1, i2, 0] = rho[i1] * cos_theta[i2]
                cp[i1, i2, 1] = rho[i1] * sin_theta[i2]
        F_std = SplineMapping.from_control_points(V0_h, cp)
        proj = C1Projector(F_std)
        Sp = proj.change_matrix_basis(S)
        bp = proj.change_rhs_basis(b)
        alpha = 'None'
    elif smooth_method == 'C1conga':
        raise ValueError("Only C0conga is implemented for now!")
        # gamma = 1.0  # any value would be ok.
        # alpha = alphaCONGA
        # P0 = C1CongaProjector0(V0_h, gamma=gamma, hbc=True)  # hbc imposes the boundary conditions
        # Sc = CongaLaplacian(S, M, P0, alpha)
        # A = Sc.tosparse()
        # bc = P0.T.dot(b)
    elif smooth_method == 'C0conga':
        alpha = alphaCONGA
        P0 = C0PolarProjection_V0(V0_h, hbc=True)  # hbc imposes the boundary conditions
        Sc = CongaLaplacian(S, M, P0, alpha)
        bc = P0.T.dot(b)
    elif smooth_method == 'None':
        alpha = 'None'
    t1 = time()
    timing['projection'] = t1 - t0

    # Apply homogeneous Dirichlet boundary conditions for the conforming
    # smooth_method case 'polar' and non-conforming case 'None'
    # NOTE: this does not effect ghost regions
    S_nobc = S.copy()
    e1 = V0_h.coeff_space.ends[0]
    if e1 == V0_h.coeff_space.npts[0] - 1:
        if smooth_method in ('polar-std', 'polar-spec'):
            last = bp[1].space.npts[0] - 1
            Sp[1, 1][last, :, :, :] = 0.
            Sp[1, 1][last, :, 0, 0] = 1.
            bp[1][last, :] = 0.
        elif smooth_method == 'None':
            S[e1, :, :, :] = 0.
            S[e1, :, 0, 0] = 1.
            b[e1, :] = 0.

    # ====================== SOLVE GALERKIN SYSTEM WITH CG ========================#

    # Solve linear system
    t0 = time()
    if smooth_method in ('polar-std', 'polar-spec'):
        Sp_inv = inverse(Sp, 'cg', tol = cgtol, maxiter = cgiter, verbose=verbose)
        xp = Sp_inv.dot(bp)
        xsol = proj.convert_to_tensor_basis(xp)
        info = Sp_inv.get_info()
        # from psydac.linalg.utilities import array_to_psydac
        # import scipy
        # L = proj.L[:, :, p2: -p2].reshape(3, 2 * ne2)
        # E = np.block([[L, np.zeros((3, (ne1 + p1 - 2) * ne2))],
        #               [np.zeros(((ne1 + p1 - 2) * ne2, 2 * ne2)), np.eye((ne1 + p1 - 2) * ne2)]])
        # xparray = scipy.linalg.solve(Sp.toarray(), bp.toarray())
        # xarray = E.T @ xparray
        # xsol = array_to_psydac(xarray, V0_h.coeff_space)
    elif smooth_method == 'C1conga':
        Sc_inv = inverse(Sc, 'cg', tol=cgtol, maxiter=cgiter, verbose=verbose)
        xsol = Sc_inv.dot(bc)
        info = Sc_inv.get_info()
    elif smooth_method == 'C0conga':
        Sc_inv = inverse(Sc, 'cg', tol=cgtol, maxiter=cgiter, verbose=verbose)
        xsol = Sc_inv.dot(bc)
        info = Sc_inv.get_info()
    elif smooth_method == 'None':
        pc = S.diagonal(inverse=True)
        S_inv = inverse(S, 'pcg', pc=pc, tol=cgtol, maxiter=cgiter, verbose=verbose)
        xsol = S_inv.dot(b)
        info = S_inv.get_info()
    t1 = time()
    timing['solution'] = t1 - t0

    # ========================= APPROXIMATION ERROR ===============================#

    # Create potential field for discrete solution
    phi = FemField(V0_h, coeffs=xsol)
    phi.coeffs.update_ghost_regions()

    # ref solution: projected exact solution

    # phi_ref = Pi0(model.phi_callable)

    # def P0_phys(f_phys, P0, domain, mappings_list):
    # phi = lambdify(domain.coordinates, f_phys)
    # P0(f_log)

    # L2 and H1 norms
    ref_u0L2_2 = phi_ref.coeffs.inner(M.dot(phi_ref.coeffs))  # l2 norm of ref solution
    ref_u0H1_semi2 = phi_ref.coeffs.inner(S.dot(phi_ref.coeffs))
    ref_u0L2 = np.sqrt(ref_u0L2_2)
    ref_u0H1 = np.sqrt(ref_u0H1_semi2 + ref_u0L2_2)  # H1 norm of ref solution

    # L2 and H1 errors
    t0 = time()
    phi_diff = phi_ref.coeffs - phi.coeffs

    err_l2_2 = phi_diff.inner(M.dot(phi_diff))
    err_h1_semi2 = phi_diff.inner(S.dot(phi_diff))

    err_l2 = np.sqrt(err_l2_2)
    err_h1 = np.sqrt(err_h1_semi2 + err_l2_2)

    rel_err_l2 = err_l2 / ref_u0L2
    rel_err_h1 = err_h1 / ref_u0H1

    # previous option: ok but H1 norm has problems ?
    # ref_u0L2 = u0L2norm_h.assemble(u0 = phi_ref)
    # err2 = L2norm_h.assemble(u0 = phi_ref - phi)
    # err2 = L2norm_h.assemble(u0 = phi)

    # assert np.allclose(err2, err2_matrix, rtol = 1e-7, atol = 1e-7)

    # and H1 error
    # t0 = time()
    # errh1_semi = H1norm_h.assemble(u0 = phi_ref - phi)
    # errh1_semi = errH1_h.assemble(u0 = phi)  # BUG ??
    # u0H1_semi = u0H1norm_h.assemble(u0 = phi)
    # errh1 = np.sqrt(errh1_semi**2 + err2**2)
    # u0H1 = np.sqrt(u0H1_semi**2 + u0L2**2)
    # errh1 = np.sqrt(errh1_semi**2 + err2**2)
    # errh1_semi_sq = phi_diff.dot(S_nobc.dot(phi_diff))
    # errh1_matrix = np.sqrt(err2_sq + errh1_semi_sq)

    t1 = time()
    timing['diagnostics'] = t1 - t0

    # assert np.allclose(errh1, errh1_matrix, rtol = 1e-7, atol = 1e-7)
    # Write solution to HDF5 file
    t0 = time()
    V0_h.export_fields('fields.h5', phi=phi)
    t1 = time()
    timing['export'] += t1 - t0
    # =============================== PRINTING INFO ===============================#

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Print some information to terminal
    for i in range(mpi_size):
        if i == mpi_rank:
            print('--------------------------------------------------')
            print(' RANK = {}'.format(mpi_rank))
            print('--------------------------------------------------')
            print('> Grid                :: [{ne1},{ne2}]'.format(ne1=ne1, ne2=ne2))
            print('> Degree              :: [{p1},{p2}]'.format(p1=p1, p2=p2))
            print('> Penalization alpha  :: {alpha} '.format(alpha=alpha))
            print( '> CG info            :: ',info )
            print('> L2 norm solution    :: {:.2e}'.format(ref_u0L2))
            print('> H1 norm solution    :: {:.2e}'.format(ref_u0H1))
            print('> L2 error (relative) :: {:.2e}'.format(rel_err_l2))
            print('> H1 error (relative) :: {:.2e}'.format(rel_err_h1))
            print('')
            print('> Assembly time :: {:.2e}'.format(timing['assembly']))
            if smooth_method:
                print('> Project. time :: {:.2e}'.format(timing['projection']))
            print('> Solution time :: {:.2e}'.format(timing['solution']))
            print('> Evaluat. time :: {:.2e}'.format(timing['diagnostics']))
            print('> Export   time :: {:.2e}'.format(timing['export']))
            print('', flush=True)
            sleep(0.001)
        mpi_comm.Barrier()

    # =============================== VISUALIZATION ===============================#

    N = 10
    V.plot_2d_decomposition(mapping.get_callable_mapping(), refine=N)

    # plot only with the root process
    distribute_viz = False
    if not distribute_viz:
        # Non-master processes stop here
        if mpi_rank != 0:
            return
        if use_spline_mapping:
            geometry = Geometry(filename='geo.h5', comm=MPI.COMM_SELF)
            map_discrete = [*geometry.mappings.values()].pop()
            Vnew = map_discrete.space
            mapping = map_discrete
        else:
            dd = DomainDecomposition(ncells, [False, True], comm=MPI.COMM_SELF)
            Vnew = TensorFemSpace(dd, V1, V2)

        # Import solution vector into new serial field

    else:
        Vnew = V

    # Import solution vector into new serial field
    phi, = Vnew.import_fields('fields.h5', 'phi')

    # Callable exact solution in logical coordinates
    X, Y = model.coordinates
    x, y = model.mapping.expressions
    expr_phi_e = model.phi.subs({X: x, Y: y})
    x1, x2 = model.mapping.logical_coordinates
    phi_e = sympy.lambdify([x1, x2], expr_phi_e)

    # Compute numerical solution (and error) on refined logical grid
    [sk1, sk2], [ek1, ek2] = Vnew.local_domain
    print([sk1, sk2], [ek1, ek2])

    eta1 = refine_array_1d(V1.breaks[sk1:ek1 + 2], N)
    eta2 = refine_array_1d(V2.breaks[sk2:ek2 + 2], N)
    num = np.array([[phi(e1, e2) for e2 in eta2] for e1 in eta1])
    ex = np.array([[phi_e(e1, e2) for e2 in eta2] for e1 in eta1])
    err = num - ex
    print('num[0,0] = ', num[0, 0])
    print('ex[0,0] = ', ex[0, 0])

    # Compute physical coordinates of logical grid
    map_temp = map_discrete if use_spline_mapping else F
    pcoords = np.array([[map_temp(e1, e2) for e2 in eta2] for e1 in eta1])
    xx = pcoords[:, :, 0]
    yy = pcoords[:, :, 1]

    plot_only_sol = False

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.2)
        cbar = ax.get_figure().colorbar(im, cax=cax)
        return cbar


    if plot_only_sol:

        # plot only numerical solution on mapped domain (analytical or spline)
        fig, axes = plt.subplots(1, 1, figsize=(4.8, 4.8))


        if use_spline_mapping:
            # Recompute physical coordinates of logical grid using spline mapping
            pcoords = np.array([[map_discrete(e1, e2) for e2 in eta2] for e1 in eta1])
            xx = pcoords[:, :, 0]
            yy = pcoords[:, :, 1]

        # Plot numerical solution
        ax = axes  # [0]
        im = ax.contourf(xx, yy, num, 40, cmap='jet')
        add_colorbar(im, ax)
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$y$', rotation='horizontal')
        ax.set_title(r'$\phi(x,y)$')
        ax.plot(xx[:, ::N], yy[:, ::N], 'k')
        ax.plot(xx[::N, :].T, yy[::N, :].T, 'k')
        ax.set_aspect('equal')

        # fig.savefig(f'plots/phi_{rp_str}.png')
        fig.show()
        # plt.cla()

        # Plot numerical error
        fig, axes = plt.subplots(1, 1, figsize=(4.8, 4.8))
        ax = axes  # [2]
        im = ax.contourf(xx, yy, err, 40, cmap='jet')
        add_colorbar(im, ax)
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$y$', rotation='horizontal')
        ax.set_title(r'$\phi(x,y) - \phi_{ex}(x,y)$')
        # ax.plot( xx[:,::N]  , yy[:,::N]  , 'k' )
        # ax.plot( xx[::N,:].T, yy[::N,:].T, 'k' )
        ax.set_aspect('equal')

        # fig.savefig(f'plots/err_{rp_str}.png')
        fig.show()
        # plt.clf()
        # fig.clf()


    else:

        # Create figure with 3 subplots:
        #  1. exact solution on exact domain
        #  2. numerical solution on mapped domain (analytical or spline)
        #  3. numerical error    on mapped domain (analytical or spline)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

        if use_spline_mapping:
            # Recompute physical coordinates of logical grid using spline mapping
            pcoords = np.array([[map_discrete(e1, e2) for e2 in eta2] for e1 in eta1])
            xx = pcoords[:, :, 0]
            yy = pcoords[:, :, 1]

        # Plot exact solution
        ax = axes[0]
        im = ax.contourf(xx, yy, ex, 40, cmap='jet')
        add_colorbar(im, ax)
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$y$', rotation='horizontal')
        ax.set_title(r'$\phi_{ex}(x,y)$')
        ax.plot(xx[:, ::N], yy[:, ::N], 'k')
        ax.plot(xx[::N, :].T, yy[::N, :].T, 'k')
        ax.set_aspect('equal')


        # Plot numerical solution
        ax = axes[1]
        im = ax.contourf(xx, yy, num, 40, cmap='jet')
        add_colorbar(im, ax)
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$y$', rotation='horizontal')
        ax.set_title(r'$\phi(x,y)$')
        ax.plot(xx[:, ::N], yy[:, ::N], 'k')
        ax.plot(xx[::N, :].T, yy[::N, :].T, 'k')
        ax.set_aspect('equal')

        # Plot numerical error
        ax = axes[2]
        im = ax.contourf(xx, yy, err, 40, cmap='jet')
        add_colorbar(im, ax)
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$y$', rotation='horizontal')
        ax.set_title(r'$\phi(x,y) - \phi_{ex}(x,y)$')
        ax.plot( xx[:,::N]  , yy[:,::N]  , 'k' )
        ax.plot( xx[::N,:].T, yy[::N,:].T, 'k' )
        ax.set_aspect('equal')

        # Show figure
        #fig.savefig(f'plots/phi_and_err_{rp_str}.png')
        fig.suptitle(f'Rank {mpi_rank}')
        fig.tight_layout()
        fig.show()

    return locals()


# ==============================================================================
# Parser
# ==============================================================================
def parse_input_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Solve Poisson's equation on a 2D polar domain."
    )

    parser.add_argument('-t',
                        type=str,
                        choices=('disk', 'target', 'czarny'),
                        default='disk',
                        dest='test_case',
                        help='Test case'
                        )

    parser.add_argument('-D',
                        type=float,
                        default=0,
                        dest='shift_D',
                        help='Shafranov shift for parametrization of Disk'
                        )

    parser.add_argument('-R',
                        type=float,
                        default=1.0,
                        dest='R',
                        help='Radius of the disk'
                        )

    parser.add_argument('-d',
                        type=int,
                        nargs=2,
                        default=[2, 2],
                        metavar=('P1', 'P2'),
                        dest='degree',
                        help='Spline degree along each dimension'
                        )

    parser.add_argument('-n', '--ncells',
                        type=int,
                        nargs=2,
                        default=[10, 20],
                        metavar=('N1', 'N2'),
                        dest='ncells',
                        help='Number of grid cells (elements) along each dimension'
                        )

    parser.add_argument('-S',
                        action='store_true',
                        dest='use_spline_mapping',
                        help='Use spline mapping in finite element calculations'
                        )

    parser.add_argument('-m',
                        choices=('polar-spec', 'polar-std', 'C0conga', 'C1conga', 'None'),
                        default='C1conga',
                        dest='smooth_method',
                        help='Apply smoothing method at pole either C1-conforming geometry specific / C1-conforming standardized / C1-CONGA / C0-CONGA'
                        )

    parser.add_argument('-v',
                        type=bool,
                        default=False,
                        dest='verbose',
                        help='See CG iterations and L2 norm of the residuals'
                        )

    parser.add_argument('-a',
                        type=float,
                        default=1000,
                        dest='alphaCONGA',
                        help='Penalization constant for CONGA methods'
                        )

    parser.add_argument('--cgtol',
                        type=float,
                        default=1e-10,
                        dest='cgtol',
                        help='absolute tol for the residual error to stop CG'
                        )

    parser.add_argument('--maxiter',
                        type=int,
                        default=100000,
                        dest='cgiter',
                        help='max number of iterations for CG'
                        )

    return parser.parse_args()


# ==============================================================================
# Script functionality
# ==============================================================================
if __name__ == '__main__':

    args = parse_input_arguments()
    namespace = run_poisson_2d(**vars(args))

    import __main__

    if hasattr(__main__, '__file__'):
        try:
            __IPYTHON__
        except NameError:
            import matplotlib.pyplot as plt

            plt.show()

## example of run:
# mpirun -n 2 python poisson_2d.py -S -n 2 3 -d 2 2  -t disk -D 0.2 -m 'C0conga'
