"""
Solve the Transverse Electric Time dependent Maxwell Problem on an analytical disk domain.

Example of run:
python maxwell_2d.py -S -n 16 32 -d 3 3 -T 1 -D 0.2 -s 1
"""

import os

import numpy as np
from mpi4py import MPI


# ==============================================================================
# TIME DISCRETIZATION
# ==============================================================================
def compute_stable_dt(cfl, C_m, dC_m, V, tau=None, light_c=1):
    """
    compute stable time step for a leap-frog Maxwell solver,
    given the (discrete) primal and dual curl operators

        cfl: stability factor (1 to choose the maximum stable dt)
        C_m: (primal) curl V -> dV
        dC_m: (dual) curl dV -> V
        V: FEM space
        tau: (optional) time to solve with an integer number of time steps
        light_c: Maxwell parameter

    """
    print(f" .. compute stable dt by estimating the operator norm of ")
    print(f" ..   dual_curl_h @ curl_h:   V_h -> V_h ")
    print(f" ..   with dim(V_h) =  {V.coeff_space.dimension}      ... ")

    def vect_norm_2(vv):
        return np.sqrt(vv.inner(vv))

    vv = V.coeff_space.zeros()
    print(f"type(V.coeff_space) = {type(V.coeff_space)}")
    print(
        f"V.coeff_space.shape = {V.coeff_space.shape}, V.coeff_space.dimension = {V.coeff_space.dimension}"
    )
    vv[:] = np.random.random(size=V.coeff_space.shape)
    norm_vv = vect_norm_2(vv)
    max_ncfl = 500
    ncfl = 0
    spectral_rho = 1
    conv = False
    CC_m = dC_m @ C_m
    while not (conv or ncfl > max_ncfl):
        ncfl += 1
        vv *= 1.0 / norm_vv
        vv = CC_m @ vv
        norm_vv = vect_norm_2(vv)
        old_spectral_rho = spectral_rho
        spectral_rho = norm_vv.copy()  # copy ??
        conv = abs((spectral_rho - old_spectral_rho) / spectral_rho) < 0.001
        print(
            f"    ... spectral radius iteration: spectral_rho( dC_m @ C_m ) ~= {spectral_rho}"
        )
    norm_op = np.sqrt(spectral_rho)
    c_dt_max = 2.0 / norm_op
    dt = cfl * c_dt_max / light_c
    if tau is not None:
        Nt_per_tau = int(np.ceil(tau / dt))
        assert Nt_per_tau >= 1
        dt = tau / Nt_per_tau
    else:
        Nt_per_tau = 1
    assert light_c * dt <= cfl * c_dt_max
    print(f"  Time step dt computed for Maxwell solver:")
    print(
        f"  with cfl = repr({cfl} we found dt = {dt} -- that is Nt_per_tau = {Nt_per_tau} on tau = {tau}."
    )
    print(
        f"   -- note that c*Dt = {light_c * dt} and c_dt_max = {c_dt_max} thus c * dt / c_dt_max = {light_c * dt / c_dt_max}"
    )
    print(
        f"   -- and spectral_radius((c*dt)**2* dC_m @ C_m ) = {(light_c * dt * norm_op) ** 2} (should be < 4)"
    )
    return Nt_per_tau, dt, norm_op


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_field_and_error(name, t, x, y, field_h, field_ex, *gridlines, only_field=True):

    import matplotlib.pyplot as plt
    from psydac.feec.polar.examples.utils_congapol import add_colorbar

    if only_field:
        fig, ax0 = plt.subplots(1, 1, figsize=(7, 6))
        axes = [ax0]
    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))
        axes = [ax0, ax1]
    im0 = ax0.contourf(x, y, field_h, 50)
    ax0.set_title(r"${0}_h$".format(name))
    if not only_field:
        im1 = ax1.contourf(x, y, field_ex - field_h, 50)
        ax1.set_title(r"${0} - {0}_h$".format(name))
    for ax in axes:
        ax.plot(*gridlines[0], color="k")
        ax.plot(*gridlines[1], color="k")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14, rotation="horizontal")
        ax.set_aspect("equal")
    add_colorbar(im0, ax0)
    if not only_field:
        add_colorbar(im1, ax1)
    fig.suptitle("Time t = {:10.3e}".format(t))
    fig.tight_layout()
    return fig


def update_plot(fig, t, x, y, field_h, field_ex):
    ax0, ax1, cax0, cax1 = fig.axes
    im0 = ax0.contourf(x, y, field_h, 50)
    im1 = ax1.contourf(x, y, field_ex - field_h, 50)
    fig.colorbar(im0, cax=cax0)
    fig.colorbar(im1, cax=cax1)
    fig.suptitle("Time t = {:10.3e}".format(t))
    fig.canvas.draw()


def plot_curve_along_s(
    name,
    s_str,
    time_str,
    theta0,
    s,
    curve_h,
    curve_ref=None,
    left_s=None,
    left_curve_h=None,
    left_curve_ref=None,
):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(s, curve_h, label=f"{name}_h")
    ax.plot(s, curve_ref, label=f"{name}_ref")

    if left_s is not None:
        ax.plot(left_s, left_curve_h, label=f"{name}_h (left)")
        if left_curve_ref is not None:
            ax.plot(left_s, left_curve_ref, label=f"{name}_ref (left)")

    ax.set(
        xlabel=s_str,
        title=f"field {name} along {s_str} for theta={theta0}, at {time_str}",
    )
    ax.legend()
    return fig


# ==============================================================================
# SIMULATION
# ==============================================================================
def run_maxwell_2d_TE(
    *,
    ncells,
    smooth,
    degree,
    nsteps,
    tend,
    splitting_order,
    shift_D,
    use_spline_mapping,
    tol,
    cfl=0.9,
    show_figs=True,
    study="maxwell_bessel",
    use_scipy=True,
    verbose=False,
):
    import matplotlib.pyplot as plt
    from sympde.calculus import dot
    from sympde.expr import BilinearForm, LinearForm, integral
    from sympde.topology import Derham, Domain, Square, element_of, elements_of
    from sympde.topology.analytical_mapping import PolarMapping, TargetMapping
    from sympde.topology.mapping import Mapping
    from sympy import Tuple, cos, pi, sin, sqrt

    from psydac.api.discretization import discretize
    from psydac.api.settings import PSYDAC_BACKENDS
    from psydac.cad.geometry import Geometry
    from psydac.feec.polar.conga_projections import (
        C0PolarProjection_V1,
        C0PolarProjection_V2,
        C1PolarProjection_V1,
        C1PolarProjection_V2,
    )
    from psydac.feec.polar.examples.analyticalTE import CircularCavitySolution
    from psydac.feec.polar.examples.utils_congapol import (
        add_colorbar,
        check_regular_ring_map,
        create_tensor_spline_space,
    )
    from psydac.feec.polar.examples.waveTE import GaussianSolution
    from psydac.feec.pull_push import push_2d_hcurl, push_2d_l2
    from psydac.fem.basic import FemField
    from psydac.linalg.basic import IdentityOperator
    from psydac.linalg.solvers import inverse
    from psydac.mapping.discrete import SplineMapping
    from psydac.utilities.utils import refine_array_1d

    assert splitting_order in [2, 4]

    # Backend is Pyccel with GCC compiler. By default: Fortran, no OpenMP
    backend = PSYDAC_BACKENDS["pyccel-gcc"]

    # Radius physical domain
    R = 1.0  # 2.0

    # Speed of light and scaling
    c = 1.0
    scale = 1.0

    # Mode number
    m, n = (2, 3)

    # Exact/initial solution
    assert study in ["L2_proj", "maxwell_bessel", "maxwell_wave"]
    study_L2_proj = study == "L2_proj"

    visdir = f"plots_{study}"
    os.makedirs(visdir, exist_ok=True)

    if study == "maxwell_wave":
        # This is just the initial solution
        exact_solution = GaussianSolution(sigma=1e-1, x0=0, y0=0, scale=scale)
    else:
        exact_solution = CircularCavitySolution(R=R, c=c, m=m, n=n, scale=scale)

    Ex_ex_t = exact_solution.Ex_ex
    Ey_ex_t = exact_solution.Ey_ex
    Bz_ex_t = exact_solution.Bz_ex

    # Logical domain: [0, R] x [0, 2pi]
    logical_bounds = [[0, R], [0, 2 * pi]]
    logical_domain = Square(
        "Omega", bounds1=logical_bounds[0], bounds2=logical_bounds[1]
    )

    # Physical domain: disk of radius R obtained as image of the logical_domain
    # with the analytical mapping of a circle
    polar_mapping = False
    if polar_mapping:
        mapping = PolarMapping("PM", c1=0, c2=0, rmin=0, rmax=1)
    else:
        # domain   = ((0, 1), (0, 2 * np.pi))
        mapping = TargetMapping("TM", c1=shift_D * R * R, c2=0, k=0, D=shift_D)

    # run parameters string
    rp_str = f"{ncells[0]}_{ncells[1]}_p{degree[0]}_D{shift_D}_s{smooth}"
    if use_spline_mapping:
        rp_str += "_sm"
    else:
        rp_str += "_pm"  # WARNING: check that polar_mapping == True ?

    # Communicator, size, rank
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    if use_spline_mapping:

        # ----------------------------------------------------------------------
        # Spline space for spline mappings
        # ----------------------------------------------------------------------
        V = create_tensor_spline_space(
            ncells,
            degree,
            [False, True],
            (logical_bounds[0], logical_bounds[1]),
            mpi_comm,
        )

        # ----------------------------------------------------------------------
        # Mapping & physical domain
        # ----------------------------------------------------------------------

        # Create spline mapping by interpolation of analytical mapping
        map_analytic = mapping.get_callable_mapping()
        map_discrete = SplineMapping.from_mapping(V, map_analytic)

        check_regular_ring_map(map_discrete)

        # Create symbolic mapping with callable mapping as spline
        mapping = Mapping("M", dim=2)
        mapping.set_callable_mapping(map_discrete)
        # In order to create a sympde.Domain object from this mapping we have
        # to create first a HDF5 file and then load as sympde.Domain.fromfile
        geometry = Geometry.from_discrete_mapping(map_discrete, comm=mpi_comm)
        geometry.export("geo.h5")
        domain = Domain.from_file("geo.h5")

    else:
        # Only symbolic mapping is necessary
        domain = mapping(logical_domain)

    # DeRham sequence
    derham = Derham(domain, sequence=["h1", "hcurl", "l2"])

    # Trial and test functions
    u1, v1 = elements_of(derham.V1, names="u1, v1")  # electric field E = (Ex, Ey)
    u2, v2 = elements_of(derham.V2, names="u2, v2")  # magnetic field Bz

    # Bilinear forms that correspond to mass matrices for spaces V1 and V2
    a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    a2 = BilinearForm((u2, v2), integral(domain, u2 * v2))

    # Discrete physical domain and discrete DeRham sequence
    if use_spline_mapping:
        domain_h = discretize(domain, filename="geo.h5", comm=mpi_comm)
        derham_h = discretize(derham, domain_h)
        F = list(domain_h.mappings.values()).pop()
    else:
        domain_h = discretize(
            domain, ncells=ncells, periodic=[False, True], comm=mpi_comm
        )
        derham_h = discretize(derham, domain_h, degree=degree)
        F = mapping.get_callable_mapping()

    def l2_norm_of(f_log):
        """
        Compute the l2 norm of f over the physical domain.
        Interface needed since FEM space `integral` function is over the symbolic domain...
        """
        f2_with_det = lambda eta1, eta2: f_log(eta1, eta2) ** 2 * np.sqrt(
            F.metric_det(eta1, eta2)
        )
        return np.sqrt(derham_h.V0.integral(f2_with_det))

    def build_eval_context():
        """
        Build all serial objects needed for plotting fields.

        This is used by:
        - study='L2_proj'
        - initial Maxwell plots
        - final Maxwell plots
        """
        if mpi_rank != 0:
            return None

        if use_spline_mapping:
            domain_h_serial = discretize(domain, filename="geo.h5")
            F_serial = [*domain_h_serial.mappings.values()].pop()
        else:
            domain_h_serial = discretize(domain, ncells=ncells, periodic=[False, True])
            F_serial = F

        derham_h_serial = discretize(derham, domain_h_serial, degree=degree)
        V0_s, V1_s, V2_s = derham_h_serial.spaces
        V1_sx, V1_sy = V1_s.spaces

        grid_x1 = V0_s.breaks[0].copy()
        grid_x2 = V0_s.breaks[1].copy()

        # Fix division by zero without taking care of the limit as s --> 0
        grid_x1[0] = 1e-20

        N = 5
        x1 = refine_array_1d(grid_x1, N)
        x2 = refine_array_1d(grid_x2, N)

        x1, x2 = np.meshgrid(x1, x2, indexing="ij")

        x = np.empty_like(x1)
        y = np.empty_like(x1)

        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                x[i, j], y[i, j] = F_serial(x1[i, j], x2[i, j])

        gridlines_x1 = (x[:, ::N], y[:, ::N])
        gridlines_x2 = (x[::N, :].T, y[::N, :].T)
        gridlines = (gridlines_x1, gridlines_x2)

        return {
            "domain_h_serial": domain_h_serial,
            "derham_h_serial": derham_h_serial,
            "V0_s": V0_s,
            "V1_s": V1_s,
            "V2_s": V2_s,
            "V1_sx": V1_sx,
            "V1_sy": V1_sy,
            "F_serial": F_serial,
            "x1": x1,
            "x2": x2,
            "x": x,
            "y": y,
            "gridlines": gridlines,
        }

    def run_study_L2_proj():
        """
        Study the H(curl) L2 projection on the mapped disk. Compares the standard and
        filtered projected fields against a manufactured physical vector field,
        prints L2 projection errors and plots the projected components and their errors.
        Executed when --study is 'L2_proj'.
        Only for serial runs
        """
        omega = 4
        print(f"studying L2 proj of f in H_0(curl) .. with omega = {omega}")
        xs, ys = domain.coordinates
        r = sqrt(xs * xs + ys * ys)

        # f in H_0(curl;Omega)
        f_x = -xs + sin(omega * (ys + 2 * xs * xs)) * (r - R)
        f_y = -ys + cos(omega * (2 * xs - ys * ys)) * (r - R)
        f_phys = Tuple(f_x, f_y)

        from sympy import lambdify

        fx_call = lambdify([xs, ys], f_x)
        fy_call = lambdify([xs, ys], f_y)

        v = element_of(derham.V1, name="u")

        l = LinearForm(v, integral(domain, dot(f_phys, v)))
        lh = discretize(l, domain_h, V1_h, backend=backend)
        tilde_f = lh.assemble()

        # create fields and point to coeffs

        fh = Pi1((fx_call, fy_call))
        fh_filter = Pi1((fx_call, fy_call))

        M1_inv = inverse(M1, "cg", verbose=verbose, tol=tol)
        print("using standard L2 projection")
        fh_c = M1_inv @ tilde_f
        fh_c = P1 @ fh_c
        print("fh_c:")
        print(fh_c.toarray()[:])

        print("using filtered L2 projection")
        PTtilde_f = P1.T @ tilde_f
        fh_filter_c = M1_inv @ PTtilde_f
        fh_filter_c = P1 @ fh_filter_c

        # compute error and exit

        errx = lambda x1, x2: push_2d_hcurl(fh.fields[0], fh.fields[1], x1, x2, F)[
            0
        ] - fx_call(F(x1, x2)[0], F(x1, x2)[1])
        erry = lambda x1, x2: push_2d_hcurl(fh.fields[0], fh.fields[1], x1, x2, F)[
            1
        ] - fy_call(F(x1, x2)[0], F(x1, x2)[1])
        errx_filter = lambda x1, x2: push_2d_hcurl(
            fh_filter.fields[0], fh_filter.fields[1], x1, x2, F
        )[0] - fx_call(F(x1, x2)[0], F(x1, x2)[1])
        erry_filter = lambda x1, x2: push_2d_hcurl(
            fh_filter.fields[0], fh_filter.fields[1], x1, x2, F
        )[1] - fy_call(F(x1, x2)[0], F(x1, x2)[1])

        error_l2_fx = l2_norm_of(errx)
        error_l2_fy = l2_norm_of(erry)
        error_l2_fx_filter = l2_norm_of(errx_filter)
        error_l2_fy_filter = l2_norm_of(erry_filter)
        print("L2 norm of projection error on fx: {:.2e}".format(error_l2_fx))
        print("L2 norm of projection error on fy: {:.2e}".format(error_l2_fy))
        print(
            "L2 norm of projection error on fx (filtered): {:.2e}".format(
                error_l2_fx_filter
            )
        )
        print(
            "L2 norm of projection error on fy (filtered): {:.2e}".format(
                error_l2_fy_filter
            )
        )

        cst_wo_det = lambda x1, x2: 1
        cst_wi_det = lambda x1, x2: 1 * np.sqrt(F.metric_det(x1, x2))
        int_wo_det = derham_h.V0.integral(cst_wo_det)
        int_wi_det = derham_h.V0.integral(cst_wi_det)
        print("V0 - integral of 1 (no det):   {:.2e}".format(int_wo_det))
        print("V0 - integral of 1 (with det): {:.2e}".format(int_wi_det))
        print("radius of disk: {:.2e}".format(R))
        print("area of log domain: {:.2e}".format(2 * np.pi * R))
        print("area of disk:       {:.2e}".format(np.pi * R * R))

        int_wo_det = derham_h.V1.spaces[0].integral(cst_wo_det)
        int_wi_det = derham_h.V1.spaces[0].integral(cst_wi_det)
        print("V1.x - integral of 1 (no det):   {:.2e}".format(int_wo_det))
        print("V1.x - integral of 1 (with det): {:.2e}".format(int_wi_det))

        if not show_figs:
            return locals()

        plot_data = build_eval_context()
        if plot_data is None:
            return locals()

        x1 = plot_data["x1"]
        x2 = plot_data["x2"]
        x = plot_data["x"]
        y = plot_data["y"]
        gridlines = plot_data["gridlines"]

        # plot

        fx_values = np.empty_like(x1)
        fy_values = np.empty_like(x1)
        fx_filter_values = np.empty_like(x1)
        fy_filter_values = np.empty_like(x1)

        fx_ex_values = np.empty_like(x1)
        fy_ex_values = np.empty_like(x1)

        for i, x1i in enumerate(x1[:, 0]):
            for j, x2j in enumerate(x2[0, :]):

                xij, yij = F(x1i, x2j)
                fx_values[i, j], fy_values[i, j] = push_2d_hcurl(
                    fh.fields[0], fh.fields[1], x1i, x2j, F
                )
                fx_filter_values[i, j], fy_filter_values[i, j] = push_2d_hcurl(
                    fh_filter.fields[0], fh_filter.fields[1], x1i, x2j, F
                )
                fx_ex_values[i, j], fy_ex_values[i, j] = fx_call(xij, yij), fy_call(
                    xij, yij
                )

        fig1 = plot_field_and_error(
            r"f^x", 0, x, y, fx_values, fx_ex_values, *gridlines
        )
        # fig2.savefig(f'{visdir}/fx_{rp_str}.png')

        fig2 = plot_field_and_error(
            r"f^y", 0, x, y, fy_values, fy_ex_values, *gridlines
        )
        # fig3.savefig(f'{visdir}/fy_{rp_str}.png')

        print("done: showing fh")

        fig3 = plot_field_and_error(
            r"f^x filter", 0, x, y, fx_filter_values, fx_ex_values, *gridlines
        )
        # fig2.savefig(f'{visdir}/fx_filter_{rp_str}.png')

        fig4 = plot_field_and_error(
            r"f^y filter", 0, x, y, fy_filter_values, fy_ex_values, *gridlines
        )
        # fig3.savefig(f'{visdir}/fy_filter_{rp_str}.png')

        print("done: showing fh_filter")
        return locals()

    # --------------------------------------------------------------------------
    # Discretization
    # --------------------------------------------------------------------------

    # Differential operators
    D0, D1 = derham_h.derivatives(kind="linop")
    D1_T = D1.T

    # Extract spaces
    V0_h, V1_h, V2_h = derham_h.spaces

    I1 = IdentityOperator(V1_h.coeff_space)
    I2 = IdentityOperator(V2_h.coeff_space)

    # Conga projectors
    if smooth == 0:
        P1 = C0PolarProjection_V1(V1_h, hbc=True)
        P2 = C0PolarProjection_V2(V2_h)
    else:
        P1 = C1PolarProjection_V1(V1_h, hbc=True)
        P2 = C1PolarProjection_V2(V2_h)
    P1_T = P1.T
    P2_T = P2.T

    # Discrete bilinear forms
    a1_h = discretize(a1, domain_h, (derham_h.V1, derham_h.V1), backend=backend)
    a2_h = discretize(a2, domain_h, (derham_h.V2, derham_h.V2), backend=backend)

    hs = R / ncells[0]
    htheta = 2 * np.pi / ncells[1]

    # Mass matrices (StencilMatrix objects)

    # raw Mass matrices in W1 and W2 have unbounded values, this seems to be fine for the assembly but inverting these matrices leads to virtual nonsense...
    # so they need to be regularized below
    M1_raw = a1_h.assemble()
    M2_raw = a2_h.assemble()

    # regularization of M1 and M2 so that they are bounded and invertible:
    M1 = (htheta * hs) * (I1 - P1.T) @ (I1 - P1) + P1.T @ M1_raw @ P1
    M2 = (htheta * hs) * (I2 - P2.T) @ (I2 - P2) + P2.T @ M2_raw @ P2

    Pi0, Pi1, Pi2 = derham_h.projectors(nquads=[degree[0] + 10, degree[1] + 10])

    if study_L2_proj:
        run_study_L2_proj()
        return locals()

    # Geometric Projectors

    # --------------------------------------------------------------------------
    # Time integration setup
    # --------------------------------------------------------------------------
    t = 0

    # Callable exact fields
    Ex_ex = lambda t: (lambda x, y, t0=t: Ex_ex_t(t0, x, y))
    Ey_ex = lambda t: (lambda x, y, t0=t: Ey_ex_t(t0, x, y))
    Bz_ex = lambda t: (lambda x, y, t0=t: Bz_ex_t(t0, x, y))

    # Initial conditions, discrete fields -- here with a pull-back in the projections
    E_log = Pi1((Ex_ex(t), Ey_ex(t)))
    E_log.coeffs.update_ghost_regions()
    B_log = Pi2(Bz_ex(t))
    B_log.coeffs.update_ghost_regions()

    # Initial conditions, spline coefficients
    e = E_log.coeffs
    b = B_log.coeffs

    if study == "maxwell_wave":
        b = D1 @ e

    # Conga Projection
    e = P1 @ e
    b = P2 @ b

    V1_s, V1_theta = V1_h.spaces
    Ex_field = FemField(V1_s, coeffs=e[0])
    Ey_field = FemField(V1_theta, coeffs=e[1])
    B_field = FemField(V2_h, coeffs=b)
    V1_s.export_fields("Ex.h5", Ex_field=Ex_field)
    V1_theta.export_fields("Ey.h5", Ey_field=Ey_field)
    V2_h.export_fields("B.h5", B_field=B_field)

    if use_scipy:

        print(" -------------- SCIPY operators ------------ ")
        print(
            " This option, at the moment, will raise an Assertion error in feec/derivatives.py:"
        )
        print(" assert not (self.domain.parallel and not with_pads)")
        print(" For the moment, commenting out that line solves the issue.")
        print(" ------------------------------------------- ")

        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import inv as sp_inv

        from psydac.linalg.basic import MatrixFreeLinearOperator
        from psydac.linalg.utilities import array_to_psydac

        conga_curl_sp = (D1 @ P1).tosparse()
        strong_curl_sp = conga_curl_sp
        lambda_strong = lambda x: array_to_psydac(
            strong_curl_sp @ x.toarray(), V2_h.coeff_space
        )

        M2_sp = M2.tosparse()
        M1_sp = csc_matrix(M1.tosparse())
        M1inv_sp = sp_inv(M1_sp)
        weak_curl_sp = M1inv_sp @ strong_curl_sp.T @ M2_sp
        lambda_weak = lambda x: array_to_psydac(
            weak_curl_sp @ x.toarray(), V1_h.coeff_space
        )

        step_faraday_2d = MatrixFreeLinearOperator(
            V1_h.coeff_space, V2_h.coeff_space, lambda_strong
        )
        step_ampere_2d = MatrixFreeLinearOperator(
            V2_h.coeff_space, V1_h.coeff_space, lambda_weak
        )

    else:
        M1_inv = inverse(M1, "cg", verbose=verbose, tol=tol)
        step_ampere_2d = M1_inv @ P1_T @ D1_T @ M2
        step_faraday_2d = D1 @ P1

    Nt, dt, norm_curlh = compute_stable_dt(
        cfl, C_m=step_ampere_2d, dC_m=step_faraday_2d, V=V2_h, tau=tend, light_c=1
    )

    # If final time is given, recompute number of time steps
    if tend is None:
        tend = nsteps * dt
        print(f"final time (re)computed: {tend}")
    else:
        nsteps = Nt
        print(f"nsteps recomputed: {nsteps}")

    # --------------------------------------------------------------------------
    # Visualization setup
    # --------------------------------------------------------------------------
    def plot_fields_along_s(tstr):

        j0 = 0
        j1 = x2.shape[1] // 2
        theta0 = x2[0, j0]
        theta1 = x2[0, j1]
        name = "Ex"
        fig_line = plot_curve_along_s(
            name,
            "s",
            tstr,
            f"{theta0} and {theta1}",
            x1[:, j0],
            Ex_values[:, j0],
            Ex_ex_values[:, j0],
            -x1[:, j1],
            Ex_values[:, j1],
            Ex_ex_values[:, j1],
        )
        fig_line.savefig(f"{visdir}/{name}_line_{tstr}_{rp_str}.png")
        plt.close(fig_line)
        name = "Ey"
        fig_line = plot_curve_along_s(
            name,
            "s",
            tstr,
            f"{theta0} and {theta1}",
            x1[:, j0],
            Ey_values[:, j0],
            Ey_ex_values[:, j0],
            -x1[:, j1],
            Ey_values[:, j1],
            Ey_ex_values[:, j1],
        )
        fig_line.savefig(f"{visdir}/{name}_line_{tstr}_{rp_str}.png")
        plt.close(fig_line)
        name = "Bz"
        fig_line = plot_curve_along_s(
            name,
            "s",
            tstr,
            f"{theta0} and {theta1}",
            x1[:, j0],
            Bz_values[:, j0],
            Bz_ex_values[:, j0],
            -x1[:, j1],
            Bz_values[:, j1],
            Bz_ex_values[:, j1],
        )
        fig_line.savefig(f"{visdir}/{name}_line_{tstr}_{rp_str}.png")
        plt.close(fig_line)

    # Plot initial conditions
    eval_data = build_eval_context()
    if (
        eval_data is not None
    ):  # when mpi_rank is 0, since this needs to be done in serial
        V1_sx = eval_data["V1_sx"]
        V1_sy = eval_data["V1_sy"]
        V2_s = eval_data["V2_s"]
        F_serial = eval_data["F_serial"]
        x1 = eval_data["x1"]
        x2 = eval_data["x2"]
        x = eval_data["x"]
        y = eval_data["y"]
        gridlines = eval_data["gridlines"]

        (Ex_serial,) = V1_sx.import_fields("Ex.h5", "Ex_field")
        (Ey_serial,) = V1_sy.import_fields("Ey.h5", "Ey_field")
        (B_serial,) = V2_s.import_fields("B.h5", "B_field")

        Ex_ex_values = np.empty_like(x1)
        Ey_ex_values = np.empty_like(x1)
        Bz_ex_values = np.empty_like(x1)

        Ex_values = np.empty_like(x1)
        Ey_values = np.empty_like(x1)
        Bz_values = np.empty_like(x1)

        for i, x1i in enumerate(x1[:, 0]):
            for j, x2j in enumerate(x2[0, :]):

                Ex_values[i, j], Ey_values[i, j] = push_2d_hcurl(
                    Ex_serial, Ey_serial, x1i, x2j, F_serial
                )

                Bz_values[i, j] = push_2d_l2(B_serial, x1i, x2j, F_serial)

                xij, yij = F_serial(x1i, x2j)
                Ex_ex_values[i, j], Ey_ex_values[i, j] = Ex_ex_t(t, xij, yij), Ey_ex_t(
                    t, xij, yij
                )

                Bz_ex_values[i, j] = Bz_ex_t(t, xij, yij)

        # fields along s for fixed theta
        plot_fields_along_s(tstr="t0")

        # Electric field, x component
        fig = plot_field_and_error(r"E^x", 0, x, y, Ex_values, Ex_ex_values, *gridlines)
        fig.savefig(f"{visdir}/Ex_t0_{rp_str}.png")
        plt.close(fig)

        # Electric field, y component
        fig = plot_field_and_error(r"E^y", 0, x, y, Ey_values, Ey_ex_values, *gridlines)
        fig.savefig(f"{visdir}/Ey_t0_{rp_str}.png")
        plt.close(fig)

        # Magnetic field, z component
        fig = plot_field_and_error(r"B^z", 0, x, y, Bz_values, Bz_ex_values, *gridlines)
        fig.savefig(f"{visdir}/Bz_t0_{rp_str}.png")
        plt.close(fig)

        if show_figs:
            # Plot exact and approximate solutions at t = 0
            fig, axs = plt.subplots(3, 3, figsize=(12, 12))
            im0 = axs[0, 0].contourf(x, y, Ex_ex_values, 50)
            im1 = axs[0, 1].contourf(x, y, Ey_ex_values, 50)
            im2 = axs[0, 2].contourf(x, y, Bz_ex_values, 50)
            im3 = axs[1, 0].contourf(x, y, Ex_values, 50)
            im4 = axs[1, 1].contourf(x, y, Ey_values, 50)
            im5 = axs[1, 2].contourf(x, y, Bz_values, 50)
            im6 = axs[2, 0].contourf(x, y, Ex_values - Ex_ex_values, 50)
            im7 = axs[2, 1].contourf(x, y, Ey_values - Ey_ex_values, 50)
            im8 = axs[2, 2].contourf(x, y, Bz_values - Bz_ex_values, 50)
            axs[0, 0].set_title(r"$E^x$ at t = 0")
            axs[0, 1].set_title(r"$E^y$ at t = 0")
            axs[0, 2].set_title(r"$B^z$ at t = 0")
            axs[1, 0].set_title(r"$E_h^x$ at t = 0")
            axs[1, 1].set_title(r"$E_h^y$ at t = 0")
            axs[1, 2].set_title(r"$B_h^z$ at t = 0")
            axs[2, 0].set_title(r"$E^x - E_h^x$ at t = 0")
            axs[2, 1].set_title(r"$E^y - E_h^y$ at t = 0")
            axs[2, 2].set_title(r"$B^z - B_h^z$ at t = 0")
            for i in range(3):
                for j in range(3):
                    axs[i, j].plot(*gridlines[0], color="k")
                    axs[i, j].plot(*gridlines[1], color="k")
                    axs[i, j].set_xlabel("x", fontsize=14)
                    axs[i, j].set_ylabel("y", fontsize=14, rotation="horizontal")
                    axs[i, j].set_aspect("equal")
            add_colorbar(im0, axs[0, 0])
            add_colorbar(im1, axs[0, 1])
            add_colorbar(im2, axs[0, 2])
            add_colorbar(im3, axs[1, 0])
            add_colorbar(im4, axs[1, 1])
            add_colorbar(im5, axs[1, 2])
            add_colorbar(im6, axs[2, 0])
            add_colorbar(im7, axs[2, 1])
            add_colorbar(im8, axs[2, 2])
            fig.suptitle(
                "Compare Exact Solution and Approximate solution at initial time"
            )
            fig.tight_layout()

            # Need a small pause to show the plot of the initial condition
            plt.pause(0.1)

    # L2 norms (of ref solution)
    normx = lambda x1, x2: Ex_ex_t(t, *F(x1, x2))
    normy = lambda x1, x2: Ey_ex_t(t, *F(x1, x2))
    normz = lambda x1, x2: Bz_ex_t(t, *F(x1, x2))

    norm_l2_Ex = l2_norm_of(normx)
    norm_l2_Ey = l2_norm_of(normy)
    norm_l2_Bz = l2_norm_of(normz)

    # L2 errors
    errx = lambda x1, x2: push_2d_hcurl(E_log.fields[0], E_log.fields[1], x1, x2, F)[
        0
    ] - Ex_ex_t(t, *F(x1, x2))
    erry = lambda x1, x2: push_2d_hcurl(E_log.fields[0], E_log.fields[1], x1, x2, F)[
        1
    ] - Ey_ex_t(t, *F(x1, x2))
    errz = lambda x1, x2: push_2d_l2(B_log, x1, x2, F) - Bz_ex_t(t, *F(x1, x2))

    error_l2_Ex = l2_norm_of(errx) / norm_l2_Ex
    error_l2_Ey = l2_norm_of(erry) / norm_l2_Ey
    error_l2_Bz = l2_norm_of(errz) / norm_l2_Bz

    print(
        "L2 norm of rel. error on Ex(t,x,y) at initial time: {:.2e}".format(error_l2_Ex)
    )
    print(
        "L2 norm of rel. error on Ey(t,x,y) at initial time: {:.2e}".format(error_l2_Ey)
    )
    print(
        "L2 norm of rel. error on Bz(t,x,y) at initial time: {:.2e}".format(error_l2_Bz)
    )

    # --------------------------------------------------------------------------
    # Solution
    # --------------------------------------------------------------------------
    def Strang_update(dtau):
        # Strang splitting, 2nd order

        # b := b - dt/2 * curl e
        db = step_faraday_2d @ e
        b.mul_iadd(-dtau / 2, db)

        # e := e + dt * curl b
        de = step_ampere_2d @ b
        e.mul_iadd(dtau, de)

        # b := b - dt/2 * curl e
        db = step_faraday_2d @ e
        b.mul_iadd(-dtau / 2, db)

        # weights for Suzuki-Yoshida composition (4th-order splitting)

    gamma_1 = 1 / (2 - 2 ** (1 / 3))
    gamma_2 = 1 - 2 * gamma_1

    # Time loop
    for ts in range(1, nsteps + 1):

        print(f"step = {ts}/{nsteps}")

        if splitting_order == 2:

            Strang_update(dt)

        elif splitting_order == 4:

            Strang_update(dt * gamma_1)
            Strang_update(dt * gamma_2)
            Strang_update(dt * gamma_1)

        else:
            raise NotImplementedError("splitting_order must be 2 or 4")

        t += dt

        # diag
        e.update_ghost_regions()
        b.update_ghost_regions()

        e = P1 @ e
        b = P2 @ b

        print("ts = {:4d},  t = {:8.4f}".format(ts, t))

    N = 10
    if show_figs:
        V0_h.plot_2d_decomposition(F, refine=N)

    e = P1 @ e
    b = P2 @ b

    Ex_field = FemField(V1_s, coeffs=e[0])
    Ey_field = FemField(V1_theta, coeffs=e[1])
    B_field = FemField(V2_h, coeffs=b)
    V1_s.export_fields("Ex_final.h5", Ex_field=Ex_field)
    V1_theta.export_fields("Ey_final.h5", Ey_field=Ey_field)
    V2_h.export_fields("B_final.h5", B_field=B_field)

    if eval_data is not None:
        (Ex_serial,) = V1_sx.import_fields("Ex_final.h5", "Ex_field")
        (Ey_serial,) = V1_sy.import_fields("Ey_final.h5", "Ey_field")
        (B_serial,) = V2_s.import_fields("B_final.h5", "B_field")

        for i, x1i in enumerate(x1[:, 0]):
            for j, x2j in enumerate(x2[0, :]):

                Ex_values[i, j], Ey_values[i, j] = push_2d_hcurl(
                    Ex_serial, Ey_serial, x1i, x2j, F_serial
                )

                Bz_values[i, j] = push_2d_l2(B_serial, x1i, x2j, F_serial)

                xij, yij = F_serial(x1i, x2j)
                Ex_ex_values[i, j], Ey_ex_values[i, j] = Ex_ex_t(t, xij, yij), Ey_ex_t(
                    t, xij, yij
                )

                Bz_ex_values[i, j] = Bz_ex_t(t, xij, yij)

        # Error at final time
        error_Ex = abs(Ex_ex_values - Ex_values).max()
        error_Ey = abs(Ey_ex_values - Ey_values).max()
        error_Bz = abs(Bz_ex_values - Bz_values).max()
        print()
        print("Max-norm of error on Ex(t,x) at final time: {:.2e}".format(error_Ex))
        print("Max-norm of error on Ey(t,x) at final time: {:.2e}".format(error_Ey))
        print("Max-norm of error on Bz(t,x) at final time: {:.2e}".format(error_Bz))

    # L2 norms (of ref solution)
    normx = lambda x1, x2: Ex_ex_t(t, *F(x1, x2))
    normy = lambda x1, x2: Ey_ex_t(t, *F(x1, x2))
    normz = lambda x1, x2: Bz_ex_t(t, *F(x1, x2))

    norm_l2_Ex = l2_norm_of(normx)
    norm_l2_Ey = l2_norm_of(normy)
    norm_l2_Bz = l2_norm_of(normz)

    E1_final = FemField(V1_s, coeffs=e[0])
    E2_final = FemField(V1_theta, coeffs=e[1])
    B_final = FemField(V2_h, coeffs=b)

    # L2 errors
    errx = lambda x1, x2: push_2d_hcurl(E1_final, E2_final, x1, x2, F)[0] - Ex_ex_t(
        t, *F(x1, x2)
    )
    erry = lambda x1, x2: push_2d_hcurl(E1_final, E2_final, x1, x2, F)[1] - Ey_ex_t(
        t, *F(x1, x2)
    )
    errz = lambda x1, x2: push_2d_l2(B_final, x1, x2, F) - Bz_ex_t(t, *F(x1, x2))

    error_l2_Ex = l2_norm_of(errx) / norm_l2_Ex
    error_l2_Ey = l2_norm_of(erry) / norm_l2_Ey
    error_l2_Bz = l2_norm_of(errz) / norm_l2_Bz

    print()
    print(
        "L2 norm of rel. error on Ex(t,x,y) at final time: {:.2e}".format(error_l2_Ex)
    )
    print(
        "L2 norm of rel. error on Ey(t,x,y) at final time: {:.2e}".format(error_l2_Ey)
    )
    print(
        "L2 norm of rel. error on Bz(t,x,y) at final time: {:.2e}".format(error_l2_Bz)
    )

    if eval_data is not None and show_figs:
        # Plot exact and approximate solution at final time
        fig1, axs = plt.subplots(3, 3, figsize=(12, 12))
        im0 = axs[0, 0].contourf(x, y, Ex_ex_values, 50)
        im1 = axs[0, 1].contourf(x, y, Ey_ex_values, 50)
        im2 = axs[0, 2].contourf(x, y, Bz_ex_values, 50)
        im3 = axs[1, 0].contourf(x, y, Ex_values, 50)
        im4 = axs[1, 1].contourf(x, y, Ey_values, 50)
        im5 = axs[1, 2].contourf(x, y, Bz_values, 50)
        im6 = axs[2, 0].contourf(x, y, Ex_values - Ex_ex_values, 50)
        im7 = axs[2, 1].contourf(x, y, Ey_values - Ey_ex_values, 50)
        im8 = axs[2, 2].contourf(x, y, Bz_values - Bz_ex_values, 50)
        axs[0, 0].set_title(r"$E^x$ at t = {:10.3e}".format(t))
        axs[0, 1].set_title(r"$E^y$ at t = {:10.3e}".format(t))
        axs[0, 2].set_title(r"$B$ at t = {:10.3e}".format(t))
        axs[1, 0].set_title(r"$E_h^x$ at t = {:10.3e}".format(t))
        axs[1, 1].set_title(r"$E_h^y$ at t = {:10.3e}".format(t))
        axs[1, 2].set_title(r"$B_h$ at t = {:10.3e}".format(t))
        axs[2, 0].set_title(r"$E^x - E^x_h$ at t = {:10.3e}".format(t))
        axs[2, 1].set_title(r"$E^y - E^y_h$ at t = {:10.3e}".format(t))
        axs[2, 2].set_title(r"$B - B_h$ at t = {:10.3e}".format(t))
        for i in range(3):
            for j in range(3):
                axs[i, j].plot(*gridlines[0], color="k")
                axs[i, j].plot(*gridlines[1], color="k")
                axs[i, j].set_xlabel("x", fontsize=14)
                axs[i, j].set_ylabel("y", fontsize=14, rotation="horizontal")
                axs[i, j].set_aspect("equal")
        add_colorbar(im0, axs[0, 0])
        add_colorbar(im1, axs[0, 1])
        add_colorbar(im2, axs[0, 2])
        add_colorbar(im3, axs[1, 0])
        add_colorbar(im4, axs[1, 1])
        add_colorbar(im5, axs[1, 2])
        add_colorbar(im6, axs[2, 0])
        add_colorbar(im7, axs[2, 1])
        add_colorbar(im8, axs[2, 2])
        fig1.suptitle("Compare Exact Solution and Approximate solution at final time")
        fig1.tight_layout()

        # fields along s, final time
        plot_fields_along_s(tstr="T")

        # Electric field, x component
        fig = plot_field_and_error(
            r"E^x", tend, x, y, Ex_values, Ex_ex_values, *gridlines
        )
        fig.savefig(f"{visdir}/Ex_T_{rp_str}.png")
        plt.close(fig)  # fig.clf()

        # Electric field, y component
        fig = plot_field_and_error(
            r"E^y", tend, x, y, Ey_values, Ey_ex_values, *gridlines
        )
        fig.savefig(f"{visdir}/Ey_T_{rp_str}.png")
        plt.close(fig)  # fig.clf()

        # Magnetic field, z component
        fig = plot_field_and_error(
            r"B^z", tend, x, y, Bz_values, Bz_ex_values, *gridlines
        )
        fig.savefig(f"{visdir}/Bz_T_{rp_str}.png")
        plt.close(fig)

    return locals()


# ==============================================================================
# PARSER
# ==============================================================================
def parse_input_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Solve Transverse Time Harmonic Maxwell system on analytical disk with CONGA polar spline method.",
    )

    parser.add_argument(
        "--study",
        choices=("L2_proj", "maxwell_bessel", "maxwell_wave"),
        default="maxwell_bessel",
        dest="study",
        help="Study to be performed",
    )

    parser.add_argument(
        "-S",
        action="store_true",
        dest="use_spline_mapping",
        help="Use spline mapping in finite element calculations",
    )

    parser.add_argument(
        "-D",
        type=float,
        default=0,
        dest="shift_D",
        help="Shafranov shift for parametrization of Disk",
    )

    parser.add_argument(
        "-n",
        "--ncells",
        nargs=2,
        type=int,
        default=[10, 20],
        dest="ncells",
        help="Number of grid cells (elements) along each dimension",
    )

    parser.add_argument(
        "-s",
        "--smoothness",
        type=int,
        default=0,
        dest="smooth",
        help="Smoothness at the pole. Only C0 and C1 possible. C0 as default.",
    )

    parser.add_argument(
        "-d",
        "--degree",
        nargs=2,
        type=int,
        default=[3, 3],
        dest="degree",
        help="Polynomial spline degrees",
    )

    parser.add_argument(
        "-o",
        "--splitting_order",
        type=int,
        default=2,
        choices=[2, 4],
        help="Order of accuracy of operator splitting",
    )

    # ...
    time_opts = parser.add_mutually_exclusive_group()
    time_opts.add_argument(
        "-t",
        type=int,
        default=1,
        dest="nsteps",
        metavar="NSTEPS",
        help="Number of time-steps to be taken",
    )
    time_opts.add_argument(
        "-T",
        type=float,
        dest="tend",
        metavar="END_TIME",
        help="Run simulation until given final time",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="Tolerance for iterative solver (L2-norm of residual)",
    )

    parser.add_argument(
        "--scipy",
        action="store_true",
        dest="use_scipy",
        help="Use scipy matrices and direct inverses",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print convergence information of iterative solver",
    )

    # Read and return input arguments
    return parser.parse_args()


# ==============================================================================
# SCRIPT FUNCTIONALITY
# ==============================================================================
if __name__ == "__main__":

    # Parse CLI arguments
    args = parse_input_arguments()

    # Run simulation
    print(f"Running function 'run_maxwell_2d_TE' with arguments:")
    print(f"{args}")
    namespace = run_maxwell_2d_TE(**vars(args))

    # Make all variables available (including imported modules)
    globals().update(namespace)

    # Keep matplotlib windows open
    plt.show()
