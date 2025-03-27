"""
    solver for the TD Maxwell problem: find E(t) in H(curl), B in L2, such that

      dt E - curl B = -J             on \\Omega
      dt B + curl E = 0              on \\Omega
      n x E = n x E_bc      on \\partial \\Omega

    with Ampere discretized weakly and Faraday discretized strongly, in a broken-FEEC approach on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h
                     (Eh)          (Bh)
"""

from pytest import param
from mpi4py import MPI

import os
import numpy as np
import scipy as sp
from collections import OrderedDict
import matplotlib.pyplot as plt

from sympy import lambdify, Matrix

from scipy.sparse.linalg import spsolve
from scipy import special

from sympde.calculus import dot
from sympde.topology import element_of
from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral, Norm
from sympde.topology import Derham

from psydac.api.settings import PSYDAC_BACKENDS
from psydac.feec.pull_push import pull_2d_hcurl
from psydac.feec.multipatch.api import discretize
from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator
from psydac.feec.multipatch.operators import HodgeOperator, get_K0_and_K0_inv, get_K1_and_K1_inv
# , write_field_to_diag_grid,
from psydac.fem.plotting_utilities import plot_field_2d as plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
# , get_praxial_Gaussian_beam_E, get_easy_Gaussian_beam_E, get_easy_Gaussian_beam_B,get_easy_Gaussian_beam_E_2, get_easy_Gaussian_beam_B_2
from psydac.feec.multipatch.examples.ppc_test_cases import get_source_and_solution_hcurl, get_div_free_pulse, get_curl_free_pulse, get_Delta_phi_pulse, get_Gaussian_beam
from psydac.feec.multipatch.utils_conga_2d import DiagGrid, P0_phys, P1_phys, P2_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities import time_count  # , export_sol, import_sol
from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic import FemField
from psydac.feec.multipatch.non_matching_operators import construct_hcurl_conforming_projection, construct_h1_conforming_projection
from psydac.feec.multipatch.multipatch_domain_utilities import build_cartesian_multipatch_domain

from psydac.api.postprocessing import OutputManager, PostProcessManager


def solve_td_maxwell_pbm(*,
                         nc=4,
                         deg=4,
                         final_time=20,
                         cfl_max=0.8,
                         dt_max=None,
                         domain_name='pretzel_f',
                         backend='pyccel-gcc',
                         source_type='zero',
                         source_omega=None,
                         source_proj='P_geom',
                         conf_proj='BSP',
                         gamma_h=10.,
                         project_sol=False,
                         filter_source=True,
                         quad_param=1,
                         E0_type='zero',
                         E0_proj='P_L2',
                         hide_plots=True,
                         plot_dir=None,
                         plot_time_ranges=None,
                         plot_source=False,
                         plot_divE=False,
                         diag_dt=None,
                         #        diag_dtau        = None,
                         cb_min_sol=None,
                         cb_max_sol=None,
                         m_load_dir=None,
                         th_sol_filename="",
                         source_is_harmonic=False,
                         domain_lims=None
                         ):
    """
    solver for the TD Maxwell problem: find E(t) in H(curl), B in L2, such that

      dt E - curl B = -J             on \\Omega
      dt B + curl E = 0              on \\Omega
      n x E = n x E_bc      on \\partial \\Omega

    with Ampere discretized weakly and Faraday discretized strongly, in a broken-FEEC approach on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h
                     (Eh)          (Bh)

    Parameters
    ----------
    nc : int
        Number of cells (same along each direction) in every patch.

    deg : int
        Polynomial degree (same along each direction) in every patch, for the
        spline space V0 in H1.

    final_time : float
        Final simulation time. Given that the speed of light is set to c=1,
        this can be easily chosen based on the wave transit time in the domain.

    cfl_max : float
        Maximum Courant parameter in the simulation domain, used to determine
        the time step size.

    dt_max : float
        Maximum time step size, which has to be met together with cfl_max. This
        additional constraint is useful to resolve a time-dependent source.

    domain_name : str
        Name of the multipatch geometry used in the simulation, to be chosen
        among those available in the function `build_multipatch_domain`.

    backend : str
        Name of the backend used for acceleration of the computational kernels,
        to be chosen among the available keys of the PSYDAC_BACKENDS dict.

    source_type : str {'zero' | 'pulse' | 'cf_pulse' | 'Il_pulse'}
        Name that identifies the space-time profile of the current source, to be
        chosen among those available in the function get_source_and_solution().
        Available options:
            - 'zero'    : no current source
            - 'pulse'   : div-free current source, time-harmonic
            - 'cf_pulse': curl-free current source, time-harmonic
            - 'Il_pulse': Issautier-like pulse, with both a div-free and a
                          curl-free component, not time-harmonic.

    source_omega : float
        Pulsation of the time-harmonic component (if any) of a time-dependent
        current source.

    source_proj : str {'P_geom' | 'P_L2'}
        Name of the approximation operator for the current source: 'P_geom' is
        a geometric projector (based on inter/histopolation) which yields the
        primal degrees of freedom; 'P_L2' is an L2 projector which yields the
        dual degrees of freedom. Change of basis from primal to dual (and vice
        versa) is obtained through multiplication with the proper Hodge matrix.

    conf_proj : str {'BSP' | 'GSP'}
        Kind of conforming projection operator. Choose 'BSP' for an operator
        based on the spline coefficients, which has maximum data locality.
        Choose 'GSP' for an operator based on the geometric degrees of freedom,
        which requires a change of basis (from B-spline to geometric, and then
        vice versa) on the patch interfaces.

    gamma_h : float
        Jump penalization parameter.

    project_sol : bool
        Whether the solution fields should be projected onto the corresponding
        conforming spaces before plotting them.

    filter_source : bool
        If True, the current source will be filtered with the conforming
        projector operator (or its dual, depending on which basis is used).

    quad_param : int
        Multiplicative factor for the number of quadrature points; set
        `quad_param` > 1 if you suspect that the quadrature is not accurate.

    E0_type : str {'zero', 'th_sol', 'pulse'}
        Initial conditions for the electric field. Choose 'zero' for E0=0,
        'th_sol' for a field obtained from the time-harmonic Maxwell solver
        (must provide a time-harmonic current source and set `source_omega`),
        and 'pulse' for a non-zero field localized in a small region.

    E0_proj : str {'P_geom' | 'P_L2'}
        Name of the approximation operator for the initial electric field E0
        (see source_proj for details). Only relevant if E0 is not zero.

    hide_plots : bool
        If True, no windows are opened to show the figures interactively.

    plot_dir : str
        Path to the directory where the figures will be saved.

    plot_time_ranges : list
        List of lists, of the form `[[start, end], dtp]`, where `[start, end]`
        is a time interval and `dtp` is the time between two successive plots.

    plot_source : bool
        If True, plot the discrete field that approximates the current source.

    plot_divE : bool
        If True, compute and plot the (weak) divergence of the electric field.

    diag_dt : float
        Time elapsed between two successive calculations of scalar diagnostic
        quantities.

    cb_min_sol : float
        Minimum value to be used in colorbars when visualizing the solution.

    cb_max_sol : float
        Maximum value to be used in colorbars when visualizing the solution.

    m_load_dir : str
        Path to directory for matrix storage.

    th_sol_filename : str
        Path to file with time-harmonic solution (to be used in conjuction with
        `source_is_harmonic = True` and `E0_type = 'th_sol'`).

    """
    diags = {}

    # ncells = [nc, nc]
    degree = [deg, deg]

    if source_omega is not None:
        period_time = 2 * np.pi / source_omega
        Nt_pp = period_time // dt_max

    if plot_time_ranges is None:
        plot_time_ranges = [
            [[0, final_time], final_time]
        ]

    if diag_dt is None:
        diag_dt = 0.1

    # if backend is None:
    #     if domain_name in ['pretzel', 'pretzel_f'] and nc > 8:
    #         backend = 'numba'
    #     else:
    #         backend = 'python'
    # print('[note: using '+backend_language+ ' backends in discretize functions]')
    if m_load_dir is not None:
        if not os.path.exists(m_load_dir):
            os.makedirs(m_load_dir)

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_td_maxwell_pbm function with: ')
    print(' ncells = {}'.format(nc))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' E0_type = {}'.format(E0_type))
    print(' E0_proj = {}'.format(E0_proj))
    print(' source_type = {}'.format(source_type))
    print(' source_proj = {}'.format(source_proj))
    print(' backend = {}'.format(backend))
    # TODO: print other parameters
    print('---------------------------------------------------------------------------------------------------------')

    debug = False

    print()
    print(' -- building discrete spaces and operators  --')

    t_stamp = time_count()
    print(' .. multi-patch domain...')
    if domain_name == 'refined_square' or domain_name == 'square_L_shape':
        int_x, int_y = domain_lims
        domain = build_cartesian_multipatch_domain(nc, int_x, int_y, mapping='identity')

    else:
        domain = build_multipatch_domain(domain_name=domain_name)

    if isinstance(nc, int):
        ncells = [nc, nc]
    elif ncells.ndim == 1:
        ncells = {patch.name: [nc[i], nc[i]]
                    for (i, patch) in enumerate(domain.interior)}
    elif ncells.ndim == 2:
        ncells = {patch.name: [nc[int(patch.name[2])][int(patch.name[4])], 
                nc[int(patch.name[2])][int(patch.name[4])]] for patch in domain.interior}

    mappings = OrderedDict([(P.logical_domain, P.mapping)
                           for P in domain.interior])
    mappings_list = list(mappings.values())

    # for diagnosttics
    diag_grid = DiagGrid(mappings=mappings, N_diag=100)

    t_stamp = time_count(t_stamp)
    print(' .. derham sequence...')
    derham = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print(' .. discrete domain...')
    domain_h = discretize(domain, ncells=ncells)

    t_stamp = time_count(t_stamp)
    print(' .. discrete derham sequence...')

    derham_h = discretize(derham, domain_h, degree=degree)

    t_stamp = time_count(t_stamp)
    print(' .. commuting projection operators...')
    nquads = [4 * (d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    t_stamp = time_count(t_stamp)
    print(' .. multi-patch spaces...')
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))
    diags['ndofs_V0'] = V0h.nbasis
    diags['ndofs_V1'] = V1h.nbasis
    diags['ndofs_V2'] = V2h.nbasis

    t_stamp = time_count(t_stamp)
    print(' .. Id operator and matrix...')
    I1 = IdLinearOperator(V1h)
    I1_m = I1.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    # other option: define as Hodge Operators:
    H0 = HodgeOperator(
        V0h,
        domain_h,
        backend_language=backend,
        load_dir=m_load_dir,
        load_space_index=0)
    H1 = HodgeOperator(
        V1h,
        domain_h,
        backend_language=backend,
        load_dir=m_load_dir,
        load_space_index=1)
    H2 = HodgeOperator(
        V2h,
        domain_h,
        backend_language=backend,
        load_dir=m_load_dir,
        load_space_index=2)

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H0_m = M0_m ...')
    H0_m = H0.to_sparse_matrix()
    t_stamp = time_count(t_stamp)
    print(' .. dual Hodge matrix dH0_m = inv_M0_m ...')
    dH0_m = H0.get_dual_Hodge_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix H1_m = M1_m ...')
    H1_m = H1.to_sparse_matrix()
    t_stamp = time_count(t_stamp)
    print(' .. dual Hodge matrix dH1_m = inv_M1_m ...')
    dH1_m = H1.get_dual_Hodge_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. Hodge matrix dH2_m = M2_m ...')
    H2_m = H2.to_sparse_matrix()
    print(' .. dual Hodge matrix dH2_m = inv_M2_m ...')
    dH2_m = H2.get_dual_Hodge_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print(' .. conforming Projection operators...')
    cP0_m = construct_h1_conforming_projection(V0h, hom_bc=False)
    cP1_m = construct_hcurl_conforming_projection(V1h, hom_bc=False)

    if conf_proj == 'GSP':
        print(' [* GSP-conga: using Geometric Spline conf Projections ]')
        K0, K0_inv = get_K0_and_K0_inv(V0h, uniform_patches=False)
        cP0_m = K0_inv @ cP0_m @ K0
        K1, K1_inv = get_K1_and_K1_inv(V1h, uniform_patches=False)
        cP1_m = K1_inv @ cP1_m @ K1
    elif conf_proj == 'BSP':
        print(' [* BSP-conga: using B-Spline conf Projections ]')
    else:
        raise ValueError(conf_proj)

    t_stamp = time_count(t_stamp)
    print(' .. broken differential operators...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()

    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Conga (projection-based) matrices
    t_stamp = time_count(t_stamp)
    dH1_m = dH1_m.tocsr()
    H2_m = H2_m.tocsr()
    cP1_m = cP1_m.tocsr()
    bD1_m = bD1_m.tocsr()

    print(' .. matrix of the primal curl (in primal bases)...')
    C_m = bD1_m @ cP1_m
    print(' .. matrix of the dual curl (also in primal bases)...')

    from sympde.calculus import grad, dot, curl, cross
    from sympde.topology import NormalVector
    from sympde.expr.expr import BilinearForm
    from sympde.topology import elements_of

    u, v = elements_of(derham.V1, names='u, v')
    nn = NormalVector('nn')
    boundary = domain.boundary
    expr_b = cross(nn, u) * cross(nn, v)

    a = BilinearForm((u, v), integral(boundary, expr_b))
    ah = discretize(a, domain_h, [V1h, V1h], backend=PSYDAC_BACKENDS[backend],)
    A_eps = ah.assemble().tosparse()

    dC_m = dH1_m @ C_m.transpose() @ H2_m

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute stable time step size based on max CFL and max dt
    dt = compute_stable_dt(C_m=C_m, dC_m=dC_m, cfl_max=cfl_max, dt_max=dt_max)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Absorbing dC_m
    CH2 = C_m.transpose() @ H2_m
    H1A = H1_m + dt * A_eps
    dC_m = sp.sparse.linalg.spsolve(H1A, CH2)

    dCH1_m = sp.sparse.linalg.spsolve(H1A, H1_m)

    print(' .. matrix of the dual div (still in primal bases)...')
    div_m = dH0_m @ cP0_m.transpose() @ bD0_m.transpose() @ H1_m

    # jump stabilization (may not be needed)
    t_stamp = time_count(t_stamp)
    print(' .. jump stabilization matrix...')
    jump_penal_m = I1_m - cP1_m
    JP_m = jump_penal_m.transpose() * H1_m * jump_penal_m

    # t_stamp = time_count(t_stamp)
    # print(' .. full operator matrix...')
    # print('STABILIZATION: gamma_h = {}'.format(gamma_h))
    # pre_A_m = cP1_m.transpose() @ ( eta * H1_m + mu * pre_CC_m - nu * pre_GD_m )  # useful for the boundary condition (if present)
    # A_m = pre_A_m @ cP1_m + gamma_h * JP_m

    print(" Reduce time step to match the simulation final time:")
    Nt = int(np.ceil(final_time / dt))
    dt = final_time / Nt
    print(f"   . Time step size  : dt = {dt}")
    print(f"   . Nb of time steps: Nt = {Nt}")

    # ...
    def is_plotting_time(nt, *, dt=dt, Nt=Nt,
                         plot_time_ranges=plot_time_ranges):
        if nt in [0, Nt]:
            return True
        for [start, end], dt_plots in plot_time_ranges:
            # number of time steps between two successive plots
            ds = max(dt_plots // dt, 1)
            if (start <= nt * dt <= end) and (nt % ds == 0):
                return True
        return False
    # ...

    # Number of time step between two successive calculations of the scalar
    # diagnostics
    diag_nt = max(int(diag_dt // dt), 1)

    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(
        ' total nb of time steps: Nt = {}, final time: T = {:5.4f}'.format(
            Nt,
            final_time))
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' plotting times: the solution will be plotted for...')
    for nt in range(Nt + 1):
        if is_plotting_time(nt):
            print(' * nt = {}, t = {:5.4f}'.format(nt, dt * nt))
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # source

    t_stamp = time_count(t_stamp)
    print()
    print(' -- getting source --')
    f0_c = None
    f0_harmonic_c = None
    if source_type == 'zero':

        f0 = None
        f0_harmonic = None

    elif source_type == 'pulse':

        f0 = get_div_free_pulse(x_0=1.0, y_0=1.0, domain=domain)

    elif source_type == 'cf_pulse':

        f0 = get_curl_free_pulse(x_0=1.0, y_0=1.0, domain=domain)

    elif source_type == 'Il_pulse':  # Issautier-like pulse
        # source will be
        #   J = curl A + cos(om*t) * grad phi
        # so that
        #   dt rho = - div J = - cos(om*t) Delta phi
        # for instance, with rho(t=0) = 0 this  gives
        #   rho = - sin(om*t)/om * Delta phi
        # and Gauss' law reads
        #  div E = rho = - sin(om*t)/om * Delta phi
        f0 = get_div_free_pulse(
            x_0=1.0, y_0=1.0, domain=domain)  # this is curl A
        f0_harmonic = get_curl_free_pulse(
            x_0=1.0, y_0=1.0, domain=domain)  # this is grad phi
        assert not source_is_harmonic

        rho0 = get_Delta_phi_pulse(
            x_0=1.0, y_0=1.0, domain=domain)  # this is Delta phi
        tilde_rho0_c = derham_h.get_dual_dofs(
            space='V0',
            f=rho0,
            backend_language=backend,
            return_format='numpy_array')
        tilde_rho0_c = cP0_m.transpose() @ tilde_rho0_c
        rho0_c = dH0_m.dot(tilde_rho0_c)
    else:

        f0, u_bc, u_ex, curl_u_ex, div_u_ex = get_source_and_solution_hcurl(
            source_type=source_type, domain=domain, domain_name=domain_name,
        )
        assert u_bc is None  # only homogeneous BC's for now

    # f0_c = np.zeros(V1h.nbasis)

    if source_omega is not None:
        f0_harmonic = f0
        f0 = None
        if E0_type == 'th_sol':
            # use source enveloppe for smooth transition from 0 to 1
            def source_enveloppe(tau):
                return (special.erf((tau / 25) - 2) - special.erf(-2)) / 2
        else:
            def source_enveloppe(tau):
                return 1

    t_stamp = time_count(t_stamp)
    tilde_f0_c = f0_c = None
    tilde_f0_harmonic_c = f0_harmonic_c = None
    if source_proj == 'P_geom':
        print(' .. projecting the source with commuting projection...')
        if f0 is not None:
            f0_h = P1_phys(f0, P1, domain, mappings_list)
            f0_c = f0_h.coeffs.toarray()
            tilde_f0_c = H1_m.dot(f0_c)
        if f0_harmonic is not None:
            f0_harmonic_h = P1_phys(f0_harmonic, P1, domain, mappings_list)
            f0_harmonic_c = f0_harmonic_h.coeffs.toarray()
            tilde_f0_harmonic_c = H1_m.dot(f0_harmonic_c)

    elif source_proj == 'P_L2':
        # helper: save/load coefs
        if f0 is not None:
            if source_type == 'Il_pulse':
                source_name = 'Il_pulse_f0'
            else:
                source_name = source_type
            sdd_filename = m_load_dir + '/' + source_name + \
                '_dual_dofs_qp{}.npy'.format(quad_param)
            if os.path.exists(sdd_filename):
                print(
                    ' .. loading source dual dofs from file {}'.format(sdd_filename))
                tilde_f0_c = np.load(sdd_filename)
            else:
                print(' .. projecting the source f0 with L2 projection...')
                tilde_f0_c = derham_h.get_dual_dofs(
                    space='V1', f=f0, backend_language=backend, return_format='numpy_array')
                print(' .. saving source dual dofs to file {}'.format(sdd_filename))
                np.save(sdd_filename, tilde_f0_c)
        if f0_harmonic is not None:
            if source_type == 'Il_pulse':
                source_name = 'Il_pulse_f0_harmonic'
            else:
                source_name = source_type
            sdd_filename = m_load_dir + '/' + source_name + \
                '_dual_dofs_qp{}.npy'.format(quad_param)
            if os.path.exists(sdd_filename):
                print(
                    ' .. loading source dual dofs from file {}'.format(sdd_filename))
                tilde_f0_harmonic_c = np.load(sdd_filename)
            else:
                print(' .. projecting the source f0_harmonic with L2 projection...')
                tilde_f0_harmonic_c = derham_h.get_dual_dofs(
                    space='V1', f=f0_harmonic, backend_language=backend, return_format='numpy_array')
                print(' .. saving source dual dofs to file {}'.format(sdd_filename))
                np.save(sdd_filename, tilde_f0_harmonic_c)

    else:
        raise ValueError(source_proj)

    t_stamp = time_count(t_stamp)
    if filter_source:
        print(' .. filtering the source...')
        if tilde_f0_c is not None:
            tilde_f0_c = cP1_m.transpose() @ tilde_f0_c
        if tilde_f0_harmonic_c is not None:
            tilde_f0_harmonic_c = cP1_m.transpose() @ tilde_f0_harmonic_c

    if tilde_f0_c is not None:
        f0_c = dH1_m.dot(tilde_f0_c)

        if debug:
            title = 'f0 part of source'
            params_str = 'omega={}_gamma_h={}_Pf={}'.format(
                source_omega, gamma_h, source_proj)
            plot_field(numpy_coeffs=f0_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title,
                       filename=plot_dir + '/' + params_str + '_f0.pdf',
                       plot_type='components', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)
            plot_field(numpy_coeffs=f0_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title,
                       filename=plot_dir + '/' + params_str + '_f0_vf.pdf',
                       plot_type='vector_field', cb_min=None, cb_max=None, hide_plot=hide_plots)
            divf0_c = div_m @ f0_c
            title = 'div f0'
            plot_field(numpy_coeffs=divf0_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title,
                       filename=plot_dir + '/' + params_str + '_divf0.pdf',
                       plot_type='components', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

    if tilde_f0_harmonic_c is not None:
        f0_harmonic_c = dH1_m.dot(tilde_f0_harmonic_c)

        if debug:
            title = 'f0_harmonic part of source'
            params_str = 'omega={}_gamma_h={}_Pf={}'.format(
                source_omega, gamma_h, source_proj)
            plot_field(numpy_coeffs=f0_harmonic_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title,
                       filename=plot_dir + '/' + params_str + '_f0_harmonic.pdf',
                       plot_type='components', cb_min=None, cb_max=None, hide_plot=hide_plots)
            plot_field(numpy_coeffs=f0_harmonic_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title,
                       filename=plot_dir + '/' + params_str + '_f0_harmonic_vf.pdf',
                       plot_type='vector_field', cb_min=None, cb_max=None, hide_plot=hide_plots)
            divf0_c = div_m @ f0_harmonic_c
            title = 'div f0_harmonic'
            plot_field(numpy_coeffs=divf0_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title,
                       filename=plot_dir + '/' + params_str + '_divf0_harmonic.pdf',
                       plot_type='components', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

    # else:
    #     raise NotImplementedError

    if f0_c is None:
        f0_c = np.zeros(V1h.nbasis)

    # if plot_source and plot_dir:
    #     plot_field(numpy_coeffs=f0_c, Vh=V1h, space_kind='hcurl', domain=domain, title='f0_h with P = '+source_proj, filename=plot_dir+'/f0h_'+source_proj+'.png', hide_plot=hide_plots)
        # plot_field(numpy_coeffs=f0_c, Vh=V1h, plot_type='vector_field', space_kind='hcurl', domain=domain, title='f0_h with P = '+source_proj, filename=plot_dir+'/f0h_'+source_proj+'_vf.png', hide_plot=hide_plots)

    t_stamp = time_count(t_stamp)

    def plot_J_source_nPlusHalf(f_c, nt):
        print(' .. plotting the source...')
        title = r'source $J^{n+1/2}_h$ (amplitude)' + \
            ' for $\\omega = {}$, $n = {}$'.format(source_omega, nt)
        params_str = 'omega={}_gamma_h={}_Pf={}'.format(
            source_omega, gamma_h, source_proj)
        plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title,
                   filename=plot_dir + '/' + params_str +
                   '_Jh_nt={}.pdf'.format(nt),
                   plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)
        title = r'source $J^{n+1/2}_h$' + \
            ' for $\\omega = {}$, $n = {}$'.format(source_omega, nt)
        plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, title=title,
                   filename=plot_dir + '/' + params_str +
                   '_Jh_vf_nt={}.pdf'.format(nt),
                   plot_type='vector_field', vf_skip=1, hide_plot=hide_plots)

    def plot_E_field(E_c, nt, project_sol=False, plot_divE=False):

        # only E for now
        if plot_dir:

            plot_omega_normalized_sol = (source_omega is not None)
            # project the homogeneous solution on the conforming problem space
            if project_sol:
                # t_stamp = time_count(t_stamp)
                print(
                    ' .. projecting the homogeneous solution on the conforming problem space...')
                Ep_c = cP1_m.dot(E_c)
            else:
                Ep_c = E_c
                print(
                    ' .. NOT projecting the homogeneous solution on the conforming problem space')
            if plot_omega_normalized_sol:
                print(' .. plotting the E/omega field...')
                u_c = (1 / source_omega) * Ep_c
                title = r'$u_h = E_h/\omega$ (amplitude) for $\omega = {:5.4f}$, $t = {:5.4f}$'.format(
                    source_omega, dt * nt)
                params_str = 'omega={:5.4f}_gamma_h={}_Pf={}_Nt_pp={}'.format(
                    source_omega, gamma_h, source_proj, Nt_pp)
            else:
                print(' .. plotting the E field...')
                if E0_type == 'pulse':
                    title = r'$t = {:5.4f}$'.format(dt * nt)
                else:
                    title = r'$E_h$ (amplitude) at $t = {:5.4f}$'.format(
                        dt * nt)
                u_c = Ep_c
                params_str = f'gamma_h={gamma_h}_dt={dt}'

            plot_field(numpy_coeffs=u_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, title=title,
                       filename=plot_dir + '/' + params_str +
                       '_Eh_nt={}.pdf'.format(nt),
                       plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

            if plot_divE:
                params_str = f'gamma_h={gamma_h}_dt={dt}'
                if source_type == 'Il_pulse':
                    plot_type = 'components'
                    rho_c = rho0_c * \
                        np.sin(source_omega * dt * nt) / source_omega
                    rho_norm2 = np.dot(rho_c, H0_m.dot(rho_c))
                    title = r'$\rho_h$ at $t = {:5.4f}, norm = {}$'.format(
                        dt * nt, np.sqrt(rho_norm2))
                    plot_field(numpy_coeffs=rho_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title,
                               filename=plot_dir + '/' + params_str +
                               '_rho_nt={}.pdf'.format(nt),
                               plot_type=plot_type, cb_min=None, cb_max=None, hide_plot=hide_plots)
                else:
                    plot_type = 'amplitude'

                divE_c = div_m @ Ep_c
                divE_norm2 = np.dot(divE_c, H0_m.dot(divE_c))
                if project_sol:
                    title = r'div $P^1_h E_h$ at $t = {:5.4f}, norm = {}$'.format(
                        dt * nt, np.sqrt(divE_norm2))
                else:
                    title = r'div $E_h$ at $t = {:5.4f}, norm = {}$'.format(
                        dt * nt, np.sqrt(divE_norm2))
                plot_field(numpy_coeffs=divE_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title,
                           filename=plot_dir + '/' + params_str +
                           '_divEh_nt={}.pdf'.format(nt),
                           plot_type=plot_type, cb_min=None, cb_max=None, hide_plot=hide_plots)

        else:
            print(' -- WARNING: unknown plot_dir !!')

    def plot_B_field(B_c, nt):

        if plot_dir:

            print(' .. plotting B field...')
            params_str = f'gamma_h={gamma_h}_dt={dt}'

            title = r'$B_h$ (amplitude) for $t = {:5.4f}$'.format(dt * nt)
            plot_field(numpy_coeffs=B_c, Vh=V2h, space_kind='l2', domain=domain, surface_plot=False, title=title,
                       filename=plot_dir + '/' + params_str +
                       '_Bh_nt={}.pdf'.format(nt),
                       plot_type='amplitude', cb_min=cb_min_sol, cb_max=cb_max_sol, hide_plot=hide_plots)

        else:
            print(' -- WARNING: unknown plot_dir !!')

    def plot_time_diags(time_diag, E_norm2_diag, B_norm2_diag, divE_norm2_diag, nt_start, nt_end,
                        GaussErr_norm2_diag=None, GaussErrP_norm2_diag=None,
                        PE_norm2_diag=None, I_PE_norm2_diag=None, J_norm2_diag=None, skip_titles=True):

        nt_start = max(nt_start, 0)
        nt_end = min(nt_end, Nt)

        td = time_diag[nt_start:nt_end + 1]
        t_label = r'$t$'

        # norm || E ||
        fig, ax = plt.subplots()
        ax.plot(td,
                np.sqrt(E_norm2_diag[nt_start:nt_end + 1]),
                '-',
                ms=7,
                mfc='None',
                mec='k')  # , label='||E||', zorder=10)
        if skip_titles:
            title = ''
        else:
            title = r'$||E_h(t)||$ vs ' + t_label
        ax.set_xlabel(t_label, fontsize=16)
        ax.set_title(title, fontsize=18)
        fig.tight_layout()
        diag_fn = plot_dir + \
            f'/diag_E_norm_gamma={gamma_h}_dt={dt}_trange=[{dt*nt_start}, {dt*nt_end}].pdf'
        print(f"saving plot for '{title}' in figure '{diag_fn}")
        fig.savefig(diag_fn)

        # energy
        fig, ax = plt.subplots()
        E_energ = .5 * E_norm2_diag[nt_start:nt_end + 1]
        B_energ = .5 * B_norm2_diag[nt_start:nt_end + 1]
        ax.plot(td, E_energ, '-', ms=7, mfc='None', c='k',
                label=r'$\frac{1}{2}||E||^2$')  # , zorder=10)
        ax.plot(td, B_energ, '-', ms=7, mfc='None', c='g',
                label=r'$\frac{1}{2}||B||^2$')  # , zorder=10)
        ax.plot(td, E_energ + B_energ, '-', ms=7, mfc='None', c='b',
                label=r'$\frac{1}{2}(||E||^2+||B||^2)$')  # , zorder=10)
        ax.legend(loc='best')
        if skip_titles:
            title = ''
        else:
            title = r'energy vs ' + t_label
        if E0_type == 'pulse':
            ax.set_ylim([0, 5])
        ax.set_xlabel(t_label, fontsize=16)
        ax.set_title(title, fontsize=18)
        fig.tight_layout()
        diag_fn = plot_dir + \
            f'/diag_energy_gamma={gamma_h}_dt={dt}_trange=[{dt*nt_start},{dt*nt_end}].pdf'
        print(f"saving plot for '{title}' in figure '{diag_fn}")
        fig.savefig(diag_fn)

        # One curve per plot from now on.
        # Collect information in a list where each item is of the form [tag,
        # data, title]
        time_diagnostics = []

        if project_sol:
            time_diagnostics += [['divPE', divE_norm2_diag,
                                  r'$||div_h P^1_h E_h(t)||$ vs ' + t_label]]
        else:
            time_diagnostics += [['divE', divE_norm2_diag,
                                  r'$||div_h E_h(t)||$ vs ' + t_label]]

        time_diagnostics += [
            ['I_PE', I_PE_norm2_diag, r'$||(I-P^1)E_h(t)||$ vs ' + t_label],
            ['PE', PE_norm2_diag, r'$||(I-P^1)E_h(t)||$ vs ' + t_label],
            ['GaussErr', GaussErr_norm2_diag,
                r'$||(\rho_h - div_h E_h)(t)||$ vs ' + t_label],
            ['GaussErrP', GaussErrP_norm2_diag,
                r'$||(\rho_h - div_h E_h)(t)||$ vs ' + t_label],
            ['J_norm', J_norm2_diag, r'$||J_h(t)||$ vs ' + t_label],
        ]

        for tag, data, title in time_diagnostics:
            if data is None:
                continue
            fig, ax = plt.subplots()
            ax.plot(td,
                    np.sqrt(I_PE_norm2_diag[nt_start:nt_end + 1]),
                    '-',
                    ms=7,
                    mfc='None',
                    mec='k')  # , label='||E||', zorder=10)
            diag_fn = plot_dir + \
                f'/diag_{tag}_gamma={gamma_h}_dt={dt}_trange=[{dt*nt_start},{dt*nt_end}].pdf'
            ax.set_xlabel(t_label, fontsize=16)
            if not skip_titles:
                ax.set_title(title, fontsize=18)
            fig.tight_layout()
            print(f"saving plot for '{title}' in figure '{diag_fn}")
            fig.savefig(diag_fn)

    # diags arrays
    E_norm2_diag = np.zeros(Nt + 1)
    B_norm2_diag = np.zeros(Nt + 1)
    divE_norm2_diag = np.zeros(Nt + 1)
    time_diag = np.zeros(Nt + 1)
    PE_norm2_diag = np.zeros(Nt + 1)
    I_PE_norm2_diag = np.zeros(Nt + 1)
    J_norm2_diag = np.zeros(Nt + 1)
    if source_type == 'Il_pulse':
        GaussErr_norm2_diag = np.zeros(Nt + 1)
        GaussErrP_norm2_diag = np.zeros(Nt + 1)
    else:
        GaussErr_norm2_diag = None
        GaussErrP_norm2_diag = None

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # initial solution

    print(' .. initial solution ..')

    # initial B sol
    B_c = np.zeros(V2h.nbasis)

    # initial E sol
    if E0_type == 'th_sol':

        if os.path.exists(th_sol_filename):
            print(
                ' .. loading time-harmonic solution from file {}'.format(th_sol_filename))
            E_c = source_omega * np.load(th_sol_filename)
            assert len(E_c) == V1h.nbasis
        else:
            print(
                ' .. Error: time-harmonic solution file given {}, but not found'.format(th_sol_filename))
            raise ValueError(th_sol_filename)

    elif E0_type == 'zero':
        E_c = np.zeros(V1h.nbasis)

    elif E0_type == 'pulse':

        E0 = get_div_free_pulse(x_0=1.0, y_0=1.0, domain=domain)

        if E0_proj == 'P_geom':
            print(' .. projecting E0 with commuting projection...')
            E0_h = P1_phys(E0, P1, domain, mappings_list)
            E_c = E0_h.coeffs.toarray()

        elif E0_proj == 'P_L2':
            # helper: save/load coefs
            E0dd_filename = m_load_dir + \
                '/E0_pulse_dual_dofs_qp{}.npy'.format(quad_param)
            if os.path.exists(E0dd_filename):
                print(' .. loading E0 dual dofs from file {}'.format(E0dd_filename))
                tilde_E0_c = np.load(E0dd_filename)
            else:
                print(' .. projecting E0 with L2 projection...')
                tilde_E0_c = derham_h.get_dual_dofs(
                    space='V1', f=E0, backend_language=backend, return_format='numpy_array')
                print(' .. saving E0 dual dofs to file {}'.format(E0dd_filename))
                np.save(E0dd_filename, tilde_E0_c)
            E_c = dH1_m.dot(tilde_E0_c)

    elif E0_type == 'pulse_2':
        # E0 = get_praxial_Gaussian_beam_E(x_0=3.14, y_0=3.14, domain=domain)

        # E0 = get_easy_Gaussian_beam_E_2(x_0=0.05, y_0=0.05, domain=domain)
        # B0 = get_easy_Gaussian_beam_B_2(x_0=0.05, y_0=0.05, domain=domain)

        E0, B0 = get_Gaussian_beam(y_0=3.14, x_0=3.14, domain=domain)
        # B0 = get_easy_Gaussian_beam_B(x_0=3.14, y_0=0.05, domain=domain)

        if E0_proj == 'P_geom':
            print(' .. projecting E0 with commuting projection...')

            E0_h = P1_phys(E0, P1, domain, mappings_list)
            E_c = E0_h.coeffs.toarray()

            # B_c = np.real( - 1j * C_m @ E_c)
            # E_c = np.real(E_c)
            B0_h = P2_phys(B0, P2, domain, mappings_list)
            B_c = B0_h.coeffs.toarray()

        elif E0_proj == 'P_L2':
            # helper: save/load coefs
            E0dd_filename = m_load_dir + \
                '/E0_pulse_dual_dofs_qp{}.npy'.format(quad_param)
            if False:  # os.path.exists(E0dd_filename):
                print(' .. loading E0 dual dofs from file {}'.format(E0dd_filename))
                tilde_E0_c = np.load(E0dd_filename)
            else:
                print(' .. projecting E0 with L2 projection...')

                tilde_E0_c = derham_h.get_dual_dofs(
                    space='V1', f=E0, backend_language=backend, return_format='numpy_array')
                print(' .. saving E0 dual dofs to file {}'.format(E0dd_filename))
                # np.save(E0dd_filename, tilde_E0_c)

            E_c = dH1_m.dot(tilde_E0_c)
            dH2_m = H2.get_dual_sparse_matrix()
            tilde_B0_c = derham_h.get_dual_dofs(
                space='V2', f=B0, backend_language=backend, return_format='numpy_array')
            B_c = dH2_m.dot(tilde_B0_c)

            # B_c = np.real( - C_m @ E_c)
            # E_c = np.real(E_c)
    else:
        raise ValueError(E0_type)

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # time loop

    def compute_diags(E_c, B_c, J_c, nt):
        time_diag[nt] = (nt) * dt
        PE_c = cP1_m.dot(E_c)
        I_PE_c = E_c - PE_c
        E_norm2_diag[nt] = np.dot(E_c, H1_m.dot(E_c))
        PE_norm2_diag[nt] = np.dot(PE_c, H1_m.dot(PE_c))
        I_PE_norm2_diag[nt] = np.dot(I_PE_c, H1_m.dot(I_PE_c))
        J_norm2_diag[nt] = np.dot(J_c, H1_m.dot(J_c))
        B_norm2_diag[nt] = np.dot(B_c, H2_m.dot(B_c))
        divE_c = div_m @ E_c
        divE_norm2_diag[nt] = np.dot(divE_c, H0_m.dot(divE_c))
        if source_type == 'Il_pulse':
            rho_c = rho0_c * np.sin(source_omega * nt * dt) / omega
            GaussErr = rho_c - divE_c
            GaussErrP = rho_c - div_m @ PE_c
            GaussErr_norm2_diag[nt] = np.dot(GaussErr, H0_m.dot(GaussErr))
            GaussErrP_norm2_diag[nt] = np.dot(GaussErrP, H0_m.dot(GaussErrP))

    if plot_dir:
        OM1 = OutputManager(plot_dir + '/spaces1.yml', plot_dir + '/fields1.h5')
        OM1.add_spaces(V1h=V1h)
        OM1.export_space_info()

        OM2 = OutputManager(plot_dir + '/spaces2.yml', plot_dir + '/fields2.h5')
        OM2.add_spaces(V2h=V2h)
        OM2.export_space_info()

        stencil_coeffs_E = array_to_psydac(cP1_m @ E_c, V1h.coeff_space)
        Eh = FemField(V1h, coeffs=stencil_coeffs_E)
        OM1.add_snapshot(t=0, ts=0)
        OM1.export_fields(Eh=Eh)

        stencil_coeffs_B = array_to_psydac(B_c, V2h.coeff_space)
        Bh = FemField(V2h, coeffs=stencil_coeffs_B)
        OM2.add_snapshot(t=0, ts=0)
        OM2.export_fields(Bh=Bh)

    # PM = PostProcessManager(domain=domain, space_file=plot_dir+'/spaces1.yml', fields_file=plot_dir+'/fields1.h5' )
    # PM.export_to_vtk(plot_dir+"/Eh",grid=None, npts_per_cell=[6]*2, snapshots='all', fields='vh' )

    # OM1.close()
    # PM.close()

    # plot_E_field(E_c, nt=0, project_sol=project_sol, plot_divE=plot_divE)
    # plot_B_field(B_c, nt=0)

    f_c = np.copy(f0_c)
    for nt in range(Nt):
        print(' .. nt+1 = {}/{}'.format(nt + 1, Nt))

        # 1/2 faraday: Bn -> Bn+1/2
        B_c[:] -= (dt / 2) * C_m @ E_c

        # ampere: En -> En+1
        if f0_harmonic_c is not None:
            f_harmonic_c = f0_harmonic_c * (np.sin(source_omega * (nt + 1) * dt) - np.sin(
                source_omega * (nt) * dt)) / (dt * source_omega)  # * source_enveloppe(omega*(nt+1/2)*dt)
            f_c[:] = f0_c + f_harmonic_c

        if nt == 0:
            if plot_dir:
                plot_J_source_nPlusHalf(f_c, nt=0)
            compute_diags(E_c, B_c, f_c, nt=0)

        E_c[:] = dCH1_m @ E_c + dt * (dC_m @ B_c - f_c)

        # if abs(gamma_h) > 1e-10:
        #    E_c[:] -= dt * gamma_h * JP_m @ E_c

        # 1/2 faraday: Bn+1/2 -> Bn+1
        B_c[:] -= (dt / 2) * C_m @ E_c

        # diags:
        compute_diags(E_c, B_c, f_c, nt=nt + 1)

        # PE_c = cP1_m.dot(E_c)
        # I_PE_c = E_c-PE_c
        # E_norm2_diag[nt+1] = np.dot(E_c,H1_m.dot(E_c))
        # PE_norm2_diag[nt+1] = np.dot(PE_c,H1_m.dot(PE_c))
        # I_PE_norm2_diag[nt+1] = np.dot(I_PE_c,H1_m.dot(I_PE_c))
        # B_norm2_diag[nt+1] = np.dot(B_c,H2_m.dot(B_c))
        # time_diag[nt+1] = (nt+1)*dt

        # diags: div
        # if project_sol:
        #     Ep_c = PE_c # = cP1_m.dot(E_c)
        # else:
        #     Ep_c = E_c
        # divE_c = div_m @ Ep_c
        # divE_norm2 = np.dot(divE_c, H0_m.dot(divE_c))
        # # print('in diag[{}]: divE_norm = {}'.format(nt+1, np.sqrt(divE_norm2)))
        # divE_norm2_diag[nt+1] = divE_norm2

        # if source_type == 'Il_pulse':
        #     rho_c = rho0_c * np.sin(omega*dt*(nt+1))/omega
        #     GaussErr = rho_c - div_m @ E_c
        #     GaussErrP = rho_c - div_m @ (cP1_m.dot(E_c))
        #     GaussErr_norm2_diag[nt+1] = np.dot(GaussErr, H0_m.dot(GaussErr))
        #     GaussErrP_norm2_diag[nt+1] = np.dot(GaussErrP, H0_m.dot(GaussErrP))

        if debug:
            divCB_c = div_m @ dC_m @ B_c
            divCB_norm2 = np.dot(divCB_c, H0_m.dot(divCB_c))
            print('-- [{}]: dt*|| div CB || = {}'.format(nt +
                  1, dt * np.sqrt(divCB_norm2)))

            divf_c = div_m @ f_c
            divf_norm2 = np.dot(divf_c, H0_m.dot(divf_c))
            print('-- [{}]: dt*|| div f || = {}'.format(nt +
                  1, dt * np.sqrt(divf_norm2)))

            divE_c = div_m @ E_c
            divE_norm2 = np.dot(divE_c, H0_m.dot(divE_c))
            print('-- [{}]: || div E || = {}'.format(nt + 1, np.sqrt(divE_norm2)))

        if is_plotting_time(nt + 1) and plot_dir:
            print("Plot Stuff")
            # plot_E_field(E_c, nt=nt+1, project_sol=True, plot_divE=False)
            # plot_B_field(B_c, nt=nt+1)
            # plot_J_source_nPlusHalf(f_c, nt=nt)

            stencil_coeffs_E = array_to_psydac(cP1_m @ E_c, V1h.coeff_space)
            Eh = FemField(V1h, coeffs=stencil_coeffs_E)
            OM1.add_snapshot(t=nt * dt, ts=nt)
            OM1.export_fields(Eh=Eh)

            stencil_coeffs_B = array_to_psydac(B_c, V2h.coeff_space)
            Bh = FemField(V2h, coeffs=stencil_coeffs_B)
            OM2.add_snapshot(t=nt * dt, ts=nt)
            OM2.export_fields(Bh=Bh)

        # if (nt+1) % diag_nt == 0:
            # plot_time_diags(time_diag, E_norm2_diag, B_norm2_diag, divE_norm2_diag, nt_start=(nt+1)-diag_nt, nt_end=(nt+1),
            # PE_norm2_diag=PE_norm2_diag, I_PE_norm2_diag=I_PE_norm2_diag, J_norm2_diag=J_norm2_diag,
            # GaussErr_norm2_diag=GaussErr_norm2_diag,
            # GaussErrP_norm2_diag=GaussErrP_norm2_diag)
    if plot_dir:
        OM1.close()

        print("Do some PP")
        PM = PostProcessManager(
            domain=domain,
            space_file=plot_dir +
            '/spaces1.yml',
            fields_file=plot_dir +
            '/fields1.h5')
        PM.export_to_vtk(
            plot_dir + "/Eh",
            grid=None,
            npts_per_cell=2,
            snapshots='all',
            fields='Eh')
        PM.close()

        PM = PostProcessManager(
            domain=domain,
            space_file=plot_dir +
            '/spaces2.yml',
            fields_file=plot_dir +
            '/fields2.h5')
        PM.export_to_vtk(
            plot_dir + "/Bh",
            grid=None,
            npts_per_cell=2,
            snapshots='all',
            fields='Bh')
        PM.close()

   # plot_time_diags(time_diag, E_norm2_diag, B_norm2_diag, divE_norm2_diag, nt_start=0, nt_end=Nt,
   # PE_norm2_diag=PE_norm2_diag, I_PE_norm2_diag=I_PE_norm2_diag, J_norm2_diag=J_norm2_diag,
   # GaussErr_norm2_diag=GaussErr_norm2_diag,
   # GaussErrP_norm2_diag=GaussErrP_norm2_diag)

    # Eh = FemField(V1h, coeffs=array_to_stencil(E_c, V1h.coeff_space))
    # t_stamp = time_count(t_stamp)

    # if sol_filename:
    #     raise NotImplementedError
    # print(' .. saving final solution coeffs to file {}'.format(sol_filename))
    # np.save(sol_filename, E_c)

    # time_count(t_stamp)

    # print()
    # print(' -- plots and diagnostics  --')

    # # diagnostics: errors
    # err_diags = diag_grid.get_diags_for(v=uh, space='V1')
    # for key, value in err_diags.items():
    #     diags[key] = value

    # if u_ex is not None:
    #     check_diags = get_Vh_diags_for(v=uh, v_ref=uh_ref, M_m=H1_m, msg='error between Ph(u_ex) and u_h')
    #     diags['norm_Pu_ex'] = check_diags['sol_ref_norm']
    #     diags['rel_l2_error_in_Vh'] = check_diags['rel_l2_error']

    # if curl_u_ex is not None:
    #     print(' .. diag on curl_u:')
    #     curl_uh_c = bD1_m @ cP1_m @ uh_c
    #     title = r'curl $u_h$ (amplitude) for $\eta = $'+repr(eta)
    #     params_str = 'eta={}_mu={}_nu={}_gamma_h={}_Pf={}'.format(eta, mu, nu, gamma_h, source_proj)
    #     plot_field(numpy_coeffs=curl_uh_c, Vh=V2h, space_kind='l2', domain=domain, surface_plot=False, title=title, filename=plot_dir+'/'+params_str+'_curl_uh.png',
    # plot_type='amplitude', cb_min=None, cb_max=None, hide_plot=hide_plots)

    #     curl_uh = FemField(V2h, coeffs=array_to_stencil(curl_uh_c, V2h.coeff_space))
    #     curl_diags = diag_grid.get_diags_for(v=curl_uh, space='V2')
    #     diags['curl_error (to be checked)'] = curl_diags['rel_l2_error']

    #     title = r'div_h $u_h$ (amplitude) for $\eta = $'+repr(eta)
    #     params_str = 'eta={}_mu={}_nu={}_gamma_h={}_Pf={}'.format(eta, mu, nu, gamma_h, source_proj)
    #     plot_field(numpy_coeffs=div_uh_c, Vh=V0h, space_kind='h1', domain=domain, surface_plot=False, title=title, filename=plot_dir+'/'+params_str+'_div_uh.png',
    # plot_type='amplitude', cb_min=None, cb_max=None, hide_plot=hide_plots)

    #     div_uh = FemField(V0h, coeffs=array_to_stencil(div_uh_c, V0h.coeff_space))
    #     div_diags = diag_grid.get_diags_for(v=div_uh, space='V0')
    #     diags['div_error (to be checked)'] = div_diags['rel_l2_error']

    return diags


# def compute_stable_dt(cfl_max, dt_max, C_m, dC_m, V1_dim):
def compute_stable_dt(*, C_m, dC_m, cfl_max, dt_max=None):
    """
    Compute a stable time step size based on the maximum CFL parameter in the
    domain. To this end we estimate the operator norm of

    `dC_m @ C_m: V1h -> V1h`,

    find the largest stable time step compatible with Strang splitting, and
    rescale it by the provided `cfl_max`. Setting `cfl_max = 1` would run the
    scheme exactly at its stability limit, which is not safe because of the
    unavoidable round-off errors. Hence we require `0 < cfl_max < 1`.

    Optionally the user can provide a maximum time step size in order to
    properly resolve some time scales of interest (e.g. a time-dependent
    current source).

    Parameters
    ----------
    C_m : scipy.sparse.spmatrix
        Matrix of the Curl operator.

    dC_m : scipy.sparse.spmatrix
        Matrix of the dual Curl operator.

    cfl_max : float
        Maximum Courant parameter in the domain, intended as a stability
        parameter (=1 at the stability limit). Must be `0 < cfl_max < 1`.

    dt_max : float, optional
        If not None, restrict the computed dt by this value in order to
        properly resolve time scales of interest. Must be > 0.

    Returns
    -------
    dt : float
        Largest stable dt which satisfies the provided constraints.

    """

    print(" .. compute_stable_dt by estimating the operator norm of ")
    print(" ..     dC_m @ C_m: V1h -> V1h ")
    print(" ..     with dim(V1h) = {}      ...".format(C_m.shape[1]))

    if not (0 < cfl_max < 1):
        print(' ******  ****** ******  ****** ******  ****** ')
        print('         WARNING !!!  cfl = {}  '.format(cfl))
        print(' ******  ****** ******  ****** ******  ****** ')

    def vect_norm_2(vv):
        return np.sqrt(np.dot(vv, vv))

    t_stamp = time_count()
    vv = np.random.random(C_m.shape[1])
    norm_vv = vect_norm_2(vv)
    max_ncfl = 500
    ncfl = 0
    spectral_rho = 1
    conv = False
    CC_m = dC_m @ C_m

    while not (conv or ncfl > max_ncfl):

        vv[:] = (1. / norm_vv) * vv
        ncfl += 1
        vv[:] = CC_m.dot(vv)

        norm_vv = vect_norm_2(vv)
        old_spectral_rho = spectral_rho
        spectral_rho = vect_norm_2(vv)  # approximation
        conv = abs((spectral_rho - old_spectral_rho) / spectral_rho) < 0.001
        print("    ... spectral radius iteration: spectral_rho( dC_m @ C_m ) ~= {}".format(spectral_rho))
    t_stamp = time_count(t_stamp)

    norm_op = np.sqrt(spectral_rho)
    c_dt_max = 2. / norm_op

    light_c = 1
    dt = cfl_max * c_dt_max / light_c

    if dt_max is not None:
        dt = min(dt, dt_max)

    print("  Time step dt computed for Maxwell solver:")
    print(
        f"     Based on cfl_max = {cfl_max} and dt_max = {dt_max}, we set dt = {dt}")
    print(
        f"     -- note that c*Dt = {light_c*dt} and c_dt_max = {c_dt_max}, thus c * dt / c_dt_max = {light_c*dt/c_dt_max}")
    print(
        f"     -- and spectral_radius((c*dt)**2* dC_m @ C_m ) = {(light_c * dt * norm_op)**2} (should be < 4).")

    return dt
