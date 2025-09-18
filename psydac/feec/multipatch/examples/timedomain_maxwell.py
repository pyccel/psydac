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
from psydac.linalg.basic import IdentityOperator

from psydac.api.settings import PSYDAC_BACKENDS
from psydac.api.discretization import discretize

from psydac.fem.plotting_utilities import plot_field_2d as plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

from psydac.feec.multipatch.examples.ppc_test_cases import get_source_and_solution_hcurl, get_div_free_pulse, get_curl_free_pulse, get_Delta_phi_pulse, get_Gaussian_beam
from psydac.feec.multipatch.utils_conga_2d import DiagGrid, P0_phys, P1_phys, P2_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities import time_count 
from psydac.fem.basic import FemField
from psydac.feec.multipatch.multipatch_domain_utilities import build_cartesian_multipatch_domain

from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.fem.projectors import get_dual_dofs


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
                         source_proj='P_L2',
                         project_sol=False,
                         filter_source=True,
                         E0_type='pulse_2',
                         E0_proj='P_L2',
                         plot_dir=None,
                         plot_time_ranges=None,
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

    project_sol : bool
        Whether the solution fields should be projected onto the corresponding
        conforming spaces before plotting them.

    filter_source : bool
        If True, the current source will be filtered with the conforming
        projector operator (or its dual, depending on which basis is used).

    E0_type : str {'zero', 'pulse'}
        Initial conditions for the electric field. Choose 'zero' for E0=0
        and 'pulse' for a non-zero field localized in a small region.

    E0_proj : str {'P_geom' | 'P_L2'}
        Name of the approximation operator for the initial electric field E0
        (see source_proj for details). Only relevant if E0 is not zero.

    plot_dir : str
        Path to the directory where the figures will be saved.

    plot_time_ranges : list
        List of lists, of the form `[[start, end], dtp]`, where `[start, end]`
        is a time interval and `dtp` is the time between two successive plots.

    domain_lims : list
        If the domain_name is 'refined_square' or 'square_L_shape', this
        parameter must be set to the list of the two intervals defining the
        rectangular domain, i.e. `[[x_min, x_max], [y_min, y_max]]`.

    """
    degree = [deg, deg]

    if source_omega is not None:
        period_time = 2 * np.pi / source_omega
        Nt_pp = period_time // dt_max

    if plot_time_ranges is None:
        plot_time_ranges = [
            [[0, final_time], final_time]
        ]


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
    print('---------------------------------------------------------------------------------------------------------')


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
    elif nc.ndim == 1:
        ncells = {patch.name: [nc[i], nc[i]]
                    for (i, patch) in enumerate(domain.interior)}
    elif nc.ndim == 2:
        ncells = {patch.name: [nc[int(patch.name[2])][int(patch.name[4])], 
                nc[int(patch.name[2])][int(patch.name[4])]] for patch in domain.interior}

    mappings = OrderedDict([(P.logical_domain, P.mapping)
                           for P in domain.interior])
    mappings_list = list(mappings.values())


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
    V0h, V1h, V2h = derham_h.spaces

    t_stamp = time_count(t_stamp)
    print(' .. Id operator and matrix...')
    I1 = IdentityOperator(V1h.coeff_space)

    t_stamp = time_count(t_stamp)
    print(' .. Hodge operators...')
    H0, H1, H2 = derham_h.hodge_operators(kind='linop')
    dH0, dH1, dH2 = derham_h.hodge_operators(kind='linop', dual=True)


    t_stamp = time_count(t_stamp)
    print(' .. conforming Projection operators...')
    cP0, cP1, cP2 = derham_h.conforming_projectors(kind='linop', p_moments = degree[0]+2, hom_bc = False)

    t_stamp = time_count(t_stamp)
    print(' .. broken differential operators...')
    bD0, bD1 = derham_h.derivatives(kind='linop')


    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    print(' .. matrix of the primal curl (in primal bases)...')
    C = bD1 @ cP1
    print(' .. matrix of the dual curl (also in primal bases)...')
    dC = dH1 @ C.T @ H2


    ### Silvermueller ABC
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
    A_eps = ah.assemble()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute stable time step size based on max CFL and max dt
    dt = compute_stable_dt(C=C, dC=dC, cfl_max=cfl_max, dt_max=dt_max)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Absorbing dC
    CH2 = C.T @ H2
    H1A = H1 + dt * A_eps

    # alternative inverse
    # from psydac.linalg.solvers     import inverse
    # H1A_inv = inverse(H1A, solver='cg', tol=1e-8)
    ###    
    M = H1A
    from scipy.linalg import inv
    from scipy.sparse import csr_matrix
    from psydac.linalg.sparse import SparseMatrixLinearOperator
    M_inv = inv(M.toarray())
    M_inv = csr_matrix(M_inv)
    H1A_inv = SparseMatrixLinearOperator(M.codomain, M.domain, M_inv)
    ####

    dC   = H1A_inv @ CH2 
    dCH1 = H1A_inv @ H1 

    print(' .. matrix of the dual div (still in primal bases)...')
    D = dH0 @ cP0.T @ bD0.T @ H1


    print(" Reduce time step to match the simulation final time:")
    Nt = int(np.ceil(final_time / dt))
    dt = final_time / Nt
    print(f"   . Time step size  : dt = {dt}")
    print(f"   . Nb of time steps: Nt = {Nt}")

    # ...
    def is_plotting_time(nt, *, dt=dt, Nt=Nt, plot_time_ranges=plot_time_ranges):
        if nt in [0, Nt]:
            return True
        for [start, end], dt_plots in plot_time_ranges:
            # number of time steps between two successive plots
            ds = max(dt_plots // dt, 1)
            if (start <= nt * dt <= end) and (nt % ds == 0):
                return True
        return False
    # ...


    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' total nb of time steps: Nt = {}, final time: T = {:5.4f}'.format(Nt, final_time))
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')
    print(' ------ ------ ------ ------ ------ ------ ------ ------ ')

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # source

    t_stamp = time_count(t_stamp)
    print()
    print(' -- getting source --')
    f0_h = None
    f0_harmonic_h = None
    rho0_h = None
    
    if source_type == 'zero':

        f0 = None
        f0_harmonic = None

    elif source_type == 'pulse':

        f0 = get_div_free_pulse(x_0=np.pi/2, y_0=np.pi/2, domain=domain)

    elif source_type == 'cf_pulse':

        f0 = get_curl_free_pulse(x_0=np.pi/2, y_0=np.pi/2, domain=domain)

    elif source_type == 'Il_pulse':  # Issautier-like pulse
        # source will be
        #   J = curl A + cos(om*t) * grad phi
        # so that
        #   dt rho = - div J = - cos(om*t) Delta phi
        # for instance, with rho(t=0) = 0 this  gives
        #   rho = - sin(om*t)/om * Delta phi
        # and Gauss' law reads
        #  div E = rho = - sin(om*t)/om * Delta phi
        f0 = get_div_free_pulse(x_0=np.pi/2, y_0=np.pi/2, domain=domain)  # this is curl A
        f0_harmonic = get_curl_free_pulse( x_0=np.pi/2, y_0=np.pi/2, domain=domain)  # this is grad phi

        rho0 = get_Delta_phi_pulse(x_0=np.pi/2, y_0=np.pi/2, domain=domain)  # this is Delta phi
        tilde_rho0_h = get_dual_dofs(Vh=V0h, f=rho0, domain_h=domain_h, backend_language=backend)
        tilde_rho0_h = cP0.T @ tilde_rho0_h
        rho0_h = dH0.dot(tilde_rho0_h)
    else:

        f0, u_bc, u_ex, curl_u_ex, div_u_ex = get_source_and_solution_hcurl(source_type=source_type, domain=domain, domain_name=domain_name)
        assert u_bc is None  # only homogeneous BC's for now


    if source_omega is not None:
        f0_harmonic = f0
        f0 = None

        def source_enveloppe(tau):
            return 1

    t_stamp = time_count(t_stamp)
    tilde_f0_h = f0_h = None
    tilde_f0_harmonic_h = f0_harmonic_h = None

    if source_proj == 'P_geom':
        print(' .. projecting the source with commuting projection...')

        if f0 is not None:
            f0_h = P1_phys(f0, P1, domain).coeffs
            tilde_f0_h = H1.dot(f0_h)

        if f0_harmonic is not None:
            f0_harmonic_h = P1_phys(f0_harmonic, P1, domain).coeffs
            tilde_f0_harmonic_h = H1.dot(f0_harmonic_h)

    elif source_proj == 'P_L2':

        if f0 is not None:
            if source_type == 'Il_pulse':
                source_name = 'Il_pulse_f0'
            else:
                source_name = source_type

            print(' .. projecting the source f0 with L2 projection...')
            tilde_f0_h = get_dual_dofs(Vh=V1h, f=f0, domain_h=domain_h, backend_language=backend)

        if f0_harmonic is not None:
            if source_type == 'Il_pulse':
                source_name = 'Il_pulse_f0_harmonic'
            else:
                source_name = source_type

            print(' .. projecting the source f0_harmonic with L2 projection...')
            tilde_f0_harmonic_h = get_dual_dofs(Vh=V1h, f=f0_harmonic, domain_h=domain_h, backend_language=backend)

    else:
        raise ValueError(source_proj)

    t_stamp = time_count(t_stamp)
    if filter_source:
        print(' .. filtering the source...')
        if tilde_f0_h is not None:
            tilde_f0_h = cP1.T @ tilde_f0_h

        if tilde_f0_harmonic_h is not None:
            tilde_f0_harmonic_h = cP1.T @ tilde_f0_harmonic_h

    if tilde_f0_h is not None:
        f0_h = dH1.dot(tilde_f0_h)

    if tilde_f0_harmonic_h is not None:
        f0_harmonic_h = dH1.dot(tilde_f0_harmonic_h)


    if f0_h is None:
        f0_h = V1h.coeff_space.zeros()

    t_stamp = time_count(t_stamp)

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
    B_h = V2h.coeff_space.zeros()
    E_h = V1h.coeff_space.zeros()

    # initial E sol
    if E0_type == 'zero':
        E_h = V1h.coeff_space.zeros()

    elif E0_type == 'pulse':

        E0 = get_div_free_pulse(x_0=np.pi/2, y_0=np.pi/2, domain=domain)

        if E0_proj == 'P_geom':
            print(' .. projecting E0 with commuting projection...')
            E0_h = P1_phys(E0, P1, domain)
            E_h = E0_h.coeffs

        elif E0_proj == 'P_L2':

            print(' .. projecting E0 with L2 projection...')
            tilde_E0_h = get_dual_dofs(Vh=V1h, f=E0, domain_h=domain_h, backend_language=backend)
            E_h = dH1.dot(tilde_E0_h)

    elif E0_type == 'pulse_2':

        E0, B0 = get_Gaussian_beam(y_0=np.pi/2, x_0=np.pi/2, domain=domain)

        if E0_proj == 'P_geom':
            print(' .. projecting E0 with commuting projection...')

            E0_h = P1_phys(E0, P1, domain)
            E_h = E0_h.coeffs

            B0_h = P2_phys(B0, P2, domain)
            B_h = B0_h.coeffs

        elif E0_proj == 'P_L2':
           
            print(' .. projecting E0 with L2 projection...')
            tilde_E0_h = get_dual_dofs(Vh=V1h, f=E0, domain_h=domain_h, backend_language=backend)
            E_h = dH1.dot(tilde_E0_h)

            tilde_B0_h = get_dual_dofs(Vh=V2h, f=B0, domain_h=domain_h, backend_language=backend)
            B_h = dH2.dot(tilde_B0_h)

    else:
        raise ValueError(E0_type)

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # time loop

    def compute_diags(E_h, B_h, J_h, nt):
        time_diag[nt] = (nt) * dt
        PE_h = cP1.dot(E_h)
        I_PE_h = E_h - PE_h
        E_norm2_diag[nt] = E_h.inner(H1.dot(E_h))
        PE_norm2_diag[nt] = PE_h.inner(H1.dot(PE_h))
        I_PE_norm2_diag[nt] = I_PE_h.inner(H1.dot(I_PE_h))
        J_norm2_diag[nt] = J_h.inner(H1.dot(J_h))
        B_norm2_diag[nt] = B_h.inner(H2.dot(B_h))
        divE_h = D @ E_h
        divE_norm2_diag[nt] = divE_h.inner(H0.dot(divE_h))
        if source_type == 'Il_pulse' and source_omega is not None:
            rho_h = rho0_h * np.sin(source_omega * nt * dt) / omega
            GaussErr = rho_h - divE_h
            GaussErrP = rho_h - D @ PE_h
            GaussErr_norm2_diag[nt] = GaussErr.inner(H0.dot(GaussErr))
            GaussErrP_norm2_diag[nt] = GaussErrP.inner(H0.dot(GaussErrP))

    if plot_dir:
        OM1 = OutputManager(plot_dir + '/spaces1.yml', plot_dir + '/fields1.h5')
        OM1.add_spaces(V1h=V1h)
        OM1.export_space_info()

        OM2 = OutputManager(plot_dir + '/spaces2.yml', plot_dir + '/fields2.h5')
        OM2.add_spaces(V2h=V2h)
        OM2.export_space_info()

        Eh = FemField(V1h, coeffs=cP1 @ E_h)
        OM1.add_snapshot(t=0, ts=0)
        OM1.export_fields(Eh=Eh)

        Bh = FemField(V2h, coeffs=B_h)
        OM2.add_snapshot(t=0, ts=0)
        OM2.export_fields(Bh=Bh)


    f_h = f0_h.copy()
    for nt in range(Nt):
        print(' .. nt+1 = {}/{}'.format(nt + 1, Nt))

        # 1/2 faraday: Bn -> Bn+1/2
        B_h -= (dt / 2) * C @ E_h

        # ampere: En -> En+1
        if f0_harmonic_h is not None and source_omega is not None:
            f_harmonic_h = f0_harmonic_h * (np.sin(source_omega * (nt + 1) * dt) - np.sin(source_omega * (nt) * dt)) / (dt * source_omega)  # * source_enveloppe(omega*(nt+1/2)*dt)
            f_h = f0_h + f_harmonic_h

        E_h = dCH1 @ E_h + dt * (dC @ B_h - f_h)

        # 1/2 faraday: Bn+1/2 -> Bn+1
        B_h -= (dt / 2) * C @ E_h

        # diags:
        compute_diags(E_h, B_h, f_h, nt=nt + 1)



        if is_plotting_time(nt + 1) and plot_dir:
            print("Plot fields")

            Eh = FemField(V1h, coeffs=cP1 @ E_h)
            OM1.add_snapshot(t=nt * dt, ts=nt)
            OM1.export_fields(Eh=Eh)

            Bh = FemField(V2h, coeffs=B_h)
            OM2.add_snapshot(t=nt * dt, ts=nt)
            OM2.export_fields(Bh=Bh)


    if plot_dir:
        OM1.close()

        print("Post process fields")
        PM = PostProcessManager(
            domain=domain,
            space_file=plot_dir + '/spaces1.yml',
            fields_file=plot_dir + '/fields1.h5')
        PM.export_to_vtk(
            plot_dir + "/Eh",
            grid=None,
            npts_per_cell=4,
            snapshots='all',
            fields='Eh')
        PM.close()

        PM = PostProcessManager(
            domain=domain,
            space_file=plot_dir + '/spaces2.yml',
            fields_file=plot_dir + '/fields2.h5')
        PM.export_to_vtk(
            plot_dir + "/Bh",
            grid=None,
            npts_per_cell=4,
            snapshots='all',
            fields='Bh')
        PM.close()



def compute_stable_dt(*, C, dC, cfl_max, dt_max=None):
    """
    Compute a stable time step size based on the maximum CFL parameter in the
    domain. To this end we estimate the operator norm of

    `dC @ C: V1h -> V1h`,

    find the largest stable time step compatible with Strang splitting, and
    rescale it by the provided `cfl_max`. Setting `cfl_max = 1` would run the
    scheme exactly at its stability limit, which is not safe because of the
    unavoidable round-off errors. Hence we require `0 < cfl_max < 1`.

    Optionally the user can provide a maximum time step size in order to
    properly resolve some time scales of interest (e.g. a time-dependent
    current source).

    Parameters
    ----------
    C : LinearOperator
        Matrix of the Curl operator.

    dC : LinearOperator
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
    print(" ..     with dim(V1h) = {}      ...".format(C.domain.dimension))

    if not (0 < cfl_max < 1):
        print(' ******  ****** ******  ****** ******  ****** ')
        print('         WARNING !!!  cfl = {}  '.format(cfl))
        print(' ******  ****** ******  ****** ******  ****** ')

    t_stamp = time_count()
    V = C.domain
    from psydac.linalg.utilities import array_to_psydac
    vv = array_to_psydac(np.random.rand(V.dimension), V)

    norm_vv = np.sqrt(vv.inner(vv))

    max_ncfl = 500
    ncfl = 0
    spectral_rho = 1
    conv = False
    CC = dC @ C

    while not (conv or ncfl > max_ncfl):

        vv *= (1. / norm_vv)
        ncfl += 1
        CC.dot(vv, out=vv)

        norm_vv = np.sqrt(vv.inner(vv))
        old_spectral_rho = spectral_rho
        spectral_rho = norm_vv  # approximation
        conv = abs((spectral_rho - old_spectral_rho) / spectral_rho) < 0.001
        print("    ... spectral radius iteration: spectral_rho( dC @ C ) ~= {}".format(spectral_rho))
    t_stamp = time_count(t_stamp)

    norm_op = np.sqrt(spectral_rho)
    c_dt_max = 2. / norm_op

    light_c = 1
    dt = cfl_max * c_dt_max / light_c

    if dt_max is not None:
        dt = min(dt, dt_max)

    print("  Time step dt computed for Maxwell solver:")
    print(f"     Based on cfl_max = {cfl_max} and dt_max = {dt_max}, we set dt = {dt}")
    print(f"     -- note that c*Dt = {light_c*dt} and c_dt_max = {c_dt_max}, thus c * dt / c_dt_max = {light_c*dt/c_dt_max}")
    print(f"     -- and spectral_radius((c*dt)**2* dC @ C ) = {(light_c * dt * norm_op)**2} (should be < 4).")

    return dt
