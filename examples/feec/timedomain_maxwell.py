#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
    solver for the TD Maxwell problem: find E(t) in H(curl), B in L2, such that

      dt E - curl B = -J             on \\Omega
      dt B + curl E = 0              on \\Omega
      n x E = n x E_bc      on \\partial \\Omega

    with Ampere discretized weakly and Faraday discretized strongly, in a broken-FEEC approach on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h
                     (Eh)          (Bh)
"""
import os
import numpy as np

from sympde.calculus        import grad, dot, curl, cross
from sympde.topology        import NormalVector
from sympde.topology        import elements_of
from sympde.topology        import Derham
from sympde.expr.expr       import integral
from sympde.expr.expr       import BilinearForm

from psydac.linalg.basic    import IdentityOperator

from psydac.api.settings       import PSYDAC_BACKENDS
from psydac.api.discretization import discretize
from psydac.api.postprocessing import OutputManager, PostProcessManager

from psydac.feec.multipatch_domain_utilities    import build_multipatch_domain, build_cartesian_multipatch_domain

from psydac.fem.basic       import FemField
from psydac.fem.projectors  import get_dual_dofs

#==============================================================================
# Solver for the TD Maxwell problem
#==============================================================================
def solve_td_maxwell_pbm(*,
                         nc=4,
                         deg=4,
                         final_time=20,
                         cfl_max=0.8,
                         dt_max=None,
                         domain_name='pretzel_f',
                         backend='pyccel-gcc',
                         source_type='zero',
                         E0_type='pulse_2',
                         plot_dir=None,
                         domain_lims=None,
                         p_moments=-1,
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

    source_type : str {'zero' | 'pulse' | 'cf_pulse' }
        Name that identifies the space-time profile of the current source, to be
        chosen among those available in the function get_source_and_solution().
        Available options:
            - 'zero'    : no current source
            - 'pulse'   : div-free current source, time-harmonic
            - 'cf_pulse': curl-free current source, time-harmonic

    E0_type : str {'zero', 'pulse'}
        Initial conditions for the electric field. Choose 'zero' for E0=0
        and 'pulse' for a non-zero field localized in a small region.

    plot_dir : str
        Path to the directory where the figures will be saved.

    domain_lims : list
        If the domain_name is 'refined_square' or 'square_L_shape', this
        parameter must be set to the list of the two intervals defining the
        rectangular domain, i.e. `[[x_min, x_max], [y_min, y_max]]`.
    
    p_moments : int
        Degree of the polynomial moments used in the conforming projection.
    """
    degree = [deg, deg]


    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_td_maxwell_pbm function with: ')
    print(' ncells = {}'.format(nc))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' backend = {}'.format(backend))
    print('---------------------------------------------------------------------------------------------------------')


    print()
    print(' -- building discrete spaces and operators  --')

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


    print(' .. derham sequence...')
    derham = Derham(domain, ["H1", "Hcurl", "L2"])

    print(' .. discrete domain...')
    domain_h = discretize(domain, ncells=ncells)

    print(' .. discrete derham sequence...')
    derham_h = discretize(derham, domain_h, degree=degree)

    print(' .. commuting projection operators...')
    nquads = [4 * (d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    print(' .. multi-patch spaces...')
    V0h, V1h, V2h = derham_h.spaces

    print(' .. Id operator and matrix...')
    I1 = IdentityOperator(V1h.coeff_space)

    print(' .. Hodge operators...')
    H0, H1, H2 = derham_h.hodge_operators(kind='linop')
    dH0, dH1, dH2 = derham_h.hodge_operators(kind='linop', dual=True)

    print(' .. conforming Projection operators...')
    cP0, cP1, cP2 = derham_h.conforming_projectors(kind='linop', p_moments = p_moments, hom_bc = False)

    print(' .. broken differential operators...')
    bD0, bD1 = derham_h.derivatives(kind='linop')

    print(' .. matrix of the primal curl (in primal bases)...')
    C = bD1 @ cP1

    print(' .. matrix of the dual curl (also in primal bases)...')
    dC = dH1 @ C.T @ H2

    ### Silvermueller ABC
    u, v = elements_of(derham.V1, names='u, v')
    nn = NormalVector('nn')
    boundary = domain.boundary
    expr_b = cross(nn, u) * cross(nn, v)

    a = BilinearForm((u, v), integral(boundary, expr_b))
    ah = discretize(a, domain_h, [V1h, V1h], backend=PSYDAC_BACKENDS[backend],)
    A_eps = ah.assemble()

    # Compute stable time step size based on max CFL and max dt
    dt = compute_stable_dt(C=C, dC=dC, cfl_max=cfl_max, dt_max=dt_max)
    print(" Reduce time step to match the simulation final time:")
    Nt = int(np.ceil(final_time / dt))
    dt = final_time / Nt
    print(f"   . Time step size  : dt = {dt}")
    print(' total nb of time steps: Nt = {}, final time: T = {:5.4f}'.format(Nt, final_time))

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

    # Absorbing dC
    dC   = H1A_inv @ C.T @ H2 
    dCH1 = H1A_inv @ H1 


    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # source
    print()
    print(' -- getting source --')
    
    if source_type == 'zero':

        f0 = None
        f0_harmonic = None

    elif source_type == 'pulse':

        f0 = get_div_free_pulse(x_0=np.pi/2, y_0=np.pi/2, domain=domain)

    elif source_type == 'cf_pulse':

        f0 = get_curl_free_pulse(x_0=np.pi/2, y_0=np.pi/2, domain=domain)

    else:

        raise ValueError(source_type)


    if f0 is not None:
        print(' .. projecting the source f0 with L2 projection...')
        tilde_f0_h = get_dual_dofs(Vh=V1h, f=f0, domain_h=domain_h, backend_language=backend)

        print(' .. filtering the source...')
        tilde_f0_h = cP1.T @ tilde_f0_h

        f0_h = dH1.dot(tilde_f0_h)

    else:

        f0_h = V1h.coeff_space.zeros()

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # initial solution

    print(' -- initial solution --')

    # initial B sol
    B_h = V2h.coeff_space.zeros()
    E_h = V1h.coeff_space.zeros()

    # initial E sol
    if E0_type == 'zero':
        E_h = V1h.coeff_space.zeros()

    elif E0_type == 'pulse':

        E0 = get_div_free_pulse(x_0=np.pi/2, y_0=np.pi/2, domain=domain)

        print(' .. projecting E0 with L2 projection...')
        tilde_E0_h = get_dual_dofs(Vh=V1h, f=E0, domain_h=domain_h, backend_language=backend)
        E_h = dH1.dot(tilde_E0_h)

    elif E0_type == 'pulse_2':

        E0, B0 = get_Gaussian_beam(y_0=np.pi/2, x_0=np.pi/2, domain=domain)

        print(' .. projecting E0 with L2 projection...')
        tilde_E0_h = get_dual_dofs(Vh=V1h, f=E0, domain_h=domain_h, backend_language=backend)
        E_h = dH1.dot(tilde_E0_h)

        tilde_B0_h = get_dual_dofs(Vh=V2h, f=B0, domain_h=domain_h, backend_language=backend)
        B_h = dH2.dot(tilde_B0_h)

    elif E0_type == 'Gaussian':
        
        E0, B0 = get_Gaussian_beam(y_0=np.pi/2, x_0=np.pi/2, domain=domain)

        print(' .. projecting E0 with L2 projection...')
        tilde_E0_h = get_dual_dofs(Vh=V1h, f=E0, domain_h=domain_h, backend_language=backend)
        E_h = dH1.dot(tilde_E0_h)

        tilde_B0_h = get_dual_dofs(Vh=V2h, f=B0, domain_h=domain_h, backend_language=backend)
        B_h = dH2.dot(tilde_B0_h)

    else:
        raise ValueError(E0_type)

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    # time loop


    if plot_dir is not None and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

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
    Btemp_h = B_h.copy()
    Etemp_h = E_h.copy()

    print(" -- time loop --")
    for nt in range(Nt):
        print(' .. nt+1 = {}/{}'.format(nt+1, Nt))

        # 1/2 faraday: Bn -> Bn+1/2
        # B_h -=  (dt/2) * C @ E_h
        # E_h = A_eps @ E_h + dt * dC @ B_h
        # B_h -= (dt/2) * C @ E_h
        
        C.dot(E_h, out=Btemp_h)
        B_h -= (dt/2) * Btemp_h
        
        dCH1.dot(E_h, out=E_h)
        dC.dot(B_h, out=Etemp_h) 
        E_h += dt * (Etemp_h - f_h)

        C.dot(E_h, out=Btemp_h)
        B_h -= (dt/2) * Btemp_h

        if plot_dir:
            Eh = FemField(V1h, coeffs=cP1 @ E_h)
            OM1.add_snapshot(t=nt*dt, ts=nt) 
            OM1.export_fields(Eh = Eh)

            Bh = FemField(V2h, coeffs=B_h)
            OM2.add_snapshot(t=nt*dt, ts=nt) 
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

# ==============================================================================
# Compute stable time step size
# ==============================================================================
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

# ==============================================================================
# Test Sources
# ==============================================================================
def get_div_free_pulse(x_0, y_0, domain=None):

    from sympy import pi, cos, sin, Tuple, exp

    x, y = domain.coordinates
    ds2_0 = (0.02)**2
    sigma_0 = (x - x_0)**2 + (y - y_0)**2
    phi_0 = exp(-sigma_0**2 / (2 * ds2_0))
    dx_sig_0 = 2 * (x - x_0)
    dy_sig_0 = 2 * (y - y_0)
    dx_phi_0 = - dx_sig_0 * sigma_0 / ds2_0 * phi_0
    dy_phi_0 = - dy_sig_0 * sigma_0 / ds2_0 * phi_0
    f_x = dy_phi_0
    f_y = - dx_phi_0
    f_vect = Tuple(f_x, f_y)

    return f_vect


def get_curl_free_pulse(x_0, y_0, domain=None, pp=False):

    from sympy import pi, cos, sin, Tuple, exp

    # return -grad phi_0
    x, y = domain.coordinates
    if pp:
        # psi=phi
        ds2_0 = (0.02)**2
    else:
        ds2_0 = (0.1)**2
    sigma_0 = (x - x_0)**2 + (y - y_0)**2
    phi_0 = exp(-sigma_0**2 / (2 * ds2_0))
    dx_sig_0 = 2 * (x - x_0)
    dy_sig_0 = 2 * (y - y_0)
    dx_phi_0 = - dx_sig_0 * sigma_0 / ds2_0 * phi_0
    dy_phi_0 = - dy_sig_0 * sigma_0 / ds2_0 * phi_0
    f_x = -dx_phi_0
    f_y = -dy_phi_0
    f_vect = Tuple(f_x, f_y)

    return f_vect

def get_Gaussian_beam(x_0, y_0, domain=None):

    from sympy import pi, cos, sin, Tuple, exp

    # return E = cos(k*x) exp( - x^2 + y^2 / 2 sigma^2) v
    x, y = domain.coordinates

    x = x - x_0
    y = y - y_0

    sigma = 0.1

    xy = x**2 + y**2
    ef = 1 / (sigma**2) * exp(- xy / (2 * sigma**2))

    # E = curl exp
    E = Tuple(y * ef, -x * ef)

    # B = curl E
    B = (xy / (sigma**2) - 2) * ef

    return E, B

if __name__ == '__main__':
    domain_name = 'refined_square'
    domain_lims = [[0, np.pi], [0, np.pi]]

    nc = 20
    ncells  = np.array([[nc, nc, nc],
                        [nc, 2*nc, nc], 
                        [nc, nc, nc]])

    deg = 3
    p_moments = deg+1

    final_time = 2

    plot_dir = './td_maxwell_pulse/'  

    solve_td_maxwell_pbm(nc=ncells, deg=deg, p_moments=p_moments, final_time=final_time, 
                        domain_name=domain_name, domain_lims=domain_lims, 
                        source_type='zero', E0_type='pulse', plot_dir=plot_dir)
