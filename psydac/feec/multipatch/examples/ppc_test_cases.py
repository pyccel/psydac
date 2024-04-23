# coding: utf-8

from mpi4py import MPI

import os
import numpy as np

from sympy import pi, cos, sin, Tuple, exp, sqrt, acos, atan

from sympde.topology import Derham
from sympde.calculus  import dot

from psydac.fem.basic                                   import FemField
from psydac.feec.multipatch.api                         import discretize
from psydac.feec.multipatch.operators                   import HodgeOperator
from psydac.feec.multipatch.plotting_utilities          import plot_field
from psydac.feec.multipatch.plotting_utilities          import get_plotting_grid, my_small_plot, my_small_streamplot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

from psydac.feec.multipatch.utilities import sol_ref_fn, error_fn, get_method_name, get_fem_name, get_load_dir

comm = MPI.COMM_WORLD


# todo [MCP, 12/02/2022]:  add an 'equation' argument to be able to return 'exact solution'

def get_source_and_sol_for_magnetostatic_pbm(
    source_type=None,
    domain=None, domain_name=None,
    refsol_params=None
):
    x,y    = domain.coordinates
    if source_type == 'dipole_J':
        # we compute two possible source terms:
        #   . a dipole current j_scal = phi_0 - phi_1   (two blobs)
        #   . and f_vect = curl j_scal
        x_0 = 1.0
        y_0 = 1.0
        ds2_0 = (0.02)**2
        sigma_0 = (x-x_0)**2 + (y-y_0)**2
        phi_0 = exp(-sigma_0**2/(2*ds2_0))
        dx_sig_0 = 2*(x-x_0)
        dy_sig_0 = 2*(y-y_0)
        dx_phi_0 = - dx_sig_0 * sigma_0 / ds2_0 * phi_0
        dy_phi_0 = - dy_sig_0 * sigma_0 / ds2_0 * phi_0

        x_1 = 2.0
        y_1 = 2.0
        ds2_1 = (0.02)**2
        sigma_1 = (x-x_1)**2 + (y-y_1)**2
        phi_1 = exp(-sigma_1**2/(2*ds2_1))
        dx_sig_1 = 2*(x-x_1)
        dy_sig_1 = 2*(y-y_1)
        dx_phi_1 = - dx_sig_1 * sigma_1 / ds2_1 * phi_1
        dy_phi_1 = - dy_sig_1 * sigma_1 / ds2_1 * phi_1

        f_scal = None #
        j_scal = phi_0 - phi_1
        f_x    =   dy_phi_0 - dy_phi_1
        f_y    = - dx_phi_0 + dx_phi_1
        f_vect = Tuple(f_x, f_y)

    else:
        raise ValueError(source_type)

    # ref solution in V1h:
    uh_ref = get_sol_ref_V1h(source_type, domain, domain_name, refsol_params)

    return f_scal, f_vect, j_scal, uh_ref


def get_source_and_solution(source_type=None, eta=0, mu=0, nu=0,
                            domain=None, domain_name=None,
                            refsol_params=None):
    """
    compute source and reference solution (exact, or reference values) when possible, depending on the source_type
    """

    # ref solution (values on diag grid)
    ph_ref = None
    uh_ref = None

    # exact solutions (if available)
    u_ex = None
    p_ex = None

    # bc solution: describe the bc on boundary. Inside domain, values should not matter. Homogeneous bc will be used if None
    u_bc = None
    # only hom bc on p (for now...)

    # source terms
    f_vect = None
    f_scal = None

    # auxiliary term (for more diagnostics)
    grad_phi = None
    phi = None

    x,y    = domain.coordinates

    if source_type == 'manu_J':
        # todo: remove if not used ?
        # use a manufactured solution, with ad-hoc (homogeneous or inhomogeneous) bc
        if domain_name in ['square_2', 'square_6', 'square_8', 'square_9']:
            t = 1
        else:
            t = pi

        u_ex   = Tuple(sin(t*y), sin(t*x)*cos(t*y))
        f_vect = Tuple(
            sin(t*y) * (eta + t**2 *(mu - cos(t*x)*(mu-nu))),
            sin(t*x) * cos(t*y) * (eta + t**2 *(mu+nu) )
        )

        # boundary condition: (here we only need to coincide with u_ex on the boundary !)
        if domain_name in ['square_2', 'square_6', 'square_9']:
            u_bc = None
        else:
            u_bc = u_ex

    elif source_type == 'manutor_poisson':
        # todo: remove if not used ?
        # same as manu_poisson, with arbitrary value for tor
        x0 = 1.5
        y0 = 1.5
        s  = (x-x0) - (y-y0)
        t  = (x-x0) + (y-y0)
        a = (1/1.9)**2
        b = (1/1.2)**2
        sigma2 = 0.0121
        tor = 2
        tau = a*s**2 + b*t**2 - 1
        phi = exp(-tau**tor/(2*sigma2))
        dx_tau = 2*( a*s + b*t)
        dy_tau = 2*(-a*s + b*t)
        dxx_tau = 2*(a + b)
        dyy_tau = 2*(a + b)
        f_scal = -((tor*tau**(tor-1)*dx_tau/(2*sigma2))**2 - (tau**(tor-1)*dxx_tau + (tor-1)*tau**(tor-2)*dx_tau**2)*tor/(2*sigma2)
                   +(tor*tau**(tor-1)*dy_tau/(2*sigma2))**2 - (tau**(tor-1)*dyy_tau + (tor-1)*tau**(tor-2)*dy_tau**2)*tor/(2*sigma2))*phi
        p_ex = phi

    elif source_type == 'manu_maxwell':
        # used for Maxwell equation with manufactured solution
        alpha   = eta
        u_ex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
        f_vect  = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                        alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))
        u_bc = u_ex

    elif source_type in ['manu_poisson', 'elliptic_J']:
        # 'manu_poisson': used for Poisson pbm with manufactured solution
        # 'elliptic_J': used for Maxwell pbm (no manufactured solution)   --   (was 'ellnew_J' in previous version)
        x0 = 1.5
        y0 = 1.5
        s  = (x-x0) - (y-y0)
        t  = (x-x0) + (y-y0)
        a = (1/1.9)**2
        b = (1/1.2)**2
        sigma2 = 0.0121
        tau = a*s**2 + b*t**2 - 1
        phi = exp(-tau**2/(2*sigma2))
        dx_tau = 2*( a*s + b*t)
        dy_tau = 2*(-a*s + b*t)
        dxx_tau = 2*(a + b)
        dyy_tau = 2*(a + b)

        dx_phi = (-tau*dx_tau/sigma2)*phi
        dy_phi = (-tau*dy_tau/sigma2)*phi
        grad_phi = Tuple(dx_phi, dy_phi)


        f_scal = -( (tau*dx_tau/sigma2)**2 - (tau*dxx_tau + dx_tau**2)/sigma2
                   +(tau*dy_tau/sigma2)**2 - (tau*dyy_tau + dy_tau**2)/sigma2 )*phi

        # exact solution of  -p'' = f  with hom. bc's on pretzel domain
        p_ex = phi

        if not domain_name in ['pretzel', 'pretzel_f']:
            print("WARNING (87656547) -- I'm not sure we have an exact solution -- check the bc's on the domain "+domain_name)
            # raise NotImplementedError(domain_name)

        f_x =   dy_tau * phi
        f_y = - dx_tau * phi
        f_vect = Tuple(f_x, f_y)

    elif source_type == 'manu_poisson_2':
        f_scal = -4
        p_ex   = x**2+y**2
        phi    = p_ex
        u_bc   = p_ex
        u_ex   = p_ex
    elif source_type == 'curl_dipole_J':
        # used for the magnetostatic problem

        # was 'dicurl_J' in previous version

        # here, f is the curl of a dipole current j = phi_0 - phi_1 (two blobs) that correspond to a scalar current density
        #
        # the solution u of the curl-curl problem with free-divergence constraint
        #   curl curl u = curl j
        #
        # then corresponds to a magnetic density,
        # see Beirão da Veiga, Brezzi, Dassi, Marini and Russo, Virtual Element approx of 2D magnetostatic pbms, CMAME 327 (2017)

        x_0 = 1.0
        y_0 = 1.0
        ds2_0 = (0.02)**2
        sigma_0 = (x-x_0)**2 + (y-y_0)**2
        phi_0 = exp(-sigma_0**2/(2*ds2_0))
        dx_sig_0 = 2*(x-x_0)
        dy_sig_0 = 2*(y-y_0)
        dx_phi_0 = - dx_sig_0 * sigma_0 / ds2_0 * phi_0
        dy_phi_0 = - dy_sig_0 * sigma_0 / ds2_0 * phi_0

        x_1 = 2.0
        y_1 = 2.0
        ds2_1 = (0.02)**2
        sigma_1 = (x-x_1)**2 + (y-y_1)**2
        phi_1 = exp(-sigma_1**2/(2*ds2_1))
        dx_sig_1 = 2*(x-x_1)
        dy_sig_1 = 2*(y-y_1)
        dx_phi_1 = - dx_sig_1 * sigma_1 / ds2_1 * phi_1
        dy_phi_1 = - dy_sig_1 * sigma_1 / ds2_1 * phi_1

        f_x =   dy_phi_0 - dy_phi_1
        f_y = - dx_phi_0 + dx_phi_1
        f_scal = 0 # phi_0 - phi_1
        f_vect = Tuple(f_x, f_y)

    elif source_type == 'old_ellip_J':

        # divergence-free f field along an ellipse curve
        if domain_name in ['pretzel', 'pretzel_f']:
            dr = 0.2
            r0 = 1
            x0 = 1.5
            y0 = 1.5
            # s0 = x0-y0
            # t0 = x0+y0
            s  = (x-x0) - (y-y0)
            t  = (x-x0) + (y-y0)
            aa = (1/1.7)**2
            bb = (1/1.1)**2
            dsigpsi2 = 0.01
            sigma = aa*s**2 + bb*t**2 - 1
            psi = exp(-sigma**2/(2*dsigpsi2))
            dx_sig = 2*( aa*s + bb*t)
            dy_sig = 2*(-aa*s + bb*t)
            f_x =   dy_sig * psi
            f_y = - dx_sig * psi

            dsigphi2 = 0.01     # this one gives approx 1e-10 at boundary for phi
            # dsigphi2 = 0.005   # if needed: smaller support for phi, to have a smaller value at boundary
            phi = exp(-sigma**2/(2*dsigphi2))
            dx_phi = phi*(-dx_sig*sigma/dsigphi2)
            dy_phi = phi*(-dy_sig*sigma/dsigphi2)

            grad_phi = Tuple(dx_phi, dy_phi)
            f_vect = Tuple(f_x, f_y)

        else:
            raise NotImplementedError

    elif source_type in ['ring_J', 'sring_J']:
        # used for the magnetostatic problem
        # 'rotating' (divergence-free) f field:

        if domain_name in ['square_2', 'square_6', 'square_8', 'square_9']:
            r0 = np.pi/4
            dr = 0.1
            x0 = np.pi/2
            y0 = np.pi/2
            omega = 43/2
            # alpha  = -omega**2  # not a square eigenvalue
            f_factor = 100

        elif domain_name in ['curved_L_shape']:
            r0 = np.pi/4
            dr = 0.1
            x0 = np.pi/2
            y0 = np.pi/2
            omega = 43/2
            # alpha  = -omega**2  # not a square eigenvalue
            f_factor = 100

        else:
            # for pretzel

            # omega = 8  # ?
            # alpha  = -omega**2

            source_option = 2

            if source_option==1:
                # big circle:
                r0 = 2.4
                dr = 0.05
                x0 = 0
                y0 = 0.5
                f_factor = 10

            elif source_option==2:
                # small circle in corner:
                if source_type == 'ring_J':
                    dr = 0.2
                else:
                    # smaller ring
                    dr = 0.1
                    assert source_type == 'sring_J'
                r0 = 1
                x0 = 1.5
                y0 = 1.5
                f_factor = 10

            else:
                raise NotImplementedError

        # note: some other currents give sympde or numba errors, see below [1]
        phi = f_factor * exp( - .5*(( (x-x0)**2 + (y-y0)**2 - r0**2 )/dr)**2 )

        f_x = - (y-y0) * phi
        f_y =   (x-x0) * phi

        f_vect = Tuple(f_x, f_y)

    else:
        raise ValueError(source_type)

    if u_ex is None:
        uh_ref = get_sol_ref_V1h(source_type, domain, domain_name, refsol_params)

    return f_scal, f_vect, u_bc, ph_ref, uh_ref, p_ex, u_ex, phi, grad_phi


def get_sol_ref_V1h( source_type=None, domain=None, domain_name=None, refsol_params=None ):
    """
    get a reference solution as a V1h FemField
    """
    uh_ref = None
    if refsol_params is not None:
        N_diag, method_ref, source_proj_ref = refsol_params
        u_ref_filename = ( get_load_dir(method=method_ref, domain_name=domain_name,nc=None,deg=None,data='solutions')
                         + sol_ref_fn(source_type, N_diag, source_proj=source_proj_ref) )
        print("no exact solution for this test-case, looking for ref solution values in file {}...".format(u_ref_filename))
        if os.path.isfile(u_ref_filename):
            print("-- file found")
            with open(u_ref_filename, 'rb') as file:
                ncells_degree = np.load(file)
                ncells   = [int(i) for i in ncells_degree['ncells_degree'][0]]
                degree   = [int(i) for i in ncells_degree['ncells_degree'][1]]

            derham   = Derham(domain, ["H1", "Hcurl", "L2"])
            domain_h = discretize(domain, ncells=ncells, comm=comm)
            V1h      = discretize(derham.V1, domain_h, degree=degree, basis='M')
            uh_ref   = FemField(V1h)
            for i,Vi in enumerate(V1h.spaces):
                for j,Vij in enumerate(Vi.spaces):
                    filename = u_ref_filename+'_%d_%d'%(i,j)
                    uij = Vij.import_fields(filename, 'phi')
                    uh_ref.fields[i].fields[j].coeffs._data = uij[0].coeffs._data

        else:
            print("-- no file, skipping it")

    return uh_ref

def get_div_free_pulse(x_0, y_0, domain=None):
    x,y    = domain.coordinates
    ds2_0 = (0.02)**2
    sigma_0 = (x-x_0)**2 + (y-y_0)**2
    phi_0 = exp(-sigma_0**2/(2*ds2_0))
    dx_sig_0 = 2*(x-x_0)
    dy_sig_0 = 2*(y-y_0)
    dx_phi_0 = - dx_sig_0 * sigma_0 / ds2_0 * phi_0
    dy_phi_0 = - dy_sig_0 * sigma_0 / ds2_0 * phi_0
    f_x    =   dy_phi_0
    f_y    = - dx_phi_0
    f_vect = Tuple(f_x, f_y)

    return f_vect


def get_cavity_solution(omega, kx, ky, t, domain):
    """
    dt B + curl E = 0
    dt E - c**2 * curl B = (1/eps) * J
    """

    x,y    = domain.coordinates

        # Exact solutions for electric and magnetic field
    E_ex = Tuple( 
         ky * cos(kx * x) * sin(ky * y) * cos(omega * t),
        -kx * sin(kx * x) * cos(ky * y) * cos(omega * t)
    )
    B_ex = cos(kx * x) * cos(ky * y) * sin(omega * t) * (kx**2 + ky**2) / omega

    return E_ex, B_ex

def get_polarized_solution(omega, kx, ky, kappa, t, domain):
    """
    dt B + curl E = 0
    D = ((1+kappa)E_x, E_y) 
    dt D - curl B = J
    """
    x,y = domain.coordinates
    
    E_ex = Tuple( 
         ky * cos(kx * x) * sin(ky * y) * cos(omega * t),
        -kx * sin(kx * x) * cos(ky * y) * cos(omega * t)
    )
    B_ex = cos(kx * x) * cos(ky * y) * sin(omega * t) * (kx**2 + ky**2) / omega

    D_ex = Tuple(
                (1 + kappa) * E_ex[0],
                              E_ex[1]
                )

    J_ex = Tuple(
        kappa * omega * ky * cos(kx * x) * sin(ky * y) * sin(omega * t),
        0
    )

    return E_ex, B_ex, D_ex, J_ex

def get_polarized_source(omega, kx, ky, kappa, t, domain):
    x,y = domain.coordinates
    return Tuple(
        kappa * omega * ky * cos(kx * x) * sin(ky * y) * sin(omega * t),
        0
    )

# dt B + curl E = (kx**2 + ky**2) + (-kx*2 - ky**2)

# dt E   = (- ky * cos(kx * x) * sin(ky * y) ,  kx * sin(kx * x) * cos(ky * y) )  omega * sin(omega * t)
# curl B = (- ky * cos(kx * x) * sin(ky * y)  , kx * sin(kx * x) * cos(ky * y) ) * (kx**2 + ky**2) / omega * sin(omega * t)

# OK IF (kx**2 + ky**2) = omega**2
from sympy import arg, I, sign

def get_polarized_annulus_potential_solution(omega, k_theta, epsilon, kappa, t, r_min, r_max, domain):
    """<
    dt B + curl E = 0
    D = ((1+kappa)E_x, E_y) 
    dt D - curl B = J
    """
    x,y = domain.coordinates
    r = sqrt(x**2 + y**2)
    theta = sign(y) * acos(x/r) #arg(x + I * y) #acos(x/r)

    delta_r = r_max - r_min

    # potential phi
    phi = -(r_max - r)/delta_r - epsilon * cos(k_theta * theta) * (r_max - r)*(r - r_min) *sin(omega *t)
    
    
    # E = - grad phi
    E_hat = Tuple( 
         1/delta_r - epsilon * cos(k_theta * theta) * (r_min + r_max - 2 * r) * sin(omega* t),
         epsilon * k_theta * sin(k_theta * theta) * (r_max - r) * (r - r_min) * sin(omega* t)
    )

    dt_E_hat = Tuple( 
         -omega * epsilon * cos(k_theta * theta) * (r_min + r_max - 2 * r) * cos(omega* t),
         omega * epsilon * k_theta * sin(k_theta * theta) * (r_max - r) * (r - r_min) * cos(omega* t)
    )

    # E = DF^{-T} \hat E
    E_phys = Tuple(
        cos(theta) * E_hat[0] - 1/r * sin(theta) * E_hat[1],
        sin(theta) * E_hat[0] + 1/r * cos(theta) * E_hat[1]
    )

    # B = curl E = 0
    B_phys = 0

    # D = E + kappa * (E - (b \dot E) b) 
    D_hat = Tuple(
                r *(1 + kappa) * E_hat[0],
                1/r *            E_hat[1]
                )

    # D = (1/r DF) \hat D
    D_phys = Tuple(
      1/r *  cos(theta) * D_hat[0] -  sin(theta) * D_hat[1],
       1/r *  sin(theta) * D_hat[0] +  cos(theta) * D_hat[1]
    )

    # J = - \partial_t D
    J_hat = Tuple(
        omega * r *(1 + kappa) * epsilon * cos(k_theta * theta) * (r_min + r_max - 2* r) * cos(omega* t),
        -omega * 1/r * epsilon * k_theta * sin(k_theta * theta) * (r_max - r) * (r - r_min) * cos(omega* t)
    )

    # J = (1/r DF) \hat J
    J_phys = Tuple(
       1/r *  cos(theta) * J_hat[0] -  sin(theta) * J_hat[1],
       1/r * sin(theta) * J_hat[0] +  cos(theta) * J_hat[1]
    )

    return  E_phys, B_phys, D_phys, J_phys

def get_polarized_annulus_potential_source(omega, k_theta, epsilon, kappa, t, r_min, r_max, domain):
    x,y = domain.coordinates
    r = sqrt(x**2 + y**2)
    theta = sign(y) * acos(x/r) #theta = arg(x + I * y) 
    dt_E_hat = Tuple( 
         -omega * epsilon * cos(k_theta * theta) * (r_min + r_max - 2 * r) * cos(omega* t),
         omega * epsilon * k_theta * sin(k_theta * theta) * (r_max - r) * (r - r_min) * cos(omega* t)
    )

    return Tuple(
       1/r *  cos(theta) * J_ex_[0] -  sin(theta) * J_ex_[1],
       1/r * sin(theta) * J_ex_[0] +  cos(theta) * J_ex_[1]
    )



def get_polarized_annulus_potential_solution_old(b, omega, k_theta, epsilon, kappa, t, r_min, r_max, domain):
    """<
    dt B + curl E = 0
    D = ((1+kappa)E_x, E_y) 
    dt D - curl B = J
    """
    x,y = domain.coordinates
    r = sqrt(x**2 + y**2)
    theta = sign(y) * acos(x/r) #arg(x + I * y) #acos(x/r)

    delta_r = r_max - r_min

    # potential phi
    #phi = -(r_max - r)/delta_r - epsilon * cos(k_theta * theta) * (r_max - r)*(r - r_min) *sin(omega *t)
    
    
    # E = - grad phi
    E_hat = Tuple( 
         1/delta_r * cos(omega * t) - epsilon * cos(k_theta * theta) * (r_min + r_max - 2 * r) * cos(omega* t),
         epsilon * k_theta * sin(k_theta * theta) * (r_max - r) * (r - r_min) * cos(omega* t)
    )

    dt_E_hat = Tuple( 
         -1/delta_r * omega * sin(omega * t) +omega * epsilon * cos(k_theta * theta) * (r_min + r_max - 2 * r) * sin(omega* t),
         -omega * epsilon * k_theta * sin(k_theta * theta) * (r_max - r) * (r - r_min) * sin(omega* t)
    )

    # E = DF^{-T} \hat E
    E_phys = Tuple(
        cos(theta) * E_hat[0] - 1/r * sin(theta) * E_hat[1],
        sin(theta) * E_hat[0] + 1/r * cos(theta) * E_hat[1]
    )

    dt_E_phys = Tuple(
        cos(theta) * dt_E_hat[0] - 1/r * sin(theta) * dt_E_hat[1],
        sin(theta) * dt_E_hat[0] + 1/r * cos(theta) * dt_E_hat[1]
    )

    # B = curl E = 0
    #B_phys = 100* cos((r - r_min)/delta_r * 2*pi) * sin(theta) #100 * r # = B_hat
    B_phys = 100 * r
    H_phys = B_phys

    B_hat = r * B_phys #cos((r - r_min)/delta_r * 2*pi) # = H_phys
    H_hat = H_phys 

    # curl B
    # 1/r DF 
   # curl_H_hat = Tuple(100* cos((r - r_min)/delta_r * 2*pi) * cos(theta), 100* (2*pi)/delta_r * sin((r - r_min)/delta_r * 2*pi)* sin(theta))
    curl_H_hat = Tuple(0, -100) #cos((r - r_min)/delta_r * 2*pi) - r * (2*pi)/delta_r * sin((r - r_min)/delta_r * 2*pi))

    curl_H_phys = Tuple(
        1/r * cos(theta) * curl_H_hat[0] -  sin(theta) * curl_H_hat[1],
        1/r * sin(theta) * curl_H_hat[0] +  cos(theta) * curl_H_hat[1]
    )

    b1 = b[0]
    b2 = b[1]
    E1 = E_phys[0]
    E2 = E_phys[1]
    Eb = E1*b1 + E2*b2

    # D = (1/r DF) \hat D
    D_phys = Tuple( (1+kappa) * E1 - kappa * Eb * b1, 
                    (1+kappa) * E2 - kappa * Eb * b2)

    dtE1 = dt_E_phys[0]
    dtE2 = dt_E_phys[1]
    dtEb = dtE1*b1 + dtE2*b2

    # J = - \partial_t D + curl H
    J_phys = Tuple( -(1+kappa) * dtE1 + kappa * dtEb * b1 + curl_H_phys[0], 
                    -(1+kappa) * dtE2 + kappa * dtEb * b2 + curl_H_phys[1]) #+ curl_H_phys

    return   E_phys, B_phys, D_phys, J_phys

def get_polarized_annulus_potential_source_old(b, omega, k_theta, epsilon, kappa, t, r_min, r_max, domain):
    x,y = domain.coordinates
    r = sqrt(x**2 + y**2)
    theta = sign(y) * acos(x/r) #theta = arg(x + I * y) 
    delta_r = r_max - r_min
    
    dt_E_hat = Tuple( 
         - 1/delta_r * omega * sin(omega * t) +omega * epsilon * cos(k_theta * theta) * (r_min + r_max - 2 * r) * sin(omega* t),
         -omega * epsilon * k_theta * sin(k_theta * theta) * (r_max - r) * (r - r_min) * sin(omega* t)
    )

    dt_E_phys = Tuple(
        cos(theta) * dt_E_hat[0] - 1/r * sin(theta) * dt_E_hat[1],
        sin(theta) * dt_E_hat[0] + 1/r * cos(theta) * dt_E_hat[1]
    )

    b1 = b[0]
    b2 = b[1]
    dtE1 = dt_E_phys[0]
    dtE2 = dt_E_phys[1]
    dtEb = dtE1*b1 +  dtE2*b2

    #curl_H_hat = Tuple(100 * cos((r - r_min)/delta_r * 2*pi) * cos(theta), 100 * (2*pi)/delta_r * sin((r - r_min)/delta_r * 2*pi)* sin(theta))
    curl_H_hat = Tuple(0, -100)#cos((r - r_min)/delta_r * 2*pi) - r * (2*pi)/delta_r * sin((r - r_min)/delta_r * 2*pi))
    curl_H_phys = Tuple(
        1/r * cos(theta) * curl_H_hat[0] -  sin(theta) * curl_H_hat[1],
        1/r * sin(theta) * curl_H_hat[0] +  cos(theta) * curl_H_hat[1]
    )

    return Tuple( -(1+kappa) * dtE1 + kappa * dtEb * b1 + curl_H_phys[0], 
                    -(1+kappa) * dtE2 + kappa * dtEb * b2 + curl_H_phys[1]) #+ curl_H_phys



def get_polarized_annulus_potential_solution(b, omega, k_theta, epsilon, kappa, t, r_min, r_max, domain):
    """
    dt B + curl E = 0
    D = ((1+kappa)E_x, E_y) 
    dt D - curl B = J
    """
    x,y = domain.coordinates
    r = sqrt(x**2 + y**2)
    theta = sign(y) * acos(x/r) #arg(x + I * y) #acos(x/r)

    delta_r = r_max - r_min

    # potential phi(r, theta)
    phi = cos(k_theta * theta) * (r_max - r)*(r - r_min) 
    dr_phi = cos(k_theta * theta) * (r_min + r_max - 2 * r)
    dtheta_phi = -k_theta * sin(k_theta * theta) * (r_max - r)*(r - r_min)
    
    barphi = (r - r_min)/delta_r 
    dr_barphi = 1/delta_r
    dtheta_barphi = 0
    # E =  grad phi cos(omega t) + epsilon (phi, 0)^T sin(omega t)
    E_hat = Tuple( 
        cos(omega * t) * dr_barphi     + epsilon * sin(omega * t) * phi,
        cos(omega * t) * dtheta_barphi
    )

    dt_E_hat = Tuple( 
        - omega * sin(omega * t) * dr_barphi     + epsilon * omega * cos(omega * t) * phi,
        - omega * sin(omega * t) * dtheta_barphi
    )

    # E = DF^{-T} \hat E
    E_phys = Tuple(
        cos(theta) * E_hat[0] - 1/r * sin(theta) * E_hat[1],
        sin(theta) * E_hat[0] + 1/r * cos(theta) * E_hat[1]
    )

    dt_E_phys = Tuple(
        cos(theta) * dt_E_hat[0] - 1/r * sin(theta) * dt_E_hat[1],
        sin(theta) * dt_E_hat[0] + 1/r * cos(theta) * dt_E_hat[1]
    )

    # dt B = - curl E =  epsilon * sin(omega * t) * k_theta * sin(k_theta * theta) * (r_max - r)*(r - r_min)
    # dt B = - curl E = epsilon * sin(omega * t)  * dtheta_phi 
    # B = -1/omega * cos(omega * t) * epsilon * dtheta_phi
    B_phys = -1/omega * cos(omega * t) * epsilon * dtheta_phi
    H_phys = B_phys

    B_hat = r * B_phys 
    H_hat = H_phys 

    # curl H =  (dtheta B, -dr B)
    curl_H_hat = Tuple(1/omega * cos(omega * t) * epsilon * k_theta**2 * cos(k_theta * theta) * (r_max - r)*(r - r_min),
                       -1/omega * cos(omega * t) * epsilon * k_theta * sin(k_theta * theta) * (r_max + r_min - 2 * r)
    )

    curl_H_phys = Tuple(
        1/r * cos(theta) * curl_H_hat[0] -  sin(theta) * curl_H_hat[1],
        1/r * sin(theta) * curl_H_hat[0] +  cos(theta) * curl_H_hat[1]
    )

    b1 = b[0]
    b2 = b[1]
    E1 = E_phys[0]
    E2 = E_phys[1]
    Eb = E1*b1 + E2*b2

    # D = (1/r DF) \hat Dtheth
    D_phys = Tuple( (1+kappa) * E1 - kappa * Eb * b1, 
                    (1+kappa) * E2 - kappa * Eb * b2)

    dtE1 = dt_E_phys[0]
    dtE2 = dt_E_phys[1]
    dtEb = dtE1*b1 + dtE2*b2

    # J = - \partial_t D + curl H
    J_phys = Tuple( -(1+kappa) * dtE1 + kappa * dtEb * b1 + curl_H_phys[0], 
                    -(1+kappa) * dtE2 + kappa * dtEb * b2 + curl_H_phys[1]) 

    return   E_phys, B_phys, D_phys, J_phys


def get_polarized_annulus_potential_source(b, omega, k_theta, epsilon, kappa, t, r_min, r_max, domain):
    x,y = domain.coordinates
    r = sqrt(x**2 + y**2)
    theta = sign(y) * acos(x/r) #theta = arg(x + I * y) 
    delta_r = r_max - r_min
    
    phi = cos(k_theta * theta) * (r_max - r)*(r - r_min) 
    dr_phi = cos(k_theta * theta) * (r_min + r_max - 2 * r)
    dtheta_phi = -k_theta * sin(k_theta * theta) * (r_max - r)*(r - r_min)
    
    barphi = (r - r_min)/delta_r 
    dr_barphi = 1/delta_r
    dtheta_barphi = 0

    dt_E_hat = Tuple( 
        - omega * sin(omega * t) * dr_barphi     + epsilon * omega * cos(omega * t) * phi,
        - omega * sin(omega * t) * dtheta_barphi
    )

    dt_E_phys = Tuple(
        cos(theta) * dt_E_hat[0] - 1/r * sin(theta) * dt_E_hat[1],
        sin(theta) * dt_E_hat[0] + 1/r * cos(theta) * dt_E_hat[1]
    )

    b1 = b[0]
    b2 = b[1]
    dtE1 = dt_E_phys[0]
    dtE2 = dt_E_phys[1]
    dtEb = dtE1*b1 +  dtE2*b2

    curl_H_hat = Tuple(1/omega * cos(omega * t) * epsilon * k_theta**2 * cos(k_theta * theta) * (r_max - r)*(r - r_min),
                       -1/omega * cos(omega * t) * epsilon * k_theta * sin(k_theta * theta) * (r_max + r_min - 2 * r)
    )

    curl_H_phys = Tuple(
        1/r * cos(theta) * curl_H_hat[0] -  sin(theta) * curl_H_hat[1],
        1/r * sin(theta) * curl_H_hat[0] +  cos(theta) * curl_H_hat[1]
    )

    return Tuple( -(1+kappa) * dtE1 + kappa * dtEb * b1 + curl_H_phys[0], 
                    -(1+kappa) * dtE2 + kappa * dtEb * b2 + curl_H_phys[1]) 

def get_poisson_annulus_solution(r_min, r_max, domain):
    x, y = domain.coordinates
    r = sqrt(x**2 + y**2)
    theta = sign(y) * acos(x/r) #theta = arg(x + I * y) 

    alpha = 1
    f = -alpha * (15 * r**2 - 8 * (r_min + r_max)*r + 3*r_min*r_max) * sin(theta)
    phi = alpha * (r**4 - (r_min + r_max)*r**3 + r_min*r_max*r**2) * sin(theta)

    return f, phi

def get_poisson_solution(x_min, x_max, y_min, y_max, domain):
    # on [0, pi] x [0, pi]
    x, y = domain.coordinates
   
    
    k_x = 2
    k_y = 2

    gamma = 1 + exp( ((x - pi/2)**2 + (y - pi/2)**2)/ (20**2) ) 
    dxgamma = 2/(20**2) * (x - pi/2) * exp( ((x - pi/2)**2 + (y - pi/2)**2)/ 20**2 )
    dygamma = 2/(20**2) * (y - pi/2) * exp( ((x - pi/2)**2 + (y - pi/2)**2)/ 20**2 )

    phi = sin( k_x * x ) * sin( k_y * y )

    dxphi = k_x * cos( k_x * x ) * sin( k_y * y)
    dyphi = k_y * sin( k_x * x ) * cos( k_y * y)

    dxxphi =  -phi * (k_x)**2
    dyyphi =  -phi * (k_y)**2
    # f = ((2*pi*k_x/delta_x)**2 + (2*pi*k_y/delta_y)**2) * phi
    f = -(dxxphi * gamma + dxphi * dxgamma + dyyphi * gamma + dyphi * dygamma)
    return f, phi