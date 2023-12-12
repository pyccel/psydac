# coding: utf-8

from mpi4py import MPI

import os
import numpy as np

from sympy import pi, cos, sin, Tuple, exp, atan, atan2

from sympde.topology import Derham

from psydac.fem.basic                                   import FemField
from psydac.feec.multipatch.api                         import discretize
from psydac.feec.multipatch.operators                   import HodgeOperator
from psydac.feec.multipatch.plotting_utilities          import plot_field
from psydac.feec.multipatch.plotting_utilities          import get_plotting_grid, my_small_plot, my_small_streamplot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

comm = MPI.COMM_WORLD


# todo [MCP, 12/02/2022]:  add an 'equation' argument to be able to return 'exact solution'

def get_phi_pulse(x_0, y_0, domain=None):
    x,y    = domain.coordinates
    ds2_0 = (0.02)**2
    sigma_0 = (x-x_0)**2 + (y-y_0)**2
    phi_0 = exp(-sigma_0**2/(2*ds2_0))

    return phi_0

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

def get_curl_free_pulse(x_0, y_0, domain=None, pp=False):
    # return -grad phi_0
    x,y    = domain.coordinates
    if pp:
        # psi=phi
        ds2_0 = (0.02)**2
    else:
        ds2_0 = (0.1)**2
    sigma_0 = (x-x_0)**2 + (y-y_0)**2
    phi_0 = exp(-sigma_0**2/(2*ds2_0))
    dx_sig_0 = 2*(x-x_0)
    dy_sig_0 = 2*(y-y_0)
    dx_phi_0 = - dx_sig_0 * sigma_0 / ds2_0 * phi_0
    dy_phi_0 = - dy_sig_0 * sigma_0 / ds2_0 * phi_0
    f_x    = -dx_phi_0
    f_y    = -dy_phi_0
    f_vect = Tuple(f_x, f_y)

    return f_vect

def get_Delta_phi_pulse(x_0, y_0, domain=None, pp=False):
    # return -Delta phi_0, with same phi_0 as in get_curl_free_pulse()
    x,y    = domain.coordinates
    if pp:
        # psi=phi
        ds2_0 = (0.02)**2
    else:
        ds2_0 = (0.1)**2
    sigma_0 = (x-x_0)**2 + (y-y_0)**2
    phi_0 = exp(-sigma_0**2/(2*ds2_0))
    dx_sig_0 = 2*(x-x_0)
    dy_sig_0 = 2*(y-y_0)
    dxx_sig_0 = 2
    dyy_sig_0 = 2
    dxx_phi_0 = ((dx_sig_0 * sigma_0 / ds2_0)**2 - ((dx_sig_0)**2 + dxx_sig_0 * sigma_0)/ds2_0 ) * phi_0
    dyy_phi_0 = ((dy_sig_0 * sigma_0 / ds2_0)**2 - ((dy_sig_0)**2 + dyy_sig_0 * sigma_0)/ds2_0 ) * phi_0
    f    = - dxx_phi_0 - dyy_phi_0

    return f

def get_Gaussian_beam_old(x_0, y_0, domain=None):
    # return E = cos(k*x) exp( - x^2 + y^2 / 2 sigma^2) v
    x,y    = domain.coordinates
    x = x - x_0
    y = y - y_0
    
    k = (10, 0)
    nk = np.sqrt(k[0]**2 + k[1]**2)

    v = (k[0]/nk, k[1]/nk)
    
    sigma = 0.05

    xy = x**2 + y**2
    ef = exp( - xy/(2*sigma**2) )

    E = cos(k[1] * x + k[0] * y) * ef
    B = (-v[1]*x + v[0]*y)/(sigma**2) * E 
    
    return Tuple(v[0]*E, v[1]*E), B

from sympy.functions.special.error_functions import erf
def get_Gaussian_beam(x_0, y_0, domain=None):
    # return E = cos(k*x) exp( - x^2 + y^2 / 2 sigma^2) v
    x,y    = domain.coordinates
    
    x = x - x_0
    y = y - y_0
    
    sigma = 0.1

    xy = x**2 + y**2
    ef = 1/(sigma**2) * exp( - xy/(2*sigma**2) )

    # E = curl exp
    E =  Tuple(  y * ef,  -x * ef)

    # B = curl E 
    B =  (xy/(sigma**2) - 2) * ef
    
    return E, B

def get_diag_Gaussian_beam(x_0, y_0, domain=None):
    # return E = cos(k*x) exp( - x^2 + y^2 / 2 sigma^2) v
    x,y    = domain.coordinates
    x = x - x_0
    y = y - y_0
    
    k = (np.pi, np.pi)
    nk = np.sqrt(k[0]**2 + k[1]**2)

    v = (k[0]/nk, k[1]/nk)
    
    sigma = 0.25

    xy = x**2 + y**2
    ef = exp( - xy/(2*sigma**2) )

    E = cos(k[1] * x + k[0] * y) * ef
    B = (-v[1]*x + v[0]*y)/(sigma**2) * E 
    
    return Tuple(v[0]*E, v[1]*E), B

def get_easy_Gaussian_beam(x_0, y_0, domain=None):
    # return E = cos(k*x) exp( - x^2 + y^2 / 2 sigma^2) v
    x,y    = domain.coordinates
    x = x - x_0
    y = y - y_0
    
    k = pi 
    sigma = 0.5

    xy = x**2 + y**2
    ef = exp( - xy/(2*sigma**2) )

    E = cos(k * y) * ef
    B = -y/(sigma**2) * E 
    
    return Tuple(E, 0), B

def get_Gaussian_beam2(x_0, y_0, domain=None):
    """ 
    Gaussian beam
    Beam inciding from the left, centered and normal to wall:
        x: axial normalized distance to the beam's focus
        y: radial normalized distance to the center axis of the beam
    """
    x,y    = domain.coordinates


    x0 = x_0
    y0 = y_0
    theta = pi/2
    w0 = 1

    t = [(x-x0)*cos(theta) - (y - y0) * sin(theta), (x-x0)*sin(theta) + (y-y0) * cos(theta)]

    
    
    EW0 = 1.0  # amplitude at the waist
    k0 = 2 * pi # free-space wavenumber

    x_ray = pi * w0 ** 2 # Rayleigh range

    w = w0 * ( 1 + t[0]**2/x_ray**2 )**0.5 #width
    curv = t[0] / ( t[0]**2 + x_ray**2 ) #curvature

    gouy_psi = -0.5 * atan2(t[0] / x_ray, 1.)  # corresponds to atan(x / x_ray), which is the Gouy phase

    EW_mod = EW0 * (w0 / w)**0.5 * exp(-(t[1] ** 2) / (w ** 2)) # Amplitude
    phase = k0 * t[0] + 0.5 * k0 * curv * t[1] ** 2 + gouy_psi # Phase

    EW_r = EW_mod * cos(phase) # Real part
    EW_i = EW_mod * sin(phase) # Imaginary part

    B = 0#t[1]/(w**2) * EW_r

    return Tuple(0,EW_r), B


def get_source_and_sol_for_magnetostatic_pbm(
    source_type=None,
    domain=None, domain_name=None,
    refsol_params=None
):
    """
    provide source, and exact solutions when available, for:
    
    Find u=B in H(curl) such that
    
        div B = 0
        curl B = j

    written as a mixed problem, see solve_magnetostatic_pbm()
    """
    u_ex = None  # exact solution
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

    return f_scal, f_vect, j_scal, u_ex

    
def get_source_and_solution_hcurl(
    source_type=None, eta=0, mu=0, nu=0,
    domain=None, domain_name=None):
    """
    provide source, and exact solutions when available, for:
    
    Find u in H(curl) such that
    
      A u = f             on \Omega
      n x u = n x u_bc    on \partial \Omega

    with

      A u := eta * u  +  mu * curl curl u  -  nu * grad div u
    
    see solve_hcurl_source_pbm()
    """

    # exact solutions (if available)
    u_ex = None
    curl_u_ex = None
    div_u_ex = None

    # bc solution: describe the bc on boundary. Inside domain, values should not matter. Homogeneous bc will be used if None
    u_bc = None

    # source terms
    f_vect = None
    
    # auxiliary term (for more diagnostics)
    grad_phi = None
    phi = None

    x,y    = domain.coordinates

    if source_type == 'manu_maxwell_inhom':
        # used for Maxwell equation with manufactured solution
        f_vect  = Tuple(eta*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                        eta*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))
        if nu == 0:
            u_ex  = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
            curl_u_ex = pi*(cos(pi*x)*cos(pi*y) - cos(pi*y))
            div_u_ex = -pi*sin(pi*x)*sin(pi*y)
        else:
            raise NotImplementedError
        u_bc = u_ex

    elif source_type == 'elliptic_J':
        # no manufactured solution for Maxwell pbm
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

        f_x =   dy_tau * phi
        f_y = - dx_tau * phi
        f_vect = Tuple(f_x, f_y)

    else:
        raise ValueError(source_type)

    # u_ex = Tuple(0, 1)  # DEBUG
    return f_vect, u_bc, u_ex, curl_u_ex, div_u_ex #, phi, grad_phi

def get_source_and_solution_h1(source_type=None, eta=0, mu=0,
                            domain=None, domain_name=None):
    """
    provide source, and exact solutions when available, for:
    
    Find u in H^1, such that

      A u = f             on \Omega
        u = u_bc          on \partial \Omega

    with

      A u := eta * u  -  mu * div grad u
    
    see solve_h1_source_pbm()
    """

    # exact solutions (if available)
    u_ex = None

    # bc solution: describe the bc on boundary. Inside domain, values should not matter. Homogeneous bc will be used if None
    u_bc = None

    # source terms
    f_scal = None

    # auxiliary term (for more diagnostics)
    grad_phi = None
    phi = None

    x,y    = domain.coordinates

    if source_type in ['manu_poisson_elliptic']:
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
        if mu == 1 and eta == 0:
            u_ex = phi
        else:
            print('WARNING (54375385643): exact solution not available in this case!') 

        if not domain_name in ['pretzel', 'pretzel_f']:
            # we may have non-hom bc's            
            u_bc = u_ex

    elif source_type == 'manu_poisson_2':
        f_scal = -4
        if mu == 1 and eta == 0:
            u_ex = x**2+y**2
        else:
            raise NotImplementedError
        u_bc   = u_ex

    elif source_type == 'manu_poisson_sincos':
        u_ex    = sin(pi*x)*cos(pi*y)
        f_scal  = (eta + 2*mu*pi**2) * u_ex
        u_bc = u_ex

    else:
        raise ValueError(source_type)

    return f_scal, u_bc, u_ex



def get_source_and_solution_OBSOLETE(source_type=None, eta=0, mu=0, nu=0,
                            domain=None, domain_name=None,
                            refsol_params=None):
    """
    OBSOLETE: kept for some test-cases
    """

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
        # same as manu_poisson_ellip, with arbitrary value for tor
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

        if source_type == 'manu_poisson' and mu == 1 and eta == 0:
            u_ex = phi

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

    return f_scal, f_vect, u_bc, p_ex, u_ex, phi, grad_phi


