#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
    solver for the problem: find u in H(curl), such that

      A u = f             on \\Omega
      n x u = n x u_bc    on \\partial \\Omega

    where the operator

      A u := eta * u  +  mu * curl curl u  -  nu * grad div u

    is discretized as  Ah: V1h -> V1h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h
"""

import os
import numpy as np

from sympde.topology import Derham

from psydac.api.discretization import discretize
from psydac.api.postprocessing import OutputManager, PostProcessManager

from psydac.feec.multipatch_domain_utilities import build_multipatch_domain

from psydac.fem.basic       import FemField
from psydac.fem.projectors  import get_dual_dofs

from psydac.linalg.basic       import IdentityOperator
from psydac.linalg.solvers     import inverse

#==============================================================================
# Solver for H(curl) source problems
#==============================================================================
def solve_hcurl_source_pbm(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_type='manu_maxwell_inhom',
        eta=-10., mu=1., nu=0., gamma_h=10.,
        project_sol=True, plot_dir=None):
    """
    solver for the problem: find u in H(curl), such that

      A u = f             on \\Omega
      n x u = n x u_bc    on \\partial \\Omega

    where the operator

      A u := eta * u  +  mu * curl curl u  -  nu * grad div u

    is discretized as  Ah: V1h -> V1h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \\Omega,

      V0h  --grad->  V1h  -—curl-> V2h

    Examples:

      - time-harmonic maxwell equation with
          eta = -omega**2
          mu  = 1
          nu  = 0

      - Hodge-Laplacian operator L = A with
          eta = 0
          mu  = 1
          nu  = 1

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma_h: jump penalization parameter
    :param source_type: must be implemented in get_source_and_solution()
    """
    degree = [deg, deg]


    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_hcurl_source_pbm function with: ')
    print(' ncells = {}'.format(nc))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    print()
    print(' -- building discrete spaces and operators  --')

    print(' .. multi-patch domain...')
    domain = build_multipatch_domain(domain_name=domain_name)
 
    if isinstance(nc, int):
        ncells = [nc, nc]
    else:
        ncells = {patch.name: [nc[i], nc[i]]
                    for (i, patch) in enumerate(domain.interior)}

    print(' .. derham sequence...')
    derham = Derham(domain, ["H1", "Hcurl", "L2"])

    print(' .. discrete domain...')
    domain_h = discretize(domain, ncells=ncells)

    print(' .. discrete derham sequence...')
    derham_h = discretize(derham, domain_h, degree=degree)

    print(' .. commuting projection operators...')
    nquads = [10 * (d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    print(' .. multi-patch spaces...')
    V0h, V1h, V2h = derham_h.spaces
    mappings = derham_h.callable_mapping

    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))


    print(' .. Id operator and matrix...')
    I1 = IdentityOperator(V1h.coeff_space)

    print(' .. Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    # other option: define as Hodge Operators:
    H0, H1, H2 = derham_h.hodge_operators(kind='linop', backend_language=backend_language)
    dH0, dH1, dH2 = derham_h.hodge_operators(kind='linop', dual=True, backend_language=backend_language)

    print(' .. conforming Projection operators...')
    # conforming Projections (should take into account the boundary conditions
    # of the continuous deRham sequence)
    cP0, cP1, cP2 = derham_h.conforming_projectors(kind='linop', hom_bc = True)

    print(' .. broken differential operators...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.derivatives(kind='linop')

    # Conga (projection-based) stiffness matrices
    # curl curl:
    print(' .. curl-curl stiffness matrix...')
    pre_CC = bD1.T @ H2 @ bD1

    # grad div:
    print(' .. grad-div stiffness matrix...')
    pre_GD = - H1 @ bD0 @ cP0 @ dH0 @ cP0.T @ bD0.T @ H1

    # jump stabilization:
    print(' .. jump stabilization matrix...')
    JS = (I1 - cP1).T @ H1 @ (I1 - cP1)

    print(' .. full operator matrix...')
    print('eta = {}'.format(eta))
    print('mu = {}'.format(mu))
    print('nu = {}'.format(nu))
    print('STABILIZATION: gamma_h = {}'.format(gamma_h))
    # useful for the boundary condition (if present)
    pre_A = eta * cP1.T @  H1 
    if mu != 0: 
        pre_A += mu * cP1.T @ pre_CC
    if nu != 0: 
        pre_A -= nu * cP1.T @ pre_GD

    A = pre_A @ cP1 + gamma_h * JS

    print()
    print(' -- getting source --')

    f_vect, u_bc, u_ex = get_source_and_solution_hcurl(source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,)

    # compute approximate source f_h
    # f_h = L2 projection of f_vect, with filtering if tilde_Pi
    tilde_f = get_dual_dofs(Vh=V1h, f=f_vect, domain_h=domain_h, backend_language=backend_language)

    print(' .. filtering the discrete source with P1.T ...')
    tilde_f = cP1.T @ tilde_f

    def lift_u_bc(u_bc):
        if u_bc is not None:
            ubc = P1(u_bc).coeffs
            ubc -= cP1.dot(ubc)

        else:
            ubc = None
            
        return ubc

    ubc = lift_u_bc(u_bc)

    if ubc is not None:
        # modified source for the homogeneous pbm
        print(' .. modifying the source with lifted bc solution...')
        tilde_f -= pre_A.dot(ubc)

    # direct solve with scipy spsolve
    print('solving source problem with conjugate gradient...')
    solver = inverse(A, solver='cg', tol=1e-8)
    u = solver.solve(tilde_f)

    # project the homogeneous solution on the conforming problem space
    if project_sol:
        print(' .. projecting the homogeneous solution on the conforming problem space...')
        u = cP1.dot(u)

    if ubc is not None:
        # adding the lifted boundary condition
        print(' .. adding the lifted boundary condition...')
        u += ubc

    uh = FemField(V1h, coeffs=u)
    f = dH1.dot(tilde_f)
    jh = FemField(V1h, coeffs=f)

    print(' -- plots and diagnostics  --')
    if plot_dir:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        OM = OutputManager(plot_dir + '/spaces.yml', plot_dir + '/fields.h5')
        OM.add_spaces(V1h=V1h)
        OM.set_static()
        OM.export_fields(vh=uh)
        OM.export_fields(jh=jh)
        OM.export_space_info()
        OM.close()

        PM = PostProcessManager(
            domain=domain,
            space_file=plot_dir +
            '/spaces.yml',
            fields_file=plot_dir +
            '/fields.h5')
        PM.export_to_vtk(
            plot_dir + "/sol",
            grid=None,
            npts_per_cell=[6] * 2,
            snapshots='all',
            fields='vh')
        PM.export_to_vtk(
            plot_dir + "/source",
            grid=None,
            npts_per_cell=[6] * 2,
            snapshots='all',
            fields='jh')

        PM.close()


    if u_ex:
        u_ex_p = P1(u_ex).coeffs
    
        err = u_ex_p - u
        l2_error = np.sqrt( H1.dot_inner(err, err) / H1.dot_inner(u_ex_p, u_ex_p))
        print("L2 error: ", l2_error)

        return l2_error

#==============================================================================
# Test sources and exact solutions
#==============================================================================
def get_source_and_solution_hcurl(
        source_type=None, eta=0, mu=0, nu=0,
        domain=None, domain_name=None):
    """
    provide source, and exact solutions when available, for:

    Find u in H(curl) such that

      A u = f             on \\Omega
      n x u = n x u_bc    on \\partial \\Omega

    with

      A u := eta * u  +  mu * curl curl u  -  nu * grad div u

    see solve_hcurl_source_pbm()
    """
    from sympy import pi, cos, sin, Tuple, exp

    # exact solutions (if available)
    u_ex = None

    # bc solution: describe the bc on boundary. Inside domain, values should
    # not matter. Homogeneous bc will be used if None
    u_bc = None

    # source terms
    f_vect = None

    # auxiliary term (for more diagnostics)
    grad_phi = None
    phi = None

    x, y = domain.coordinates

    if source_type == 'manu_maxwell_inhom':
        # used for Maxwell equation with manufactured solution
        f_vect = Tuple(eta * sin(pi * y) - pi**2 * sin(pi * y) * cos(pi * x) + pi**2 * sin(pi * y),
                       eta * sin(pi * x) * cos(pi * y) + pi**2 * sin(pi * x) * cos(pi * y))
        if nu == 0:
            u_ex = Tuple(sin(pi * y), sin(pi * x) * cos(pi * y))
            curl_u_ex = pi * (cos(pi * x) * cos(pi * y) - cos(pi * y))
            div_u_ex = -pi * sin(pi * x) * sin(pi * y)
        else:
            raise NotImplementedError
        u_bc = u_ex

    elif source_type == 'elliptic_J':
        # no manufactured solution for Maxwell pbm
        x0 = 1.5
        y0 = 1.5
        s = (x - x0) - (y - y0)
        t = (x - x0) + (y - y0)
        a = (1 / 1.9)**2
        b = (1 / 1.2)**2
        sigma2 = 0.0121
        tau = a * s**2 + b * t**2 - 1
        phi = exp(-tau**2 / (2 * sigma2))
        dx_tau = 2 * (a * s + b * t)
        dy_tau = 2 * (-a * s + b * t)

        f_x = dy_tau * phi
        f_y = - dx_tau * phi
        f_vect = Tuple(f_x, f_y)

    else:
        raise ValueError(source_type)

    from sympy import lambdify
    u_bc_x = lambdify(domain.coordinates, u_bc[0])
    u_bc_y = lambdify(domain.coordinates, u_bc[1])

    u_ex_x = lambdify(domain.coordinates, u_ex[0])
    u_ex_y = lambdify(domain.coordinates, u_ex[1])

    return f_vect, [u_bc_x, u_bc_y], [u_ex_x, u_ex_y]

if __name__ == '__main__':
    nc = 5
    deg = 3

    source_type = 'manu_maxwell_inhom'
    domain_name = 'pretzel_f'

    omega = np.pi
    eta = -omega**2  # source

    err = solve_hcurl_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc')
        