from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict

from sympy import pi, cos, sin, Tuple, exp
from sympy import lambdify

from sympde.expr     import TerminalExpr
from sympde.calculus import grad, dot, inner, rot, div, curl, cross
from sympde.calculus import minus, plus
from sympde.topology import NormalVector
from sympde.expr     import Norm

from sympde.topology import element_of, elements_of, Domain

from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral


from scipy.sparse.linalg import spsolve, spilu, cg, lgmres
from scipy.sparse.linalg import LinearOperator, eigsh, minres, gmres

from scipy.sparse.linalg import inv
from scipy.sparse.linalg import norm as spnorm
from scipy.linalg        import eig, norm
from scipy.sparse import save_npz, load_npz, bmat

from sympde.topology import Derham
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping



from psydac.fem.basic   import FemField

from psydac.api.settings import PSYDAC_BACKENDS

from psydac.feec.multipatch.api import discretize
from psydac.feec.pull_push import pull_2d_h1, pull_2d_hcurl, push_2d_hcurl, push_2d_l2

from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator
from psydac.feec.multipatch.operators import time_count, HodgeOperator
from psydac.feec.multipatch.plotting_utilities import plot_field
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, my_small_plot, my_small_streamplot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

from psydac.feec.multipatch.utils_conga_2d import rhs_fn, sol_ref_fn, hf_fn, error_fn, get_method_name, get_fem_name, get_load_dir

comm = MPI.COMM_WORLD

def solve_source_pbm(nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_proj='P_geom', source_type='manu_J',
                     eta=-10, mu=1, nu=1, gamma_h=10,
                     plot_source=False, plot_dir=None, hide_plots=True):
    """
    solver for the problem: find u in H(curl), such that

      A u = f             on \Omega
      n x u = n x u_bc    on \partial \Omega

    where the operator

      A u := eta * u  +  mu * curl curl u  -  nu * grad div u

    is discretized as  Ah: V1h -> V1h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \Omega,

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
    :param source_proj: approximation operator for the source, possible values are 'P_geom' or 'P_L2'
    :param source_type: must be implemented in get_source_and_solution()
    """

    ncells = [nc, nc]
    degree = [deg,deg]

    if backend_language is None:
        if domain_name in ['pretzel', 'pretzel_f'] and nc > 8:
            backend_language='numba'
        else:
            backend_language='python'
    print('[note: using '+backend_language+ ' backends in discretize functions]')


    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_source_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' source_proj = {}'.format(source_proj))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    t_stamp = time_count()
    print('building symbolic domain sequence...')
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    t_stamp = time_count(t_stamp)
    print('building derham sequence...')
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    t_stamp = time_count(t_stamp)
    print('building discrete domain...')
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    t_stamp = time_count(t_stamp)
    print('building discrete derham sequence...')
    derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    t_stamp = time_count(t_stamp)
    print('building commuting projection operators...')
    nquads = [4*(d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    # multi-patch (broken) spaces
    t_stamp = time_count(t_stamp)
    print('calling the multi-patch spaces...')
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))

    t_stamp = time_count(t_stamp)
    print('building the Id operator and matrix...')
    I1 = IdLinearOperator(V1h)
    I1_m = I1.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print('instanciating the Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    # other option: define as Hodge Operators:
    H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language)
    H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language)
    H2 = HodgeOperator(V2h, domain_h, backend_language=backend_language)

    t_stamp = time_count(t_stamp)
    print('building the dual Hodge matrix dH0_m = M0_m ...')
    dH0_m = H0.get_dual_Hodge_sparse_matrix()  # = mass matrix of V0

    t_stamp = time_count(t_stamp)
    print('building the primal Hodge matrix H0_m = inv_M0_m ...')
    H0_m  = H0.to_sparse_matrix()              # = inverse mass matrix of V0

    t_stamp = time_count(t_stamp)
    print('building the dual Hodge matrix dH1_m = M1_m ...')
    dH1_m = H1.get_dual_Hodge_sparse_matrix()  # = mass matrix of V1

    t_stamp = time_count(t_stamp)
    print('building the primal Hodge matrix H1_m = inv_M1_m ...')
    H1_m  = H1.to_sparse_matrix()              # = inverse mass matrix of V1

    # print("dH1_m @ H1_m == I1_m: {}".format(np.allclose((dH1_m @ H1_m).todense(), I1_m.todense())) )   # CHECK: OK

    t_stamp = time_count(t_stamp)
    print('building the dual Hodge matrix dH2_m = M2_m ...')
    dH2_m = H2.get_dual_Hodge_sparse_matrix()  # = mass matrix of V2

    t_stamp = time_count(t_stamp)
    print('building the conforming Projection operators and matrices...')
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    cP0 = derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language)
    cP1 = derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language)
    cP0_m = cP0.to_sparse_matrix()
    cP1_m = cP1.to_sparse_matrix()

    t_stamp = time_count(t_stamp)
    print('building the broken differential operators and matrices...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    def lift_u_bc(u_bc):
        if u_bc is not None:
            # note: for simplicity we apply the full P1 on u_bc, but we only need to set the boundary dofs
            u_bc_x = lambdify(domain.coordinates, u_bc[0])
            u_bc_y = lambdify(domain.coordinates, u_bc[1])
            u_bc_log = [pull_2d_hcurl([u_bc_x, u_bc_y], m) for m in mappings_list]
            # it's a bit weird to apply P1 on the list of (pulled back) logical fields -- why not just apply it on u_bc ?
            uh_bc = P1(u_bc_log)
            ubc_c = uh_bc.coeffs.toarray()
            # removing internal dofs (otherwise ubc_c may already be a very good approximation of uh_c ...)
            ubc_c = ubc_c - cP1_m.dot(ubc_c)
        else:
            ubc_c = None
        return ubc_c

    # Conga (projection-based) stiffness matrices
    # curl curl:
    t_stamp = time_count(t_stamp)
    print('computing the curl-curl stiffness matrix...')
    pre_CC_m = bD1_m.transpose() @ dH2_m @ bD1_m
    # CC_m = cP1_m.transpose() @ pre_CC_m @ cP1_m  # Conga stiffness matrix

    # grad div:
    t_stamp = time_count(t_stamp)
    print('computing the grad-div stiffness matrix...')
    pre_GD_m = - dH1_m @ bD0_m @ cP0_m @ H0_m @ cP0_m.transpose() @ bD0_m.transpose() @ dH1_m
    # GD_m = cP1_m.transpose() @ pre_GD_m @ cP1_m  # Conga stiffness matrix

    # jump penalization:
    t_stamp = time_count(t_stamp)
    print('computing the jump penalization matrix...')
    jump_penal_m = I1_m - cP1_m
    JP_m = jump_penal_m.transpose() * dH1_m * jump_penal_m

    t_stamp = time_count(t_stamp)
    print('computing the full operator matrix...')
    print('eta = {}'.format(eta))
    print('mu = {}'.format(mu))
    print('nu = {}'.format(nu))
    pre_A_m = cP1_m.transpose() @ ( eta * dH1_m + mu * pre_CC_m - nu * pre_GD_m )  # useful for the boundary condition (if present)
    A_m = pre_A_m @ cP1_m + gamma_h * JP_m

    # get exact source, bcs, ref solution...
    # note: design of source and solution should also be thought over -- here I'm only copying old function from electromag_pbms.py
    t_stamp = time_count(t_stamp)
    print('getting the source and ref solution...')
    N_diag = 200
    method = 'conga'
    f_scal, f_vect, u_bc, ph_ref, uh_ref, p_ex, u_ex, phi, grad_phi = get_source_and_solution(
        source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,
        refsol_params=[N_diag, method, source_proj],
    )

    # compute approximate source f_h
    t_stamp = time_count(t_stamp)
    b_c = f_c = None
    if source_proj == 'P_geom':
        # f_h = P1-geometric (commuting) projection of f_vect
        print('projecting the source with commuting projection...')
        f_x = lambdify(domain.coordinates, f_vect[0])
        f_y = lambdify(domain.coordinates, f_vect[1])
        f_log = [pull_2d_hcurl([f_x, f_y], m) for m in mappings_list]
        f_h = P1(f_log)
        f_c = f_h.coeffs.toarray()
        b_c = dH1_m.dot(f_c)

    elif source_proj == 'P_L2':
        # f_h = L2 projection of f_vect
        print('projecting the source with L2 projection...')
        v  = element_of(V1h.symbolic_space, name='v')
        expr = dot(f_vect,v)
        l = LinearForm(v, integral(domain, expr))
        lh = discretize(l, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])
        b  = lh.assemble()
        b_c = b.toarray()
        if plot_source:
            f_c = H1_m.dot(b_c)
    else:
        raise ValueError(source_proj)

    if plot_source:
        plot_field(numpy_coeffs=f_c, Vh=V1h, space_kind='hcurl', domain=domain, title='f_h with P = '+source_proj, filename=plot_dir+'/fh_'+source_proj+'.png', hide_plot=hide_plots)

    ubc_c = lift_u_bc(u_bc)

    if ubc_c is not None:
        # modified source for the homogeneous pbm
        t_stamp = time_count(t_stamp)
        print('modifying the source with lifted bc solution...')
        b_c = b_c - pre_A_m.dot(ubc_c)

    # direct solve with scipy spsolve
    t_stamp = time_count(t_stamp)
    print('solving source problem with scipy.spsolve...')
    uh_c = spsolve(A_m, b_c)

    # project the homogeneous solution on the conforming problem space
    t_stamp = time_count(t_stamp)
    print('projecting the homogeneous solution on the conforming problem space...')
    uh_c = cP1_m.dot(uh_c)

    if ubc_c is not None:
        # adding the lifted boundary condition
        t_stamp = time_count(t_stamp)
        print('adding the lifted boundary condition...')
        uh_c += ubc_c

    t_stamp = time_count(t_stamp)
    print('getting and plotting the FEM solution from numpy coefs array...')
    title = r'solution $u_h$ (amplitude) for $\eta = $'+repr(eta)
    params_str = 'eta={}'.format(eta) + '_mu={}'.format(mu) + '_nu={}'.format(nu)+ '_gamma_h={}'.format(gamma_h)
    plot_field(numpy_coeffs=uh_c, Vh=V1h, space_kind='hcurl', domain=domain, title=title, filename=plot_dir+params_str+'_uh.png', hide_plot=hide_plots)

    time_count(t_stamp)



def get_source_and_solution(source_type=None, eta=0, mu=0, nu=0,
                            domain=None, domain_name=None,
                            refsol_params=None):
    """
    compute source and reference solution (exact, or reference values) when possible, depending on the source_type
    """

    assert refsol_params
    N_diag, method_ref, source_proj_ref = refsol_params

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
        alpha   = eta
        u_ex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
        f_vect  = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                        alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))
        u_bc = u_ex
    elif source_type in ['manu_poisson', 'ellnew_J']:

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

        f_x =   dy_tau * phi
        f_y = - dx_tau * phi
        f_vect = Tuple(f_x, f_y)


    elif source_type == 'dipcurl_J':
        # here, f will be the curl of a dipole + phi_0 - phi_1 (two blobs) that correspond to a scalar current density
        # the solution of the curl-curl problem with free-divergence constraint
        #   curl curl u = curl j
        #
        # then corresponds to a magnetic density,
        # see Beirão da Veiga, Brezzi, Dassi, Marini and Russo, Virtual Element approx of 2D magnetostatic pbms, CMAME 327 (2017)

        x_0 = 1.0
        y_0 = 1.0
        # x_0 = 0.3
        # y_0 = 2.7
        # x_0 = -0.7
        # y_0 = 2.3
        ds2_0 = (0.02)**2
        sigma_0 = (x-x_0)**2 + (y-y_0)**2
        phi_0 = exp(-sigma_0**2/(2*ds2_0))
        dx_sig_0 = 2*(x-x_0)
        dy_sig_0 = 2*(y-y_0)
        dx_phi_0 = - dx_sig_0 * sigma_0 / ds2_0 * phi_0
        dy_phi_0 = - dy_sig_0 * sigma_0 / ds2_0 * phi_0

        x_1 = 2.0
        y_1 = 2.0
        # x_1 = 0.7
        # y_1 = 2.3
        # x_1 = -0.3
        # y_1 = 2.7
        ds2_1 = (0.02)**2
        sigma_1 = (x-x_1)**2 + (y-y_1)**2
        phi_1 = exp(-sigma_1**2/(2*ds2_1))
        dx_sig_1 = 2*(x-x_1)
        dy_sig_1 = 2*(y-y_1)
        dx_phi_1 = - dx_sig_1 * sigma_1 / ds2_1 * phi_1
        dy_phi_1 = - dy_sig_1 * sigma_1 / ds2_1 * phi_1

        f_x =   dy_phi_0 - dy_phi_1
        f_y = - dx_phi_0 + dx_phi_1
        f_scal = phi_0 - phi_1
        f_vect = Tuple(f_x, f_y)

    elif source_type == 'ellip_J':

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

    assert f_vect is not None
    if u_ex is None:
        u_ref_filename = get_load_dir(method=method_ref, domain_name=domain_name,nc=None,deg=None,data='solutions')+sol_ref_fn(source_type, N_diag, source_proj=source_proj_ref)
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

    return f_scal, f_vect, u_bc, ph_ref, uh_ref, p_ex, u_ex, phi, grad_phi


if __name__ == '__main__':

    t_stamp_full = time_count()

    quick_run = True
    # quick_run = False

    omega = np.sqrt(170) # source
    # source_type = 'ellnew_J'
    source_type = 'manu_J'

    if quick_run:
        domain_name = 'curved_L_shape'
        nc = 4
        deg = 2
    else:
        nc = 8
        deg = 4

    domain_name = 'pretzel_f'
    domain_name = 'curved_L_shape'
    # nc = 8
    deg = 2

    nc = 2
    # deg = 2

    solve_source_pbm(
        nc=nc, deg=deg,
        eta=-omega**2,
        nu=0,
        mu=1, #1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='numba',
        plot_source=True,
        plot_dir='./plots/tests_source_february/',
        hide_plots=True,
    )

    time_count(t_stamp_full, msg='full program')