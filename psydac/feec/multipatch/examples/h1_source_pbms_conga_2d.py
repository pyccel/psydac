from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict

from sympy import lambdify
from scipy.sparse.linalg import spsolve

from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral, Norm
from sympde.topology  import Derham
from sympde.topology import element_of


from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.feec.multipatch.api import discretize
from psydac.feec.pull_push      import pull_2d_h1

from psydac.feec.multipatch.fem_linear_operators        import IdLinearOperator
from psydac.feec.multipatch.operators                   import time_count, HodgeOperator
from psydac.feec.multipatch.plotting_utilities          import plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution

from psydac.linalg.utilities import array_to_stencil
from psydac.fem.basic        import FemField

def solve_h1_source_pbm(
        nc=4, deg=4, domain_name='pretzel_f', backend_language=None, source_proj='P_L2', source_type='manu_poisson',
        eta=-10., mu=1., gamma_h=10.,
        plot_source=False, plot_dir=None, hide_plots=True
):
    """
    solver for the problem: find u in H^1, such that

      A u = f             on \Omega
        u = u_bc          on \partial \Omega

    where the operator

      A u := eta * u  -  mu * div grad u

    is discretized as  Ah: V0h -> V0h  in a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \Omega,

      V0h  --grad->  V1h  -—curl-> V2h

    Examples:

      - Helmholtz equation with
          eta = -omega**2
          mu  = 1

      - Poisson equation with Laplace operator L = A,
          eta = 0
          mu  = 1

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma_h: jump penalization parameter
    :param source_proj: approximation operator for the source, possible values are 'P_geom' or 'P_L2'
    :param source_type: must be implemented in get_source_and_solution()
    """

    ncells = [nc, nc]
    degree = [deg,deg]

    # if backend_language is None:
    #     if domain_name in ['pretzel', 'pretzel_f'] and nc > 8:
    #         backend_language='numba'
    #     else:
    #         backend_language='python'
    # print('[note: using '+backend_language+ ' backends in discretize functions]')

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_h1_source_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    print(' domain_name = {}'.format(domain_name))
    print(' source_proj = {}'.format(source_proj))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    print('building the multipatch domain...')
    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    domain_h = discretize(domain, ncells=ncells)

    print('building the symbolic and discrete deRham sequences...')
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    # multi-patch (broken) spaces
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2
    print('dim(V0h) = {}'.format(V0h.nbasis))
    print('dim(V1h) = {}'.format(V1h.nbasis))
    print('dim(V2h) = {}'.format(V2h.nbasis))

    print('broken differential operators...')
    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    # bD1_m = bD1.to_sparse_matrix()

    print('building the discrete operators:')
    print('commuting projection operators...')
    nquads = [4*(d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    I0 = IdLinearOperator(V0h)
    I0_m = I0.to_sparse_matrix()

    print('Hodge operators...')
    # multi-patch (broken) linear operators / matrices
    H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language)
    H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language)

    dH0_m = H0.get_dual_Hodge_sparse_matrix()  # = mass matrix of V0
    H0_m  = H0.to_sparse_matrix()              # = inverse mass matrix of V0
    dH1_m = H1.get_dual_Hodge_sparse_matrix()  # = mass matrix of V1
    # H1_m  = H1.to_sparse_matrix()              # = inverse mass matrix of V1

    print('conforming projection operators...')
    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    cP0 = derham_h.conforming_projection(space='V0', hom_bc=True, backend_language=backend_language)
    cP0_m = cP0.to_sparse_matrix()
    # cP1 = derham_h.conforming_projection(space='V1', hom_bc=True, backend_language=backend_language)
    # cP1_m = cP1.to_sparse_matrix()

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    def lift_u_bc(u_bc):
        if u_bc is not None:
            print('lifting the boundary condition in V0h...  [warning: Not Tested Yet!]')
            # note: for simplicity we apply the full P1 on u_bc, but we only need to set the boundary dofs
            u_bc = lambdify(domain.coordinates, u_bc)
            u_bc_log = [pull_2d_h1(u_bc, m) for m in mappings_list]
            # it's a bit weird to apply P1 on the list of (pulled back) logical fields -- why not just apply it on u_bc ?
            uh_bc = P0(u_bc_log)
            ubc_c = uh_bc.coeffs.toarray()
            # removing internal dofs (otherwise ubc_c may already be a very good approximation of uh_c ...)
            ubc_c = ubc_c - cP0_m.dot(ubc_c)
        else:
            ubc_c = None
        return ubc_c

    # Conga (projection-based) stiffness matrices:
    # div grad:
    pre_DG_m = - bD0_m.transpose() @ dH1_m @ bD0_m

    # jump penalization:
    jump_penal_m = I0_m - cP0_m
    JP0_m = jump_penal_m.transpose() * dH0_m * jump_penal_m

    pre_A_m = cP0_m.transpose() @ ( eta * dH0_m - mu * pre_DG_m )  # useful for the boundary condition (if present)
    A_m = pre_A_m @ cP0_m + gamma_h * JP0_m

    print('getting the source and ref solution...')
    # (not all the returned functions are useful here)
    N_diag = 200
    method = 'conga'
    f_scal, f_vect, u_bc, ph_ref, uh_ref, p_ex, u_ex, phi, grad_phi = get_source_and_solution(
        source_type=source_type, eta=eta, mu=mu, domain=domain, domain_name=domain_name,
        refsol_params=[N_diag, method, source_proj],
    )

    # compute approximate source f_h
    b_c = f_c = None
    if source_proj == 'P_geom':
        print('projecting the source with commuting projection P0...')
        f = lambdify(domain.coordinates, f_scal)
        f_log = [pull_2d_h1(f, m) for m in mappings_list]
        f_h = P0(f_log)
        f_c = f_h.coeffs.toarray()
        b_c = dH0_m.dot(f_c)

    elif source_proj == 'P_L2':
        print('projecting the source with L2 projection...')
        v  = element_of(V0h.symbolic_space, name='v')
        expr = f_scal * v
        l = LinearForm(v, integral(domain, expr))
        lh = discretize(l, domain_h, V0h, backend=PSYDAC_BACKENDS[backend_language])
        b  = lh.assemble()
        b_c = b.toarray()
        if plot_source:
            f_c = H0_m.dot(b_c)
    else:
        raise ValueError(source_proj)

    if plot_source:
        plot_field(numpy_coeffs=f_c, Vh=V0h, space_kind='h1', domain=domain, title='f_h with P = '+source_proj, filename=plot_dir+'/fh_'+source_proj+'.png', hide_plot=hide_plots)

    ubc_c = lift_u_bc(u_bc)

    if ubc_c is not None:
        # modified source for the homogeneous pbm
        print('modifying the source with lifted bc solution...')
        b_c = b_c - pre_A_m.dot(ubc_c)

    # direct solve with scipy spsolve
    print('solving source problem with scipy.spsolve...')
    uh_c = spsolve(A_m, b_c)

    # project the homogeneous solution on the conforming problem space
    print('projecting the homogeneous solution on the conforming problem space...')
    uh_c = cP0_m.dot(uh_c)

    if ubc_c is not None:
        # adding the lifted boundary condition
        print('adding the lifted boundary condition...')
        uh_c += ubc_c

    print('getting and plotting the FEM solution from numpy coefs array...')
    title = r'solution $\phi_h$ (amplitude)'
    params_str = 'eta={}_mu={}_gamma_h={}'.format(eta, mu, gamma_h)
    plot_field(numpy_coeffs=uh_c, Vh=V0h, space_kind='h1', domain=domain, title=title, filename=plot_dir+params_str+'_phi_h.png', hide_plot=hide_plots)

if __name__ == '__main__':

    t_stamp_full = time_count()

    quick_run = True
    # quick_run = False

    omega = np.sqrt(170) # source
    roundoff = 1e4
    eta = int(-omega**2 * roundoff)/roundoff
    # print(eta)
    # source_type = 'elliptic_J'
    source_type = 'manu_poisson'

    # if quick_run:
    #     domain_name = 'curved_L_shape'
    #     nc = 4
    #     deg = 2
    # else:
    #     nc = 8
    #     deg = 4

    domain_name = 'pretzel_f'
    # domain_name = 'curved_L_shape'
    nc = 10
    deg = 2

    # nc = 2
    # deg = 2

    run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
    solve_h1_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        mu=1, #1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_source=True,
        plot_dir='./plots/h1_tests_source_february/'+run_dir,
        hide_plots=True,
    )

    time_count(t_stamp_full, msg='full program')
