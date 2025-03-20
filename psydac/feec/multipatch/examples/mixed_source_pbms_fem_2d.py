from mpi4py import MPI

import os
import numpy as np
from collections import OrderedDict

from sympy import lambdify
from sympy  import pi, sin, cos, Tuple, Matrix

from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from sympde.calculus import grad, div, curl, dot, inner
from sympde.calculus import minus, plus, dot, cross
from sympde.topology import NormalVector
from sympde.topology import element_of
from sympde.expr.expr import BilinearForm, LinearForm
from sympde.expr.expr import integral
from sympde.topology import Derham
from sympde.expr.equation import find

from sympde.topology import VectorFunctionSpace, ScalarFunctionSpace, VectorFunction

from psydac.api.settings import PSYDAC_BACKENDS

from psydac.feec.pull_push import pull_2d_h1, pull_2d_hcurl, pull_2d_l2

from psydac.feec.multipatch.api import discretize
from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator
from psydac.feec.multipatch.operators import HodgeOperator
from psydac.fem.plotting_utilities import plot_field_2d as plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.multipatch_domain_utilities import build_cartesian_multipatch_domain

from psydac.feec.multipatch.examples.ppc_test_cases import get_source_and_sol_for_magnetostatic_pbm
from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d import get_eigenvalues
from psydac.feec.multipatch.utilities import time_count

from psydac.feec.multipatch.non_matching_operators import construct_h1_conforming_projection, construct_hcurl_conforming_projection


def first_eigenmodes_hlap(
        nc=4, deg=4, domain=None,
        method_type='H1_fem',
        backend_language=None,
        bc_type='H0curl',
        gamma0_h=10., gamma1_h=10.,
        nb_eigenmodes=0,
        project_solution=False,
        plot_eigenmodes=False,
        plot_source=False,
        plot_dir=None, hide_plots=True,
        m_load_dir="",
):
    """
    computes the lowest eigenmodes of vector laplacian operator with H1 fem and strong differential operators

    Note: the harmonic forms (H2 in inhom sequence, or H1 in hom sequence) are
        H = {div u = curl u = 0, and nxu = 0 on B}

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma0_h: jump penalization parameter in V0h
    :param gamma1_h: jump penalization parameter in V1h
    :param bc_type: 'H0curl' or 'H0div'
    :param m_load_dir: directory for matrix storage
    """

    ncells = [nc, nc]
    degree = [deg, deg]

    # if backend_language is None:
    #     backend_language='python'
    # print('[note: using '+backend_language+ ' backends in discretize functions]')
    assert bc_type in ['H0curl', 'H0div']
    assert nb_eigenmodes > 0

    print('---------------------------------------------------------------------------------------------------------')
    print('Starting solve_mixed_source_pbm function with: ')
    print(' ncells = {}'.format(ncells))
    print(' degree = {}'.format(degree))
    # print(' domain_name = {}'.format(domain_name))
    # print(' source_proj = {}'.format(source_proj))
    print(' bc_type = {}'.format(bc_type))
    print(' backend_language = {}'.format(backend_language))
    print('---------------------------------------------------------------------------------------------------------')

    print('building symbolic and discrete domain...')
    # domain = build_multipatch_domain(domain_name=domain_name)

    mappings = OrderedDict([(P.logical_domain, P.mapping)
                        for P in domain.interior])
    mappings_list = list(mappings.values())
    domain_h = discretize(domain, ncells=ncells)

    if method_type == 'H1_fem':
        V = VectorFunctionSpace('V', domain, kind='h1')

        x,y = domain.coordinates

        v = element_of(V, name='v')
        u = element_of(V, name='u')

        nn = NormalVector("nn")

        I = domain.interfaces
        B = domain.boundary

        kappa  = 10**3

        avr    = lambda u:0.5*plus(u)+0.5*minus(u)
        jump   = lambda u: minus(u)-plus(u)

        expr = (curl(v) * curl(u)) + (div(v) * div(u))  # doesn't work
        # expr = inner(grad(v), grad(u))
        expr_I = kappa * dot(jump(u), jump(v))

        if bc_type == 'H0curl':
            expr_B = kappa * (cross(nn, u)*cross(nn, v) + div(u)*div(v))
        else:
            print('WARNING: curl is not implemented yet so we only impose n.u = 0')
            expr_B = kappa * (dot(nn, u)*dot(nn, v))
            # print(f'type(u) = {type(u)}')
            # print(f'u[0] = {u[0]}, type(u[0]) = {type(u[0])}, ')
            # print(f'type(rotate(u)) = {type(rotate(u))}')
            # expr_B = kappa * (dot(nn, u)*dot(nn, v) + curl(u)*curl(v))        

        a = BilinearForm((v,u), integral(domain, expr) + integral(I, expr_I) + integral(B, expr_B)) #, check_linearity=False)

        ## Harmonic forms = kernel of A

        aM = BilinearForm((v,u), integral(domain, dot(u,v)))

        # Vh = discretize(V, domain_h)
        # domain_h = discretize(domain, ncells=ncells, comm=comm)
        print('discretizing the space...')
        Vh       = discretize(V, domain_h, degree=degree) #, basis='M')

        # coeffs = np.arange(Vh.nbasis)
        # plot_field(numpy_coeffs=coeffs, Vh=Vh, space_kind='h1', plot_type='vector_field', domain=domain, title='some u',
        #     filename=plot_dir + f'some_u.png', hide_plot=hide_plots)

        # exit()

        # Discrete bilinear forms
        print('discretizing the bilinear form a...')
        nquads = [deg + 1, deg + 1]
        a_h = discretize(a, domain_h, [Vh, Vh], nquads=nquads, backend=PSYDAC_BACKENDS[backend_language])

        # print(domain_h)
        # expr = dot(f, v)
        # f = Tuple(1., 0.)
        # l = LinearForm(v, integral(domain, dot(f,v)))
        # equation = find(u, forall=v, lhs=a(u,v), rhs=l(v))
        # equation_h = discretize(equation, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[backend_language])
        # a_h = equation_h.lhs

        print('discretizing the bilinear form aM...')
        aM_h = discretize(aM, domain_h, [Vh, Vh], nquads=nquads, backend=PSYDAC_BACKENDS[backend_language])
        A = a_h.assemble()
        A_m = A.tosparse()

        M = aM_h.assemble()
        M_m = M.tosparse()

        push_kind = 'h1'

    else:
        print('--* solving with feec *--')

        print('building symbolic and discrete derham sequences...')

        print('using grad -> curl sequence')
        derham = Derham(domain, ["H1", "Hcurl", "L2"])
        push_kind = 'hcurl'

        hom_bc = (bc_type == 'H0curl')
        print('with hom_bc = {}'.format(hom_bc))

        # if :
        #     hom_bc = True #(bc_type == 'pseudo-vacuum')  # /!\  here u = B is in H(curl), not E  /!\

        #     # print('using grad -> curl sequence')
        #     # derham = Derham(domain, ["H1", "Hcurl", "L2"])
        #     # push_kind = 'hcurl'
        # else:
            # print('using curl -> div sequence')
            # derham = Derham(domain, ["H1", "Hdiv", "L2"])
            # push_kind = 'hdiv'

        derham_h = discretize(derham, domain_h, degree=degree)

        V0h = derham_h.V0
        V1h = derham_h.V1
        V2h = derham_h.V2
        print('dim(V0h) = {}'.format(V0h.nbasis))
        print('dim(V1h) = {}'.format(V1h.nbasis))
        print('dim(V2h) = {}'.format(V2h.nbasis))

        print('building the discrete operators:')
        print('commuting projection operators...')
        nquads = [4 * (d + 1) for d in degree]
        P0, P1, P2 = derham_h.projectors(nquads=nquads)

        I0_m = IdLinearOperator(V0h).to_sparse_matrix()
        I1_m = IdLinearOperator(V1h).to_sparse_matrix()

        print('Hodge operators...')
        # multi-patch (broken) linear operators / matrices
        H0 = HodgeOperator(
            V0h,
            domain_h,
            backend_language=backend_language,
            load_dir=m_load_dir,
            load_space_index=0)
        H1 = HodgeOperator(
            V1h,
            domain_h,
            backend_language=backend_language,
            load_dir=m_load_dir,
            load_space_index=1)
        H2 = HodgeOperator(
            V2h,
            domain_h,
            backend_language=backend_language,
            load_dir=m_load_dir,
            load_space_index=2)

        H0_m = H0.to_sparse_matrix()                # = mass matrix of V0
        dH0_m = H0.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V0
        H1_m = H1.to_sparse_matrix()                # = mass matrix of V1
        dH1_m = H1.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V1
        H2_m = H2.to_sparse_matrix()                # = mass matrix of V2
        dH2_m = H2.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V2

        M0_m = H0_m
        M1_m = H1_m  # usual notation

        print('conforming projection operators...')
        # conforming Projections (should take into account the boundary conditions
        # of the continuous deRham sequence)
        cP0_m = construct_h1_conforming_projection(V0h, hom_bc=hom_bc)
        cP1_m = construct_hcurl_conforming_projection(V1h, hom_bc=hom_bc)

        print('broken differential operators...')
        bD0, bD1 = derham_h.broken_derivatives_as_operators
        bD0_m = bD0.to_sparse_matrix()
        bD1_m = bD1.to_sparse_matrix()

        # Conga (projection-based) operator matrices
        print('grad [or curl] matrix...')
        G_m = bD0_m @ cP0_m
        tG_m = H1_m @ G_m  # grad: V0h -> tV1h

        print('curl-curl [or grad-div] stiffness matrix...')
        C_m = bD1_m @ cP1_m
        CC_m = C_m.transpose() @ H2_m @ C_m

        # jump penalization and stabilization operators:
        JP0_m = I0_m - cP0_m
        S0_m = JP0_m.transpose() @ H0_m @ JP0_m

        JP1_m = I1_m - cP1_m
        S1_m = JP1_m.transpose() @ H1_m @ JP1_m

        if not hom_bc:
            # very small regularization to avoid constant p=1 in the kernel
            reg_S0_m = 1e-16 * M0_m + gamma0_h * S0_m
        else:
            reg_S0_m = gamma0_h * S0_m


        gamma_Lh = 10  # penalization value should not change the kernel

        GD_m = - tG_m @ dH0_m @ G_m.transpose() @ H1_m   # todo: check with paper
        L_m = CC_m - GD_m + gamma_Lh * S1_m

        A_m = L_m
        M_m = H1_m
        Vh = V1h

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    hf_cs = []
    print(f'computing the {nb_eigenmodes} first eigenmodes...')
    gamma_Lh = 10  # penalization value should not change the kernel

    eigenvalues, eigenvectors = get_eigenvalues(
        nb_eigenmodes + 1, 1e-6, A_m, M_m)

    for i in range(nb_eigenmodes):
        lambda_i = eigenvalues[i]
        print(
            ".. storing eigenmode #{}, with eigenvalue = {}".format(
                i, lambda_i))
        # check:
        if abs(lambda_i) > 1e-8:
            print(" ****** WARNING! this eigenvalue should be 0!   ****** ")
        hf_cs.append(eigenvectors[:, i])

        if plot_eigenmodes:
            plot_field(numpy_coeffs=eigenvectors[:, i], Vh=Vh, space_kind=push_kind, plot_type='vector_field', domain=domain, title='eigenmode {0} with lambda_{0} = {1:.4f}'.format(i,lambda_i),
                filename=plot_dir + f'eigmode_{i}_vf.png', hide_plot=hide_plots)

            plot_field(numpy_coeffs=eigenvectors[:, i], Vh=Vh, space_kind=push_kind, plot_type='amplitude', domain=domain, title='eigenmode {0} with lambda_{0} = {1:.4f}'.format(i,lambda_i),
                filename=plot_dir + f'eigmode_{i}.png', hide_plot=hide_plots)

if __name__ == '__main__':

    t_stamp_full = time_count()

    bc_type = 'H0curl'
    bc_type = 'H0div'

    method_type = 'H1_fem'
    # method_type = 'feec'


    # nc = 20
    # deg = 4
    nc = 10
    deg = 2

    # domain_name = 'square_mp'
    # domain_name = 'square'
    # domain_name = 'pretzel_f'
    domain_name = 'annulus_4'

    nb_eigenmodes = 4

    #
    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    backend_language = 'python' # 'pyccel-gcc'

    print(f' .. building domain {domain_name}..')
    if domain_name in ['pretzel_f', 'annulus_4']:
        domain = build_multipatch_domain(domain_name=domain_name)
    elif domain_name == 'square_mp':
        int_x = [0, np.pi]
        int_y = [0, np.pi]

        square_with_hole = False
        if square_with_hole:
            ncells_patch_grid = np.array([
                [2,  2,    2],
                [2,  None, 2],
                [2,  2,    2]
                ])
        else:
            ncells_patch_grid = np.array([
                [4, 4],
                [4, 4]
                ])

        domain = build_cartesian_multipatch_domain(ncells_patch_grid, int_x, int_y, mapping='identity')
    run_dir = f'{method_type}_{bc_type}_{domain_name}_nc={nc}_deg={deg}/'
    # m_load_dir = 'matrices_{}_nc={}_deg={}/'.format(domain_name, nc, deg)
    first_eigenmodes_hlap(
        nc=nc, deg=deg,
        domain=domain,
        method_type=method_type,
        bc_type=bc_type,
        backend_language='pyccel-gcc',
        nb_eigenmodes=nb_eigenmodes,
        # plot_source=True,
        plot_eigenmodes=True,
        plot_dir='./plots/first_eigenmodes/' + run_dir,
        hide_plots=True,
        # m_load_dir=m_load_dir
    )

    time_count(t_stamp_full, msg='full program')
