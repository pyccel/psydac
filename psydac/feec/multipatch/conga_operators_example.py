# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np

from collections import OrderedDict

from sympde.topology import Derham
from psydac.feec.multipatch.api import discretize

from psydac.feec.pull_push     import pull_2d_h1, pull_2d_hcurl, pull_2d_l2
from psydac.utilities.utils    import refine_array_1d

from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.feec.multipatch.fem_linear_operators import ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, get_K0_and_K0_inv, get_K1_and_K1_inv
from psydac.feec.multipatch.operators import ConformingProjection_V0, ConformingProjection_V1
from psydac.feec.multipatch.operators import time_count
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_patch_knots_gridlines, my_small_plot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

comm = MPI.COMM_WORLD



def conga_operators_2d():
    """
    - assembles several multipatch operators and a conforming projection

    - performs several tests:
      - ...
      -

    """

    r_min = 0.5
    r_max = 1
    # domain_name = 'pretzel'
    # domain_name = 'pretzel_f'
    # domain_name = 'curved_L_shape'
    domain_name = 'square_8'

    domain = build_multipatch_domain(domain_name=domain_name, r_min=r_min, r_max=r_max)

    print("nb of patches: ", len(domain.interior))
    print("interiors (patches): ", domain.interior)
    print("nb of boundaries: ", len(domain.boundary))
    print("boundaries: ", domain.boundary)
    print("nb of interfaces: ", len(domain.interfaces))
    print("interfaces: ", domain.interfaces)

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])

    mappings_list = list(mappings.values())

    F = [f.get_callable_mapping() for f in mappings_list]

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    only_cP_check = True

    #+++++++++++++++++++++++++++++++
    # . Discrete space
    #+++++++++++++++++++++++++++++++

    ncells = [4, 4]
    # ncells = [2, 2]
    degree = [2, 2]
    nquads = [d + 1 for d in degree]

    t_stamp = time_count()
    print('# AA...' )

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1

    t_stamp = time_count(t_stamp)
    print('# Comm Proj...' )
    # Projectors for broken spaces
    # - image is a discrete field in the multipatch (broken) space V0, V1 or V2
    # - when applied to a conforming field, the resulting discrete field actually belongs to the conforming discrete subspace
    # - they should commute with the differential operators (the broken or Conga ones, since these two coincide on conforming fields)
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    t_stamp = time_count(t_stamp)
    print('# Conf Proj...' )
    # Conforming projection operators
    Pconf_0 = ConformingProjection_V0(V0h, domain_h)#, verbose=False)
    Pconf_1 = ConformingProjection_V1(V1h, domain_h)# hom_bc=True)#, verbose=False)

    if only_cP_check:
        pass
    else:
        t_stamp = time_count(t_stamp)
        print('# Mass...' )
        # Mass matrices for multipatch spaces (block-diagonal)
        M0 = BrokenMass(V0h, domain_h, is_scalar=True)
        M1 = BrokenMass(V1h, domain_h, is_scalar=False)

        t_stamp = time_count(t_stamp)
        print('# broken derivatives...' )
        # Broken derivative operators
        bD0, bD1 = derham_h.broken_derivatives_as_operators

    t_stamp = time_count(t_stamp)

    #+++++++++++++++++++++++++++++++
    # . some target functions
    #+++++++++++++++++++++++++++++++

    x,y       = domain.coordinates

    # u and E = grad u
    u_solution  = x**2 + y**2
    E_solution_x = 2*x
    E_solution_y = 2*y

    # A and B = curl A (scalar curl)
    A_solution_x = -y*2
    A_solution_y = x*2
    B_solution = 2*(x+y)

    # TODO: use non polynomial fields to check commuting diagram properties...

    from sympy import lambdify
    u_sol   = lambdify(domain.coordinates, u_solution)
    E_sol_x = lambdify(domain.coordinates, E_solution_x)
    E_sol_y = lambdify(domain.coordinates, E_solution_y)
    A_sol_x = lambdify(domain.coordinates, A_solution_x)
    A_sol_y = lambdify(domain.coordinates, A_solution_y)
    B_sol   = lambdify(domain.coordinates, B_solution)

    # pull-backs of u and E
    # u_sol_log = [lambda xi1, xi2 : u_sol(*f(xi1,xi2)) for f in F]
    u_sol_log = [pull_2d_h1(u_sol, f) for f in mappings_list]
    E_sol_log = [pull_2d_hcurl([E_sol_x,E_sol_y], f) for f in mappings_list]
    # this gives a list of list of functions : E_sol_log[k][d] : xi1, xi2 -> E_d(xi1, xi2) on the logical patch k

    # pull-backs of A and B
    A_sol_log = [pull_2d_hcurl([A_sol_x,A_sol_y], f) for f in mappings_list]
    B_sol_log = [pull_2d_l2(B_sol, f) for f in mappings_list]

    # discontinuous targets (scalar, and vector-valued)
    nb_patches = len(domain)
    # v_sol_log = [(lambda y:lambda xi1, xi2 : y)(i) for i in range(len(domain))]
    v_sol_log = [lambda xi1, xi2, ii=i : ii for i in range(nb_patches)]

    # G_sol_log = [[(lambda y:lambda xi1, xi2 : y)(i) for d in [0,1]] for i in range(len(domain))]
    G_sol_log = [[lambda xi1, xi2, ii=i : ii+1 for d in [0,1]] for i in range(nb_patches)]

    # note: in other tests, the target functions are given directly as lambda functions -- what is best ?
    # fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    # D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)
    # D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)
    # fun2    = lambda xi1, xi2 : .5*np.sin(xi1)*np.sin(xi2)

    # for the plots:
    N=20
    etas, xx, yy = get_plotting_grid(mappings, N)
    gridlines_x1, gridlines_x2 = get_patch_knots_gridlines(V0h, N, mappings, plotted_patch=1)

    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

    # I. check the qualitative properties of the Conforming Projections

    # - in V0 with a discontinuous v
    v0   = P0(v_sol_log)
    v0c  = Pconf_0(v0)   # should be H1-conforming (ie, continuous)

    M0, M0_inv = get_K0_and_K0_inv(V0h, uniform_patches=True)

    print('******     check K0:   ****************************')
    K0, K0_inv = get_K0_and_K0_inv(V0h, uniform_patches=True)
    cP0 = Pconf_0.to_sparse_matrix()

    v0_bc = v0.coeffs.toarray()
    v0_2c = K0_inv @ (cP0 @ (K0 @ v0_bc))

    v0_2 = FemField(V0h, coeffs=array_to_stencil(v0_2c, V0h.vector_space))

    print('more shapes are: \n K0 = {0}\n K0_inv = {1}\n'.format(K0.shape,K0_inv.shape))


    if only_cP_check:
        # plot v, v0 and v0c
        v_vals   = grid_vals_h1(v_sol_log)
        v0_vals  = grid_vals_h1(v0)
        v0c_vals = grid_vals_h1(v0c)

        my_small_plot(
            title=r'broken and conforming approximation of some $v$',
            vals=[v_vals, v0_vals, v0c_vals],
            titles=[r'$v^{ex}(x,y)$', r'$v^h(x,y)$', r'$P^{0,c} v^h(x,y)$'],
            xx=xx, yy=yy,
            surface_plot=True,
            cmap='jet'
        )


        # plot v, v0 and v0c
        v_vals   = grid_vals_h1(v_sol_log)
        v02_vals = grid_vals_h1(v0_2)
        v0c_vals = grid_vals_h1(v0c)

        my_small_plot(
            title=r'broken and conforming approximation of some $v$',
            vals=[v_vals, v02_vals, v0c_vals],
            titles=[r'$v^{ex}(x,y)$', r'$P^{0,2c}v^h(x,y)$', r'$P^{0,c} v^h(x,y)$'],
            xx=xx, yy=yy,
            surface_plot=True,
            cmap='jet'
        )


        G1   = P1(G_sol_log)
        G1c  = Pconf_1(G1)  # should be curl-conforming


        print('******     check K1:   ****************************')
        K1, K1_inv = get_K1_and_K1_inv(V1h, uniform_patches=True)
        cP1 = Pconf_1.to_sparse_matrix()

        print('V1h shapes: \n cP1 = {0}, K1 = {1}\n K1_inv = {2}\n'.format(cP1.shape, K0.shape,K0_inv.shape))

        G1_bc = G1.coeffs.toarray()   # broken
        G1_2c = K1_inv @ (cP1 @ (K1 @ G1_bc))

        G1_2 = FemField(V1h, coeffs=array_to_stencil(G1_2c, V1h.vector_space))

        # plotting
        G_x_vals, G_y_vals     = grid_vals_hcurl(G_sol_log)
        G1_x_vals, G1_y_vals   = grid_vals_hcurl(G1)
        G1c_x_vals, G1c_y_vals = grid_vals_hcurl(G1c)
        G1c2_x_vals, G1c2_y_vals = grid_vals_hcurl(G1_2)

        # # plot G, G1 and G1c, x component
        # my_small_plot(
        #     title=r'broken and conforming approximation of some $v$',
        #     vals=[G_x_vals,G1_x_vals,G1c_x_vals],
        #     titles=[r'$G^{ex}_x(x,y)$',r'$G^h_x(x,y)$',r'$(P^{1,c}G)_x v^h(x,y)$'],
        #     xx=xx,
        #     yy=yy,
        # )
        #
        # # plot G, G1 and G1c, y component
        # my_small_plot(
        #     title=r'broken and conforming approx of some $G$, y component',
        #     vals=[G_y_vals, G1_y_vals, G1c_y_vals],
        #     titles=[r'$G^{ex}_y(x,y)$', r'$G^h_y(x,y)$', r'$(P^{1,c}G)_y v^h(x,y)$'],
        #     xx=xx,
        #     yy=yy,
        # )
        #
        # plot G1x, G1y, and confP1 approx
        my_small_plot(
            title=r'broken and conforming approximation: comp x',
            vals=[G1_x_vals,G1c_x_vals,G1c2_x_vals],
            titles=[r'$G^h_x(x,y)$',r'$(P^{1,c}G)_x v^h(x,y)$', 'new P1 x'],
            xx=xx,
            yy=yy,
            surface_plot=True,
        )

        # plot G1x, G1y, and confP1 approx
        my_small_plot(
            title=r'broken and conforming approximation: comp y',
            vals=[G1_y_vals,G1c_y_vals,G1c2_y_vals],
            titles=[r'$G^h_y(x,y)$',r'$(P^{1,c}G)_y v^h(x,y)$', 'new P1 y'],
            xx=xx,
            yy=yy,
            surface_plot=True,
        )

        print('******     DEBUG   end   ****************************')


        exit()

    cDv0 = bD0(v0c)

    D0 = ComposedLinearOperator([bD0, Pconf_0])
    cDv0 = D0(v0)

    # - in V1 with a discontinuous field G
    cDG1 = bD1(G1c)

    # II. check the commuting diag properties

    # - for the gradient:  D0 P0 u = P1 grad u
    u0 = P0(u_sol_log)
    grad_u0 = bD0(u0)
    # using E = grad u
    E1 = P1(E_sol_log)

    # - for the (scalar) curl:  D1 P1 A = P2 curl A
    A1 = P1(A_sol_log)
    curl_A1 = bD1(A1)

    # using B = curl A
    B2 = P2(B_sol_log)


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # VISUALIZATION
    #   adapted from examples/poisson_2d_multi_patch.py and
    #   and psydac/api/tests/test_api_feec_2d.py
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # I - 1. qualitative assessment of conf Projection in V0, with discontinuous v

    only_cP1_test = True

    if not only_cP1_test:

        # plot v, v0 and v0c
        v_vals   = grid_vals_h1(v_sol_log)
        v0_vals  = grid_vals_h1(v0)
        v0c_vals = grid_vals_h1(v0c)

        my_small_plot(
            title=r'broken and conforming approximation of some $v$',
            vals=[v_vals, v0_vals, v0c_vals],
            titles=[r'$v^{ex}(x,y)$', r'$v^h(x,y)$', r'$P^{0,c} v^h(x,y)$'],
            xx=xx, yy=yy,
            surface_plot=True,
            cmap='jet'
        )

            # gridlines_x1=gridlines_x1,
            # gridlines_x2=gridlines_x2,


        # plot v0 and cDv0
        cDv0_x_vals, cDv0_y_vals = grid_vals_hcurl(cDv0)

        my_small_plot(
            title=r'discontinuous $v^h$ and its Conga gradient $D^0 = D^{0,b}P^{0,c}$',
            vals=[v0_vals, cDv0_x_vals, cDv0_y_vals],
            titles=[r'$v^h(x,y)$', r'$(D^0 v^h)_x(x,y)$' , r'$(D^0 v^h)_y(x,y)$' ],
            xx=xx, yy=yy,
            surface_plot=True,
        )
            # gridlines_x1=gridlines_x1,
            # gridlines_x2=gridlines_x2,

        # print(" ok stop for now -- confP1 will be checked later ")
        # exit()

    # I - 2. qualitative assessment of conf Projection in V1, with discontinuous G


    G_x_vals, G_y_vals     = grid_vals_hcurl(G_sol_log)
    G1_x_vals, G1_y_vals   = grid_vals_hcurl(G1)
    G1c_x_vals, G1c_y_vals = grid_vals_hcurl(G1c)

    # plot G, G1 and G1c, x component
    my_small_plot(
        title=r'broken and conforming approximation of some $v$',
        vals=[G_x_vals,G1_x_vals,G1c_x_vals],
        titles=[r'$G^{ex}_x(x,y)$',r'$G^h_x(x,y)$',r'$(P^{1,c}G)_x v^h(x,y)$'],
        xx=xx,
        yy=yy,
    )

    # plot G, G1 and G1c, y component
    my_small_plot(
        title=r'broken and conforming approx of some $G$, y component',
        vals=[G_y_vals, G1_y_vals, G1c_y_vals],
        titles=[r'$G^{ex}_y(x,y)$', r'$G^h_y(x,y)$', r'$(P^{1,c}G)_y v^h(x,y)$'],
        xx=xx,
        yy=yy,
    )

    # plot G1x, G1y, and confP1 approx
    my_small_plot(
        title=r'broken and conforming approximation: comp x',
        vals=[G1_x_vals,G1c_x_vals],
        titles=[r'$G^h_x(x,y)$',r'$(P^{1,c}G)_x v^h(x,y)$'],
        xx=xx,
        yy=yy,
        surface_plot=True,
    )

    # plot G1x, G1y, and confP1 approx
    my_small_plot(
        title=r'broken and conforming approximation: comp y',
        vals=[G1_y_vals,G1c_y_vals],
        titles=[r'$G^h_y(x,y)$',r'$(P^{1,c}G)_y v^h(x,y)$'],
        xx=xx,
        yy=yy,
        surface_plot=True,
    )

    # todo: plot G1, broken_curl G1 and conga_curl G1

    print(" ok, confP1 done now")
    exit()


    # II. visual check of commuting diag properties
    # - for the gradient:  D0 P0 u = P1 grad u (P1 E)

    # plot u and u0 = u_h

    u_vals  = grid_vals_h1(u_sol_log)
    u0_vals = grid_vals_h1(u0)
    u_err = [abs(u1 - u2) for u1, u2 in zip(u_vals, u0_vals)]
    # u_err   = abs(u_vals - u0_vals)

    my_small_plot(
        title=r'approximation of a potential $u$',
        vals=[u_vals, u0_vals, u_err],
        titles=[r'$u^{ex}(x,y)$', r'$u^h(x,y)$', r'$|(u^{ex}-u^h)(x,y)|$'],
        xx=xx,
        yy=yy,
    )

    # plot (compare) E1 and grad_u0

    E_x_vals, E_y_vals   = grid_vals_hcurl(E_sol_log)
    E1_x_vals, E1_y_vals = grid_vals_hcurl(E1)

    E_x_err = [abs(e1 - e2) for e1, e2 in zip(E_x_vals, E1_x_vals)]
    E_y_err = [abs(e1 - e2) for e1, e2 in zip(E_y_vals, E1_y_vals)]
    # E_x_err = abs(E_x_vals - E1_x_vals)
    # E_y_err = abs(E_y_vals - E1_y_vals)

    my_small_plot(
        title=r'approximation of a field $E$, $x$ component',
        vals=[E_x_vals, E1_x_vals, E_x_err],
        titles=[r'$E^{ex}_x(x,y)$', r'$E^h_x(x,y)$', r'$|(E^{ex}-E^h)_x(x,y)|$'],
        xx=xx,
        yy=yy,
    )

    my_small_plot(
        title=r'approximation of a field $E$, $y$ component',
        vals=[E_y_vals, E1_y_vals, E_y_err],
        titles=[r'$E^{ex}_y(x,y)$', r'$E^h_y(x,y)$', r'$|(E^{ex}-E^h)_y(x,y)|$'],
        xx=xx,
        yy=yy,
    )

    grad_u0_x_vals, grad_u0_y_vals = grid_vals_hcurl(grad_u0)
    gu_x_err = [abs(e1 - e2) for e1, e2 in zip(grad_u0_x_vals, E1_x_vals)]
    gu_y_err = [abs(e1 - e2) for e1, e2 in zip(grad_u0_y_vals, E1_y_vals)]
    # gu_x_err = abs(grad_u0_x_vals - E1_x_vals)
    # gu_y_err = abs(grad_u0_y_vals - E1_y_vals)

    my_small_plot(
        title=r'commuting diagram property ($x$ component)',
        vals=[grad_u0_x_vals, E1_x_vals, gu_x_err],
        titles=[r'$(D^0u^h)_x(x,y)$' , r'$E^h_x(x,y)$', r'$|(D^0u^h - E^h)_x(x,y)|$'],
        xx=xx,
        yy=yy,
    )

    my_small_plot(
        title=r'commuting diagram property ($y$ component)',
        vals=[grad_u0_y_vals, E1_y_vals, gu_y_err],
        titles=[r'$(D^0u^h)_y(x,y)$' , r'$E^h_y(x,y)$', r'$|(D^0u^h - E^h)_y(x,y)|$'],
        xx=xx,
        yy=yy,
    )



if __name__ == '__main__':

    conga_operators_2d()

