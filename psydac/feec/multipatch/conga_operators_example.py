# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

# checking import path...
# import sympde
# print(sympde.__file__)
# exit()

from sympde.calculus import grad, dot, inner, rot, div
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus

from sympde.topology import Derham
# from sympde.topology import ProductSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

#from psydac.api.discretization import discretize
from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above

from psydac.linalg.basic import LinearOperator
# ProductSpace
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix
from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.direct_solvers import SparseSolver
from psydac.linalg.identity import IdentityLinearOperator #, IdentityStencilMatrix as IdentityMatrix

from psydac.fem.basic   import FemField
from psydac.fem.vector import ProductFemSpace, VectorFemSpace

from psydac.feec.pull_push     import pull_2d_h1, pull_2d_hcurl, pull_2d_l2

from psydac.feec.derivatives import Gradient_2D
from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl

from psydac.utilities.utils    import refine_array_1d

from psydac.feec.multipatch.operators import BrokenMass_V0, BrokenMass_V1
# <<<<<<< HEAD
# from psydac.feec.multipatch.operators import IdLinearOperator, SumLinearOperator, MultLinearOperator
# from psydac.feec.multipatch.operators import BrokenGradient_2D, BrokenTransposedGradient_2D
# from psydac.feec.multipatch.operators import ConformingProjection_V0, ConformingProjection_V1, ComposedLinearOperator
# from psydac.feec.multipatch.operators import Multipatch_Projector_H1, Multipatch_Projector_Hcurl
# from psydac.feec.multipatch.operators import get_scalar_patch_fields, get_vector_patch_fields
# =======
from psydac.feec.multipatch.operators import ConformingProjection_V0, ConformingProjection_V1, DummyConformingProjection_V1, ComposedLinearOperator
# from psydac.feec.multipatch.operators import get_scalar_patch_fields, get_vector_patch_fields
from psydac.feec.multipatch.operators import get_grid_vals_V0, get_grid_vals_V1, get_grid_vals_V2
# >>>>>>> [feec-multipatch] add dummy test for conforming projection in V1

comm = MPI.COMM_WORLD


def conga_operators_2d():
    """
    - assembles several multipatch operators and a conforming projection

    - performs several tests:
      - ...
      -

    """


    #+++++++++++++++++++++++++++++++
    # . Domain
    #+++++++++++++++++++++++++++++++

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    cartesian = False

    if cartesian:
        mapping_1 = IdentityMapping('M1', 2)
        mapping_2 = IdentityMapping('M2',2)
    else:
        mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
        mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    domain_1     = mapping_1(A)
    domain_2     = mapping_2(B)

    domain = domain_1.join(domain_2, name = 'domain',
                bnd_minus = domain_1.get_boundary(axis=1, ext=1),
                bnd_plus  = domain_2.get_boundary(axis=1, ext=-1))

    mappings  = {A.interior:mapping_1, B.interior:mapping_2}

    mappings_obj = [mapping_1, mapping_2]
    F = [f.get_callable_mapping() for f in mappings_obj]


    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])


    #+++++++++++++++++++++++++++++++
    # . Discrete space
    #+++++++++++++++++++++++++++++++

    ncells = [2**2, 2**2]
    degree = [2, 2]
    nquads = [d + 1 for d in degree]

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1

    # Mass matrices for multipatch spaces (block-diagonal)
    M0 = BrokenMass_V0(V0h, domain_h)
    M1 = BrokenMass_V1(V1h, domain_h)

    # Projectors for broken spaces
    # - image is a discrete field in the multipatch (broken) space V0, V1 or V2
    # - when applied to a conforming field, the resulting discrete field actually belongs to the conforming discrete subspace
    # - they should commute with the differential operators (the broken or Conga ones, since these two coincide on conforming fields)
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    # Conforming projection operators
    Pconf_0 = ConformingProjection_V0(V0h, domain_h)#, verbose=False)
    # Pconf_1 = DummyConformingProjection_V1(V1h, domain_h)#, verbose=False)
    Pconf_1 = ConformingProjection_V1(V1h, domain_h)#, verbose=False)

    # Broken derivative operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators

    # Conga derivative operators (not needed in this example file)
    # cD0 = ComposedLinearOperator(bD0, Pconf_0)
    # cD1 = ComposedLinearOperator(bD1, Pconf_1)

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
    u_sol = lambdify(domain.coordinates, u_solution)
    E_sol_x = lambdify(domain.coordinates, E_solution_x)
    E_sol_y = lambdify(domain.coordinates, E_solution_y)
    A_sol_x = lambdify(domain.coordinates, A_solution_x)
    A_sol_y = lambdify(domain.coordinates, A_solution_y)
    B_sol = lambdify(domain.coordinates, B_solution)

    # pull-backs of u and E
    # u_sol_log = [lambda xi1, xi2 : u_sol(*f(xi1,xi2)) for f in F]
    u_sol_log = [pull_2d_h1(u_sol, f) for f in mappings_obj]
    E_sol_log = [pull_2d_hcurl([E_sol_x,E_sol_y], f) for f in mappings_obj]
    # this gives a list of list of functions : E_sol_log[k][d] : xi1, xi2 -> E_d(xi1, xi2) on the logical patch k

    # pull-backs of A and B
    A_sol_log = [pull_2d_hcurl([A_sol_x,A_sol_y], f) for f in mappings_obj]
    B_sol_log = [pull_2d_l2(B_sol, f) for f in mappings_obj]

    # discontinuous targets (scalar, and vector-valued)
    v_sol_log = [
        lambda xi1, xi2 : 1, # u_sol(*F[0](xi1,xi2)),
        lambda xi1, xi2 : 0,
        ]

    G_sol_log = [
        [lambda xi1, xi2 : d for d in [0,1]],
        [lambda xi1, xi2 : 1-d for d in [0,1]],
        ]

    # note: in other tests, the target functions are given directly as lambda functions -- what is best ?
    # fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    # D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)
    # D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)
    # fun2    = lambda xi1, xi2 : .5*np.sin(xi1)*np.sin(xi2)

    # I. check the qualitative properties of the Conforming Projections

    # - in V0 with a discontinuous v
    v0 = P0(v_sol_log)
    v0c = Pconf_0(v0)   # should be H1-conforming (ie, continuous)
    cDv0 = bD0(v0c)

    # - in V1 with a discontinuous field G
    G1 = P1(G_sol_log)
    G1c = Pconf_1(G1)  # should be curl-conforming
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

    plot_fields = True

    if plot_fields:

        N = 20

        etas     = [[refine_array_1d( bounds, N ) for bounds in zip(D.min_coords, D.max_coords)] for D in mappings]

        mappings = [lambdify(M.logical_coordinates, M.expressions) for d,M in mappings.items()]

        pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings, etas)]
        pcoords  = np.concatenate(pcoords, axis=1)

        xx = pcoords[:,:,0]
        yy = pcoords[:,:,1]

        # to plot a patch grid
        plotted_patch = 1
        if plotted_patch in [0, 1]:
            grid_x1 = V0h.spaces[plotted_patch].breaks[0]
            grid_x2 = V0h.spaces[plotted_patch].breaks[1]
            x1 = refine_array_1d(grid_x1, N)
            x2 = refine_array_1d(grid_x2, N)
            x1, x2 = np.meshgrid(x1, x2, indexing='ij')
            x, y = F[plotted_patch](x1, x2)
            gridlines_x1 = (x[:, ::N],   y[:, ::N]  )
            gridlines_x2 = (x[::N, :].T, y[::N, :].T)
            gridlines = (gridlines_x1, gridlines_x2)


        # v, v0 and conf proj v0c
        # todo: best with a H1 push-forward...
        # v_vals  = [np.array( [[phi(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(v_sol_log,etas)]
        # # or: v_vals  = [np.array( [[v_sol( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        # v_vals  = np.concatenate(v_vals,     axis=1)

        # v0_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(v0.fields, etas)]
        # v0_vals  = np.concatenate(v0_vals,     axis=1)
        # v0c_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(v0c.fields, etas)]
        # v0c_vals = np.concatenate(v0c_vals,     axis=1)



        # I - 1. qualitative assessment of conf Projection in V0, with discontinuous v

        # plot v, v0 and v0c

        v_vals   = get_grid_vals_V0(v_sol_log, None, etas, mappings_obj)
        v0_vals  = get_grid_vals_V0(v0, V0h, etas, mappings_obj)
        v0c_vals = get_grid_vals_V0(v0c, V0h, etas, mappings_obj)


        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'broken and conforming approximation of some $v$', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)

        if plotted_patch is not None:
            ax.plot(*gridlines_x1, color='k')
            ax.plot(*gridlines_x2, color='k')

        cp = ax.contourf(xx, yy, v_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$v^{ex}(x,y)$' )


        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, v0_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$v^h(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, v0c_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$P^{0,c} v^h(x,y)$' )

        plt.show()

        # show v0 and cDv0

        cDv0_x_vals, cDv0_y_vals = get_grid_vals_V1(cDv0, V1h, etas, mappings_obj)

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'discontinuous $v^h$ and its Conga gradient $D^0 = D^{0,b}P^{0,c}$', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)

        if plotted_patch is not None:
            ax.plot(*gridlines_x1, color='k')
            ax.plot(*gridlines_x2, color='k')

        cp = ax.contourf(xx, yy, v0_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$v^h(x,y)$' )


        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, cDv0_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$(D^0 v^h)_x(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, cDv0_y_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$(D^0 v^h)_y(x,y)$' )

        plt.show()


        # I - 2. qualitative assessment of conf Projection in V1, with discontinuous G

        G_x_vals, G_y_vals     = get_grid_vals_V1(G_sol_log, None, etas, mappings_obj)
        G1_x_vals, G1_y_vals   = get_grid_vals_V1(G1, V1h, etas, mappings_obj)
        G1c_x_vals, G1c_y_vals = get_grid_vals_V1(G1c, V1h, etas, mappings_obj)

        # plot G, G1 and G1c, x component

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'broken and conforming approximation of some $v$', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)

        if plotted_patch is not None:
            ax.plot(*gridlines_x1, color='k')
            ax.plot(*gridlines_x2, color='k')

        cp = ax.contourf(xx, yy, G_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$G^{ex}_x(x,y)$' )


        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, G1_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$G^h_x(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, G1c_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$(P^{1,c}G)_x v^h(x,y)$' )

        plt.show()

        # plot G, G1 and G1c, y component

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'broken and conforming approximation of some $v$', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)

        if plotted_patch is not None:
            ax.plot(*gridlines_x1, color='k')
            ax.plot(*gridlines_x2, color='k')

        cp = ax.contourf(xx, yy, G_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$G^{ex}_y(x,y)$' )


        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, G1_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$G^h_y(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, G1c_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$(P^{1,c}G)_y v^h(x,y)$' )

        plt.show()



        # todo: plot G1, broken_curl G1 and conga_curl G1



        # II. visual check of commuting diag properties
        # - for the gradient:  D0 P0 u = P1 grad u (P1 E)

        # plot u and u0 = u_h

        u_vals  = get_grid_vals_V0(u_sol_log, None, etas, mappings_obj)
        u0_vals = get_grid_vals_V0(u0, V0h, etas, mappings_obj)
        u_err   = abs(u_vals - u0_vals)

        # u_vals  = [np.array( [[u_sol( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        # u_vals  = np.concatenate(u_vals,     axis=1)
        # u0s = get_scalar_patch_fields(u0, V0h)
        # u0_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(u0s, etas)]
        # # u0_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(u0.fields, etas)]
        # u0_vals  = np.concatenate(u0_vals,     axis=1)

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'approximation of a potential $u$', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)

        if plotted_patch is not None:
            ax.plot(*gridlines_x1, color='k')
            ax.plot(*gridlines_x2, color='k')

        cp = ax.contourf(xx, yy, u_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$u^{ex}(x,y)$' )


        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, u0_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$u^h(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, u_err, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$|(u^{ex}-u^h)(x,y)|$' )

        plt.show()


        # plot (compare) E1 and grad_u0

        E_x_vals, E_y_vals   = get_grid_vals_V1(E_sol_log, None, etas, mappings_obj)
        E1_x_vals, E1_y_vals = get_grid_vals_V1(E1, V1h, etas, mappings_obj)
        grad_u0_x_vals, grad_u0_y_vals = get_grid_vals_V1(grad_u0, V1h, etas, mappings_obj)

        E_x_err = abs(E_x_vals - E1_x_vals)
        E_y_err = abs(E_y_vals - E1_y_vals)

        gu_x_err = abs(grad_u0_x_vals - E1_x_vals)
        gu_y_err = abs(grad_u0_y_vals - E1_y_vals)


        # E_x_vals = [np.array( [[E_sol_x( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        # E_y_vals = [np.array( [[E_sol_y( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        # E_x_vals = np.concatenate(E_x_vals,     axis=1)
        # E_y_vals = np.concatenate(E_y_vals,     axis=1)


        # E1_x_vals = 2*[None]
        # E1_y_vals = 2*[None]
        # grad_u0_x_vals = 2*[None]
        # grad_u0_y_vals = 2*[None]
        #
        # E1s = get_vector_patch_fields(E1, V1h)
        # grad_u0s = get_vector_patch_fields(grad_u0, V1h)
        # for k in [0,1]:
        #     # patch k
        #     eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        #     E1_x_vals[k] = np.empty_like(eta_1)
        #     E1_y_vals[k] = np.empty_like(eta_1)
        #     grad_u0_x_vals[k] = np.empty_like(eta_1)
        #     grad_u0_y_vals[k] = np.empty_like(eta_1)
        #     for i, x1i in enumerate(eta_1[:, 0]):
        #         for j, x2j in enumerate(eta_2[0, :]):
        #             E1_x_vals[k][i, j], E1_y_vals[k][i, j] = \
        #                 push_2d_hcurl(E1s[k].fields[0], E1s[k].fields[1], x1i, x2j, mappings_obj[k])
        #             grad_u0_x_vals[k][i, j], grad_u0_y_vals[k][i, j] = \
        #                 push_2d_hcurl(grad_u0s[k].fields[0], grad_u0s[k].fields[1], x1i, x2j, mappings_obj[k])
        #
        # E1_x_vals = np.concatenate(E1_x_vals,     axis=1)
        # E1_y_vals = np.concatenate(E1_y_vals,     axis=1)
        #
        # grad_u0_x_vals = np.concatenate(grad_u0_x_vals,     axis=1)
        # grad_u0_y_vals = np.concatenate(grad_u0_y_vals,     axis=1)
        #

        # plot E_x and E1_x

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'approximation of a field $E$, $x$ component', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)
        cp = ax.contourf(xx, yy, E_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$E^{ex}_x(x,y)$' )

        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, E1_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$E^h_x(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, E_x_err, 50, cmap='jet')
        # cp3 = ax.contourf(xx, yy, err, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$|(E^{ex}-E^h)_x(x,y)|$' )

        plt.show()

        # plot E_y and E1_y

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'approximation of a field $E$, $y$ component', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)
        cp = ax.contourf(xx, yy, E_y_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$E^{ex}_y(x,y)$' )

        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, E1_y_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$E^h_y(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, E_y_err, 50, cmap='jet')
        # cp3 = ax.contourf(xx, yy, err, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$|(E^{ex}-E^h)_y(x,y)|$' )

        plt.show()


        # show grad_u0_x and E1_x

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'commuting diagram property (x component)', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)
        cp = ax.contourf(xx, yy, grad_u0_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$(D^0u^h)_x(x,y)$' )

        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, E1_x_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$E^h_x(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, gu_x_err, 50, cmap='jet')
        # cp3 = ax.contourf(xx, yy, err, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$|(D^0u^h - E^h)_x(x,y)|$' )

        plt.show()


        # show grad_u0_y and E1_y

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'commuting diagram property (y component)', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)
        cp = ax.contourf(xx, yy, grad_u0_y_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$(D^0u^h)_y(x,y)$' )

        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, E1_y_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$E^h_y(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, gu_y_err, 50, cmap='jet')
        # cp3 = ax.contourf(xx, yy, err, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$|(D^0u^h - E^h)_y(x,y)|$' )

        plt.show()


    print(" done. ")
    exit()



if __name__ == '__main__':

    conga_operators_2d()

