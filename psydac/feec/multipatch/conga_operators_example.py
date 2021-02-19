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

from psydac.api.discretization import discretize

from psydac.linalg.basic import LinearOperator
# ProductSpace
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix
from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.direct_solvers import SparseSolver
from psydac.linalg.identity import IdentityLinearOperator #, IdentityStencilMatrix as IdentityMatrix

from psydac.fem.basic   import FemField
from psydac.fem.vector import ProductFemSpace, VectorFemSpace

from psydac.feec.pull_push     import push_2d_hcurl, pull_2d_hcurl  #, push_2d_l2

from psydac.feec.derivatives import Gradient_2D
from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl

from psydac.utilities.utils    import refine_array_1d

from psydac.feec.multipatch.operators import BrokenMass_V0, BrokenMass_V1
from psydac.feec.multipatch.operators import IdLinearOperator, SumLinearOperator, MultLinearOperator
from psydac.feec.multipatch.operators import BrokenGradient_2D, BrokenTransposedGradient_2D
from psydac.feec.multipatch.operators import ConformingProjection, ComposedLinearOperator
from psydac.feec.multipatch.operators import Multipatch_Projector_H1, Multipatch_Projector_Hcurl
from psydac.feec.multipatch.operators import get_scalar_patch_fields, get_vector_patch_fields

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

    ncells=[2**2, 2**2]
    degree=[2,2]

    domain_h = discretize(domain, ncells=ncells, comm=comm)

    ## this should eventually work:
    # derham_h = discretize(derham, domain_h, degree=degree)      # build them by hand if this doesn't work
    # V0h       = derham_h.V0
    # V1h       = derham_h.V1

    ## meanwhile, we define the broken multipatch spaces individually:
    V0h = discretize(derham.V0, domain_h, degree=degree)
    V1h = discretize(derham.V1, domain_h, degree=degree, basis='M')

    assert isinstance(V1h, ProductFemSpace)
    assert isinstance(V1h.vector_space, BlockVectorSpace)

    ## and also as list of patches:
    domains = [domain_1, domain_2]
    derhams = [Derham(dom, ["H1", "Hcurl", "L2"]) for dom in domains]

    domains_h = [discretize(dom, ncells=ncells, comm=comm) for dom in domains]
    derhams_h = [discretize(derh, dom_h, degree=degree)
                 for dom_h, derh in zip(domains_h, derhams)]


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # TODO [YG, 12.02.2021]: Multi-patch de Rham sequence should provide this!
    V0h._spaces = tuple(derh_h.V0 for derh_h in derhams_h)
    V1h._spaces = tuple(derh_h.V1 for derh_h in derhams_h)

    V0h._vector_space = BlockVectorSpace(*[V.vector_space for V in V0h.spaces])
    V1h._vector_space = BlockVectorSpace(*[V.vector_space for V in V1h.spaces])
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # Mass matrix for multipatch space V1

    M0 = BrokenMass_V0(V0h, domain_h)
    M1 = BrokenMass_V1(V1h, domain_h)

    #+++++++++++++++++++++++++++++++
    # . some target functions
    #+++++++++++++++++++++++++++++++

    # fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    # D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)
    # D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)
    # fun2    = lambda xi1, xi2 : .5*np.sin(xi1)*np.sin(xi2)

    x,y       = domain.coordinates
    u_solution  = x**2 + y**2
    E_solution_x = 2*x
    E_solution_y = 2*y

    from sympy import lambdify
    u_sol = lambdify(domain.coordinates, u_solution)
    E_sol_x = lambdify(domain.coordinates, E_solution_x)
    E_sol_y = lambdify(domain.coordinates, E_solution_y)

    # pull-backs of u and E
    u_sol_log = [lambda xi1, xi2 : u_sol(*f(xi1,xi2)) for f in F]

    # list: E_sol_log[k][d] : xi1, xi2 -> E_x(xi1, xi2) on patch k
    E_sol_log = [pull_2d_hcurl([E_sol_x,E_sol_y], f) for f in mappings_obj]

    # discontinuous target
    v_sol_log = [
        lambda xi1, xi2 : 1, # u_sol(*F[0](xi1,xi2)),
        lambda xi1, xi2 : 0,
        ]

    #+++++++++++++++++++++++++++++++
    # . Multipatch H1, Hcurl (commuting) projectors
    #+++++++++++++++++++++++++++++++

    # I. multipatch V0 projection

    P0 = Multipatch_Projector_H1(V0h)
    u0 = P0(u_sol_log)
    v0 = P0(v_sol_log)

    # II. conf projection V0 -> V0
    # projection from broken multipatch space to conforming subspace (using the same basis)

    Pconf_0 = ConformingProjection(V0h, domain_h, verbose=False)
    v0c = Pconf_0(v0)

    # III. multipatch V1 projection

    P1 = Multipatch_Projector_Hcurl(V1h, nquads=[5,5])
    E1 = P1(E_sol_log)

    # IV.  multipatch (broken) grad operator on V0h

    bD0 = BrokenGradient_2D(V0h, V1h)
    grad_u0 = bD0(u0)



    # V. Conga grad operator on V0h

    cD0 = ComposedLinearOperator(bD0, Pconf_0)
    cDv0 = cD0(v0)


    # Transpose of the Conga grad operator (using the symmetry of Pconf_0)

    # bD0_T = BrokenTransposedGradient_2D(V0hs, V1hs, V0h, V1h)
    # cD0_T = ComposedLinearOperator(Pconf_0,bD0_T)
    #
    # I0 = IdLinearOperator(V0h)
    # # I0 = IdentityLinearOperator(V0h.vector_space)
    #
    #
    # #+++++++++++++++++++++++++++++++
    # # . Conga Poisson solver
    # #+++++++++++++++++++++++++++++++
    #
    #
    # x,y = domain.coordinates
    # solution = x**2 + y**2
    #
    # f        = -4
    #
    # v = element_of(derham.V0, 'v')
    # l = LinearForm(v,  integral(domain, f*v))
    # lh = discretize(l, domain_h, V0h)
    # b = lh.assemble()
    #
    # cD0T_M1_cD0 = ComposedLinearOperator( cD0_T, ComposedLinearOperator( M1, cD0 ) )
    # # A = cD0T_M1_cD0 + (I0 - Pconf_0)
    # minus_cP0 = MultLinearOperator(-1,Pconf_0)
    # I_minus_cP0 = SumLinearOperator( I0, minus_cP0 )
    # A = SumLinearOperator( cD0T_M1_cD0, I_minus_cP0 )



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
        # solution = lambdify(domain.coordinates, solution)

        pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings, etas)]
        pcoords  = np.concatenate(pcoords, axis=1)

        # u exact
        u_vals  = [np.array( [[u_sol( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        u_vals  = np.concatenate(u_vals,     axis=1)

        # Poisson sol
        # sol_vals  = [np.array( [[sol( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        # sol_vals  = np.concatenate(sol_vals,     axis=1)

        # E exact
        E_x_vals = [np.array( [[E_sol_x( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        E_y_vals = [np.array( [[E_sol_y( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        E_x_vals = np.concatenate(E_x_vals,     axis=1)
        E_y_vals = np.concatenate(E_y_vals,     axis=1)

        # u0
        u0s = get_scalar_patch_fields(u0, V0h)
        u0_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(u0s, etas)]
        # u0_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(u0.fields, etas)]
        u0_vals  = np.concatenate(u0_vals,     axis=1)
        u_err = abs(u_vals - u0_vals)

        # Poisson sol_h
        # sols_h = get_scalar_patch_fields(sol_h, V0h)
        # sol_h_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(sols_h, etas)]
        # sol_h_vals  = np.concatenate(sol_h_vals,     axis=1)
        # sol_err = abs(sol_vals - sol_h_vals)


        # v, v0 and conf proj v0c
        v_vals  = [np.array( [[phi(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(v_sol_log,etas)]
        # v_vals  = [np.array( [[v_sol( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        v_vals  = np.concatenate(v_vals,     axis=1)
        v0_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(v0.fields, etas)]
        v0_vals  = np.concatenate(v0_vals,     axis=1)
        v0c_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(v0c.fields, etas)]
        v0c_vals = np.concatenate(v0c_vals,     axis=1)

        cDv0_x_vals = 2*[None]
        cDv0_y_vals = 2*[None]
        cDv0s = get_vector_patch_fields(cDv0, V1h)
        for k in [0,1]:
            # patch k
            eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
            cDv0_x_vals[k] = np.empty_like(eta_1)
            cDv0_y_vals[k] = np.empty_like(eta_1)
            for i, x1i in enumerate(eta_1[:, 0]):
                for j, x2j in enumerate(eta_2[0, :]):
                    cDv0_x_vals[k][i, j], cDv0_y_vals[k][i, j] = \
                        push_2d_hcurl(cDv0s[k].fields[0], cDv0s[k].fields[1], x1i, x2j, mappings_obj[k])
        cDv0_x_vals = np.concatenate(cDv0_x_vals,     axis=1)
        cDv0_y_vals = np.concatenate(cDv0_y_vals,     axis=1)

        # E1 and grad_u0
        E1_x_vals = 2*[None]
        E1_y_vals = 2*[None]
        grad_u0_x_vals = 2*[None]
        grad_u0_y_vals = 2*[None]

        E1s = get_vector_patch_fields(E1, V1h)
        grad_u0s = get_vector_patch_fields(grad_u0, V1h)
        for k in [0,1]:
            # patch k
            eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
            E1_x_vals[k] = np.empty_like(eta_1)
            E1_y_vals[k] = np.empty_like(eta_1)
            grad_u0_x_vals[k] = np.empty_like(eta_1)
            grad_u0_y_vals[k] = np.empty_like(eta_1)
            for i, x1i in enumerate(eta_1[:, 0]):
                for j, x2j in enumerate(eta_2[0, :]):
                    E1_x_vals[k][i, j], E1_y_vals[k][i, j] = \
                        push_2d_hcurl(E1s[k].fields[0], E1s[k].fields[1], x1i, x2j, mappings_obj[k])
                    grad_u0_x_vals[k][i, j], grad_u0_y_vals[k][i, j] = \
                        push_2d_hcurl(grad_u0s[k].fields[0], grad_u0s[k].fields[1], x1i, x2j, mappings_obj[k])

        E1_x_vals = np.concatenate(E1_x_vals,     axis=1)
        E1_y_vals = np.concatenate(E1_y_vals,     axis=1)
        E_x_err = abs(E_x_vals - E1_x_vals)
        E_y_err = abs(E_y_vals - E1_y_vals)

        grad_u0_x_vals = np.concatenate(grad_u0_x_vals,     axis=1)
        grad_u0_y_vals = np.concatenate(grad_u0_y_vals,     axis=1)
        gu_x_err = abs(grad_u0_x_vals - E1_x_vals)
        gu_y_err = abs(grad_u0_y_vals - E1_y_vals)


        # plots

        xx = pcoords[:,:,0]
        yy = pcoords[:,:,1]

        # plot one patch grid
        plotted_patch = 1
        if plotted_patch in [0, 1]:

            #patch_derham = derhams_h[plotted_patch]
            grid_x1 = derhams_h[plotted_patch].V0.breaks[0]
            grid_x2 = derhams_h[plotted_patch].V0.breaks[1]

            print("grid_x1 = ", grid_x1)

            x1 = refine_array_1d(grid_x1, N)
            x2 = refine_array_1d(grid_x2, N)

            x1, x2 = np.meshgrid(x1, x2, indexing='ij')
            x, y = F[plotted_patch](x1, x2)

            print("x1 = ", x1)

            gridlines_x1 = (x[:, ::N],   y[:, ::N]  )
            gridlines_x2 = (x[::N, :].T, y[::N, :].T)
            gridlines = (gridlines_x1, gridlines_x2)


        # plot poisson solutions

        # fig = plt.figure(figsize=(17., 4.8))
        # fig.suptitle(r'approximation of some $v$', fontsize=14)
        #
        # ax = fig.add_subplot(1, 3, 1)
        #
        # if plotted_patch is not None:
        #     ax.plot(*gridlines_x1, color='k')
        #     ax.plot(*gridlines_x2, color='k')
        #
        # cp = ax.contourf(xx, yy, u0_vals, 50, cmap='jet')
        # # cp = ax.contourf(xx, yy, sol_vals, 50, cmap='jet')
        # cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        # ax.set_xlabel( r'$x$', rotation='horizontal' )
        # ax.set_ylabel( r'$y$', rotation='horizontal' )
        # ax.set_title ( r'$u^h(x,y)$' )
        # # ax.set_title ( r'$\phi^{ex}(x,y)$' )
        #
        #
        # ax = fig.add_subplot(1, 3, 2)
        # cp2 = ax.contourf(xx, yy, sol_h_vals, 50, cmap='jet')
        # cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)
        #
        # ax.set_xlabel( r'$x$', rotation='horizontal' )
        # ax.set_ylabel( r'$y$', rotation='horizontal' )
        # ax.set_title ( r'$\Delta_h u^h(x,y)$' )
        # # ax.set_title ( r'$\phi^h(x,y)$' )
        #
        # ax = fig.add_subplot(1, 3, 3)
        # cp3 = ax.contourf(xx, yy, sol_err, 50, cmap='jet')
        # cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)
        #
        # ax.set_xlabel( r'$x$', rotation='horizontal' )
        # ax.set_ylabel( r'$y$', rotation='horizontal' )
        # ax.set_title ( r'$|(\phi^{ex}-\phi^h)(x,y)|$' )
        #
        # plt.show()

        # plot v, v_h and v_hc

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'approximation of some $v$', fontsize=14)

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
        ax.set_title ( r'$P^c v^h(x,y)$' )

        plt.show()

        # show v0 and cDv0

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'discontinuous $v^h$ and its Conga gradient', fontsize=14)

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
        ax.set_title ( r'$(D^0P^c v^h)_x(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, cDv0_y_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$(D^0P^c v^h)_y(x,y)$' )

        plt.show()


        # plot u and u_h

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'approximation of potential $u$', fontsize=14)

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




        # plot E_x and E1_x

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'approximation of field $E_x$', fontsize=14)

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
        fig.suptitle(r'approximation of field $E_y$', fontsize=14)

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


    exit()



    # conga D0 operator



    #############     continue below, but needs a lot of cleaning....




    Dfun_h    = D0(u0)
    Dfun_proj = u1

    # todo: plot the different fields for visual check

    # P0 should map into a conforming function, so we should have u0_conf = u0
    error = (u0.coeffs-u0_conf.coeffs).toarray().max()
    assert abs(error)<1e-9
    print(error)

    # test commuting diagram on the multipatch spaces
    error = (Dfun_proj.coeffs-Dfun_h.coeffs).toarray().max()
    assert abs(error)<1e-9
    print(error)





    #+++++++++++++++++++++++++++++++
    # . Some matrices
    #+++++++++++++++++++++++++++++++

    # identity operator on V0h
    # I0_1 = IdentityMatrix(V0h_1)
    # I0_2 = IdentityMatrix(V0h_2)
    # I0 = BlockMatrix(V0h, V0h, blocks=[[I0_1, None],[None, I0_2]])

    # local (single patch) de Rham sequences:
    derham_1  = Derham(domain_1, ["H1", "Hcurl"])
    derham_2  = Derham(domain_2, ["H1", "Hcurl"])

    domain_h_1 = discretize(domain_1, ncells=ncells, comm=comm)
    domain_h_2 = discretize(domain_2, ncells=ncells, comm=comm)

    # mass matrix of V1   (mostly taken from psydac/api/tests/test_api_feec_3d.py)


    if 0:
        # this would be nice but doesn't work:
        u1, v1 = elements_of(derham.V1, names='u1, v1')
        a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
        a1_h = discretize(a1, domain_h, [V1h, V1h])  # , backend=PSYDAC_BACKEND_GPYCCEL)
        M1 = a1_h.assemble()  #.tosparse().tocsc()
    else:
        # so, block construction
        u1_1, v1_1 = elements_of(derham_1.V1, names='u1_1, v1_1')
        a1_1 = BilinearForm((u1_1, v1_1), integral(domain_1, dot(u1_1, v1_1)))
        a1_h_1 = discretize(a1_1, domain_h_1, [V1h_1, V1h_1])  # , backend=PSYDAC_BACKEND_GPYCCEL)
        M1_1 = a1_h_1.assemble()  #.tosparse().tocsc()

        u1_2, v1_2 = elements_of(derham_2.V1, names='u1_2, v1_2')
        a1_2 = BilinearForm((u1_2, v1_2), integral(domain_2, dot(u1_2, v1_2)))
        a1_h_2 = discretize(a1_2, domain_h_2, [V1h_2, V1h_2])  # , backend=PSYDAC_BACKEND_GPYCCEL)
        M1_2 = a1_h_2.assemble()  #.tosparse().tocsc()

        M1 = BlockMatrix(V1h.vector_space, V1h.vector_space, blocks=[[M1_1, None],[None, M1_2]])

    #+++++++++++++++++++++++++++++++
    # . Differential operators
    #   on conforming and broken spaces
    #+++++++++++++++++++++++++++++++

    # "broken grad" operator, coincides with the grad on the conforming subspace of V0h
    # later: broken_D0 = Gradient_2D(V0h, V1h)   # on multipatch domains we should maybe provide the "BrokenGradient"
    # or broken_D0 = derham_h.D0 ?

    # Note: here we should use the lists V0hs, V1hs defined above

    broken_D0 = BlockMatrix(V0h.vector_space, V1h.vector_space, blocks=[[D0_1, None],[None, D0_2]])


    # plot ?
    # (use example from poisson_2d_multipatch ??)

    # xx = pcoords[:,:,0]
    # yy = pcoords[:,:,1]
    #
    # fig = plt.figure(figsize=(17., 4.8))
    #
    # ax = fig.add_subplot(1, 3, 1)
    #
    # cp = ax.contourf(xx, yy, ex, 50, cmap='jet')
    # cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
    # ax.set_xlabel( r'$x$', rotation='horizontal' )
    # ax.set_ylabel( r'$y$', rotation='horizontal' )
    # ax.set_title ( r'$\phi_{ex}(x,y)$' )
    #
    # ax = fig.add_subplot(1, 3, 2)
    # cp2 = ax.contourf(xx, yy, num, 50, cmap='jet')
    # cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)
    #
    # ax.set_xlabel( r'$x$', rotation='horizontal' )
    # ax.set_ylabel( r'$y$', rotation='horizontal' )
    # ax.set_title ( r'$\phi(x,y)$' )
    #
    # ax = fig.add_subplot(1, 3, 3)
    # cp3 = ax.contourf(xx, yy, err, 50, cmap='jet')
    # cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)
    #
    # ax.set_xlabel( r'$x$', rotation='horizontal' )
    # ax.set_ylabel( r'$y$', rotation='horizontal' )
    # ax.set_title ( r'$\phi(x,y) - \phi_{ex}(x,y)$' )
    # plt.show()



    # next test:

    #+++++++++++++++++++++++++++++++
    # . test Poisson solver
    #+++++++++++++++++++++++++++++++

    # x,y = domain.coordinates
    # solution = x**2 + y**2
    # f        = -4
    #
    # v = element_of(derham.V0, 'v')
    # l = LinearForm(v, f*v)
    # b = discretize(l, domain_h, V0h)
    #
    # D0T_M1_D0 = ComposedLinearOperator( D0_transp, ComposedLinearOperator( M1,D0 ) )
    #
    # A = D0T_M1_D0 + (I0 - Pconf_0)
    #
    # solution, info = cg( A, b, tol=1e-13, verbose=True )
    #
    # l2_error, h1_error = run_poisson_2d(solution, f, domain, )
    #
    # # todo: plot the solution for visual check
    #
    # print(l2_error)




if __name__ == '__main__':

    conga_operators_2d()

