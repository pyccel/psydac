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
from psydac.fem.vector import ProductFemSpace, VectorFemSpace, VectorFemField

from psydac.feec.pull_push     import push_2d_hcurl, pull_2d_hcurl  #, push_2d_l2

from psydac.feec.derivatives import Gradient_2D
from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl

from psydac.utilities.utils    import refine_array_1d

from psydac.feec.multipatch.operators import BrokenMass_V0, BrokenMass_V1, ortho_proj_Hcurl
from psydac.feec.multipatch.operators import IdLinearOperator, SumLinearOperator, MultLinearOperator
from psydac.feec.multipatch.operators import BrokenGradient_2D, BrokenTransposedGradient_2D
from psydac.feec.multipatch.operators import ConformingProjection, ComposedLinearOperator
from psydac.feec.multipatch.operators import Multipatch_Projector_H1, Multipatch_Projector_Hcurl
from psydac.feec.multipatch.operators import get_scalar_patch_fields, get_vector_patch_fields

comm = MPI.COMM_WORLD


def conga_poisson_2d():
    """
    - assembles several multipatch operators and a conforming projection

    - performs several tests:
      - ...
      -

    """


    #+++++++++++++++++++++++++++++++
    # . Domain
    #+++++++++++++++++++++++++++++++

    cartesian = True

    if cartesian:

        A = Square('A',bounds1=(0, 1), bounds2=(0, 0.5))
        B = Square('B',bounds1=(0, 1), bounds2=(0.5, 1))
        mapping_1 = IdentityMapping('M1', 2)
        mapping_2 = IdentityMapping('M2', 2)

    else:

        A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
        B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))
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
    domains_h = [discretize(dom, ncells=ncells, comm=comm) for dom in domains]
    derhams  = [Derham(dom, ["H1", "Hcurl", "L2"]) for dom in domains]
    derhams_h = [discretize(derh, dom_h, degree=degree)
                 for dom_h, derh in zip(domains_h, derhams)]

    V0hs = [  discretize(derh.V0, dom_h, degree=degree)
                  for dom_h, derh in zip(domains_h, derhams)]

    V1hs = [  discretize(derh.V1, dom_h, degree=degree, basis='M')
                  for dom_h, derh in zip(domains_h, derhams)]


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
    if cartesian:
        phi_exact = x*(1-x)*y*(1-y)
        f         = -2*(y*(1-y)+x*(1-x))

    else:
        phi_exact = (x**2 + y**2 - 1/4)*(x**2 + y**2 - 1)
        f         = (x+y)*(2*(x**2 + y**2)-5/4) + 8*(x**2 + y**2)

    # u_solution  = x**2 + y**2

    # affine field, for testing
    u_exact_x = 2*x
    u_exact_y = 2*y

    from sympy import lambdify
    phi_ex = lambdify(domain.coordinates, phi_exact)
    u_ex_x = lambdify(domain.coordinates, u_exact_x)
    u_ex_y = lambdify(domain.coordinates, u_exact_y)

    # pull-back of phi_ex
    phi_ex_log = [lambda xi1, xi2 : phi_ex(*f(xi1,xi2)) for f in F]

    # list: E_sol_log[k][d] : xi1, xi2 -> E_x(xi1, xi2) on patch k
    # E_sol_log = [pull_2d_hcurl([E_sol_x,E_sol_y], f) for f in mappings_obj]

    # discontinuous target
    # v_sol_log = [
    #     lambda xi1, xi2 : 1, # u_sol(*F[0](xi1,xi2)),
    #     lambda xi1, xi2 : 0,
    #     ]

    #+++++++++++++++++++++++++++++++
    # . Multipatch H1, Hcurl (commuting) projectors
    #+++++++++++++++++++++++++++++++

    # I. multipatch V0 projection

    P0 = Multipatch_Projector_H1(V0hs, V0h)
    phi = P0(phi_ex_log)

    # u0 = P0(u_sol_log)
    # v0 = P0(v_sol_log)

    # II. conf projection V0 -> V0

    Pconf_0 = ConformingProjection(V0hs[0], V0hs[1], domains_h[0], domains_h[1], V0h, domain_h, verbose=False)
    # test_Pc = False

    # if test_Pc:
        # projection from broken multipatch space to conforming subspace (using the same basis)
        # v0c = Pconf_0(v0)

    # validation of M1 matrix

    # P1 = Multipatch_Projector_Hcurl(V1hs, V1h, nquads=[5,5])

    # from sympy import Tuple
    # EE = Tuple(E_solution_x, E_solution_y)
    # E1 = ortho_proj_Hcurl(EE, V1h, domain_h, M1)

    # multipatch (broken) grad operator on V0h

    bD0 = BrokenGradient_2D(V0hs, V1hs, V0h, V1h)
    # grad_u0 = bD0(u0)

    # V. Conga grad operator on V0h

    cD0 = ComposedLinearOperator(bD0,Pconf_0)
    # cDv0 = cD0(v0)

    # Transpose of the Conga grad operator (using the symmetry of Pconf_0)
    bD0_T = BrokenTransposedGradient_2D(V0hs, V1hs, V0h, V1h)
    cD0_T = ComposedLinearOperator(Pconf_0,bD0_T)

    I0 = IdLinearOperator(V0h)
    # I0 = IdentityLinearOperator(V0h.vector_space)


    #+++++++++++++++++++++++++++++++
    # . Conga Poisson solver
    #+++++++++++++++++++++++++++++++


    # x,y = domain.coordinates
    # solution = x**2 + y**2

    # f        = -4

    v = element_of(derham.V0, 'v')
    l = LinearForm(v,  integral(domain, f*v))
    lh = discretize(l, domain_h, V0h)
    b = lh.assemble()

    cD0T_M1_cD0 = ComposedLinearOperator( cD0_T, ComposedLinearOperator( M1, cD0 ) )
    # A = cD0T_M1_cD0 + (I0 - Pconf_0)
    minus_cP0 = MultLinearOperator(-1,Pconf_0)
    I_minus_cP0 = SumLinearOperator( I0, minus_cP0 )
    A = SumLinearOperator( cD0T_M1_cD0, I_minus_cP0 )

    solve_poisson = False
    if solve_poisson:
        # then the desired 'solution' is phi, the solution of the Poisson equation

        sol_coeffs, info = cg( A, b, tol=1e-3, verbose=True )
        sol_h = VectorFemField(V0h, coeffs=sol_coeffs)
        sol_h = Pconf_0(sol_h)
        sol_exact = phi_exact
    else:
        # not solving Poisson, just checking direct equation
        # so the desired 'solution' is f, the source term

        # \int grad P u -> f ?
        # sol_coeffs = A.dot(u0.coeffs)

        b = cD0T_M1_cD0.dot(phi.coeffs)
        sol_coeffs, info = pcg(M0.mat(), b, pc="jacobi", tol=1e-10)
        sol_h = VectorFemField(V0h, coeffs=sol_coeffs)
        sol_exact = f
        #sol_h = Pconf_0(sol_h)

    sol_ex = lambdify(domain.coordinates, sol_exact)





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


        # solution
        sol_vals  = [np.array( [[sol_ex( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
        sol_vals  = np.concatenate(sol_vals,     axis=1)

        # Poisson sol_h
        sols_h = get_scalar_patch_fields(sol_h, V0hs)
        sol_h_vals = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(sols_h, etas)]
        sol_h_vals  = np.concatenate(sol_h_vals,     axis=1)
        sol_err = abs(sol_vals - sol_h_vals)




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

        fig = plt.figure(figsize=(17., 4.8))
        fig.suptitle(r'approximation of some $v$', fontsize=14)

        ax = fig.add_subplot(1, 3, 1)

        if plotted_patch is not None:
            ax.plot(*gridlines_x1, color='k')
            ax.plot(*gridlines_x2, color='k')

        cp = ax.contourf(xx, yy, sol_vals, 50, cmap='jet')
        # cp = ax.contourf(xx, yy, sol_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$u^h(x,y)$' )
        # ax.set_title ( r'$\phi^{ex}(x,y)$' )


        ax = fig.add_subplot(1, 3, 2)
        cp2 = ax.contourf(xx, yy, sol_h_vals, 50, cmap='jet')
        cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$\Delta_h u^h(x,y)$' )
        # ax.set_title ( r'$\phi^h(x,y)$' )

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, sol_err, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$|(\phi^{ex}-\phi^h)(x,y)|$' )

        plt.show()


    exit()



if __name__ == '__main__':

    conga_poisson_2d()

