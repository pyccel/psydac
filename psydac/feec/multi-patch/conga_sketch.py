# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

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
from psydac.fem.vector import ProductFemSpace, VectorFemField

from psydac.feec.derivatives import Gradient_2D
from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl

from psydac.utilities.utils    import refine_array_1d


comm = MPI.COMM_WORLD


#===============================================================================
class ConformingProjection( LinearOperator ):
    """
    Conforming projection from global broken space to conforming global space

    proj.dot(v) returns the conforming projection of v, computed by solving linear system

    Parameters
    ----------
    todo

    """
    def __init__( self, V0h_1, V0h_2, domain_h_1, domain_h_2, V0h, domain_h):

        V0 = V0h.symbolic_space
        domain = V0.domain
        # domain_h = V0h.domain  # would be nice
        
        self._domain   = V0h
        self._codomain = V0h

        u, v = elements_of(V0, names='u, v')
        expr   = u*v  # dot(u,v)

        kappa  = 1000 # 10**20
        I = domain.interfaces  # note: interfaces does not include the boundary 
        expr_I = kappa*( plus(u)-minus(u) )*( plus(v)-minus(v) )   # this penalization is for an H1-conforming space

        a = BilinearForm((u,v), integral(domain, expr) + integral(I, expr_I))

        ah = discretize(a, domain_h, [V0h, V0h])    # ... or (V0h, V0h)?

        self._A = ah.assemble() #.toarray()
        self._solver = SparseSolver( self._A.tosparse() )

        V0_1 = V0h_1.symbolic_space
        V0_2 = V0h_2.symbolic_space

        v1, f1 = elements_of(V0_1, names='v1, f1')
        v2, f2 = elements_of(V0_2, names='v2, f2')
 
        l1 = LinearForm(v1, integral(V0_1.domain, f1*v1))
        l2 = LinearForm(v2, integral(V0_2.domain, f2*v2))

        self._lh_1 = discretize(l1, domain_h_1, V0h_1)
        self._lh_2 = discretize(l2, domain_h_2, V0h_2)

    @property
    def domain( self ):
        return self._domain

    @property
    def codomain( self ):
        return self._codomain


    def __call__( self, f ):

        f1,f2 = f.fields

        b1 = self._lh_1.assemble(f1=f1)
        b2 = self._lh_2.assemble(f2=f2)
        b  = BlockVector(self.codomain.vector_space, blocks=[b1, b2])

        # sol_coeffs, info = cg( self._A, b, tol=1e-13, verbose=True )
        sol_coeffs, info = pcg( self._A, b, pc="jacobi", tol=1e-6, verbose=True )  # doesn't cv
        #
        # sol_coeffs = self._solver.solve( b )

        return  VectorFemField(self.codomain, coeffs=sol_coeffs)

    def dot( self, f_coeffs, out=None ):

        f = VectorFemField(self.domain, coeffs=f_coeffs)

        return self(f).coeffs

class ComposedLinearOperator( LinearOperator ):

    def __init__( self, B, A ):

        assert isinstance(A, LinearOperator)
        assert isinstance(B, LinearOperator)
        assert B.domain == A.codomain

        self._domain   = A.domain
        self._codomain = B.codomain

        self._A = A
        self._B = B

    def __call__( self, f ):

        return  self._B(self._A(f))

    def dot( self, f_coeffs, out=None ):

        return  self._B.dot(self._A.dot(f_coeffs))


def test_conga_2d():
    """
    assembles a conforming projection on the conforming subspace,
    and the corresponding gradient operator on the broken multipatch space

    performs two tests:
      - commuting diagram
      -

    """


    #+++++++++++++++++++++++++++++++
    # . Domain
    #+++++++++++++++++++++++++++++++

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    domain_1     = mapping_1(A)
    domain_2     = mapping_2(B)

    # local (single patch) de Rham sequences:
    derham_1  = Derham(domain_1, ["H1", "Hcurl"])
    derham_2  = Derham(domain_2, ["H1", "Hcurl"])

    domain = domain_1.join(domain_2, name = 'domain',
                bnd_minus = domain_1.get_boundary(axis=1, ext=1),
                bnd_plus  = domain_2.get_boundary(axis=1, ext=-1))

    mappings  = {A.interior:mapping_1, B.interior:mapping_2}

    # multi-patch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl"])



    #+++++++++++++++++++++++++++++++
    # . Discrete space
    #+++++++++++++++++++++++++++++++

    ncells=[2**2, 2**2]
    degree=[2,2]

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    
    ## derham for later
    # derham_h = discretize(derham, domain_h, degree=degree)      # build them by hand if this doesn't work
    # V0h       = derham_h.V0
    # V1h       = derham_h.V1

    # broken multipatch spaces
    V0h = discretize(derham.V0, domain_h, degree=degree)
    V1h = discretize(derham.V1, domain_h, degree=degree)

    assert isinstance(V1h, ProductFemSpace)
    assert isinstance(V1h.vector_space, BlockVectorSpace)

    # local construction

    domain_h_1 = discretize(domain_1, ncells=ncells, comm=comm)
    domain_h_2 = discretize(domain_2, ncells=ncells, comm=comm)

    V0h_1 = discretize(derham_1.V0, domain_h_1, degree=degree)
    V0h_2 = discretize(derham_2.V0, domain_h_2, degree=degree)
    V1h_1 = discretize(derham_1.V1, domain_h_1, degree=degree)
    V1h_2 = discretize(derham_2.V1, domain_h_2, degree=degree)

    # try also
    patch_domains = [domain_1, domain_2]
    patch_domains_h = [discretize(patch_domain, ncells=ncells, comm=comm) for patch_domain in patch_domains]
    patch_derhams  = [Derham(patch_domain, ["H1", "Hcurl"]) for patch_domain in patch_domains]
    patch_derhams_h = [discretize(patch_derham, patch_domain_h, degree=degree)
                 for patch_domain_h, patch_derham in zip(patch_domains_h, patch_derhams)]

    patch_V0s_h = [  discretize(patch_derham.V0, patch_domain_h, degree=degree)
                  for patch_domain_h, patch_derham in zip(patch_domains_h, patch_derhams)]

    # patch_V0h = patch_derhams_h[0].V0
    # print("*-*:", patch_V0h)
    #
    # patch_V0h = patch_V0s_h[0]
    # print("*+*:", patch_V0h)
    # exit()


    V0h_vector_space = BlockVectorSpace(V0h_1.vector_space, V0h_2.vector_space)
    V1h_vector_space = BlockVectorSpace(V1h_1.vector_space, V1h_2.vector_space)

    # print("V0h_vector_space: ", V0h_vector_space)
    # print("V0h.vector_space: ", V0h.vector_space)
    # print("%%%%%%%%% %%%%%%%%% %%%%%%%%%")

    if 0:
        # alternative (but with vector spaces)
        V0h_1 = V0h.spaces[0]  # V0h on domain 1
        V0h_2 = V0h.spaces[1]  # V0h on domain 2
        V1h_1 = BlockVectorSpace(V1h.spaces[0], V1h.spaces[1])  # V1h on domain 1
        V1h_2 = BlockVectorSpace(V1h.spaces[2], V1h.spaces[3])  # V1h on domain 2

    #+++++++++++++++++++++++++++++++
    # . some target functions
    #+++++++++++++++++++++++++++++++


    x,y       = domain.coordinates
    solution  = x**2 + y**2
    solution_aux = 0

    from sympy import lambdify
    solution = lambdify(domain.coordinates, solution)
    solution_aux = lambdify(domain.coordinates, solution_aux)

    fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)
    D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)

    # fun2    = lambda xi1, xi2 : .5*np.sin(xi1)*np.sin(xi2)


    # pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings, etas)]

    # print(mapping_1.items()[1])
    # M.logical_coordinates, M.expressions) for d,M in mappings.items()]
    F1 = mapping_1.get_callable_mapping()
    F2 = mapping_2.get_callable_mapping()
    F = [F1,F2]

    solution_log_1 = lambda xi1, xi2 : solution(*F1(xi1,xi2))
    solution_log_2 = lambda xi1, xi2 : solution_aux(*F2(xi1,xi2))


    #+++++++++++++++++++++++++++++++
    # . H1, Hcurl (commuting) projectors
    #+++++++++++++++++++++++++++++++

    # create an instance of the H1 projector class
    # P0 = Projector_H1(V0h)   # todo
    # P1 = Projector_Hcurl(V1h)
    # u0        = P0(fun1)
    # u1        = P1((D1fun1, D2fun1))

    P0_1 = Projector_H1(V0h_1)
    P0_2 = Projector_H1(V0h_2)

    u0_1 = P0_1(solution_log_1)
    u0_2 = P0_2(solution_log_2)

    # Create empty vector FEM field, then copy coefficients from scalar fields
    u0 = VectorFemField(V0h)
    u0.coeffs[0][:] = u0_1.coeffs[:]
    u0.coeffs[1][:] = u0_2.coeffs[:]
    u0.coeffs.update_ghost_regions()

    n_quads = [5,5]
    P1_1 = Projector_Hcurl(V1h_1, n_quads)
    P1_2 = Projector_Hcurl(V1h_2, n_quads)

    u1_1 = P1_1((D1fun1, D2fun1))
    u1_2 = P1_2((D1fun1, D2fun1))




    #+++++++++++++++++++++++++++++++
    # . conf projection operators
    #+++++++++++++++++++++++++++++++

    test_Pconf = False

    if test_Pconf:

        # projection from broken multipatch space to conforming subspace (using the same basis)
        Pconf_0 = ConformingProjection(V0h_1, V0h_2, domain_h_1, domain_h_2, V0h, domain_h)
        # u0_conf   = Pconf_0(u0)
        u0c = Pconf_0(u0)



    #+++++++++++++++++++++++++++++++
    # . Differential operators
    #   on conforming and broken spaces
    #+++++++++++++++++++++++++++++++

    test_D0 = True

    # "broken grad" operator, coincides with the grad on the conforming subspace of V0h
    # later: broken_D0 = Gradient_2D(V0h, V1h)   # on multi-patch domains we should maybe provide the "BrokenGradient"

    if test_D0:

        # building multi-patch gradient by hand (later would be nice to have it as an operator between multi-patch spaces)
        D0_1 = Gradient_2D(V0h_1, V1h_1)
        D0_2 = Gradient_2D(V0h_2, V1h_2)

        du0_1 = D0_1(u0_1)
        du0_2 = D0_2(u0_2)

        # Create empty multi-patch FEM field, then copy coefficients from single-patch fields
        du0 = VectorFemField(V1h)

        # patch 1
        du0.coeffs[0][0][:] = du0_1[0].coeffs[:]
        du0.coeffs[0][1][:] = du0_1[1].coeffs[:]
        du0.coeffs.update_ghost_regions()

        # patch 2
        du0.coeffs[1][0][:] = du0_2[0].coeffs[:]
        du0.coeffs[1][1][:] = du0_2[1].coeffs[:]
        du0.coeffs.update_ghost_regions()

        # > this should allow to define a multi-patch operator: broken_D0

    test_conga_D0 = False
    if test_conga_D0:

        # Conga grad operator (on the broken V0h)
        conga_D0 = ComposedLinearOperator(broken_D0,Pconf_0)



    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # VISUALIZATION  adapted from examples/poisson_2d_multi_patch.py
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    N = 20

    etas     = [[refine_array_1d( bounds, N ) for bounds in zip(D.min_coords, D.max_coords)] for D in mappings]

    mappings = [lambdify(M.logical_coordinates, M.expressions) for d,M in mappings.items()]
    # solution = lambdify(domain.coordinates, solution)

    pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings, etas)]
    num     = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(u0.fields, etas)]
    ex      = [np.array( [[solution( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]
    if test_Pconf:
        num_c   = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(u0c.fields, etas)]

    elif test_D0:
        num_du_x = 2*[None]
        num_du_y = 2*[None]

        for k in [0,1]:
            # patch k
            eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
            for i, x1i in enumerate(eta_1[:, 0]):
                for j, x2j in enumerate(eta_2[0, :]):
                    num_du_x[k][i, j], num_du_y[k][i, j] = \
                        push_2d_hcurl(du0.fields[k][0], du0.fields[k][1], x1i, x2j, F[k])

                        # [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(du0.fields[0], etas)]
            #
            #         Ex_values[i, j], Ey_values[i, j] = \
            #             push_2d_hcurl(E.fields[0], E.fields[1], x1i, x2j, mapping)


    pcoords  = np.concatenate(pcoords, axis=1)
    num      = np.concatenate(num,     axis=1)
    ex       = np.concatenate(ex,      axis=1)
    err      = abs(num - ex)

    if test_Pconf:
        num_c    = np.concatenate(num_c,     axis=1)

    elif test_D0:
        num_du_x = np.concatenate(num_du_x,     axis=1)
        num_du_y = np.concatenate(num_du_y,     axis=1)

    xx = pcoords[:,:,0]
    yy = pcoords[:,:,1]

    plotted_patch = 1

    if plotted_patch is not None:
        assert plotted_patch in [0, 1]

        #patch_derham = patch_derhams_h[plotted_patch]
        grid_x1 = patch_derhams_h[plotted_patch].V0.breaks[0]
        grid_x2 = patch_derhams_h[plotted_patch].V0.breaks[1]

        print("grid_x1 = ", grid_x1)

        x1 = refine_array_1d(grid_x1, N)
        x2 = refine_array_1d(grid_x2, N)

        x1, x2 = np.meshgrid(x1, x2, indexing='ij')
        x, y = F[plotted_patch](x1, x2)

        print("x1 = ", x1)

        gridlines_x1 = (x[:, ::N],   y[:, ::N]  )
        gridlines_x2 = (x[::N, :].T, y[::N, :].T)
        gridlines = (gridlines_x1, gridlines_x2)


    fig = plt.figure(figsize=(17., 4.8))

    ax = fig.add_subplot(1, 3, 1)

    if plotted_patch is not None:
        ax.plot(*gridlines_x1, color='k')
        ax.plot(*gridlines_x2, color='k')
    cp = ax.contourf(xx, yy, ex, 50, cmap='jet')
    cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
    ax.set_xlabel( r'$x$', rotation='horizontal' )
    ax.set_ylabel( r'$y$', rotation='horizontal' )
    ax.set_title ( r'$u_{ex}(x,y)$' )


    ax = fig.add_subplot(1, 3, 2)
    cp2 = ax.contourf(xx, yy, num, 50, cmap='jet')
    cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)

    ax.set_xlabel( r'$x$', rotation='horizontal' )
    ax.set_ylabel( r'$y$', rotation='horizontal' )
    ax.set_title ( r'$u_h(x,y)$' )


    if test_Pconf:
        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, num_c, 50, cmap='jet')
        # cp3 = ax.contourf(xx, yy, err, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$u_c(x,y)$' )
        # ax.set_title ( r'$u_h(x,y) - u_{ex}(x,y)$' )


    elif test_D0:

        ax = fig.add_subplot(1, 3, 3)
        cp3 = ax.contourf(xx, yy, num_du_x, 50, cmap='jet')
        # cp3 = ax.contourf(xx, yy, err, 50, cmap='jet')
        cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)

        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )
        ax.set_title ( r'$du_x(x,y)$' )



    plt.show()










    #+++++++++++++++++++++++++++++++
    # . test commuting diagram
    #+++++++++++++++++++++++++++++++







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

    # I0 = IdentityLinearOperator(V0h)
    I0 = IdentityLinearOperator(V0h.vector_space)


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
    # later: broken_D0 = Gradient_2D(V0h, V1h)   # on multi-patch domains we should maybe provide the "BrokenGradient"
    # or broken_D0 = derham_h.D0 ?

    D0_1 = Gradient_2D(V0h_1, V1h_1)
    D0_2 = Gradient_2D(V0h_2, V1h_2)
    
    broken_D0 = BlockMatrix(V0h.vector_space, V1h.vector_space, blocks=[[D0_1, None],[None, D0_2]])
    
    # projection from broken multipatch space to conforming subspace
    Pconf_0 = ConformingProjection(V0h)

    # Conga grad operator (on the broken V0h)
    D0 = ComposedLinearOperator(broken_D0,Pconf_0)

    # Transpose of the Conga grad operator (using the symmetry of Pconf_0)
    D0_transp = ComposedLinearOperator(Pconf_0,broken_D0.T)


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

    test_conga_2d()

