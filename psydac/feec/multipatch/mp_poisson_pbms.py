# coding: utf-8

# Multipatch Poisson problems (source, eigenvalue) solved with a Conga or Nitsche method

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigs, eigsh

from collections import OrderedDict

import scipy.sparse.linalg as scipy_solvers

from sympde.topology import Derham, Union
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping
from sympde.topology import Boundary, NormalVector

from sympde.expr.expr import Norm
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral
from sympde.expr      import find, EssentialBC
from sympde.calculus  import grad, dot, inner, rot, div
from sympde.calculus  import jump, avg, Dn, minus, plus


from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.utilities import array_to_stencil
from psydac.utilities.utils    import refine_array_1d
from psydac.fem.basic   import FemField
from psydac.api.discretization import discretize as api_discretize  # MCP: there's a conflict with feec.multipatch.api.discretize ...

from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.api import discretize
from psydac.feec.multipatch.operators import BrokenMass
from psydac.feec.multipatch.operators import ConformingProjection_V0
from psydac.feec.multipatch.operators import get_patch_index_from_face
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_grid_vals_scalar, get_patch_knots_gridlines, my_small_plot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain

comm = MPI.COMM_WORLD

#==============================================================================
def solve_poisson_2d(conga=True, domain=None, ncells=None, degree=None, nb_eigs=None, strong_penalization=True):
    """
    solves a Poisson problem (source or eigenvalue) with a Conga or a Nitsche method
    """

    if conga:
        method_name = "Conga"
    else:
        method_name = "Nitsche"

    assert ncells and degree and domain

    eigenproblem = (nb_eigs > 0)

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())
    F = [f.get_callable_mapping() for f in mappings_list]

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    #+++++++++++++++++++++++++++++++
    # . Discrete space
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1

    # Mass matrices for broken spaces (block-diagonal)
    M0 = BrokenMass(V0h, domain_h, is_scalar=True)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)

    # Projectors for broken spaces
    # nquads = [d + 1 for d in degree]
    # bP0, bP1, bP2 = derham_h.projectors(nquads=nquads)

    # Broken derivative operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators

    #+++++++++++++++++++++++++++++++
    # . exact solution and source
    #+++++++++++++++++++++++++++++++

    from sympy import cos, sin
    x,y       = domain.coordinates
    phi_exact = sin(x)*cos(y)
    f         = 2*phi_exact

    from sympy import lambdify
    phi_ex = lambdify(domain.coordinates, phi_exact)
    f_ex   = lambdify(domain.coordinates, f)

    # V0 pull-backs on logical domain
    phi_ex_log = [lambda xi1, xi2,ff=f : phi_ex(*ff(xi1,xi2)) for f in F]
    f_log      = [lambda xi1, xi2,ff=f : f_ex(*ff(xi1,xi2)) for f in F]

    v  = element_of(derham.V0, 'v')
    u  = element_of(derham.V0, 'u')

    error  = u - phi_exact  # u is phi ....

    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    l2norm_h = discretize(l2norm, domain_h, V0h)
    h1norm_h = discretize(h1norm, domain_h, V0h)

    # RHS (L2 projection here)
    l  = LinearForm(v,  integral(domain, f*v))
    lh = discretize(l, domain_h, V0h)
    b  = lh.assemble()

    poisson_tol = 5e-13

    if conga:

        if strong_penalization:
            # jump penalization factor from Buffa, Perugia and Warburton
            nc = ncells[0]
            deg = degree[0]
            h = 1/nc
            DG_gamma = 10*(deg+1)**2/h
            # DG_gamma = 10*(deg)**2/h
            gamma_jump = DG_gamma
        else:
            gamma_jump = 1

        #+++++++++++++++++++++++++++++++
        # . Multipatch operators
        #+++++++++++++++++++++++++++++++

        # phi_ref = bP0(phi_ex_log)

        # Conforming projection V0 -> V0
        ## note: there are problems (eg at the interface) when the conforming projection is not accurate (low penalization or high tolerance)
        cP0     = ConformingProjection_V0(V0h, domain_h)
        cP0_hom = ConformingProjection_V0(V0h, domain_h, hom_bc=True)

        I0 = IdLinearOperator(V0h)

        D0_hom = ComposedLinearOperator([bD0,cP0_hom])

        M0_m = M0.to_sparse_matrix()
        M1_m = M1.to_sparse_matrix()
        bD0_m = bD0.to_sparse_matrix()
        cP0_m = cP0.to_sparse_matrix()
        cP0_hom_m = cP0_hom.to_sparse_matrix()
        I0_m = I0.to_sparse_matrix()

        D0_hom_m = bD0_m * cP0_hom_m
        D0_m = bD0_m * cP0_m

        # Conga Poisson matrix is
        # A = (cD0)^T * M1 * cD0 + alpha_jump * (I0 - Pc)^T * M0 * (I0 - Pc)    ## but here, Pc is a symmetric matrix
        #     + gamma_jump * jump_penal_m.transpose() * M1_m * jump_penal_m
        #
        # cD0T_M1_cD0_hom = ComposedLinearOperator([cP0_hom, bD0.transpose(), M1, bD0, cP0_hom])
        # A_hom = ComposedLinearOperator([(I0-cP0_hom), M0, (I0-cP0_hom)]) + cD0T_M1_cD0_hom
        #
        # cD0T_M1_cD0_hom = ComposedLinearOperator([cP0_hom, bD0.transpose(), M1, bD0, cP0_hom])

        A_hom_m = gamma_jump * (I0_m-cP0_hom_m) * M0_m * (I0_m-cP0_hom_m) + D0_hom_m.transpose() * M1_m * D0_hom_m
        A_m     = gamma_jump * (I0_m-cP0_m    ) * M0_m * (I0_m-cP0_m    ) + D0_m.transpose()     * M1_m * D0_m

        if not eigenproblem:

            # apply boundary conditions
            boundary = Union(*[j for i in domain.interior for j in i.boundary])
            a0 = BilinearForm((u,v), integral(boundary, u*v))
            l0 = LinearForm(v, integral(boundary, phi_exact*v))

            a0_h = discretize(a0, domain_h, [V0h, V0h])
            l0_h = discretize(l0, domain_h, V0h)

            x0, info = cg(a0_h.assemble(), l0_h.assemble(), tol=poisson_tol)
            x0 = x0.toarray()

            x0 = cP0_m.dot(x0)-cP0_hom_m.dot(x0)
            # x0 = x0 - cP0_hom_m.dot(x0)   # should also work since x0 is continuous

            ## other option: get the lifted boundary solution by direct interpolation of the boundary data:
            # bP0, bP1, bP2 = derham_h.projectors(nquads=nquads)
            # lambdify bP0
            # x0 = bP0(phi_exact)

            b = b.toarray()
            b = b - A_m.dot(x0)
            # ...

            print("solving Poisson equation with the Conga method...")
            x = scipy_solvers.spsolve(A_hom_m, b)
            phi_coeffs = cP0_hom_m.dot(x) + x0

            phi_coeffs = array_to_stencil(phi_coeffs, V0h.vector_space)
            phi_h = FemField(V0h, coeffs=phi_coeffs)

        else:
            phi_h = None

    else:
        # Nitsche's method

        nn   = NormalVector('nn')
        bc   = EssentialBC(u, phi_exact, domain.boundary)
        I = domain.interfaces
        kappa  = 10**3

        #    expr_I = - kappa*plus(u)*minus(v)\
        #             - kappa*plus(v)*minus(u)\
        #             + kappa*minus(u)*minus(v)\
        #             + kappa*plus(u)*plus(v)

        expr_I =- 0.5*dot(grad(plus(u)),nn)*minus(v)  + 0.5*dot(grad(minus(v)),nn)*plus(u)  - kappa*plus(u)*minus(v)\
                + 0.5*dot(grad(minus(u)),nn)*plus(v)  - 0.5*dot(grad(plus(v)),nn)*minus(u)  - kappa*plus(v)*minus(u)\
                - 0.5*dot(grad(minus(v)),nn)*minus(u) - 0.5*dot(grad(minus(u)),nn)*minus(v) + kappa*minus(u)*minus(v)\
                - 0.5*dot(grad(plus(v)),nn)*plus(u)   - 0.5*dot(grad(plus(u)),nn)*plus(v)   + kappa*plus(u)*plus(v)

        expr   = dot(grad(u),grad(v))

        a = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I))
        l  = LinearForm(v,  integral(domain, f*v))

        if eigenproblem:
            ah = discretize(a, domain_h, [V0h, V0h])
            A = ah.assemble()
            # MCP: consider the matrix for the homogeneous problem ?
            A_m = A.tosparse()

        else:
            # solving source problem
            print("solving Poisson equation with Nitsche's method...")
            equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)
            equation_h = discretize(equation, domain_h, [V0h, V0h])
            phi_h, info  = equation_h.solve(info=True, tol=1e-14)

    if eigenproblem:
        # Find eigenmodes and eigenvalues with scipy.sparse
        # A = A.to_sparse_matrix()
        M0 = M0.to_sparse_matrix()
        # eigenvalues, eigenvectors = eigs(A, k=nb_eigs, which='SM' )   # 'SM' = smallest magnitude
        ncv = 4*nb_eigs
        # mode='cayley'
        mode='normal'
        print("solving Poisson eigenvalue problem with the "+method_name+" method...")
        eigenvalues, eigenvectors = eigsh(A_m, k=nb_eigs, M=M0, sigma=1, mode=mode, which='LM', ncv=ncv)

        print("eigenvalues:")
        print(eigenvalues)

    else:

        l2_error = l2norm_h.assemble(u=phi_h)
        h1_error = h1norm_h.assemble(u=phi_h)

        print( '> L2 error      :: {:.2e}'.format( l2_error ) )
        print( '> H1 error      :: {:.2e}'.format( h1_error ) )

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # VISUALIZATION
        #   adapted from examples/poisson_2d_multi_patch.py and
        #   and psydac/api/tests/test_api_feec_2d.py
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        N=20
        etas, xx, yy = get_plotting_grid(mappings, N)
        gridlines_x1, gridlines_x2 = get_patch_knots_gridlines(V0h, N, mappings, plotted_patch=1)

        grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')

        # phi_ref_vals = get_grid_vals_scalar(phi_ex_log, etas, domain, list(mappings.values())) #_obj)
        # phi_h_vals   = get_grid_vals_scalar(phi_h, etas, domain, list(mappings.values())) #


        phi_ref_vals = grid_vals_h1(phi_ex_log)
        phi_h_vals   = grid_vals_h1(phi_h)
        phi_err = [abs(pr - ph) for pr, ph in zip(phi_ref_vals, phi_h_vals)]

        print([np.array(a).max() for a in phi_err])
        my_small_plot(
            title=r'Solution of Poisson problem $\Delta \phi = f$',
            vals=[phi_ref_vals, phi_h_vals, phi_err],
            titles=[r'$\phi^{ex}(x,y)$', r'$\phi^h(x,y)$', r'$|(\phi-\phi^h)(x,y)|$'],
            xx=xx, yy=yy,
            gridlines_x1=gridlines_x1,
            gridlines_x2=gridlines_x2,
            surface_plot=True,
            cmap='jet',
        )



if __name__ == '__main__':

    ## main parameters ------------------------------------------------------------------------------------------------

    nc = 2**2
    deg = 2

    # nb of computed eigenvalues -- if 0, solve the source problem
    nb_eigs = 10

    ## chose method:
    method = 'Conga'
    # method = 'Nitsche'

    strong_penalization = True # only for Conga for now

    ## chose domain:
    # domain_name = 'pretzel'
    domain_name = 'curved_L_shape'

    if method == 'Conga':
        conga = True
    elif method == 'Nitsche':
        conga = False
    else:
        raise ValueError(method)

    #+++++++++++++++++++++++++++++++
    # . Domain
    #+++++++++++++++++++++++++++++++

    domain = build_multipatch_domain(domain_name=domain_name, n_patches=None)

    solve_poisson_2d(
        nb_eigs=nb_eigs,
        conga=conga, strong_penalization=strong_penalization,
        domain=domain, ncells=[nc, nc], degree=[deg, deg],
    )


