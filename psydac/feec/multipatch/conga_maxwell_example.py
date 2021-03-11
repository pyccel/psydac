# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

# checking import path...
# import sympde
# print(sympde.__file__)
# exit()

from sympy import pi, cos, sin, Matrix, Tuple
from sympy import symbols
from sympy import lambdify

from sympde.calculus import grad, dot, inner, rot, div, curl, cross
from sympde.topology import NormalVector
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus
from sympde.expr import Norm

from sympde.topology import Derham
# from sympde.topology import ProductSpace
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

#from psydac.api.discretization import discretize
from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above

from psydac.api.tests.test_api_2d_system_mapping_multipatch import run_maxwell_2d

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

from psydac.feec.multipatch.operators import BrokenMass_V0, BrokenMass_V1, BrokenMass_V2, ortho_proj_Hcurl
from psydac.feec.multipatch.operators import IdLinearOperator, SumLinearOperator, MultLinearOperator
from psydac.feec.multipatch.operators import BrokenGradient_2D, BrokenTransposedGradient_2D
from psydac.feec.multipatch.operators import ConformingProjection_V1, ComposedLinearOperator
from psydac.feec.multipatch.operators import Multipatch_Projector_H1, Multipatch_Projector_Hcurl
from psydac.feec.multipatch.operators import get_grid_vals_V0, get_grid_vals_V1, get_grid_vals_V2
from psydac.feec.multipatch.operators import my_small_plot

comm = MPI.COMM_WORLD

from psydac.api.essential_bc import *
from sympde.topology      import Boundary, Interface


#==============================================================================
def run_conga_maxwell_2d(uex, f, alpha, domain, ncells, degree, comm=None):
    """
    - assemble and solve a Maxwell problem on a multipatch domain, using Conga approach
    - this problem is adapted from the single patch test_api_system_3_2d_dir_1
    """

    maxwell_tol = 5e-13
    nquads = [d + 1 for d in degree]

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    # Mass matrices for broken spaces (block-diagonal)
    # TODO: (MCP 10.03.2021) define them as Hodge FemLinearOperators
    M0 = BrokenMass_V0(V0h, domain_h)
    M1 = BrokenMass_V1(V1h, domain_h)
    M2 = BrokenMass_V2(V2h, domain_h)

    # Projectors for broken spaces
    # bP0, bP1, bP2 = derham_h.projectors(nquads=nquads)

    # Broken derivative operators
    bD0, bD1, bD2 = derham_h.broken_derivatives_as_operators

    # Transposed operator(s)
    bD1_T = bD1.transpose()

    cP1 = ConformingProjection_V1(V1h, domain_h)


    # todo (MCP): simplify using matrices

    # Conga grad operator on V0h
    cD1 = ComposedLinearOperator(bD1, cP1)

    # Transpose of the Conga grad operator (using the symmetry of Pconf_0)
    cD1_T = ComposedLinearOperator(cP1, bD1_T)

    I1 = IdLinearOperator(V1h)

    cD1T_M2_cD1 = ComposedLinearOperator( cD1_T, ComposedLinearOperator( M2, cD1 ) )
    alpha_M1   = MultLinearOperator(alpha,M1)
    A_aux = SumLinearOperator( cD1T_M2_cD1, alpha_M1)
    minus_cP1   = MultLinearOperator(-1,cP1)
    I_minus_cP1 = SumLinearOperator( I1, minus_cP1 )
    I_minus_cP1_squared = ComposedLinearOperator(I_minus_cP1, I_minus_cP1)

    A1 = SumLinearOperator( A_aux, I_minus_cP1) #_squared )

    # boundary conditions
    u, v, F  = elements_of(V, names='u, v, F')
    nn  = NormalVector('nn')
    # error   = Matrix([F[0]-uex[0],F[1]-uex[1]])
    # kappa        = 10**8*degree[0]**2*ncells[0]
    penalization = 10**7
    boundary = domain.boundary

    # todo (MCP): simplify
    expr_b = penalization * cross(u, nn) * cross(v, nn)
    a_b = BilinearForm((u,v), integral(boundary, expr_b))
    a_b_h = discretize(a_b, domain_h, [V1h, V1h])
    A_b = FemLinearOperator(fem_domain=V1h, fem_codomain=V1h, matrix=a_b_h.assemble())
    A = SumLinearOperator(A1, A_b)

    expr   = dot(f,v)
    expr_b = penalization * cross(uex, nn) * cross(v, nn)

    l = LinearForm(v, integral(domain, expr) + integral(boundary, expr_b))
    lh = discretize(l, domain_h, V1h)
    b  = lh.assemble()

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system

    u_coeffs, info = pcg( A, b, pc='jacobi', tol=maxwell_tol, verbose=True )

    uh = FemField(V1h, coeffs=u_coeffs)

    # error
    error   = Matrix([F[0]-uex[0],F[1]-uex[1]])
    l2_norm = Norm(error, domain, kind='l2')
    l2_norm_h   = discretize(l2_norm, domain_h, V1h)
    l2_error = l2_norm_h.assemble(F=uh)

    return l2_error, uh

from psydac.api.tests.test_api_2d_compatible_spaces import test_api_system_3_2d_dir_1

if __name__ == '__main__':


    bounds1   = (0.5, 1.)
    bounds2_A = (0, np.pi/2)
    bounds2_B = (np.pi/2, np.pi)

    A = Square('A',bounds1=bounds1, bounds2=bounds2_A)
    B = Square('B',bounds1=bounds1, bounds2=bounds2_B)

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    domain = D1.join(D2, name = 'domain',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    x,y    = domain.coordinates
    alpha    = 1.
    uex      = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f        = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                     alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))


    ## call solver

    # l2_error, uh = run_maxwell_2d(uex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2,2], return_sol=True)

    l2_error, uh = run_conga_maxwell_2d(uex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2,2], return_sol=True)

    print("max2d: ", l2_error)
    # print("conga_max2d: ", conga_l2_error)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # VISUALIZATION
    #   adapted from examples/poisson_2d_multi_patch.py and
    #   and psydac/api/tests/test_api_feec_2d.py
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    N = 20

    mappings  = [mapping_1, mapping_2]
    etas     = [[refine_array_1d( bounds, N ) for bounds in zip(D.min_coords, D.max_coords)] for D in mappings]

    # mappings_lambda = [lambdify(M.logical_coordinates, M.expressions) for d,M in mappings.items()]
    mappings_lambda = [lambdify(M.logical_coordinates, M.expressions) for M in mappings]

    pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings_lambda, etas)]
    pcoords  = np.concatenate(pcoords, axis=1)


    # plots

    xx = pcoords[:,:,0]
    yy = pcoords[:,:,1]

    # plot one patch grid
    plotted_patch = -1
    if plotted_patch in [0, 1]:

        #patch_derham = derhams_h[plotted_patch]
        grid_x1 = V0h.spaces[plotted_patch].breaks[0]
        grid_x2 = V0h.spaces[plotted_patch].breaks[1]

        print("grid_x1 = ", grid_x1)

        x1 = refine_array_1d(grid_x1, N)
        x2 = refine_array_1d(grid_x2, N)

        x1, x2 = np.meshgrid(x1, x2, indexing='ij')
        x, y = F[plotted_patch](x1, x2)

        print("x1 = ", x1)

        gridlines_x1 = (x[:, ::N],   y[:, ::N]  )
        gridlines_x2 = (x[::N, :].T, y[::N, :].T)
        gridlines = (gridlines_x1, gridlines_x2)
    else:
        gridlines_x1 = None
        gridlines_x2 = None

    u_x_vals, u_y_vals   = get_grid_vals_V1(uex, None, etas, mappings)
    uh_x_vals, uh_y_vals = get_grid_vals_V1(uh, V1h, etas, mappings)

    u_x_err = abs(u_x_vals - uh_x_vals)
    u_y_err = abs(u_y_vals - uh_y_vals)

    my_small_plot(
        title=r'approximation of solution $u$, $x$ component',
        vals=[u_x_vals, uh_x_vals, u_x_err],
        titles=[r'$u^{ex}_x(x,y)$', r'$E^h_x(x,y)$', r'$|(E^{ex}-E^h)_x(x,y)|$'],
        xx=xx,
        yy=yy,
        gridlines_x1=gridlines_x1,
        gridlines_x2=gridlines_x2,
    )

