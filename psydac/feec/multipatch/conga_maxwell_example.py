# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from sympy import pi, cos, sin, Matrix, Tuple
from sympy import symbols
from sympy import lambdify

from sympde.calculus import grad, dot, inner, rot, div, curl, cross
from sympde.topology import NormalVector
from sympde.expr import Norm

from sympde.topology import Derham
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above

from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.feec.pull_push     import push_2d_hcurl, pull_2d_hcurl

from psydac.utilities.utils    import refine_array_1d

from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, ortho_proj_Hcurl
from psydac.feec.multipatch.operators import ConformingProjection_V1
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_patch_knots_gridlines, my_small_plot
from psydac.feec.multipatch.multipatch_domain_utilities import get_pretzel

comm = MPI.COMM_WORLD

#==============================================================================
def run_conga_maxwell_2d(uex, f, alpha, domain, ncells, degree, comm=None, return_sol=False):
    """
    - assemble and solve a Maxwell problem on a multipatch domain, using Conga approach
    - this problem is adapted from the single patch test_api_system_3_2d_dir_1
    """

    use_scipy = True
    maxwell_tol = 5e-3
    nquads = [d + 1 for d in degree]

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V1h = derham_h.V1
    V2h = derham_h.V2

    # Mass matrices for broken spaces (block-diagonal)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)
    M2 = BrokenMass(V2h, domain_h, is_scalar=True)

    bD0, bD1 = derham_h.broken_derivatives_as_operators
    cP1 = ConformingProjection_V1(V1h, domain_h)
    I1 = IdLinearOperator(V1h)

    A1 = alpha * M1 + ComposedLinearOperator([I1-cP1, M1, I1-cP1]) + ComposedLinearOperator([cP1, bD1.transpose(), M2, bD1, cP1])

    # boundary conditions
    u, v, F  = elements_of(V1h.symbolic_space, names='u, v, F')
    nn  = NormalVector('nn')
    penalization = 10**7
    boundary = domain.boundary
    expr_b = penalization * cross(u, nn) * cross(v, nn)
    a_b = BilinearForm((u,v), integral(boundary, expr_b))
    a_b_h = discretize(a_b, domain_h, [V1h, V1h])
    A_b = FemLinearOperator(fem_domain=V1h, fem_codomain=V1h, matrix=a_b_h.assemble())

    A = A1 + A_b

    expr   = dot(f,v)
    expr_b = penalization * cross(uex, nn) * cross(v, nn)

    l = LinearForm(v, integral(domain, expr) + integral(boundary, expr_b))
    lh = discretize(l, domain_h, V1h)
    b  = lh.assemble()

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system

    if use_scipy:
        print("solving with scipy...")
        A = A.to_sparse_matrix()
        b = b.toarray()     # todo MCP: why not 'to_array', for consistency with array_to_stencil ?

        x        = spsolve(A, b)
        u_coeffs = array_to_stencil(x, V1h.vector_space)

    else:
        u_coeffs, info = cg( A, b, tol=maxwell_tol, verbose=True )

    uh = FemField(V1h, coeffs=u_coeffs)
    uh = cP1(uh)

    # error
    error       = Matrix([F[0]-uex[0],F[1]-uex[1]])
    l2_norm     = Norm(error, domain, kind='l2')
    l2_norm_h   = discretize(l2_norm, domain_h, V1h)
    l2_error    = l2_norm_h.assemble(F=uh)

    return l2_error, uh

def run_maxwell_2d_time_harmonic():
    """
    curl-curl problem with 0 order term and source
    """

    domain = get_pretzel(h=0.5, r_min=1, r_max=1.5, debug_option=0)

    x,y    = domain.coordinates
    alpha  = -1
    uex    = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f      = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                     alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    uex_x = lambdify(domain.coordinates, uex[0])
    uex_y = lambdify(domain.coordinates, uex[1])
    uex_log = [pull_2d_hcurl([uex_x,uex_y], f) for f in mappings_list]

    ## call solver
    l2_error, uh = run_conga_maxwell_2d(uex, f, alpha, domain, ncells=[2**2, 2**2], degree=[2,2], return_sol=True)
    # else:
    #     # Nitsche
    #     l2_error, uh = run_system_3_2d_dir(uex, f, alpha, domain, ncells=[2**3, 2**3], degree=[2,2], return_sol=True)


    print("max2d: ", l2_error)


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # VISUALIZATION
    #   adapted from examples/poisson_2d_multi_patch.py and
    #   and psydac/api/tests/test_api_feec_2d.py
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    N=20
    etas, xx, yy = get_plotting_grid(mappings, N)
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')
    gridlines_x1 = None
    gridlines_x2 = None

    u_x_vals, u_y_vals   = grid_vals_hcurl(uex_log)
    uh_x_vals, uh_y_vals = grid_vals_hcurl(uh)
    u_x_err = [abs(u1 - u2) for u1, u2 in zip(u_x_vals, uh_x_vals)]
    u_y_err = [abs(u1 - u2) for u1, u2 in zip(u_y_vals, uh_y_vals)]
    # u_x_err = abs(u_x_vals - uh_x_vals)
    # u_y_err = abs(u_y_vals - uh_y_vals)

    my_small_plot(
        title=r'approximation of solution $u$, $x$ component',
        vals=[u_x_vals, uh_x_vals, u_x_err],
        titles=[r'$u^{ex}_x(x,y)$', r'$u^h_x(x,y)$', r'$|(u^{ex}-u^h)_x(x,y)|$'],
        xx=xx,
        yy=yy,
        gridlines_x1=gridlines_x1,
        gridlines_x2=gridlines_x2,
    )

    my_small_plot(
        title=r'approximation of solution $u$, $y$ component',
        vals=[u_y_vals, uh_y_vals, u_y_err],
        titles=[r'$u^{ex}_y(x,y)$', r'$u^h_y(x,y)$', r'$|(u^{ex}-u^h)_y(x,y)|$'],
        xx=xx,
        yy=yy,
        gridlines_x1=gridlines_x1,
        gridlines_x2=gridlines_x2,
    )


if __name__ == '__main__':

    run_maxwell_2d_time_harmonic()
