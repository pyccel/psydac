# coding: utf-8

# small script to test a Conga Poisson solver on a multipatch domains,

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

import scipy.sparse.linalg as scipy_solvers

from sympde.topology import Derham, Union
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from sympde.expr.expr import Norm
from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.utilities import array_to_stencil
from psydac.utilities.utils    import refine_array_1d
from psydac.fem.basic   import FemField

from psydac.feec.multipatch.fem_linear_operators import IdLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above
from psydac.feec.multipatch.operators import BrokenMass
from psydac.feec.multipatch.operators import ConformingProjection_V0
from psydac.feec.multipatch.operators import get_patch_index_from_face
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_grid_vals_scalar, get_patch_knots_gridlines, my_small_plot
from psydac.feec.multipatch.multipatch_domain_utilities import get_annulus_fourpatches, get_pretzel

comm = MPI.COMM_WORLD

from psydac.api.essential_bc import apply_essential_bc_stencil

#==============================================================================
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

    pretzel = True
    cartesian = False
    use_scipy = True
    poisson_tol = 5e-13

    domain = get_pretzel(h=0.5, r_min=1, r_max=1.5, debug_option=0)

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])

    mappings_list = list(mappings.values())
    F = [f.get_callable_mapping() for f in mappings_list]

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

    # Mass matrices for broken spaces (block-diagonal)
    M0 = BrokenMass(V0h, domain_h, is_scalar=True)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)

    # Projectors for broken spaces
    bP0, bP1, bP2 = derham_h.projectors(nquads=nquads)

    # Broken derivative operators
    bD0, bD1 = derham_h.broken_derivatives_as_operators

    #+++++++++++++++++++++++++++++++
    # . some target functions
    #+++++++++++++++++++++++++++++++

    # fun1    = lambda xi1, xi2 : np.sin(xi1)*np.sin(xi2)
    # D1fun1  = lambda xi1, xi2 : np.cos(xi1)*np.sin(xi2)
    # D2fun1  = lambda xi1, xi2 : np.sin(xi1)*np.cos(xi2)
    # fun2    = lambda xi1, xi2 : .5*np.sin(xi1)*np.sin(xi2)

    from sympy import cos, sin
    x,y       = domain.coordinates
    phi_exact = sin(x)*cos(y)
    f         = 2*phi_exact

    # affine field, for testing
    u_exact_x = 2*x
    u_exact_y = 2*y

    from sympy import lambdify
    phi_ex = lambdify(domain.coordinates, phi_exact)
    f_ex   = lambdify(domain.coordinates, f)
    u_ex_x = lambdify(domain.coordinates, u_exact_x)
    u_ex_y = lambdify(domain.coordinates, u_exact_y)

    # pull-back of phi_ex
    phi_ex_log = [lambda xi1, xi2,ff=f : phi_ex(*ff(xi1,xi2)) for f in F]
    f_log      = [lambda xi1, xi2,ff=f : f_ex(*ff(xi1,xi2)) for f in F]

    #+++++++++++++++++++++++++++++++
    # . Multipatch operators
    #+++++++++++++++++++++++++++++++

    # phi_ref = bP0(phi_ex_log)

    # Conforming projection V0 -> V0
    ## note: there are problems (eg at the interface) when the conforming projection is not accurate (low penalization or high tolerance)
    cP0     = ConformingProjection_V0(V0h, domain_h)
    cP0_hom = ConformingProjection_V0(V0h, domain_h, hom_bc=True)

    I0 = IdLinearOperator(V0h)

    v  = element_of(derham.V0, 'v')
    u  = element_of(derham.V0, 'u')

    error  = u - phi_exact

    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    l  = LinearForm(v,  integral(domain, f*v))
    lh = discretize(l, domain_h, V0h)
    b  = lh.assemble()

    # Conga Poisson matrix is
    # A = (cD0)^T * M1 * cD0 + (I0 - Pc)^2

    cD0T_M1_cD0 = ComposedLinearOperator([cP0, bD0.transpose(), M1, bD0, cP0])
    A = ComposedLinearOperator([(I0-cP0), M0, (I0-cP0)]) + cD0T_M1_cD0

    cD0T_M1_cD0_hom = ComposedLinearOperator([cP0_hom, bD0.transpose(), M1, bD0, cP0_hom])
    A_hom = ComposedLinearOperator([(I0-cP0_hom), M0, (I0-cP0_hom)]) + cD0T_M1_cD0_hom

    # apply boundary conditions
    boundary = Union(*[j for i in domain.interior for j in i.boundary])
    a0 = BilinearForm((u,v), integral(boundary, u*v))
    l0 = LinearForm(v, integral(boundary, phi_exact*v))

    a0_h = discretize(a0, domain_h, [V0h, V0h])
    l0_h = discretize(l0, domain_h, V0h)

    x0, info = cg(a0_h.assemble(), l0_h.assemble(), tol=poisson_tol)

    x0 = cP0.dot(x0)-cP0_hom.dot(x0)
    # x0 = x0 - cP0_hom.dot(x0)   # should also work since x0 is continuous

    ## other option: get the lifted boundary solution by direct interpolation of the boundary data:
    # bP0, bP1, bP2 = derham_h.projectors(nquads=nquads)
    # lambdify bP0
    # x0 = bP0(phi_exact)

    b = b - A.dot(x0)
    # ...
    if use_scipy:
        print("solving Poisson with scipy...")
        A_hom = A_hom.to_sparse_matrix()
        b = b.toarray()

        x = scipy_solvers.spsolve(A_hom, b)
        phi_coeffs = array_to_stencil(x, V0h.vector_space)

    else:
        phi_coeffs, info = cg( A_hom, b, tol=poisson_tol, verbose=True )

    phi_coeffs = cP0_hom.dot(phi_coeffs) + x0

    phi_h = FemField(V0h, coeffs=phi_coeffs)

    l2norm_h = discretize(l2norm, domain_h, V0h)
    h1norm_h = discretize(h1norm, domain_h, V0h)


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

    conga_poisson_2d()

