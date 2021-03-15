# coding: utf-8

# small script to test a Conga Poisson solver on a multipatch domains,

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve

from sympde.topology import Derham
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

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
from psydac.feec.multipatch.operators import get_grid_vals_scalar
from psydac.feec.multipatch.operators import get_plotting_grid, get_patch_knots_gridlines, my_small_plot

comm = MPI.COMM_WORLD

from psydac.api.essential_bc import apply_essential_bc_stencil
from sympde.topology         import Boundary, Interface

def get_space_indices_from_target(domain, target):
    domains = domain.interior.args
    if isinstance(target, Interface):
        raise NotImplementedError("TODO")
    elif isinstance(target, Boundary):
        i = domains.index(target.domain)
    else:
        i = domains.index(target)
    return i

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

    cartesian = False
    use_scipy = True
    poisson_tol = 5e-13

    nc = 2

    if cartesian:
        A = Square('A',bounds1=(0.5, 1), bounds2=(0, 0.5))
        B = Square('B',bounds1=(0.5, 1), bounds2=(0.5, 1))
        mapping_1 = IdentityMapping('M1', 2)
        mapping_2 = IdentityMapping('M2', 2)

    else:
        A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
        B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))
#        A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi))
#        B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi, 2*np.pi))
        mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
        mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    domain_1     = mapping_1(A)
    domain_2     = mapping_2(B)

    domain = domain_1.join(domain_2, name = 'domain',
                bnd_minus = domain_1.get_boundary(axis=1, ext=1),
                bnd_plus  = domain_2.get_boundary(axis=1, ext=-1))

    # todo (MCP): improve/clarify how mappings are used below: as dicts, lists ?
    mappings  = {A.interior:mapping_1, B.interior:mapping_2}
    mappings_obj = [mapping_1, mapping_2]
    F = [f.get_callable_mapping() for f in mappings_obj]

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    #+++++++++++++++++++++++++++++++
    # . Discrete space
    #+++++++++++++++++++++++++++++++

    ncells = [nc**2, nc**2]
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
    phi_ex_log = [lambda xi1, xi2 : phi_ex(*f(xi1,xi2)) for f in F]
    f_log = [lambda xi1, xi2 : f_ex(*f(xi1,xi2)) for f in F]

    #+++++++++++++++++++++++++++++++
    # . Multipatch operators
    #+++++++++++++++++++++++++++++++

    # phi_ref = bP0(phi_ex_log)

    # Conforming projection V0 -> V0
    ## note: there are problems (eg at the interface) when the conforming projection is not accurate (low penalization or high tolerance)
    cP0 = ConformingProjection_V0(V0h, domain_h)

    I0 = IdLinearOperator(V0h)

    v  = element_of(derham.V0, 'v')
    u  = element_of(derham.V0, 'u')

    l  = LinearForm(v,  integral(domain, f*v))
    lh = discretize(l, domain_h, V0h)
    b  = lh.assemble()

    # Conga Poisson matrix is
    # A = (cD0)^T * M1 * cD0 + (I0 - Pc)^2

    cD0T_M1_cD0 = ComposedLinearOperator([cP0, bD0.transpose(), M1, bD0, cP0])
    A = (I0-cP0) + cD0T_M1_cD0

    # apply boundary conditions
    a0 = BilinearForm((u,v), integral(domain.boundary, u*v))
    l0 = LinearForm(v, integral(domain.boundary, phi_exact*v))

    a0_h = discretize(a0, domain_h, [V0h, V0h])
    l0_h = discretize(l0, domain_h, V0h)

    x0, info = cg(a0_h.assemble(), l0_h.assemble(), tol=poisson_tol)

    b = b-A.dot(x0)
    for bn in domain.boundary:
        i = get_space_indices_from_target(domain, bn)
        for j in range(len(domain)):
            apply_essential_bc_stencil(cP0._A[i,j], axis=bn.axis, ext=bn.ext, order=0)
        apply_essential_bc_stencil(b[i], axis=bn.axis, ext=bn.ext, order=0)

    # ...

    if use_scipy:
        print("solving Poisson with scipy...")
        A = A.to_sparse_matrix()
        b = b.toarray()

        x = spsolve(A, b)
        phi_coeffs = array_to_stencil(x, V0h.vector_space)

    else:
        phi_coeffs, info = cg( A, b, tol=poisson_tol, verbose=True )


    phi_h = FemField(V0h, coeffs=phi_coeffs)
    phi_h = FemField(V0h, coeffs=cP0(phi_h).coeffs+x0)


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # VISUALIZATION
    #   adapted from examples/poisson_2d_multi_patch.py and
    #   and psydac/api/tests/test_api_feec_2d.py
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    N=20
    etas, xx, yy = get_plotting_grid(mappings, N)
    gridlines_x1, gridlines_x2 = get_patch_knots_gridlines(V0h, N, mappings, plotted_patch=1)

    phi_ref_vals = get_grid_vals_scalar(phi_ex_log, etas, mappings) #_obj)
    phi_h_vals = get_grid_vals_scalar(phi_h, etas, mappings) #
    phi_err = abs(phi_ref_vals - phi_h_vals)

    my_small_plot(
        title=r'Solution of Poisson problem $\Delta \phi = f$',
        vals=[phi_ref_vals, phi_h_vals, phi_err],
        titles=[r'$\phi^{ex}(x,y)$', r'$\phi^h(x,y)$', r'$|(\phi-\phi^h)(x,y)|$'],
        xx=xx, yy=yy,
        gridlines_x1=gridlines_x1,
        gridlines_x2=gridlines_x2,
    )


if __name__ == '__main__':

    conga_poisson_2d()

