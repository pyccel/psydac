import numpy as np

from sympde.topology import NCube
from sympde.topology import domain
from sympde.topology import Square, PolarMapping

from psydac.mapping.discrete import SplineCallableMapping
from psydac.cad.geometry     import Geometry
from psydac.api.tests.build_domain import build_11_patch_pretzel

import pytest


@pytest.mark.parametrize('spline_mapping', [False, True])
def test_poisson_mapping(spline_mapping):
    _, _, l2_error = _solve_poisson_mapping(spline_mapping)
    # Actual: ~1e-5 (True, 8x8 deg-2 spline mesh -- the geometry approximation
    # dominates) / ~2e-5 (False, 10x10 deg-2 -- x**2+y**2 is not a degree-2
    # polynomial once pulled back through the non-affine polar map). Thresholds
    # leave ~50x / ~10x headroom.
    assert l2_error < (1e-3 if spline_mapping else 2e-4)


def _solve_poisson_mapping(spline_mapping):
    # WP07c-2: a coupled multipatch Poisson solve (SIPG interface + Nitsche
    # boundary, manufactured solution x**2 + y**2) on a 2-patch mapped geometry.
    #   spline_mapping=False : analytic PolarMapping geometry
    #                          -> discretize(Omega, ncells=..., periodic=...)
    #   spline_mapping=True  : each patch a DiscreteMapping wrapping a
    #                          SplineCallableMapping approximation of that PolarMapping
    #                          -> discretize(Omega) builds the Geometry from the
    #                          splines (is_analytical=False, so assembly is by
    #                          grid evaluation of the spline, not the analytic
    #                          Jacobian). Same abstract model, same solve.
    #
    # Returns (Omega, uh, l2_error) -- not just the l2_error -- so `__main__`
    # below can plot `uh` on `Omega` for each variant (e.g. to derive a
    # documentation example from this test).
    from sympde.calculus       import grad, dot, minus, plus
    from sympde.topology       import Domain, ScalarFunctionSpace, elements_of, NormalVector
    from sympde.expr.expr      import BilinearForm, LinearForm, integral, Norm
    from sympde.expr           import find

    from psydac.api.discretization import discretize
    from psydac.api.settings       import PSYDAC_BACKENDS

    rmin, rmax        = 0.3, 1.
    F_degree, F_ncells = (2, 2), (8, 8)

    # Two quarter-annulus patches (the geometry the test has always used).
    domain_log_1 = Square('A_1', bounds1=(0., 1.), bounds2=(0.,     0.5 * np.pi))
    domain_log_2 = Square('A_2', bounds1=(0., 1.), bounds2=(np.pi,  1.5 * np.pi))
    F_1 = PolarMapping('F_1', dim=2, c1=0.,          c2=0., rmin=rmin, rmax=rmax)
    F_2 = PolarMapping('F_2', dim=2, c1=rmin + rmax, c2=0., rmin=rmax, rmax=rmin)

    if spline_mapping:
        F_1s = SplineCallableMapping.from_mapping(
            F_1, ncells=F_ncells, degree=F_degree,
            bounds=zip(domain_log_1.min_coords, domain_log_1.max_coords))
        F_2s = SplineCallableMapping.from_mapping(
            F_2, ncells=F_ncells, degree=F_degree,
            bounds=zip(domain_log_2.min_coords, domain_log_2.max_coords))
        # A fresh DefinedMapping per patch whose callable IS the spline.
        M_1, M_2 = F_1s.to_defined_mapping('F_1'), F_2s.to_defined_mapping('F_2')
        assert M_1.is_analytical is False
        Omega_1, Omega_2 = M_1(domain_log_1), M_2(domain_log_2)
    else:
        Omega_1, Omega_2 = F_1(domain_log_1), F_2(domain_log_2)

    connectivity = [((0, 1, -1), (1, 1, -1), 1)]
    Omega = Domain.join([Omega_1, Omega_2], connectivity, 'domain')

    # ------------------------------------------------------------------ model
    x, y     = Omega.coordinates
    solution = x**2 + y**2
    f        = -4

    V    = ScalarFunctionSpace('V', Omega, kind=None)
    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')
    I    = Omega.interfaces
    bnd  = Omega.boundary
    kappa = 1e3

    expr_I = (- 0.5*dot(grad(plus(u)), nn)*minus(v)  + 0.5*dot(grad(minus(v)), nn)*plus(u)  - kappa*plus(u)*minus(v)
              + 0.5*dot(grad(minus(u)), nn)*plus(v)  - 0.5*dot(grad(plus(v)), nn)*minus(u)  - kappa*plus(v)*minus(u)
              - 0.5*dot(grad(minus(v)), nn)*minus(u) - 0.5*dot(grad(minus(u)), nn)*minus(v) + kappa*minus(u)*minus(v)
              + 0.5*dot(grad(plus(v)), nn)*plus(u)   + 0.5*dot(grad(plus(u)), nn)*plus(v)   + kappa*plus(u)*plus(v))
    expr_b = -dot(grad(u), nn)*v - dot(grad(v), nn)*u + kappa*u*v

    a = BilinearForm((u, v), integral(Omega, dot(grad(u), grad(v)))
                             + integral(I, expr_I) + integral(bnd, expr_b))
    l = LinearForm(v, integral(Omega, f*v)
                      + integral(bnd, -dot(grad(v), nn)*solution + kappa*solution*v))
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v))
    l2norm   = Norm(u - solution, Omega, kind='l2')

    # --------------------------------------------------------------- discretize
    backend = PSYDAC_BACKENDS['python']
    if spline_mapping:
        # No filename / ncells: Omega carries its own discrete geometry.
        # degree / ncells are taken from the spline spaces (F_degree, F_ncells).
        Omega_h = discretize(Omega)
        Vh      = discretize(V, Omega_h)
    else:
        Omega_h = discretize(Omega, ncells=[10, 10], periodic=[False, False])
        Vh      = discretize(V, Omega_h, degree=[2, 2])

    equation_h = discretize(equation, Omega_h, [Vh, Vh], backend=backend)
    l2norm_h   = discretize(l2norm,   Omega_h, Vh, backend=backend)

    uh       = equation_h.solve()
    l2_error = float(l2norm_h.assemble(u=uh))
    print(f'test_poisson_mapping[spline_mapping={spline_mapping}]: L2 error = {l2_error:.2e}')

    return Omega, uh, l2_error


def test_poisson_2d_single_patch_discrete_mapping():
    _, _, l2_error = _solve_poisson_2d_single_patch_discrete_mapping()
    assert l2_error < 1e-4


def _solve_poisson_2d_single_patch_discrete_mapping():
    # WP07b: a Poisson solve on a *single-patch* domain whose mapping is a
    # DiscreteMapping wrapping a SplineCallableMapping (is_analytical=False), i.e.
    # psydac assembles the geometry via grid evaluation of the spline -- the
    # same path as Domain.from_file, but built in memory. Manufactured solution
    # x**2 + y**2 on a spline-approximated quarter annulus.
    #
    # Returns (Omega, uh, l2_error) -- see _solve_poisson_mapping.
    from sympy import pi

    from sympde.calculus       import grad, dot
    from sympde.topology       import ScalarFunctionSpace, elements_of
    from sympde.expr.expr      import BilinearForm, LinearForm, integral, Norm
    from sympde.expr           import find, EssentialBC

    from psydac.api.discretization import discretize
    from psydac.api.settings       import PSYDAC_BACKENDS

    rmin, rmax = 0.3, 1.0
    A = Square('A', bounds1=(0., 1.), bounds2=(0., 0.5 * float(pi)))
    F = PolarMapping('F', dim=2, c1=0., c2=0., rmin=rmin, rmax=rmax)

    # spline approximation of the geometry (degree 3, coarse grid)
    geo_ncells, geo_degree = (8, 8), (3, 3)
    F_h = SplineCallableMapping.from_mapping(
        F, ncells=geo_ncells, degree=geo_degree,
        bounds=zip(A.min_coords, A.max_coords))

    F_disc = F_h.to_defined_mapping('F')            # DiscreteMapping, is_analytical=False
    assert F_disc.is_analytical is False
    Omega  = F_disc(A)

    patches = [Omega.interior]
    Omega_h = Geometry(
        domain   = Omega, pdim = 2,
        ncells   = {p.name: [12, 12]        for p in patches},
        periodic = {p.name: [False, False]  for p in patches},
        mappings = {p.name: p.mapping.get_callable_mapping() for p in patches},
    )

    x, y = Omega.coordinates
    ue   = x**2 + y**2
    f    = -4

    V    = ScalarFunctionSpace('V', Omega)
    u, v = elements_of(V, names='u, v')
    a    = BilinearForm((u, v), integral(Omega, dot(grad(u), grad(v))))
    l    = LinearForm(v, integral(Omega, f * v))
    bc   = EssentialBC(u, ue, Omega.boundary)
    eq   = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)
    l2   = Norm(u - ue, Omega, kind='l2')

    backend = PSYDAC_BACKENDS['python']
    Vh   = discretize(V,  Omega_h, degree=[3, 3])
    eqh  = discretize(eq, Omega_h, [Vh, Vh], backend=backend)
    l2h  = discretize(l2, Omega_h, Vh, backend=backend)

    uh = eqh.solve()
    l2_error = float(l2h.assemble(u=uh))
    print(f'single-patch DiscreteMapping Poisson: L2 error = {l2_error:.2e}')

    return Omega, uh, l2_error


def test_poisson_2d_two_patch_discrete_mapping():
    _, _, l2_error = _solve_poisson_2d_two_patch_discrete_mapping()
    assert l2_error < 1e-3


def _solve_poisson_2d_two_patch_discrete_mapping():
    # WP07c-1: a coupled (interface-term) Poisson solve on a *two-patch* domain
    # whose patches are DiscreteMappings wrapping SplineCallableMappings. Omega_h comes
    # straight from `discretize(Omega)` -- no filename, no ncells: the domain
    # carries its own discrete geometry, and Geometry.from_discrete_domain wires
    # the coefficient-space interface connectivity. SIPG interface + Nitsche
    # boundary, manufactured solution x**2 + y**2 on a spline half-annulus.
    #
    # Returns (Omega, uh, l2_error) -- see _solve_poisson_mapping.
    from sympy import pi

    from sympde.calculus       import grad, dot, minus, plus
    from sympde.topology       import Domain, ScalarFunctionSpace, elements_of, NormalVector
    from sympde.expr.expr      import BilinearForm, LinearForm, integral, Norm
    from sympde.expr           import find

    from psydac.api.discretization import discretize
    from psydac.api.settings       import PSYDAC_BACKENDS

    geo_degree, geo_ncells = (3, 3), (8, 8)
    A = Square('A', bounds1=(0.5, 1.0), bounds2=(0.0,           0.5 * float(pi)))
    B = Square('B', bounds1=(0.5, 1.0), bounds2=(0.5 * float(pi),      float(pi)))

    def approx(pm, sq):
        return SplineCallableMapping.from_mapping(
            pm, ncells=geo_ncells, degree=geo_degree,
            bounds=zip(sq.min_coords, sq.max_coords))

    M_A = approx(PolarMapping('MA', dim=2, c1=0., c2=0., rmin=0., rmax=1.), A).to_defined_mapping('MA')
    M_B = approx(PolarMapping('MB', dim=2, c1=0., c2=0., rmin=0., rmax=1.), B).to_defined_mapping('MB')
    assert M_A.is_analytical is False
    Omega = Domain.join([M_A(A), M_B(B)], [((0, 1, 1), (1, 1, -1), 1)], 'half_annulus')

    Omega_h = discretize(Omega)                       # <-- the WP07c-1 entry point

    x, y     = Omega.coordinates
    solution = x**2 + y**2
    f        = -4

    V    = ScalarFunctionSpace('V', Omega, kind=None)
    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')
    I    = Omega.interfaces
    bnd  = Omega.boundary
    kappa = 1e3

    expr_I = (- 0.5*dot(grad(plus(u)), nn)*minus(v)  + 0.5*dot(grad(minus(v)), nn)*plus(u)  - kappa*plus(u)*minus(v)
              + 0.5*dot(grad(minus(u)), nn)*plus(v)  - 0.5*dot(grad(plus(v)), nn)*minus(u)  - kappa*plus(v)*minus(u)
              - 0.5*dot(grad(minus(v)), nn)*minus(u) - 0.5*dot(grad(minus(u)), nn)*minus(v) + kappa*minus(u)*minus(v)
              + 0.5*dot(grad(plus(v)), nn)*plus(u)   + 0.5*dot(grad(plus(u)), nn)*plus(v)   + kappa*plus(u)*plus(v))
    expr_b = -dot(grad(u), nn)*v - dot(grad(v), nn)*u + kappa*u*v

    a = BilinearForm((u, v), integral(Omega, dot(grad(u), grad(v)))
                             + integral(I, expr_I) + integral(bnd, expr_b))
    l = LinearForm(v, integral(Omega, f*v)
                      + integral(bnd, -dot(grad(v), nn)*solution + kappa*solution*v))
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v))
    l2norm   = Norm(u - solution, Omega, kind='l2')

    backend = PSYDAC_BACKENDS['python']
    Vh  = discretize(V, Omega_h)
    eqh = discretize(equation, Omega_h, [Vh, Vh], backend=backend)
    l2h = discretize(l2norm, Omega_h, Vh, backend=backend)

    uh = eqh.solve()
    l2_error = float(l2h.assemble(u=uh))
    print(f'two-patch DiscreteMapping Poisson: L2 error = {l2_error:.2e}')

    return Omega, uh, l2_error


if __name__ == '__main__':
    # Run directly (`python test_spline_mapping_2d.py`) to solve and,
    # optionally, plot each of the Poisson problems above -- handy for
    # deriving a documentation example from this file. Plotting is off by
    # default so a plain run needs no display; flip PLOT to True (or set
    # MPLBACKEND to an interactive backend) to pop up the figures.
    PLOT = False

    solutions = []
    for spline_mapping in [False, True]:
        print(f'Running test_poisson_mapping with spline_mapping={spline_mapping}')
        Omega, uh, l2_error = _solve_poisson_mapping(spline_mapping)
        solutions.append((f'poisson_mapping[spline_mapping={spline_mapping}]', Omega, uh))

    Omega, uh, l2_error = _solve_poisson_2d_single_patch_discrete_mapping()
    solutions.append(('poisson_2d_single_patch_discrete_mapping', Omega, uh))

    Omega, uh, l2_error = _solve_poisson_2d_two_patch_discrete_mapping()
    solutions.append(('poisson_2d_two_patch_discrete_mapping', Omega, uh))

    if PLOT:
        from psydac.fem.plotting_utilities import plot_field_2d as plot_field
        for title, Omega, uh in solutions:
            plot_field(fem_field=uh, domain=Omega, title=title, hide_plot=False)
