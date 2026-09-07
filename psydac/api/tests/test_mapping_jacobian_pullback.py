#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Unit tests for stage 4 (mapping/Jacobian handling) of the assembly pipeline,
for the 3D sum-factorization path with an analytical mapping.

We use a diagonal scaling mapping F(x1, x2, x3) = (a*x1, b*x2, c*x3), whose
Jacobian determinant (`a*b*c`) and gradient pullback (`1/a, 1/b, 1/c` per
direction) are known in closed form. This lets us cross-check the physical
assembly (which exercises `LogicalExpr`'s analytical Jacobian pullback) against
an independently-assembled reference on the *unmapped* logical domain (which
does not exercise any mapping-specific logic at all).

`PSYDAC_BACKEND_PYTHON` is used throughout, so no pyccel/Fortran compilation
is required.
"""
import numpy as np
import pytest

from sympde.topology import AnalyticMapping, Cube, ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import dx1, dx2, dx3
from sympde.calculus  import dot, grad
from sympde.expr      import BilinearForm, integral

from psydac.api.discretization    import discretize
from psydac.api.settings          import PSYDAC_BACKEND_PYTHON
from psydac.api.fem_bilinear_form import DiscreteBilinearForm

#==============================================================================
A, B, C = 2., 3., 4.

class ScalingMapping3D(AnalyticMapping):
    _expressions = {'x': f'{A}*x1', 'y': f'{B}*x2', 'z': f'{C}*x3'}
    _ldim        = 3
    _pdim        = 3

#==============================================================================
def test_mass_matrix_total_equals_physical_volume():

    logical_domain = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
    mapping        = ScalingMapping3D('M')
    domain         = mapping(logical_domain)

    V = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    a = BilinearForm((u, v), integral(domain, u * v))

    domain_h = discretize(domain, ncells=(2, 2, 2))
    Vh       = discretize(V, domain_h, degree=(2, 2, 2))

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
    assert type(ah) is DiscreteBilinearForm  # sum-factorization path (3D, interior)

    M = ah.assemble()

    # Sum of all mass-matrix entries = integral of 1*1 over the physical
    # domain (by the B-spline partition-of-unity property) = physical volume.
    physical_volume = A * B * C
    assert M.toarray().sum() == pytest.approx(physical_volume)

#==============================================================================
def test_stiffness_matrix_matches_manual_jacobian_pullback():

    ncells = (2, 2, 2)
    degree = (2, 2, 2)

    logical_domain = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
    mapping        = ScalingMapping3D('M')
    domain         = mapping(logical_domain)

    # --- physical assembly: exercises the analytical-mapping Jacobian pullback
    V_map = ScalarFunctionSpace('Vmap', domain)
    u_m, v_m = elements_of(V_map, names='u, v')
    a_phys = BilinearForm((u_m, v_m), integral(domain, dot(grad(u_m), grad(v_m))))

    domain_h_map = discretize(domain, ncells=ncells)
    Vh_map       = discretize(V_map, domain_h_map, degree=degree)

    ah_phys = discretize(a_phys, domain_h_map, [Vh_map, Vh_map], backend=PSYDAC_BACKEND_PYTHON)
    assert type(ah_phys) is DiscreteBilinearForm
    K_phys = ah_phys.assemble().toarray()

    # --- reference assembly: three purely-logical directional stiffness
    # matrices on the *unmapped* domain (mapping_option is None here, so no
    # stage-4 pullback logic is exercised at all).
    V_ref = ScalarFunctionSpace('Vref', logical_domain)
    u_r, v_r = elements_of(V_ref, names='u, v')

    domain_h_ref = discretize(logical_domain, ncells=ncells)
    Vh_ref       = discretize(V_ref, domain_h_ref, degree=degree)

    K_dirs = []
    for dxi in (dx1, dx2, dx3):
        a_dir = BilinearForm((u_r, v_r), integral(logical_domain, dxi(u_r) * dxi(v_r)))
        ah_dir = discretize(a_dir, domain_h_ref, [Vh_ref, Vh_ref], backend=PSYDAC_BACKEND_PYTHON)
        K_dirs.append(ah_dir.assemble().toarray())

    # For a diagonal scaling map, grad_x = (1/a) dx1, grad_y = (1/b) dx2,
    # grad_z = (1/c) dx3, and the volume form is det(DF) = a*b*c. Hence
    # K_phys = sum_i (det / scale_i**2) * K_dir_i.
    det = A * B * C
    K_expected = (det / A**2) * K_dirs[0] + (det / B**2) * K_dirs[1] + (det / C**2) * K_dirs[2]

    assert np.allclose(K_phys, K_expected, rtol=1e-12, atol=1e-12)
