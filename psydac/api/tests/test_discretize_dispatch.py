#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Unit tests for the class-dispatch logic of `discretize()` when applied to
`BilinearForm`/`LinearForm` objects (see `psydac.api.discretization.discretize`).

These tests use the pure-Python backend and tiny meshes/degrees so that they
run quickly (no pyccel/Fortran compilation), and they only check the *type* of
the object returned by `discretize()`, without calling `.assemble()`.
"""
from sympde.topology import Square, Cube
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of, element_of
from sympde.calculus  import dot, grad
from sympde.expr      import BilinearForm, LinearForm, integral

from psydac.api.discretization  import discretize
from psydac.api.settings        import PSYDAC_BACKEND_PYTHON
from psydac.api.fem_bilinear_form import DiscreteBilinearForm as DiscreteBilinearFormSF
from psydac.api.fem_sum_form    import DiscreteSumForm
from psydac.api.fem             import DiscreteBilinearForm
from psydac.api.fem             import DiscreteLinearForm

#==============================================================================
def test_discretize_bilinear_3d_uses_sum_factorization_by_default():

    domain = Cube()
    V = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v))))

    domain_h = discretize(domain, ncells=(2, 2, 2))
    Vh       = discretize(V, domain_h, degree=(1, 1, 1))

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)

    assert type(ah) is DiscreteBilinearFormSF

#==============================================================================
def test_discretize_bilinear_2d_uses_legacy_path():

    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v))))

    domain_h = discretize(domain, ncells=(4, 4))
    Vh       = discretize(V, domain_h, degree=(2, 2))

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)

    assert type(ah) is DiscreteBilinearForm

#==============================================================================
def test_discretize_bilinear_3d_can_force_legacy_path():

    domain = Cube()
    V = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    a = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v))))

    domain_h = discretize(domain, ncells=(2, 2, 2))
    Vh       = discretize(V, domain_h, degree=(1, 1, 1))

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON,
                     sum_factorization=False)

    assert type(ah) is DiscreteBilinearForm

#==============================================================================
def test_discretize_bilinear_mixed_interior_boundary_uses_sum_form():

    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    boundary = domain.boundary
    a = BilinearForm((u, v), integral(domain, u*v) + integral(boundary, u*v))

    domain_h = discretize(domain, ncells=(4, 4))
    Vh       = discretize(V, domain_h, degree=(2, 2))

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)

    assert type(ah) is DiscreteSumForm

#==============================================================================
def test_discretize_linear_form_uses_discrete_linear_form():

    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    v = element_of(V, name='v')

    l = LinearForm(v, integral(domain, v))

    domain_h = discretize(domain, ncells=(4, 4))
    Vh       = discretize(V, domain_h, degree=(2, 2))

    lh = discretize(l, domain_h, Vh, backend=PSYDAC_BACKEND_PYTHON)

    assert type(lh) is DiscreteLinearForm
