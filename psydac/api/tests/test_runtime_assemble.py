#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Unit tests for stage 6 (runtime `assemble()`) of the assembly pipeline:
`reset_arrays()`/`compute_free_arguments()` in `psydac.api.fem_common`, and
the free-argument-resolution + zero-then-fill behavior of
`DiscreteBilinearForm.assemble()` (`psydac.api.fem_bilinear_form`).

`PSYDAC_BACKEND_PYTHON` is used throughout, so no pyccel/Fortran compilation
is required.
"""
import numpy as np
import pytest

from sympde.core     import Constant
from sympde.topology import Cube, ScalarFunctionSpace
from sympde.topology import elements_of, element_of
from sympde.expr      import BilinearForm, LinearForm, integral
from sympde.expr.evaluation import TerminalExpr

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_PYTHON
from psydac.api.fem_common     import reset_arrays, compute_free_arguments
from psydac.fem.basic          import FemField

#==============================================================================
def test_reset_arrays_zeroes_real_array():
    a = np.array([1., 2., 3.])
    reset_arrays(a)
    assert np.all(a == 0.)

#==============================================================================
def test_reset_arrays_zeroes_complex_array():
    a = np.array([1 + 2j, 3 + 4j])
    reset_arrays(a)
    assert np.all(a == 0j)

#==============================================================================
def test_reset_arrays_handles_multiple_arrays_independently():
    a = np.array([1., 2.])
    b = np.array([3., 4., 5.])
    reset_arrays(a, b)
    assert np.all(a == 0.) and np.all(b == 0.)

#==============================================================================
def test_compute_free_arguments_detects_fields_and_constants():

    domain = Cube()
    V = ScalarFunctionSpace('V', domain)
    v = element_of(V, name='v')
    w = element_of(V, name='w')
    c = Constant('c', real=True)

    l = LinearForm(v, integral(domain, c * w * v))
    kernel_expr = TerminalExpr(l, domain)[0]

    free_args = compute_free_arguments(l, kernel_expr)

    assert set(free_args) == {'w', 'c'}

#==============================================================================
def test_assemble_called_twice_gives_same_result():
    """`reset_arrays()` must zero the StencilMatrix before every assembly, so
    that repeated calls to `assemble()` never accumulate contributions."""

    domain = Cube()
    V = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    a = BilinearForm((u, v), integral(domain, u * v))

    domain_h = discretize(domain, ncells=(2, 2, 2))
    Vh       = discretize(V, domain_h, degree=(2, 2, 2))
    ah       = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)

    M1 = ah.assemble().toarray().copy()
    M2 = ah.assemble().toarray().copy()

    assert np.allclose(M1, M2, rtol=1e-13, atol=1e-13)

#==============================================================================
def test_assemble_re_resolves_free_field_argument_every_call():
    """Free `FemField` arguments must be re-resolved on every `assemble()`
    call (not cached from a previous call)."""

    domain = Cube()
    V = ScalarFunctionSpace('V', domain)
    u, v, w = elements_of(V, names='u, v, w')

    a = BilinearForm((u, v), integral(domain, w * u * v))

    domain_h = discretize(domain, ncells=(2, 2, 2))
    Vh       = discretize(V, domain_h, degree=(2, 2, 2))
    ah       = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)

    coeffs_1 = Vh.coeff_space.zeros()
    coeffs_1._data[:] = 1.
    field_1  = FemField(Vh, coeffs_1)

    coeffs_2 = Vh.coeff_space.zeros()
    coeffs_2._data[:] = 2.
    field_2  = FemField(Vh, coeffs_2)

    M1 = ah.assemble(w=field_1).toarray().copy()
    M2 = ah.assemble(w=field_2).toarray().copy()
    M1_again = ah.assemble(w=field_1).toarray().copy()

    # Different field values must give different (non-accumulated) results...
    assert not np.allclose(M1, M2)
    # ...and re-using the original field must reproduce the original result.
    assert np.allclose(M1, M1_again, rtol=1e-13, atol=1e-13)
