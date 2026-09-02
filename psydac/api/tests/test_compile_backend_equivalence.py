#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Unit tests for stage 5 (compilation) of the assembly pipeline:
`BasicCodeGen._compile()` in `psydac.api.basic`, which either leaves the
generated assembly function as plain Python (`backend='python'`) or compiles
it to Fortran via `pyccel.epyccel()` (`backend='pyccel-gcc'`).

These tests discretize the *same* symbolic form with both backends and check
that (a) `self._func` is a compiled callable only for the pyccel backend, and
(b) the assembled matrices/vectors are numerically identical, i.e. that
compilation preserves the semantics of the generated Python code.

Requires a working Fortran compiler (marked `@pytest.mark.pyccel`, consistent
with `test_epyccel_flags.py`).
"""
import types

import numpy as np
import pytest

from sympde.topology import Cube, ScalarFunctionSpace
from sympde.topology import elements_of, element_of
from sympde.calculus  import dot, grad
from sympde.expr      import BilinearForm, LinearForm, integral

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL

#==============================================================================
@pytest.mark.pyccel
def test_compiled_and_uncompiled_backends_give_same_bilinear_form():

    domain = Cube()
    V = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    a = BilinearForm((u, v), integral(domain, u * v + dot(grad(u), grad(v))))

    domain_h = discretize(domain, ncells=(2, 2, 2))
    Vh       = discretize(V, domain_h, degree=(2, 2, 2))

    ah_python = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
    ah_pyccel = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)

    # Only the 'python' backend leaves the assembly function as plain Python.
    assert isinstance(ah_python._func, types.FunctionType)
    assert not isinstance(ah_pyccel._func, types.FunctionType)

    A_python = ah_python.assemble().toarray()
    A_pyccel = ah_pyccel.assemble().toarray()

    assert np.allclose(A_python, A_pyccel, rtol=1e-13, atol=1e-13)

#==============================================================================
@pytest.mark.pyccel
def test_compiled_and_uncompiled_backends_give_same_linear_form():

    domain = Cube()
    V = ScalarFunctionSpace('V', domain)
    v = element_of(V, name='v')

    l = LinearForm(v, integral(domain, v))

    domain_h = discretize(domain, ncells=(2, 2, 2))
    Vh       = discretize(V, domain_h, degree=(2, 2, 2))

    lh_python = discretize(l, domain_h, Vh, backend=PSYDAC_BACKEND_PYTHON)
    lh_pyccel = discretize(l, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)

    assert isinstance(lh_python._func, types.FunctionType)
    assert not isinstance(lh_pyccel._func, types.FunctionType)

    b_python = lh_python.assemble().toarray()
    b_pyccel = lh_pyccel.assemble().toarray()

    assert np.allclose(b_python, b_pyccel, rtol=1e-13, atol=1e-13)
