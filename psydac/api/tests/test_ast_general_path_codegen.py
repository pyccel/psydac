#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
Unit tests for the general AST code-generation path in `psydac.api.ast.fem`
(`class AST` -> `DefNode`), as used by `DiscreteLinearForm`, `DiscreteFunctional`,
boundary bilinear forms, and non-sum-factorized (e.g. 2D) bilinear forms.

Unlike the print-only smoke tests in `psydac/psydac/api/ast/tests/` (which just
check that `AST`/`parse`/`pycode` do not crash), these tests make real
assertions on the resulting `DefNode` structure and on `PSYDAC_BACKEND_PYTHON`
(no pyccel/Fortran compilation) numeric results.
"""
import types

import pytest

from sympde.topology import Square, Cube, ScalarFunctionSpace
from sympde.topology import elements_of, element_of
from sympde.calculus  import dot, grad
from sympde.expr      import BilinearForm, LinearForm, Norm, integral

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_PYTHON
from psydac.api.fem            import DiscreteBilinearForm, DiscreteLinearForm, DiscreteFunctional

#==============================================================================
def test_ast_linear_form_structure():

    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    v = element_of(V, name='v')

    l = LinearForm(v, integral(domain, v))

    domain_h = discretize(domain, ncells=(4, 4))
    Vh       = discretize(V, domain_h, degree=(2, 2))

    lh = discretize(l, domain_h, Vh, backend=PSYDAC_BACKEND_PYTHON)
    assert isinstance(lh, DiscreteLinearForm)

    assert lh.ast.expr.kind == 'linearform'
    assert lh.ast.expr.name.startswith('assemble_vector_')
    assert lh.max_nderiv == 0  # no derivatives appear in `v`
    assert isinstance(lh._func, types.FunctionType)  # no compilation with 'python' backend

    # Sum of all entries = integral of 1*v = domain volume (partition of unity).
    b = lh.assemble()
    assert b.toarray().sum() == pytest.approx(1.0)

#==============================================================================
def test_ast_bilinear_form_structure_2d():

    domain = Square()  # 2D => general AST path, never sum-factorization
    V = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    a = BilinearForm((u, v), integral(domain, u * v + dot(grad(u), grad(v))))

    domain_h = discretize(domain, ncells=(4, 4))
    Vh       = discretize(V, domain_h, degree=(2, 2))

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)
    assert isinstance(ah, DiscreteBilinearForm)

    assert ah.ast.expr.kind == 'bilinearform'
    assert ah.ast.expr.name.startswith('assemble_matrix_')
    assert ah.max_nderiv == 1  # dot(grad(u), grad(v)) needs first derivatives

#==============================================================================
def test_ast_functional_form_structure():

    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    u = element_of(V, name='u')

    norm = Norm(u, domain, kind='l2')

    domain_h = discretize(domain, ncells=(4, 4))
    Vh       = discretize(V, domain_h, degree=(2, 2))

    nh = discretize(norm, domain_h, Vh, backend=PSYDAC_BACKEND_PYTHON)
    assert isinstance(nh, DiscreteFunctional)

    assert nh.ast.expr.kind == 'functionalform'
    assert nh.ast.expr.name.startswith('assemble_scalar_')

#==============================================================================
def test_boundary_bilinear_form_uses_general_ast_path_even_in_3d():
    """A pure-boundary BilinearForm must always use the general AST path
    (`psydac.api.fem.DiscreteBilinearForm`), never the 3D sum-factorization
    path, since `TerminalExpr` on it yields a `BoundaryExpression`."""

    domain = Cube()
    boundary = domain.get_boundary(axis=0, ext=1)

    V = ScalarFunctionSpace('V', domain)
    u, v = elements_of(V, names='u, v')

    a = BilinearForm((u, v), integral(boundary, u * v))

    domain_h = discretize(domain, ncells=(2, 2, 2))
    Vh       = discretize(V, domain_h, degree=(1, 1, 1))

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_PYTHON)

    assert isinstance(ah, DiscreteBilinearForm)
    assert ah.ast.expr.kind == 'bilinearform'
