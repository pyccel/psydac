#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from sympde.topology import Line, Square
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.core     import Constant
from sympde.expr     import BilinearForm
from sympde.expr     import LinearForm
from sympde.expr     import integral
from sympde.expr     import find
from sympde.expr     import EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKENDS

#==============================================================================
@pytest.fixture(params=[None, 'pyccel-gcc'])
def backend(request):
    return request.param

#==============================================================================
def test_field_and_constant(backend):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    # Symbolic problem definition with SymPDE
    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    u = element_of(V, name='u')
    v = element_of(V, name='v')
    f = element_of(V, name='f')
    c = Constant(name='c', real=True)

    g = c * f**2
    a = BilinearForm((u, v), integral(domain, u * v))
    l = LinearForm(v, integral(domain, g * v))
    bc = EssentialBC(u, g, domain.boundary)

    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v), bc=bc)

    # Discretization and solution with PSYDAC
    ncells = (5, 5)
    degree = (3, 3)
    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)
    equation_h = discretize(equation, domain_h, [Vh, Vh], **kwargs)

    # Discrete field is set to 1, and constant is set to 3
    fh = FemField(Vh)
    fh.coeffs[:] = 1
    c_value = 3.0

    # Solve call should not crash if correct arguments are used
    xh = equation_h.solve(c=c_value, f=fh)

    # Verify that solution is equal to c_value
    assert np.allclose(xh.coeffs.toarray(), c_value, rtol=1e-9, atol=1e-16)
