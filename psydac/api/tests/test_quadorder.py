import pytest

from sympde.topology import Line, Square
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.core     import Constant
from sympde.expr     import BilinearForm
from sympde.expr     import LinearForm
from sympde.expr     import integral

import numpy as np

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_PYTHON

#==============================================================================
@pytest.mark.parametrize( 'test_nquads', [(3,3), (4,4), (5,3)] )
def test_custom_nquads(test_nquads):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.

    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    u = element_of(V, name='u')
    v = element_of(V, name='v')
    c = Constant(name='c')

    a = BilinearForm((u, v), integral(domain, u * v))
    l = LinearForm(v, integral(domain, v))

    ncells = (12, 12)
    degree = (2, 2)

    domain_h = discretize(domain, ncells=ncells)

    Vh = discretize(V, domain_h, degree=degree, nquads=test_nquads)

    # NOTE: we _need_ the Python backend here for range checking, otherwise we'd only get segfaults at best
    _ = discretize(a, domain_h, [Vh, Vh], nquads=test_nquads, backend=PSYDAC_BACKEND_PYTHON).assemble()
    _ = discretize(l, domain_h,      Vh , nquads=test_nquads, backend=PSYDAC_BACKEND_PYTHON).assemble()

    assert np.array_equal(Vh.nquads, test_nquads)
