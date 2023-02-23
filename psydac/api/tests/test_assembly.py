import pytest
from sympy import pi, sin, cos, tan, atan, atan2
from sympy import exp, sinh, cosh, tanh, atanh

from sympde.topology import Line, Square
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of
from sympde.core     import Constant
from sympde.expr     import BilinearForm
from sympde.expr     import LinearForm
from sympde.expr     import integral

from psydac.api.discretization import discretize
from psydac.fem.basic          import FemField
from psydac.api.settings       import PSYDAC_BACKENDS

#==============================================================================
@pytest.fixture(params=[None, 'numba', 'pyccel-gcc'])
def backend(request):
    return request.param

#==============================================================================
def test_field_and_constant(backend):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    u = element_of(V, name='u')
    v = element_of(V, name='v')
    f = element_of(V, name='f')
    c = Constant(name='c')

    g = c * f**2
    a = BilinearForm((u, v), integral(domain, u * v * g))
    l = LinearForm(v, integral(domain, g * v))

    ncells = (5, 5)
    degree = (3, 3)
    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)
    ah = discretize(a, domain_h, [Vh, Vh], **kwargs)
    lh = discretize(l, domain_h,      Vh , **kwargs)

    fh = FemField(Vh)
    fh.coeffs[:] = 1

    gh = FemField(Vh)
    gh.coeffs[:] = 2

    # Assembly call should not crash if correct arguments are used
    A = ah.assemble(c=1.0, f=fh)
    b = lh.assemble(f=fh, c=1.0)

    # The results should not be affected by this assemble 
    C = ah.assemble(c=1.0, f=gh)
    d = lh.assemble(f=gh, c=1.0)

    # Test matrix A
    x = fh.coeffs
    assert abs(A.dot(x).dot(x) - 1) < 1e-12

    # Test vector b
    assert abs(b.toarray().sum() - 1) < 1e-12

    print("PASSED")

#==============================================================================
def test_multiple_fields(backend):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Line()
    V = ScalarFunctionSpace('V', domain)
    u = element_of(V, name='u')
    v = element_of(V, name='v')

    f1 = element_of(V, name='f1')
    f2 = element_of(V, name='f2')

    g = 0.5 * (f1**2 + f2)
    a = BilinearForm((u, v), integral(domain, u * v * g))
    l = LinearForm(v, integral(domain, g * v))

    ncells = (5,)
    degree = (3,)
    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)
    ah = discretize(a, domain_h, [Vh, Vh], **kwargs)
    lh = discretize(l, domain_h,      Vh , **kwargs)

    fh = FemField(Vh)
    fh.coeffs[:] = 1

    # Assembly call should not crash if correct arguments are used
    A = ah.assemble(f1=fh, f2=fh)
    b = lh.assemble(f1=fh, f2=fh)

    # Test matrix A
    x = fh.coeffs
    assert abs(A.dot(x).dot(x) - 1) < 1e-12

    # Test vector b
    assert abs(b.toarray().sum() - 1) < 1e-12

    print("PASSED")

#==============================================================================
def test_math_imports(backend):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    (x, y) = domain.coordinates

    V = ScalarFunctionSpace('V', domain)
    u = element_of(V, name='u')
    v = element_of(V, name='v')

    k_a = sin(pi * x) + cos(pi * y) + tan(pi/2 * y/x) + atan(y/x) + atan2(x, y)
    k_l = exp(x + y) + sinh(x) + cosh(y) + tanh(x - y) + atanh(x - y)

    a = BilinearForm((u, v), integral(domain, k_a * u * v))
    l = LinearForm(v, integral(domain, k_l * v))

    ncells = (4, 4)
    degree = (2, 2)
    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)

    # Python code generation works if printer recognizes all Sympy functions
    ah = discretize(a, domain_h, [Vh, Vh], **kwargs)
    lh = discretize(l, domain_h,      Vh , **kwargs)

    # Assembly works if math functions' imports are compatible with calls
    A = ah.assemble()
    b = lh.assemble()

    # TODO: add meaningful assert statement
    print("PASSED")

#==============================================================================
def test_non_symmetric_BilinearForm(backend):

    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V1 = ScalarFunctionSpace('V1', domain)
    V2 = VectorFunctionSpace('V2', domain)

    u = element_of(V2, name='u')
    v = element_of(V1, name='v')

    a = BilinearForm((u, v), integral(domain, u[0] * v))

    ncells = (5, 5)
    degree = (3, 3)
    domain_h = discretize(domain, ncells=ncells)
    Vh1 = discretize(V1, domain_h, degree=degree)
    Vh2 = discretize(V2, domain_h, degree=degree)
    ah = discretize(a, domain_h, [Vh2, Vh1], **kwargs)

    A = ah.assemble()
    # Test using the dot product of such form
    x = Vh2.vector_space.zeros()
    b = A.dot(x)

    y = Vh1.vector_space.zeros()
    c = A.T.dot(y)

    print("PASSED")

#==============================================================================
if __name__ == '__main__':
    test_field_and_constant(None)
    test_multiple_fields(None)
    test_math_imports(None)
    test_non_symmetric_BilinearForm('python')
