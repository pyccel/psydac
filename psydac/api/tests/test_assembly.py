from sympde.topology import Line, Square
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.core     import Constant
from sympde.expr     import BilinearForm
from sympde.expr     import LinearForm
from sympde.expr     import integral

from psydac.api.discretization import discretize
from psydac.fem.basic          import FemField

#==============================================================================
def test_field_and_constant():

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
    ah = discretize(a, domain_h, [Vh, Vh])
    lh = discretize(l, domain_h, Vh)

    fh = FemField(Vh)
    fh.coeffs[:] = 1

    # Assembly call should not crash if correct arguments are used
    A = ah.assemble(c=1, f=fh)
    b = lh.assemble(f=fh, c=1)

    # Test matrix A
    x = fh.coeffs
    assert abs(A.dot(x).dot(x) - 1) < 1e-12

    # Test vector b
    assert abs(b.toarray().sum() - 1) < 1e-12

    print("PASSED")

#==============================================================================
def test_multiple_fields():

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
    ah = discretize(a, domain_h, [Vh, Vh])
    lh = discretize(l, domain_h, Vh)

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
if __name__ == '__main__':
    test_field_and_constant()
    test_multiple_fields()
