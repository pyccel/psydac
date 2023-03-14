import pytest
import numpy as np
from sympy import pi, sin, cos, tan, atan, atan2
from sympy import exp, sinh, cosh, tanh, atanh, Tuple


from sympde.topology import Line, Square
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of, Derham
from sympde.core     import Constant
from sympde.expr     import LinearForm, BilinearForm, Functional
from sympde.expr     import integral

from psydac.linalg.solvers     import inverse
from psydac.api.discretization import discretize
from psydac.fem.basic          import FemField
from psydac.api.settings       import PSYDAC_BACKENDS
from psydac.linalg.utilities   import array_to_psydac

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

    # Assembly call should not crash if correct arguments are used
    A = ah.assemble(c=1.0, f=fh)
    b = lh.assemble(f=fh, c=1.0)

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

    print("PASSED")

#==============================================================================
def test_assembly_no_synchr_args(backend):

    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    nc       = 5
    ncells   = (nc,)
    degree   = (2,)
    periodic = (True,)

    domain   = Line()
    domain_h = discretize(domain, ncells=ncells, periodic=periodic)

    derham   = Derham(domain)
    derham_h = discretize(derham, domain_h, degree=degree)

    #spaces
    V0h = derham_h.V0
    V1h = derham_h.V1

    #differential operator
    div, = derham_h.derivatives_as_matrices

    rho  = element_of(V1h.symbolic_space, name='rho')
    g    = element_of(V1h.symbolic_space, name='g')
    h    = element_of(V1h.symbolic_space, name='h')

    #L2 proj rho u -> V1
    expr = g*h*rho
    weight_int_prod = BilinearForm((g,h), integral(domain, expr))
    weight_int_prod_h = discretize(weight_int_prod, domain_h, (V1h,V1h), **kwargs)

    expr = g*rho
    int_prod = LinearForm(g, integral(domain, expr))
    int_prod_h = discretize(int_prod, domain_h, V1h, **kwargs)

    func  = Functional(rho, domain)
    func_h = discretize(func, domain_h, V1h, **kwargs)

    uh      = array_to_psydac(np.array([i for i in range(nc)]), V0h.vector_space)
    const_1 = array_to_psydac(np.array([1/nc]*nc), V1h.vector_space)

    rhoh1 = div.dot(uh)
    rhof1 = FemField(V1h, rhoh1)
    rhoh2 = div.dot(uh)
    rhof2  = FemField(V1h, rhoh2)
    rhoh3 = div.dot(uh)
    rhof3  = FemField(V1h, rhoh3)
    weight_mass_matrix = weight_int_prod_h.assemble(rho=rhof1)
    inte_bilin = const_1.dot(weight_mass_matrix.dot(const_1))

    int_prod_rho = int_prod_h.assemble(rho=rhof2)
    inte_lin = int_prod_rho.dot(const_1)

    inte_norm = func_h.assemble(rho=rhof3)

    assert( abs(inte_bilin) < 1.e-12)    
    assert( abs(inte_lin) < 1.e-12)
    assert( abs(inte_norm) < 1.e-12)

#==============================================================================
if __name__ == '__main__':
    #test_field_and_constant(None)
    #test_multiple_fields(None)
    #test_math_imports(None)
    #test_non_symmetric_BilinearForm(None)
    test_assembly_no_synchr_args(None)
