import pytest
import numpy as np

from sympy import pi, sin, cos, tan, atan, atan2, exp, sinh, cosh, tanh, atanh, Tuple, I


from sympde.topology import Line, Square
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of, Derham
from sympde.core     import Constant
from sympde.expr     import LinearForm, BilinearForm, Functional, Norm
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
@pytest.fixture(params=['real','complex'])
def dtype(request):
    return request.param


    # The assembly method of a BilinearForm applied a conjugate on the theoretical matrices to solve the good equation.
    # In theory, we have the system A.conj(u)=conj(b) due to the complex dot product between the tests functions.
    # In psydac, we have decided to assemble the matrix conj(A) and b to get the good solution.

#==============================================================================
def test_field_and_constant(backend, dtype):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    # TODO: remove codomain_type when It is implemented in sympde
    u = element_of(V, name='u')
    v = element_of(V, name='v')
    f = element_of(V, name='f')
    c = Constant(name='c')

    if dtype == 'complex':
        V.codomain_type = dtype
        g = I * c * f**2
        res = 1.j
    else:
        g = c * f**2
        res = 1

    a = BilinearForm((u, v), integral(domain, g * u * v))
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
    #TODO change res into np.conj(res) when the conjugate is applied in the dot product in sympde
    assert abs(x.dot(A.dot(x)) - res) < 1e-12

    # Test vector b
    assert abs(b.toarray().sum() - res) < 1e-12
    print("PASSED")

#==============================================================================
def test_bilinearForm_complex(backend, dtype='complex'):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    # TODO: remove codomain_type when It is implemented in sympde
    V.codomain_type = dtype
    u = element_of(V, name='u')
    v = element_of(V, name='v')
    f = element_of(V, name='f')
    c = Constant(name='c')

    res=(1.+1.j)/2
    # We try to put complex as a sympy object in the expression
    g1 = (1.+I)/2 * c * f**2

    # We try to put complex as a python scalar in the expression
    g2 = res * c * f**2

    # We try to put complex in a Sympde Constant in the expression or in a PSYDAC FemField in the expression
    g3 = c * f**2

    a1 = BilinearForm((u, v), integral(domain, u * v * g1))
    a2 = BilinearForm((u, v), integral(domain, u * v * g2))
    a3 = BilinearForm((u, v), integral(domain, u * v * g3))

    ncells = (5, 5)
    degree = (3, 3)
    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)
    a1h = discretize(a1, domain_h, [Vh, Vh], **kwargs)
    a2h = discretize(a2, domain_h, [Vh, Vh], **kwargs)
    a3h = discretize(a3, domain_h, [Vh, Vh], **kwargs)

    fh = FemField(Vh)
    fh.coeffs[:] = 1
    fh2 = FemField(Vh)
    fh2.coeffs[:] = np.sqrt(res)

    # Assembly call should not crash if correct arguments are used
    A1 = a1h.assemble(c=1.0, f=fh)
    A2 = a2h.assemble(c=1.0, f=fh)
    A3 = a3h.assemble(c=res, f=fh)
    A4 = a3h.assemble(c=1.0, f=fh2)

    # Test matrix A
    x = fh.coeffs

    #TODO change res into np.conj(res) when the conjugate is applied in the dot product in sympde
    assert abs(x.dot(A1.dot(x)) - res) < 1e-12
    assert abs(x.dot(A2.dot(x)) - res) < 1e-12
    assert abs(x.dot(A3.dot(x)) - res) < 1e-12
    assert abs(x.dot(A4.dot(x)) - res) < 1e-12

    print("PASSED")

#==============================================================================
def test_linearForm_complex(backend, dtype='complex'):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    # TODO: remove codomain_type when It is implemented in sympde
    V.codomain_type = dtype
    v = element_of(V, name='v')
    f = element_of(V, name='f')
    c = Constant(name='c')

    res = (1.+1.j)/2
    # We try to put complex as a sympy object in the expression
    g1 = (1.+I)/2 * c * f**2

    # We try to put complex as a python scalar in the expression
    g2 = res * c * f**2

    # We try to put complex in a Sympde Constant in the expression or in a PSYDAC FemField in the expression
    g3 = c * f**2

    l1 = LinearForm(v, integral(domain, g1 * v))
    l2 = LinearForm(v, integral(domain, g2 * v))
    l3 = LinearForm(v, integral(domain, g3 * v))

    ncells = (5, 5)
    degree = (3, 3)
    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)
    l1h = discretize(l1, domain_h, Vh, **kwargs)
    l2h = discretize(l2, domain_h, Vh, **kwargs)
    l3h = discretize(l3, domain_h, Vh, **kwargs)

    fh = FemField(Vh)
    fh.coeffs[:] = 1
    fh2 = FemField(Vh)
    fh2.coeffs[:] = np.sqrt(res)

    # Assembly call should not crash if correct arguments are used
    b1 = l1h.assemble(c=1.0, f=fh)
    b2 = l2h.assemble(c=1.0, f=fh)
    b3 = l3h.assemble(c=res, f=fh)
    b4 = l3h.assemble(c=1.0, f=fh2)


    # Test vector b
    assert abs(b1.toarray().sum() - res) < 1e-12
    assert abs(b2.toarray().sum() - res) < 1e-12
    assert abs(b3.toarray().sum() - res) < 1e-12
    assert abs(b4.toarray().sum() - res) < 1e-12

    print("PASSED")

#==============================================================================
def test_Norm_complex(backend, dtype='complex'):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    # TODO: remove codomain_type when It is implemented in sympde
    V.codomain_type = dtype
    v = element_of(V, name='v')
    c = Constant(name='c')

    res = (1.+1.j)/np.sqrt(2)

    # We try to put complex as a sympy object in the expression
    g1  = (1.+I)/np.sqrt(2)

    # We try to put complex as a python scalar in the expression
    g2  = res

    # We try to put complex in a Sympde Constant in the expression
    g3  = c

    n1 = Norm(v-g1, domain)
    n2 = Norm(v-g2, domain)
    n3 = Norm(v-g3, domain)

    # We try to put complex in a PSYDAC FemField in the expression
    n4 = Norm(v, domain)


    ncells = (5, 5)
    degree = (3, 3)
    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)
    n1h = discretize(n1, domain_h, Vh, **kwargs)
    n2h = discretize(n2, domain_h, Vh, **kwargs)
    n3h = discretize(n3, domain_h, Vh, **kwargs)
    n4h = discretize(n4, domain_h, Vh, **kwargs)

    fh = FemField(Vh)
    fh.coeffs[:] = 1

    fh2 = FemField(Vh)
    fh2.coeffs[:] = np.sqrt(res)

    # Assembly call should not crash if correct arguments are used
    r1 = n1h.assemble(v=fh)
    r2 = n2h.assemble(v=fh)
    r3 = n3h.assemble(v=fh, c=res)
    r4 = n4h.assemble(v=fh2)

    # Test matrix A
    assert abs(r1-0.7653668647301748) < 1e-12
    assert abs(r1 - r2) < 1e-12
    assert abs(r1 - r3) < 1e-12
    assert abs(r4 - 1) < 1e-12
    print("PASSED")

#==============================================================================
def test_multiple_fields(backend, dtype):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Line()
    V = ScalarFunctionSpace('V', domain)

    # TODO: remove codomain_type when It is implemented in sympde
    V.codomain_type = dtype
    u = element_of(V, name='u')
    v = element_of(V, name='v')

    f1 = element_of(V, name='f1')
    f2 = element_of(V, name='f2')

    if dtype == 'complex':
        g = 0.5j * (f1**2 + f2)
        res = 1.j
    else:
        g = 0.5 * (f1**2 + f2)
        res = 1

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

    x = fh.coeffs

    # Test matrix A
    #TODO change res into np.conj(res) when the conjugate is applied in the dot product in sympde
    assert abs(x.dot(A.dot(x)) - res) < 1e-12

    # Test vector b
    assert abs(b.toarray().sum() - res) < 1e-12

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
    test_field_and_constant(None)
    test_multiple_fields(None)
    test_math_imports(None)
    test_non_symmetric_BilinearForm(None)
    test_assembly_no_synchr_args(None)
