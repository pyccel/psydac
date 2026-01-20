#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np
from mpi4py import MPI
from sympy import pi, sin, cos, tan, atan, atan2, exp, sinh, cosh, tanh, atanh, Tuple, I, sqrt

from sympde.topology import Line, Square
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of, Derham
from sympde.core     import Constant
from sympde.expr     import LinearForm, BilinearForm, Functional, Norm
from sympde.expr     import integral
from sympde.calculus import Inner

from psydac.linalg.solvers     import inverse
from psydac.api.discretization import discretize
from psydac.fem.basic          import FemField
from psydac.api.settings       import PSYDAC_BACKENDS
from psydac.linalg.utilities   import array_to_psydac

#==============================================================================
@pytest.fixture(params=[None, 'pyccel-gcc'])
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

    if dtype == 'complex':
        c = Constant(name='c', complex=True)
        V.codomain_type = dtype
        g = I * c * f**2
        res = 1.j
        cst=complex(1.0)
    else:
        c = Constant(name='c', real=True)
        g = c * f**2
        res = 1
        cst=1.0

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
    A = ah.assemble(c=cst, f=fh)
    b = lh.assemble(f=fh, c=cst)

    # Test matrix A
    x = fh.coeffs
    #TODO change res into np.conj(res) when the conjugate is applied in the dot product in sympde
    assert abs(x.inner(A.dot(x)) - res) < 1e-12

    # Test vector b
    assert abs(b.toarray().sum() - res) < 1e-12
    print("PASSED")

#==============================================================================
def test_field_and_constant_deg_0(backend, dtype):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    # TODO: remove codomain_type when It is implemented in sympde
    u = element_of(V, name='u')
    v = element_of(V, name='v')
    f = element_of(V, name='f')

    if dtype == 'complex':
        c = Constant(name='c', complex=True)
        V.codomain_type = dtype
        g = I * c * f**2
        res = 1.j
        cst=complex(1.0)
    else:
        c = Constant(name='c', real=True)
        g = c * f**2
        res = 1
        cst=1.0

    a = BilinearForm((u, v), integral(domain, g * u * v))
    l = LinearForm(v, integral(domain, g * v))

    ncells = (5, 5)
    degree = (0, 0)
    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)
    ah = discretize(a, domain_h, [Vh, Vh], **kwargs)
    lh = discretize(l, domain_h,      Vh , **kwargs)

    fh = FemField(Vh)
    fh.coeffs[:] = 1

    # Assembly call should not crash if correct arguments are used
    A = ah.assemble(c=cst, f=fh)
    b = lh.assemble(f=fh, c=cst)

    # Test matrix A
    x = fh.coeffs

    #TODO change res into np.conj(res) when the conjugate is applied in the dot product in sympde
    assert abs(x.inner(A.dot(x)) - res) < 1e-12

    # Test vector b
    assert abs(b.toarray().sum() - res) < 1e-12
    print("PASSED")

#==============================================================================
def test_bilinearForm_complex(backend):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    # TODO: remove codomain_type when It is implemented in sympde
    V.codomain_type = 'complex'
    u = element_of(V, name='u')
    v = element_of(V, name='v')
    f = element_of(V, name='f')
    c = Constant(name='c', complex=True)

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
    A1 = a1h.assemble(c=complex(1.0), f=fh)
    A2 = a2h.assemble(c=complex(1.0), f=fh)
    A3 = a3h.assemble(c=res, f=fh)
    A4 = a3h.assemble(c=complex(1.0), f=fh2)

    # Test matrix A
    x = fh.coeffs

    #TODO change res into np.conj(res) when the conjugate is applied in the dot product in sympde
    assert abs(x.inner(A1.dot(x)) - res) < 1e-12
    assert abs(x.inner(A2.dot(x)) - res) < 1e-12
    assert abs(x.inner(A3.dot(x)) - res) < 1e-12
    assert abs(x.inner(A4.dot(x)) - res) < 1e-12

    print("PASSED")

#==============================================================================
def test_linearForm_complex(backend):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    # TODO: remove codomain_type when It is implemented in sympde
    V.codomain_type = 'complex'
    v = element_of(V, name='v')
    f = element_of(V, name='f')
    c = Constant(name='c', complex=True)

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
    b1 = l1h.assemble(c=complex(1.0), f=fh)
    b2 = l2h.assemble(c=complex(1.0), f=fh)
    b3 = l3h.assemble(c=res, f=fh)
    b4 = l3h.assemble(c=complex(1.0), f=fh2)


    # Test vector b
    assert abs(b1.toarray().sum() - res) < 1e-12
    assert abs(b2.toarray().sum() - res) < 1e-12
    assert abs(b3.toarray().sum() - res) < 1e-12
    assert abs(b4.toarray().sum() - res) < 1e-12

    print("PASSED")

#==============================================================================
def test_Norm_complex(backend):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)

    # TODO: remove codomain_type when It is implemented in sympde
    V.codomain_type = 'complex'
    v = element_of(V, name='v')
    c = Constant(name='c', complex=True)

    res = (1.+1j)/np.sqrt(2)

    # We try to put complex as a sympy object in the expression
    g1  = (1.+I)/sqrt(2)

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
@pytest.mark.mpi
def test_assemble_complex_parallel(backend):

    # If 'backend' is specified, accelerate Python code by passing **kwargs
    # to discretization of bilinear forms, linear forms and functionals.
    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = ScalarFunctionSpace('V', domain)
    V.codomain_type = 'complex'

    Vr = ScalarFunctionSpace('Vr', domain)
    Vr.codomain_type = 'complex'

    # TODO: remove codomain_type when It is implemented in sympde
    u = element_of(V, name='u')
    v = element_of(V, name='v')
    f = element_of(V, name='f')

    c = Constant(name='c', complex=True)
    gr = c * f**2
    gc = I * c * f**2
    cst = 1.0+0.0j

    ac = BilinearForm((u, v), integral(domain, gc * u * v))
    ar = BilinearForm((u, v), integral(domain, gr * u * v))
    lc = LinearForm(v, integral(domain, gc * v))
    lr = LinearForm(v, integral(domain, gr * v))
    nr = Norm(1.0 *v, domain, kind='l2')
    nc = Norm(1.0j*v, domain, kind='l2')

    ncells = (5, 5)
    degree = (3, 3)
    domain_h = discretize(domain, ncells=ncells, comm=MPI.COMM_WORLD)
    Vh = discretize(V, domain_h, degree=degree)
    Vrh = discretize(Vr, domain_h, degree=degree)
    ach = discretize(ac, domain_h, [Vh, Vh], **kwargs)
    arh = discretize(ar, domain_h, [Vrh, Vrh], **kwargs)
    lch = discretize(lc, domain_h,      Vh , **kwargs)
    lrh = discretize(lr, domain_h,      Vrh , **kwargs)
    nch = discretize(nc, domain_h,      Vh , **kwargs)
    nrh = discretize(nr, domain_h,      Vrh , **kwargs)

    fh = FemField(Vh)
    fh.coeffs[:] = 1

    # Assembly call should not crash if correct arguments are used
    Ac = ach.assemble(c=cst, f=fh)
    Ar = arh.assemble(c=cst, f=fh)
    bc = lch.assemble(f=fh, c=cst)
    br = lrh.assemble(f=fh, c=cst)
    nc = nch.assemble(v=fh)
    nr = nrh.assemble(v=fh)

    # Test matrix Ac and Ar
    #TODO change Ar*1j into -Ar*1j when the conjugate is applied in the dot product in sympde
    assert np.all(abs((Ac)._data-(Ar)._data*1j))<1e-16

    # Test vector bc and br
    assert np.all(abs((bc)._data-(br)._data*1j)<1e-16)

    # Test Norm nc and nr
    assert abs(nc - nr) < 1e-8

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
    assert abs(x.inner(A.dot(x)) - res) < 1e-12

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
def test_non_symmetric_different_space_BilinearForm(backend):

    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square()
    V = VectorFunctionSpace('V', domain, kind='Hdiv')
    X = VectorFunctionSpace('X', domain, kind='h1')

    u = element_of(X, name='u')    
    w = element_of(V, name='w')

    A = BilinearForm((u, w), integral(domain, Inner(u, w)))

    ncells = [4, 4]
    degree = [2, 2]

    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=degree)
    Xh = discretize(X, domain_h, degree=degree)

    ah = discretize(A, domain_h, (Xh, Vh), **kwargs)
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
    div, = derham_h.derivatives(kind='linop')

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

    uh      = array_to_psydac(np.array([i for i in range(nc)]), V0h.coeff_space)
    const_1 = array_to_psydac(np.array([1/nc]*nc), V1h.coeff_space)

    rhoh1 = div.dot(uh)
    rhof1 = FemField(V1h, rhoh1)
    rhoh2 = div.dot(uh)
    rhof2  = FemField(V1h, rhoh2)
    rhoh3 = div.dot(uh)
    rhof3  = FemField(V1h, rhoh3)
    weight_mass_matrix = weight_int_prod_h.assemble(rho=rhof1)
    inte_bilin = const_1.inner(weight_mass_matrix.dot(const_1))

    int_prod_rho = int_prod_h.assemble(rho=rhof2)
    inte_lin = int_prod_rho.inner(const_1)

    inte_norm = func_h.assemble(rho=rhof3)

    assert( abs(inte_bilin) < 1.e-12)    
    assert( abs(inte_lin) < 1.e-12)
    assert( abs(inte_norm) < 1.e-12)

#==============================================================================
if __name__ == '__main__':
    test_Norm_complex(None)
    test_field_and_constant(None, 'real')
    test_multiple_fields(None, 'real')
    test_math_imports(None)
    test_non_symmetric_BilinearForm(None)
    test_non_symmetric_different_space_BilinearForm(None)
    test_assembly_no_synchr_args(None)
