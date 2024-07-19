import pytest
import numpy as np
from mpi4py import MPI
from sympy import Domain, pi, sin, cos, tan, atan, atan2, exp, sinh, cosh, tanh, atanh, Tuple, I, sqrt
from sympy import lambdify
from sympde.topology import Line, Square
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of, Derham, elements_of
from sympde.core     import Constant
from sympde.expr     import LinearForm, BilinearForm, Functional, Norm, SemiNorm
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
    assert abs(x.dot(A.dot(x)) - res) < 1e-12

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
    assert abs(x.dot(A1.dot(x)) - res) < 1e-12
    assert abs(x.dot(A2.dot(x)) - res) < 1e-12
    assert abs(x.dot(A3.dot(x)) - res) < 1e-12
    assert abs(x.dot(A4.dot(x)) - res) < 1e-12

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
@pytest.mark.parallel
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

# def norm_projected(expr, Vh, nquads):
#     from psydac.feec.global_projectors import Projector_L2
#     P = Projector_L2(Vh, nquads=nquads)
#     solution_call = lambdify(Domain.coordinates, solution)
#     sol_h = P(solution_call)
#     sol_c = sol_h.coeffs
#     a = BilinearForm((u, v),  integral(domain, u*v))
#     a_h = discretize(a, domain_h, [Vh,Vh], backend=None)
#     M = a_h.assemble()
#     l2_norm_psydac = np.sqrt(sol_c.dot(M.dot(sol_c)))

    

def test_sympde_norm(backend):

    kwargs = {'backend': PSYDAC_BACKENDS[backend]} if backend else {}

    domain = Square('domain', bounds1=(0, 1), bounds2=(0, 1))

    U = ScalarFunctionSpace('U', domain, kind=None)
    U.codomain_type='real'    
    V = ScalarFunctionSpace('V', domain, kind=None)
    V.codomain_type='complex'
    u1, u2 = elements_of(U, names='u1, u2')
    v1, v2 = elements_of(V, names='v1, v2')

    x, y = domain.coordinates
    kappa = 2*pi
    rho = sin(kappa * y)    # real    
    dx_rho = 0 * y
    dy_rho = kappa * cos(kappa * y)
    phi = exp(1j * kappa * x) * rho  # complex
    # other possible expressions? also fail:
    # phi = exp(I * kappa * x) * rho 
    # phi = (cos(kappa * x) + 1j * sin(kappa * x)) * rho  
    # phi = (cos(kappa * x) + 1.j * sin(kappa * x)) * rho  
    dx_phi = 1j * kappa * phi
    dy_phi = exp(1j * kappa * x) * dy_rho

    # sympde L2 norms
    rho_l2_sym = Norm(rho, domain, kind='l2')
    phi_l2_sym = Norm(phi, domain, kind='l2') 
    # sympde H1 semi-norms
    rho_h1s_sym = SemiNorm(rho, domain, kind='h1')
    phi_h1s_sym = SemiNorm(phi, domain, kind='h1')

    # Q: can we evaluate these norms directly (in sympde)?
    
    # aux 
    rho_l2_usym = Norm(rho - u1, domain, kind='l2')
    rho_l2_vsym = Norm(rho - v1, domain, kind='l2')
    # phi_l2_vsym = Norm(phi - v1, domain, kind='l2')
    phi_l2_vsym = Norm(v1 - phi, domain, kind='l2')
    rho_h1s_usym = SemiNorm(rho - u1, domain, kind='h1')
    rho_h1s_vsym = SemiNorm(rho - v1, domain, kind='h1')
    phi_h1s_vsym = SemiNorm(phi - v1, domain, kind='h1')
    
    # aux 2
    l2_usym = Norm(u1, domain, kind='l2')
    l2_vsym = Norm(v1, domain, kind='l2')
    h1s_usym = SemiNorm(u1, domain, kind='h1')
    h1s_vsym = SemiNorm(v1, domain, kind='h1')

    # exact norms
    rho_l2_ex = sqrt((1-sin(2*kappa)/(2*kappa))/2)
    rho_h1s_ex = kappa * sqrt(1-rho_l2_ex**2) # todo
    rho_h1_ex = sqrt(rho_l2_ex**2 + rho_h1s_ex**2)

    phi_l2_ex = rho_l2_ex
    phi_h1s_ex = kappa
    phi_h1_ex = sqrt(phi_l2_ex**2 + phi_h1s_ex**2)
    
    # discretize norms
    ncells = [8, 8]
    degree = [4, 4]
    domain_h = discretize(domain, ncells=ncells, periodic=[False, False])
    Uh       = discretize(U, domain_h, degree=degree)
    Vh       = discretize(V, domain_h, degree=degree)

    # commented because currently not working. TODO: fix this
    # rho_l2_h = discretize(rho_l2_sym, domain_h, Uh, **kwargs)  # todo: also try in Vh
    # phi_l2_h = discretize(phi_l2_sym, domain_h, Vh, **kwargs)    
    # rho_l2 = rho_l2_h.assemble()
    # phi_l2 = phi_l2_h.assemble()
    rho_l2 = 'NOT IMPLEMENTED (todo)'
    phi_l2 = 'NOT IMPLEMENTED (todo)'

    # aux 
    rho_l2_usym_h = discretize(rho_l2_usym, domain_h, Uh, **kwargs)
    zero_h = FemField(Uh)
    rho_l2_u0 = rho_l2_usym_h.assemble(u1=zero_h)

    rho_l2_vsym_h = discretize(rho_l2_vsym, domain_h, Vh, **kwargs)
    zero_h = FemField(Vh)
    rho_l2_v0 = rho_l2_vsym_h.assemble(v1=zero_h)

    phi_l2_vsym_h = discretize(phi_l2_vsym, domain_h, Vh, **kwargs)
    zero_h = FemField(Vh)
    phi_l2_v0 = phi_l2_vsym_h.assemble(v1=zero_h)

    rho_h1s_usym_h = discretize(rho_h1s_usym, domain_h, Uh, **kwargs)
    zero_h = FemField(Uh)
    rho_h1s_u0 = rho_h1s_usym_h.assemble(u1=zero_h)

    rho_h1s_vsym_h = discretize(rho_h1s_vsym, domain_h, Vh, **kwargs)
    zero_h = FemField(Vh)
    rho_h1s_v0 = rho_h1s_vsym_h.assemble(v1=zero_h)

    phi_h1s_vsym_h = discretize(phi_h1s_vsym, domain_h, Vh, **kwargs)
    zero_h = FemField(Vh)
    phi_h1s_v0 = phi_h1s_vsym_h.assemble(v1=zero_h)

    # discretize norms through L2 projection
    m_U = BilinearForm((u1, v1),  integral(domain, u1*v1))
    m_V = BilinearForm((u2, v2),  integral(domain, u2*v2))
    mh_U = discretize(m_U, domain_h, [Uh,Uh], **kwargs)
    mh_V = discretize(m_V, domain_h, [Vh,Vh], **kwargs)
    M_U = mh_U.assemble()
    M_V = mh_V.assemble()
    
    from psydac.feec.global_projectors import Projector_L2
    P_U = Projector_L2(Uh, nquads=[2*d+1 for d in degree])
    P_V = Projector_L2(Vh, nquads=[2*d+1 for d in degree])

    def proj(fun_sym, space = 'U'):
        fun = lambdify(domain.coordinates, fun_sym)
        if space == 'U':
            return P_U(fun)
        elif space == 'V':
            return P_V(fun)
        else:
            raise ValueError('space must be either "U" or "V"')

    rho_h = proj(rho, space = 'U') # todo: try with V
    dx_rho_h = proj(dx_rho, space = 'U')
    dy_rho_h = proj(dy_rho, space = 'U')
    phi_h = proj(phi, space = 'V')
    dx_phi_h = proj(dx_phi, space = 'V')
    dy_phi_h = proj(dy_phi, space = 'V')

    rho_c = rho_h.coeffs
    dx_rho_c = dx_rho_h.coeffs
    dy_rho_c = dy_rho_h.coeffs
    phi_c = phi_h.coeffs
    dx_phi_c = dx_phi_h.coeffs
    dy_phi_c = dy_phi_h.coeffs

    rho_h_l2  = np.sqrt(rho_c.dot(M_U.dot(rho_c)))
    rho_h_h1s = np.sqrt(dx_rho_c.dot(M_U.dot(dx_rho_c)) + dy_rho_c.dot(M_U.dot(dy_rho_c))) 

    phi_h_l2 = np.sqrt(np.real(phi_c.dot(M_V.dot(phi_c))))
    phi_h_h1s = np.sqrt(np.real(dx_phi_c.dot(M_V.dot(dx_phi_c)) + dy_phi_c.dot(M_V.dot(dy_phi_c))))

    # aux 2
    l2_usym_h  = discretize(l2_usym, domain_h, Uh, **kwargs)   # todo: try with V
    h1s_usym_h = discretize(h1s_usym, domain_h, Uh, **kwargs)
    l2_vsym_h  = discretize(l2_vsym, domain_h, Vh, **kwargs)
    h1s_vsym_h = discretize(h1s_vsym, domain_h, Vh, **kwargs)

    l2_urho  = l2_usym_h.assemble(u1=rho_h)
    h1s_urho = h1s_usym_h.assemble(u1=rho_h)

    l2_vphi  = l2_vsym_h.assemble(v1=phi_h)
    h1s_vphi = h1s_vsym_h.assemble(v1=phi_h)


    # compare
        
    tol  = 1e-12
    htol = 1e-4

    print(" ---- ---- ---- ---- ---- ---- ---- ")
    print(" ---- ---- ---- ---- ---- ---- ---- ")
    print(" rho L2 norms: ")    
    print(f'rho_l2     = {rho_l2}')
    print(f'rho_l2_u0  = {rho_l2_u0}')
    print(f'rho_l2_v0  = {rho_l2_v0}')
    print(f'rho_l2_ex  = {rho_l2_ex.evalf()}')
    print(f'rho_h_l2   = {rho_h_l2}')
    print(f'l2_urho    = {l2_urho}')
    rho_l2_ref = rho_l2_ex.evalf()    
    # assert abs(rho_l2 - rho_l2_ref) < tol  # TODO
    assert abs(rho_l2_u0 - rho_l2_ref) < tol
    assert abs(rho_l2_v0 - rho_l2_ref) < tol
    assert abs(rho_h_l2 - rho_l2_ref) < htol
    assert abs(l2_urho - rho_l2_ref) < htol
    print(" ---- ---- ---- ---- ---- ---- ---- ")
    print(" phi L2 norms: ")
    print(f'phi_l2     = {phi_l2}')
    print(f'phi_l2_v0  = {phi_l2_v0}')
    print(f'phi_l2_ex  = {phi_l2_ex.evalf()}')
    print(f'phi_h_l2   = {phi_h_l2}')
    print(f'l2_vphi    = {l2_vphi}')
    phi_l2_ref = phi_l2_ex.evalf()
    # assert abs(phi_l2 - phi_l2_ref) < tol  # TODO
    assert abs(phi_l2_v0 - phi_l2_ref) < tol  # FAILS!
    assert abs(phi_h_l2 - phi_l2_ref) < htol
    assert abs(l2_vphi - phi_l2_ref) < htol  # FAILS!
    print(" ---- ---- ---- ---- ---- ---- ---- ")
    print(" ---- ---- ---- ---- ---- ---- ---- ")
    print(" rho H1-seminorms: ")
    print(f'rho_h1s_u0  = {rho_h1s_u0}')
    print(f'rho_h1s_v0  = {rho_h1s_v0}')
    print(f'rho_h1s_ex  = {rho_h1s_ex.evalf()}')
    print(f'rho_h_h1s   = {rho_h_h1s}')
    print(f'h1s_urho    = {h1s_urho}')
    rho_h1s_ref = rho_h1s_ex.evalf()
    assert abs(rho_h1s_u0 - rho_h1s_ref) < tol  # FAILS!
    assert abs(rho_h1s_v0 - rho_h1s_ref) < tol  # FAILS!
    assert abs(rho_h_h1s - rho_h1s_ref) < htol
    assert abs(h1s_urho - rho_h1s_ref) < htol
    print(" ---- ---- ---- ---- ---- ---- ---- ")
    print(" phi H1-seminorms: ")
    print(f'phi_h1s_v0  = {phi_h1s_v0}')
    print(f'phi_h1s_ex  = {phi_h1s_ex.evalf()}')
    print(f'phi_h_h1s   = {phi_h_h1s}')
    print(f'h1s_vphi    = {h1s_vphi}')
    phi_h1s_ref = phi_h1s_ex.evalf()
    assert abs(phi_h1s_v0 - phi_h1s_ref) < tol # FAILS!
    assert abs(phi_h_h1s - phi_h1s_ref) < htol
    assert abs(h1s_vphi - phi_h1s_ref) < htol # FAILS!
    print(" ---- ---- ---- ---- ---- ---- ---- ")
    print(" ---- ---- ---- ---- ---- ---- ---- ")

    

#==============================================================================
if __name__ == '__main__':

    test_Norm_complex(None)
    test_sympde_norm(None)
    exit()
    test_field_and_constant(None)
    test_multiple_fields(None)
    test_math_imports(None)
    test_non_symmetric_BilinearForm(None)
    test_non_symmetric_different_space_BilinearForm(None)
    test_assembly_no_synchr_args(None)
