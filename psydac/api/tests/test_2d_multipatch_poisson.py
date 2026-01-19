#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from sympy import pi, sin

from sympde.calculus      import grad, dot
from sympde.calculus      import minus, plus
from sympde.topology      import Square, Domain
from sympde.topology      import ScalarFunctionSpace
from sympde.topology      import elements_of
from sympde.topology      import NormalVector
from sympde.topology      import IdentityMapping, AffineMapping
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.expr     import Norm, SemiNorm
from sympde.expr.equation import find, EssentialBC

from psydac.api.discretization import discretize

#==============================================================================
def run_poisson_2d(solution, f, domain, ncells, degree):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V   = ScalarFunctionSpace('V', domain, kind=None)

    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')

    bc   = EssentialBC(u, 0, domain.boundary)

    error  = u - solution

    I = domain.interfaces

    kappa  = 10**3

    #expr_I =- dot(grad(plus(u)),nn)*minus(v)  + dot(grad(minus(v)),nn)*plus(u) - kappa*plus(u)*minus(v)\
    #        + dot(grad(minus(u)),nn)*plus(v)  - dot(grad(plus(v)),nn)*minus(u) - kappa*plus(v)*minus(u)\
    #        - dot(grad(plus(v)),nn)*plus(u)   + kappa*plus(u)*plus(v)\
    #        - dot(grad(minus(v)),nn)*minus(u) + kappa*minus(u)*minus(v)

    expr_I =- 0.5*dot(grad(plus(u)),nn)*minus(v)  + 0.5*dot(grad(minus(v)),nn)*plus(u)  - kappa*plus(u)*minus(v)\
            + 0.5*dot(grad(minus(u)),nn)*plus(v)  - 0.5*dot(grad(plus(v)),nn)*minus(u)  - kappa*plus(v)*minus(u)\
            - 0.5*dot(grad(minus(v)),nn)*minus(u) - 0.5*dot(grad(minus(u)),nn)*minus(v) + kappa*minus(u)*minus(v)\
            + 0.5*dot(grad(plus(v)),nn)*plus(u)   + 0.5*dot(grad(plus(u)),nn)*plus(v)   + kappa*plus(u)*plus(v)

    expr   = dot(grad(u),grad(v))

    a = BilinearForm((u,v), integral(domain, expr) + integral(I, expr_I))
    l = LinearForm(v, integral(domain, f*v))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)

    l2norm =     Norm(error, domain, kind='l2')
    h1norm = SemiNorm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    domain_h = discretize(domain, ncells=ncells)
    Vh       = discretize(V, domain_h, degree=degree)

    equation_h = discretize(equation, domain_h, [Vh, Vh])

    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)

    uh = equation_h.solve()

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return l2_error, h1_error

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_0():
    A = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, 1))

    connectivity = [((0,0,1),(1,0,-1))]
    patches = [A,B]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y = domain.coordinates

    solution = x*y*(1-y)*(1-x)
    f        = -2*x*(x - 1) -2*y*(y - 1)

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 2.176726763610992e-09
    expected_h1_error = 2.9725703533101877e-09

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_1():
    A = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, 1))

    connectivity = [((0,0,1),(1,0,-1))]
    patches = [A,B]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.002035229666394183
    expected_h1_error = 0.056796387991647795

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_2():
    A = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, 1))

    connectivity = [((0,0,1),(1,0,-1))]
    patches = [A,B]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.002035229666394183
    expected_h1_error = 0.056796387991647795

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_3():
    A = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B = Square('B',bounds1=(0, 0.5), bounds2=(0, 1))

    M1 = IdentityMapping('M1',2)
    M2 = AffineMapping('M2',2, c1=1, c2=0,
        a11=-1, a12=0,
        a21=0, a22=1)

    D1 = M1(A)
    D2 = M2(B)

    connectivity = [((0,0,1),(1,0,1))]
    patches = [D1,D2]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.0020352296663948295
    expected_h1_error = 0.05679638799164739

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#------------------------------------------------------------------------------
def test_poisson_2d_2_patch_dirichlet_4():
    A = Square('A',bounds1=(0, 0.5), bounds2=(0, 1))
    B = Square('B',bounds1=(0, 0.5), bounds2=(0, 1))

    M1 = AffineMapping('M1',2, c1=0.5, c2=0.,a11=1,  a22=1, a12=0, a21=0)
    M2 = AffineMapping('M2',2, c1=0.5, c2=0, a11=-1, a12=0, a21=0, a22=1)

    D1 = M1(A)
    D2 = M2(B)

    connectivity = [((0,0,-1),(1,0,-1))]
    patches = [D1,D2]
    domain = Domain.join(patches, connectivity, 'domain')

    x,y = domain.coordinates
    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*solution

    l2_error, h1_error = run_poisson_2d(solution, f, domain, ncells=[2**2,2**2], degree=[2,2])

    expected_l2_error = 0.0020352296663934746
    expected_h1_error = 0.05679638799164659

    assert ( abs(l2_error - expected_l2_error) < 1e-7 )
    assert ( abs(h1_error - expected_h1_error) < 1e-7 )

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

