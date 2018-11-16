# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S

from sympde.core import dx, dy, dz
from sympde.core import Mapping
from sympde.core import Constant
from sympde.core import Field
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import FunctionSpace, VectorFunctionSpace
from sympde.core import TestFunction
from sympde.core import VectorTestFunction
from sympde.core import BilinearForm, LinearForm, Integral
from sympde.core import Norm
from sympde.core import Equation, DirichletBC
from sympde.core import Domain
from sympde.core import Boundary, trace_0, trace_1
from sympde.core import ComplementBoundary
from sympde.gallery import Poisson, Stokes

from spl.fem.context import fem_context
from spl.fem.basic   import FemField
from spl.fem.splines import SplineSpace
from spl.fem.tensor  import TensorFemSpace
from spl.api.discretization import discretize
from spl.api.boundary_condition import DiscreteBoundary
from spl.api.boundary_condition import DiscreteComplementBoundary
from spl.api.boundary_condition import DiscreteDirichletBC

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose
from utils import assert_identical_coo

DEBUG = False

domain = Domain('\Omega', dim=1)

def create_discrete_space(p=2, ne=2):
    # ... discrete spaces
    # Input data: degree, number of elements

    # Create uniform grid
    grid = linspace( 0., 1., num=ne+1 )

    # Create finite element space and precompute quadrature data
    V = SplineSpace( p, grid=grid )
    V.init_fem()
    # ...

    return V

def test_api_poisson_1d_dir_1():
    print('============ test_api_poisson_1d_dir_1 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)

    x = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = pi**2*sin(pi*x)*v
    l = LinearForm(v, expr)

    error = F-sin(pi*x)
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(i) for i in [B1, B2]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    bc = [DiscreteDirichletBC(i) for i in [B1, B2]]
    equation_h = discretize(equation, [Vh, Vh], bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh)
    h1norm_h = discretize(h1norm, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:] = x[:]
    # ...

    # ... compute norms
    error = l2norm_h.assemble(F=phi)
    print('> L2 norm      = ', error)

    error = h1norm_h.assemble(F=phi)
    print('> H1 seminorm  = ', error)
    # ...

def test_api_poisson_1d_dirneu_1():
    print('============ test_api_poisson_1d_dirneu_1 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    solution = sin(0.5*pi*x)

    expr = (1./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = l0(v) + l_B2(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(-B2)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space()
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    bc = [DiscreteDirichletBC(-B2)]
    equation_h = discretize(equation, [Vh, Vh], boundary=B2, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh)
    h1norm_h = discretize(h1norm, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:] = x[:]
    # ...

    # ... compute norms
    error = l2norm_h.assemble(F=phi)
    print('> L2 norm      = ', error)

    error = h1norm_h.assemble(F=phi)
    print('> H1 seminorm  = ', error)
    # ...


###############################################
if __name__ == '__main__':

    # ... examples without mapping
    test_api_poisson_1d_dir_1()
    test_api_poisson_1d_dirneu_1()
    # ...
