# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S
from sympy import Tuple
from sympy import Matrix

from sympde.core import dx, dy, dz
from sympde.core import Mapping
from sympde.core import Constant
from sympde.core import Field
from sympde.core import VectorField
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
from spl.fem.vector  import ProductFemSpace, VectorFemField
from spl.api.discretization import discretize
from spl.api.boundary_condition import DiscreteBoundary
from spl.api.boundary_condition import DiscreteComplementBoundary
from spl.api.boundary_condition import DiscreteDirichletBC
from spl.api.settings import SPL_BACKEND_PYTHON, SPL_BACKEND_PYCCEL

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose
from utils import assert_identical_coo

domain = Domain('\Omega', dim=3)

def create_discrete_space(p=(2,2,2), ne=(2,2,2)):
    # ... discrete spaces
    # Input data: degree, number of elements
    p1,p2,p3 = p
    ne1,ne2,ne3 = ne

    # Create uniform grid
    grid_1 = linspace( 0., 1., num=ne1+1 )
    grid_2 = linspace( 0., 1., num=ne2+1 )
    grid_3 = linspace( 0., 1., num=ne3+1 )

    # Create 1D finite element spaces and precompute quadrature data
    V1 = SplineSpace( p1, grid=grid_1 ); V1.init_fem()
    V2 = SplineSpace( p2, grid=grid_2 ); V2.init_fem()
    V3 = SplineSpace( p3, grid=grid_3 ); V3.init_fem()

    # Create 3D tensor product finite element space
    V = TensorFemSpace( V1, V2, V3 )
    # ...

    return V


def test_api_poisson_3d_dir_1(backend=SPL_BACKEND_PYTHON):
    print('============ test_api_poisson_3d_dir_1 =============')

    # ... abstract model
    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)
    B5 = Boundary(r'\Gamma_5', domain)
    B6 = Boundary(r'\Gamma_6', domain)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)*v
    l = LinearForm(v, expr)

    error = F - sin(pi*x)*sin(pi*y)*sin(pi*z)
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    bc = [DirichletBC(i) for i in [B1, B2, B3, B4, B5, B6]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh = create_discrete_space(p=(2,2,2), ne=(2**3,2**3,2**3))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)
    B5 = DiscreteBoundary(B5, axis=2, ext=-1)
    B6 = DiscreteBoundary(B6, axis=2, ext= 1)

    bc = [DiscreteDirichletBC(i) for i in [B1, B2, B3, B4, B5, B6]]
    equation_h = discretize(equation, [Vh, Vh], bc=bc, backend=backend)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, backend=backend)
    h1norm_h = discretize(h1norm, Vh, backend=backend)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    error = l2norm_h.assemble(F=phi)
    print('> L2 norm      = ', error)

    error = h1norm_h.assemble(F=phi)
    print('> H1 seminorm  = ', error)
    # ...

def test_api_poisson_3d_dir_1_mapping(backend=SPL_BACKEND_PYTHON):
    print('============ test_api_poisson_3d_dir_1_mapping =============')

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)
    B5 = Boundary(r'\Gamma_5', domain)
    B6 = Boundary(r'\Gamma_6', domain)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    expr = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)*v
    l = LinearForm(v, expr, mapping=mapping)

    error = F - sin(pi*x)*sin(pi*y)*sin(pi*z)
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(i) for i in [B1, B2, B3, B4, B5, B6]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context('identity_3d.h5')
#    Vh, mapping = fem_context('collela_3d.h5')
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)
    B5 = DiscreteBoundary(B5, axis=2, ext=-1)
    B6 = DiscreteBoundary(B6, axis=2, ext= 1)

    bc = [DiscreteDirichletBC(i) for i in [B1, B2, B3, B4, B5, B6]]
    equation_h = discretize(equation, [Vh, Vh], mapping, bc=bc, backend=backend)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping, backend=backend)
    h1norm_h = discretize(h1norm, Vh, mapping, backend=backend)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    error = l2norm_h.assemble(F=phi)
    print('> L2 norm      = ', error)

    error = h1norm_h.assemble(F=phi)
    print('> H1 seminorm  = ', error)
    # ...


###############################################
if __name__ == '__main__':

    # ... without mapping
#    test_api_poisson_3d_dir_1(backend=SPL_BACKEND_PYCCEL)
    # ...

    # ... with mapping
    test_api_poisson_3d_dir_1_mapping(backend=SPL_BACKEND_PYCCEL)
    # ...
