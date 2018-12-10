# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin, exp
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

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose

import os

base_dir = os.path.dirname(os.path.realpath(__file__))
mesh_dir = os.path.join(base_dir, 'mesh')

domain = Domain('\Omega', dim=2)


#==============================================================================
def test_api_poisson_2d_dir_identity():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    expr = 2*pi**2*sin(pi*x)*sin(pi*y)*v
    l = LinearForm(v, expr, mapping=mapping)

    error = F-sin(pi*x)*sin(pi*y)
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(i) for i in [B1, B2, B3, B4]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(i) for i in [B1, B2, B3, B4]]
    equation_h = discretize(equation, [Vh, Vh], mapping, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0006542603581247817
    expected_h1_error =  0.039070712161073926

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...



def test_api_poisson_2d_dir_collela():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain)
    B2 = Boundary(r'\Gamma_2', domain)
    B3 = Boundary(r'\Gamma_3', domain)
    B4 = Boundary(r'\Gamma_4', domain)

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    expr = 2*pi**2*sin(pi*x)*sin(pi*y)*v
    l = LinearForm(v, expr, mapping=mapping)

    error = F-sin(pi*x)*sin(pi*y)
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(i) for i in [B1, B2, B3, B4]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(i) for i in [B1, B2, B3, B4]]
    equation_h = discretize(equation, [Vh, Vh], mapping, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.09098801047984553
    expected_h1_error =  1.2367524458055985

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_identity_1():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.5*pi*(1.-x))*sin(pi*y)

    expr = (5./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B1(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B1)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)

    bc = [DiscreteDirichletBC(-B1)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B1, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0004663817338795957
    expected_h1_error =  0.027807908353581108

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_identity_2():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.5*pi*x)*sin(pi*y)

    expr = (5./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B2(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B2)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    bc = [DiscreteDirichletBC(-B2)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B2, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0004663817338528758
    expected_h1_error =  0.02780790835358418

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...


#==============================================================================
def test_api_poisson_2d_dirneu_identity_3():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(pi*x)*sin(0.5*pi*(1.-y))

    expr = (5./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B3(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B3)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)

    bc = [DiscreteDirichletBC(-B3)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B3, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.00046638173388566565
    expected_h1_error =  0.02780790835358236

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_identity_4():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B4 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(pi*x)*sin(0.5*pi*y)

    expr = (5./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B4(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B4)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(-B4)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B4, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.00046638173385220663
    expected_h1_error =  0.02780790835356847

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_collela_1():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B1

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.25*pi*(1.-x))*sin(pi*y)

    expr = (17./16.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B1(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B1)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)

    bc = [DiscreteDirichletBC(-B1)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B1, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.04062215219339022
    expected_h1_error =  0.5936839071478804

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_collela_2():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.25*pi*(x+1.))*sin(pi*y)

    expr = (17./16.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B2(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B2)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    bc = [DiscreteDirichletBC(-B2)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B2, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.03867254728233516
    expected_h1_error =  0.5866068983918493

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_collela_3():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B3 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B3

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.25*pi*(1.-y))*sin(pi*x)

    expr = (17./16.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B3(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B3)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)

    bc = [DiscreteDirichletBC(-B3)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B3, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.04062215219345227
    expected_h1_error =  0.5936839071478792

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_collela_4():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B4 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.25*pi*(y+1.))*sin(pi*x)

    expr = (17./16.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B4(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B4)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(-B4)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B4, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.03867254728233587
    expected_h1_error =  0.58660689839185

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...


#==============================================================================
def test_api_poisson_2d_dirneu_square_mod_a():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.5*pi*x)*sin(pi*y)

    expr = (5./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B2(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B2)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'square_mod_a.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    bc = [DiscreteDirichletBC(-B2)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=B2, bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  1.274563312270839e-05
    expected_h1_error =  0.0012737740540020835

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_square_mod_b():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.5*pi*(x-0.5))*sin(pi*y)

    expr = (5./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B1(v) + l_B2(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-(B1+B2))]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'square_mod_b.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)

    bc = [DiscreteDirichletBC(-(B1+B2))]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1,B2], bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  1.0055191929198286e-05
    expected_h1_error =  0.0010058235188393288

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_identity_13():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)

    expr = (1./2.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B1(v) + l_B3(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-(B1+B3))]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)

    bc = [DiscreteDirichletBC(-(B1+B3))]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1,B3], bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  7.835967808039396e-05
    expected_h1_error =  0.004809729086396761

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_dirneu_identity_123():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain) # Dirichlet H. bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(0.25*pi*x)*cos(0.5*pi*y)

    expr = (5./16.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(B4)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(B4)]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1,B2,B3], bc=bc)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  7.106271222962881e-05
    expected_h1_error =  0.004357097026355032

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...


#==============================================================================
def test_api_poisson_2d_neu_identity():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u)) + v*u
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(pi*x)*cos(pi*y)

    expr = (2.*pi**2 + 1.)*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v) + l_B4(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    equation = Equation(a(v,u), l(v))
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1,B2,B3,B4])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0006518539616462576
    expected_h1_error =  0.038954558964370896

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_2d_neu_collela():

    # ... abstract model
    mapping = Mapping('M', rdim=2, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B4

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u)) + v*u
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(pi*x)*cos(pi*y)

    expr = (2.*pi**2 + 1.)*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v) + l_B4(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    equation = Equation(a(v,u), l(v))
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_2d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1,B2,B3,B4])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, mapping)
    h1norm_h = discretize(h1norm, Vh, mapping)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    phi = FemField( Vh, 'phi' )
    phi.coeffs[:,:] = x[:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.08881000572443457
    expected_h1_error =  1.2203240282935726

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
