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

import os

base_dir = os.path.dirname(os.path.realpath(__file__))
mesh_dir = os.path.join(base_dir, 'mesh')

domain = Domain('\Omega', dim=3)

def test_api_poisson_3d_dir_collela():

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
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_3d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)
    B5 = DiscreteBoundary(B5, axis=2, ext=-1)
    B6 = DiscreteBoundary(B6, axis=2, ext= 1)

    bc = [DiscreteDirichletBC(i) for i in [B1, B2, B3, B4, B5, B6]]
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.8151461486397859
    expected_h1_error =  7.887790839303131

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...


#==============================================================================
def test_api_poisson_3d_dirneu_identity_2():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # neumann  bc will be applied on B2

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.5*pi*x)*sin(pi*y)*sin(pi*z)

    expr = (9./4.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B2(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B2)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_3d.h5'))
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.007476406034615364
    expected_h1_error =  0.20417783668832656

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_3d_dirneu_identity_24():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # neumann  bc will be applied on B2
    B4 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B4

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.5*pi*x)*sin(0.5*pi*y)*sin(pi*z)

    expr = (3./2.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr)

    expr = l0(v) + l_B2(v) + l_B4(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-(B2+B4))]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_3d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(-(B2+B4))]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B2,B4], bc=bc)
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.005339281019682795
    expected_h1_error =  0.14517242816351233

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_3d_dirneu_identity_13():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)*sin(pi*z)

    expr = (3./2.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B3(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-(B1+B3))]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_3d.h5'))
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.005339281019684631
    expected_h1_error =  0.14517242816351372

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_3d_dirneu_identity_123():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain)
    B5 = Boundary(r'\Gamma_5', domain)
    B6 = Boundary(r'\Gamma_6', domain)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*sin(pi*z)

    expr = (21./16.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(B) for B in [B4,B5,B6]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_3d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)
    B5 = DiscreteBoundary(B5, axis=2, ext=-1)
    B6 = DiscreteBoundary(B6, axis=2, ext= 1)

    bc = [DiscreteDirichletBC(B) for B in [B4,B5,B6]]
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.00681948184967118
    expected_h1_error =  0.18416036905795535

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_3d_dirneu_identity_1235():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain)
    B5 = Boundary(r'\Gamma_5', domain) # Neumann bc will be applied on B5
    B6 = Boundary(r'\Gamma_6', domain)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*cos(0.5*pi*z)

    expr = (9./16.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B5)
    l_B5 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v) + l_B5(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(B) for B in [B4,B6]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'identity_3d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)
    B5 = DiscreteBoundary(B5, axis=2, ext=-1)
    B6 = DiscreteBoundary(B6, axis=2, ext= 1)

    bc = [DiscreteDirichletBC(B) for B in [B4,B6]]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1,B2,B3,B5], bc=bc)
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.0010224893148868808
    expected_h1_error =  0.030546175685500224

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_3d_dirneu_collela_2():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # neumann  bc will be applied on B2

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.25*pi*(x+1.))*sin(pi*y)*sin(pi*z)

    expr = (33./16.)*pi**2*solution*v

    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = l0(v) + l_B2(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-B2)]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_3d.h5'))
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.31651011929794615
    expected_h1_error =  3.315168073716373

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_3d_dirneu_collela_24():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B2 = Boundary(r'\Gamma_2', domain) # neumann  bc will be applied on B2
    B4 = Boundary(r'\Gamma_4', domain) # Neumann bc will be applied on B4

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.25*pi*(x+1.))*sin(0.25*pi*(y+1.))*sin(pi*z)

    expr = (9./8.)*pi**2*solution*v
    l0 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr, mapping=mapping)

    expr = v*trace_1(grad(solution), B4)
    l_B4 = LinearForm(v, expr)

    expr = l0(v) + l_B2(v) + l_B4(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-(B2+B4))]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_3d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)

    bc = [DiscreteDirichletBC(-(B2+B4))]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B2,B4], bc=bc)
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.19713579709344806
    expected_h1_error =  1.9973824048200948

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_3d_dirneu_collela_13():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = sin(0.25*pi*(1.-x))*sin(0.25*pi*(1.-y))*sin(pi*z)

    expr = (9./8.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B3(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(-(B1+B3))]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_3d.h5'))
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.1967707543568222
    expected_h1_error =  1.997279767542349

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_3d_dirneu_collela_123():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain)
    B5 = Boundary(r'\Gamma_5', domain)
    B6 = Boundary(r'\Gamma_6', domain)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(pi*x)*sin(0.25*pi*(1.-y))*sin(pi*z)

    expr = (33./16.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(B) for B in [B4,B5,B6]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_3d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)
    B5 = DiscreteBoundary(B5, axis=2, ext=-1)
    B6 = DiscreteBoundary(B6, axis=2, ext= 1)

    bc = [DiscreteDirichletBC(B) for B in [B4,B5,B6]]
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.6216671144774677
    expected_h1_error =  5.854903143947668

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...

#==============================================================================
def test_api_poisson_3d_dirneu_collela_1235():

    # ... abstract model
    mapping = Mapping('M', rdim=3, domain=domain)

    U = FunctionSpace('U', domain)
    V = FunctionSpace('V', domain)

    B1 = Boundary(r'\Gamma_1', domain) # Neumann bc will be applied on B1
    B2 = Boundary(r'\Gamma_2', domain) # Neumann bc will be applied on B2
    B3 = Boundary(r'\Gamma_3', domain) # Neumann bc will be applied on B3
    B4 = Boundary(r'\Gamma_4', domain)
    B5 = Boundary(r'\Gamma_5', domain) # Neumann bc will be applied on B5
    B6 = Boundary(r'\Gamma_6', domain)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(U, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr, mapping=mapping)

    solution = cos(pi*x)*sin(0.25*pi*(1.-y))*sin(0.25*pi*(1.-z))

    expr = (9./8.)*pi**2*solution*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B1)
    l_B1 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B2)
    l_B2 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B3)
    l_B3 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B5)
    l_B5 = LinearForm(v, expr)

    expr = l0(v) + l_B1(v) + l_B2(v) + l_B3(v) + l_B5(v)
    l = LinearForm(v, expr, mapping=mapping)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2', name='u', mapping=mapping)
    h1norm = Norm(error, domain, kind='h1', name='u', mapping=mapping)

    bc = [DirichletBC(B) for B in [B4,B6]]
    equation = Equation(a(v,u), l(v), bc=bc)
    # ...

    # ... discrete spaces
    Vh, mapping = fem_context(os.path.join(mesh_dir, 'collela_3d.h5'))
    # ...

    # ... dsicretize the equation using Dirichlet bc
    B1 = DiscreteBoundary(B1, axis=0, ext=-1)
    B2 = DiscreteBoundary(B2, axis=0, ext= 1)
    B3 = DiscreteBoundary(B3, axis=1, ext=-1)
    B4 = DiscreteBoundary(B4, axis=1, ext= 1)
    B5 = DiscreteBoundary(B5, axis=2, ext=-1)
    B6 = DiscreteBoundary(B6, axis=2, ext= 1)

    bc = [DiscreteDirichletBC(B) for B in [B4,B6]]
    equation_h = discretize(equation, [Vh, Vh], mapping, boundary=[B1,B2,B3,B5], bc=bc)
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
    phi.coeffs[:,:,:] = x[:,:,:]
    # ...

    # ... compute norms
    l2_error = l2norm_h.assemble(F=phi)
    h1_error = h1norm_h.assemble(F=phi)

    expected_l2_error =  0.6863456565612966
    expected_h1_error =  5.177931817581279

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


###############################################
if __name__ == '__main__':
    test_api_poisson_3d_dirneu_collela_2()
    print('')
