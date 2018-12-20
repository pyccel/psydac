# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S
from sympy import Tuple
from sympy import Matrix

from sympde.core import Constant
from sympde.core import grad, dot, inner, cross, rot, curl, div
from sympde.core import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import Field, VectorField
from sympde.topology import ProductSpace
from sympde.topology import TestFunction
from sympde.topology import VectorTestFunction
from sympde.topology import Unknown
from sympde.topology import InteriorDomain, Union
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.topology import Mapping
from sympde.topology import Domain
from sympde.expr import BilinearForm, LinearForm, Integral
from sympde.expr import Norm
from sympde.expr import Equation, DirichletBC

from spl.fem.basic   import FemField
from spl.fem.vector  import ProductFemSpace, VectorFemField
from spl.api.discretization import discretize

from spl.mapping.discrete import SplineMapping

from numpy import linspace, zeros, allclose

import os

base_dir = os.path.dirname(os.path.realpath(__file__))
mesh_dir = os.path.join(base_dir, 'mesh')

#==============================================================================
def run_poisson_3d_dir(filename, solution, f):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = FunctionSpace('V', domain)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(V, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = f*v
    l = LinearForm(v, expr)

    error = F - solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    equation = Equation(a(v,u), l(v), bc=DirichletBC(domain.boundary))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
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
    # ...

    return l2_error, h1_error

#==============================================================================
def run_poisson_3d_dirneu(filename, solution, f, boundary):

    assert( isinstance(boundary, (list, tuple)) )

    # ... abstract model
    domain = Domain.from_file(filename)

    V = FunctionSpace('V', domain)

    B_neumann = [domain.get_boundary(i) for i in boundary]
    if len(B_neumann) == 1:
        B_neumann = B_neumann[0]

    else:
        B_neumann = Union(*B_neumann)

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(V, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = f*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B_neumann)
    l_B_neumann = LinearForm(v, expr)

    expr = l0(v) + l_B_neumann(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    B_dirichlet = domain.boundary.complement(B_neumann)

    equation = Equation(a(v,u), l(v), bc=DirichletBC(B_dirichlet))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
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
    # ...

    return l2_error, h1_error

#==============================================================================
def run_laplace_3d_neu(filename, solution, f):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = FunctionSpace('V', domain)

    B_neumann = domain.boundary

    x,y,z = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(V, name='u')

    expr = dot(grad(v), grad(u)) + v*u
    a = BilinearForm((v,u), expr)

    expr = f*v
    l0 = LinearForm(v, expr)

    expr = v*trace_1(grad(solution), B_neumann)
    l_B_neumann = LinearForm(v, expr)

    expr = l0(v) + l_B_neumann(v)
    l = LinearForm(v, expr)

    error = F-solution
    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    equation = Equation(a(v,u), l(v))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)
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
    # ...

    return l2_error, h1_error


#==============================================================================
def test_api_poisson_3d_dir_collela():

    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    from sympy.abc import x,y,z

    solution = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f        = 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

    l2_error, h1_error = run_poisson_3d_dir(filename, solution, f)

    expected_l2_error =  0.8151461486397859
    expected_h1_error =  7.887790839303131

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
def test_api_poisson_3d_dirneu_identity_2():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    from sympy.abc import x,y,z

    solution = sin(0.5*pi*x)*sin(pi*y)*sin(pi*z)
    f        = (9./4.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f, ['Gamma_2'])

    expected_l2_error =  0.007476406034615364
    expected_h1_error =  0.20417783668832656

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
def test_api_poisson_3d_dirneu_identity_13():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    from sympy.abc import x,y,z

    solution = cos(0.5*pi*x)*cos(0.5*pi*y)*sin(pi*z)
    f        = (3./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               ['Gamma_1', 'Gamma_3'])

    expected_l2_error =  0.005339281019684631
    expected_h1_error =  0.14517242816351372

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_identity_24():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    from sympy.abc import x,y,z

    solution = sin(0.5*pi*x)*sin(0.5*pi*y)*sin(pi*z)
    f        = (3./2.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               ['Gamma_2', 'Gamma_4'])

    expected_l2_error =  0.005339281019682795
    expected_h1_error =  0.14517242816351233

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_identity_123():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    from sympy.abc import x,y,z

    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*sin(pi*z)
    f        = (21./16.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               ['Gamma_1', 'Gamma_2', 'Gamma_3'])

    expected_l2_error =  0.00681948184967118
    expected_h1_error =  0.18416036905795535

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_identity_1235():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    from sympy.abc import x,y,z

    solution = cos(0.25*pi*x)*cos(0.5*pi*y)*cos(0.5*pi*z)
    f        = (9./16.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               ['Gamma_1', 'Gamma_2', 'Gamma_3', 'Gamma_5'])

    expected_l2_error =  0.0010224893148868808
    expected_h1_error =  0.030546175685500224

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
def test_api_poisson_3d_dirneu_collela_2():
    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    from sympy.abc import x,y,z

    solution = sin(0.25*pi*(x+1.))*sin(pi*y)*sin(pi*z)
    f        = (33./16.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f, ['Gamma_2'])

    expected_l2_error =  0.31651011929794615
    expected_h1_error =  3.315168073716373

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
def test_api_poisson_3d_dirneu_collela_13():
    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    from sympy.abc import x,y,z

    solution = sin(0.25*pi*(1.-x))*sin(0.25*pi*(1.-y))*sin(pi*z)
    f        = (9./8.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               ['Gamma_1', 'Gamma_3'])

    expected_l2_error =  0.1967707543568222
    expected_h1_error =  1.997279767542349

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_collela_24():
    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    from sympy.abc import x,y,z

    solution = sin(0.25*pi*(x+1.))*sin(0.25*pi*(y+1.))*sin(pi*z)
    f        = (9./8.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               ['Gamma_2', 'Gamma_4'])

    expected_l2_error =  0.19713579709344806
    expected_h1_error =  1.9973824048200948

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_collela_123():
    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    from sympy.abc import x,y,z

    solution = cos(pi*x)*sin(0.25*pi*(1.-y))*sin(pi*z)
    f        = (33./16.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               ['Gamma_1', 'Gamma_2', 'Gamma_3'])

    expected_l2_error =  0.6216671144774677
    expected_h1_error =  5.854903143947668

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_poisson_3d_dirneu_collela_1235():
    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    from sympy.abc import x,y,z

    solution = cos(pi*x)*sin(0.25*pi*(1.-y))*sin(0.25*pi*(1.-z))
    f        = (9./8.)*pi**2*solution

    l2_error, h1_error = run_poisson_3d_dirneu(filename, solution, f,
                                               ['Gamma_1', 'Gamma_2', 'Gamma_3', 'Gamma_5'])

    expected_l2_error =  0.6863456565612966
    expected_h1_error =  5.177931817581279

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_laplace_3d_neu_identity():
    filename = os.path.join(mesh_dir, 'identity_3d.h5')

    from sympy.abc import x,y,z

    solution = cos(pi*x)*cos(pi*y)*cos(pi*z)
    f        = (3.*pi**2 + 1.)*solution

    l2_error, h1_error = run_laplace_3d_neu(filename, solution, f)

    expected_l2_error =  0.008820692250536439
    expected_h1_error =  0.24426625779804703

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

#==============================================================================
def test_api_laplace_3d_neu_collela():
    filename = os.path.join(mesh_dir, 'collela_3d.h5')

    from sympy.abc import x,y,z

    solution = cos(pi*x)*cos(pi*y)*cos(pi*z)
    f        = (3.*pi**2 + 1.)*solution

    l2_error, h1_error = run_laplace_3d_neu(filename, solution, f)

    expected_l2_error =  0.918680010922823
    expected_h1_error =  8.85217673379022

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
