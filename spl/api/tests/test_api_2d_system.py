# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin, Tuple, Matrix

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import Field, VectorField
from sympde.topology import ProductSpace
from sympde.topology import TestFunction
from sympde.topology import VectorTestFunction
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from spl.fem.basic   import FemField
from spl.fem.vector   import VectorFemField
from spl.api.discretization import discretize

from numpy import linspace, zeros, allclose
import numpy as np
from mpi4py import MPI
import pytest

#==============================================================================
def run_system_1_2d_dir(Fe, Ge, f0, f1, ncells, degree):

    # ... abstract model
    domain = Square()

    W = VectorFunctionSpace('W', domain)
    V = FunctionSpace('V', domain)
    X = ProductSpace(W, V)

    x,y = domain.coordinates

    F = VectorField(W, name='F')
    G = Field(V, name='G')

    u,v = [VectorTestFunction(W, name=i) for i in ['u', 'v']]
    p,q = [      TestFunction(V, name=i) for i in ['p', 'q']]

    a0 = BilinearForm((v,u), inner(grad(v), grad(u)))
    a1 = BilinearForm((q,p), p*q)
    a  = BilinearForm(((v,q),(u,p)), a0(v,u) + a1(q,p))

    l0 = LinearForm(v, dot(f0, v))
    l1 = LinearForm(q, f1*q)
    l  = LinearForm((v,q), l0(v) + l1(q))

    error = Matrix([F[0]-Fe[0], F[1]-Fe[1]])
    l2norm_F = Norm(error, domain, kind='l2')
    h1norm_F = Norm(error, domain, kind='h1')

    error = G-Ge
    l2norm_G = Norm(error, domain, kind='l2')
    h1norm_G = Norm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find([u,p], forall=[v,q], lhs=a((u,p),(v,q)), rhs=l(v,q), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    Wh = discretize(W, domain_h, degree=degree)
    Xh = discretize(X, domain_h, degree=degree)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Xh, Xh])
    # ...

    # ... discretize norms
    l2norm_F_h = discretize(l2norm_F, domain_h, Wh)
    h1norm_F_h = discretize(h1norm_F, domain_h, Wh)

    l2norm_G_h = discretize(l2norm_G, domain_h, Vh)
    h1norm_G_h = discretize(h1norm_G, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    x = equation_h.solve()
    # ...

    # ...
    Fh = VectorFemField( Wh )
    Fh.coeffs[0][:,:] = x[0][:,:]
    Fh.coeffs[1][:,:] = x[1][:,:]
    # ...

    # ...
    Gh = FemField( Vh )
    Gh.coeffs[:,:] = x[2][:,:]
    # ...

    # ... compute norms
    l2_error_F = l2norm_F_h.assemble(F=Fh)
    h1_error_F = h1norm_F_h.assemble(F=Fh)

    l2_error_G = l2norm_G_h.assemble(G=Gh)
    h1_error_G = h1norm_G_h.assemble(G=Gh)
    # ...

    l2_error = np.asarray([l2_error_F, l2_error_G])
    h1_error = np.asarray([h1_error_F, h1_error_G])

    return l2_error, h1_error

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_system_1_2d_dir_1():

    from sympy.abc import x,y

    Fe = Tuple(sin(pi*x)*sin(pi*y), sin(pi*x)*sin(pi*y))
    f0 = Tuple(2*pi**2*sin(pi*x)*sin(pi*y),
              2*pi**2*sin(pi*x)*sin(pi*y))

    Ge = cos(pi*x)*cos(pi*y)
    f1 = cos(pi*x)*cos(pi*y)

    l2_error, h1_error = run_system_1_2d_dir(Fe, Ge, f0, f1,
                                            ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  np.asarray([0.00030842129059875065,
                                     0.0002164796555228256])
    expected_h1_error =  np.asarray([0.018418110343264293,
                                     0.012987988507232278])

    assert( np.allclose(l2_error, expected_l2_error, 1.e-7) )
    assert( np.allclose(h1_error, expected_h1_error, 1.e-7) )
