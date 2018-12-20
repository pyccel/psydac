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
mesh_dir = os.path.join(base_dir, '..', 'mesh')



#==============================================================================
def assert_identical_coo(A, B):

    if isinstance(A, (list, tuple)) and isinstance(B, (list, tuple)):
        assert len(A) == len(B)

        for a,b in zip(A, B): assert_identical_coo(a, b)

    elif not(isinstance(A, (list, tuple))) and not(isinstance(B, (list, tuple))):
        A = A.tocoo()
        B = B.tocoo()

        assert(A.shape == B.shape)
        assert(A.nnz == B.nnz)

        assert(allclose(A.row,  B.row))
        assert(allclose(A.col,  B.col))
        assert(allclose(A.data, B.data))

    else:
        raise TypeError('Wrong types for entries')




#==============================================================================
def test_api_poisson_2d_dir_identity():

    # filename where the geometry is defined
    filename = 'identity_2d.h5'

    # ... abstract model
    domain = Domain.from_file(filename)

    V = FunctionSpace('V', domain)

    x,y = domain.coordinates

    F = Field('F', V)

    v = TestFunction(V, name='v')
    u = TestFunction(V, name='u')

    expr = dot(grad(v), grad(u))
    a = BilinearForm((v,u), expr)

    expr = 2*pi**2*sin(pi*x)*sin(pi*y)*v
    l = LinearForm(v, expr)

    error = F-sin(pi*x)*sin(pi*y)
    l2norm = Norm(error, domain, kind='l2', name='u')
    h1norm = Norm(error, domain, kind='h1', name='u')

    equation = Equation(a(v,u), l(v), bc=DirichletBC(domain.boundary))
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    equation_h = discretize(equation, [Vh, Vh], domain_h)
    # ...

    # ... discretize norms
    l2norm_h = discretize(l2norm, Vh, domain_h)
    h1norm_h = discretize(h1norm, Vh, domain_h)
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

    print('> l2_error = ', l2_error)
    print('> h1_error = ', h1_error)

#    expected_l2_error =  0.0006542603581247817
#    expected_h1_error =  0.039070712161073926
#
#    assert( abs(l2_error - expected_l2_error) < 1.e-7)
#    assert( abs(h1_error - expected_h1_error) < 1.e-7)
    # ...





###############################################
if __name__ == '__main__':
    test_api_poisson_2d_dir_identity()
