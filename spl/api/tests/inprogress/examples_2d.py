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

from mpi4py import MPI

#########################
def plot_field(phi):
    # Plot solution on refined grid
    from matplotlib import pyplot as plt
    import numpy as np

    xx = np.linspace( 0., 1., num=101 )
    yy = np.linspace( 0., 1., num=101 )
    zz = np.array( [[phi( xi,yi ) for yi in yy] for xi in xx] )
    fig, ax = plt.subplots( 1, 1 )
    im = ax.contourf( xx, yy, zz.transpose(), 40, cmap='jet' )
    fig.colorbar( im )
    ax.set_xlabel( r'$x$', rotation='horizontal' )
    ax.set_ylabel( r'$y$', rotation='horizontal' )
    ax.set_title ( r'$\phi(x,y)$' )
    ax.grid()

    # Show figure and keep it open if necessary
    fig.tight_layout()
    fig.show()
    plt.show()
#########################


#==============================================================================
def run_poisson_2d_dir(solution, f, ncells, degree, comm=MPI.COMM_WORLD):

    # ... abstract model
    domain = Square()

    V = FunctionSpace('V', domain)

    x,y = domain.coordinates

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
    domain_h = discretize(domain, ncells=ncells, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
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


#    #######################
#    import numpy as np
#    phi_anal  = lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)
#    integrand = lambda *x: (phi(*x)-phi_anal(*x))**2
#    err2 = np.sqrt( Vh.integral( integrand ) )
#    print('> err2 = ', err2)
#    #######################
#
    return l2_error, h1_error


#==============================================================================
def test_api_poisson_2d_dir_1():

    from sympy.abc import x,y

    solution = sin(pi*x)*sin(pi*y)
    f        = 2*pi**2*sin(pi*x)*sin(pi*y)

    l2_error, h1_error = run_poisson_2d_dir(solution, f,
                                            ncells=[2**3,2**3], degree=[2,2])

#    print(l2_error, h1_error)

    expected_l2_error =  0.00021808678604760232
    expected_h1_error =  0.013023570720360362

    assert( abs(l2_error - expected_l2_error) < 1.e-7)
    assert( abs(h1_error - expected_h1_error) < 1.e-7)

###############################################
if __name__ == '__main__':
    test_api_poisson_2d_dir_1()
