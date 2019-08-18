# -*- coding: UTF-8 -*-

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube, PeriodicDomain
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm, TerminalExpr
from sympde.expr import find, EssentialBC


from psydac.fem.basic   import FemField
from psydac.fem.vector  import VectorFemField
from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil
from psydac.api.essential_bc   import apply_essential_bc
from psydac.linalg.dense       import DenseVectorSpace , DenseVector, DenseMatrix
from psydac.linalg.block       import BlockMatrix

from numpy import linspace, zeros, allclose
import numpy as np
from mpi4py import MPI
import pytest

#from scipy.sparse.linalg import cg, gmres
from scipy.sparse.linalg import spsolve
#from scipy import linalg
from psydac.linalg.iterative_solvers import cg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

def run_system_1_2d_dir(ncells, degree):

               
    # ... abstract model
    domain = Square()
    x,y = domain.coordinates
    
    t  = Constant('t')

    from sympy import pi, cos, sin, exp,Tuple

    Bext  =  sin(3*pi*x)*sin(4*pi*y)*exp(5*t+2)
    Eext  = Tuple(-0.8/pi*sin(3*pi*x)*cos(4*pi*y)*exp(5*t+2)
         ,0.6/pi*cos(3*pi*x)*sin(4*pi*y)*exp(5*t+2))
    
    j =  Tuple((4*pi**2 + 4.0)/pi*exp(5*t+2)*sin(3*pi*x)*cos(4*pi*y), 
               -(3*pi**2 + 3.0)/pi*exp(5*t+2)*sin(4*pi*y)*cos(3*pi*x))


    V1 = VectorFunctionSpace('V1', domain,kind='H1')
    V2 = ScalarFunctionSpace('V2', domain,kind='H1')
    
    X  = V1*V2

    F = element_of(V2, name='F')
    
    error    = F-Bext
    norm_l2  = Norm(error, domain, kind='l2')
    norm_h1  = Norm(error, domain, kind='h1')

    E,P = [element_of(V1, name=i) for i in ['E', 'P']]
    B,v = [element_of(V2, name=i) for i in ['B', 'v']]

    int_0 = lambda expr: integral(domain , expr)
    
    a1  = BilinearForm(((E,B),(P,v)), int_0(dot(E,rot(v)) + dot(rot(B),P)))
    a2  = BilinearForm(((E,B),(P,v)), int_0(dot(E,P) + B*v))

    l1  = LinearForm((P,v), int_0(dot(j,P)))
    l2  = LinearForm((P,v), int_0(dot(P, Eext) + v*Bext))
    
 
    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    V1h = discretize(V1, domain_h, degree=degree)
    V2h = discretize(V2, domain_h, degree=degree)
    Xh  = V1h*V2h

    # ... dsicretize the equation using Dirichlet bc
    a1_h = discretize(a1, domain_h, [Xh, Xh])
    a2_h = discretize(a2, domain_h, [Xh, Xh])
    
    l1_h = discretize(l1, domain_h, Xh)
    l2_h = discretize(l2, domain_h, Xh)
    
    norm_l2h = discretize(norm_l2, domain_h, V2h)

    # ...
    
    # ...
    
    t = 0
    dt = 0.00001
    n  = 0
    
    M    = a2_h.assemble()
    S    = a1_h.assemble()
    u0   = l2_h.assemble(t=0)
    
    for i in range(3):
        u0[i]._data[2,:] = 0
        u0[i]._data[-2,:]= 0
        u0[i]._data[:,2] = 0
        u0[i]._data[:,-2]= 0
        M[i,i]._data[2,:,:,:] = 0
        M[i,i]._data[-2,:,:,:] = 0
        M[i,i]._data[:,2,:,:] = 0
        M[i,i]._data[:,-2,:,:] = 0


    u ,info = cg( M, u0, tol=1e-10)
    
    phi = FemField(V2h, u[2])
    error_l2 = norm_l2h.assemble(F=phi, t=t)
#    print(error_l2)
    
    for i in range(n):
        t +=dt
        jn = l1_h.assemble(t=t)
        u  = M.dot(u) - dt*S.dot(u) + dt*jn
        for j in range(3):
            u[j]._data[2,:] = 0
            u[j]._data[-2,:]= 0
            u[j]._data[:,2] = 0
            u[j]._data[:,-2]= 0
        u ,info = cg( M, u, tol=1e-10)
    

        phi = FemField(V2h, u[2])
        error_l2 = norm_l2h.assemble(F=phi, t=t)
#        print(error_l2)
    
    x = np.linspace(0., 1., 101)
    y = np.linspace(0., 1., 101)
    
    phi = np.array([[phi(xi,yj) for xi in x] for yj in y])

    X, Y = np.meshgrid(x, y, indexing='ij')

    model = lambda x,y:np.sin(3*np.pi*x)*np.sin(4*np.pi*y)*np.exp(5*t+2)
    Z = model(X,Y)
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')

#    Axes3D.plot_wireframe(ax, X, Y, phi, color='b')
#    Axes3D.plot_wireframe(ax, X, Y, Z, color='r')

#    plt.show()
    
    return error_l2

def run_system_1_2d_per(ncells, degree):

               
    # ... abstract model
    domain = Square()
    x,y = domain.coordinates
    
    domain = PeriodicDomain(domain, [True, True])
    
    t  = Constant('t')

    from sympy import pi, cos, sin, exp, Tuple

    Bext  =  sin(3*pi*x)*sin(4*pi*y)*exp(5*t+2)
    Eext  = Tuple(-0.8/pi*sin(3*pi*x)*cos(4*pi*y)*exp(5*t+2)
         ,0.6/pi*cos(3*pi*x)*sin(4*pi*y)*exp(5*t+2))
    
    j =  Tuple((4*pi**2 + 4.0)/pi*exp(5*t+2)*sin(3*pi*x)*cos(4*pi*y), 
               -(3*pi**2 + 3.0)/pi*exp(5*t+2)*sin(4*pi*y)*cos(3*pi*x))


    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = ScalarFunctionSpace('V2', domain, kind='H1')
    
    X  = V1*V2

    F = element_of(V2, name='F')
    
    error    = F-Bext
    norm_l2  = Norm(error, domain, kind='l2')
    norm_h1  = Norm(error, domain, kind='h1')

    E,P = [element_of(V1, name=i) for i in ['E', 'P']]
    B,v = [element_of(V2, name=i) for i in ['B', 'v']]

    int_0 = lambda expr: integral(domain , expr)
    
    a1  = BilinearForm(((E,B),(P,v)), int_0(dot(E,rot(v)) + dot(rot(B),P)))
    a2  = BilinearForm(((E,B),(P,v)), int_0(dot(E,P) + B*v))

    l1  = LinearForm((P,v), int_0(dot(j,P)))
    l2  = LinearForm((P,v), int_0(dot(P, Eext) + v*Bext))
    l3  = LinearForm((P,v), int_0(v))
    l4  = LinearForm((P,v), int_0(P))
    
 
    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    V1h = discretize(V1, domain_h, degree=degree)
    V2h = discretize(V2, domain_h, degree=degree)
    Xh  = V1h*V2h

    # ... dsicretize the equation using Dirichlet bc
    a1_h = discretize(a1, domain_h, [Xh, Xh])
    a2_h = discretize(a2, domain_h, [Xh, Xh])
    
    l1_h = discretize(l1, domain_h, Xh)
    l2_h = discretize(l2, domain_h, Xh)
    l3_h = discretize(l3, domain_h, Xh)
    l4_h = discretize(l4, domain_h, Xh)
    
    norm_l2h = discretize(norm_l2, domain_h, V2h)
    
    # ...
    
    t = 0
    dt = 0.00001
    n  = 10
    
    M    = a2_h.assemble()
    S    = a1_h.assemble()
    u0   = l2_h.assemble(t=0)
    
    l3 = l3_h.assemble()
    l4 = l4_h.assemble()
    
    V   = DenseVectorSpace(1)

    pb11 = DenseMatrix(V, Xh.spaces[0].vector_space)
    pb12 = DenseMatrix(V, Xh.spaces[1].vector_space)
    pb13 = DenseMatrix(V, Xh.spaces[2].vector_space)
    
    pb21 = DenseMatrix(Xh.spaces[0].vector_space, V)
    pb22 = DenseMatrix(Xh.spaces[1].vector_space, V)
    pb23 = DenseMatrix(Xh.spaces[2].vector_space, V)

    pb11._data[0] = l4[0]
    pb21._data[0] = l4[0]
    pb12._data[0] = l4[1]
    pb22._data[0] = l4[1]
    pb13._data[0] = l3[2]
    pb23._data[0] = l3[2]
    
    M2 = BlockMatrix(Xh*V, Xh*V)
    S2 = BlockMatrix(Xh*V, Xh*V)
    
    for i in range(3):
        M2[i,i] = M[i,i]
        
    M2[0,4] = pb11
    M2[1,4] = pb12
    M2[2,4] = pb13
    M2[4,0] = pb21
    M2[4,1] = pb22
    M2[4,2] = pb23
    
 
    u ,info = cg( M, u0, tol=1e-10)
    
    phi = FemField(V2h, u[2])
    error_l2 = norm_l2h.assemble(F=phi, t=t)
#    print(error_l2)
    
    for i in range(n):
        t +=dt
        jn = l1_h.assemble(t=t)
        u  = M.dot(u) - dt*S.dot(u) + dt*jn

        u ,info = cg( M, u, tol=1e-10)
    

        phi = FemField(V2h, u[2])
        error_l2 = norm_l2h.assemble(F=phi, t=t)
#        print(error_l2)
    
    x = np.linspace(0., 1., 101)
    y = np.linspace(0., 1., 101)
    
    phi = np.array([[phi(xi,yj) for xi in x] for yj in y])

    X, Y = np.meshgrid(x, y, indexing='ij')

    model = lambda x,y:np.sin(3*np.pi*x)*np.sin(4*np.pi*y)*np.exp(5*t+2)
    Z = model(X,Y)
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')

#    Axes3D.plot_wireframe(ax, X, Y, phi, color='b')
#    Axes3D.plot_wireframe(ax, X, Y, Z, color='r')

#    plt.show()
    
    return error_l2

def test_api_system_1_2d_dir_1():
    error_l2 = run_system_1_2d_dir(ncells=[2**3, 2**3], degree=[2,2])

@pytest.mark.xfail
def test_api_system_1_2d_per_1():
    error_l2 = run_system_1_2d_per(ncells=[2**5, 2**5], degree=[2,2])
    
    
#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()

