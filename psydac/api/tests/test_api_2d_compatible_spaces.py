# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin, Tuple, Matrix

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import ScalarField, VectorField
from sympde.topology import ProductSpace
from sympde.topology import ScalarTestFunction
from sympde.topology import VectorTestFunction
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm, TerminalExpr
from sympde.expr import find, EssentialBC


from psydac.fem.basic   import FemField
from psydac.fem.vector  import VectorFemField
from psydac.api.discretization import discretize

from numpy import linspace, zeros, allclose
import numpy as np
from mpi4py import MPI
import pytest

from scipy.sparse.linalg import cg, gmres
from scipy.sparse.linalg import spsolve
from scipy import linalg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
#==============================================================================

def run_system_1_2d_dir(f0, u, ncells, degree):
    # ... abstract model
    domain = Square()

    V1 = VectorFunctionSpace('V1', domain, kind='Hdiv')
    V2 = FunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    x,y = domain.coordinates
    
    F = ScalarField(V2, name='F')

    
    p,q = [VectorTestFunction(V1, name=i) for i in ['p', 'q']]
    u,v = [ScalarTestFunction(V2, name=i) for i in ['u', 'v']]

    a  = BilinearForm(((p,u),(q,v)),dot(p,q) + div(q)*u + div(p)*v )
    l  = LinearForm((q,v), f0*v)

    error = F-u
    l2norm_F = Norm(error, domain, kind='l2')
    h1norm_F = Norm(error, domain, kind='h1')

    equation = find([p,u], forall=[q,v], lhs=a((p,u),(q,v)), rhs=l(q,v))
 
    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...
    # ... discrete spaces
    V1h = discretize(V1, domain_h, degree=degree)
    V2h = discretize(V2, domain_h, degree=degree)
    Xh  = discretize(X , domain_h, degree=degree)

    # ... dsicretize the equation using Dirichlet bc
    ah = discretize(equation, domain_h, [Xh, Xh], symbolic_space=[X, X])
    # ...
    
    # ... discretize norms
    l2norm_F_h = discretize(l2norm_F, domain_h, V2h)
    h1norm_F_h = discretize(h1norm_F, domain_h, V2h)

    ah.assemble()

    M   = ah.linear_system.lhs.tosparse()
    rhs = ah.linear_system.rhs.toarray()
    x = spsolve(M, rhs)
    
    
    s31,s32 = V2h.vector_space.starts
    e31,e32 = V2h.vector_space.ends
    # ...
    Fh = FemField( V2h )

    Fh.coeffs[:,:] = x[-(e31-s31+1)*(e32-s32+1):].reshape((e31-s31+1, e32-s32+1))
    # ...
    # ... compute norms
    l2_error = l2norm_F_h.assemble(F=Fh)
    h1_error = h1norm_F_h.assemble(F=Fh)

    return l2_error, h1_error
    
def run_system_2_2d_dir(f1, f2,u1, u2, ncells, degree):
    # ... abstract model
    domain = Square()

    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = FunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    x,y = domain.coordinates

    F = VectorField(W, name='F')
    
    u,v = [VectorTestFunction(V1, name=i) for i in ['u', 'v']]
    p,q = [ScalarTestFunction(V2, name=i) for i in ['p', 'q']]

    a  = BilinearForm(((u,p),(v,q)),inner(grad(u),grad(v)) + div(u)*q - p*div(v) )
    l  = LinearForm((v,q), f1*v[0]+f2*v[1]+q)
    
    bc = EssentialBC(u, 0, domain.boundary)
    equation = find([u,p], forall=[v,q], lhs=a((u,p),(v,q)), rhs=l(v,q), bc=bc)

    error = Matrix([F[0]-u1, F[1]-u2])
    l2norm_F = Norm(error, domain, kind='l2')
    h1norm_F = Norm(error, domain, kind='h1')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ... discrete spaces
    V1h = discretize(V1, domain_h, degree=degree)
    V2h = discretize(V2, domain_h, degree=degree)
    Xh  = discretize(X , domain_h, degree=degree)
    
    s11,s12 = V1h.spaces[0].vector_space.starts
    e11,e12 = V1h.spaces[0].vector_space.ends
    s21,s22 = V1h.spaces[1].vector_space.starts
    e21,e22 = V1h.spaces[1].vector_space.ends
    s31,s32 = V2h.vector_space.starts
    e31,e32 = V2h.vector_space.ends
    
    # ... dsicretize the equation using Dirichlet bc
    ah = discretize(equation, domain_h, [Xh, Xh], symbolic_space=[X, X])
    
    # ... discretize norms
    l2norm_F_h = discretize(l2norm_F, domain_h, V1h)
    h1norm_F_h = discretize(h1norm_F, domain_h, V1h)

    ah.assemble()
    
    M     = ah.linear_system.lhs
    M[0,0][0,:,0,0] = 1.
    M[0,0][:,0,0,0] = 1.
    M[0,0][e11,:,0,0] = 1.
    M[0,0][:,e12,0,0] = 1.
    M[1,1][0,:,0,0] = 1.
    M[1,1][:,0,0,0] = 1.
    M[1,1][e21,:,0,0] = 1.
    M[1,1][:,e22,0,0] = 1.
    M = M.toarray()
    
    rhs   = ah.linear_system.rhs.toarray()

    M_1   = np.zeros((len(rhs)+1,len(rhs)+1))
    rhs_1 = np.zeros(len(rhs)+1)
    v2h_s = (e11+1)*(e12+1)+(e21+1)*(e22+1)

    M_1[:-1,:-1] = M
    M_1[-1,v2h_s:-1]  = rhs[v2h_s:]
    M_1[v2h_s:-1,-1]  = rhs[v2h_s:]
    rhs_1[:v2h_s]   = rhs[:v2h_s] 

    M_inv = linalg.inv(M_1)
    x = M_inv.dot(rhs_1)
    
    phi1 = VectorFemField(V1h)
    phi2 = FemField(V2h)

    
    phi1[0].coeffs[s11:e11+1, s12:e12+1] = x[:(e11+1)*(e12+1)].reshape((e11+1-s11, e12+1-s12))
    phi1[1].coeffs[s21:e21+1, s22:e22+1] = x[(e11+1)*(e12+1):v2h_s].reshape((e21+1-s21, e22+1-s22))
    phi2.coeffs[s31:e31+1, s32:e32+1]    = x[v2h_s:-1].reshape((e31-s31+1, e32-s32+1))
    
        # ... compute norms
    l2_error = l2norm_F_h.assemble(F=phi1)
    h1_error = h1norm_F_h.assemble(F=phi1)
    return l2_error, h1_error

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_system_2_2d_dir_1():
    from sympy.abc import x,y
    from sympy import cos
    
    f1 = -x**2*(x - 1)**2*(24*y - 12) - 4*y*(x**2 + 4*x*(x - 1) + (x - 1)**2)*(2*y**2 - 3*y + 1) - 2*pi*cos(2*pi*x)
    f2 = 4*x*(2*x**2 - 3*x + 1)*(y**2 + 4*y*(y - 1) + (y - 1)**2) + y**2*(24*x - 12)*(y - 1)**2 + 2*pi*cos(2*pi*y)
    u1 = x**2*(-x + 1)**2*(4*y**3 - 6*y**2 + 2*y)
    u2 =-y**2*(-y + 1)**2*(4*x**3 - 6*x**2 + 2*x)
    p  = sin(2*pi*x) - sin(2*pi*y)
    
    x = run_system_2_2d_dir(f1, f2, ncells=[2**3,2**3], degree=[2,2])
            
def test_api_system_1_2d_dir_1():
    from sympy.abc import x,y

    f0 =  -2*(2*pi)**2*sin(2*pi*x)*sin(2*pi*y)
    u  = sin(2*pi*x)*sin(2*pi*y)
    x = run_system_1_2d_dir(f0,u, ncells=[10,10], degree=[2,2])

test_api_system_1_2d_dir_1()
