# -*- coding: UTF-8 -*-

from sympy import Tuple, Matrix

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
from psydac.fem.vector         import ProductFemSpace
from psydac.feec.utilities     import Interpolation, interpolation_matrices
from psydac.feec.derivatives         import Grad, Curl, Div
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

def run_system_1_2d_dir(f0, sol, ncells, degree):
    # ... abstract model
    domain = Square()

    H1    = FunctionSpace('V2', domain, kind='H1')
    Hdiv  = VectorFunctionSpace('Hdiv', domain, kind='Hdiv')
    L2    = FunctionSpace('L2', domain, kind='L2')    
    X     = ProductSpace(Hdiv, L2)

    F = ScalarField(L2, name='F')

    p,q = [VectorTestFunction(Hdiv, name=i) for i in ['p', 'q']]
    u,v = [ScalarTestFunction(L2, name=i) for i in ['u', 'v']]

    a  = BilinearForm(((p,u),(q,v)),dot(p,q) + div(q)*u )
    l  = LinearForm((q,v), 2*v)
    
    error = F-sol
    l2norm_F = Norm(error, domain, kind='l2')

    equation = find([p,u], forall=[q,v], lhs=a((p,u),(q,v)), rhs=l(q,v))
 
    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...
    
    # ... discrete spaces
    H1_Vh   = discretize(H1, domain_h, degree=degree)
    Hdiv_Vh = discretize(Hdiv, domain_h, degree=degree)
    L2_Vh   = discretize(L2, domain_h, degree=degree)
    Xh      = discretize(X , domain_h, degree=degree)
    
    # ... dsicretize the equation
    ah = discretize(equation, domain_h, [Xh, Xh], symbolic_space=[X, X])

    ah.assemble()

    M   = ah.linear_system.lhs
    rhs = ah.linear_system.rhs
    
    # ...
    DIV = Div(H1_Vh, Hdiv_Vh.vector_space, L2_Vh.vector_space)
    Int = Interpolation(H1=H1_Vh, Hdiv=Hdiv_Vh, L2=L2_Vh)

    # ...
    M[2,0] = DIV._matrix[0,0]
    M[2,1] = DIV._matrix[0,1]
    
    # ...
    f      = lambda x,y: -2*(2*np.pi)**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    rhs[2] = Int('L2', f)

    # ...
    M   = M.tosparse().tocsc()
    rhs = rhs.toarray()
    
    x   = spsolve(M, rhs)

    # ...
    s31,s32 = Xh.spaces[2].vector_space.starts
    e31,e32 = Xh.spaces[2].vector_space.ends
    
    # ...
    Fh = FemField( Xh.spaces[2], normalize=True )

    Fh.coeffs[s31:e31+1, s32:e32+1] = x[-(e31-s31+1)*(e32-s32+1):].reshape((e31-s31+1, e32-s32+1))
    
    # ...  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x      = np.linspace( 0., 1., 101 )
    y      = np.linspace( 0., 1., 101 )

    phi = np.array( [[Fh(xi, yj) for xi in x] for yj in y] )
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    model = lambda x,y:np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    Z = model(X,Y)
    
    Axes3D.plot_wireframe(ax, X, Y, phi, color='b')
    Axes3D.plot_wireframe(ax, X, Y, Z, color='r')
    
    plt.show()

    

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
            
def test_api_system_1_2d_dir_1():
    from sympy.abc import x,y
    from sympy import sin, cos, pi

    f0 =  -2*(2*pi)**2*sin(2*pi*x)*sin(2*pi*y)
    u  = sin(2*pi*x)*sin(2*pi*y)
    run_system_1_2d_dir(f0,u, ncells=[5,5], degree=[2,2])

