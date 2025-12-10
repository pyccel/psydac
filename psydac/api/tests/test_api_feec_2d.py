# -*- coding: UTF-8 -*-

from mpi4py import MPI
from scipy.sparse.linalg import spsolve
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from sympde.calculus import dot, div
from sympde.topology import Derham
from sympde.topology import ProductSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

#==============================================================================

def run_system_1_2d_dir(f0, sol, ncells, degree):
    # ... abstract model
    domain = Square()
    
    derham = Derham(domain, sequence=['H1', 'Hdiv', 'L2'])
    H1     = derham.V0
    Hdiv   = derham.V1
    L2     = derham.V2  
    X      = ProductSpace(Hdiv, L2)

    p, q = elements_of(Hdiv, names='p, q')
    u, v = elements_of(  L2, names='u, v')

    F = element_of(L2, name='F')

    int_0 = lambda expr: integral(domain , expr)

    a  = BilinearForm(((p,u),(q,v)), int_0(dot(p,q) + div(q)*u) )
    l  = LinearForm((q,v), int_0(2*v))
    
    error  = F-sol
    l2norm = Norm(error, domain, kind='l2')

    equation = find([p,u], forall=[q,v], lhs=a((p,u),(q,v)), rhs=l(q,v))
 
    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...
    
    # ... discrete spaces
    derham_Vh = discretize(derham, domain_h, degree=degree)
    Xh        = discretize(X , domain_h, degree=degree)
    
    Hdiv_Vh = derham_Vh.V1
    L2_Vh   = derham_Vh.V2
    
    
    # ... dsicretize the equation
    ah       = discretize(equation, domain_h, [Xh, Xh], symbolic_space=[X, X])
    l2norm_h = discretize(l2norm, domain_h, L2_Vh)

    ah.assemble()
    
    M   = ah.linear_system.lhs
    rhs = ah.linear_system.rhs
    # ...
    M[2,0] = Hdiv_Vh.div._matrix[0,0]
    M[2,1] = Hdiv_Vh.div._matrix[0,1]
    
    # ...
    f      = lambda x,y: -2*(2*np.pi)**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    rhs[2] = L2_Vh.interpolate(f)

    # ...
    M   = M.tosparse().tocsc()
    rhs = rhs.toarray()
    
    x   = spsolve(M, rhs)

    # ...
    s31,s32 = L2_Vh.vector_space.starts
    e31,e32 = L2_Vh.vector_space.ends
    
    u = x[-(e31-s31+1)*(e32-s32+1):].reshape((e31-s31+1, e32-s32+1))
    
    # ...
    Fh = FemField( L2_Vh )

    Fh.coeffs[s31:e31+1, s32:e32+1] = u
    

    # ...  
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')

    x      = np.linspace( 0., 1., 101 )
    y      = np.linspace( 0., 1., 101 )

    phi = np.array( [[Fh(xi, yj) for xi in x] for yj in y] )
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    model = lambda x,y:np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    Z = model(X,Y)
    
    error = l2norm_h.assemble(F=Fh)
    # TODO fix bug it gives the wrong error
    
    error = np.abs(Z-phi).max()
     
#    Axes3D.plot_wireframe(ax, X, Y, phi, color='b')
#    Axes3D.plot_wireframe(ax, X, Y, Z, color='r')
    
#    plt.show()
    
    return error

    

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
            
def test_api_system_1_2d_dir_1():
    from sympy.abc import x,y
    from sympy import sin, cos, pi

    f0 =  -2*(2*pi)**2*sin(2*pi*x)*sin(2*pi*y)
    u  = sin(2*pi*x)*sin(2*pi*y)

    error = run_system_1_2d_dir(f0,u, ncells=[5, 5], degree=[2,2])

