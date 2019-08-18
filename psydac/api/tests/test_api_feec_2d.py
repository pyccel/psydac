# -*- coding: UTF-8 -*-

from sympy import Tuple, Matrix

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)

from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace, Derham
from sympde.topology import ScalarField, VectorField
from sympde.topology import element_of
from sympde.topology import ProductSpace
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm, TerminalExpr
from sympde.expr import find, EssentialBC


from psydac.fem.basic   import FemField
from psydac.fem.vector  import VectorFemField
from psydac.api.discretization import discretize
from psydac.fem.vector         import ProductFemSpace
from psydac.linalg.block       import BlockVector, BlockMatrix
from psydac.linalg.utilities   import array_to_stencil

from scipy.sparse.linalg import cg, gmres
from scipy.sparse.linalg import spsolve
from scipy import linalg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import numpy as np
from sympy import lambdify
#==============================================================================

def run_system_1_2d_dir(f0, sol, ncells, degree):
    # ... abstract model
    domain = Square()

    derham = Derham(domain, sequence=['H1', 'Hdiv', 'L2'])
    V0 = derham.V0
    V1 = derham.V1
    V2 = derham.V2

    V0, V1, V2 = derham.spaces

    F = element_of(V2, name='F')

    p,q = [element_of(V1, name=i) for i in ['p', 'q']]
    u,v = [element_of(V2, name=i) for i in ['u', 'v']]

    int_0 = lambda expr: integral(domain , expr)
    
    a  = BilinearForm(((p,u),(q,)), int_0(dot(p,q) + div(q)*u) )


#    a  = BilinearForm(((p,u),(q,v)), int_0(dot(p,q) + div(q)*u) )
    l  = LinearForm((q,v), int_0(2*v))
    
    error  = F-sol
    l2norm = Norm(error, domain, kind='l2')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    derham_h = discretize(derham, domain_h, degree=degree)

    V1_h = derham_h.V1
    V2_h = derham_h.V2
    Xh   = V1_h * V2_h

    V0_h, V1_h, V2_h = derham_h.spaces
    GRAD, DIV        = derham_h.derivatives_as_matrices
    # ...

    ah       = discretize(a, domain_h, [Xh, V1_h])
    l2norm_h = discretize(l2norm, domain_h, V2_h)
    # ...

    # ...
    M   = ah.assemble()
    rhs = BlockVector(Xh.vector_space)
    # ...

    # ...
    blocks  = [list(block) for block in M.blocks]
    blocks += [[None, None, None]]

    blocks[2][0] = DIV[0,0]
    blocks[2][1] = DIV[0,1]
    
    M = BlockMatrix(Xh.vector_space, Xh.vector_space, blocks=blocks)
     
    # ...
    rhs[2] = V2_h.interpolate(f0)

    # ...
    M   = M.tosparse().tocsc()
    rhs = rhs.toarray()

    x   = spsolve(M, rhs)
    
    u = array_to_stencil(x, Xh.vector_space)

    # ...
    Fh = FemField( V2_h )

    Fh.coeffs[:,:] = u[2][:,:]

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

    f0 = lambda x,y: -2*(2*np.pi)**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    u  = sin(2*pi*x)*sin(2*pi*y)

    error = run_system_1_2d_dir(f0,u, ncells=[5, 5], degree=[2,2])

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
