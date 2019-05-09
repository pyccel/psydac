# -*- coding: UTF-8 -*-

from sympy import Tuple, Matrix

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace, Derham
from sympde.topology import element_of_space
from sympde.topology import ProductSpace
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.expr import BilinearForm, LinearForm
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

def run_system_1_3d_dir(f0, sol, ncells, degree):
    # ... abstract model
    domain = Cube()

    derham = Derham(domain)

    V0, V1, V2, V3 = derham.spaces

    F = element_of_space(V3, name='F')

    p,q = [element_of_space(V2, name=i) for i in ['p', 'q']]
    u,v = [element_of_space(V3, name=i) for i in ['u', 'v']]

    a  = BilinearForm(((p,u),(q,)), dot(p,q) + div(q)*u )

    error  = F-sol
    l2norm = Norm(error, domain, kind='l2')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    derham_h = discretize(derham, domain_h, degree=degree)

    V2_h = derham_h.V2
    V3_h = derham_h.V3
    Xh   = V2_h * V3_h

    V0_h, V1_h, V2_h, V3_h = derham_h.spaces
    GRAD, CURL, DIV        = derham_h.derivatives_as_matrices
    # ...
    ah       = discretize(a, domain_h, [Xh, V2_h])
    l2norm_h = discretize(l2norm, domain_h, V3_h)
    # ...

    # ...
    M   = ah.assemble()
    rhs = BlockVector(Xh.vector_space)
    # ...

    # ...
    blocks  = [list(block) for block in M.blocks]
    blocks += [[None, None, None, None]]

    blocks[3][0] = DIV[0,0]
    blocks[3][1] = DIV[0,1]
    blocks[3][2] = DIV[0,2]
    
    M = BlockMatrix(Xh.vector_space, Xh.vector_space, blocks=blocks)
     
    # ...
    rhs[3] = V3_h.interpolate(f0)

    # ...
    M   = M.tosparse().tocsc()
    rhs = rhs.toarray()

    x   = spsolve(M, rhs)
    
    u = array_to_stencil(x, Xh.vector_space)

    # ...
    Fh = FemField( V3_h )

    Fh.coeffs[:,:,:] = u[3][:,:,:]

    # ...
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')

#    x      = np.linspace( 0., 1., 101 )
#    y      = np.linspace( 0., 1., 101 )
#    z      = np.linspace( 0., 1., 101 )
    
#    phi = np.array( [[[Fh(xi, yj, zk) for xi in x] for yj in y] for zk in z])

#    X, Y = np.meshgrid(x, y, z, indexing='ij')

#    model = lambda x,y,z:np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*np.sin(2*np.pi*z)
#    R = model(X, Y, Z)

    error = l2norm_h.assemble(F=Fh)
    # TODO fix bug it gives the wrong error

#    error = np.abs(Z-phi).max()

#    Axes3D.plot_wireframe(ax, X, Y, phi, color='b')
#    Axes3D.plot_wireframe(ax, X, Y, Z, color='r')

#    plt.show()

    return error



###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================

def test_api_system_1_3d_dir_1():
    from sympy.abc import x,y,z
    from sympy import sin, cos, pi

    f0 = lambda x,y,z: -3*(2*np.pi)**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*np.sin(2*np.pi*z)
    u  = sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)

    error = run_system_1_3d_dir(f0,u, ncells=[5, 5, 5], degree=[2, 2, 2])

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()
