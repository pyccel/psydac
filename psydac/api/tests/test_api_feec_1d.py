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

def run_system_1_1d_dir(f0, sol, ncells, degree):
    # ... abstract model
    domain = Line()

    derham = Derham(domain)
    
    V0, V1 = derham.spaces

    F = element_of_space(V1, name='F')

    p,q = [element_of_space(V0, name=i) for i in ['p', 'q']]
    u,v = [element_of_space(V1, name=i) for i in ['u', 'v']]

    a  = BilinearForm(((p,u),(q,)), dot(p,q) + div(q)*u )

    error  = F-sol
    l2norm = Norm(error, domain, kind='l2')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    derham_h = discretize(derham, domain_h, degree=degree)

    V0_h = derham_h.V0
    V1_h = derham_h.V1
    Xh   = V0_h * V1_h

    V0_h, V1_h = derham_h.spaces
    GRAD       = derham_h.derivatives_as_matrices
    # ...

    ah       = discretize(a, domain_h, [Xh, V0_h])
    l2norm_h = discretize(l2norm, domain_h, V1_h)
    # ...

    # ...
    M   = ah.assemble()
    rhs = BlockVector(Xh.vector_space)
    # ...

    # ...
    blocks  = [list(block) for block in M.blocks]
    blocks += [[None, None]]

    blocks[1][0] = GRAD[0,0]
    
    M = BlockMatrix(Xh.vector_space, Xh.vector_space, blocks=blocks)
     
    # ...
    rhs[1] = V1_h.interpolate(f0)

    # ...
    M   = M.tosparse().tocsc()
    rhs = rhs.toarray()

    x   = spsolve(M, rhs)
    
    u = array_to_stencil(x, Xh.vector_space)

    # ...
    Fh = FemField( V1_h )

    Fh.coeffs[:] = u[1][:]

    # ...
#    fig,ax = plt.subplots( 1, 1 )

    x      = np.linspace( 0., 1., 101 )

    phi = np.array( [Fh(xi) for xi in x] )
    #TODO fig bug calculate the right field

    model = lambda x:np.sin(2*np.pi*x)
    y = model(x)

    error = l2norm_h.assemble(F=Fh)
    # TODO fix bug it gives the wrong error

    error = np.abs(y-phi).max()

#    ax.plot( x, phi )
#    ax.plot( x, y )

#    plt.show()

    return error



###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================

def test_api_system_1_2d_dir_1():
    from sympy.abc import x
    import sympy as sp

    f0 = lambda x: -(2*np.pi)**2*np.sin(2*np.pi*x)
    u  = sp.sin(2*sp.pi*x)

    error = run_system_1_1d_dir(f0,u, ncells=[10], degree=[2])

