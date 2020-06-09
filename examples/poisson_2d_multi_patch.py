import pytest      
                
from sympy.core.containers import Tuple
from sympy                 import Matrix               
from sympy                 import Function                                
from sympy                 import pi, cos, sin, exp                        
      
from sympde.core     import Constant
from sympde.calculus import grad, dot, inner, rot, div
from sympde.calculus import laplace, bracket, convect
from sympde.calculus import jump, avg, Dn, minus, plus

from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import InteriorDomain, Union
from sympde.topology import Boundary, NormalVector
from sympde.topology import Domain
from sympde.topology import trace_1
from sympde.topology import Square
from sympde.topology import ElementDomain
from sympde.topology import Area
from sympde.topology import IdentityMapping, PolarMapping, AffineMapping
                         
from sympde.expr.expr          import LinearExpr
from sympde.expr.expr          import LinearForm, BilinearForm
from sympde.expr.expr          import integral              
from sympde.expr.expr          import Functional, Norm                       
from sympde.expr.expr          import linearize                      
from sympde.expr.evaluation    import TerminalExpr
from sympde.expr               import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.fem.vector         import VectorFemField
from psydac.fem.basic          import FemField
from psydac.utilities.utils    import refine_array_1d

import matplotlib.pyplot as plt
import numpy as np

#==============================================================================

def run_poisson_2d(solution, f, domain, mappings, ncells, degree, comm=None):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    V   = ScalarFunctionSpace('V', domain, kind=None)

    u, v = elements_of(V, names='u, v')
    nn   = NormalVector('nn')

    bc   = EssentialBC(u, solution, domain.boundary)

    error  = u - solution

    I = domain.interfaces

    kappa  = 10**9

   # expr_I =(
   #         - dot(grad(plus(u)),nn)*minus(v)  + dot(grad(minus(v)),nn)*plus(u) - kappa*plus(u)*minus(v)
   #         + dot(grad(minus(u)),nn)*plus(v)  - dot(grad(plus(v)),nn)*minus(u) - kappa*plus(v)*minus(u)
   #         - dot(grad(plus(v)),nn)*plus(u)   + kappa*plus(u)*plus(v)
   #         - dot(grad(minus(v)),nn)*minus(u) + kappa*minus(u)*minus(v))

    expr_I =- 0.5*dot(grad(plus(u)),nn)*minus(v)  + 0.5*dot(grad(minus(v)),nn)*plus(u)  - kappa*plus(u)*minus(v)\
            + 0.5*dot(grad(minus(u)),nn)*plus(v)  - 0.5*dot(grad(plus(v)),nn)*minus(u)  - kappa*plus(v)*minus(u)\
            - 0.5*dot(grad(minus(v)),nn)*minus(u) - 0.5*dot(grad(minus(u)),nn)*minus(v) + kappa*minus(u)*minus(v)\
            - 0.5*dot(grad(plus(v)),nn)*plus(u)   - 0.5*dot(grad(plus(u)),nn)*plus(v)   + kappa*plus(u)*plus(v)

    expr   = dot(grad(u),grad(v))

    a = BilinearForm((u,v),  integral(domain, expr) + integral(I, expr_I))
    l = LinearForm(v, integral(domain, f*v))

    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc)

    l2norm = Norm(error, domain, kind='l2')
    h1norm = Norm(error, domain, kind='h1')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++
    
    domain_h = discretize(domain, ncells=ncells)
    Vh       = discretize(V, domain_h, mapping=mappings, degree=degree)  
    
    equation_h = discretize(equation, domain_h, [Vh, Vh])

    l2norm_h = discretize(l2norm, domain_h, Vh)
    h1norm_h = discretize(h1norm, domain_h, Vh)

    x  = equation_h.solve()
    
    uh = VectorFemField( Vh )

    for i in range(len(uh.coeffs[:])):
        uh.coeffs[i][:,:] = x[i][:,:]

    l2_error = l2norm_h.assemble(u=uh)
    h1_error = h1norm_h.assemble(u=uh)

    return uh, l2_error, h1_error


if __name__ == '__main__':

    A = Square('A',bounds1=(0.5, 1.), bounds2=(-1., 0.))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, np.pi))
    C = Square('C',bounds1=(0.5, 1.), bounds2=(np.pi, np.pi + 1))

    domain     = A.join(B, name = 'AB',
                bnd_minus = A.get_boundary(axis=1, ext=1),
                bnd_plus  = B.get_boundary(axis=1, ext=-1))

    domain    = domain.join(C, name = 'ABC',
                bnd_minus = B.get_boundary(axis=1, ext=1),
                bnd_plus  = C.get_boundary(axis=1, ext=-1))

    mapping_1 = IdentityMapping('M1', 2)
    mapping_2 = PolarMapping   ('M2', 2, c1 = 0., c2 = 0., rmin = 0., rmax=1.)
    mapping_3 = AffineMapping  ('M3', 2, c1 = 0., c2 = np.pi, a11 = -1, a22 = -1, a21 = 0, a12 = 0)

    mappings  = [mapping_1, mapping_2, mapping_3]
    domains   = [A, B, C]
    x,y       = domain.coordinates
    solution  = x**2 + y**2
    f         = -4

    uh, l2_error, h1_error = run_poisson_2d(solution, f, domain, mappings, ncells=[2**2,2**2], degree=[2,2])

    # ...

    from sympy import lambdify

    mappings = [lambdify(M.logical_coordinates, M.expressions) for M in mappings]
    solution = lambdify(domain.coordinates, solution)

    etas     = [[refine_array_1d( D.bounds1, 10 ), refine_array_1d( D.bounds2, 10 )] for D in domains]
    # ...

    pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings, etas)]
    num     = [np.array( [[phi( e1,e2 ) for e2 in eta[1]] for e1 in eta[0]] ) for phi,eta in zip(uh.fields, etas)]
    ex      = [np.array( [[solution( *f(e1,e2) ) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings,etas)]

    pcoords  = np.concatenate(pcoords, axis=1)
    num      = np.concatenate(num,     axis=1)
    ex       = np.concatenate(ex,      axis=1)
    
    err      = abs(num - ex)

    xx = pcoords[:,:,0]
    yy = pcoords[:,:,1]

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 3, 1)

    cp = ax.contourf(xx, yy, num, 50, cmap='jet')
    cbar = fig.colorbar(cp, ax=ax,  pad=0.05)
    ax = fig.add_subplot(1, 3, 2)
    cp2 = ax.contourf(xx, yy, ex, 50, cmap='jet') 
    cbar = fig.colorbar(cp2, ax=ax,  pad=0.05)
    ax = fig.add_subplot(1, 3, 3)
    cp3 = ax.contourf(xx, yy, err, 50, cmap='jet')   
    cbar = fig.colorbar(cp3, ax=ax,  pad=0.05)
    plt.show()

