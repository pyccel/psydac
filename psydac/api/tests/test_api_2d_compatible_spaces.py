# -*- coding: UTF-8 -*-

import numpy as np
from sympy import pi, cos, sin, Matrix, Tuple
from scipy import linalg
from scipy.sparse.linalg import spsolve

from sympde.calculus import grad, dot, inner, div, curl, cross
from sympde.topology import NormalVector
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of
from sympde.topology import Square
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil

from psydac.linalg.iterative_solvers import pcg

#==============================================================================

def run_system_1_2d_dir(f0, sol, ncells, degree):
    # ... abstract model
    domain = Square()

    V1 = VectorFunctionSpace('V1', domain, kind='Hdiv')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    x,y = domain.coordinates

    F = element_of(V2, name='F')


    p,q = [element_of(V1, name=i) for i in ['p', 'q']]
    u,v = [element_of(V2, name=i) for i in ['u', 'v']]

    int_0 = lambda expr: integral(domain , expr)
    
    a  = BilinearForm(((p,u),(q,v)), int_0(dot(p,q) + div(q)*u + div(p)*v) )
    l  = LinearForm((q,v), int_0(f0*v))
    
    # ...
    error = F-sol
    l2norm_F = Norm(error, domain, kind='l2')

    # ...
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
    # ...

    # ...
    ah.assemble()
    M   = ah.linear_system.lhs.tosparse().tocsc()
    rhs = ah.linear_system.rhs.toarray()

    x   = spsolve(M, rhs)
    x   = array_to_stencil(x, Xh.vector_space)
    
    # ...
    Fh = FemField( V2h )
    Fh.coeffs[:, :] = x[2][:,:]
    # ...

    # ... compute norms
#    l2norm_F_h._set_func('dependencies_evlw0ux7','assembly')
    l2_error = l2norm_F_h.assemble(F=Fh)

    return l2_error
    
def run_system_2_2d_dir(f1, f2,u1, u2, ncells, degree):
    # ... abstract model
    domain = Square()

    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    x,y = domain.coordinates

    F = element_of(V1, name='F')

    u,v = [element_of(V1, name=i) for i in ['u', 'v']]
    p,q = [element_of(V2, name=i) for i in ['p', 'q']]

    int_0 = lambda expr: integral(domain , expr)

    
    a  = BilinearForm(((u,p),(v,q)), int_0(inner(grad(u),grad(v)) + div(u)*q - p*div(v)) )
    l  = LinearForm((v,q), int_0(f1*v[0]+f2*v[1]+q))
    
    bc = EssentialBC(u, 0, domain.boundary)
    equation = find([u,p], forall=[v,q], lhs=a((u,p),(v,q)), rhs=l(v,q), bc=bc)

    error = Matrix([F[0]-u1, F[1]-u2])
    l2norm_F = Norm(error, domain, kind='l2')

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
    
    phi1 = FemField(V1h)
    phi2 = FemField(V2h)

    
    phi1.coeffs[0][s11:e11+1, s12:e12+1] = x[:(e11+1)*(e12+1)].reshape((e11+1-s11, e12+1-s12))
    phi1.coeffs[1][s21:e21+1, s22:e22+1] = x[(e11+1)*(e12+1):v2h_s].reshape((e21+1-s21, e22+1-s22))
    phi2.coeffs[s31:e31+1, s32:e32+1]    = x[v2h_s:-1].reshape((e31-s31+1, e32-s32+1))
    
        # ... compute norms
    l2_error = l2norm_F_h.assemble(F=phi1)
    return l2_error

def run_system_3_2d_dir(uex, f, alpha, ncells, degree):

    domain = Square('A')
    B_dirichlet_0 = domain.boundary

    V  = VectorFunctionSpace('V', domain, kind='hcurl')

    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    F  = element_of(V, name='F')

    # Bilinear form a: V x V --> R
    a   = BilinearForm((u, v), integral(domain, curl(u)*curl(v) + alpha*dot(u,v)))

    nn   = NormalVector('nn')
    a_bc = BilinearForm((u, v), integral(domain.boundary, 1e30 * cross(u, nn) * cross(v, nn)))


    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, dot(f,v)))

    # l2 error
    error   = Matrix([F[0]-uex[0],F[1]-uex[1]])
    l2norm  = Norm(error, domain, kind='l2')

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, ncells=ncells)

    # Discrete spaces
    Vh = discretize(V, domain_h, degree=degree)

    # Discretize bi-linear and linear form
    a_h    = discretize(a, domain_h, [Vh, Vh])
    a_bc_h = discretize(a_bc, domain_h, [Vh, Vh])

    l_h          = discretize(l, domain_h, Vh)
    l2_norm_h    = discretize(l2norm, domain_h, Vh)

    M = a_h.assemble() + a_bc_h.assemble()
    b = l_h.assemble()

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    sol, info  = pcg(M ,b, pc='jacobi', tol=1e-8)

    uh       = FemField( Vh, sol )
    l2_error = l2_norm_h.assemble(F=uh)

    return l2_error

###############################################################################
#            SERIAL TESTS
###############################################################################

def test_api_system_1_2d_dir_1():
    from sympy import symbols
    x1, x2 = symbols('x1, x2')

    f0 =  -2*x1*(1-x1) -2*x2*(1-x2)
    u  = x1*(1-x1)*x2*(1-x2)

    l2_error = run_system_1_2d_dir(f0,u, ncells=[2**3,2**3], degree=[2,2])
    assert l2_error-0.00030070020628128664<1e-13

def test_api_system_2_2d_dir_1():

    from sympy import symbols
    x1, x2 = symbols('x1, x2')
 
    f1 = -x1**2*(x1 - 1)**2*(24*x2 - 12) - 4*x2*(x1**2 + 4*x1*(x1 - 1) + (x1 - 1)**2)*(2*x2**2 - 3*x2 + 1) - 2*pi*cos(2*pi*x1)
    f2 = 4*x1*(2*x1**2 - 3*x1 + 1)*(x2**2 + 4*x2*(x2 - 1) + (x2 - 1)**2) + x2**2*(24*x1 - 12)*(x2 - 1)**2 + 2*pi*cos(2*pi*x2)
    u1 = x1**2*(-x1 + 1)**2*(4*x2**3 - 6*x2**2 + 2*x2)
    u2 =-x2**2*(-x2 + 1)**2*(4*x1**3 - 6*x1**2 + 2*x1)
    p  = sin(2*pi*x1) - sin(2*pi*x2)
    
    l2_error = run_system_2_2d_dir(f1, f2, u1, u2, ncells=[2**4,2**4], degree=[2,2])
    assert l2_error-0.020113712082281063<1e-13
    #TODO verify convergence rate

def test_api_system_3_2d_dir_1():
    from sympy import symbols
    x,y,z    = symbols('x1, x2, x3')

    alpha    = 1.
    uex      = Tuple(sin(pi*y), sin(pi*x)*cos(pi*y))
    f        = Tuple(alpha*sin(pi*y) - pi**2*sin(pi*y)*cos(pi*x) + pi**2*sin(pi*y),
                     alpha*sin(pi*x)*cos(pi*y) + pi**2*sin(pi*x)*cos(pi*y))

    l2_error = run_system_3_2d_dir(uex, f, alpha, ncells=[2**3,2**3], degree=[2,2])
    assert abs(l2_error-0.0029394893438220502)<1e-13

