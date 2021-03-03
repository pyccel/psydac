# -*- coding: UTF-8 -*-

import numpy as np
from sympy import pi, cos, sin, Matrix, Tuple
from scipy import linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres as sp_gmres
from scipy.sparse.linalg import minres as sp_minres
from scipy.sparse.linalg import bicg as sp_bicg
from scipy.sparse.linalg import bicgstab as sp_bicgstab

from sympde.calculus import grad, dot, inner, div
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.fem.vector         import ProductFemSpace
from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil

from psydac.linalg.iterative_solvers import bicg
from psydac.linalg.block import BlockVector

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

#==============================================================================
def run_stokes_2d_dir(f, ue, pe, *, ncells, degree, scipy=False):

    # ... abstract model
    domain = Square()

    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    u, v = elements_of(V1, names='u, v')
    p, q = elements_of(V2, names='p, q')

    x, y  = domain.coordinates
    int_0 = lambda expr: integral(domain , expr)

    a  = BilinearForm(((u, p), (v, q)), int_0(inner(grad(u), grad(v)) - div(u)*q - p*div(v)) )
    l  = LinearForm((v, q), int_0(dot(f, v)))
    bc = EssentialBC(u, 0, domain.boundary)

    equation = find((u, p), forall=(v, q), lhs=a((u, p), (v, q)), rhs=l(v, q), bc=bc)

    error_u = [ue[0]-u[0], ue[1]-u[1]]
    error_p = pe - p

    l2norm_u = Norm(error_u, domain, kind='l2')
    l2norm_p = Norm(error_p, domain, kind='l2')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)

    # ... discrete spaces
    V1h = discretize(V1, domain_h, degree=degree)
    V2h = discretize(V2, domain_h, degree=degree, basis='M')
    Xh  = discretize(X , domain_h, degree=degree, basis='M')

    # ... discretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Xh, Xh])

    # ... discretize error norms
    l2norm_uh = discretize(l2norm_u, domain_h, V1h)
    l2norm_ph = discretize(l2norm_p, domain_h, V2h)

    # ... assemble linear system
    equation_h.assemble()


    # ... solve linear system
    if scipy:
        # Select Scipy's sparse iterative solver among 3 available
        sp_solver = [sp_gmres, sp_minres, sp_bicg, sp_bicgstab][1]
        As = equation_h.linear_system.lhs.tosparse()
        bs = equation_h.linear_system.rhs.toarray()
        xs, info = sp_solver(As, bs, tol=1e-12)
        x = array_to_stencil(xs, Xh.vector_space)
        print('\n', info)

    else:
        # Use Psydac's bi-conjugate gradient solver
        A = equation_h.linear_system.lhs
        b = equation_h.linear_system.rhs
        x, info = bicg(A, A.T, b, tol=1e-12)
        print('\n', info)

    # Numerical solution: velocity field
    # TODO: allow this: uh = FemField(V1h, coeffs=x[0:2]) or similar
    uh = FemField(V1h)
    uh.coeffs[0] = x[0]
    uh.coeffs[1] = x[1]

    # Numerical solution: pressure field
    # TODO: allow this: uh = FemField(V2h, coeffs=x[2])
    ph = FemField(V2h)
    ph.coeffs[:] = x[2][:]

    # Compute norms
    l2_error_u = l2norm_uh.assemble(u = uh)
    l2_error_p = l2norm_ph.assemble(p = ph)

    # Average value of the pressure (should be 0)
    domain_area = V2h.integral(lambda x1, x2: 1.0)
    p_avg = V2h.integral(ph) / domain_area

    print('l2_error(u) = {}'.format(l2_error_u))
    print('l2_error(p) = {}'.format(l2_error_p))
    print('average(p)  = {}'.format(p_avg))

    return locals()

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

#------------------------------------------------------------------------------
def test_stokes_2d_dir_1():

    # ... Exact solution
    from sympy import symbols
    x1, x2 = symbols('x1, x2')
 
    f1 = -x1**2*(x1 - 1)**2*(24*x2 - 12) - 4*x2*(x1**2 + 4*x1*(x1 - 1) + (x1 - 1)**2)*(2*x2**2 - 3*x2 + 1) - 2*pi*cos(2*pi*x1)
    f2 = 4*x1*(2*x1**2 - 3*x1 + 1)*(x2**2 + 4*x2*(x2 - 1) + (x2 - 1)**2) + x2**2*(24*x1 - 12)*(x2 - 1)**2 + 2*pi*cos(2*pi*x2)
    f  = Tuple(f1, f2)

    u1 = x1**2*(-x1 + 1)**2*(4*x2**3 - 6*x2**2 + 2*x2)
    u2 =-x2**2*(-x2 + 1)**2*(4*x1**3 - 6*x1**2 + 2*x1)
    ue = Tuple(u1, u2)
    pe = -sin(2*pi*x1) + sin(2*pi*x2)
    # ...

    # ... Check that exact solution is correct
    from sympde.calculus import laplace, grad
    from sympde.expr import TerminalExpr

    kwargs = dict(dim=2, logical=True)
    a = TerminalExpr(-laplace(ue), **kwargs)
    b = TerminalExpr(    grad(pe), **kwargs)
    c = TerminalExpr(   Matrix(f), **kwargs)
    err = (a.T + b - c).simplify()

    assert err[0] == 0
    assert err[1] == 0
    # ...

    # Run test
    namespace = run_stokes_2d_dir(f, ue, pe, ncells=[2**3, 2**3], degree=[2, 2], scipy=False)

    # Check error on velocity and pressure fields
    assert abs(namespace['l2_error_u'] - 1.86072381490785e-5) < 1e-13
    assert abs(namespace['l2_error_p'] - 2.44281724609567e-2) < 1e-13

    #TODO verify convergence rate

#------------------------------------------------------------------------------
def test_stokes_2d_dir_2():

    # ... Exact solution
    from sympy import symbols
    x1, x2 = symbols('x1, x2')

    u1 =  2000 * x1**2 * (1-x1)**2 * x2 * (1-x2) * (1-2*x2)
    u2 = -2000 * x2**2 * (1-x2)**2 * x1 * (1-x1) * (1-2*x1)
    ue = Tuple(u1, u2)
    pe = x1**2 + x2**2 - 2/3
    # ...

    # Verify that div(u) = 0
    assert (u1.diff(x1) + u2.diff(x2)).simplify() == 0

    # ... Compute right-hand side
    from sympde.calculus import laplace, grad
    from sympde.expr import TerminalExpr

    kwargs = dict(dim=2, logical=True)
    a = TerminalExpr(-laplace(ue), **kwargs)
    b = TerminalExpr(    grad(pe), **kwargs)
    f = (a.T + b).simplify()
    # ...

    # Run test
    namespace = run_stokes_2d_dir(f, ue, pe, ncells=[2**3, 2**3], degree=[4, 4], scipy=True)
