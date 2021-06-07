# -*- coding: UTF-8 -*-

import pytest
import numpy as np
from sympy import pi, cos, sin, sqrt, ImmutableDenseMatrix as Matrix, Tuple, lambdify
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres as sp_gmres
from scipy.sparse.linalg import minres as sp_minres
from scipy.sparse.linalg import cg as sp_cg
from scipy.sparse.linalg import bicg as sp_bicg
from scipy.sparse.linalg import bicgstab as sp_bicgstab

from sympde.calculus import grad, dot, inner, div, curl, cross
from sympde.topology import NormalVector
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square, Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.api.essential_bc   import apply_essential_bc
from psydac.cad.geometry       import refine_knots
from psydac.fem.basic          import FemField
from psydac.fem.vector         import ProductFemSpace
from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil
from psydac.linalg.stencil     import *
from psydac.linalg.block       import *
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL

from sympde.calculus import laplace, grad, Transpose
from sympde.expr     import TerminalExpr

from scipy.sparse.linalg import minres
#==============================================================================
def get_boundaries(*args):

    if not args:
        return ()
    else:
        assert all(1 <= a <= 4 for a in args)
        assert len(set(args)) == len(args)

    boundaries = {1: {'axis': 0, 'ext': -1},
                  2: {'axis': 0, 'ext':  1},
                  3: {'axis': 1, 'ext': -1},
                  4: {'axis': 1, 'ext':  1}}

    return tuple(boundaries[i] for i in args)
#==============================================================================
def run_navier_stokes_2d(domain, f, ue, pe, *, ncells, degree):

    # Maximum number of Newton iterations and convergence tolerance
    N = 20
    TOL = 1e-12

    # ... abstract model
    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    u, v = elements_of(V1, names='u, v')
    p, q = elements_of(V2, names='p, q')

    x, y  = domain.coordinates
    int_0 = lambda expr: integral(domain , expr)

    du = element_of(V1, name='du')
    dp = element_of(V2, name='dp')

    a = BilinearForm(((du,dp),(v, q)), integral(domain, dot(Transpose(grad(du))*u, v) + dot(Transpose(grad(u))*du, v) + inner(grad(du), grad(v)) - div(du)*q - dp*div(v)) )
    l = LinearForm((v, q), integral(domain, dot(Transpose(grad(u))*u, v) + inner(grad(u), grad(v)) - div(u)*q - p*div(v) - dot(f, v)) )

    boundary = Union(*[domain.get_boundary(**kw) for kw in get_boundaries(1,2)])
    bc = EssentialBC(du, ue, boundary)
    equation = find((du, dp), forall=(v, q), lhs=a((du, dp), (v, q)), rhs=l(v, q), bc=bc)

    # Define (abstract) norms
    l2norm_u   = Norm(Matrix([u[0]-ue[0],u[1]-ue[1]]), domain, kind='l2')
    l2norm_p   = Norm(p-pe  , domain, kind='l2')

    l2norm_du  = Norm(Matrix([du[0],du[1]]), domain, kind='l2')
    l2norm_dp  = Norm(dp     , domain, kind='l2')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)

    knots1 = np.array([0, 0, 0, 1.0000, 1.0000, 1.0000])
    knots2 = np.array([0, 0, 0, 1.0000, 1.0000, 1.0000])

    knots1,knots2 = refine_knots([knots1, knots2], ncells=ncells, degree=degree, multiplicity=[2,2])
    knots  = {domain.name:[knots1, knots2]}
    
    # ... discrete spaces
    V1h = discretize(V1, domain_h, degree=degree, knots=knots)
    V2h = discretize(V2, domain_h, degree=degree, knots=knots)
    Xh  = V1h*V2h

    x = BlockVector(Xh.vector_space)

    # ... discretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Xh, Xh], backend=PSYDAC_BACKEND_GPYCCEL)
    a_h        = equation_h.lhs
    l_h        = equation_h.rhs

    # Discretize norms
    l2norm_u_h = discretize(l2norm_u, domain_h, V1h, backend=PSYDAC_BACKEND_GPYCCEL)
    l2norm_p_h = discretize(l2norm_p, domain_h, V2h, backend=PSYDAC_BACKEND_GPYCCEL)

    l2norm_du_h = discretize(l2norm_du, domain_h, V1h, backend=PSYDAC_BACKEND_GPYCCEL)
    l2norm_dp_h = discretize(l2norm_dp, domain_h, V2h, backend=PSYDAC_BACKEND_GPYCCEL)

    x0 = equation_h.compute_dirichlet_bd_conditions()

    # First guess: zero solution
    u_h = FemField(V1h)
    p_h = FemField(V2h)

    u_h[0].coeffs[:,:] = x0[0].coeffs[:,:]
    u_h[1].coeffs[:,:] = x0[1].coeffs[:,:]
    p_h.coeffs[:,:]    = x0[2].coeffs[:,:]

    du_h = FemField(V1h)
    dp_h = FemField(V2h)

    # Newton iteration
    for n in range(N):

        print()
        print('==== iteration {} ===='.format(n))

        M = a_h.assemble(u=u_h, p=p_h)
        b = l_h.assemble(u=u_h, p=p_h)
        apply_essential_bc(M, *equation_h.bc)
        apply_essential_bc(b, *equation_h.bc)
        x,info = minres(M.tosparse().tocsr(), b.toarray(), tol=1e-9)
        x = array_to_stencil(x, b.space)

        du_h[0].coeffs[:] = x[0][:]
        du_h[1].coeffs[:] = x[1][:]
        dp_h.coeffs[:]    = x[2][:]
        dp_h.coeffs[:]    = dp_h.coeffs[:]

        # update field
        u_h -= du_h
        p_h -= dp_h

        # Compute L2 norm of increment
        l2_error_du = l2norm_du_h.assemble(du=du_h)
        l2_error_dp = l2norm_dp_h.assemble(dp=dp_h)

        print('L2_error_norm(du) = {}'.format(l2_error_du))
        print('L2_error_norm(dp) = {}'.format(l2_error_dp))

        if abs(l2_error_du+l2_error_dp) <= TOL:
            print()
            print('CONVERGED')
            break

    l2_error_u = l2norm_u_h.assemble(u=u_h)
    l2_error_p = l2norm_p_h.assemble(p=p_h)

    return l2_error_u, l2_error_p

###############################################################################
#            SERIAL TESTS
###############################################################################
#------------------------------------------------------------------------------
def test_navier_stokes_2d(scipy=True):

    # ... Exact solution
    domain = Square()
    x, y   = domain.coordinates

    mu = 1
    ux = cos(y*pi)
    uy = x*(x-1)
    ue = Matrix([[ux], [uy]])
    pe = sin(pi*y)
    # ...

    # Verify that div(u) = 0
    assert (ux.diff(x) + uy.diff(y)).simplify() == 0

    # ... Compute right-hand side
    from sympde.calculus import laplace, grad 
    from sympde.expr     import TerminalExpr 

    kwargs = dict(dim=2, logical=True)
    a = TerminalExpr(-mu*laplace(ue), **kwargs)
    b = TerminalExpr(    grad(ue), **kwargs)
    c = TerminalExpr(    grad(pe), **kwargs)
    f = (a.T + b.T*ue + c).simplify()

    fx = -mu*(ux.diff(x, 2) + ux.diff(y, 2)) + ux*ux.diff(x) + uy*ux.diff(y) + pe.diff(x)
    fy = -mu*(uy.diff(x, 2) - uy.diff(y, 2)) + ux*uy.diff(x) + uy*uy.diff(y) + pe.diff(y)

    assert (f[0]-fx).simplify() == 0
    assert (f[1]-fy).simplify() == 0

    f  = Tuple(fx, fy)
    # ...

    # Run test
    l2_error_u, l2_error_p = run_navier_stokes_2d(domain, f, ue, pe, ncells=[2**3, 2**3], degree=[2, 2])

    # Check that expected absolute error on velocity and pressure fields
    assert abs(0.00020452836013053793 - l2_error_u ) < 1e-7
    assert abs(0.004127752838826402 - l2_error_p  ) < 1e-7

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
