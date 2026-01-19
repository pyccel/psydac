#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

import pytest
import numpy as np
from sympy import pi, cos, sin, sqrt, exp, ImmutableDenseMatrix as Matrix, Tuple, lambdify
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres as sp_gmres
from scipy.sparse.linalg import minres as sp_minres
from scipy.sparse.linalg import cg as sp_cg
from scipy.sparse.linalg import bicg as sp_bicg
from scipy.sparse.linalg import bicgstab as sp_bicgstab

from sympde.calculus import grad, dot, inner, div, curl, cross
from sympde.calculus import Transpose, laplace
from sympde.topology import NormalVector
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Domain, Square, Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC
from sympde.core     import Constant
from sympde.expr     import TerminalExpr
from sympde.expr     import linearize

from psydac.api.essential_bc   import apply_essential_bc
from psydac.fem.basic          import FemField
from psydac.fem.vector         import VectorFemSpace
from psydac.core.bsplines      import make_knots
from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_psydac
from psydac.linalg.stencil     import *
from psydac.linalg.block       import *
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
from psydac.utilities.utils    import refine_array_1d, animate_field, split_space, split_field
from psydac.linalg.solvers     import inverse

from mpi4py import MPI
comm = MPI.COMM_WORLD
#==============================================================================

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

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

#------------------------------------------------------------------------------
def scipy_solver(M, b):
    x  = spsolve(M.tosparse().tocsr(), b.toarray())
    x  = array_to_psydac(x, b.space)
    return x,0

#------------------------------------------------------------------------------
def psydac_solver(M, b):
    M_inv = inverse(M, 'lsmr', maxiter=10000, tol=1e-6)
    x = M_inv @ b
    info = M_inv.get_info()
    return x, info

#==============================================================================
def run_time_dependent_navier_stokes_2d(filename, dt_h, nt, newton_tol=1e-4, max_newton_iter=100, scipy=True):
    """
        Time dependent Navier Stokes solver in a 2d domain.
        this example was taken from the pyiga library
        https://github.com/c-f-h/pyiga/blob/master/notebooks/solve-navier-stokes.ipynb
    """
    domain  = Domain.from_file(filename)

    # ... abstract model
    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    u0, u, v, du = elements_of(V1, names='u0, u, v, du')
    p0, p, q, dp = elements_of(V2, names='p0, p, q, dp')

    x, y  = domain.coordinates
    int_0 = lambda expr: integral(domain , expr)

    # time step
    dt = Constant(name='dt', real=True)

    # Boundaries
    boundary_h = Union(*[domain.get_boundary(**kw) for kw in get_boundaries(3,4)])
    boundary   = Union(*[domain.get_boundary(**kw) for kw in get_boundaries(1)])
    ue         = Tuple(40*y*(0.5-y)*exp(-100*(y-0.25)**2), 0)
    bc         = [EssentialBC(du, ue, boundary), EssentialBC(du, 0, boundary_h)]

    # Reynolds number
    Re = 1e4

    Fl = lambda u,p: Re**-1*inner(grad(u), grad(v)) - div(u)*q - p*div(v) + 1e-10*p*q
    F  = lambda u,p: dot(Transpose(grad(u))*u,v) + Fl(u,p)
    
    l = LinearForm((v, q), integral(domain, dot(u,v)-dot(u0,v) + dt/2 * (F(u,p) + F(u0,p0)) ))
    a = linearize(l, (u,p), trials=(du, dp))

    equation  = find((du, dp), forall=(v, q), lhs=a((du, dp), (v, q)), rhs=l(v, q), bc=bc)

    # Use the stokes equation to compute the initial solution
    a_stokes = BilinearForm(((du,dp),(v, q)), integral(domain, Fl(du,dp)) )
    l_stokes = LinearForm((v, q), integral(domain, dot(v,Tuple(0,0)) ))

    equation_stokes = find((du, dp), forall=(v, q), lhs=a_stokes((du, dp), (v, q)), rhs=l_stokes(v, q), bc=bc)

    # Define (abstract) norms
    l2norm_du  = Norm(Matrix([du[0],du[1]]), domain, kind='l2')
    l2norm_dp  = Norm(dp, domain, kind='l2')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename)

    # ... discrete spaces
    V1h = discretize(V1, domain_h)
    V2h = discretize(V2, domain_h)
    Xh  = VectorFemSpace(V1h, V2h)

    # ... discretize the equations
    equation_h        = discretize(equation,        domain_h, [Xh, Xh], backend=PSYDAC_BACKEND_GPYCCEL)
    equation_stokes_h = discretize(equation_stokes, domain_h, [Xh, Xh], backend=PSYDAC_BACKEND_GPYCCEL)

    a_h        = equation_h.lhs
    l_h        = equation_h.rhs

    # Discretize the norms
    l2norm_du_h = discretize(l2norm_du, domain_h, V1h, backend=PSYDAC_BACKEND_GPYCCEL)
    l2norm_dp_h = discretize(l2norm_dp, domain_h, V2h, backend=PSYDAC_BACKEND_GPYCCEL)

    # compute the initial solution
    equation_stokes_h.set_solver('bicg', tol=1e-15)
    x0 = equation_stokes_h.solve()

    u0_h = FemField(V1h)
    p0_h = FemField(V2h)

    u_h  = FemField(V1h)
    p_h  = FemField(V2h)

    du_h = FemField(V1h)
    dp_h = FemField(V2h)

    # First guess
    u_h[0].coeffs[:,:] = x0[0].coeffs[:,:]
    u_h[1].coeffs[:,:] = x0[1].coeffs[:,:]
    p_h.coeffs[:,:]    = x0[2].coeffs[:,:]

    # store the solutions
    solutions                    = [FemField(V1h)]
    solutions[-1][0].coeffs[:,:] = u_h[0].coeffs[:,:]
    solutions[-1][1].coeffs[:,:] = u_h[1].coeffs[:,:]
    Tf = dt_h*(nt+1)
    t  = 0

    solver = scipy_solver if scipy else psydac_solver

    while t<Tf:
        t += dt_h
        print()
        print('======= time {}/{} ======='.format(t,Tf))

        u0_h[0].coeffs[:,:] = u_h[0].coeffs[:,:]
        u0_h[1].coeffs[:,:] = u_h[1].coeffs[:,:]
        p0_h.coeffs[:,:]    = p_h.coeffs[:,:]

        # Newton iteration
        for n in range(max_newton_iter):
            print()
            print('==== iteration {} ===='.format(n))

            M = a_h.assemble(u=u_h, p=p_h, dt=dt_h)
            b = l_h.assemble(u=u_h, p=p_h, u0=u0_h, p0=p0_h, dt=dt_h)

            apply_essential_bc(M, *equation_h.bc, identity=True)
            apply_essential_bc(b, *equation_h.bc)

            x,info = solver(M, b)

            du_h[0].coeffs[:] = x[0][:]
            du_h[1].coeffs[:] = x[1][:]
            dp_h.coeffs[:]    = x[2][:]

            # Compute L2 norm of increment
            l2_error_du = l2norm_du_h.assemble(du=du_h)
            l2_error_dp = l2norm_dp_h.assemble(dp=dp_h)

            print('L2_error_norm(du) = {}'.format(l2_error_du))
            print('L2_error_norm(dp) = {}'.format(l2_error_dp))

            if abs(l2_error_du) <= newton_tol:
                print()
                print('CONVERGED')
                break
            elif n == max_newton_iter-1 or abs(l2_error_du)>1/newton_tol or abs(l2_error_dp) > 1/newton_tol:
                print()
                print('NOT CONVERGED')
                t = Tf
                return solutions, p_h, domain, domain_h

            # update field
            u_h -= du_h
            p_h -= dp_h

        solutions.append(FemField(V1h))
        solutions[-1][0].coeffs[:,:] = u_h[0].coeffs[:,:]
        solutions[-1][1].coeffs[:,:] = u_h[1].coeffs[:,:]

    return solutions, p_h, domain, domain_h

#==============================================================================
def run_steady_state_navier_stokes_2d(domain, f, ue, pe, *, ncells, degree, multiplicity):
    """
        Navier Stokes solver for the 2d steady-state problem.
    """
    # Maximum number of Newton iterations and convergence tolerance
    N = 20
    TOL = 1e-4

    # ... abstract model
    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    u, v = elements_of(V1, names='u, v')
    p, q = elements_of(V2, names='p, q')

    du = element_of(V1, name='du')
    dp = element_of(V2, name='dp')

    boundary = Union(*[domain.get_boundary(**kw) for kw in get_boundaries(1,2)])

    a_b = BilinearForm(((u,p),(v, q)), integral(boundary, dot(u,v)) )
    l_b = LinearForm((v, q), integral(boundary, dot(ue, v)) )

    equation_b = find((u, p), forall=(v, q), lhs=a_b((u, p), (v, q)), rhs=l_b(v, q))

    l = LinearForm((v, q), integral(domain, dot(Transpose(grad(u))*u, v) + inner(grad(u), grad(v)) - div(u)*q - p*div(v) - dot(f, v)) )
    a = linearize(l, (u,p), trials=(du, dp))

    bc       = EssentialBC(du, 0, boundary)

    equation = find((du, dp), forall=(v, q), lhs=a((du, dp), (v, q)), rhs=l(v, q), bc=bc)

    # Define (abstract) norms
    l2norm_u   = Norm(Matrix([u[0]-ue[0], u[1]-ue[1]]), domain, kind='l2')
    l2norm_p   = Norm(p - pe, domain, kind='l2')

    l2norm_du  = Norm(Matrix([du[0],du[1]]), domain, kind='l2')
    l2norm_dp  = Norm(dp, domain, kind='l2')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    breaks1 = np.linspace(0, 1, ncells[0]+1)
    breaks2 = np.linspace(0, 1, ncells[1]+1)

    knots1 = make_knots(breaks1, degree=degree[0], multiplicity=multiplicity[0], periodic=False)
    knots2 = make_knots(breaks2, degree=degree[1], multiplicity=multiplicity[1], periodic=False)

    knots  = [knots1, knots2]

    # ... discrete spaces
    Xh  = discretize(X, domain_h, degree=degree, knots=knots, sequence='TH')

    V1h, V2h = split_space(Xh)

    # ... discretize the equation and equation_b
    equation_b_h = discretize(equation_b, domain_h, [Xh, Xh], backend=PSYDAC_BACKEND_GPYCCEL)
    equation_h   = discretize(equation,   domain_h, [Xh, Xh], backend=PSYDAC_BACKEND_GPYCCEL)

    # Discretize norms
    l2norm_u_h = discretize(l2norm_u, domain_h, V1h, backend=PSYDAC_BACKEND_GPYCCEL)
    l2norm_p_h = discretize(l2norm_p, domain_h, V2h, backend=PSYDAC_BACKEND_GPYCCEL, nquads=[deg+1 for deg in degree])

    l2norm_du_h = discretize(l2norm_du, domain_h, V1h, backend=PSYDAC_BACKEND_GPYCCEL)
    l2norm_dp_h = discretize(l2norm_dp, domain_h, V2h, backend=PSYDAC_BACKEND_GPYCCEL)

    x0 = equation_b_h.solve()

    # First guess: zero solution
    u_h = FemField(V1h)
    p_h = FemField(V2h)

    u_h, p_h = split_field(x0, (V1h, V2h))

    du_h = FemField(V1h)
    dp_h = FemField(V2h)

    equation_h.set_solver('bicg', tol=1e-9, info=True)

    # Newton iteration
    for n in range(N):
        print('==== iteration {} ===='.format(n))

        xh, info   = equation_h.solve(u=u_h, p=p_h)

        split_field(xh, (V1h, V2h), out=(du_h, dp_h))

        # update field
        u_h -= du_h
        p_h -= dp_h

        # Compute L2 norm of increment
        l2_error_du = l2norm_du_h.assemble(du=du_h)
        l2_error_dp = l2norm_dp_h.assemble(dp=dp_h)

        print('L2_error_norm(du) = {}'.format(l2_error_du))
        print('L2_error_norm(dp) = {}'.format(l2_error_dp))

        if abs(l2_error_du)<= TOL and abs(l2_error_dp) <= TOL:
            print()
            print('CONVERGED')
            break

    l2_error_u = l2norm_u_h.assemble(u=u_h)
    l2_error_p = l2norm_p_h.assemble(p=p_h)

    return l2_error_u, l2_error_p

###############################################################################
#            SERIAL TESTS
###############################################################################
def test_st_navier_stokes_2d():

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

    a = TerminalExpr(-mu*laplace(ue), domain)
    b = TerminalExpr(    grad(ue), domain)
    c = TerminalExpr(    grad(pe), domain)    
    f = (a + b.T*ue + c).simplify()
    
    fx = -mu*(ux.diff(x, 2) + ux.diff(y, 2)) + ux*ux.diff(x) + uy*ux.diff(y) + pe.diff(x)
    fy = -mu*(uy.diff(x, 2) + uy.diff(y, 2)) + ux*uy.diff(x) + uy*uy.diff(y) + pe.diff(y)

    assert (f[0]-fx).simplify() == 0
    assert (f[1]-fy).simplify() == 0
    
    f  = Tuple(fx, fy)
    # ...

    # Run test

    l2_error_u, l2_error_p = run_steady_state_navier_stokes_2d(domain, f, ue, pe, ncells=[2**3,2**3], degree=[2, 2], multiplicity=[2,2])

    # Check that expected absolute error on velocity and pressure fields
    assert abs(0.00020452836013053793 - l2_error_u ) < 1e-7
    assert abs(0.004127752838826402 - l2_error_p  ) < 1e-7

#------------------------------------------------------------------------------
def test_navier_stokes_2d():
    Tf       = 1.
    dt_h     = 0.05
    nt       = Tf//dt_h
    filename = os.path.join(mesh_dir, 'bent_pipe.h5')
    solutions, p_h, domain, domain_h = run_time_dependent_navier_stokes_2d(filename, dt_h=dt_h, nt=nt, newton_tol=1e-10, scipy=True)

###############################################################################
#            PARALLEL TESTS
###############################################################################
@pytest.mark.mpi
def test_st_navier_stokes_2d_parallel():

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

    a = TerminalExpr(-mu*laplace(ue), domain)
    b = TerminalExpr(    grad(ue), domain)
    c = TerminalExpr(    grad(pe), domain)
    f = (a + b.T*ue + c).simplify()

    fx = -mu*(ux.diff(x, 2) + ux.diff(y, 2)) + ux*ux.diff(x) + uy*ux.diff(y) + pe.diff(x)
    fy = -mu*(uy.diff(x, 2) + uy.diff(y, 2)) + ux*uy.diff(x) + uy*uy.diff(y) + pe.diff(y)

    assert (f[0]-fx).simplify() == 0
    assert (f[1]-fy).simplify() == 0

    f  = Tuple(fx, fy)
    # ...

    # Run test

    l2_error_u, l2_error_p = run_steady_state_navier_stokes_2d(domain, f, ue, pe, ncells=[2**3,2**3], degree=[2, 2], multiplicity=[2,2])

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

#------------------------------------------------------------------------------
if __name__ == '__main__':

    # do some handpicked tests

    verify = 'st_navier_stokes_2d'
    # verify = 'st_navier_stokes_2d_parallel'

    if verify == 'st_navier_stokes_2d':
        print('Running test_st_navier_stokes_2d()')
        test_st_navier_stokes_2d()
    
    elif verify == 'st_navier_stokes_2d_parallel':
        print('Running test_st_navier_stokes_2d_parallel()')
        test_st_navier_stokes_2d_parallel()
        
    else:
    
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from time       import time

        Tf       = 3.
        dt_h     = 0.05
        nt       = Tf//dt_h
        filename = os.path.join(mesh_dir, 'bent_pipe.h5')

        solutions, p_h, domain, domain_h = run_time_dependent_navier_stokes_2d(filename, dt_h=dt_h, nt=nt, scipy=False)

        domain = domain.logical_domain
        mapping = domain_h.mappings['patch_0']

        anim = animate_field(solutions, domain, mapping, res=(150,150), progress=True)
        anim.save('animated_fields_{}_{}.mp4'.format(str(Tf).replace('.','_'), str(dt_h).replace('.','_')), writer=animation.FFMpegWriter(fps=60))

