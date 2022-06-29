# -*- coding: UTF-8 -*-

import os
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
from sympde.calculus import minus, plus
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
from psydac.fem.vector         import ProductFemSpace
from psydac.core.bsplines      import make_knots
from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil
from psydac.linalg.stencil     import *
from psydac.linalg.block       import *
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
from psydac.utilities.utils    import refine_array_1d, animate_field, split_space, split_field
from psydac.linalg.iterative_solvers import cg, pcg, bicg, lsmr

from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_grid_vals
from psydac.feec.multipatch.plotting_utilities import get_patch_knots_gridlines, my_small_plot

import matplotlib.pyplot as plt
from matplotlib import animation
from time       import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
#==============================================================================
# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

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
    x  = array_to_stencil(x, b.space)
    return x,0

#------------------------------------------------------------------------------
def psydac_solver(M, b):
    return lsmr(M, M.T, b, maxiter=10000, tol=1e-6)

#==============================================================================
def run_time_dependent_navier_stokes_2d(domain, f, ue, pe, mu, *,
                                        boundary, boundary_h, boundary_n, 
                                        ncells, degree, multiplicity,
                                        filename, dt_h, nt, newton_tol=1e-5,
                                        max_newton_iter=50, scipy=True):


    assert filename is not None

    t0 = time()
    # ... abstract model
    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    ut, u, v, un = elements_of(V1, names='ut, u, v, un')
    pt, p, q, pn = elements_of(V2, names='pt, p, q, pn')

    nn    = NormalVector('nn')

    # time step
    dt = Constant(name='dt')

    int_0  = lambda expr: integral(domain , expr)
    int_1  = lambda expr: integral(boundary, expr)
    int_2  = lambda expr: integral(boundary_h, expr)
    int_3  = lambda expr: integral(boundary_n, expr)
    int_4  = lambda expr: integral(domain.interfaces, expr)
    jump   = lambda u: -plus(u)*Transpose(nn)+minus(u)*Transpose(nn)
    avr    = lambda u: 0.5*minus(u)+0.5*plus(u)
    grad_s = lambda u:0.5*Transpose(grad(u))+0.5*grad(u)

    Fl = lambda u,p,v,q: mu*inner(grad_s(u), grad_s(v)) - div(u)*q - p*div(v)
    Fn = lambda u,p,v,q:Fl(u,p,v,q) + dot(Transpose(grad(u))*u, v)

    bd_pen   = 10**5
    jump_pen = 10**5

    a = BilinearForm(((u, p),(v, q)), int_0(dot(u,v)*(2/dt) + dot(Transpose(grad(u))*un, v) + dot(Transpose(grad(un))*u, v) + Fl(u,p,v,q)) 
                                     +int_1(-mu*inner(grad_s(v),u*Transpose(nn)) - mu*inner(grad_s(u),v*Transpose(nn)) + bd_pen*mu*inner(u*Transpose(nn),v*Transpose(nn)))
                                     +int_2(-mu*inner(grad_s(v),u*Transpose(nn)) - mu*inner(grad_s(u),v*Transpose(nn)) + bd_pen*mu*inner(u*Transpose(nn),v*Transpose(nn)))
                                     +int_4(-mu*inner(grad_s(avr(v)), jump(u))-mu*inner(grad_s(avr(u)), jump(v)) + 2*mu*jump_pen*inner(jump(u), jump(v)))
                                     )

    l = LinearForm((v, q), int_0(dot(ut, v)*(2/dt) - Fn(ut,pt,v,q) + dot(f, v) + dot(Transpose(grad(un))*un, v)) 
                        + int_1(-mu*inner(grad_s(v),ue*Transpose(nn)) + bd_pen*mu*inner(ue*Transpose(nn),v*Transpose(nn)))
                        + int_3(inner(grad(0.5*mu*ue),v*Transpose(nn))+inner(Transpose(grad(0.5*mu*ue)),v*Transpose(nn))- pe*dot(v,nn))
                        )

    equation  = find((u, p), forall=(v, q), lhs=a((u, p), (v, q)), rhs=l(v, q))

    # Use the stokes equation to compute the initial solution
    a_stokes = BilinearForm(((u,p),(v, q)), int_0(Fl(u,p,v,q))
                                           +int_1(-mu*inner(grad(v),u*Transpose(nn)) - mu*inner(grad(u),v*Transpose(nn)) + bd_pen*mu*inner(u*Transpose(nn),v*Transpose(nn)))
                                           +int_2(-mu*inner(grad(v),u*Transpose(nn)) - mu*inner(grad(u),v*Transpose(nn)) + bd_pen*mu*inner(u*Transpose(nn),v*Transpose(nn)))
                                           +int_4(-mu*inner(grad(avr(v)), jump(u))-mu*inner(grad(avr(u)), jump(v)) + 2*mu*jump_pen*inner(jump(u), jump(v)))
                                           )

    l_stokes = LinearForm((v, q), int_1(-mu*inner(grad_s(v),ue*Transpose(nn)) + bd_pen*mu*inner(ue*Transpose(nn),v*Transpose(nn)))
                                  )

    equation_stokes = find((u, p), forall=(v, q), lhs=a_stokes((u, p), (v, q)), rhs=l_stokes(v, q))

    # Define (abstract) norms
    l2norm_du  = Norm(Matrix([u[0],u[1]]), domain, kind='l2')
    l2norm_dp  = Norm(p     , domain, kind='l2')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename)

    # ... discrete spaces
    V1h = discretize(V1, domain_h)
    V2h = discretize(V2, domain_h)
    Xh  = discretize(X, domain_h)

    # ... discretize the equations
    equation_h        = discretize(equation,        domain_h, [Xh, Xh], backend=PSYDAC_BACKEND_GPYCCEL)
    equation_stokes_h = discretize(equation_stokes, domain_h, [Xh, Xh], backend=PSYDAC_BACKEND_GPYCCEL)

    # Discretize the norms
    l2norm_du_h = discretize(l2norm_du, domain_h, V1h, backend=PSYDAC_BACKEND_GPYCCEL)
    l2norm_dp_h = discretize(l2norm_dp, domain_h, V2h, backend=PSYDAC_BACKEND_GPYCCEL)

    t1 = time()
    print("Elapsed time for the problem discretization :", t1-t0)

    print("Begin solvers")
    # compute the initial solution
    equation_stokes_h.set_solver('direct')
    x0 = equation_stokes_h.solve(t=0.)

    ut_h = FemField(V1h)
    pt_h = FemField(V2h)

    u_h  = FemField(V1h)
    p_h  = FemField(V2h)

    new_u_h = FemField(V1h)
    new_p_h = FemField(V2h)

    # store the solutions
    solutions  = [FemField(V1h)]
    for i in range(len(domain)):
        # First guess
        ut_h[i][0].coeffs[:,:] = x0[i][0].coeffs[:,:]
        ut_h[i][1].coeffs[:,:] = x0[i][1].coeffs[:,:]
        pt_h[i].coeffs[:,:]    = x0[i][2].coeffs[:,:]

        solutions[-1][i][0].coeffs[:,:] = ut_h[i][0].coeffs[:,:]
        solutions[-1][i][1].coeffs[:,:] = ut_h[i][1].coeffs[:,:]

    equation_h.set_solver(solver='direct')
    Tf = dt_h*(nt+1)
    t  = 0
    while t<Tf:
        t += dt_h
        print()
        print('======= time {}/{} ======='.format(t,Tf))

        for i in range(len(domain)):
            ut_h[i][0].coeffs[:,:] = u_h[i][0].coeffs[:,:]
            ut_h[i][1].coeffs[:,:] = u_h[i][1].coeffs[:,:]
            pt_h[i].coeffs[:,:]    = p_h[i].coeffs[:,:]

        # Newton iteration
        for n in range(max_newton_iter):
            print()
            print('==== iteration {} ===='.format(n))
            xh = equation_h.solve(un=u_h, pn=p_h, ut=ut_h, pt=pt_h, dt=dt_h, t=t)

            if len(domain)>1:
                for i in range(len(domain)):
                    new_u_h[i][0].coeffs[:,:] = xh[i][0].coeffs[:,:]
                    new_u_h[i][1].coeffs[:,:] = xh[i][1].coeffs[:,:]
                    new_p_h[i].coeffs[:,:]    = xh[i][2].coeffs[:,:]

            # Compute L2 norm of increment
            l2_error_du = l2norm_du_h.assemble(u=(new_u_h-u_h))
            l2_error_dp = l2norm_dp_h.assemble(p=(new_p_h-p_h))

            print('L2_error_norm(du) = {}'.format(l2_error_du))
            print('L2_error_norm(dp) = {}'.format(l2_error_dp))

            u_h = new_u_h.copy()
            p_h = new_p_h.copy()

            if abs(l2_error_du) <= newton_tol and abs(l2_error_dp) <= newton_tol:
                print()
                print('CONVERGED')
                break
            elif n == max_newton_iter-1:
                print()
                print('NOT CONVERGED')
                t = Tf
                return solutions, p_h, domain, domain_h

        solutions.append(FemField(V1h))
        for i in range(len(domain)):
            solutions[-1][i][0].coeffs[:,:] = u_h[i][0].coeffs[:,:]
            solutions[-1][i][1].coeffs[:,:] = u_h[i][1].coeffs[:,:]

    return solutions, p_h, domain_h

#==============================================================================
def run_steady_state_navier_stokes_2d(domain, f, ue, pe, mu, *, boundary, boundary_h, boundary_n, ncells, degree, multiplicity, filename):
    """
        Navier Stokes solver for the 2d steady-state problem.
    """
    # Maximum number of Newton iterations and convergence tolerance
    N = 100
    TOL = 1e-5

    # ... abstract model
    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    u, v = elements_of(V1, names='u, v')
    p, q = elements_of(V2, names='p, q')

    un = element_of(V1, name='un')
    pn = element_of(V2, name='pn')

    nn = NormalVector("nn")

    interface = domain.interfaces

#    jump_pen   = 5*(degree[0]+1)
#    bd_pen     = 5*(degree[0]+1)

    bd_pen   = 10**5
    jump_pen = 10**5

    grad_s = lambda u:0.5*Transpose(grad(u))+0.5*grad(u)
    a = BilinearForm(((u, p),(v, q)), integral(domain, dot(Transpose(grad(u))*un, v) + dot(Transpose(grad(un))*u, v) + mu*inner(grad_s(u), grad_s(v)) - div(u)*q - p*div(v)) 
                                     +integral(boundary,   -mu*inner(grad_s(v),u*Transpose(nn)) - mu*inner(grad_s(u),v*Transpose(nn)) + bd_pen*mu*inner(u*Transpose(nn),v*Transpose(nn)))
                                     +integral(boundary_h, -mu*inner(grad_s(v),u*Transpose(nn)) - mu*inner(grad_s(u),v*Transpose(nn)) + bd_pen*mu*inner(u*Transpose(nn),v*Transpose(nn)))
                                     )

    l = LinearForm((v,q), integral(domain,  dot(f, v) + dot(Transpose(grad(un))*un, v)) 
                        + integral(boundary, -mu*inner(grad_s(v),ue*Transpose(nn)) + bd_pen*mu*inner(ue*Transpose(nn),v*Transpose(nn)))
#                        + integral(boundary_n, inner(grad(0.5*mu*ue),v*Transpose(nn))+inner(Transpose(grad(0.5*mu*ue)),v*Transpose(nn))- pe*dot(v,nn))
                        )

    jump  = lambda u: -plus(u)*Transpose(nn)+minus(u)*Transpose(nn)
    avr   = lambda u: 0.5*minus(u)+0.5*plus(u)
    expr_I = integral(interface, -mu*inner(grad_s(avr(v)), jump(u))-mu*inner(grad_s(avr(u)), jump(v)) + 2*mu*jump_pen*inner(jump(u), jump(v)))

    equation = find((u, p), forall=(v, q), lhs=(a((u, p), (v, q))+expr_I), rhs=l(v, q))

    # Define (abstract) norms
    l2norm_u   = Norm(Matrix([u[0]-ue[0],u[1]-ue[1]]), domain, kind='l2')
    l2norm_p   = Norm(p-pe  , domain, kind='l2')

    l2norm_du  = Norm(Matrix([u[0],u[1]]), domain, kind='l2')
    l2norm_dp  = Norm(p     , domain, kind='l2')

    # ... create the computational domain from a topological domain


#    min_coords = domain.logical_domain.min_coords
#    max_coords = domain.logical_domain.max_coords
#    breaks1 = np.linspace(min_coords[0], max_coords[0], ncells[0]+1)
#    breaks2 = np.linspace(min_coords[1], max_coords[1], ncells[1]+1)

#    knots1 = make_knots(breaks1, degree=degree[0], multiplicity=multiplicity[0], periodic=False)
#    knots2 = make_knots(breaks2, degree=degree[1], multiplicity=multiplicity[1], periodic=False)

#    knots  = [knots1, knots2]

    if filename is None:
        domain_h = discretize(domain, ncells=ncells, comm=comm)
        # ... discrete spaces
        Xh   = discretize(X, domain_h, degree=degree)
        V1h  = discretize(V1, domain_h, degree=degree)
        V2h  = discretize(V2, domain_h, degree=degree)

    else:
        domain_h = discretize(domain, filename=filename, comm=comm)
        # ... discrete spaces
        Xh   = discretize(X, domain_h)
        V1h  = discretize(V1, domain_h)
        V2h  = discretize(V2, domain_h)
    # ... discretize the equation

    equation_h   = discretize(equation, domain_h, [Xh, Xh], backend=PSYDAC_BACKEND_GPYCCEL)

    # Discretize norms
    l2norm_u_h = discretize(l2norm_u, domain_h, V1h, backend=PSYDAC_BACKEND_GPYCCEL)
    l2norm_p_h = discretize(l2norm_p, domain_h, V2h, backend=PSYDAC_BACKEND_GPYCCEL)

    l2norm_du_h = discretize(l2norm_du, domain_h, V1h, backend=PSYDAC_BACKEND_GPYCCEL)
    l2norm_dp_h = discretize(l2norm_dp, domain_h, V2h, backend=PSYDAC_BACKEND_GPYCCEL)

    # First guess: zero solution
    u_h = FemField(V1h)
    p_h = FemField(V2h)

    new_u_h = FemField(V1h)
    new_p_h = FemField(V2h)

    equation_h.set_solver('direct', tol=1e-13, info=True)

    # Newton iteration
    for n in range(N):
        print('==== iteration {} ===='.format(n))
        xh, info = equation_h.solve(un=u_h, pn=p_h)

        if len(domain)>1:
            for i in range(len(domain)):
                new_u_h[i][0].coeffs[:,:] = xh[i][0].coeffs[:,:]
                new_u_h[i][1].coeffs[:,:] = xh[i][1].coeffs[:,:]
                new_p_h[i].coeffs[:,:]    = xh[i][2].coeffs[:,:]

        else:
            new_u_h[0].coeffs[:,:] = xh[0].coeffs[:,:]
            new_u_h[1].coeffs[:,:] = xh[1].coeffs[:,:]
            new_p_h.coeffs[:,:]    = xh[2].coeffs[:,:]

        # Compute L2 norm of increment
        l2_error_du = l2norm_du_h.assemble(u=(new_u_h-u_h))
        l2_error_dp = l2norm_dp_h.assemble(p=(new_p_h-p_h))

        u_h = new_u_h.copy()
        p_h = new_p_h.copy()

        print('L2_error_norm(du) = {}'.format(l2_error_du))
        print('L2_error_norm(dp) = {}'.format(l2_error_dp))

        if abs(l2_error_du)<= TOL and abs(l2_error_dp) <= TOL:
            print()
            print('CONVERGED')
            break

    l2_error_u = l2norm_u_h.assemble(u=u_h)
    l2_error_p = l2norm_p_h.assemble(p=p_h)

    return l2_error_u, l2_error_p, u_h, domain_h

###############################################################################
#            SERIAL TESTS
###############################################################################
def test_st_navier_stokes_2d():

    from sympde.topology      import IdentityMapping, PolarMapping, AffineMapping
    # ... Exact solution
    mapping_1 = IdentityMapping('M1', 2)
    mapping_2 = PolarMapping   ('M2', 2, c1 = 0., c2 = 0.5, rmin = 0., rmax=1.)
    mapping_3 = AffineMapping  ('M3', 2, c1 = 0., c2 = np.pi, a11 = -1, a22 = -1, a21 = 0, a12 = 0)

    A = Square('A',bounds1=(0.5, 1.), bounds2=(-1., 0.5))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(0, np.pi))
    C = Square('C',bounds1=(0.5, 1.), bounds2=(np.pi-0.5, np.pi + 1))

    D1     = mapping_1(A)
    D2     = mapping_2(B)
    D3     = mapping_3(C)

    D1D2      = D1.join(D2, name = 'D1D2',
                bnd_minus = D1.get_boundary(axis=1, ext=1),
                bnd_plus  = D2.get_boundary(axis=1, ext=-1))

    domain    = D1D2.join(D3, name = 'D1D2D3',
                bnd_minus = D2.get_boundary(axis=1, ext=1),
                bnd_plus  = D3.get_boundary(axis=1, ext=-1))

    boundary_h = None
    boundary   = domain.boundary
    boundary_n = domain.boundary.complement(boundary)

    x, y     = domain.coordinates
    mu = 0.007
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

    a = TerminalExpr(-mu*div(0.5*grad(ue)+0.5*Transpose(grad(ue))), domain)
    b = TerminalExpr(    grad(ue), domain)
    c = TerminalExpr(    grad(pe), domain)
    f = (a + b.T*ue + c).simplify()

    l2_error_u, l2_error_p, u_h, domain_h = run_steady_state_navier_stokes_2d(domain, f, ue, pe, mu, boundary=boundary, boundary_h=boundary_h, boundary_n=boundary_n,
                                                                    ncells=[2**4,2**4], degree=[2, 2], multiplicity=[1,1])

    print(l2_error_u, l2_error_p)
#    domains  = [domain.logical_domain.interior]
#    mappings = [domain.mapping]

#    etas, xx, yy         = get_plotting_grid({I:M for I,M in zip(domains, mappings)}, N=20)
#    grid_vals_h1         = lambda v: get_grid_vals(v, etas, mappings, space_kind='h1')
#    uh_x_vals, uh_y_vals = grid_vals_h1(u_h)
#    my_small_plot(
#        title=r'approximation of solution $u$',
#        vals=[np.sqrt(uh_x_vals**2+uh_y_vals**2)],
#        titles=[r'$|uh^(x,y)|$'],
#        xx=xx,
#        yy=yy,
#        gridlines_x1=None,
#        gridlines_x2=None,
#    )
    # Check that expected absolute error on velocity and pressure fields
    assert abs(0.007847028803941369 - l2_error_u ) < 1e-7
    assert abs(0.04955682156571245 - l2_error_p  ) < 1e-7


def test_st_navier_stokes_2d_2():
    filename = os.path.join(mesh_dir, 'multipatch/plate_with_hole_mp_7.h5')
    domain   = Domain.from_file(filename)
    x,y = domain.coordinates
    # Boundaries
    patches = domain.interior.args
    boundary_h = [patches[0].get_boundary(axis=1, ext=-1),
                  patches[1].get_boundary(axis=1, ext=-1),
                  patches[1].get_boundary(axis=1, ext=1),
                  patches[2].get_boundary(axis=1, ext=-1),
                  patches[3].get_boundary(axis=1, ext=-1),
                  patches[3].get_boundary(axis=1, ext=1),
                  patches[4].get_boundary(axis=0, ext=-1),
                  patches[4].get_boundary(axis=0, ext=1),
                  ]

    boundary = [patches[0].get_boundary(axis=1, ext=1)]
    boundary_h = Union(*boundary_h)
    boundary   = boundary[0]
    boundary_n = patches[4].get_boundary(axis=1, ext=1)
    ue         = Matrix((4*0.3*y*(0.41-y)/(0.41**2), 0))
    f          = Tuple(0,0)
    pe         = 0
    mu         = 0.001
    _, _, u_h, domain_h = run_steady_state_navier_stokes_2d(domain, f, ue, pe, mu, boundary=boundary, boundary_h=boundary_h, boundary_n=boundary_n, 
                                                      ncells=None, degree=None, multiplicity=None,filename=filename)

    domains  = domain.logical_domain.interior
    mappings = list(domain_h.mappings.values())

    etas, xx, yy         = get_plotting_grid({I:M for I,M in zip(domains, mappings)}, N=20)
    grid_vals_h1         = lambda v: get_grid_vals(v, etas, mappings, space_kind='h1')
    uh_x_vals, uh_y_vals = grid_vals_h1(u_h)
    my_small_plot(
        title=r'approximation of solution $u$',
        vals=[np.sqrt(uh_x_vals**2+uh_y_vals**2)],
        titles=[r'$|uh^(x,y)|$'],
        xx=xx,
        yy=yy,
        gridlines_x1=None,
        gridlines_x2=None,
    )
#test_st_navier_stokes_2d_2()
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
@pytest.mark.parallel
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
    fy = -mu*(uy.diff(x, 2) - uy.diff(y, 2)) + ux*uy.diff(x) + uy*uy.diff(y) + pe.diff(y)

    assert (f[0]-fx).simplify() == 0
    assert (f[1]-fy).simplify() == 0

    f  = Tuple(fx, fy)
    # ...

    # Run test

    l2_error_u, l2_error_p = run_steady_state_navier_stokes_2d(domain, f, ue, pe, ncells=[2**3,2**3], degree=[2, 2], multiplicity=[2,2])

    # Check that expected absolute error on velocity and pressure fields
    assert abs(0.00020452836013053793 - l2_error_u ) < 1e-7
    assert abs(0.004127752838826402 - l2_error_p  ) < 1e-7

##==============================================================================
## CLEAN UP SYMPY NAMESPACE
##==============================================================================

#def teardown_module():
#    from sympy.core import cache
#    cache.clear_cache()

#def teardown_function():
#    from sympy.core import cache
#    cache.clear_cache()

#------------------------------------------------------------------------------
if __name__ == '__main__':

    Tf       = 6.
    dt_h     = 0.05
    nt       = Tf//dt_h

#    filename = os.path.join(mesh_dir, 'bent_pipe.h5')
#    filename = os.path.join(mesh_dir, 'multipatch/plate_with_hole_mp.h5')
    filename = os.path.join(mesh_dir, 'multipatch/plate_with_hole_mp_7.h5')

    domain   = Domain.from_file(filename)
    x,y      = domain.coordinates

    # Boundaries
    patches = domain.interior.args
    boundary_h = [patches[0].get_boundary(axis=1, ext=-1),
                  patches[1].get_boundary(axis=1, ext=-1),
                  patches[1].get_boundary(axis=1, ext=1),
                  patches[2].get_boundary(axis=1, ext=-1),
                  patches[3].get_boundary(axis=1, ext=-1),
                  patches[3].get_boundary(axis=1, ext=1),
                  patches[4].get_boundary(axis=0, ext=-1),
                  patches[4].get_boundary(axis=0, ext=1),
                  ]

    boundary = [patches[0].get_boundary(axis=1, ext=1)]
    boundary_h = Union(*boundary_h)
    boundary   = boundary[0]
    boundary_n = patches[4].get_boundary(axis=1, ext=1)

    t          = Constant(name='t')
    ue         = Matrix((4*1.5*sin(pi*t/8)*y*(0.41-y)/(0.41**2), 0))
    f          = Tuple(0,0)
    pe         = 0
    mu         = 0.001

    solutions, p_h, domain_h = run_time_dependent_navier_stokes_2d(domain, f, ue, pe, mu,
                                                                   boundary=boundary, boundary_h=boundary_h, boundary_n=boundary_n,
                                                                   ncells=None, degree=None, multiplicity=None,
                                                                   filename=filename, dt_h=dt_h, nt=nt, scipy=False)


    domains  = domain.logical_domain.interior
    mappings = list(domain_h.mappings.values())

    etas, xx, yy         = get_plotting_grid({I:M for I,M in zip(domains, mappings)}, N=20)
    grid_vals_h1         = lambda v: get_grid_vals(v, etas, mappings, space_kind='h1')
    uh_x_vals, uh_y_vals = grid_vals_h1(solutions[-1])
    my_small_plot(
        title=r'approximation of solution $u$, $x$ component',
        vals=[uh_x_vals**2+uh_y_vals**2],
        titles=[r'$|uh^(x,y)|$'],
        xx=xx,
        yy=yy,
        gridlines_x1=None,
        gridlines_x2=None,
    )

    raise SystemExit()
    anim = animate_field(solutions, domain, mapping, res=(150,150), progress=True)
    anim.save('animated_fields_{}_{}.mp4'.format(str(Tf).replace('.','_'), str(dt_h).replace('.','_')), writer=animation.FFMpegWriter(fps=60))

