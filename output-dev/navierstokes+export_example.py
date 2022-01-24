

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

from psydac.api.essential_bc   import apply_essential_bc
from psydac.api.postprocessing import OutputManager
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

def scipy_solver(M, b):
    x  = spsolve(M.tosparse().tocsr(), b.toarray())
    x  = array_to_stencil(x, b.space)
    return x,0

def psydac_solver(M, b):
    return lsmr(M, M.T, b, maxiter=10000, tol=1e-10)


def run_time_dependent_navier_stokes_2d(filename, dt_h, nt, newton_tol=1e-4, max_newton_iter=50, scipy=True):
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

    u, v = elements_of(V1, names='u, v')
    p, q = elements_of(V2, names='p, q')

    x, y  = domain.coordinates
    int_0 = lambda expr: integral(domain , expr)

    u0 = element_of(V1, name='u0')
    du = element_of(V1, name='du')
    dp = element_of(V2, name='dp')

    # time step
    dt = Constant(name='dt')

    # boundaries
    boundary_h = Union(*[domain.get_boundary(**kw) for kw in get_boundaries(3,4)])
    boundary   = Union(*[domain.get_boundary(**kw) for kw in get_boundaries(1)])
    ue         = Tuple(40*y*(0.5-y)*exp(-100*(y-0.25)**2), 0)
    bc         = [EssentialBC(du, ue, boundary), EssentialBC(du, 0, boundary_h)]

    # Reynolds number
    Re = 1e4

    # define the linearized navier stokes
    a = BilinearForm(((du,dp),(v, q)), integral(domain, dot(du,v) + dt*dot(Transpose(grad(du))*u, v) + dt*dot(Transpose(grad(u))*du, v)
                                                      + dt*Re**-1*inner(grad(du), grad(v)) - dt*div(du)*q - dt*dp*div(v) + dt*1e-10*dp*q) )
    l = LinearForm((v, q),             integral(domain, dot(u,v)-dot(u0,v) + dt*dot(Transpose(grad(u))*u, v)
                                                      + dt*Re**-1*inner(grad(u), grad(v)) - dt*div(u)*q - dt*p*div(v) + dt*1e-10*p*q) )

    # use the stokes equation to compute the initial solution
    a_stokes = BilinearForm(((du,dp),(v, q)), integral(domain, Re**-1*inner(grad(du), grad(v)) - div(du)*q - dp*div(v) + 1e-10*dp*q) )
    l_stokes = LinearForm((v, q), integral(domain, dot(v,Tuple(0,0)) ))

    equation        = find((du, dp), forall=(v, q), lhs=a((du, dp), (v, q)), rhs=l(v, q), bc=bc)
    equation_stokes = find((du, dp), forall=(v, q), lhs=a_stokes((du, dp), (v, q)), rhs=l_stokes(v, q), bc=bc)

    # Define (abstract) norms
    l2norm_du  = Norm(Matrix([du[0],du[1]]), domain, kind='l2')
    l2norm_dp  = Norm(dp     , domain, kind='l2')

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename)

    # ... discrete spaces
    V1h = discretize(V1, domain_h)
    V2h = discretize(V2, domain_h)
    Xh  = V1h*V2h

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

    #Output Management
    O = OutputManager('spaces_test.yml','fields_test.h5')
    O.add_spaces(V1h, V2h) #Xh)
    O.export_fields(0., 0, u0 = u0_h, p0 = p0_h, u = u_h, p = p_h, du = du_h, dp =dp_h)
    ts = 0

    while t<Tf:
        ts += 1
        t  += dt_h
        print()
        print('======= time {}/{} ======='.format(t,Tf),flush = True)

        u0_h[0].coeffs[:,:] = u_h[0].coeffs[:,:]
        u0_h[1].coeffs[:,:] = u_h[1].coeffs[:,:]
        p0_h.coeffs[:,:]    = p_h.coeffs[:,:]

        # Newton iteration
        for n in range(max_newton_iter):
            print()
            print('==== iteration {} ===='.format(n), flush = True)

            M = a_h.assemble(u=u_h, p=p_h, dt=dt_h)
            b = l_h.assemble(u=u_h, p=p_h, u0=u0_h, dt=dt_h)

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

            if abs(l2_error_du) <= newton_tol and abs(l2_error_dp) <= newton_tol:
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

        O.export_fields(t, ts, u0=u0_h, p0=p0_h, u=u_h, p=p_h, du=du_h, dp=dp_h)

    O.export_space_info()
    return solutions, p_h, domain, domain_h

if __name__ == '__main__':

    filename = '../mesh/bent_pipe.h5'

    Tf   = 0.30
    dt_h = 0.05
    nt   = Tf//dt_h
    solutions, p_h, domain, domain_h = run_time_dependent_navier_stokes_2d(filename, dt_h=dt_h, nt=nt, scipy=False)