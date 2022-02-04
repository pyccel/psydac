import os
import pytest
import pyevtk
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
from psydac.cad.geometry       import refine_knots
from psydac.fem.basic          import FemField
from psydac.fem.vector         import ProductFemSpace
from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil
from psydac.linalg.stencil     import *
from psydac.linalg.block       import *
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
from psydac.utilities.utils    import refine_array_1d, animate_field
from psydac.linalg.iterative_solvers import cg, pcg, bicg, lsmr
from psydac.api.postprocessing import OutputManager, PostProcessManager

from mpi4py import MPI


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


# ==============================================================================
def run_time_dependent_navier_stokes_2d(geometry_file, dt_h, Tf,
        newton_tol=1e-4, max_newton_iter=50, scipy=True):
    """
        Run simulation: solve time-dependent Navier Stokes' equations in a 2d domain.
        this example was taken from the pyiga library
        https://github.com/c-f-h/pyiga/blob/master/notebooks/solve-navier-stokes.ipynb
    """

    # SymPDE multipatch domain
    domain = Domain.from_file(geometry_file)

    # ... abstract model
    V1 = VectorFunctionSpace('V1', domain, kind='H1')
    V2 = ScalarFunctionSpace('V2', domain, kind='L2')
    X  = ProductSpace(V1, V2)

    u, v = elements_of(V1, names='u, v')
    p, q = elements_of(V2, names='p, q')

    x, y  = domain.coordinates
    int_0 = lambda expr: integral(domain, expr)

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
    domain_h = discretize(domain, filename=geometry_file)

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

    u0_h = FemField(V1h)
    p0_h = FemField(V2h)

    u_h  = FemField(V1h)
    p_h  = FemField(V2h)

    du_h = FemField(V1h)
    dp_h = FemField(V2h)

    # compute the initial solution
    x0 = equation_stokes_h.solve(solver='bicg', tol=1e-15)

    # First guess
    u_h[0].coeffs[:,:] = x0[0].coeffs[:,:]
    u_h[1].coeffs[:,:] = x0[1].coeffs[:,:]
    p_h.coeffs[:,:]    = x0[2].coeffs[:,:]

    t  = 0.0 # time value
    ts = 0   # time step number

    # select linear solver
    solver = scipy_solver if scipy else psydac_solver

    # =============================================
    # OutputManager init
    # =============================================
    Om = OutputManager('spaces_nv.yml', 'fields_nv.h5')
    Om.add_spaces(V1h, V2h)

    mapping = domain_h.mappings['patch_0']

    mesh_coords = mapping.build_mesh(refine_factor=0)
    x_mesh = np.ascontiguousarray(mesh_coords[..., 0:1])
    y_mesh = np.ascontiguousarray(mesh_coords[..., 1:])
    z_mesh = np.zeros_like(x_mesh)

    uh_grids = []
    ph_grids = []

    # =========================================================================
    # Solving
    # =========================================================================
    while t < Tf:

        t  += dt_h
        ts += 1

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
            b = l_h.assemble(u=u_h, p=p_h, u0=u0_h, dt=dt_h)

            apply_essential_bc(M, *equation_h.bc, identity=True)
            apply_essential_bc(b, *equation_h.bc)

            x, info = solver(M, b)
            print(info)

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
                break

            # update field
            u_h -= du_h
            p_h -= dp_h

        # ------------------------------------------------
        # OUTPUT TO FILE
        # ------------------------------------------------
        Om.add_snapshot(t, ts).export_fields(u=u_h, p=p_h)

        # ------------------------------------------------
        # OUTPUT TO VTK
        # ------------------------------------------------
        ph_grid = V2h.eval_fields(p_h, refine_factor=1)
        uh_grid_x = V1h.spaces[0].eval_fields(u_h.fields[0], refine_factor=1)
        uh_grid_y = V1h.spaces[1].eval_fields(u_h.fields[1], refine_factor=1)
        uh_grid_z = np.zeros_like(uh_grid_x)

        ph_grids.append(ph_grid)
        uh_grids.append((uh_grid_x, uh_grid_y))
        # pyevtk.hl.gridToVTK(f'bent_pipe_nv_{ts:0>4}', x_mesh, y_mesh, z_mesh,
        #                   pointData={'Pressure': ph_grid,
        #                               'velocity': (uh_grid_x, uh_grid_y, uh_grid_z)})

    Om.export_space_info()

    Pm = PostProcessManager(geometry_file, 'spaces_nv.yml', 'fields_nv.h5')
    Pm.reconstruct_scope()

    fields_dict = Pm.fields
    spaces_dict = Pm.spaces

    for k,v in fields_dict.items():
        if k!='static':
            for k_field, field in v['fields'].items():

                print(f'snapshot {k}, field {k_field}')
                if k_field == 'p':
                    new_p_grid = field.space.eval_fields(field, refine_factor=1)
                    print(np.allclose(new_p_grid,ph_grids[k]))
                elif k_field == 'u':  # Because vector Space
                    new_u_grid_x = field.space.spaces[0].eval_fields(field.fields[0], refine_factor=1)
                    new_u_grid_y = field.space.spaces[1].eval_fields(field.fields[1], refine_factor=1)
                    print(f'x : {np.allclose(new_u_grid_x,uh_grids[k][0])}')
                    print(f'y : {np.allclose(new_u_grid_y, uh_grids[k][1])}')


def teardown_module():
    from sympy.core import cache
    cache.clear_cache()


def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

#==============================================================================
# PARSER
#==============================================================================
def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description = "Solve the time-dependent Navier-Stokes equations in a 2D bent pipe."
    )

    parser.add_argument('--dt',
        type = float,
        help = 'Time step size (constant)',
        default = 0.05,
    )

    parser.add_argument('--tend',
        type = float,
        help = 'Final simulation time',
        default = 0.2,
    )

    parser.add_argument('-o', '--output_dir',
        type = str,
        help = 'Output directory',
        default = 'ns_output'
    )

    # Read input arguments
    return parser.parse_args()

if __name__ == '__main__':

    # Read input parameters from terminal
    args = parse_args()

    # ... Imports (after parser to speed up help message)
    import os
    import sys
    import glob
    import shutil
    import subprocess
    from pathlib  import Path
    from datetime import datetime, timezone

    from matplotlib import animation

    import psydac
    #...

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO: make sure that script works in parallel!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Path to geometry file
    geometry_file = Path(psydac.__path__[0] + '/../mesh/bent_pipe.h5').resolve()

    # Get time and date
    timestamp = datetime.now(timezone.utc).astimezone() \
                        .strftime("%Y-%m-%d %H:%M:%S UTC%z")

    # Path to current Python interpreter
    python_exe = Path(sys.executable).resolve()

    # ... Get information about Psydac library using pip
    cmd = [python_exe, *'-m pip freeze'.split()]
    pip = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    cmd = ['grep', 'psydac']
    grep = subprocess.Popen(cmd, stdin=pip.stdout, stdout=subprocess.PIPE, encoding='utf-8')

    pip.stdout.close()
    psydac_lib, _ = grep.communicate()
    psydac_lib    = psydac_lib.strip()
    # ...

    # Command used to run the script
    run_cmd = ' '.join(sys.argv)

    # Create output directory (provided path is relative to this file)
    output_dir = (Path(__file__).parent / args.output_dir).resolve()
    os.makedirs(output_dir)

    # Save general information to file

    print('time    :', timestamp)
    print('psydac  :', psydac_lib)
    print('python  :', python_exe)
    print('command :', run_cmd)
    print('geometry:', geometry_file)

    # Copy current script to output directory
    shutil.copy2(__file__, output_dir)

    # Copy geometry file to output directory
    shutil.copy2(geometry_file, output_dir)

    # Run simulation in output directory
    curdir = Path('.').resolve()
    os.chdir(output_dir)
    try:
        run_time_dependent_navier_stokes_2d(
            geometry_file,
            dt_h=args.dt,
            Tf=args.tend,
            scipy=True
        )
    finally:
        os.chdir(curdir)
