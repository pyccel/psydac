#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

# A note on the mappings used in these tests:
#
#   - 'identity_2d.h5' is the identity mapping on the unit square [0, 1] X [0, 1]
#
#   - 'collela_2d.h5' is a NURBS mapping from the unit square [0, 1]^2 to the
#      larger square [-1, 1]^2, with deformations going as sin(pi x) * sin(pi y)
#
#   - 'quarter_annulus.h5' is a NURBS transformation from the unit square [0, 1]^2
#      to the quarter annulus in the lower-left quadrant of the Cartesian plane
#      (hence both x and y are negative), with r_min = 0.5 and r_max = 1
#
#      Please note that the logical coordinates (x1, x2) correspond to the polar
#      coordinates (r, theta), but with reversed order: hence x1=theta and x2=r

import os
from pathlib import Path

from mpi4py import MPI
from sympy import pi, cos, sin, log, exp, lambdify, symbols
import pytest

from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import NormalVector
from sympde.topology import Domain
from sympde.topology import Union
from sympde.topology import Square
from sympde.expr     import linearize
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.fem.basic                        import FemField
from psydac.api.discretization               import discretize
from psydac.api.settings                     import PSYDAC_BACKEND_GPYCCEL
from psydac.feec.global_geometric_projectors import GlobalGeometricProjectorH1

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

x, y = symbols('x, y', real=True)

#------------------------------------------------------------------------------
def run_field_test(filename, f):

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++

    domain = Domain.from_file(filename)

    V  = ScalarFunctionSpace('V', domain)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    F  = element_of(V, name='F')

    # Bilinear form a: V x V --> R
    a = BilinearForm((u, v), integral(domain, u * v))

    # Linear form l: V --> R
    l = LinearForm(v, integral(domain, f * v))

    # Variational model
    equation = find(u, forall=v, lhs=a(u, v), rhs=l(v))

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, filename=filename)

    # Discrete spaces
    Vh = discretize(V, domain_h)

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Vh, Vh])

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    # Solve linear system
    # uh is the L2-projection of the analytical field "f"
    uh = equation_h.solve()

    #+++++++++++++++++++++++++++++++
    l1   = LinearForm( v, integral(domain, F*v))
    l2   = LinearForm( v, integral(domain, f*v))
    l1_h = discretize(l1, domain_h,  Vh)
    l2_h = discretize(l2, domain_h,  Vh)

    a1   = BilinearForm( (u,v), integral(domain, F*u*v))
    a2   = BilinearForm( (u,v), integral(domain, f*u*v))
    a1_h = discretize(a1, domain_h,  [Vh, Vh])
    a2_h = discretize(a2, domain_h,  [Vh, Vh])

    x1 = l1_h.assemble(F=uh)
    x2 = l2_h.assemble()

    A1 = a1_h.assemble(F=uh)
    A2 = a2_h.assemble()

    error_1 = abs((x1-x2).toarray()).max()
    error_2 = abs((A1-A2).toarray()).max()

    return error_1, error_2

#------------------------------------------------------------------------------
def run_boundary_field_test(domain, boundary, f, ncells):
    V  = ScalarFunctionSpace('V', domain)
    u  = element_of(V, name='u')
    v  = element_of(V, name='v')
    F  = element_of(V, name='F')

    nn = NormalVector('nn')

    # Bilinear form a: V x V --> R
    aF = BilinearForm((u, v), integral(boundary, F * u * v))
    af = BilinearForm((u, v), integral(boundary, f * u * v))

    a_grad_F = BilinearForm((u, v), integral(boundary, dot(grad(F), nn) * u * v))
    a_grad_f = BilinearForm((u, v), integral(boundary, dot(grad(f), nn) * u * v))

    # Linear form l: V --> R
    lF   = LinearForm( v, integral(boundary, F*v))
    lf   = LinearForm( v, integral(boundary, f*v))

    l_grad_F   = LinearForm( v, integral(boundary, dot(grad(F), nn) * v))
    l_grad_f   = LinearForm( v, integral(boundary, dot(grad(f), nn) * v))

    domain_h = discretize(domain, ncells=ncells)
    Vh = discretize(V, domain_h, degree=[3,3])

    x,y = domain.coordinates
    f_lambda = lambdify([x,y], f, 'math')
    Pi0 = GlobalGeometricProjectorH1(Vh)
    fh = Pi0(f_lambda)
    fh.coeffs.update_ghost_regions()

    aF_h = discretize(aF, domain_h, [Vh, Vh])
    af_h = discretize(af, domain_h, [Vh, Vh])
    aF_x = aF_h.assemble(F=fh)
    af_x = af_h.assemble()

    a_grad_F_h = discretize(a_grad_F, domain_h, [Vh, Vh])
    a_grad_f_h = discretize(a_grad_f, domain_h, [Vh, Vh])
    a_grad_F_x = a_grad_F_h.assemble(F=fh)
    a_grad_f_x = a_grad_f_h.assemble()

    lF_h = discretize(lF, domain_h,  Vh)
    lf_h = discretize(lf, domain_h,  Vh)
    lF_x = lF_h.assemble(F=fh)
    lf_x = lf_h.assemble()

    l_grad_F_h = discretize(l_grad_F, domain_h,  Vh)
    l_grad_f_h = discretize(l_grad_f, domain_h,  Vh)
    l_grad_F_x = l_grad_F_h.assemble(F=fh)
    l_grad_f_x = l_grad_f_h.assemble()

    error_BilinearForm = (aF_x-af_x).toarray()
    rel_error_BilinearForm = abs(error_BilinearForm).max() / abs(af_x.toarray()).max()

    error_BilinearForm_grad = (a_grad_F_x-a_grad_f_x).toarray()
    rel_error_BilinearForm_grad = abs(error_BilinearForm_grad).max() / abs(a_grad_f_x.toarray()).max()

    error_LinearForm = (lF_x-lf_x).toarray()
    rel_error_LinearForm = abs(error_LinearForm).max() / abs(lf_x.toarray()).max()

    error_LinearForm_grad = (l_grad_F_x-l_grad_f_x).toarray()
    rel_error_LinearForm_grad = abs(error_LinearForm_grad).max() / abs(l_grad_f_x.toarray()).max()

    return rel_error_BilinearForm, rel_error_BilinearForm_grad, rel_error_LinearForm, rel_error_LinearForm_grad

#------------------------------------------------------------------------------
def run_non_linear_poisson(filename, comm=None):

    # Maximum number of Newton iterations and convergence tolerance
    N = 20
    TOL = 1e-14

    # Define topological domain
    Omega = Domain.from_file(filename)

    # Method of manufactured solutions: define exact
    # solution phi_e, then compute right-hand side f
    x, y  = Omega.coordinates
    u_e = 2 * log(0.5 * (x**2 + y**2) + 0.5)

    # Define abstract model
    V = ScalarFunctionSpace('V', Omega)
    v = element_of(V, name='v')
    u = element_of(V, name='u')

    f = -2.*exp(-u)
    l = LinearForm( v, integral(Omega, dot(grad(v), grad(u)) - f*v ))

    du = element_of(V, name='du')
    dir_boundary = Omega.get_boundary(axis=0, ext=1)
    bc = EssentialBC(du, 0, dir_boundary)

    # Linearized model (for Newton iteration)
    a = linearize(l, u, trials=du)
    equation = find(du, forall=v, lhs=a(du, v), rhs=-l(v), bc=bc)

    # Define (abstract) norms
    l2norm_err = Norm(u - u_e, Omega, kind='l2')
    l2norm_du  = Norm(du     , Omega, kind='l2')

    # Create computational domain from topological domain
    Omega_h = discretize(Omega, filename=filename, comm=comm)

    # Create discrete spline space
    Vh = discretize(V, Omega_h)

    # Discretize equation (u is free parameter and must be provided later)
    equation_h = discretize(equation, Omega_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)

    # Discretize norms
    l2norm_err_h = discretize(l2norm_err, Omega_h, Vh)
    l2norm_du_h  = discretize(l2norm_du , Omega_h, Vh)

    # First guess: zero solution
    u_h = FemField(Vh)

    # Newton iteration
    for n in range(N):

        print()
        print('==== iteration {} ===='.format(n))
        du_h = equation_h.solve(u=u_h)

        # Compute L2 norm of increment
        l2_du = l2norm_du_h.assemble(du=du_h)
        print('L2_norm(du) = {}'.format(l2_du))

        if l2_du <= TOL:
            print('CONVERGED')
            break

        # update field
        u_h += du_h

    # Compute L2 error norm from solution field
    l2_error = l2norm_err_h.assemble(u=u_h)

    return l2_error

###############################################################################
#            SERIAL TESTS
###############################################################################

TOL = 1e-12

@pytest.mark.parametrize('n1', [10, 31, 42])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('ext', [-1, 1])
def test_boundary_field_identity(n1, axis, ext):

    domain = Square('domain', bounds1=(0., 0.5), bounds2=(0., 1.))
    boundary = domain.get_boundary(axis=axis, ext=ext)

    x,y = domain.coordinates
    f = (1+x)**3 + (1+y)**3

    rel_error_BilinearForm, rel_error_BilinearForm_grad, rel_error_LinearForm, rel_error_LinearForm_grad = run_boundary_field_test(domain, boundary, f, [n1,2*n1])

    assert rel_error_BilinearForm < TOL
    assert rel_error_BilinearForm_grad < TOL
    assert rel_error_LinearForm < TOL
    assert rel_error_LinearForm_grad < TOL


def test_field_identity_1():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    f        = sin(pi*x)*sin(pi*y)

    error_1, error_2 = run_field_test(filename, f)

    expected_error_1 =  4.77987181085604e-12
    expected_error_2 =  1.196388887893425e-07

    assert abs(error_1 - expected_error_1) < 1.e-7
    assert abs(error_2 - expected_error_2) < 1.e-7

#------------------------------------------------------------------------------
def test_field_identity_2():

    filename = os.path.join(mesh_dir, 'identity_2d.h5')
    f        = x*y*(x-1)*(y-1)

    error_1, error_2 = run_field_test(filename, f)

    expected_error_1 =  5.428295909559039e-11
    expected_error_2 =  2.9890068935570224e-11

    assert abs(error_1 - expected_error_1) < 1.e-10
    assert abs(error_2 - expected_error_2) < 1.e-10

#------------------------------------------------------------------------------
def test_field_collela():

    filename = os.path.join(mesh_dir, 'collela_2d.h5')
    f        = sin(pi*x)*sin(pi*y)

    error_1, error_2 = run_field_test(filename, f)

    expected_error_1 =  1.9180860719170134e-10
    expected_error_2 =  0.00010748308338081464

    assert abs(error_1 - expected_error_1) < 1.e-7
    assert abs(error_2 - expected_error_2) < 1.e-7

#------------------------------------------------------------------------------
def test_field_quarter_annulus():

    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    c        = pi / (1. - 0.5**2)
    r2       = 1. - x**2 - y**2
    f        = x*y*sin(c * r2)

    error_1, error_2 = run_field_test(filename, f)

    expected_error_1 =  1.1146377538410329e-10
    expected_error_2 =  9.18920469410037e-08

    assert abs(error_1 - expected_error_1) < 1.e-7
    assert abs(error_2 - expected_error_2) < 1.e-7

#==============================================================================
def test_nonlinear_poisson_circle():

    filename = os.path.join(mesh_dir, 'circle.h5')
    l2_error = run_non_linear_poisson(filename)

    expected_l2_error =  0.004026218710733066

    assert abs(l2_error - expected_l2_error) < 1.e-7

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

if __name__ == '__main__':
    test_field_quarter_annulus()
