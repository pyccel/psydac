import logging

from psydac.fem.basic              import FemField
from psydac.feec.global_projectors import Projector_H1, Projector_Hdiv
from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_J_direct_annulus
from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_J_direct_with_bc
from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_distorted_annulus
from psydac.feec.pull_push         import pull_2d_hdiv


import numpy as np
import sympy
from typing import Tuple

from sympde.topology  import Derham, Square, PolarMapping
from sympde.topology.domain import Domain, Union
from sympde.topology.mapping import Mapping


from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.fem  import DiscreteLinearForm
from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.fem.basic import FemField
from psydac.fem.vector import VectorFemSpace
from psydac.linalg.utilities import array_to_psydac
from psydac.linalg.stencil import StencilVector

from sympde.calculus      import grad, dot
from sympde.expr import LinearForm, integral
import sympde.topology as top
import numpy as np

from psydac.fem.basic             import FemField
from psydac.fem.vector import VectorFemSpace

from sympde.topology.domain       import Domain

from psydac.fem.tests.get_integration_function import solve_poisson_2d_annulus

def _create_domain_and_derham() -> Tuple[Domain, Derham]:
    """ Creates domain and de Rham sequence on annulus with rmin=1. and rmax=2."""
    logical_domain = Square(name='logical_domain', bounds1=(0,1), bounds2=(0,2*np.pi))
    boundary_logical_domain = Union(logical_domain.get_boundary(axis=0, ext=-1),
                                    logical_domain.get_boundary(axis=0, ext=1))
    logical_domain = Domain(name='logical_domain',
                            interiors=logical_domain.interior,
                            boundaries=boundary_logical_domain,
                            dim=2)
    polar_mapping = PolarMapping(name='polar_mapping', dim=2, c1=0., c2=0.,
                                 rmin=1.0, rmax=2.0)
    annulus = polar_mapping(logical_domain)
    derham = Derham(domain=annulus, sequence=['H1', 'Hdiv', 'L2'])
    return annulus, derham

def compute_curve_integral_rhs(derham, annulus, J, annulus_h, derham_h, 
                                psi_h, c_0):
    sigma, tau = top.elements_of(derham.V0, names='sigma tau')
    inner_prod_J = LinearForm(tau, integral(annulus, J*tau))
    inner_prod_J_h = discretize(inner_prod_J, annulus_h, space=derham_h.V0)
    assert isinstance(inner_prod_J_h, DiscreteLinearForm)
    inner_prod_J_h_stencil = inner_prod_J_h.assemble()
    # Try changing this to the evaluation using the dicrete linear form directly
    assert isinstance(inner_prod_J_h_stencil, StencilVector)
    inner_prod_J_h_vec = inner_prod_J_h_stencil.toarray()
    psi_h_coeffs = psi_h.coeffs.toarray()
    curve_integral_rhs = c_0 + np.dot(inner_prod_J_h_vec, psi_h_coeffs)
    return curve_integral_rhs

def test_solve_J_direct_annulus_with_poisson_psi():
    N1 = 8
    N2 = 8
    ncells = [N1,N2]
    annulus, derham = _create_domain_and_derham()
    annulus_h = discretize(annulus, ncells=ncells, periodic=[False, True])
    derham_h = discretize(derham, annulus_h, degree=[2,2])
    assert isinstance(derham_h, DiscreteDerham)
    
    # Compute right hand side
    x,y = sympy.symbols(names='x y')
    boundary_values_poisson = 1/3*(x**2 + y**2 - 1)  # Equals one 
        # on the exterior boundary and zero on the interior boundary
    psi_h = solve_poisson_2d_annulus(annulus_h, derham_h.V0, rhs=1e-10, 
                                     boundary_values=boundary_values_poisson)

    J = 4*x**2 - 12*x**2/sympy.sqrt(x**2 + y**2) + 4*y**2 - 12*y**2/sympy.sqrt(x**2 + y**2) + 8

    curve_integral_rhs = compute_curve_integral_rhs(derham, annulus, J, annulus_h, derham_h, psi_h, c_0=0.)

    B_h_coeffs_arr = solve_magnetostatic_pbm_J_direct_annulus(J, psi_h, rhs_curve_integral=curve_integral_rhs,
                                                     derham_h=derham_h,
                                                     derham=derham,
                                                     annulus_h=annulus_h)
    B_h_coeffs = array_to_psydac(B_h_coeffs_arr, derham_h.V1.vector_space)
    B_h = FemField(derham_h.V1, coeffs=B_h_coeffs)

    does_plot_psi = False
    if does_plot_psi:
        output_manager = OutputManager('magnetostatic_V0.yml',
                                             'psi_h.h5')
        output_manager.add_spaces(V0=derham_h.V0)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(psi_h=psi_h)
        post_processor = PostProcessManager(domain=annulus,
                                            space_file='magnetostatic_V0.yml',
                                            fields_file='psi_h.h5')
        post_processor.export_to_vtk('plot_files/manufactured_poisson_psi/psi_h_vtk', npts_per_cell=5, fields='psi_h')

    does_plot = False
    if does_plot:
        output_manager = OutputManager('spaces_magnetostatic.yml', 
                                       'fields_magnetostatic.h5')
        output_manager.add_spaces(V1=derham_h.V1)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(B_h=B_h)
        post_processor = PostProcessManager(domain=annulus, 
                                            space_file='spaces_magnetostatic.yml',
                                            fields_file='fields_magnetostatic.h5')
        post_processor.export_to_vtk('plot_files/manufactured_poisson_psi/B_h_vtk', npts_per_cell=5,
                                        fields=("B_h"))

    eval_grid = [np.array([0.25, 0.5, 0.75]), np.array([np.pi/2, np.pi])]
    V1h = derham_h.V1
    assert isinstance(V1h, VectorFemSpace)
    B_h_eval = V1h.eval_fields(eval_grid, B_h)
    assert np.linalg.norm(B_h_eval[0][0]) < 1e-5
    assert abs( B_h_eval[0][1][0,1] - (0.25-1)**2 * (0.25+1)) < 0.01
    assert abs( B_h_eval[0][1][1,0] - (0.5-1)**2 * (0.5+1)) < 0.01
    assert abs( B_h_eval[0][1][2,1] - (0.75-1)**2 * (0.75+1)) < 0.01

def test_solve_J_direct_annulus_inner_curve():
    annulus, derham = _create_domain_and_derham()

    N1 = 8
    N2 = 8
    ncells = [N1,N2]
    annulus_h = discretize(annulus, ncells=ncells, periodic=[False, True])
    derham_h = discretize(derham, annulus_h, degree=[2,2])
    assert isinstance(derham_h, DiscreteDerham)

    psi = lambda alpha, theta : 2*alpha if alpha <= 0.5 else 1.0
    h1_proj = Projector_H1(derham_h.V0)
    psi_h = h1_proj(psi) 
    x, y = sympy.symbols(names='x, y')
    J = 4*x**2 - 12*x**2/sympy.sqrt(x**2 + y**2) + 4*y**2 - 12*y**2/sympy.sqrt(x**2 + y**2) + 8
    # f = sympy.Tuple(8*y - 12*y/sympy.sqrt(x**2 + y**2), -8*x + 12*x/sympy.sqrt(x**2 + y**2))
    c_0 = -1.125*np.pi

    rhs_curve_integral = compute_rhs_inner_curve(N1, N2, psi, J, c_0)

    B_h_coeffs_arr = solve_magnetostatic_pbm_J_direct_annulus(J, psi_h=psi_h, rhs_curve_integral=rhs_curve_integral,
                                                     derham=derham,
                                                     derham_h=derham_h,
                                                     annulus_h=annulus_h)

    B_h_coeffs = array_to_psydac(B_h_coeffs_arr, derham_h.V1.vector_space)
    B_h = FemField(derham_h.V1, coeffs=B_h_coeffs)

    does_plot_psi = False
    if does_plot_psi:
        output_manager = OutputManager('magnetostatic_V0.yml',
                                             'psi_h.h5')
        output_manager.add_spaces(V0=derham_h.V0)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(psi_h=psi_h)
        post_processor = PostProcessManager(domain=annulus,
                                            space_file='magnetostatic_V0.yml',
                                            fields_file='psi_h.h5')
        post_processor.export_to_vtk('plot_files/manufactured_inner_curve/psi_h_vtk', npts_per_cell=5, fields='psi_h')

    does_plot = False
    if does_plot:
        output_manager = OutputManager('spaces_magnetostatic.yml', 
                                       'fields_magnetostatic.h5')
        output_manager.add_spaces(V1=derham_h.V1)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(B_h=B_h)
        post_processor = PostProcessManager(domain=annulus, 
                                            space_file='spaces_magnetostatic.yml',
                                            fields_file='fields_magnetostatic.h5')
        post_processor.export_to_vtk('plot_files/manufactured_inner_curve/B_h_vtk', npts_per_cell=5,
                                        fields=("B_h"))


    eval_grid = [np.array([0.25, 0.5, 0.75]), np.array([np.pi/2, np.pi])]
    V1h = derham_h.V1
    assert isinstance(V1h, VectorFemSpace)
    B_h_eval = V1h.eval_fields(eval_grid, B_h)
    print(B_h_eval)
    assert np.linalg.norm(B_h_eval[0][0]) < 1e-5
    assert abs( B_h_eval[0][1][0,1] - (0.25-1)**2 * (0.25+1)) < 0.01
    assert abs( B_h_eval[0][1][1,0] - (0.5-1)**2 * (0.5+1)) < 0.01
    assert abs( B_h_eval[0][1][2,1] - (0.75-1)**2 * (0.75+1)) < 0.01


def compute_rhs_inner_curve(N1, N2, psi, J, c_0):
    logical_domain_gamma = Square(name='logical_domain_gamma', bounds1=(0,0.5), bounds2=(0,2*np.pi))
    boundary_logical_domain_gamma = Union(logical_domain_gamma.get_boundary(axis=0, ext=-1),
                                    logical_domain_gamma.get_boundary(axis=0, ext=1))
    logical_domain_gamma = Domain(name='logical_domain_gamma',
                            interiors=logical_domain_gamma.interior,
                            boundaries=boundary_logical_domain_gamma,
                            dim=2)
    polar_mapping = PolarMapping(name='polar_mapping', dim=2, c1=0., c2=0.,
                                 rmin=1.0, rmax=2.0)
    omega_gamma = polar_mapping(logical_domain_gamma)
    derham_gamma = Derham(domain=omega_gamma, sequence=['H1', 'Hdiv', 'L2'])
    omega_gamma_h = discretize(omega_gamma, ncells=[N1//2,N2], periodic=[False, True])
    derham_gamma_h = discretize(derham_gamma, omega_gamma_h, degree=[2,2])
    h1_proj_gamma = Projector_H1(derham_gamma_h.V0)
    psi_h_gamma = h1_proj_gamma(psi)
    
    does_plot_psi_omega = False
    if does_plot_psi_omega:
        output_manager_gamma = OutputManager('V0_gamma.yml', 'psi_h_gamma.h5')
        output_manager_gamma.add_spaces(V0_gamma=derham_gamma_h.V0)
        output_manager_gamma.export_space_info()
        output_manager_gamma.set_static()
        output_manager_gamma.export_fields(psi_h_gamma=psi_h_gamma)
        post_processor_gamma = PostProcessManager(domain=omega_gamma,
                                                  space_file='V0_gamma.yml',
                                                  fields_file='psi_h_gamma.h5')
        post_processor_gamma.export_to_vtk('psi_h_gamma_vtk', npts_per_cell=5,
                                           fields=('psi_h_gamma'))

    sigma, tau = top.elements_of(derham_gamma.V0, names='sigma tau')
    inner_prod_J = LinearForm(tau, integral(omega_gamma, J*tau))
    inner_prod_J_h = discretize(inner_prod_J, omega_gamma_h, space=derham_gamma_h.V0)
    assert isinstance(inner_prod_J_h, DiscreteLinearForm)
    inner_prod_J_h_stencil = inner_prod_J_h.assemble()
    # Try changing this to the evaluation using the dicrete linear form directly
    assert isinstance(inner_prod_J_h_stencil, StencilVector)
    inner_prod_J_h_vec = inner_prod_J_h_stencil.toarray()
    psi_h_gamma_coeffs = psi_h_gamma.coeffs.toarray()
    rhs_curve_integral = c_0 + np.dot(inner_prod_J_h_vec, psi_h_gamma_coeffs)
    return rhs_curve_integral


def test_biot_savart():
    annulus, derham = _create_domain_and_derham()
    N1 = 8
    N2 = 8
    ncells = [N1,N2]
    annulus_h = discretize(annulus, ncells=ncells, periodic=[False, True])
    derham_h = discretize(derham, annulus_h, degree=[2,2])
    assert isinstance(derham_h, DiscreteDerham)

    psi = lambda alpha, theta : 2*alpha if alpha <= 0.5 else 1.0
    h1_proj = Projector_H1(derham_h.V0)
    psi_h = h1_proj(psi) 
    x, y = sympy.symbols(names='x, y')
    J = 1e-10
    c_0 = -4*np.pi
    rhs_curve_integral = compute_rhs_inner_curve(N1, N2, psi, J, c_0)

    B_h_coeffs_arr = solve_magnetostatic_pbm_J_direct_annulus(J, psi_h=psi_h, rhs_curve_integral=rhs_curve_integral,
                                                     derham=derham,
                                                     derham_h=derham_h,
                                                     annulus_h=annulus_h)
    
    B_h_coeffs = array_to_psydac(B_h_coeffs_arr, derham_h.V1.vector_space)
    B_h = FemField(derham_h.V1, coeffs=B_h_coeffs)

    does_plot_psi = True
    if does_plot_psi:
        output_manager = OutputManager('magnetostatic_V0.yml',
                                             'psi_h.h5')
        output_manager.add_spaces(V0=derham_h.V0)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(psi_h=psi_h)
        post_processor = PostProcessManager(domain=annulus,
                                            space_file='magnetostatic_V0.yml',
                                            fields_file='psi_h.h5')
        post_processor.export_to_vtk('plot_files/biot_savart_annulus/psi_h_vtk', npts_per_cell=5, fields='psi_h')

    does_plot = True
    if does_plot:
        output_manager = OutputManager('spaces_magnetostatic.yml', 
                                       'fields_magnetostatic.h5')
        output_manager.add_spaces(V1=derham_h.V1)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(B_h=B_h)
        post_processor = PostProcessManager(domain=annulus, 
                                            space_file='spaces_magnetostatic.yml',
                                            fields_file='fields_magnetostatic.h5')
        post_processor.export_to_vtk('plot_files/biot_savart_annulus/B_h_vtk', npts_per_cell=3,
                                        fields=("B_h"))
    
    eval_grid = [np.array([0.25, 0.5, 0.75]), np.array([np.pi/2, np.pi])]
    V1h = derham_h.V1
    assert isinstance(V1h, VectorFemSpace)
    B_h_eval = V1h.eval_fields(eval_grid, B_h)
    print(B_h_eval)
    assert np.linalg.norm(B_h_eval[0][0]) < 1e-5
    assert abs( B_h_eval[0][1][0,1] - 2/(1+0.25)) < 0.01
    assert abs( B_h_eval[0][1][1,0] - 2/(1+0.5)) < 0.01
    assert abs( B_h_eval[0][1][2,1] - 2/(1+0.75)) < 0.01

def test_constant_one():
    annulus, derham = _create_domain_and_derham()
    N1 = 16
    N2 = 16
    ncells = [N1,N2]
    annulus_h = discretize(annulus, ncells=ncells, periodic=[False, True])
    derham_h = discretize(derham, annulus_h, degree=[2,2])
    assert isinstance(derham_h, DiscreteDerham)

    psi = lambda alpha, theta : 2*alpha if alpha <= 0.5 else 1.0
    h1_proj = Projector_H1(derham_h.V0)
    psi_h = h1_proj(psi) 
    x, y = sympy.symbols(names='x, y')
    J = 1e-10
    c_0 = 0.

    rhs_curve_integral = compute_rhs_inner_curve(N1, N2, psi, J, c_0)

    h_div_projector = Projector_Hdiv(derham_h.V1)
    B_exact_1 = lambda x,y: 1.0
    B_exact_2 = lambda x,y: 1.0
    B_exact = (B_exact_1, B_exact_2)
    P0, P1, P2 = derham_h.projectors()
    boundary_data = P1(B_exact) # B_exact is not correct it is zero on the boundary


    B_h_coeffs_arr = solve_magnetostatic_pbm_J_direct_with_bc(J, psi_h=psi_h, rhs_curve_integral=rhs_curve_integral,
                                                     boundary_data=boundary_data,
                                                     derham=derham,
                                                     derham_h=derham_h,
                                                     annulus_h=annulus_h)
    
    B_h_coeffs = array_to_psydac(B_h_coeffs_arr, derham_h.V1.vector_space)
    B_h = FemField(derham_h.V1, coeffs=B_h_coeffs)

    does_plot = True
    if does_plot:
        output_manager = OutputManager('spaces_magnetostatic_constant_one.yml', 
                                       'fields_magnetostatic_constant_one.h5')
        output_manager.add_spaces(V1=derham_h.V1)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(B_h=B_h, B_exact=boundary_data)
        post_processor = PostProcessManager(domain=annulus, 
                                            space_file='spaces_magnetostatic_constant_one.yml',
                                            fields_file='fields_magnetostatic_constant_one.h5')
        post_processor.export_to_vtk('magnetostatic_pbm_constant_one_vtk', npts_per_cell=3,
                                        fields=("B_h", "B_exact"))


    eval_grid = [np.array([0.25, 0.5, 0.75]), np.array([np.pi/2, np.pi])]
    V1h = derham_h.V1
    assert isinstance(V1h, VectorFemSpace)
    B_h_eval = V1h.eval_fields(eval_grid, B_h)
    print(B_h_eval)
    assert abs( B_h_eval[0][0][0,1] - (0.25+1) * (np.sin(np.pi)+np.cos(np.pi))) < 0.02, f"B_h_eval[0][0][0,1] - (0.25+1) * (np.sin(np.pi)+np.cos(np.pi)):{B_h_eval[0][0][0,1] - (0.25+1) * (np.sin(np.pi)+np.cos(np.pi))}"
    assert abs( B_h_eval[0][1][0,1] - (np.cos(np.pi) -  np.sin(np.pi))) < 0.02
    assert abs( B_h_eval[0][0][2,1] - (0.75+1) * (np.sin(np.pi)+np.cos(np.pi))) < 0.03

class DistortedPolarMapping(Mapping):
    """

    Examples

    """
    _expressions = {'x': '3.0*(x1 + 1)*cos(x2)*(cos(x2)**2+1)',
                    'y': '(x1 + 1)*sin(x2)*(cos(x2)**2+1)'}

    _ldim        = 2
    _pdim        = 2

def _create_distorted_annulus_and_derham() -> Tuple[Domain, Derham]:
    logical_domain = Square(name='logical_domain', bounds1=(0,1), bounds2=(0,2*np.pi))
    boundary_logical_domain = Union(logical_domain.get_boundary(axis=0, ext=-1),
                                    logical_domain.get_boundary(axis=0, ext=1))
    logical_domain = Domain(name='logical_domain',
                            interiors=logical_domain.interior,
                            boundaries=boundary_logical_domain,
                            dim=2)
    distorted_polar_mapping = DistortedPolarMapping(name='distorted_polar_mapping', dim=2)
    domain = distorted_polar_mapping(logical_domain)
    derham = Derham(domain=domain, sequence=['H1', 'Hdiv', 'L2'])
    return domain, derham

def compute_rhs_distorted_inner_curve(N1, N2, psi, J, c_0, distorted_polar_mapping):
    logical_domain_gamma = Square(name='logical_domain_gamma', bounds1=(0,0.5), bounds2=(0,2*np.pi))
    boundary_logical_domain_gamma = Union(logical_domain_gamma.get_boundary(axis=0, ext=-1),
                                    logical_domain_gamma.get_boundary(axis=0, ext=1))
    logical_domain_gamma = Domain(name='logical_domain_gamma',
                            interiors=logical_domain_gamma.interior,
                            boundaries=boundary_logical_domain_gamma,
                            dim=2)
    omega_gamma = distorted_polar_mapping(logical_domain_gamma)
    derham_gamma = Derham(domain=omega_gamma, sequence=['H1', 'Hdiv', 'L2'])
    omega_gamma_h = discretize(omega_gamma, ncells=[N1//2,N2], periodic=[False, True])
    derham_gamma_h = discretize(derham_gamma, omega_gamma_h, degree=[2,2])
    h1_proj_gamma = Projector_H1(derham_gamma_h.V0)
    psi_h_gamma = h1_proj_gamma(psi)
    sigma, tau = top.elements_of(derham_gamma.V0, names='sigma tau')
    inner_prod_J = LinearForm(tau, integral(omega_gamma, J*tau))
    inner_prod_J_h = discretize(inner_prod_J, omega_gamma_h, space=derham_gamma_h.V0)
    assert isinstance(inner_prod_J_h, DiscreteLinearForm)
    inner_prod_J_h_stencil = inner_prod_J_h.assemble()
    # Try changing this to the evaluation using the dicrete linear form directly
    assert isinstance(inner_prod_J_h_stencil, StencilVector)
    inner_prod_J_h_vec = inner_prod_J_h_stencil.toarray()
    psi_h_gamma_coeffs = psi_h_gamma.coeffs.toarray()
    rhs_curve_integral = c_0 + np.dot(inner_prod_J_h_vec, psi_h_gamma_coeffs)
    return rhs_curve_integral


def test_magnetostatic_pbm_annuluslike():
    domain, derham = _create_distorted_annulus_and_derham()
    N1 = 16
    N2 = 16
    ncells = [16,16]
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True])
    derham_h = discretize(derham, domain_h, degree=[2,2])
    assert isinstance(derham_h, DiscreteDerham)

    psi = lambda alpha, theta : 2*alpha if alpha <= 0.5 else 1.0
    h1_proj = Projector_H1(derham_h.V0)
    psi_h = h1_proj(psi)
    x, y = sympy.symbols(names='x, y')
    J = 1e-10
    c_0 = -4*np.pi

    does_plot_psi = True
    if does_plot_psi:
        output_manager = OutputManager('magnetostatic_V0.yml',
                                             'psi_h.h5')
        output_manager.add_spaces(V0=derham_h.V0)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(psi_h=psi_h)
        post_processor = PostProcessManager(domain=domain,
                                            space_file='magnetostatic_V0.yml',
                                            fields_file='psi_h.h5')
        post_processor.export_to_vtk('psi_h_vtk', npts_per_cell=5, fields='psi_h')
    
    distorted_polar_mapping = DistortedPolarMapping(name='polar_mapping', dim=2)
    rhs_curve_integral = compute_rhs_distorted_inner_curve(N1, N2, psi, J, c_0, distorted_polar_mapping)

    h_div_projector = Projector_Hdiv(derham_h.V1)
    B_exact_1 = lambda x,y: -2*y/(x**2 + y**2) 
    B_exact_2 = lambda x,y: 2*x/(x**2 + y**2)
    B_exact = (B_exact_1, B_exact_2)
    P0, P1, P2 = derham_h.projectors()
    boundary_data = P1(B_exact) 

    B_h_coeffs_arr = solve_magnetostatic_pbm_distorted_annulus(J, psi_h=psi_h, rhs_curve_integral=rhs_curve_integral,
                                                     boundary_data=boundary_data,
                                                     derham=derham,
                                                     derham_h=derham_h,
                                                     domain_h=domain_h)
    
    B_h_coeffs = array_to_psydac(B_h_coeffs_arr, derham_h.V1.vector_space)
    B_h = FemField(derham_h.V1, coeffs=B_h_coeffs)

    does_plot = True
    if does_plot:
        output_manager = OutputManager('spaces_magnetostatic_distorted_annulus.yml', 
                                       'fields_magnetostatic_distorted_annulus.h5')
        output_manager.add_spaces(V1=derham_h.V1)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(B_h=B_h, B_exact=boundary_data)
        post_processor = PostProcessManager(domain=domain, 
                                            space_file='spaces_magnetostatic_distorted_annulus.yml',
                                            fields_file='fields_magnetostatic_distorted_annulus.h5')
        post_processor.export_to_vtk('plot_files/biot_savart_distorted/B_h', npts_per_cell=5,
                                        fields=("B_h", "B_exact"))

    does_plot_psi = True
    if does_plot_psi:
        output_manager = OutputManager('magnetostatic_V0.yml',
                                             'psi_h.h5')
        output_manager.add_spaces(V0=derham_h.V0)
        output_manager.export_space_info()
        output_manager.set_static()
        output_manager.export_fields(psi_h=psi_h)
        post_processor = PostProcessManager(domain=domain,
                                            space_file='magnetostatic_V0.yml',
                                            fields_file='psi_h.h5')
        post_processor.export_to_vtk('plot_files/biot_savart_distorted/psi_h_vtk', npts_per_cell=5, fields='psi_h')

    eval_grid = [np.array([0.25, 0.5, 0.75]), np.array([np.pi/2, np.pi])]
    V1h = derham_h.V1
    assert isinstance(V1h, VectorFemSpace)
    B_h_eval = V1h.eval_fields(eval_grid, B_h)
    B_exact_logical = pull_2d_hdiv(B_exact, distorted_polar_mapping.get_callable_mapping())
    print(B_h_eval)
    assert abs( B_h_eval[0][0][0,1] - B_exact_logical[0](eval_grid[0][0], eval_grid[1][1])) < 0.01, f"B_h_eval[0][0][0,1] - B_exact[0](eval_grid[0,0], eval_grid[1,1]):{B_h_eval[0][0][0,1] - B_exact_logical[0](eval_grid[0][0], eval_grid[1][1])}"
    assert abs(B_h_eval[0][1][0,1] - B_exact_logical[1](eval_grid[0][0], eval_grid[1][1])) < 0.01, f"B_h_eval[0][1][0,1] - B_exact[1](eval_grid[0,0], eval_grid[1,1]):{B_h_eval[0][1][0,1] - B_exact_logical[1](eval_grid[0][0], eval_grid[1][1])}"
    assert abs( B_h_eval[0][1][2,0] - B_exact_logical[1](eval_grid[0][2], eval_grid[1][0])) < 0.01, f"B_h_eval[0][0][2,0] - B_exact[1](eval_grid[0,2], eval_grid[1,0]):{B_h_eval[0][1][2,0] - B_exact_logical[1](eval_grid[0][2], eval_grid[1][0])}"



if __name__ == '__main__':
    logging.basicConfig(filename='mydebug.log', level=logging.DEBUG, filemode='w')
    # test_magnetostatic_pbm_homogeneous()
    # test_magnetostatic_pbm_manufactured()
    # test_solve_J_direct_annulus_with_poisson_psi()
    # test_solve_J_direct_annulus_inner_curve()
    # test_constant_one()
    # test_biot_savart()
    test_magnetostatic_pbm_annuluslike()