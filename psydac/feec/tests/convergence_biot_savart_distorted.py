import numpy as np
import pytest
import logging
import pickle
import matplotlib.pyplot as plt

from psydac.core.bsplines          import make_knots
from psydac.fem.basic              import FemField
from psydac.fem.splines            import SplineSpace
from psydac.fem.tensor             import TensorFemSpace
from psydac.feec.derivatives       import VectorCurl_2D, Divergence_2D
from psydac.feec.global_projectors import Projector_H1, Projector_Hdiv
from psydac.feec.global_projectors import projection_matrix_H1_homogeneous_bc, projection_matrix_Hdiv_homogeneous_bc 
from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_annulus, solve_magnetostatic_pbm_J_direct_annulus
from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_J_direct_with_bc
from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_distorted_annulus
from psydac.feec.tests.test_magnetostatic_pbm_annulus import _create_domain_and_derham
from psydac.feec.pull_push         import pull_2d_hdiv
from psydac.ddm.cart               import DomainDecomposition


import numpy as np
import sympy
from typing import Tuple

from sympde.topology  import Derham, Square, IdentityMapping, PolarMapping
from sympde.topology.domain import Domain, Union, Connectivity
from sympde.topology.mapping import Mapping

from psydac.feec.global_projectors import projection_matrix_Hdiv_homogeneous_bc, projection_matrix_H1_homogeneous_bc

from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.fem  import DiscreteBilinearForm, DiscreteLinearForm
from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.cad.geometry     import Geometry
from psydac.fem.basic import FemField
from psydac.fem.vector import VectorFemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.linalg.block import BlockVector
from psydac.linalg.utilities import array_to_psydac
from psydac.linalg.stencil import StencilVector

from scipy.sparse._lil import lil_matrix
from scipy.sparse._coo import coo_matrix

from sympde.calculus      import grad, dot
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr.expr import Norm
from sympde.expr.equation import find, EssentialBC
import sympde.topology as top
from sympde.utilities.utils import plot_domain

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy

from psydac.cad.geometry          import Geometry
from psydac.core.bsplines         import quadrature_grid
from psydac.fem.basic             import FemField
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace
from psydac.linalg.kron           import KroneckerLinearSolver
from psydac.linalg.block          import BlockDiagonalSolver
from psydac.utilities.quadratures import gauss_legendre

from sympde.topology.domain       import Domain

from scipy.sparse import bmat
from scipy.sparse._lil import lil_matrix
from scipy.sparse.linalg import eigs, spsolve
from scipy.sparse.linalg import inv

from psydac.fem.tests.get_integration_function import solve_poisson_2d_annulus

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

def l2_error_biot_savart_distorted(N):
    logger = logging.getLogger("test_magnetostatic_pbm_annuluslike")
    domain, derham = _create_distorted_annulus_and_derham()
    ncells = [N,N]
    domain_h = discretize(domain, ncells=ncells, periodic=[False, True])
    derham_h = discretize(derham, domain_h, degree=[2,2])
    assert isinstance(derham_h, DiscreteDerham)

    psi = lambda alpha, theta : 2*alpha if alpha <= 0.5 else 1.0
    h1_proj = Projector_H1(derham_h.V0)
    psi_h = h1_proj(psi) 
    x, y = sympy.symbols(names='x, y')
    J = 1e-10

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

    # Compute right hand side of the curve integral constraint
    logical_domain_gamma = Square(name='logical_domain_gamma', bounds1=(0,0.5), bounds2=(0,2*np.pi))
    boundary_logical_domain_gamma = Union(logical_domain_gamma.get_boundary(axis=0, ext=-1),
                                    logical_domain_gamma.get_boundary(axis=0, ext=1))
    logical_domain_gamma = Domain(name='logical_domain_gamma',
                            interiors=logical_domain_gamma.interior,
                            boundaries=boundary_logical_domain_gamma,
                            dim=2)
    distorted_polar_mapping = DistortedPolarMapping(name='polar_mapping', dim=2)
    omega_gamma = distorted_polar_mapping(logical_domain_gamma)
    derham_gamma = Derham(domain=omega_gamma, sequence=['H1', 'Hdiv', 'L2'])
    omega_gamma_h = discretize(omega_gamma, ncells=[N//2,N], periodic=[False, True])
    derham_gamma_h = discretize(derham_gamma, omega_gamma_h, degree=[2,2])
    h1_proj_gamma = Projector_H1(derham_gamma_h.V0)
    assert isinstance(derham_h, DiscreteDerham)

    sigma, tau = top.elements_of(derham_gamma.V0, names='sigma tau')
    inner_prod_J = LinearForm(tau, integral(omega_gamma, J*tau))
    inner_prod_J_h = discretize(inner_prod_J, omega_gamma_h, space=derham_gamma_h.V0)
    assert isinstance(inner_prod_J_h, DiscreteLinearForm)
    inner_prod_J_h_stencil = inner_prod_J_h.assemble()
    # Try changing this to the evaluation using the dicrete linear form directly
    assert isinstance(inner_prod_J_h_stencil, StencilVector)
    inner_prod_J_h_vec = inner_prod_J_h_stencil.toarray()
    logger.debug("np.linalg.norm(inner_prod_J_h_vec):%s\n", np.linalg.norm(inner_prod_J_h_vec))

    psi_h_gamma = h1_proj_gamma(psi)
    psi_h_gamma_coeffs = psi_h_gamma.coeffs.toarray()
    c_0 = -4*np.pi
    rhs_curve_integral = c_0 + np.dot(inner_prod_J_h_vec, psi_h_gamma_coeffs)
    logger.debug("rhs_curve_integral + 4*np.pi:%s\n", rhs_curve_integral + 4*np.pi)

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
        post_processor.export_to_vtk('magnetostatic_pbm_distorted_annulus_vtk', npts_per_cell=3,
                                        fields=("B_h", "B_exact"))


    # eval_grid = [np.array([0.25, 0.5, 0.75]), np.array([np.pi/2, np.pi])]
    # V1h = derham_h.V1
    # assert isinstance(V1h, VectorFemSpace)
    # B_h_eval = V1h.eval_fields(eval_grid, B_h)
    # B_exact_logical = pull_2d_hdiv(B_exact, distorted_polar_mapping.get_callable_mapping())
    # print(B_h_eval)
    # assert abs( B_h_eval[0][0][0,1] - B_exact_logical[0](eval_grid[0][0], eval_grid[1][1])) < 0.01, f"B_h_eval[0][0][0,1] - B_exact[0](eval_grid[0,0], eval_grid[1,1]):{B_h_eval[0][0][0,1] - B_exact_logical[0](eval_grid[0][0], eval_grid[1][1])}"
    # assert abs(B_h_eval[0][1][0,1] - B_exact_logical[1](eval_grid[0][0], eval_grid[1][1])) < 0.01, f"B_h_eval[0][1][0,1] - B_exact[1](eval_grid[0,0], eval_grid[1,1]):{B_h_eval[0][1][0,1] - B_exact_logical[1](eval_grid[0][0], eval_grid[1][1])}"
    # assert abs( B_h_eval[0][1][2,0] - B_exact_logical[1](eval_grid[0][2], eval_grid[1][0])) < 0.01, f"B_h_eval[0][0][2,0] - B_exact[1](eval_grid[0,2], eval_grid[1,0]):{B_h_eval[0][1][2,0] - B_exact_logical[1](eval_grid[0][2], eval_grid[1][0])}"

    x, y = domain.coordinates
    B_ex = sympy.Tuple(2.0/(x**2 + y**2)*(-y), 2.0/(x**2 + y**2)*x)
    v, _ = top.elements_of(derham.V1, names='v, _')
    error = sympy.Matrix([v[0]-B_ex[0], v[1]-B_ex[1]])
    l2_error_symbolic = Norm(error, domain)
    l2_error_h_symbolic = discretize(l2_error_symbolic, domain_h, derham_h.V1)
    l2_error = l2_error_h_symbolic.assemble(v=B_h)

    return l2_error


if __name__ == '__main__':
    computes_l2_errors = True
    if computes_l2_errors:
        l2_error_data = {"n_cells": np.array([8,16,32,64,128]), "l2_error": np.zeros(5)}
        for i,N in enumerate([8,16,32,64,128]):
            l2_error_data['l2_error'][i] = l2_error_biot_savart_distorted(N)

        with open('l2_error_data/biot_savart_distorted.pkl', 'wb') as file:
            pickle.dump(l2_error_data, file)

    else: 
        l2_error_data = None
        with open('l2_error_data/biot_savart_distorted.pkl', 'rb') as file:
            l2_error_data = pickle.load(file)
        
        h = l2_error_data['n_cells']**(-1.0)
        h_squared = l2_error_data['n_cells']**(-2.0)
        h_cubed = l2_error_data['n_cells']**(-3.0)
        plt.loglog(l2_error_data['n_cells'], l2_error_data['l2_error'], label='l2_error')
        plt.loglog(l2_error_data['n_cells'], h)
        plt.loglog(l2_error_data['n_cells'], h_squared)
        plt.loglog(l2_error_data['n_cells'], h_cubed)
        plt.legend()
        plt.show()
