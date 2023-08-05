import numpy as np
import pytest
import logging
import pandas as pd
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


def l2_error_biot_savart_annulus_outer_curve(N, p):
    annulus, derham = _create_domain_and_derham()
    ncells = [N,N//2]
    annulus_h = discretize(annulus, ncells=ncells, periodic=[False, True])
    derham_h = discretize(derham, annulus_h, degree=[p,p])
    assert isinstance(derham_h, DiscreteDerham)

    x, y = sympy.symbols(names='x, y')
    boundary_values_poisson = 1/3*(x**2 + y**2 - 1)  # Equals one 
    psi_h = solve_poisson_2d_annulus(annulus_h, derham_h.V0, rhs=1e-12, 
                                     boundary_values=boundary_values_poisson)
    J = 1e-10

    # Compute right hand side of the curve integral constraint
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
    omega_gamma_h = discretize(omega_gamma, ncells=[N//2,N//2], periodic=[False, True])
    derham_gamma_h = discretize(derham_gamma, omega_gamma_h, degree=[p,p])
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

    # psi_h_gamma = h1_proj_gamma(psi)
    # psi_h_gamma_coeffs = psi_h_gamma.coeffs.toarray()
    c_0 = -4*np.pi
    rhs_curve_integral = c_0 #+ np.dot(inner_prod_J_h_vec, psi_h_gamma_coeffs)

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
        post_processor.export_to_vtk('psi_h_vtk', npts_per_cell=5, fields='psi_h')

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
        post_processor.export_to_vtk('magnetostatic_pbm_vtk', npts_per_cell=3,
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

    x, y = annulus.coordinates
    B_ex = sympy.Tuple(2.0/(x**2 + y**2)*(-y), 2.0/(x**2 + y**2)*x)
    v, _ = top.elements_of(derham.V1, names='v, _')
    error = sympy.Matrix([v[0]-B_ex[0], v[1]-B_ex[1]])
    l2_error_sym = Norm(error, annulus)
    l2_error_h_sym = discretize(l2_error_sym, annulus_h, derham_h.V1)
    l2_error = l2_error_h_sym.assemble(v=B_h)

    return l2_error

if __name__ == '__main__':
    computes_l2_errors = False
    if computes_l2_errors:
        l2_error_data = {"n_cells": np.array([8,16,32,64]), "l2_error": np.zeros(4)}
        for i,N in enumerate([8,16,32,64]):
            l2_error_data['l2_error'][i] = l2_error_biot_savart_annulus_outer_curve(N,3)
        l2_error_array = np.column_stack((l2_error_data['n_cells'], l2_error_data['l2_error']))
        l2_error_data_frame = pd.DataFrame(data=l2_error_array, columns=['n_cells', 'l2_error'])
        l2_error_data_frame.to_csv('l2_error_data/biot_savart_outer_curve/l2_error_data.csv',index=False, header=True)

    else: 
        l2_error_data = pd.read_csv('l2_error_data/biot_savart_outer_curve/l2_error_data.csv')

        l2_error_inner_curve = np.loadtxt('l2_error_data/biot_savart_annulus/l2_error.csv')
        
        h = l2_error_data['n_cells']**(-1.0)
        h_squared = l2_error_data['n_cells']**(-2.0)
        h_cubed = l2_error_data['n_cells']**(-3.0)
        plt.loglog(l2_error_data['n_cells'], l2_error_data['l2_error'], label='l2_error', marker='o')
        plt.loglog(l2_error_data['n_cells'], h, label='1/N')
        plt.loglog(l2_error_data['n_cells'], h_squared, label='1/N^2')
        plt.loglog(l2_error_data['n_cells'], h_cubed, label='1/N^3')
        plt.loglog(l2_error_data['n_cells'], l2_error_inner_curve, label='Inner curve')

        plt.legend()
        plt.show()


    
        