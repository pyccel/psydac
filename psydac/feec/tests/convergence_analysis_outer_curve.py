import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.fem.basic              import FemField
from psydac.fem.tests.get_integration_function import solve_poisson_2d_annulus
from psydac.fem.vector import VectorFemSpace
from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_J_direct_annulus
from psydac.feec.tests.test_magnetostatic_pbm_annulus import _create_domain_and_derham
from psydac.linalg.utilities import array_to_psydac

import sympy
from sympde.expr.expr import Norm
import sympde.topology as top

def l2_error_biot_savart_annulus_outer_curve(N, p):
    """
    Computes L2 error of solution of the Biot-Savart problem with curve integral constraint in 2D
    (see test_magnetostatic_pbm_annulus.py for details) where the domain is an annulus 
    and the curve is the circle with radius 2 i.e. equal to the outer boundary
    """
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
    c_0 = -4*np.pi
    rhs_curve_integral = c_0 

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


    
        