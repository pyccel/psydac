import numpy as np
import matplotlib.pyplot as plt
import sympy
import sympde.topology as top

from sympde.expr.expr import Norm

from psydac.api.discretization import discretize
from psydac.feec.tests.test_magnetostatic_pbm_annulus import _compute_solution_annulus_inner_curve


def l2_error_biot_savart_annulus(N, p):
    """
    Computes L2 error of solution of the Biot-Savart problem with curve integral constraint in 2D
    (see test_magnetostatic_pbm_annulus.py for details) where the domain is an annulus with rmin=1 and 
    rmax=2 and the curve is the circle with radius 1.5
    """
    
    derham, derham_h, annulus, annulus_h, B_h = _compute_solution_annulus_inner_curve(
                                        N1=N, N2=N//2, p=p, does_plot_psi=False,
                                        does_plot=False, J=1e-10, c_0=-4*np.pi)

    x, y = annulus.coordinates
    B_ex = sympy.Tuple(2.0/(x**2 + y**2)*(-y), 2.0/(x**2 + y**2)*x)
    v, _ = top.elements_of(derham.V1, names='v, _')
    error = sympy.Matrix([v[0]-B_ex[0], v[1]-B_ex[1]])
    l2_error_sym = Norm(error, annulus)
    l2_error_h_sym = discretize(l2_error_sym, annulus_h, derham_h.V1)
    l2_error = l2_error_h_sym.assemble(v=B_h)

    return l2_error

if __name__ == '__main__':
    computes_l2_errors = True
    if computes_l2_errors:
        l2_error_data = {"n_cells": np.array([8,16,32,64]), "l2_error": np.zeros(4)}
        for i,N in enumerate([8,16,32,64]):
            l2_error_data['l2_error'][i] = l2_error_biot_savart_annulus(N, 3)

        np.savetxt('l2_error_data/biot_savart_annulus/n_cells.csv', l2_error_data['n_cells'])
        np.savetxt('l2_error_data/biot_savart_annulus/l2_error.csv', l2_error_data['l2_error'])

    l2_error_data = {"n_cells": np.array([8,16,32,64]), "l2_error": np.zeros(4)}
    
    n_cells = np.loadtxt('l2_error_data/biot_savart_annulus/n_cells.csv')
    l2_error = np.loadtxt('l2_error_data/biot_savart_annulus/l2_error.csv')

    l2_error_data['n_cells'] = n_cells
    l2_error_data['l2_error'] = l2_error

    h = l2_error_data['n_cells']**(-1.0)
    h_squared = l2_error_data['n_cells']**(-2.0)
    h_cubed = 0.1* l2_error_data['n_cells']**(-3.0)
    plt.loglog(l2_error_data['n_cells'], l2_error_data['l2_error'], marker='o', label='l2_error')
    plt.loglog(l2_error_data['n_cells'], h)
    plt.loglog(l2_error_data['n_cells'], h_squared)
    plt.loglog(l2_error_data['n_cells'], h_cubed, label='h_cubed')
    plt.legend()
    plt.show()


    
        