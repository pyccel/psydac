import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from psydac.api.discretization import discretize
from psydac.feec.tests.test_magnetostatic_pbm_annulus import _compute_solution_annulus_inner_curve
import sympy
from sympde.expr.expr import Norm
import sympde.topology as top


def l2_error_manufactured_inner_curve(N, p):
    """
    Computes L2 error of solution of the magnetostatic problem with curve integral constraint in 2D
    (see test_magnetostatic_pbm_annulus.py for details) where the domain is an annulus 
    and the curve is the circle with radius 1 and J comes from the manufactured solution

    Parameters
    ----------
    N : int
        Number of cells in x1 direction
    p : int
        Spline degree
    """
    N1 = N
    N2 = N//2
    x, y = sympy.symbols(names='x, y')
    J = 4*x**2 - 12*x**2/sympy.sqrt(x**2 + y**2) + 4*y**2 - 12*y**2/sympy.sqrt(x**2 + y**2) + 8
    c_0 = -1.125*np.pi
    
    derham, derham_h, annulus, annulus_h, B_h = _compute_solution_annulus_inner_curve(
        N1, N2, p, does_plot_psi=False, does_plot=False, J=J, c_0=c_0
    )

    x, y = annulus.coordinates
    B_ex = sympy.Tuple((sympy.sqrt(x**2 + y**2)-2)**2 * (-y), 
                       (sympy.sqrt(x**2 + y**2)-2)**2 * x)
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
            l2_error_data['l2_error'][i] = l2_error_manufactured_inner_curve(N, 2)

        np.save('l2_error_data/manufactured_inner_curve/degree3/n_cells.npy', l2_error_data['n_cells'])
        np.save('l2_error_data/manufactured_inner_curve/degree3/l2_error.npy', l2_error_data['l2_error'])

    l2_error_data = {"n_cells": np.array([8,16,32,64]), "l2_error": np.zeros(4)}
    with open('l2_error_data/manufactured_inner_curve.pkl', 'rb') as file:
        l2_error_data = pickle.load(file)
    
    np.savetxt('l2_error_data/manufactured_inner_curve/n_cells.csv',
                l2_error_data['n_cells'], delimiter='\t')
    np.savetxt('l2_error_data/manufactured_inner_curve/l2_error.csv',
                l2_error_data['l2_error'], delimiter='\t')

    n_cells = np.load('l2_error_data/manufactured_inner_curve/degree3/n_cells.npy')
    l2_error = np.load('l2_error_data/manufactured_inner_curve/degree3/l2_error.npy')

    l2_error_array = np.column_stack((n_cells, l2_error))
    l2_error_data = pd.DataFrame(data=l2_error_array, columns=['n_cells', 'l2_error'])

    l2_error_data.to_csv('l2_error_data/manufactured_inner_curve/degree3/l2_error_data.csv',
                            sep='\t', index=False)

    h = l2_error_data['n_cells']**(-1.0)
    h_squared = l2_error_data['n_cells']**(-2.0)
    h_cubed = l2_error_data['n_cells']**(-3.0)
    plt.loglog(l2_error_data['n_cells'], l2_error_data['l2_error'], marker='o', label='l2_error')
    plt.loglog(l2_error_data['n_cells'], h)
    plt.loglog(l2_error_data['n_cells'], h_squared)
    plt.loglog(l2_error_data['n_cells'], h_cubed, label='h_cubed')
    plt.legend()
    plt.show()
