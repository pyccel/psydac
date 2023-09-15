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
from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_J_direct_annulus
from psydac.feec.tests.magnetostatic_pbm_annulus import solve_magnetostatic_pbm_J_direct_with_bc
from psydac.feec.tests.test_magnetostatic_pbm_annulus import _create_domain_and_derham, compute_curve_integral_rhs
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



def l2_error_manufactured_poisson_psi(N, p):
    N1 = 8
    N2 = 8
    ncells = [N,N//2]
    annulus, derham = _create_domain_and_derham()
    annulus_h = discretize(annulus, ncells=ncells, periodic=[False, True])
    derham_h = discretize(derham, annulus_h, degree=[p,p])
    assert isinstance(derham_h, DiscreteDerham)
    
    # Compute right hand side
    x,y = sympy.symbols(names='x y')
    boundary_values_poisson = 1/3*(x**2 + y**2 - 1)  # Equals one 
        # on the exterior boundary and zero on the interior boundary
    psi_h = solve_poisson_2d_annulus(annulus_h, derham_h.V0, rhs=1e-10, 
                                     boundary_values=boundary_values_poisson)
    c_0 = 0.
    J = 4*x**2 - 12*x**2/sympy.sqrt(x**2 + y**2) + 4*y**2 - 12*y**2/sympy.sqrt(x**2 + y**2) + 8
    curve_integral_rhs = compute_curve_integral_rhs(derham, annulus, J, annulus_h, 
                                                    derham_h, psi_h, c_0)

    B_h_coeffs_arr = solve_magnetostatic_pbm_J_direct_annulus(J, psi_h, rhs_curve_integral=curve_integral_rhs,
                                                     derham_h=derham_h,
                                                     derham=derham,
                                                     annulus_h=annulus_h)
    B_h_coeffs = array_to_psydac(B_h_coeffs_arr, derham_h.V1.vector_space)
    B_h = FemField(derham_h.V1, coeffs=B_h_coeffs)

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
            l2_error_data['l2_error'][i] = l2_error_manufactured_poisson_psi(N, 2)

        # with open('l2_error_data/manufactured_poisson_psi.pkl', 'wb') as file:
        #     pickle.dump(l2_error_data, file)
        np.save('l2_error_data/manufactured_poisson_psi/degree3/n_cells.npy', l2_error_data['n_cells'])
        np.save('l2_error_data/manufactured_poisson_psi/degree3/l2_error.npy', l2_error_data['l2_error'])

    # l2_error_data = None
    # with open('l2_error_data/manufactured_poisson_psi.pkl', 'rb') as file:
    #     l2_error_data = pickle.load(file)
    # np.savetxt('l2_error_data/manufactured_poisson_psi/n_cells.csv',
    #             l2_error_data['n_cells'], delimiter='\t')
    # np.savetxt('l2_error_data/manufactured_poisson_psi/l2_error.csv',
    #            l2_error_data['l2_error'], delimiter='\t')

    # l2_error_data = {"n_cells": np.array([8,16,32,64]), "l2_error": np.zeros(4)}
    n_cells = np.load('l2_error_data/manufactured_poisson_psi/degree3/n_cells.npy')
    l2_error = np.load('l2_error_data/manufactured_poisson_psi/degree3/l2_error.npy')

    l2_error_data_array = np.column_stack((n_cells, l2_error))
    l2_error_data = pd.DataFrame(data=l2_error_data_array, columns=['n_cells', 'l2_error'])

    # l2_error_data.to_csv('l2_error_data/manufactured_poisson_psi/degree3/l2_error_data.csv')
    # l2_error_data['n_cells'] = n_cells
    # l2_error_data['l2_error'] = l2_error
    
    h = l2_error_data['n_cells']**(-1.0)
    h_squared = l2_error_data['n_cells']**(-2.0)
    h_cubed = l2_error_data['n_cells']**(-3.0)
    plt.loglog(l2_error_data['n_cells'], l2_error_data['l2_error'], label='l2_error', marker='o')
    plt.loglog(l2_error_data['n_cells'], h)
    plt.loglog(l2_error_data['n_cells'], h_squared)
    plt.loglog(l2_error_data['n_cells'], h_cubed)
    plt.legend()
    plt.show()


