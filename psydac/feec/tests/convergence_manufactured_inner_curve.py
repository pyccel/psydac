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
from psydac.feec.tests.test_magnetostatic_pbm_annulus import (_create_domain_and_derham, 
                                                              _compute_solution_annulus_inner_curve,
)
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



def l2_error_manufactured_inner_curve(N, p):
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
