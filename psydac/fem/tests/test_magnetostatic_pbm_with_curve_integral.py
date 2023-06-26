from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.cad.geometry     import Geometry
from psydac.ddm.cart import DomainDecomposition
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.splines import SplineSpace
from psydac.fem.vector      import ProductFemSpace
from psydac.fem.basic import FemField
from psydac.feec.global_projectors import Projector_H1
from psydac.feec.derivatives import VectorCurl_2D, Divergence_2D
from psydac.feec.multipatch.operators import HodgeOperator
from psydac.feec.multipatch.plotting_utilities import plot_field

from sympde.calculus      import grad, dot
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr.equation import find, EssentialBC
import sympde.topology as top
from sympde.topology.domain import Domain
from sympde.utilities.utils import plot_domain

from typing import Tuple, Callable

import sympy
import numpy as np


import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

from psydac.linalg.block    import BlockVector, BlockLinearOperator
from psydac.fem.tensor      import TensorFemSpace
from psydac.linalg.identity import IdentityStencilMatrix, IdentityMatrix
#from psydac.linalg.basic    import IdentityOperator
from psydac.fem.basic       import FemField
from psydac.linalg.basic    import LinearOperator
from psydac.ddm.cart        import DomainDecomposition, CartDecomposition

from psydac.fem.tests.get_integration_function import solve_poisson_2d_annulus
from psydac.fem.tests.magnetostatic_pbm_curve_integral import solve_magnetostatic_pbm_curve_integral, CurveIntegralData

def test_zero_solution():
    rmin = 1.
    rmax = 2.
    # Create the symbolic annulus domain
    domain  = top.Square('A',bounds1=(0, 1.), bounds2=(0, 2*np.pi))
    mapping = top.PolarMapping('M1',2, c1=0., c2=0., rmin=rmin, rmax=rmax)
    annulus : top.Domain = mapping(domain)

    # Discretize the domain and the de Rham sequence
    ncells = [2**2, 2**2] # in each direction
    degree = [2, 2] # degree of what ? in each direction
    annulus_h : Geometry = discretize(annulus, ncells=ncells, 
                                      periodic=[False, True])
    derham  = top.Derham(annulus, ["H1", "Hdiv", "L2"])    
    derham_h : DiscreteDerham = discretize(derham, annulus_h, degree=degree) #checked
    
    # Compute the scalar function for the curve integral
    x : sympy.Symbol
    y : sympy.Symbol
    x, y = domain.coordinates
    boundary_expr = 1/(rmax**2 - rmin**2)*(x**2 + y**2 - rmin**2)  # Equals one 
        # on the exterior boundary and zero on the interior boundary
    psi_h = solve_poisson_2d_annulus(annulus_h=annulus_h, Vh=derham_h.V0, 
                                     rhs=1e-9, boundary_values=boundary_expr)
    curve_integral_data = CurveIntegralData(c_0=0., curve_integral_function=psi_h)


    J = lambda x_1, x_2: 0
    B_h = solve_magnetostatic_pbm_curve_integral(J, 
                                        curve_integral_data=curve_integral_data,
                                        domain_h=annulus_h,
                                        derham_h=derham_h)
    
    # Compute values of B_h at some points
    V1h = B_h.space
    assert isinstance(V1h, ProductFemSpace)
    grid = [np.array([0.05, 0.3]), np.array([0.5, 1.3])] # values of first 
        # logical coordinate in the first array and second logical coordinate 
        # in the second array
    B_h_eval : list[ Tuple[np.ndarray] ]  = V1h.eval_fields(grid, B_h)
    # B_h_eval[i][j][k,l] = the evaluation of the j-th component of the
    # i-th field at the node (x_k, y_l)

    assert abs( B_h_eval[0][0][0,0] ) < 1e-6
    assert abs( B_h_eval[0][1][0,0] ) < 1e-6

    assert abs( B_h_eval[0][0][1,1] ) < 1e-6
    assert abs( B_h_eval[0][1][1,1] ) < 1e-6


def test_magnetostatic_problem_manufactured_sol():
    rmin = 0.5
    rmax = 1.0
    # Create the symbolic annulus domain
    domain  = top.Square('A',bounds1=(0, 1.), bounds2=(0, 2*np.pi))
    mapping = top.PolarMapping('M1',2, c1=0., c2=0., rmin=rmin, rmax=rmax)
    annulus : top.Domain = mapping(domain)

    # Discretize the domain and the de Rham sequence
    ncells = [2**3, 2**3] # in each direction
    degree = [2, 2] # degree of what ? in each direction
    annulus_h : Geometry = discretize(annulus, ncells=ncells, 
                                      periodic=[False, True])
    derham  = top.Derham(annulus, ["H1", "Hdiv", "L2"])    
    derham_h : DiscreteDerham = discretize(derham, annulus_h, degree=degree)

    # Compute the scalar function for the curve integral
    x : sympy.Symbol
    y : sympy.Symbol
    x, y = annulus.coordinates
    boundary_expr = 1/(rmax**2 - rmin**2)*(x**2 + y**2 - rmin**2)  # Equals one 
        # on the exterior boundary and zero on the interior boundary
    psi_h = solve_poisson_2d_annulus(annulus_h=annulus_h, Vh=derham_h.V0, 
                                     rhs=1e-9, boundary_values=boundary_expr)
    V0h : TensorFemSpace = derham_h.V0
    curve_integral_data = CurveIntegralData(c_0=4*np.pi, curve_integral_function=psi_h)

    B_h = solve_magnetostatic_pbm_curve_integral(J=-4., 
                                        curve_integral_data=curve_integral_data,
                                        domain_h=annulus_h,
                                        derham_h=derham_h)

    ###DEBUG###
    print("type(annulus):", type(annulus))
    print("annulus.mapping:", annulus.mapping)
    print("annulus_h.mappings:", annulus_h.mappings)
    # annulus_h.export("annulus.h5")
    output_manager = OutputManager("spaces.yaml", "fields.h5")
    output_manager.add_spaces(V1h = derham_h.V1)
    output_manager.export_space_info()
    output_manager.set_static()
    output_manager.export_fields(B_h=B_h)
    post_process_manager = PostProcessManager(domain=annulus, 
            space_file="spaces.yaml", fields_file="fields.h5")
    post_process_manager.export_to_vtk("magnetic_field_vtk", npts_per_cell=3,
                                       fields="B_h")
    ###########


    # Compute values of B_h at some points
    V1h = B_h.space
    assert isinstance(V1h, ProductFemSpace)
    grid = [np.array([0.05, 0.3]), np.array([0.5, 1.3])] # values of first 
        # logical coordinate in the first array and second logical coordinate 
        # in the second array
    B_h_eval : list[ Tuple[np.ndarray] ]  = V1h.eval_fields(grid, B_h)
    # B_h_eval[i][j][k,l] = the evaluation of the j-th component of the
    # i-th field at the node (x_k, y_l) # checked

    ###DEBUG##
    print("B_h_eval:", B_h_eval)
    ##########

    assert abs(B_h_eval[0][0][0,0] - 2*0.5) < 0.01
    assert abs(B_h_eval[0][1][0,0] + 2*0.05) < 0.01
    assert abs(B_h_eval[0][0][1,1] - 2*1.3) < 0.01
    assert abs(B_h_eval[0][1][1,1] + 2*0.3) < 0.01


if __name__ == "__main__":
    test_zero_solution()