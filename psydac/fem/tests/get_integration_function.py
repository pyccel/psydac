from psydac.api.discretization import discretize
from psydac.cad.geometry     import Geometry
from psydac.ddm.cart import DomainDecomposition
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.splines import SplineSpace
from psydac.fem.basic import FemField


from sympde.calculus      import grad, dot
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr.equation import find, EssentialBC
import sympde.topology as top
from sympde.utilities.utils import plot_domain

from typing import Tuple

import sympy
import numpy as np



def solve_poisson_2d_annulus(annulus_h : Geometry, Vh : TensorFemSpace, 
                             rhs : sympy.Expr, 
                             boundary_values : sympy.Expr) -> FemField:
    """
    Solves the Poisson equation on an annulus with dirichlet boundary
    conditions on the exterior and interior boundary
    
    Parameters
    ----------
    annulus_h : Geometry
        Discretized annulus domain
    Vh : TensorFemSpace
        Discretized function space using tensor product splines
    rhs : sympy.Expr
        Symbolic expression of the right hand side of 
        the Poisson equation i.e. the source term
    boundary_values : sympy.Expr
        Symbolic expression of the function that interpolates the boundary
        conditions
    
    Returns
    -------
    solution : FemField
        The approximated solution of the Poisson equation on the annulus
    """

    if not isinstance(Vh, TensorFemSpace):
        raise ValueError("FEM space for solving Laplace"
                         "equation should be TensorFemSpace")    
        
    annulus = annulus_h.domain

    # Create symbolic equation
    V = Vh.symbolic_space
    u, v = top.elements_of(V, names='u, v')
    a = BilinearForm((u,v), integral(annulus, dot(grad(u),grad(v))))
    l = LinearForm(v, integral(annulus, rhs*v))
    boundary_dirichlet = top.Union(*[annulus.get_boundary(axis=0, ext=-1), 
                        annulus.get_boundary(axis=0, ext=1)])
    bc   = EssentialBC(u, rhs=boundary_values, boundary=boundary_dirichlet) 
    # In EssentialBC lhs is the function, rhs are the values on boundary
    equation = find(u, forall=v, lhs=a(u,v), rhs=l(v), bc=bc) 

    equation_h = discretize(equation, annulus_h, [Vh, Vh])

    uh : FemField = equation_h.solve()
    return uh

