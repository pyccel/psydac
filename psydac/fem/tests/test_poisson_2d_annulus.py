import numpy as np

from psydac.cad.geometry     import Geometry
from psydac.api.discretization import discretize
from psydac.ddm.cart import DomainDecomposition
from psydac.api.feec import DiscreteDerham
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.splines import SplineSpace
from psydac.fem.basic import FemField

from sympde.calculus      import grad, dot
from sympde.expr import BilinearForm, LinearForm, integral, Norm
from sympde.expr.equation import find, EssentialBC
import sympde.topology as top
from sympde.utilities.utils import plot_domain

from typing import Tuple

import sympy
from psydac.fem.tests.get_integration_function import solve_poisson_2d_annulus


import pytest

def test_boundary_condition():
    """ Test if the zero boundary condition is satisfied"""
    rmin = 1.5
    rmax = 2.7
    # Create the symbolic annulus domain
    domain  = top.Square('A',bounds1=(0, 1.), bounds2=(0, 2*np.pi))
    mapping = top.PolarMapping('M1',2, c1=0., c2=0., rmin=rmin, rmax=rmax)
    annulus : top.Domain = mapping(domain)
    ##DEBUG##
    plot_domain(domain=annulus, draw=True, isolines=True)
    print("type(annulus):", type(annulus))
    #########

    V    = top.ScalarFunctionSpace(name='V', domain=annulus, kind=None)
    # Discretize the domain
    ncells = [2**2, 2**2] # in each direction
    degree = [2, 2] # degree of what ? in each direction
    annulus_h : Geometry = discretize(annulus, ncells=ncells, 
                                      periodic=[False, True])
    Vh : TensorFemSpace = discretize(V, annulus_h, degree=degree)


    psi_h = solve_poisson_2d_annulus(annulus_h, Vh, rhs=1e-7, boundary_values=0)
    assert (psi_h(0,1) < 1e-3)
    assert (psi_h(0.4, 1.2) < 1e-5)

def test_manufactured_solution():
    """ Uses a manufactured solution so test the inhomogeneous problem"""
    
    # Create symbolic domain
    rmin = 1.5
    rmax = 2.5
    domain  = top.Square('A',bounds1=(0, 1.), bounds2=(0, 2*np.pi))
    mapping = top.PolarMapping('M1',2, c1=0., c2=0., rmin=rmin, rmax=rmax)
    annulus = mapping(domain)

    # Solution and resulting rhs
    x, y = annulus.coordinates
    solution = x**2 + y**2
    rhs = -4

    ## Compute the error norms

    # Create function space with boundary conditions
    V = top.ScalarFunctionSpace(name='V', domain=annulus, kind=None)
    u = top.element_of(V, name='u')

    # Discretize
    ncells = [2**2, 2**2] # in each direction
    degree = [2, 2] # degree of what ? in each direction
    annulus_h = discretize(annulus, ncells=ncells, periodic=[False, True])
    Vh       = discretize(V, annulus_h, degree=degree)

    error  = u - solution
    l2norm = Norm(error, annulus, kind='l2')
    h1norm = Norm(error, annulus, kind='h1')

    l2norm_h = discretize(l2norm, annulus_h, Vh)
    h1norm_h = discretize(h1norm, annulus_h, Vh)

    # Solve the Poisson problem
    psi_h = solve_poisson_2d_annulus(annulus_h, Vh, 
                                     rhs=-4, boundary_values=solution)
    l2_error = l2norm_h.assemble(u=psi_h)
    h1_error = h1norm_h.assemble(u=psi_h)
    assert l2_error < 1e-8
    assert h1_error < 1e-6

def test_boundary_values2():
    # Create symbolic domain
    rmin = 0.5
    rmax = 1.0
    domain  = top.Square('A',bounds1=(0, 1.), bounds2=(0, 2*np.pi))
    mapping = top.PolarMapping('M1',2, c1=0., c2=0., rmin=rmin, rmax=rmax)
    annulus = mapping(domain)

    # Discretize the domain and the de Rham sequence
    ncells = [2**3, 2**3] # in each direction
    degree = [2, 2] # degree of what ? in each direction
    annulus_h : Geometry = discretize(annulus, ncells=ncells, 
                                      periodic=[False, True])
    derham  = top.Derham(annulus, ["H1", "Hdiv", "L2"])    
    derham_h : DiscreteDerham = discretize(derham, annulus_h, degree=degree)
    
    x : sympy.Symbol
    y : sympy.Symbol
    x, y = annulus.coordinates
    boundary_expr = 1/(rmax**2 - rmin**2)*(x**2 + y**2 - rmin**2)  # Equals one 
        # on the exterior boundary and zero on the interior boundary
    psi_h = solve_poisson_2d_annulus(annulus_h=annulus_h, Vh=derham_h.V0, 
                                     rhs=1e-9, boundary_values=boundary_expr)

    # Evaluate psi_h
    V0h : TensorFemSpace = derham_h.V0
    x_eval = np.linspace(0, 1, 4)
    y_eval = np.linspace(0, 2*np.pi, 4)
    psi_h_eval = V0h.eval_fields([x_eval, y_eval], psi_h)
    
    assert abs(psi_h_eval[0][0,1]) < 1e-10
    assert abs(psi_h_eval[0][0,3]) < 1e-10
    assert abs(psi_h_eval[0][3,2] - 1) < 1e-10
    assert abs(psi_h_eval[0][3,3] - 1) < 1e-10



import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    test_boundary_condition()