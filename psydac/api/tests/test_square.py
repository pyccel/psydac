# -*- coding: UTF-8 -*-

import os
import pytest
import numpy as np
from sympy import pi, cos, sin, sqrt, exp, ImmutableDenseMatrix as Matrix, Tuple, lambdify
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres as sp_gmres
from scipy.sparse.linalg import minres as sp_minres
from scipy.sparse.linalg import cg as sp_cg
from scipy.sparse.linalg import bicg as sp_bicg
from scipy.sparse.linalg import bicgstab as sp_bicgstab

from sympde.calculus import grad, dot, inner, div, curl, cross
from sympde.calculus import Transpose, laplace
from sympde.calculus import minus, plus
from sympde.topology import NormalVector
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Domain, Square, Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC
from sympde.core     import Constant
from sympde.expr     import TerminalExpr
from sympde.expr     import linearize


from psydac.api.essential_bc   import apply_essential_bc
from psydac.fem.basic          import FemField
from psydac.fem.vector         import ProductFemSpace
from psydac.core.bsplines      import make_knots
from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil
from psydac.linalg.stencil     import *
from psydac.linalg.block       import *
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
from psydac.utilities.utils    import refine_array_1d, animate_field, split_space, split_field
from psydac.linalg.iterative_solvers import cg, pcg, bicg, lsmr

from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_grid_vals
from psydac.feec.multipatch.plotting_utilities import get_patch_knots_gridlines, my_small_plot

import matplotlib.pyplot as plt
from matplotlib import animation
from time       import time

from mpi4py import MPI

#==============================================================================
def run_test():

    filename = 'square_t.h5'
    domain   = Domain.from_file(filename)
    # ... abstract model
    V = VectorFunctionSpace('V1', domain, kind='H1')


    u, v = elements_of(V, names='u, v')

    nn = NormalVector("nn")

    grad_s = lambda u:grad(u)+Transpose(grad(u))
#    a = BilinearForm((u,v), integral(domain.get_boundary(axis=0, ext=-1), dot(grad(v[0]),nn)*u[0] + dot(grad(u[0]),nn)*v[0] + dot(grad(v[1]),nn)*u[1] + dot(grad(u[1]),nn)*v[1] ) )
    a = BilinearForm((u,v), integral(domain.boundary, dot(Transpose(grad(u))*nn, v) + dot(Transpose(grad(v))*nn, u) -10*3*dot(u,v)))
                                                       

    domain_h = discretize(domain, filename=filename)
    Vh       = discretize(V, domain_h)

    # ... discretize the equation
    a_h  = discretize(a, domain_h, [Vh, Vh])
    A = a_h.assemble()
    A = A.toarray(order='F')
    np.savetxt('myfile_2.txt', A, fmt='%.17f')

run_test()

