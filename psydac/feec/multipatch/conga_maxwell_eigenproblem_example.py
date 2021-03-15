# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs

from sympy import pi, cos, sin, Matrix, Tuple
from sympy import symbols
from sympy import lambdify

from sympde.calculus import grad, dot, inner, rot, div, curl, cross
from sympde.topology import NormalVector
from sympde.expr import Norm

from sympde.topology import Derham
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral

from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above

from psydac.linalg.iterative_solvers import cg, pcg
from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.feec.pull_push     import push_2d_hcurl, pull_2d_hcurl

from psydac.utilities.utils    import refine_array_1d

from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, ortho_proj_Hcurl
from psydac.feec.multipatch.operators import ConformingProjection_V1
from psydac.feec.multipatch.operators import get_grid_vals_scalar, get_grid_vals_vector
from psydac.feec.multipatch.operators import get_plotting_grid, get_patch_knots_gridlines, my_small_plot

comm = MPI.COMM_WORLD


def run_maxwell_2d_eigenproblem(nb_eigs, ncells, degree):
    """
    Maxwell eigenproblem solver, see eg
    Buffa, Perugia & Warburton, The Mortar-Discontinuous Galerkin Method for the 2D Maxwell Eigenproblem JSC 2009.

    :param nb_eigs: nb of eigenmodes to be computed
    :return: eigenvalues and eigenmodes
    """

    A = Square('A',bounds1=(0., 1.), bounds2=(0., 1.))
    mapping_1 = IdentityMapping('M1',2)
    domain = mapping_1(A)

    x,y    = domain.coordinates
    mappings  = {A.interior:mapping_1}
    mappings_list = list(mappings.values())

    nquads = [d + 1 for d in degree]

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V1h = derham_h.V1
    V2h = derham_h.V2

    # Mass matrices for broken spaces (block-diagonal)
    M2 = BrokenMass(V2h, domain_h, is_scalar=True)
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    cP1 = ConformingProjection_V1(V1h, domain_h, hom_bc=True)    # todo (MCP): add option hom_bc=True for hom bc
    I1 = IdLinearOperator(V1h)

    A = 1e10*ComposedLinearOperator([I1-cP1,I1-cP1]) + ComposedLinearOperator([cP1, bD1.transpose(), M2, bD1, cP1])

    # Find eigenmodes and eigenvalues with scipy.sparse
    A = A.to_sparse_matrix()
    eigenvalues, eigenvectors = eigs(A, k=nb_eigs)

    # plotting
    etas, xx, yy = get_plotting_grid(mappings, N=20)

    for k_eig in range(nb_eigs):
        evalue  = eigenvalues[k]
        emode_c = array_to_stencil(eigenvectors[k], V1h.vector_space)
        emode = FemField(V1h, coeffs=emode_c)
        emode = cP1(emode)

        eh_x_vals, eh_y_vals = get_grid_vals_vector(emode, etas, mappings)

        my_small_plot(
            title=r'mode nb k='+repr(k),
            vals=[eh_x_vals, eh_y_vals],
            titles=[r'$e^h_{k,x}(x,y)$', r'$e^h_{k,y}(x,y)$'],
            xx=xx,
            yy=yy,
        )

if __name__ == '__main__':

    run_maxwell_2d_eigenproblem()