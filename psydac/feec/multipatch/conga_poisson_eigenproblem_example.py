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
from psydac.feec.multipatch.operators import ConformingProjection_V0
from psydac.feec.multipatch.operators import get_grid_vals_scalar, get_grid_vals_vector
from psydac.feec.multipatch.operators import get_plotting_grid, get_patch_knots_gridlines, my_small_plot

comm = MPI.COMM_WORLD


def run_poisson_2d_eigenproblem(nb_eigs, ncells, degree):
    """
    Poisson (ie, Laplace) eigenproblem solver

    :param nb_eigs: nb of eigenmodes to be computed
    :return: eigenvalues and eigenmodes
    """

    # OmegaLog1 = Square('OmegaLog1',bounds1=(0., 0.5), bounds2=(0., 1.))
    # OmegaLog2 = Square('OmegaLog2',bounds1=(0.5, 1.), bounds2=(0., 1.))
    OmegaLog1 = Square('OmegaLog1',bounds1=(0., 1.), bounds2=(0., 0.5))
    OmegaLog2 = Square('OmegaLog2',bounds1=(0., 1.), bounds2=(0.5, 1.))
    mapping_1 = IdentityMapping('M1',2)
    mapping_2 = IdentityMapping('M2',2)
    domain_1     = mapping_1(OmegaLog1)
    domain_2     = mapping_2(OmegaLog2)

    domain = domain_1.join(domain_2, name = 'domain',
                bnd_minus = domain_1.get_boundary(axis=1, ext=1),
                bnd_plus  = domain_2.get_boundary(axis=1, ext=-1))

    x,y    = domain.coordinates

    mappings  = {OmegaLog1.interior:mapping_1, OmegaLog2.interior:mapping_2}
    mappings_list = list(mappings.values())

    nquads = [d + 1 for d in degree]

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1
    # V2h = derham_h.V2

    # Mass matrices for broken spaces (block-diagonal)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)
    bD0, bD1 = derham_h.broken_derivatives_as_operators

    cP0 = ConformingProjection_V0(V0h, domain_h, hom_bc=True)
    I0 = IdLinearOperator(V0h)

    A = ComposedLinearOperator([I0-cP0,I0-cP0]) + ComposedLinearOperator([cP0, bD0.transpose(), M1, bD0, cP0])

    # Find eigenmodes and eigenvalues with scipy.sparse
    A = A.to_sparse_matrix()
    eigenvalues, eigenvectors = eigs(A, k=nb_eigs, which='SM' )   # 'SM' = smallest magnitude

    print(type(eigenvalues))
    print(type(eigenvectors))
    print(eigenvectors.shape)
    # plotting
    etas, xx, yy = get_plotting_grid(mappings, N=20)

    for k_eig in range(nb_eigs):
        evalue  = eigenvalues[k_eig]
        emode_c = array_to_stencil(eigenvectors[:,k_eig], V0h.vector_space)
        emode = FemField(V0h, coeffs=emode_c)
        emode = cP0(emode)

        uh_vals = get_grid_vals_scalar(emode, etas, mappings)

        my_small_plot(
            title='mode nb k='+repr(k_eig)+'  --  eigenvalue = '+repr(evalue),
            vals=[uh_vals],
            titles=[r'$u^h_{k}(x,y)$'],
            xx=xx,
            yy=yy,
        )

if __name__ == '__main__':

    run_poisson_2d_eigenproblem(nb_eigs=6, ncells=[2**4, 2**4], degree=[2,2])