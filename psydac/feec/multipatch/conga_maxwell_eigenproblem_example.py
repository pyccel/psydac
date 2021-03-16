# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigsh

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
from psydac.feec.multipatch.operators import ConformingProjection_V0, ConformingProjection_V1
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
    V2h = derham_h.V2

    # Mass matrices for broken spaces (block-diagonal)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)
    M2 = BrokenMass(V2h, domain_h, is_scalar=True)
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    cP0 = ConformingProjection_V0(V0h, domain_h, hom_bc=True)
    cP1 = ConformingProjection_V1(V1h, domain_h, hom_bc=True)
    D0 = ComposedLinearOperator([bD0,cP0])
    D0_t = ComposedLinearOperator([cP0, bD0.transpose()])
    D1 = ComposedLinearOperator([bD1,cP1])
    D1_t = ComposedLinearOperator([cP1, bD1.transpose()])
    I1 = IdLinearOperator(V1h)

    # A = ComposedLinearOperator([I1-cP1,I1-cP1]) + ComposedLinearOperator([cP1, bD1.transpose(), M2, bD1, cP1])
    A = ( ComposedLinearOperator([D1_t, M2, D1])
        + 1000*ComposedLinearOperator([I1-cP1,M1, I1-cP1])
        + 1000*ComposedLinearOperator([M1, D0, D0_t, M1])
        )

    # Find eigenmodes and eigenvalues with scipy.sparse.eigsh (symmetric matrices)
    A = A.to_sparse_matrix()
    M1 = M1.to_sparse_matrix()
    eigenvalues, eigenvectors = eigsh(A, k=nb_eigs, M=M1, sigma=4.5)

    # plotting
    etas, xx, yy = get_plotting_grid(mappings, N=20)

    for k_eig in range(nb_eigs):
        evalue  = eigenvalues[k_eig]
        emode_sp = np.real(eigenvectors[:,k_eig])
        # normalize mode in L2
        Me = M1.dot(emode_sp)
        norm_emode = np.dot(emode_sp,Me)
        emode_c = array_to_stencil(emode_sp/norm_emode, V1h.vector_space)
        emode = FemField(V1h, coeffs=emode_c)

        cP_emode = cP1(emode)
        curl_emode = D1(emode)

        eh_x_vals, eh_y_vals = get_grid_vals_vector(emode, etas, mappings)
        cPeh_x_vals, cPeh_y_vals = get_grid_vals_vector(cP_emode, etas, mappings)
        jumps_eh_vals = abs(eh_x_vals-cPeh_x_vals)**2 + abs(eh_y_vals-cPeh_y_vals)**2
        curl_eh_vals = get_grid_vals_scalar(curl_emode, etas, mappings)

        my_small_plot(
            title='mode k='+repr(k_eig)+'  --  norm = '+ repr(norm_emode) + '  --  eigenvalue = '+repr(evalue),
            vals=[eh_x_vals, eh_y_vals, jumps_eh_vals, curl_eh_vals],
            titles=[r'$e^h_{k,x}$', r'$e^h_{k,y}$', r'$|(I-P^1_c) e^h_k|^2$', r'curl$(e^h_k)$'],
            xx=xx,
            yy=yy,
        )

if __name__ == '__main__':


    # aa = np.ones(10)
    # bb = np.ones(10)
    # cc = np.dot(aa,bb)
    # print(aa.shape, type(aa))
    # print(bb.shape, type(bb))
    # print(cc.shape, type(cc))
    # print("cc = ", cc)
    # exit()

    run_maxwell_2d_eigenproblem(nb_eigs=8, ncells=[2**4, 2**4], degree=[2,2])