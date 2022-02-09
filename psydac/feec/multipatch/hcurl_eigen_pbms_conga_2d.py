from mpi4py import MPI

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from sympy import pi, cos, sin, Matrix, Tuple, Max, exp
from sympy import symbols
from sympy import lambdify

from sympde.expr     import TerminalExpr
from sympde.calculus import grad, dot, inner, rot, div, curl, cross
from sympde.calculus import minus, plus
from sympde.topology import NormalVector
from sympde.expr     import Norm

from sympde.topology import Derham
from sympde.topology import element_of, elements_of, Domain

from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping
from sympde.topology import VectorFunctionSpace

from sympde.expr.equation import find, EssentialBC

from sympde.expr.expr import LinearForm, BilinearForm
from sympde.expr.expr import integral


from scipy.sparse.linalg import spsolve, spilu, cg, lgmres
from scipy.sparse.linalg import LinearOperator, eigsh, minres, gmres

from scipy.sparse.linalg import inv
from scipy.sparse.linalg import norm as spnorm
from scipy.linalg        import eig, norm
from scipy.sparse import save_npz, load_npz, bmat

# from scikits.umfpack import splu    # import error

from sympde.topology import Derham
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from psydac.feec.multipatch.api import discretize
from psydac.feec.pull_push     import pull_2d_h1, pull_2d_hcurl, push_2d_hcurl, push_2d_l2

from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.api.settings        import PSYDAC_BACKENDS
from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, get_K0_and_K0_inv, get_K1_and_K1_inv, get_M_and_M_inv
from psydac.feec.multipatch.operators import ConformingProjection_V0, ConformingProjection_V1, time_count
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector, get_grid_quad_weights
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, my_small_plot, my_small_streamplot
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain, get_ref_eigenvalues
from psydac.feec.multipatch.electromag_pbms import get_source_and_solution

comm = MPI.COMM_WORLD

def solve_eigenvalue_pbm(nc=4, deg=4, domain_name='pretzel_f', backend_language='python', mu=1, nu=1, gamma_h=10):
    """
    solver for the eigenvalue problem: find lambda in R and u in H0(curl), such that

      A u   = lambda * u    on \Omega

    with an operator

      A u := mu * curl curl u  -  nu * grad div u

    discretized as  Ah: V1h -> V1h  with a broken-FEEC approach involving a discrete sequence on a 2D multipatch domain \Omega,

      V0h  --grad->  V1h  -—curl-> V2h

    Examples:

      - curl-curl eigenvalue problem with
          mu  = 1
          nu  = 0

      - Hodge-Laplacian eigenvalue problem with
          mu  = 1
          nu  = 1

    :param nc: nb of cells per dimension, in each patch
    :param deg: coordinate degree in each patch
    :param gamma_h: jump penalization parameter
    """

    ncells = [nc, nc]
    degree = [deg,deg]

    domain = build_multipatch_domain(domain_name=domain_name)
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    derham  = Derham(domain, ["H1", "Hcurl", "L2"], hom_bc=True)   # the bc's should be a parameter of the continuous deRham sequence

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

    # 'geometric' commuting projection operators in the broken spaces (note: applied to smooth functions they return conforming fields)
    nquads = [4*(d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    # multi-patch (broken) spaces
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    # multi-patch (broken) linear operators / matrices
    M0 = V0h.MassMatrix
    M1 = V1h.MassMatrix
    M2 = V2h.MassMatrix
    M0_inv = V0h.InverseMassMatrix

    # was:
    # M0 = BrokenMass(V0h, domain_h, is_scalar=True, backend_language=backend_language)
    # M1 = BrokenMass(V1h, domain_h, is_scalar=False, backend_language=backend_language)
    # M2 = BrokenMass(V2h, domain_h, is_scalar=True, backend_language=backend_language)

    # other option: define as Hodge Operators:
    dH0 = V0h.dualHodge  # dH0 = M0: Hodge operator primal_V0 -> dual_V0
    dH1 = V1h.dualHodge  # dH1 = M1: Hodge operator primal_V1 -> dual_V1
    dH2 = V2h.dualHodge  # dH2 = M2: Hodge operator primal_V2 -> dual_V2
    H0  = V0h.Hodge      # dH0 = M0_inv: Hodge operator dual_V0 -> primal_V0

    # conforming Projections (should take into account the boundary conditions of the continuous deRham sequence)
    cP0 = V0h.ConformingProjection
    cP1 = V1h.ConformingProjection

    # was:
    # cP1 = ConformingProjection_V1(V1h, domain_h, hom_bc=True, backend_language=backend_language)

    # broken (patch-wise) differential operators
    bD0, bD1 = derham_h.BrokenDerivatives

    # was:
    # bD0, bD1 = derham_h.broken_derivatives_as_operators

    I1 = IdLinearOperator(V1h)

    # convert to scipy (or compose linear operators ?)
    M0_m = M0.to_sparse_matrix()  # or: dH0_m  = dH0.to_sparse_matrix() ...
    M1_m = M1.to_sparse_matrix()
    M2_m = M2.to_sparse_matrix()
    M0_inv_m = M0_inv.to_sparse_matrix()
    cP0_m = cP0.to_sparse_matrix()
    cP1_m = cP1.to_sparse_matrix()
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()
    I1_m = I1.to_sparse_matrix()

    # Conga (projection-based) stiffness matrices
    # curl curl:
    pre_CC_m = bD1_m.transpose() @ M2_m @ bD1_m
    CC_m = cP1_m.transpose() @ pre_CC_m @ cP1_m  # Conga stiffness matrix

    # grad div:
    pre_GD_m = M1_m @ bD0_m @ cP0_m @ M0_inv_m @ cP0_m.transpose() @ bD0_m.transpose() @ M1_m
    GD_m = cP1_m.transpose() @ pre_GD_m @ cP1_m  # Conga stiffness matrix

    # jump penalization:
    jump_penal_m = I1_m - cP1_m
    JP_m = jump_penal_m.transpose() * M1_m * jump_penal_m

    A_m = mu * CC_m + gamma_h * JP_m + nu * GD_m

    eigenvalues, eigenvectors = get_eigenvalues(nb_eigs, sigma, A_m, M_m)

    # plot something...

    return eigenvalues, eigenvectors



def get_eigenvalues(nb_eigs, sigma, A_m, M_m):
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- ')
    print('computing {0} eigenvalues (and eigenvectors) close to sigma={1} with scipy.sparse.eigsh...'.format(nb_eigs, sigma) )
    mode = 'normal'
    which = 'LM'
    # from eigsh docstring:
    #   ncv = number of Lanczos vectors generated ncv must be greater than k and smaller than n;
    #   it is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))
    ncv = 4*nb_eigs
    print('A_m.shape = ', A_m.shape)
    try_lgmres = True
    max_shape_splu = 17000
    if A_m.shape[0] < max_shape_splu:
        print('(via sparse LU decomposition)')
        OPinv = None
        tol_eigsh = 0
    else:

        OP_m = A_m - sigma*M_m
        tol_eigsh = 1e-7
        if try_lgmres:
            print('(via SPILU-preconditioned LGMRES iterative solver for A_m - sigma*M1_m)')
            OP_spilu = spilu(OP_m, fill_factor=15, drop_tol=5e-5)
            preconditioner = LinearOperator(OP_m.shape, lambda x: OP_spilu.solve(x) )
            tol = tol_eigsh
            OPinv = LinearOperator(
                matvec=lambda v: lgmres(OP_m, v, x0=None, tol=tol, atol=tol, M=preconditioner,
                                    callback=lambda x: print('cg -- residual = ', norm(OP_m.dot(x)-v))
                                    )[0],
                shape=M_m.shape,
                dtype=M_m.dtype
            )

        else:
            # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html:
            # the user can supply the matrix or operator OPinv, which gives x = OPinv @ b = [A - sigma * M]^-1 @ b.
            # > here, minres: MINimum RESidual iteration to solve Ax=b
            # suggested in https://github.com/scipy/scipy/issues/4170
            print('(with minres iterative solver for A_m - sigma*M1_m)')
            OPinv = LinearOperator(matvec=lambda v: minres(OP_m, v, tol=1e-10)[0], shape=M_m.shape, dtype=M_m.dtype)

    eigenvalues, eigenvectors = eigsh(A_m, k=nb_eigs, M=M_m, sigma=sigma, mode=mode, which=which, ncv=ncv, tol=tol_eigsh, OPinv=OPinv)

    print("done. eigenvalues found: " + repr(eigenvalues))
    return eigenvalues, eigenvectors
