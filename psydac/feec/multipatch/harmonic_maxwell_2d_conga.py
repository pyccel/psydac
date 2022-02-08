# ------------------------------------------------------------------------------------------------------------------------
# pseudo-code example for the time-harmonic maxwell equation
#
#   -omega**2 * u  +  mu * curl curl u  =  f                on \Omega
#
#                                 n x u = n x u_bc          on \partial \Omega
#
# solved with a conga (broken-feec) method on a multipatch domain \Omega
# ------------------------------------------------------------------------------------------------------------------------

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

comm = MPI.COMM_WORLD

# ----------------------
# parameters

nc = 4
deg = 4
domain_name = 'pretzel_f'
backend_language = 'python'

source_type = 'ellnew_J' # divergence-free J source with elliptic-shaped support
# source_type = 'manu_J'  # manufactured solution
source_proj = 'P_geom'  # or 'P_L2'

omega = 70
eta = -omega**2
mu = 1
gamma_h = 10    # penalization of the jumps in V1h (discrete H-curl)

# ----------------------

ncells = [nc, nc]
degree = [deg,deg]

domain = build_multipatch_domain(domain_name=domain_name)
mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
mappings_list = list(mappings.values())

derham  = Derham(domain, ["H1", "Hcurl", "L2"], hom_bc=True)   # the bc's should be a parameter of the continuous deRham sequence

# design of source and solution should also be thought over -- here I'm only copying old function from electromag_pbms.py
f_scal, f_vect, u_bc, ph_ref, uh_ref, p_ex, u_ex, phi, grad_phi = get_source_and_solution(
    source_type=source_type, eta=eta, mu=mu, domain=domain,
    refsol_params=[N_diag, method, Psource],
)
lift_u_bc = (u_bc is not None)

domain_h = discretize(domain, ncells=ncells, comm=comm)
derham_h = discretize(derham, domain_h, degree=degree, backend=PSYDAC_BACKENDS[backend_language])

# multi-patch (broken) spaces
V0h = derham_h.V0
V1h = derham_h.V1
V2h = derham_h.V2

# multi-patch (broken) linear operators / matrices
M0 = V0h.MassMatrix
M1 = V1h.MassMatrix
M2 = V2h.MassMatrix

# was:
# M0 = BrokenMass(V0h, domain_h, is_scalar=True, backend_language=backend_language)
# M1 = BrokenMass(V1h, domain_h, is_scalar=False, backend_language=backend_language)
# M2 = BrokenMass(V2h, domain_h, is_scalar=True, backend_language=backend_language)

# other option: define as Hodge Operators:
dH0 = V0h.dualHodge  # M0 == dH0
dH1 = V1h.dualHodge  # ..
dH2 = V2h.dualHodge  # ..

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
cP0_m = cP0.to_sparse_matrix()
cP1_m = cP1.to_sparse_matrix()
bD0_m = bD0.to_sparse_matrix()
bD1_m = bD1.to_sparse_matrix()
I1_m = I1.to_sparse_matrix()

# Conga operators
bCC_m = bD1_m.transpose() @ M2_m @ bD1_m  # broken (patch-diagonal) curl-curl stiffness matrix
CC_m = cP1_m.transpose() @ bCC_m @ cP1_m  # Conga curl-curl stiffness matrix

# jump penalization
jump_penal_m = I1_m-cP1_m
JP_m = jump_penal_m.transpose() * M1_m * jump_penal_m

A_m = mu * CC_m + gamma_h * JP_m + eta * cP1_m.transpose() @ M1_m @ cP1_m

if source_proj == 'P_geom':
    # approx source is
    # f_h = P1-geometric (commuting) projection of f_vect
    f_x = lambdify(domain.coordinates, f_vect[0])
    f_y = lambdify(domain.coordinates, f_vect[1])
    f_log = [pull_2d_hcurl([f_x, f_y], m) for m in mappings_list]
    f_h = P1(f_log)
    f_c = f_h.coeffs.toarray()
    b_c = M1_m.dot(f_c)

else:
    # approx source is
    # f_h = L2 projection of f_vect
    v  = element_of(V1h.symbolic_space, name='v')
    expr = dot(f_vect,v)
    l = LinearForm(v, integral(domain, expr))
    lh = discretize(l, domain_h, V1h, backend=PSYDAC_BACKENDS[backend_language])
    b  = lh.assemble()
    b_c = b.toarray()

if lift_u_bc:
    # Projector on broken space
    # note: we only need to set the boundary dofs -- here we apply the full P1 on u_bc
    # (btw, it's a bit weird to apply P1 on the list of (pulled back) logical fields -- why not just apply it on u_bc ? otherwise it should maybe be called logical_P1...)
    nquads = [4*(d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)
    u_bc_x = lambdify(domain.coordinates, u_bc[0])
    u_bc_y = lambdify(domain.coordinates, u_bc[1])
    u_bc_log = [pull_2d_hcurl([u_bc_x, u_bc_y], m) for m in mappings_list]
    # note: we only need the boundary dofs of u_bc
    uh_bc = P1(u_bc_log)
    ubc_c = uh_bc.coeffs.toarray()
    # removing internal dofs
    ubc_c = ubc_c-cP1_m.dot(ubc_c)
    # modified source for the homogeneous pbm

    A_bc_m = cP1_m.transpose() @ ( mu * bCC_m  + eta * M1_m )
    b_c = b_c - A_bc_m.dot(ubc_c)

else:
    ubc_c = None

# direct solve with scipy spsolve
uh_c = spsolve(A_m, b_c)

# project the solution on the conforming problem space
uh_c = cP1_m.dot(uh_c)

if lift_u_bc:
    # adding the lifted boundary condition
    uh_c += ubc_c

uh = FemField(V1h, coeffs=array_to_stencil(uh_c, V1h.vector_space))

uh.plot()
