from mpi4py import MPI
from sympde.calculus import dot, grad

from sympde.topology import ScalarFunctionSpace, element_of, elements_of, Line, NormalVector

from sympde.expr import BilinearForm, LinearForm, integral, EssentialBC, Norm, find

from sympde.topology import Square, Union

from sympy import pi, cos, sin, symbols

from sympy import I, conjugate, re

from psydac.linalg.utilities import array_to_psydac

from psydac.fem.basic import FemField
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

import numpy as np

import sympy as sy

import os

import scipy.sparse.linalg as spla

from psydac.api.discretization import *

# domain = Square(bounds1=(0, 0.5), bounds2=(0, 1))
#
# V = ScalarFunctionSpace('V', domain, kind=None)
# V.codomain_type ='complex'
#
# u, v = elements_of(V, names='u, v')
#
# bilinForm = BilinearForm((u, v), integral(domain, dot(grad(u), grad(v))))
#
# linForm = LinearForm(v, I*integral(domain, grad(v)))
#
# norm = Norm(v,domain)
#
# domain_h = discretize(domain, ncells=[8, 8])
# Vh = discretize(V, domain_h, degree=[2, 2])
#
# # Bilin_h = discretize(bilinForm, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)
# linForm_h = discretize(linForm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)

# norm_h = discretize(norm, domain_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL)
#
# equation = find(u, forall=v, lhs=bilinForm(u, v), rhs=linForm(v), bc=[])
# equation_h = discretize(equation, domain_h, [Vh, Vh], backend=PSYDAC_BACKEND_GPYCCEL)
#
# uh = equation_h.solve()
# print(uh.coeffs)

x,y,z = symbols('x1, x2, x3')
def get_boundaries(*args):

    if not args:
        return ()
    else:
        assert all(1 <= a <= 4 for a in args)
        assert len(set(args)) == len(args)

    boundaries = {1: {'axis': 0, 'ext': -1},
                  2: {'axis': 0, 'ext':  1},
                  3: {'axis': 1, 'ext': -1},
                  4: {'axis': 1, 'ext':  1}}

    return tuple(boundaries[i] for i in args)

#==============================================================================
solution = sin(pi*x)*sin(pi*y)
f        = 2*pi**2*sin(pi*x)*sin(pi*y)

dir_zero_boundary    = get_boundaries(1, 2, 3, 4)
backend=PSYDAC_BACKEND_GPYCCEL

ncells=[2**3, 2**3]
degree=[2, 2]
comm=MPI.COMM_WORLD

alpha=(I)

#+++++++++++++++++++++++++++++++
# 1. Abstract model
#+++++++++++++++++++++++++++++++
domain = Square()

B_dirichlet = Union(*[domain.get_boundary(**kw) for kw in dir_zero_boundary])

Vr  = ScalarFunctionSpace('V', domain)
ur  = element_of(Vr, name='u')
vr  = element_of(Vr, name='v')
nn = NormalVector('nn')
# Bilinear form a: V x V --> R
ar = BilinearForm((ur, vr), integral(domain, dot(grad(ur), grad(vr))))
# Linear form l: V --> R
lr = LinearForm(vr, integral(domain, f * vr))
# Dirichlet boundary conditions
bc = []
bc += [EssentialBC(ur, 0, B_dirichlet)]
# Variational model
equationr = find(ur, forall=vr, lhs=ar(ur, vr), rhs=lr(vr), bc=bc)

Vc  = ScalarFunctionSpace('V', domain)
Vc.codomain_type = 'complex'
uc  = element_of(Vc, name='u')
vc  = element_of(Vc, name='v')
nn = NormalVector('nn')
# Bilinear form a: V x V --> R
ac = BilinearForm((uc, vc), integral(domain, dot(grad(uc), grad(vc))))
# Linear form l: V --> R
lc = LinearForm(vc, alpha*integral(domain, f * vc))
# Dirichlet boundary conditions
bc = []
bc += [EssentialBC(uc, 0, B_dirichlet)]
# Variational model
equationc = find(uc, forall=vc, lhs=ac(uc, vc), rhs=lc(vc), bc=bc)

#+++++++++++++++++++++++++++++++
# 2. Discretization
#+++++++++++++++++++++++++++++++

# Create computational domain from topological domain
domain_h = discretize(domain, ncells=ncells, comm=comm)

# Discrete spaces
Vrh = discretize(Vr, domain_h, degree=degree)
Vch = discretize(Vc, domain_h, degree=degree)

# Discretize equation using Dirichlet bc
equation_hr = discretize(equationr, domain_h, [Vrh, Vrh], backend=backend)
equation_hc = discretize(equationc, domain_h, [Vch, Vch], backend=backend)

#+++++++++++++++++++++++++++++++
# 3. Solution
#+++++++++++++++++++++++++++++++

# Solve linear system
uhr = equation_hr.solve()
uhc = equation_hc.solve()

# Error norms
error_r = ur - solution
error_c = (uc/conjugate(alpha)) - solution

l2normr = Norm(error_r, domain, kind='l2')
h1normr = Norm(error_r, domain, kind='h1')
l2normc = Norm(error_c, domain, kind='l2')
h1normc = Norm(error_c, domain, kind='h1')

# Discretize error norms
l2norm_hr = discretize(l2normr, domain_h, Vrh, backend=backend)
h1norm_hr = discretize(h1normr, domain_h, Vrh, backend=backend)
l2norm_hc = discretize(l2normc, domain_h, Vch, backend=backend)
h1norm_hc = discretize(h1normc, domain_h, Vch, backend=backend)

# Compute error norms
l2_error_r = l2norm_hr.assemble(u=uhr)
h1_error_r = h1norm_hr.assemble(u=uhr)
l2_error_c = l2norm_hc.assemble(u=uhc)
h1_error_c = h1norm_hc.assemble(u=uhc)

print(l2_error_r)
print(l2_error_c)
print(h1_error_r)
print(h1_error_c)
# assert(abs(l2_error) < 1.e-7)
# assert(abs(h1_error) < 1.e-7)