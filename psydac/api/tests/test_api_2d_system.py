#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from mpi4py import MPI
from sympy import pi, cos, sin, Tuple, Matrix
import numpy as np
import pytest

from sympde.calculus import grad, dot, inner
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of
from sympde.topology import Square
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm, SemiNorm
from sympde.expr import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize

#==============================================================================

def run_system_1_2d_dir(Fe, Ge, f0, f1, ncells, degree):

    # ... abstract model
    domain = Square()

    W = VectorFunctionSpace('W', domain)
    V = ScalarFunctionSpace('V', domain)
    X = ProductSpace(W, V)

    X = W * V

    x,y = domain.coordinates

    F = element_of(W, name='F')
    G = element_of(V, name='G')

    u,v = [element_of(W, name=i) for i in ['u', 'v']]
    p,q = [element_of(V, name=i) for i in ['p', 'q']]

    int_0 = lambda expr: integral(domain , expr)

    a0 = BilinearForm((v,u), int_0(inner(grad(v), grad(u))))
    a1 = BilinearForm((q,p), int_0(p*q))
    a  = BilinearForm(((v,q),(u,p)), a0(v,u) + a1(q,p))

    l0 = LinearForm(v, int_0(dot(f0, v)))
    l1 = LinearForm(q, int_0(f1*q))
    l  = LinearForm((v,q), l0(v) + l1(q))

    error = Matrix([F[0]-Fe[0], F[1]-Fe[1]])
    l2norm_F =     Norm(error, domain, kind='l2')
    h1norm_F = SemiNorm(error, domain, kind='h1')

    error = G-Ge
    l2norm_G =     Norm(error, domain, kind='l2')
    h1norm_G = SemiNorm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find([u,p], forall=[v,q], lhs=a((u,p),(v,q)), rhs=l(v,q), bc=bc)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, ncells=ncells)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    Wh = discretize(W, domain_h, degree=degree)
    Xh = discretize(X, domain_h, degree=degree)
    # ...

#    # TODO: make this work
#    Xh = Wh * Vh
#    Wh, Vh = Xh.spaces

    # ... discretize the equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Xh, Xh])
    # ...

    # ... discretize norms
    l2norm_F_h = discretize(l2norm_F, domain_h, Wh)
    h1norm_F_h = discretize(h1norm_F, domain_h, Wh)

    l2norm_G_h = discretize(l2norm_G, domain_h, Vh)
    h1norm_G_h = discretize(h1norm_G, domain_h, Vh)
    # ...

    # ... solve the discrete equation
    Xh = equation_h.solve()
    # ...

    # TODO [YG, 12.02.2021]: Fh and Gh are temporary FEM fields needed because
    #   the blocks in Xh.coeffs have been flattened. Once this assumption is
    #   removed, just assemble the error norms passing F = Xh[0] and G = Xh[1].

    # ...
    Fh = FemField( Wh )
    Fh.coeffs[0][:,:] = Xh.coeffs[0][:,:]
    Fh.coeffs[1][:,:] = Xh.coeffs[1][:,:]
    # ...

    # ...
    Gh = FemField( Vh )
    Gh.coeffs[:,:] = Xh.coeffs[2][:,:]
    # ...

    # ... compute norms
    l2_error_F = l2norm_F_h.assemble(F=Fh)
    h1_error_F = h1norm_F_h.assemble(F=Fh)

    l2_error_G = l2norm_G_h.assemble(G=Gh)
    h1_error_G = h1norm_G_h.assemble(G=Gh)
    # ...

    l2_error = np.asarray([l2_error_F, l2_error_G])
    h1_error = np.asarray([h1_error_F, h1_error_G])

    return l2_error, h1_error

###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
def test_api_system_1_2d_dir_1():

    from sympy import symbols

    x1,x2 = symbols('x1, x2', real=True)

    Fe = Tuple(sin(pi*x1)*sin(pi*x2), sin(pi*x1)*sin(pi*x2))
    f0 = Tuple(2*pi**2*sin(pi*x1)*sin(pi*x2),
              2*pi**2*sin(pi*x1)*sin(pi*x2))

    Ge = cos(pi*x1)*cos(pi*x2)
    f1 = cos(pi*x1)*cos(pi*x2)

    l2_error, h1_error = run_system_1_2d_dir(Fe, Ge, f0, f1,
                                            ncells=[2**3,2**3], degree=[2,2])

    expected_l2_error =  np.asarray([0.00030842129059875065,
                                     0.0002164796555228256])
    expected_h1_error =  np.asarray([0.018418110343264293,
                                     0.012987988507232278])


    assert( np.allclose(l2_error, expected_l2_error, 1.e-13) )
    assert( np.allclose(h1_error, expected_h1_error, 1.e-13) )

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
