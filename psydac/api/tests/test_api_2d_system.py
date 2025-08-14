# -*- coding: UTF-8 -*-
import time

from sympy import pi, cos, sin, Tuple, Matrix
import numpy as np

from sympde.calculus import grad, dot, inner
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.expr import Norm, SemiNorm
from sympde.expr import find, EssentialBC

from psydac.api.discretization import discretize
from psydac.api.essential_bc   import apply_essential_bc
from psydac.api.equation       import _default_solver as solver_options
from psydac.linalg.solvers     import inverse
from psydac.fem.basic          import FemField

#==============================================================================
def run_stokes_2d_dir(Fe, Ge, f0, f1, *, ncells, degree,
                      comm=None, backend=None, timing=None):

    # The dictionary for the timings is modified in-place
    profile = (timing is not None)
    if profile:
        assert isinstance(timing, dict)

    #+++++++++++++++++++++++++++++++
    # 1. Abstract model
    #+++++++++++++++++++++++++++++++
    domain = Square()

    W = VectorFunctionSpace('W', domain)
    V = ScalarFunctionSpace('V', domain)
    X = ProductSpace(W, V)

    F = element_of(W, name='F')
    G = element_of(V, name='G')

    u, v = elements_of(W, names='u, v')
    p, q = elements_of(V, names='p, q')

    int_0 = lambda expr: integral(domain, expr)

    a0 = BilinearForm((u, v), int_0(inner(grad(u), grad(v))))
    a1 = BilinearForm((p, q), int_0(p * q))
    a  = BilinearForm(((u, p), (v, q)), a0(u, v) + a1(p, q))

    l0 = LinearForm(v, int_0(dot(f0, v)))
    l1 = LinearForm(q, int_0(f1 * q))
    l  = LinearForm((v,q), l0(v) + l1(q))

    error = Matrix([F[0]-Fe[0], F[1]-Fe[1]])
    l2norm_F =     Norm(error, domain, kind='l2')
    h1norm_F = SemiNorm(error, domain, kind='h1')

    error = G-Ge
    l2norm_G =     Norm(error, domain, kind='l2')
    h1norm_G = SemiNorm(error, domain, kind='h1')

    bc = EssentialBC(u, 0, domain.boundary)
    equation = find((u, p), forall=(v, q), lhs=a((u, p), (v, q)), rhs=l(v, q), bc=bc)

    #+++++++++++++++++++++++++++++++
    # 2. Discretization
    #+++++++++++++++++++++++++++++++

    # Create computational domain from topological domain
    domain_h = discretize(domain, ncells=ncells, comm=comm)

    # Discrete spaces
    Vh = discretize(V, domain_h, degree=degree)
    Wh = discretize(W, domain_h, degree=degree)
    Xh = discretize(X, domain_h, degree=degree)
    # ...

#    # TODO: make this work
#    Xh = Wh * Vh
#    Wh, Vh = Xh.spaces

    # Discretize equation using Dirichlet bc
    equation_h = discretize(equation, domain_h, [Xh, Xh], backend=backend)
    # ...

    # Discretize error norms
    l2norm_F_h = discretize(l2norm_F, domain_h, Wh, backend=backend)
    h1norm_F_h = discretize(h1norm_F, domain_h, Wh, backend=backend)

    l2norm_G_h = discretize(l2norm_G, domain_h, Vh, backend=backend)
    h1norm_G_h = discretize(h1norm_G, domain_h, Vh, backend=backend)

    #+++++++++++++++++++++++++++++++
    # 3. Solution
    #+++++++++++++++++++++++++++++++

    if profile:
        # Assemble matrix corresponding to discrete bilinear form
        tb = time.time()
        A  = equation_h.lhs.assemble()
        te = time.time()
        timing['matrix'] = te - tb

        # Assemble vector corresponding to discrete linear form
        tb = time.time()
        b  = equation_h.rhs.assemble()
        te = time.time()
        timing['rhs'] = te - tb

        # Apply essential BCs to A and b
        apply_essential_bc(A, *equation_h.bc)
        apply_essential_bc(b, *equation_h.bc)

        # Solve linear system
        tb = time.time()
        A_inv = inverse(A, **solver_options)
        x = A_inv @ b
        te = time.time()
        timing['solution'] = te - tb

        # Store result in a new FEM field
        xh = FemField(Xh, coeffs=x)

    else:
        xh = equation_h.solve()

    # TODO [YG, 12.02.2021]: Fh and Gh are temporary FEM fields needed because
    #   the blocks in Xh.coeffs have been flattened. Once this assumption is
    #   removed, just assemble the error norms passing F = Xh[0] and G = Xh[1].

    # ...
    Fh = FemField( Wh )
    Fh.coeffs[0][:,:] = xh.coeffs[0][:,:]
    Fh.coeffs[1][:,:] = xh.coeffs[1][:,:]
    # ...

    # ...
    Gh = FemField( Vh )
    Gh.coeffs[:,:] = xh.coeffs[2][:,:]
    # ...

    # Compute error norms
    if profile: tb = time.time()
    l2_error_F = l2norm_F_h.assemble(F=Fh)
    if profile: te = time.time()
    if profile: timing['L2 error (F)'] = te - tb

    if profile: tb = time.time()
    h1_error_F = h1norm_F_h.assemble(F=Fh)
    if profile: te = time.time()
    if profile: timing['H1 error (F)'] = te - tb

    if profile: tb = time.time()
    l2_error_G = l2norm_G_h.assemble(G=Gh)
    if profile: te = time.time()
    if profile: timing['L2 error (G)'] = te - tb

    if profile: tb = time.time()
    h1_error_G = h1norm_G_h.assemble(G=Gh)
    if profile: te = time.time()
    if profile: timing['H1 error (G)'] = te - tb

    l2_error = np.asarray([l2_error_F, l2_error_G])
    h1_error = np.asarray([h1_error_F, h1_error_G])

    return l2_error, h1_error

###############################################################################
#            SERIAL TESTS
###############################################################################

def test_2d_stokes_dir(backend=None, timing=None):

    from sympy import symbols

    x1,x2 = symbols('x1, x2', real=True)

    Fe = Tuple(sin(pi*x1)*sin(pi*x2), sin(pi*x1)*sin(pi*x2))
    f0 = Tuple(2*pi**2*sin(pi*x1)*sin(pi*x2),
              2*pi**2*sin(pi*x1)*sin(pi*x2))

    Ge = cos(pi*x1)*cos(pi*x2)
    f1 = cos(pi*x1)*cos(pi*x2)

    l2_error, h1_error = run_stokes_2d_dir(Fe, Ge, f0, f1,
                                           ncells=[2**3, 2**3], degree=[2, 2],
                                           backend=backend, timing=timing)

    expected_l2_error =  np.asarray([0.00030842129059875065,
                                     0.0002164796555228256])
    expected_h1_error =  np.asarray([0.018418110343264293,
                                     0.012987988507232278])

    assert np.allclose(l2_error, expected_l2_error, 1.e-13)
    assert np.allclose(h1_error, expected_h1_error, 1.e-13)

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
