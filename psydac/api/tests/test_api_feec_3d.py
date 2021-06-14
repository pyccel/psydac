# -*- coding: UTF-8 -*-

from sympde.topology import Mapping
from sympde.calculus import grad, dot
from sympde.calculus import laplace
from sympde.topology import ScalarFunctionSpace
from sympde.topology import elements_of
from sympde.topology import NormalVector
from sympde.topology import Cube, Derham
from sympde.topology import Union
from sympde.expr     import BilinearForm, LinearForm, integral
from sympde.expr     import Norm
from sympde.expr     import find, EssentialBC

from psydac.fem.basic          import FemField
from psydac.api.discretization import discretize
from psydac.feec.pull_push     import push_3d_hcurl, push_3d_hdiv
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL, PSYDAC_BACKEND_NUMBA
from psydac.linalg.utilities   import array_to_stencil
from psydac.linalg.iterative_solvers import cg

from mpi4py import MPI

import pytest
import numpy as np
import scipy as sc

#===============================================================================
def splitting_integrator_scipy(e0, b0, M1, M2, CURL, dt, niter):

    CURL_T = CURL.T
    M1_solver = sc.sparse.linalg.splu(M1)
    def M1CM2_dot(b):
        y1 = M2.dot(b)
        y2 = CURL_T.dot(y1)
        return M1_solver.solve(y2)

    e_history = [e0]
    b_history = [b0]

    for  ts in range(niter):

        b = b_history[ts]
        e = e_history[ts]

        b_new = b - dt * CURL.dot(e)
        e_new = e + dt * M1CM2_dot(b_new)

        b_history.append(b_new)
        e_history.append(e_new)
    return e_history, b_history

def splitting_integrator_stencil(e0, b0, M1, M2, CURL, dt, niter):

    CURL_T = CURL.transpose()
    def M1CM2_dot(b):
        y1 = M2.dot(b)
        y2 = CURL_T.dot(y1)
        return cg(M1, y2, tol=1e-12)[0]

    e_history = [e0]
    b_history = [b0]

    for  ts in range(niter):

        b = b_history[ts]
        e = e_history[ts]

        b_new = b - dt * CURL.dot(e)
        e_new = e + dt * M1CM2_dot(b_new)

        b_history.append(b_new)
        e_history.append(e_new)
    return e_history, b_history

def evaluation_all_times(fields, x, y, z):
    ak_value = np.empty(len(fields), dtype = 'float')

    for i in range(len(fields)):
        ak_value[i] = fields[i](x,y,z)
    return ak_value

#==================================================================================
def run_maxwell_3d_scipy(logical_domain, mapping, e_ex, b_ex, ncells, degree, periodic, dt, niter):

    domain  = mapping(logical_domain)
    derham  = Derham(domain)

    u0, v0 = elements_of(derham.V0, names='u0, v0')
    u1, v1 = elements_of(derham.V1, names='u1, v1')
    u2, v2 = elements_of(derham.V2, names='u2, v2')
    u3, v3 = elements_of(derham.V3, names='u3, v3')

    a0 = BilinearForm((u0, v0), integral(domain, u0*v0))
    a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    a2 = BilinearForm((u2, v2), integral(domain, dot(u2, v2)))
    a3 = BilinearForm((u3, v3), integral(domain, u3*v3))

    #==============================================================================
    # Discrete objects: Psydac

    domain_h = discretize(domain, ncells=ncells, comm=MPI.COMM_WORLD)
    derham_h = discretize(derham, domain_h, degree=degree, periodic=periodic)

    a1_h = discretize(a1, domain_h, (derham_h.V1, derham_h.V1), backend=PSYDAC_BACKEND_GPYCCEL)
    a2_h = discretize(a2, domain_h, (derham_h.V2, derham_h.V2), backend=PSYDAC_BACKEND_GPYCCEL)

    # StencilMatrix objects
    M1 = a1_h.assemble().tosparse().tocsc()
    M2 = a2_h.assemble().tosparse().tocsr()

    # Diff operators
    GRAD, CURL, DIV = derham_h.derivatives_as_matrices

    # Porjectors
    P0, P1, P2, P3  = derham_h.projectors(nquads=[5,5,5])

    CURL = CURL.transform(lambda block: block.tokronstencil().tostencil()).tomatrix().tosparse().tocsr()

    # initial conditions
    e0_1 = lambda x, y, z: e_ex[0](0, x, y, z)
    e0_2 = lambda x, y, z: e_ex[1](0, x, y, z)
    e0_3 = lambda x, y, z: e_ex[2](0, x, y, z)

    e0   = (e0_1, e0_2, e0_3)

    b0_1 = lambda x, y, z : b_ex[0](0, x, y, z)
    b0_2 = lambda x, y, z : b_ex[1](0, x, y, z)
    b0_3 = lambda x, y, z : b_ex[2](0, x, y, z)

    b0   = (b0_1, b0_2, b0_3)

    # project initial conditions
    e0_coeff = P1(e0).coeffs

    b0_coeff = P2(b0).coeffs

    # time integrator
    e_history, b_history = splitting_integrator_scipy(e0_coeff.toarray(), b0_coeff.toarray(), M1, M2, CURL, dt, niter)

    # study of fields
    b_history = [array_to_stencil(bi, derham_h.V2.vector_space) for bi in b_history]
    b_fields  = [FemField(derham_h.V2, bi).fields for bi in b_history]

    bx_fields = [bi[0] for bi in b_fields]
    by_fields = [bi[1] for bi in b_fields]
    bz_fields = [bi[2] for bi in b_fields]

    bx_value_fun = lambda x, y, z: evaluation_all_times(bx_fields, x, y, z)
    by_value_fun = lambda x, y, z: evaluation_all_times(by_fields, x, y, z)
    bz_value_fun = lambda x, y, z: evaluation_all_times(bz_fields, x, y, z)

    x,y,z      = derham_h.V0.breaks
    x, y       = 0.5, 0.5

    b_values_0 = []
    for zi in z:
        b_value_phys  = push_3d_hdiv(bx_value_fun, by_value_fun, bz_value_fun, x, y, zi, mapping)
        b_values_0.append(b_value_phys[0])
    b_values_0  = np.array(b_values_0)

    time_array  = np.linspace(0, dt*niter, niter + 1)
    tt, zz      = np.meshgrid(time_array, z)

    b_ex_values_0 = b_ex[0](tt, x, y, zz)

    error = abs(b_values_0-b_ex_values_0).max()
    return error

#==================================================================================
def run_maxwell_3d_stencil(logical_domain, mapping, e_ex, b_ex, ncells, degree, periodic, dt, niter):

    domain  = mapping(logical_domain)
    derham  = Derham(domain)

    u0, v0 = elements_of(derham.V0, names='u0, v0')
    u1, v1 = elements_of(derham.V1, names='u1, v1')
    u2, v2 = elements_of(derham.V2, names='u2, v2')
    u3, v3 = elements_of(derham.V3, names='u3, v3')

    a0 = BilinearForm((u0, v0), integral(domain, u0*v0))
    a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
    a2 = BilinearForm((u2, v2), integral(domain, dot(u2, v2)))
    a3 = BilinearForm((u3, v3), integral(domain, u3*v3))

    #==============================================================================
    # Discrete objects: Psydac

    domain_h = discretize(domain, ncells=ncells, comm=MPI.COMM_WORLD)
    derham_h = discretize(derham, domain_h, degree=degree, periodic=periodic)

    a1_h = discretize(a1, domain_h, (derham_h.V1, derham_h.V1), backend=PSYDAC_BACKEND_GPYCCEL)
    a2_h = discretize(a2, domain_h, (derham_h.V2, derham_h.V2), backend=PSYDAC_BACKEND_GPYCCEL)

    # StencilMatrix objects
    M1 = a1_h.assemble()
    M2 = a2_h.assemble()

    # Diff operators
    GRAD, CURL, DIV = derham_h.derivatives_as_matrices

    # Porjectors
    P0, P1, P2, P3  = derham_h.projectors(nquads=[5,5,5])

    # initial conditions
    e0_1 = lambda x, y, z: e_ex[0](0, x, y, z)
    e0_2 = lambda x, y, z: e_ex[1](0, x, y, z)
    e0_3 = lambda x, y, z: e_ex[2](0, x, y, z)

    e0   = (e0_1, e0_2, e0_3)

    b0_1 = lambda x, y, z : b_ex[0](0, x, y, z)
    b0_2 = lambda x, y, z : b_ex[1](0, x, y, z)
    b0_3 = lambda x, y, z : b_ex[2](0, x, y, z)

    b0   = (b0_1, b0_2, b0_3)

    # project initial conditions
    e0_coeff = P1(e0).coeffs

    b0_coeff = P2(b0).coeffs

    # time integrator
    e_history, b_history = splitting_integrator_stencil(e0_coeff, b0_coeff, M1, M2, CURL, dt, niter)

    # study of fields
    b_fields  = [FemField(derham_h.V2, bi).fields for bi in b_history]

    bx_fields = [bi[0] for bi in b_fields]
    by_fields = [bi[1] for bi in b_fields]
    bz_fields = [bi[2] for bi in b_fields]

    bx_value_fun = lambda x, y, z: evaluation_all_times(bx_fields, x, y, z)
    by_value_fun = lambda x, y, z: evaluation_all_times(by_fields, x, y, z)
    bz_value_fun = lambda x, y, z: evaluation_all_times(bz_fields, x, y, z)

    x,y,z      = derham_h.V0.breaks
    x, y       = 0.5, 0.5

    b_values_0 = []
    for zi in z:
        b_value_phys  = push_3d_hdiv(bx_value_fun, by_value_fun, bz_value_fun, x, y, zi, mapping)
        b_values_0.append(b_value_phys[0])
    b_values_0  = np.array(b_values_0)

    time_array  = np.linspace(0, dt*niter, niter + 1)
    tt, zz      = np.meshgrid(time_array, z)

    b_ex_values_0 = b_ex[0](tt, x, y, zz)

    error = abs(b_values_0-b_ex_values_0).max()
    return error
###############################################################################
#            SERIAL TESTS
###############################################################################

#==============================================================================
# 3D Maxwell's equations with "Collela" map
#==============================================================================
def test_maxwell_3d_1():
    class CollelaMapping3D(Mapping):

        _expressions = {'x': 'k1*(x1 + eps*sin(2.*pi*x1)*sin(2.*pi*x2))',
                        'y': 'k2*(x2 + eps*sin(2.*pi*x1)*sin(2.*pi*x2))',
                        'z': 'k3*x3'}

        _ldim        = 3
        _pdim        = 3

    M               = CollelaMapping3D('M', k1=1, k2=1, k3=1, eps=0.1)
    logical_domain  = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))

    # exact solution
    e_ex_0 = lambda t, x, y, z: 0
    e_ex_1 = lambda t, x, y, z: -np.cos(2*np.pi*t-2*np.pi*z)
    e_ex_2 = lambda t, x, y, z: 0

    e_ex   = (e_ex_0, e_ex_1, e_ex_2)

    b_ex_0 = lambda t, x, y, z : np.cos(2*np.pi*t-2*np.pi*z)
    b_ex_1 = lambda t, x, y, z : 0
    b_ex_2 = lambda t, x, y, z : 0

    b_ex   = (b_ex_0, b_ex_1, b_ex_2)

    #space parameters
    ncells   = [2**4, 2**3, 2**5]
    degree   = [2, 2, 2]
    periodic = [True, True, True]

    #time parameters
    dt    = 0.5*1/max(ncells)
    niter = 10
    T     = dt*niter

    error = run_maxwell_3d_scipy(logical_domain, M, e_ex, b_ex, ncells, degree, periodic, dt, niter)
    assert abs(error - 0.04294761712765949) < 1e-9

def test_maxwell_3d_2():
    class CollelaMapping3D(Mapping):

        _expressions = {'x': 'k1*(x1 + eps*sin(2.*pi*x1)*sin(2.*pi*x2))',
                        'y': 'k2*(x2 + eps*sin(2.*pi*x1)*sin(2.*pi*x2))',
                        'z': 'k3*x3'}

        _ldim        = 3
        _pdim        = 3

    M               = CollelaMapping3D('M', k1=1, k2=1, k3=1, eps=0.1)
    logical_domain  = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))

    # exact solution
    e_ex_0 = lambda t, x, y, z: 0
    e_ex_1 = lambda t, x, y, z: -np.cos(2*np.pi*t-2*np.pi*z)
    e_ex_2 = lambda t, x, y, z: 0

    e_ex   = (e_ex_0, e_ex_1, e_ex_2)

    b_ex_0 = lambda t, x, y, z : np.cos(2*np.pi*t-2*np.pi*z)
    b_ex_1 = lambda t, x, y, z : 0
    b_ex_2 = lambda t, x, y, z : 0

    b_ex   = (b_ex_0, b_ex_1, b_ex_2)

    #space parameters
    ncells   = [7, 7, 7]
    degree   = [2, 2, 2]
    periodic = [True, True, True]

    #time parameters
    dt    = 0.5*1/max(ncells)
    niter = 2
    T     = dt*niter

    error = run_maxwell_3d_stencil(logical_domain, M, e_ex, b_ex, ncells, degree, periodic, dt, niter)
    assert abs(error - 0.24586986658559362) < 1e-9
#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

