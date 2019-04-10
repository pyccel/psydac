# -*- coding: UTF-8 -*-

from sympy import pi, cos, sin
from sympy import S
from sympy import Tuple
from sympy import Matrix

from sympde.core import Constant
from sympde.calculus import grad, dot, inner, cross, rot, curl, div
from sympde.calculus import laplace, hessian
from sympde.topology import (dx, dy, dz)
from sympde.topology import FunctionSpace, VectorFunctionSpace
from sympde.topology import element_of_space, element_of_space
from sympde.topology import ProductSpace
from sympde.topology import element_of_space
from sympde.topology import element_of_space
from sympde.topology import element_of_space
from sympde.topology import Unknown
from sympde.topology import InteriorDomain, Union
from sympde.topology import Boundary, NormalVector, TangentVector
from sympde.topology import Domain, Line, Square, Cube
from sympde.topology import Trace, trace_0, trace_1
from sympde.topology import Union
from sympde.topology import Mapping
from sympde.expr import BilinearForm, LinearForm
from sympde.expr import Norm
from sympde.expr import find, EssentialBC

from gelato.expr import GltExpr

from psydac.fem.basic   import FemField
from psydac.fem.vector  import ProductFemSpace, VectorFemField
from psydac.api.discretization import discretize

from psydac.mapping.discrete import SplineMapping

import numpy as np
from scipy.linalg import eig as eig_solver
from mpi4py import MPI
import pytest

import os

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')
# ...

#==============================================================================
def run_poisson_2d_dir(filename, comm=None):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = FunctionSpace('V', domain)

    x,y = domain.coordinates

    v = element_of_space(V, name='v')
    u = element_of_space(V, name='u')

    a = BilinearForm((v,u), dot(grad(v), grad(u)))

    glt_a = GltExpr(a)
    # ...

    # ... create the computational domain from a topological domain
    domain_h = discretize(domain, filename=filename, comm=comm)
    # ...

    # ... discrete spaces
    Vh = discretize(V, domain_h)
    # ...

    # ... dsicretize the equation using Dirichlet bc
    ah = discretize(a, domain_h, [Vh, Vh])
    # ...

    # ... dsicretize the glt symbol
    glt_ah = discretize(glt_a, domain_h, [Vh, Vh])
    # ...

    # ...
    eigh = glt_ah.eig()
    eigh = eigh.ravel()
    eigh.sort()
    # ...

    # ... use eigenvalue solver
    M = ah.assemble().tosparse().todense()
    w, v = eig_solver(M)
    eig = w.real
    eig.sort()
    # ...

    # ...
    error = np.linalg.norm(eig-eigh) / Vh.nbasis
    # ...

    return error


#==============================================================================
def test_api_glt_poisson_2d_dir_identity():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    error = run_poisson_2d_dir(filename)
    assert(np.allclose([error], [0.029738578422276986]))


#==============================================================================
def test_api_glt_poisson_2d_dir_collela():
    filename = os.path.join(mesh_dir, 'collela_2d.h5')

    error = run_poisson_2d_dir(filename)
    assert(np.allclose([error], [0.04655602895206486]))

#==============================================================================
def test_api_glt_poisson_2d_dir_quart_circle():
    filename = os.path.join(mesh_dir, 'quart_circle.h5')

    error = run_poisson_2d_dir(filename)
    assert(np.allclose([error], [0.04139096668630673]))

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()

#test_api_glt_poisson_2d_dir_identity()
#test_api_glt_poisson_2d_dir_collela()
#test_api_glt_poisson_2d_dir_quart_circle()
