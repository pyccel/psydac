#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eig as eig_solver

from sympde.calculus import grad, dot
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of
from sympde.topology import Domain
from sympde.expr     import BilinearForm, integral

from gelato.expr import GltExpr

from psydac.api.discretization import discretize

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

#==============================================================================
def run_poisson_2d_dir(filename, comm=None):

    # ... abstract model
    domain = Domain.from_file(filename)

    V = ScalarFunctionSpace('V', domain)

    x,y = domain.coordinates

    v = element_of(V, name='v')
    u = element_of(V, name='u')

    int_0 = lambda expr: integral(domain , expr)

    a = BilinearForm((v,u), int_0(dot(grad(v), grad(u))))

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
    glt_ah = discretize(glt_a, domain_h, [Vh, Vh], expand=True)
    # ...

    # ...
    eigh = glt_ah.eig()
    eigh = eigh.ravel()
    eigh.sort()
    # ...

    # ... use eigenvalue solver
    M = ah.assemble().toarray()
    w, v = eig_solver(M)
    eig = w.real
    eig.sort()
    # ...

    # ...
    error = np.linalg.norm(eig-eigh) / Vh.nbasis
    # ...

    return error


#==============================================================================
@pytest.mark.xfail
def test_api_glt_poisson_2d_dir_identity():
    filename = os.path.join(mesh_dir, 'identity_2d.h5')

    error = run_poisson_2d_dir(filename)
    assert(np.allclose([error], [0.029738578422276986]))


#==============================================================================
@pytest.mark.xfail
def test_api_glt_poisson_2d_dir_collela():
    filename = os.path.join(mesh_dir, 'collela_2d.h5')

    error = run_poisson_2d_dir(filename)
    assert(np.allclose([error], [0.04655602895206486]))

#==============================================================================
@pytest.mark.xfail
def test_api_glt_poisson_2d_dir_quarter_annulus():
    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')

    error = run_poisson_2d_dir(filename)
    assert(np.allclose([error], [0.04139096668630673]))

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

#test_api_glt_poisson_2d_dir_identity()
#test_api_glt_poisson_2d_dir_collela()
#test_api_glt_poisson_2d_dir_quarter_annulus()
