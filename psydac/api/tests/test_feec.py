#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
import pytest

from sympde.topology import Line, Square, Cube, Domain, Derham
from sympde.topology.analytical_mapping import AffineMapping, PolarMapping

from psydac.api.discretization import discretize

#==============================================================================
# Functions of the logical coordinates, and affine mappings, in 1D, 2D and 3D
f = {1: lambda x: np.sin(np.pi * x) + x,
     2: lambda x, y: np.sin(np.pi * x) * np.cos(y) + x * y,
     3: lambda x, y, z: np.sin(np.pi * x) * np.cos(y) + x * y * z}
g = {2: lambda x, y: x**2 - y,
     3: lambda x, y, z: x**2 - y * z}
h = {3: lambda x, y, z: np.exp(-x) * z}

functions = {1: [f[1], f[1]],
             2: [f[2], (f[2], g[2]), f[2]],
             3: [f[3], (f[3], g[3], h[3]), (f[3], g[3], h[3]), f[3]]}

affine_params = {1: dict(c1=1, a11=2),
                 2: dict(c1=1, c2=0, a11=2, a12=1, a21=0, a22=3),
                 3: dict(c1=1, c2=0, c3=2, a11=2, a12=1, a13=0,
                         a21=0, a22=3, a23=1, a31=1, a32=0, a33=2)}

ncells = {1: [5], 2: [3, 5], 3: [3, 4, 5]}
degree = {1: [2], 2: [2, 3], 3: [2, 2, 3]}

def project(domain, funcs, **kwargs):
    """Project the functions onto the de Rham sequence of the domain, and return the coefficients."""
    dim = domain.dim
    sequence = ['h1', 'hcurl', 'l2'] if dim == 2 else None
    domain_h = discretize(domain, ncells=ncells[dim])
    derham_h = discretize(Derham(domain, sequence), domain_h, degree=degree[dim])
    return [P(fun).coeffs.toarray() for P, fun in zip(derham_h.projectors(**kwargs), funcs)]

#==============================================================================
@pytest.mark.parametrize('dim', [1, 2, 3])
def test_DiscreteDeRham_projectors_logical(dim):

    logical_domain = {1: Line, 2: Square, 3: Cube}[dim]('Omega')
    mapped_domain = AffineMapping('F', dim=dim, **affine_params[dim])(logical_domain)

    # With logical=True, the projection does not depend on the mapping (see #153)
    expected = project(logical_domain, functions[dim])
    actual   = project(mapped_domain, functions[dim], logical=True)
    for a, e in zip(actual, expected):
        assert np.array_equal(a, e)

    # By default, the functions are pulled back, which changes the result
    physical = project(mapped_domain, functions[dim])
    assert not all(np.allclose(p, e) for p, e in zip(physical, expected))

#==============================================================================
def test_MultipatchDiscreteDeRham_projectors_logical():

    A = Square('A', bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B', bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))
    connectivity = [((0, 1, 1), (1, 1, -1), 1)]

    logical_domain = Domain.join([A, B], connectivity, 'Omega')
    mapped_domain = Domain.join([PolarMapping('M1', 2, c1=0., c2=0., rmin=0., rmax=1.)(A),
                                 PolarMapping('M2', 2, c1=0., c2=0., rmin=0., rmax=1.)(B)],
                                connectivity, 'Omega')

    # The multipatch projectors act on a list of functions, one per patch
    funcs = [[fun, fun] for fun in functions[2]]

    nquads = [p + 1 for p in degree[2]]
    expected = project(logical_domain, funcs, nquads=nquads)
    actual   = project(mapped_domain, funcs, nquads=nquads, logical=True)
    for a, e in zip(actual, expected):
        assert np.array_equal(a, e)
