import numpy as np
import pytest
import os

from sympde.topology import Domain
from psydac.api.discretization import discretize
from psydac.core.bsplines import quadrature_grid
from psydac.utilities.quadratures import gauss_legendre

try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']
except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')


@pytest.mark.parametrize('geometry_file', ['collela_2d.h5', 'collela_3d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('k', [1, 2, 3])
def test_build_mesh(geometry_file, k):
    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)

    domainh = discretize(domain, filename=filename)

    for mapping in domainh.mappings.values():
        space = mapping.space
        glob_points = []
        for i in range(mapping.ldim):
            grid_i = space.breaks[i]

            # Gauss-Legendre quadrature rule
            u, w = gauss_legendre(k)
            u = u[::-1]
            w = w[::-1]

            # Grids
            glob_points_i, _ = quadrature_grid(grid_i, u, w)

            glob_points_i[0, 0] = grid_i[0]
            glob_points_i[-1, -1] = grid_i[-1]

            glob_points.append(glob_points_i)

        mesh_fast = mapping.build_mesh(refine_factor=k)

        mesh_slow = np.empty_like(mesh_fast)



