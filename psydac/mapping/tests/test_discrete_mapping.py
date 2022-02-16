import numpy as np
import pyevtk.hl
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


@pytest.mark.parametrize('geometry_file', ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
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

        x_mesh, y_mesh, z_mesh = mapping.build_mesh(refine_factor=k)

        if mapping.ldim == 2:

            eta1 = [glob_points[0][i // (k + 1)][i % (k + 1)] for i in range(x_mesh.shape[0])]
            eta2 = [glob_points[1][i // (k + 1)][i % (k + 1)] for i in range(x_mesh.shape[1])]

            pcoords = np.array([[mapping([e1, e2]) for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0:1]
            y_mesh_l = pcoords[..., 1:2]
            z_mesh_l = np.zeros_like(x_mesh_l)

        if mapping.ldim == 3:

            eta1 = [glob_points[0][i // (k + 1)][i % (k + 1)] for i in range(x_mesh.shape[0])]
            eta2 = [glob_points[1][i // (k + 1)][i % (k + 1)] for i in range(x_mesh.shape[1])]
            eta3 = [glob_points[2][i // (k + 1)][i % (k + 1)] for i in range(x_mesh.shape[2])]

            pcoords = np.array([[[mapping([e1, e2, e3]) for e3 in eta3] for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]
            z_mesh_l = pcoords[..., 2]



        assert np.allclose(x_mesh, x_mesh_l)
        assert np.allclose(y_mesh, y_mesh_l)
        assert np.allclose(z_mesh, z_mesh_l)
