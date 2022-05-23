import numpy as np
import pytest
import os

from sympde.topology import Domain
from sympde.topology.analytical_mapping import PolarMapping
from sympy import rad
from psydac.api.discretization import discretize
from psydac.utilities.utils import refine_array_1d

from psydac.mapping.discrete import NurbsMapping
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.splines import SplineSpace
from psydac.cad.geometry import export_nurbs_to_hdf5

from igakit.cad import circle, ruled


try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']
except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

# Checking for float equality
ATOL=1e-15
RTOL=1e-15

@pytest.mark.parametrize('geometry_file', ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('refinement', [2, 3, 4])
def test_build_mesh(geometry_file, refinement):
    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)
    domainh = discretize(domain, filename=filename)

    for mapping in domainh.mappings.values():
        space = mapping.space

        grid = [refine_array_1d(space.breaks[i], refinement, remove_duplicates=False) for i in range(mapping.ldim)]

        x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid, npts_per_cell=refinement + 1)

        if mapping.ldim == 2:

            eta1, eta2 = grid

            pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0:1]
            y_mesh_l = pcoords[..., 1:2]
            z_mesh_l = np.zeros_like(x_mesh_l)

        elif mapping.ldim == 3:

            eta1, eta2, eta3 = grid

            pcoords = np.array([[[mapping(e1, e2, e3) for e3 in eta3] for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]
            z_mesh_l = pcoords[..., 2]

        else:
            assert False

        assert x_mesh.flags['C_CONTIGUOUS'] and y_mesh.flags['C_CONTIGUOUS'] and z_mesh.flags['C_CONTIGUOUS']

        assert np.allclose(x_mesh, x_mesh_l)
        assert np.allclose(y_mesh, y_mesh_l)
        assert np.allclose(z_mesh, z_mesh_l)

def test_nurbs_circle():
    rmin, rmax = 0.2, 1
    c1, c2 = 0, 0

    # Igakit
    c_ext = circle(radius=rmax, center=(c1, c2))
    c_int = circle(radius=rmin, center=(c1, c2))

    disk = ruled(c_ext, c_int).transpose()

    w  = disk.weights
    k = disk.knots
    control = disk.points
    d = disk.degree

    # Psydac
    spaces = [SplineSpace(degree, knot) for degree, knot in zip(d, k)]
    T = TensorFemSpace(*spaces)
    mapping = NurbsMapping.from_control_points_weights(T, control_points=control[..., :2], weights=w)

    x1_pts = np.linspace(0, 1, 10) 
    x2_pts = np.linspace(0, 1, 10)

    for x2 in x2_pts:
        for x1 in x1_pts:
            x_p, y_p = mapping(x1, x2)
            x_i, y_i, z_i = disk(x1, x2)

            assert np.allclose((x_p, y_p), (x_i, y_i), atol=ATOL, rtol=RTOL) 

            J_p = mapping.jac_mat(x1, x2)
            J_i = disk.gradient(u=x1, v=x2)

            assert np.allclose(J_i[:2], J_p, atol=ATOL, rtol=RTOL)
