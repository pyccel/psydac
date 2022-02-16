import pytest
import os

from sympde.topology import Domain
from psydac.api.discretization import discretize

try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']
except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')


@pytest.mark.parametrize('geometry_file', ['collela_2d.h5', 'collela_3d.h5', 'bent_pipe.h5'])
def test_build_mesh(geometry_file):
    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)

    domainh = discretize(domain, filename=filename)

    for mapping in domainh.mappings.values():
        mapping.build_mesh()
