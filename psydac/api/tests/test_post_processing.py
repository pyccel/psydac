import pytest
import os
import numpy as np

from sympde.topology import Square, ScalarFunctionSpace, VectorFunctionSpace, Domain
from psydac.api.discretization import discretize
from psydac.fem.basic import FemField
from psydac.api.postprocessing import OutputManager, PostProcessManager

try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..','..')
    mesh_dir = os.path.join(base_dir, 'mesh')

def test_OutputManager():

    domain = Square('D')
    A = ScalarFunctionSpace('A', domain, kind='H1')
    B = VectorFunctionSpace('B', domain, kind=None)

    domain_h = discretize(domain, ncells=[5, 5])

    Ah = discretize(A, domain_h, degree=[3, 3])
    Bh = discretize(B, domain_h, degree=[2, 2])

    uh = FemField(Ah)
    vh = FemField(Bh)

    Om = OutputManager('file.yml', 'file.h5')
    Om.add_spaces(Ah=Ah, Bh=Bh)

    Om.set_static().export_fields(uh=uh, vh=vh)

    Om.add_snapshot(t=0., ts=0).export_fields(uh=uh, vh=vh)
    Om.add_snapshot(t=1., ts=1).export_fields(uh=uh, vh=vh)

    expected_spaces_info = {'ndim': 2,
                            'fields': 'file.h5',
                            'patches': [{'name': 'D',
                                         'scalar_spaces': [{'name': 'Ah',
                                                            'ldim': 2,
                                                            'kind': 'h1',
                                                            'dtype': "<class 'float'>",
                                                            'rational': False,
                                                            'periodic': [False, False],
                                                            'degree': [3, 3],
                                                            'basis': ['B', 'B'],
                                                            'knots': [
                                                                [0.0, 0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0, 1.0],
                                                                [0.0, 0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0, 1.0]
                                                            ]
                                                            },
                                                           {'name': 'Bh[0]',
                                                            'ldim': 2,
                                                            'kind': 'undefined',
                                                            'dtype': "<class 'float'>",
                                                            'rational': False,
                                                            'periodic': [False, False],
                                                            'degree': [2, 2],
                                                            'basis': ['B', 'B'],
                                                            'knots': [
                                                                [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0],
                                                                [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0]
                                                            ]
                                                            },
                                                           {'name': 'Bh[1]',
                                                            'ldim': 2,
                                                            'kind': 'undefined',
                                                            'dtype': "<class 'float'>",
                                                            'rational': False,
                                                            'periodic': [False, False],
                                                            'degree': [2, 2],
                                                            'basis': ['B', 'B'],
                                                            'knots': [
                                                                [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0],
                                                                [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                 0.6000000000000001, 0.8, 1.0, 1.0, 1.0]
                                                            ]
                                                            }
                                                           ],
                                         'vector_spaces': [{'name': 'Bh',
                                                            'kind': 'undefined',
                                                            'components': [
                                                                {'name': 'Bh[0]',
                                                                 'ldim': 2,
                                                                 'kind': 'undefined',
                                                                 'dtype': "<class 'float'>",
                                                                 'rational': False,
                                                                 'periodic': [False, False],
                                                                 'degree': [2, 2],
                                                                 'basis': ['B', 'B'],
                                                                 'knots': [
                                                                     [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                      0.6000000000000001, 0.8, 1.0, 1.0, 1.0],
                                                                     [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                      0.6000000000000001, 0.8, 1.0, 1.0, 1.0]
                                                                 ]
                                                                 },
                                                                {'name': 'Bh[1]',
                                                                 'ldim': 2,
                                                                 'kind': 'undefined',
                                                                 'dtype': "<class 'float'>",
                                                                 'rational': False,
                                                                 'periodic': [False, False],
                                                                 'degree': [2, 2],
                                                                 'basis': ['B', 'B'],
                                                                 'knots': [
                                                                     [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                      0.6000000000000001, 0.8, 1.0, 1.0, 1.0],
                                                                     [0.0, 0.0, 0.0, 0.2, 0.4,
                                                                      0.6000000000000001, 0.8, 1.0, 1.0, 1.0]
                                                                 ]
                                                                 }
                                                            ]
                                                            }]
                                         }]
                            }

    assert(Om._spaces_info == expected_spaces_info)


def test_PostProcess_Manager():
    # =================================================================
    # Part 1: Running a simulation
    # =================================================================
    geometry_file = mesh_dir+'/identity_2d.h5'

    domain = Domain.from_file(geometry_file)

    V1 = ScalarFunctionSpace('V1', domain, kind='l2')
    V2 = VectorFunctionSpace('V2', domain, kind='hcurl')
    V3 = VectorFunctionSpace('V3', domain, kind='hdiv')

    domainh = discretize(domain, filename=geometry_file)

    V1h = discretize(V1, domainh, degree=[4, 3])
    V2h = discretize(V2, domainh, degree=[[3, 2], [2, 3]])
    V3h = discretize(V3, domainh, degree=[[1, 1], [1, 1]])

    uh = FemField(V1h)
    vh = FemField(V2h)
    wh = FemField(V3h)

    # Output Manager Initialization
    output = OutputManager('space_example.yml', 'fields_example.h5')
    output.add_spaces(V1h=V1h, V2h=V2h, V3=V3h)
    output.set_static().export_fields(u=uh, v=vh)

    uh_grids = []
    vh_grids = []

    for i in range(15):
        uh.coeffs[:] = np.random.random(size=uh.coeffs[:].shape)

        vh.coeffs[0][:] = np.random.random(size=vh.coeffs[0][:].shape)
        vh.coeffs[1][:] = np.random.random(size=vh.coeffs[1][:].shape)

        # Export to HDF5
        output.add_snapshot(t=float(i), ts=i).export_fields(u=uh, v=vh, w=wh)

        # Saving for comparisons
        uh_grid = V1h.eval_fields(uh, refine_factor=2)
        vh_grid_x, vh_grid_y = V2h.eval_fields(vh, refine_factor=2)
        uh_grids.append(uh_grid)
        vh_grids.append((vh_grid_x, vh_grid_y))

    output.export_space_info()
    # End of the "simulation"

    # =================================================================================
    # Part 2: Post Processing
    # =================================================================================

    post = PostProcessManager(geometry_filename=geometry_file,
                              space_filename='space_example.yml',
                              fields_filename='fields_example.h5')

    post.reconstruct_scope()

    V1h_new = post.spaces['V1h']
    V2h_new = post.spaces['V2h']

    for i in range(len(uh_grids)):
        snapshot = post.fields[i]
        u_new = snapshot['fields']['u']
        v_new = snapshot['fields']['v']

        uh_grid_new = V1h_new.eval_fields(u_new, refine_factor=2)
        vh_grid_x_new, vh_grid_y_new = V2h_new.eval_fields(v_new, refine_factor=2)

        assert np.allclose(uh_grid_new, uh_grids[i])
        assert np.allclose(vh_grid_x_new, vh_grids[i][0])
        assert np.allclose(vh_grid_y_new, vh_grids[i][1])

    post.export_to_vtk('example_None', dt=None, u='u', v='v', w='w')
    post.export_to_vtk('example_int', dt=5, u='u', v='v', w='w')
    post.export_to_vtk('example_list', dt=[9,5,6,3], u='u', v='v', w='w')


if __name__ == '__main__':
    test_OutputManager()
    test_PostProcess_Manager()
