import pytest
import os
import glob
import numpy as np

from sympde.topology import Square, ScalarFunctionSpace, VectorFunctionSpace, Domain
from psydac.api.discretization import discretize
from psydac.fem.basic import FemField
from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.utilities.utils import refine_array_1d

try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

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

    Om.set_static()
    Om.export_fields(uh_static=uh)
    Om.export_fields(vh_static=vh)

    Om.add_snapshot(t=0., ts=0)
    Om.export_fields(uh=uh, vh=vh)

    Om.add_snapshot(t=1., ts=1)
    Om.export_fields(uh=uh)
    Om.export_fields(vh=vh)

    with pytest.raises(AssertionError):
        Om.export_fields(uh_static=uh)

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

    assert Om._spaces_info == expected_spaces_info
    Om.export_space_info()

    # Remove files
    os.remove('file.h5')
    os.remove('file.yml')


@pytest.mark.parametrize('geometry', ['identity_2d.h5',
                                      'identity_3d.h5',
                                      'bent_pipe.h5'])
@pytest.mark.parametrize('refinement', [1, 2, 3])
def test_PostProcessManager(geometry, refinement):
    # =================================================================
    # Part 1: Running a simulation
    # =================================================================
    geometry_file = os.path.join(mesh_dir, geometry)
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

    npts_per_cell = refinement + 1

    grid = [refine_array_1d(V1h.breaks[i], refinement, remove_duplicates=False) for i in range(domainh.ldim)]

    # Output Manager Initialization
    output = OutputManager('space_example.yml', 'fields_example.h5')
    output.add_spaces(V1h=V1h, V2h=V2h, V3=V3h)
    output.set_static()
    output.export_fields(w=wh)

    uh_grids = []
    vh_grids = []

    for i in range(15):
        uh.coeffs[:] = np.random.random(size=uh.coeffs[:].shape)

        vh.coeffs[0][:] = np.random.random(size=vh.coeffs[0][:].shape)
        vh.coeffs[1][:] = np.random.random(size=vh.coeffs[1][:].shape)

        # Export to HDF5
        output.add_snapshot(t=float(i), ts=i)
        output.export_fields(u=uh, v=vh)

        # Saving for comparisons
        uh_grid = V1h.eval_fields(grid, uh, npts_per_cell=npts_per_cell)
        vh_grid =  V2h.eval_fields(grid, vh, npts_per_cell=npts_per_cell)
        vh_grid_x, vh_grid_y = vh_grid[0][0], vh_grid[0][1]
        uh_grids.append(uh_grid)
        vh_grids.append((vh_grid_x, vh_grid_y))

    output.export_space_info()
    # End of the "simulation"

    # =================================================================================
    # Part 2: Post Processing
    # =================================================================================

    post = PostProcessManager(geometry_file=geometry_file,
                              space_file='space_example.yml',
                              fields_file='fields_example.h5')


    V1h_new = post.spaces['V1h']
    V2h_new = post.spaces['V2h']

    for i in range(len(uh_grids)):
        post.load_snapshot(i, 'u', 'v')
        snapshot = post._snapshot_fields

        u_new = snapshot['u']
        v_new = snapshot['v']

        uh_grid_new = V1h_new.eval_fields(grid, u_new, npts_per_cell=npts_per_cell)
        vh_grid_new = V2h_new.eval_fields(grid, v_new, npts_per_cell=npts_per_cell)
        vh_grid_x_new, vh_grid_y_new = vh_grid_new[0][0], vh_grid_new[0][1]

        assert np.allclose(uh_grid_new, uh_grids[i])
        assert np.allclose(vh_grid_x_new, vh_grids[i][0])
        assert np.allclose(vh_grid_y_new, vh_grids[i][1])

    mesh, static_fields = post.export_to_vtk('example_None', 
                                             grid, 
                                             npts_per_cell=npts_per_cell, 
                                             snapshots='none', 
                                             fields={'field1': 'u', 'field2': 'v', 'field3': 'w'}, 
                                             debug=True)
    assert list(static_fields[0].keys()) == ['field3']

    mesh, all_fields = post.export_to_vtk('example_all', 
                                          grid, 
                                          npts_per_cell=npts_per_cell, 
                                          snapshots='all', 
                                          fields={'field1': 'u', 'field2': 'v', 'field3': 'w'},
                                          debug=True)
    
    assert list(all_fields[0].keys()) == ['field3']
    assert all(list(all_fields[i + 1].keys()) == ['field1', 'field2'] for i in range(len(post._snapshot_list)))

    mesh, snapshot_fields = post.export_to_vtk('example_list', 
                                               grid, 
                                               npts_per_cell=npts_per_cell, 
                                               snapshots=[9, 5, 6, 3], 
                                               fields={'field1': 'u', 'field2': 'v', 'field3': 'w'},
                                               debug=True)

    assert all(list(snapshot_fields[i].keys()) == ['field1', 'field2'] for i in range(4))

    # Clear files
    for f in glob.glob("example*.vts"): #VTK files
        os.remove(f)
    os.remove('space_example.yml')
    os.remove('fields_example.h5')


@pytest.mark.parallel
def test_multipatch_parallel_export(interactive=False):
    bounds1   = (0.5, 1.)
    bounds2_A = (0, np.pi/2)
    bounds2_B = (np.pi/2, np.pi)

    A = Square('A',bounds1=bounds1, bounds2=bounds2_A)
    B = Square('B',bounds1=bounds1, bounds2=bounds2_B)

    domain = A.join(B, name='domain',
                    bnd_minus=A.get_boundary(axis=1, ext=1),
                    bnd_plus=B.get_boundary(axis=1, ext=-1))

    Va = ScalarFunctionSpace('Va', A)
    Vb = ScalarFunctionSpace('Vb', B)
    V = ScalarFunctionSpace('V', domain)
    Vv = VectorFunctionSpace('Vv', domain)
    Vva = VectorFunctionSpace('Vva', A)

    Om = OutputManager('spaces_multipatch.yml', 'fields_multipatch.h5', comm=comm)

    domain_h = discretize(domain, ncells = [15, 15], comm=comm)
    Ah = discretize(A, ncells = [5, 5], comm=comm)
    Bh = discretize(B, ncells = [5, 5])

    Vh = discretize(V, domain_h, degree=[3, 3])
    Vah = discretize(Va, Ah, degree=[3, 3])
    Vbh = discretize(Vb, Bh, degree=[3, 3])

    Vvh = discretize(Vv, domain_h, degree=[3,3])
    Vvah = discretize(Vva, Ah, degree=[3,3])

    uh = FemField(Vh)
    uah = FemField(Vah)
    ubh = FemField(Vbh)

    uvh = FemField(Vvh)
    uvah = FemField(Vvah)

    Om.add_spaces(V=Vh)
    Om.add_spaces(Va=Vah)
    Om.add_spaces(Vb=Vbh)
    Om.add_spaces(Vv=Vvh)
    Om.add_spaces(Vva=Vvah)

    Om.set_static()
    
    Om.export_fields(field_A=uah, field_B=ubh,
                     field_V=uh, uvh=uvh, uvah=uvah)
    Om.add_snapshot(0., 0)
    Om.export_fields(fA=uah, f_B=ubh,
                     f_V=uh, uvh_0=uvh, uvah_0=uvah)
    Om.export_space_info()
    Om.close()

    comm.Barrier()
    if comm.Get_rank() == 0 and not interactive:
        os.remove('spaces_multipatch.yml')
        os.remove('fields_multipatch.h5')


@pytest.mark.parallel
@pytest.mark.parametrize('geometry', ['identity_2d.h5',
                                      'identity_3d.h5',
                                      'bent_pipe.h5'])
def test_parallel_export(geometry, interactive=False):
    rank = comm.Get_rank()
    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domainh = discretize(domain, filename=filename, comm=comm)

    V = ScalarFunctionSpace('V', domain, kind='H1')

    Vh = discretize(V, domainh, degree=[3] * domainh.ldim)

    f = FemField(Vh)

    f.coeffs[:] = rank

    Om = OutputManager('space_test_export.yml', 'fields_test_export.h5', comm=comm, mode='w')

    Om.add_spaces(Vh=Vh)
    Om.set_static()
    Om.export_fields(f=f)

    grid = [np.linspace(Vh.breaks[i][0], Vh.breaks[i][-1], Vh.ncells[i] * 2) for i in range(Vh.ldim)]
    npts_per_cell = [2] * len(grid)
    
    for i in range(15):
        f.coeffs[:] = i
        Om.add_snapshot(t=float(i), ts=i)
        Om.export_fields(f_t=f)
    Om.export_space_info()
    Om.close()
    comm.Barrier()

    Pm = PostProcessManager(geometry_file=filename, 
                            space_file='space_test_export.yml', 
                            fields_file='fields_test_export.h5', 
                            comm=comm)

    Pm.export_to_vtk('test', grid=grid, npts_per_cell=npts_per_cell, 
                     fields={'f_t': 'time'}, snapshots='all')

    comm.Barrier()

    if rank == 0 and not interactive:
        for f in glob.glob('test*.*vts'):
            os.remove(f)
        os.remove('space_test_export.yml')
        os.remove('fields_test_export.h5')


if __name__ == "__main__":
    import sys
    # Meant to showcase examples 
    test_multipatch_parallel_export(True)
    test_parallel_export(sys.argv[1], True)