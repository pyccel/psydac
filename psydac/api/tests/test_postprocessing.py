import pytest
import os
import glob
import numpy as np

from sympde.topology import Square, Cube, ScalarFunctionSpace, VectorFunctionSpace, Domain, Derham
from sympde.topology.analytical_mapping import IdentityMapping, AffineMapping

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
                                         'breakpoints': [
                                             [0., 0.2, 0.4, 0.6000000000000001, 0.8, 1.],
                                             [0., 0.2, 0.4, 0.6000000000000001, 0.8, 1.]
                                         ],
                                         'scalar_spaces': [{'name': 'Ah',
                                                            'ldim': 2,
                                                            'kind': 'h1',
                                                            'dtype': "<class 'float'>",
                                                            'rational': False,
                                                            'periodic': [False, False],
                                                            'degree': [3, 3],
                                                            'multiplicity': [1, 1],
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
                                                            'multiplicity': [1, 1],
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
                                                            'multiplicity': [1, 1],
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
                                                                 'multiplicity': [1, 1],
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
                                                                 'multiplicity': [1, 1],
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
    Om.export_space_info()
    assert Om._space_info == expected_spaces_info
    
    # Remove files
    os.remove('file.h5')
    os.remove('file.yml')


@pytest.mark.parametrize('domain', [Square(), Cube()])
def test_reconstruct_spaces_topological_domain(domain):
    dim = domain.dim

    domain_h = discretize(domain, ncells=[5] * dim)

    Vh1 = ScalarFunctionSpace('Vh1', domain, kind='h1')
    Vvh1 = VectorFunctionSpace('Vvh1', domain, kind='h1')

    Vl2 = ScalarFunctionSpace('Vl2', domain, kind='l2')
    Vvl2 = VectorFunctionSpace('Vvl2', domain, kind='l2')

    Vhdiv = VectorFunctionSpace('Vhdiv', domain, kind='hdiv')

    Vhcurl = VectorFunctionSpace('Vhcurl', domain, kind='hcurl')

    degree1 = [2, 3, 4][:dim]
    degree2 = [5, 5, 5][:dim]
    degree3 = [2, 3, 2][:dim]
    degrees = [degree1, degree2, degree3]
    random_gen = np.random.default_rng()
    rints = random_gen.integers(low=0, high=3, size=6)

    Vhh1 = discretize(Vh1, domain_h, degree=degrees[rints[0]])
    Vvhh1 = discretize(Vvh1, domain_h, degree=degrees[rints[1]])
    Vhl2 = discretize(Vl2, domain_h, degree=degrees[rints[2]])
    Vvhl2 = discretize(Vvl2, domain_h, degree=degrees[rints[3]])
    Vhhdiv = discretize(Vhdiv, domain_h, degree=degrees[rints[4]])
    Vhhcurl = discretize(Vhcurl, domain_h, degree=degrees[rints[5]])

    Om1 = OutputManager("space.yml", "t.h5")
    Om1.add_spaces(Vector_h1=Vvhh1, Vector_l2=Vvhl2, Vector_hdiv=Vhhdiv, Vector_hcurl=Vhhcurl)
    Om1.add_spaces(Scalar_h1=Vhh1, Scalar_l2=Vhl2)
    
    space_info_1 = Om1.space_info

    Om1.export_space_info()

    Pm = PostProcessManager(domain=domain, space_file="space.yml", fields_file="t.h5")

    Om2 = OutputManager("space_2.yml", "t.h5")
    Om2.add_spaces(**Pm.spaces)

    space_info_2 = Om2.space_info
    Om2.export_space_info()

    assert space_info_1 == space_info_2
    os.remove("space.yml")
    os.remove("space_2.yml")


@pytest.mark.parametrize('domain, seq', [(Square(), ['h1', 'hdiv', 'l2']), (Square(), ['h1', 'hcurl', 'l2']), (Cube(), None)])
def test_reconstruct_DeRhamSequence_topological_domain(domain, seq):
    deRham = Derham(domain, sequence=seq)

    domain_h  = discretize(domain, ncells=[5]*domain.dim)
    derham_h = discretize(deRham, domain_h, degree=[2]*domain.dim)

    Om = OutputManager('space_derham.yml', 'f.h5')
    Om.add_spaces(**dict(((Vh.symbolic_space.name, Vh) for Vh in derham_h.spaces)))
    Om.export_space_info()

    Pm = PostProcessManager(domain=domain, space_file='space_derham.yml', fields_file='f.h5')

    Om2 = OutputManager('space_derham_2.yml', 'f.h5')

    Om2.add_spaces(**Pm.spaces)
    Om2.export_space_info()
    space_info_1 = Om.space_info
    space_info_2 = Om2.space_info

    patches_1 = space_info_1['patches']
    patches_2 = space_info_2['patches']

    for patch1 in patches_1:
        for patch2 in patches_2:
            if patch1['name'] == patch2['name']:
                for key in patch1.keys():
                    value1 = patch1[key]
                    value2 = patch2[key]
                    assert type(value1) == type(value2)
                    if 'space' in key:
                        for space1 in value1:
                            for space2 in value2:
                                if space2['name'] == space1['name']:
                                    for key_space in space1.keys():
                                        assert space1[key_space] == space2[key_space]
                    else:
                        assert value1 == value2

    os.remove('space_derham_2.yml')
    os.remove('space_derham.yml')


@pytest.mark.parametrize('geometry, seq', [('identity_2d.h5', ['h1', 'hdiv', 'l2']),
                                           ('identity_2d.h5', ['h1', 'hcurl', 'l2']),
                                           ('identity_3d.h5', None),
                                           ('bent_pipe.h5', ['h1', 'hdiv', 'l2']),
                                           ('bent_pipe.h5', ['h1', 'hcurl', 'l2'])])
def test_reconstruct_DeRhamSequence_discrete_domain(geometry, seq):
    
    geometry_file = os.path.join(mesh_dir, geometry)
    domain = Domain.from_file(geometry_file)

    deRham = Derham(domain, sequence=seq)

    domain_h  = discretize(domain, filename=geometry_file)
    derham_h = discretize(deRham, domain_h, degree=[2]*domain.dim)

    Om = OutputManager('space_derham.yml', 'f.h5')
    Om.add_spaces(**dict(((Vh.symbolic_space.name, Vh) for Vh in derham_h.spaces)))
    Om.export_space_info()

    Pm = PostProcessManager(domain=domain, space_file='space_derham.yml', fields_file='f.h5')

    Om2 = OutputManager('space_derham_2.yml', 'f.h5')

    Om2.add_spaces(**Pm.spaces)
    Om2.export_space_info()
    space_info_1 = Om.space_info
    space_info_2 = Om2.space_info

    patches_1 = space_info_1['patches']
    patches_2 = space_info_2['patches']

    for patch1 in patches_1:
        for patch2 in patches_2:
            if patch1['name'] == patch2['name']:
                for key in patch1.keys():
                    value1 = patch1[key]
                    value2 = patch2[key]
                    assert type(value1) == type(value2)
                    if 'space' in key:
                        for space1 in value1:
                            for space2 in value2:
                                if space2['name'] == space1['name']:
                                    for key_space in space1.keys():
                                        assert space1[key_space] == space2[key_space]
                    else:
                        assert value1 == value2

    os.remove('space_derham_2.yml')
    os.remove('space_derham.yml')



@pytest.mark.parametrize('geometry', ['identity_2d.h5', 'identity_3d.h5','bent_pipe.h5'])
def test_PostProcessManager(geometry):
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

    npts_per_cell = 2

    grid = [refine_array_1d(V1h.breaks[i], 1, remove_duplicates=False) for i in range(domainh.ldim)]

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

    mesh_and_infos, static_fields = post.export_to_vtk('example_None', 
                                             grid, 
                                             npts_per_cell=npts_per_cell, 
                                             snapshots='none', 
                                             fields={'field1': 'u', 'field2': 'v', 'field3': 'w'}, 
                                             debug=True)
    assert list(static_fields[0].keys()) == ['field3']

    mesh_and_infos, all_fields = post.export_to_vtk('example_all', 
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
    for f in glob.glob("example*.vtu"): #VTK files
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
@pytest.mark.parametrize('geometry', ['identity_2d.h5','identity_3d.h5','bent_pipe.h5'])
@pytest.mark.parametrize('kind', ['h1', 'l2', 'hdiv', 'hcurl'])
@pytest.mark.parametrize('space', [ScalarFunctionSpace, VectorFunctionSpace])
def test_parallel_export_discrete_domain(geometry, kind, space, interactive=False):
    rank = comm.Get_rank()

    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domain_h = discretize(domain, filename=filename, comm=comm)

    dim = domain.dim

    symbolic_space = space('V', domain, kind=kind)

    degree = [2, 3, 4][:dim]
    space_h = discretize(symbolic_space, domain_h, degree=degree)

    field = FemField(space_h)

    Om = OutputManager("space_export_discrete.yml", "fields_export_discrete.h5", comm=comm)

    Om.add_spaces(space_h=space_h)
     
    Om.set_static()
    Om.export_fields(field_static=field)
    Om.add_snapshot(t=0., ts=0)
    Om.export_fields(field_t=field)
    
    Om.export_space_info()
    Om.close()
    
    Pm = PostProcessManager(geometry_file=filename, space_file="space_export_discrete.yml", fields_file="fields_export_discrete.h5", comm=comm)
    try:
        grid = [refine_array_1d(space_h.breaks[i], 1, False) for i in range(space_h.ldim)]
    except AttributeError:
        grid = [refine_array_1d(space_h.spaces[0].breaks[i], 1, False) for i in range(space_h.ldim)]

    npts_per_cell = [2] * dim


    (x_mesh, y_mesh, z_mesh, conn, offsets, celltypes), \
    pointDatas = Pm.export_to_vtk("test_export_discrete_r", grid, npts_per_cell=npts_per_cell,
                                  snapshots='all', fields={'f1': 'field_static', 'f2': 'field_t'},
                                  debug=True)
    
    for pointData in pointDatas:
        for field_data in pointData.values():
            try:
                assert field_data.shape == x_mesh.shape
            except AttributeError:
                assert field_data[0].shape == x_mesh.shape

    (x_mesh, y_mesh, z_mesh, conn, offsets, celltypes), \
    pointDatas = Pm.export_to_vtk("test_export_discrete_i", grid, npts_per_cell=None,
                                  snapshots='all', fields={'f1': 'field_static', 'f2': 'field_t'},
                                  debug=True)

    for pointData in pointDatas:
        for field_data in pointData.values():
            try:
                assert field_data.shape == x_mesh.shape
            except AttributeError:
                assert field_data[0].shape == x_mesh.shape

    Pm.comm.Barrier()
    if rank == 0 and not interactive:
        for f in glob.glob("test_export_discrete*.*vtu"):
            os.remove(f)
        os.remove("space_export_discrete.yml")
        os.remove("fields_export_discrete.h5")


@pytest.mark.parallel
@pytest.mark.parametrize('domain', [Square(), Cube()])
@pytest.mark.parametrize('mapping', [IdentityMapping, AffineMapping])
@pytest.mark.parametrize('kind', ['h1', 'l2', 'hdiv', 'hcurl'])
@pytest.mark.parametrize('space', [ScalarFunctionSpace, VectorFunctionSpace])
def test_parallel_export_topological_domain(domain, mapping, kind, space, interactive=False):
    rank = comm.Get_rank()
    dim = domain.dim

    dim_params_dict = {
        2: {'c1': 0, 'c2': 0, 
            'a11': 1, 'a12': 3, 
            'a21': 3, 'a22': 1},
        3: {'c1': 0, 'c2': 1, 'c3': 0,
            'a11': 1, 'a12': 0, 'a13': 2,
            'a21': 0, 'a22': 1, 'a23': 0,
            'a31': 2, 'a32': 0, 'a33': 1}
    }
    F = mapping('F', dim, **dim_params_dict[dim])
    domain = F(domain)

    symbolic_space = space('V', domain, kind=kind)

    degree = [2, 3, 4][:dim]
    domain_h = discretize(domain, ncells=[5] * dim)
    space_h = discretize(symbolic_space, domain_h, degree=degree)

    field = FemField(space_h)

    Om = OutputManager("space_export_topo.yml", "fields_export_topo.h5", comm=comm)

    Om.add_spaces(space_h=space_h)
     
    Om.set_static()
    Om.export_fields(field_static=field)
    Om.add_snapshot(t=0., ts=0)
    Om.export_fields(field_t=field)
    
    Om.export_space_info()
    Om.close()
    
    Pm = PostProcessManager(domain=domain, space_file="space_export_topo.yml", fields_file="fields_export_topo.h5", comm=comm)
    try:
        grid = [refine_array_1d(space_h.breaks[i], 1, False) for i in range(space_h.ldim)]
    except AttributeError:
        grid = [refine_array_1d(space_h.spaces[0].breaks[i], 1, False) for i in range(space_h.ldim)]

    npts_per_cell = [2] * dim
    (x_mesh, y_mesh, z_mesh, conn, offsets, celltypes), \
    pointDatas = Pm.export_to_vtk("test_export_topo_r", grid, npts_per_cell=npts_per_cell,
                                 snapshots='all', fields={'f1': 'field_static', 'f2': 'field_t'}, debug=True)
    for pointData in pointDatas:
        for field_data in pointData.values():
            try:
                assert field_data.shape == x_mesh.shape
            except AttributeError:
                assert field_data[0].shape == x_mesh.shape
    
    (x_mesh, y_mesh, z_mesh, conn, offsets, celltypes), \
    pointDatas = Pm.export_to_vtk("test_export_topo_i", grid, npts_per_cell=None,
                                 snapshots='all', fields={'f1': 'field_static', 'f2': 'field_t'},
                                 debug=True)

    for pointData in pointDatas:
        for field_data in pointData.values():
            try:
                assert field_data.shape == x_mesh.shape
            except AttributeError:
                assert field_data[0].shape == x_mesh.shape

    Pm.comm.Barrier()
    if rank == 0 and not interactive:
        for f in glob.glob("test_export_topo*.*vtu"):
            os.remove(f)
        os.remove("space_export_topo.yml")
        os.remove("fields_export_topo.h5")


if __name__ == "__main__":
    import sys
    # Meant to showcase examples 
    test_multipatch_parallel_export(True)
    test_parallel_export_discrete_domain('bent_pipe.h5', 'h1', ScalarFunctionSpace, True)
    test_parallel_export_topological_domain(Square(), AffineMapping, 'h1', ScalarFunctionSpace, True)
