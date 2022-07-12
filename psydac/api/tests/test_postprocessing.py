import glob
import itertools
import os
from turtle import shape
import pytest

import numpy as np

from sympde.topology import Square, Cube, ScalarFunctionSpace, VectorFunctionSpace, Domain, Derham
from sympde.topology.analytical_mapping import IdentityMapping, AffineMapping

from psydac.api.discretization import discretize
from psydac.fem.basic import FemField
from psydac.fem.vector import VectorFemSpace, ProductFemSpace
from psydac.utilities.utils import refine_array_1d
from psydac.feec.pull_push import (push_2d_hcurl, 
                                   push_2d_h1, 
                                   push_2d_hdiv,
                                   push_2d_l2,
                                   push_3d_h1,
                                   push_3d_hcurl,
                                   push_3d_hdiv,
                                   push_3d_l2)

from psydac.api.postprocessing import OutputManager, PostProcessManager

# Get mesh_directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']
except KeyError:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

# Get a communicator
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    if comm.size == 1:
        comm = None
except ImportError:
    comm = None
    rank = 0
    size = 1

# Tolerances for float equality
ATOL=1e-15
RTOL=1e-15

# Old push-forward functions
push_function_dict = {
    'h1': {2: push_2d_h1, 3: push_3d_h1},
    'hcurl': {2: push_2d_hcurl, 3: push_3d_hcurl},
    'hdiv': {2: push_2d_hdiv, 3: push_3d_hdiv},
    'l2': {2: push_2d_l2, 3: push_3d_l2},
}


def test_add_spaces():
    domain = Square('D')
    A = ScalarFunctionSpace('A', domain, kind='H1')
    B = VectorFunctionSpace('B', domain, kind=None)

    domain_h = discretize(domain, ncells=[5, 5])

    Ah = discretize(A, domain_h, degree=[3, 3])
    Bh = discretize(B, domain_h, degree=[2, 2])

    Om = OutputManager('file.yml', 'file.h5')
    Om.add_spaces(Ah=Ah, Bh=Bh)

    with pytest.raises(AssertionError):
        Om.add_spaces(Ah=Ah)

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
    assert Om._space_info == expected_spaces_info

    Om.export_space_info()

    # Remove file
    os.remove('file.yml')


def test_export_fields():
    domain = Square('D')
    A = ScalarFunctionSpace('A', domain, kind='H1')
    B = VectorFunctionSpace('B', domain, kind=None)

    domain_h = discretize(domain, ncells=[5, 5])

    Ah = discretize(A, domain_h, degree=[3, 3])
    Bh = discretize(B, domain_h, degree=[2, 2])

    Om = OutputManager('file.yml', 'file.h5')
    Om.add_spaces(Ah=Ah, Bh=Bh)

    uh = FemField(Ah)
    vh = FemField(Bh)

    with pytest.raises(ValueError):
        Om.export_fields(uh=uh, vh=vh)

    Om.set_static()
    Om.export_fields(uh=uh)
    Om.export_fields(vh=vh)
    
    Om.add_snapshot(0., 0)
    with pytest.raises(AssertionError):
        Om.export_fields(uh=uh, vh=vh)

    Om.export_fields(vh_00=vh, uh_0=uh)

    with pytest.raises(ValueError):
        Om.export_fields(vh_00=vh)
    
    Om.export_fields(vh_01=vh)

    Om.add_snapshot(1., 1)
    Om.add_snapshot(2., 2)
    Om.export_fields(vh_00=vh, vh_01=vh)

    Om.close()

    import h5py as h5
    fh5 = h5.File('file.h5', mode='r')

    # General check
    assert fh5.attrs['spaces'] == 'file.yml'
    assert set(fh5.keys()) == {'static', 'snapshot_0000', 'snapshot_0001', 'snapshot_0002'}
    assert all(set(fh5[k].keys()) == {'D'} for k in {'static', 'snapshot_0000', 'snapshot_0002'})

    # static check
    assert set(fh5['static']['D'].keys()) == {'Ah', 'Bh[0]', 'Bh[1]'}
    assert set(fh5['static']['D']['Ah']) == {'uh'}
    assert set(fh5['static']['D']['Bh[0]']) == {'vh[0]'}
    assert set(fh5['static']['D']['Bh[1]']) == {'vh[1]'}
    assert fh5['static']['D']['Bh[0]'].attrs['parent_space'] == 'Bh'
    assert fh5['static']['D']['Bh[1]'].attrs['parent_space'] == 'Bh'
    assert fh5['static']['D']['Bh[0]']['vh[0]'].attrs['parent_field'] == 'vh'
    assert fh5['static']['D']['Bh[1]']['vh[1]'].attrs['parent_field'] == 'vh'

    # Snapshot checks
    for i in range(3):
        assert fh5[f'snapshot_000{i}'].attrs['t'] == float(i)
        assert fh5[f'snapshot_000{i}'].attrs['ts'] == i

    assert set(fh5['snapshot_0001'].keys()) == set()
    
    assert set(fh5['snapshot_0000']['D'].keys()) == {'Ah', 'Bh[0]', 'Bh[1]'}
    assert set(fh5['snapshot_0000']['D']['Ah'].keys()) == {'uh_0'}
    assert set(fh5['snapshot_0000']['D']['Bh[0]']) == {'vh_00[0]', 'vh_01[0]'}
    assert set(fh5['snapshot_0000']['D']['Bh[1]']) == {'vh_00[1]', 'vh_01[1]'}

    assert set(fh5['snapshot_0002']['D'].keys()) == {'Bh[0]', 'Bh[1]'}
    assert set(fh5['snapshot_0002']['D']['Bh[0]'].keys()) == {'vh_00[0]', 'vh_01[0]'}
    assert set(fh5['snapshot_0002']['D']['Bh[1]'].keys()) == {'vh_00[1]', 'vh_01[1]'}


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
                    assert isinstance(value1, type(value2))
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


@pytest.mark.xfail # Needs new geometry files to work See Said's PR 
@pytest.mark.parametrize('geometry, seq', [('identity_2d.h5', ['h1', 'hdiv', 'l2']),
                                           ('identity_2d.h5', ['h1', 'hcurl', 'l2']),
                                           ('identity_3d.h5', None),
                                           ('bent_pipe.h5', ['h1', 'hdiv', 'l2']),
                                           ('bent_pipe.h5', ['h1', 'hcurl', 'l2'])])
def test_reconstruct_DerhamSequence_discrete_domain(geometry, seq):
    
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
                    assert isinstance(value1, type(value2))
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


def test_reconstruct_multipatch():
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

    Om = OutputManager('spaces_multipatch.yml', 'fields_multipatch.h5')

    domain_h = discretize(domain, ncells = [15, 15])
    Ah = discretize(A, ncells = [15, 15])
    Bh = discretize(B, ncells = [15, 15])

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

    Pm = PostProcessManager(
        domain=domain, 
        space_file='spaces_multipatch.yml',
        fields_file='fields_multipatch.h5'
    )
    
    Om2 = OutputManager('__.yml', '__.h5')

    Om2.add_spaces(**Pm.spaces) 

    os.remove('spaces_multipatch.yml')
    os.remove('fields_multipatch.h5')

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
                    assert isinstance(value1, type(value2))
                    if 'space' in key:
                        for space1 in value1:
                            for space2 in value2:
                                if space2['name'] == space1['name']:
                                    for key_space in space1.keys():
                                        assert space1[key_space] == space2[key_space]
                    else:
                        assert value1 == value2


@pytest.mark.parallel
@pytest.mark.xfail # Needs new geometry files to work See Said's PR 
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

    Pm.export_to_vtk("test_export_discrete_r", grid=grid, npts_per_cell=npts_per_cell,
                                  snapshots='all', fields={'f1': 'field_static', 'f2': 'field_t'},
                                  debug=True)
    
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

    if isinstance(space_h, (VectorFemSpace, ProductFemSpace)):
        for i in range(dim):
            field.coeffs[i][:] = i
    else:
        field.coeffs[:] = 5

    Om = OutputManager("space_export_topo.yml", "fields_export_topo.h5", comm=comm)

    Om.add_spaces(space_h=space_h)

    Om.set_static()
    Om.export_fields(field_static=field)
    Om.add_snapshot(t=0., ts=0)
    Om.export_fields(field_t=field)

    Om.export_space_info()
    Om.close()

    Pm = PostProcessManager(domain=domain, space_file="space_export_topo.yml", fields_file="fields_export_topo.h5", comm=comm)

    grid = [[0.2, 0.3], [0.2, 0.5, 0.7], [0.15, 0.45]][:dim]
    shape = (2, 3, 2)[:dim]

    debug_result = Pm.export_to_vtk(
        "test_export_topo_i",
        grid=grid,
        npts_per_cell=None,
        snapshots='all',
        fields=('field_static', 'field_t'),
        debug=True
    )
    Pm.fields_file.close()

    exception_list = debug_result['Exception']
    for e in exception_list:
        raise e

    push_func_i = push_function_dict[kind][dim]

    if isinstance(space_h, (VectorFemSpace, ProductFemSpace)):
        
        if kind in ['hdiv', 'hcurl']:
            push_func = lambda *eta: push_func_i(*field, *eta, F)
        elif kind == 'l2':
            push_func = lambda *eta: tuple(
                push_func_i(f, *eta, F) for f in field
                )
        else:
            push_func = lambda *eta: tuple(
                f(*eta) for f in field
                )

        expected_push = np.zeros(((dim,) + shape))
        for index, eta in zip(itertools.product(*tuple(range(s) for s in shape)), itertools.product(*grid)):
                expected_push[(slice(0, None, 1),) + index] = push_func(*eta)
    else:
        if kind != 'h1':
            push_func = lambda *eta: push_func_i(field, *eta, F)
        else:
             push_func = lambda *eta: push_func_i(field, *eta)

        expected_push = np.zeros(shape)
        for index, eta in zip(itertools.product(*tuple(range(s) for s in shape)), itertools.product(*grid)):
                expected_push[index] = push_func(*eta)
    
    for p in debug_result['pointData']:
        for _, data in p.items():
            if isinstance(data, tuple):
                for i in range(dim):
                    if kind == 'l2':
                        assert np.allclose(data[i], np.abs(np.ravel(expected_push[i], 'F')),
                                           atol=ATOL, rtol=RTOL) # Metric det vs Jacobian det
                    else:
                        assert np.allclose(data[i], np.ravel(expected_push[i], 'F'), atol=ATOL, rtol=RTOL)

            else:
                if kind == 'l2':
                    assert np.allclose(data, np.abs(np.ravel(expected_push, 'F')),
                                       atol=ATOL, rtol=RTOL)  # Metric det vs Jacobian det
                else:
                    assert np.allclose(data, np.ravel(expected_push, 'F'), atol=ATOL, rtol=RTOL)

    try:
        Pm.comm.barrier()
    except AttributeError:
        pass
    if rank == 0 and not interactive:
        for f in glob.glob("test_export_topo*.*vtu"):
            os.remove(f)
        os.remove("space_export_topo.yml")
        os.remove("fields_export_topo.h5")


if __name__ == "__main__":
    # Meant to showcase examples 
    test_parallel_export_discrete_domain('bent_pipe.h5', 'h1', ScalarFunctionSpace, True)
    test_parallel_export_topological_domain(Square(), AffineMapping, 'hcurl', VectorFunctionSpace, True)
