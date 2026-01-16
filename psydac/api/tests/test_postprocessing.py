#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
import glob
from pathlib import Path

import pytest
import numpy as np
from mpi4py import MPI

from sympde.topology import Square, Cube, ScalarFunctionSpace, VectorFunctionSpace, Domain, Derham, Union
from sympde.topology.analytical_mapping import IdentityMapping, AffineMapping, PolarMapping

from psydac.api.discretization import discretize
from psydac.fem.basic import FemField
from psydac.fem.tensor import TensorFemSpace
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

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

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


def build_2_mapped_squares():
    A = Square('A',bounds1=(-0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(-0.5, 1.), bounds2=(np.pi/2, np.pi))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0.1, rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0.1, rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    patches = [D1, D2]
    connectivity = [((0,1,1),(1,1,-1))]
    return Domain.join(patches, connectivity, 'domain')


def build_2_squares():
    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))

    patches = [A, B]
    connectivity = [((0,1,1),(1,1,-1))]
    return Domain.join(patches, connectivity, 'domain')


def build_2_cubes():
    A = Cube('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2), bounds3=(0, 1))
    B = Cube('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi), bounds3=(0, 1))

    patches = [A, B]
    connectivity = [((0,1,1),(1,1,-1))]
    return Domain.join(patches, connectivity, 'domain')


###############################################################################
#                            Output Manager tests                             #
###############################################################################
@pytest.mark.parametrize( 'dtype', ['float', 'complex'] )
def test_add_spaces(dtype):
    domain = Square('D')
    A = ScalarFunctionSpace('A', domain, kind='H1')
    A.codomain_type = dtype
    B = VectorFunctionSpace('B', domain, kind=None)
    B.codomain_type = dtype

    domain_h = discretize(domain, ncells=[5, 5])

    Ah = discretize(A, domain_h, degree=[3, 3])
    Bh = discretize(B, domain_h, degree=[2, 2])

    Om = OutputManager('test_add_spaces_single_patch.yml',
                       'test_add_spaces_single_patch.h5')

    Om.add_spaces(Ah=Ah, Bh=Bh)

    with pytest.raises(AssertionError):
        Om.add_spaces(Ah=Ah)

    expected_spaces_info = {'ndim': 2,
                            'fields': 'test_add_spaces_single_patch.h5',
                            'patches': [{'name': 'D',
                                         'breakpoints': [
                                             [0., 0.2, 0.4, 0.6000000000000001, 0.8, 1.],
                                             [0., 0.2, 0.4, 0.6000000000000001, 0.8, 1.]
                                         ],
                                         'scalar_spaces': [{'name': 'Ah',
                                                            'ldim': 2,
                                                            'kind': 'h1',
                                                            'dtype': f"<class '{dtype}'>",
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
                                                            'dtype': f"<class '{dtype}'>",
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
                                                            'dtype': f"<class '{dtype}'>",
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
                                                                 'dtype': f"<class '{dtype}'>",
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
                                                                 'dtype': f"<class '{dtype}'>",
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
    os.remove('test_add_spaces_single_patch.yml')


@pytest.mark.parametrize( 'dtype', ['float', 'complex'] )
def test_export_fields_serial(dtype):
    domain = Square('D')
    A = ScalarFunctionSpace('A', domain, kind='H1')
    A.codomain_type = dtype
    B = VectorFunctionSpace('B', domain, kind=None)
    B.codomain_type = dtype

    domain_h = discretize(domain, ncells=[5, 5])

    Ah = discretize(A, domain_h, degree=[3, 3])
    Bh = discretize(B, domain_h, degree=[2, 2])

    Om = OutputManager('test_export_fields_serial.yml', 'test_export_fields_serial.h5')
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
    fh5 = h5.File('test_export_fields_serial.h5', mode='r')

    # General check
    assert fh5.attrs['spaces'] == 'test_export_fields_serial.yml'
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

    os.remove('test_export_fields_serial.yml')
    os.remove('test_export_fields_serial.h5')


@pytest.mark.mpi
def test_export_fields_parallel():
    comm = MPI.COMM_WORLD
    domain = Square('D')
    A = ScalarFunctionSpace('A', domain, kind='H1')

    domain_h = discretize(domain, ncells=[5, 5], comm=comm)

    Ah = discretize(A, domain_h, degree=[3, 3])

    Om = OutputManager(
        'test_export_fields_parallel.yml',
        'test_export_fields_parallel.h5',
        comm=comm,
        save_mpi_rank=True)

    Om.add_spaces(Ah=Ah)

    uh = FemField(Ah)

    Om.set_static()
    Om.export_fields(uh=uh)
    Om.close()

    comm.Barrier()
    rank = comm.Get_rank()
    if rank  == 0:
        import h5py as h5
        fh5 = h5.File('test_export_fields_parallel.h5', mode='r',)

        assert 'mpi_dd' in fh5.keys()
        assert 'D' in fh5['mpi_dd'].keys()
        # Shape should be (size, 2, ldim) here ldim is 2 (Square())
        assert fh5['mpi_dd']['D'].shape == (comm.Get_size(), 2, 2)
        fh5.close()
        os.remove('test_export_fields_parallel.yml')
        os.remove('test_export_fields_parallel.h5')


###############################################################################
#                 Output Manager and PostProcess Manager tests                #
###############################################################################
@pytest.mark.parametrize('domain', [Square(), Cube()])
@pytest.mark.parametrize( 'dtype', ['float', 'complex'] )
def test_reconstruct_spaces_topological_domain(domain, dtype):
    dim = domain.dim

    domain_h = discretize(domain, ncells=[5] * dim)

    Vh1 = ScalarFunctionSpace('Vh1', domain, kind='h1')
    Vh1.codomain_type = dtype
    Vvh1 = VectorFunctionSpace('Vvh1', domain, kind='h1')
    Vvh1.codomain_type = dtype

    Vl2 = ScalarFunctionSpace('Vl2', domain, kind='l2')
    Vl2.codomain_type = dtype
    Vvl2 = VectorFunctionSpace('Vvl2', domain, kind='l2')
    Vvl2.codomain_type = dtype

    Vhdiv = VectorFunctionSpace('Vhdiv', domain, kind='hdiv')
    Vhdiv.codomain_type = dtype

    Vhcurl = VectorFunctionSpace('Vhcurl', domain, kind='hcurl')
    Vhdiv.codomain_type = dtype

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

    Om1 = OutputManager(
        "test_reconstruct_spaces_topological_domain.yml",
        "test_reconstruct_spaces_topological_domain.h5"
    )

    Om1.add_spaces(Vector_h1=Vvhh1, Vector_l2=Vvhl2, Vector_hdiv=Vhhdiv, Vector_hcurl=Vhhcurl)
    Om1.add_spaces(Scalar_h1=Vhh1, Scalar_l2=Vhl2)

    space_info_1 = Om1.space_info

    Om1.export_space_info()
    with pytest.warns(UserWarning):
        Pm = PostProcessManager(
            domain=domain,
            space_file="test_reconstruct_spaces_topological_domain.yml",
            fields_file="test_reconstruct_spaces_topological_domain.h5"
        )

    Om2 = OutputManager(
        "test_reconstruct_spaces_topological_domain_2.yml",
        "test_reconstruct_spaces_topological_domain.h5"
    )
    Om2.add_spaces(**Pm.spaces)

    space_info_2 = Om2.space_info
    Om2.export_space_info()

    assert space_info_1 == space_info_2
    os.remove("test_reconstruct_spaces_topological_domain.yml")
    os.remove("test_reconstruct_spaces_topological_domain_2.yml")


@pytest.mark.parametrize('domain, seq', [(Square(), ['h1', 'hdiv', 'l2']), (Square(), ['h1', 'hcurl', 'l2']), (Cube(), None)])
@pytest.mark.parametrize( 'dtype', ['float', 'complex'] )
def test_reconstruct_DerhamSequence_topological_domain(domain, seq, dtype):
    derham = Derham(domain, sequence=seq)

    for V in derham.spaces:
        V.codomain_type = dtype

    domain_h  = discretize(domain, ncells=[5]*domain.dim)
    derham_h = discretize(derham, domain_h, degree=[2]*domain.dim)

    Om = OutputManager(
        'test_reconstruct_DerhamSequence_topological_domain.yml',
        'test_reconstruct_DerhamSequence_topological_domain.h5'
    )

    Om.add_spaces(**dict(((Vh.symbolic_space.name, Vh) for Vh in derham_h.spaces)))
    Om.export_space_info()
    with pytest.warns(UserWarning):
        Pm = PostProcessManager(
            domain=domain,
            space_file='test_reconstruct_DerhamSequence_topological_domain.yml',
            fields_file='test_reconstruct_DerhamSequence_topological_domain.h5'
        )

    Om2 = OutputManager(
        'test_reconstruct_DerhamSequence_topological_domain_2.yml',
        'test_reconstruct_DerhamSequence_topological_domain.h5'
    )

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

    os.remove('test_reconstruct_DerhamSequence_topological_domain_2.yml')
    os.remove('test_reconstruct_DerhamSequence_topological_domain.yml')


@pytest.mark.parametrize('geometry, seq', [('identity_2d.h5', ['h1', 'hdiv', 'l2']),
                                           ('identity_2d.h5', ['h1', 'hcurl', 'l2']),
                                           ('identity_3d.h5', None),
                                           ('pipe.h5', ['h1', 'hdiv', 'l2']),
                                           ('pipe.h5', ['h1', 'hcurl', 'l2'])])
@pytest.mark.parametrize( 'dtype', ['float', 'complex'] )
def test_reconstruct_DerhamSequence_discrete_domain(geometry, seq, dtype):

    geometry_file = os.path.join(mesh_dir, geometry)
    domain = Domain.from_file(geometry_file)

    derham = Derham(domain, sequence=seq)

    for V in derham.spaces:
        V.codomain_type = dtype

    domain_h  = discretize(domain, filename=geometry_file)
    derham_h = discretize(derham, domain_h, degree=[2]*domain.dim)

    Om = OutputManager(
        'test_reconstruct_DerhamSequence_discrete_domain.yml',
        'test_reconstruct_DerhamSequence_discrete_domain.h5'
    )
    Om.add_spaces(**dict(((Vh.symbolic_space.name, Vh) for Vh in derham_h.spaces)))
    Om.export_space_info()

    with pytest.warns(UserWarning):
        Pm = PostProcessManager(
            geometry_file=geometry_file,
            space_file='test_reconstruct_DerhamSequence_discrete_domain.yml',
            fields_file='test_reconstruct_DerhamSequence_discrete_domain.h5'
        )

    Om2 = OutputManager(
        'test_reconstruct_DerhamSequence_discrete_domain_2.yml',
        'test_reconstruct_DerhamSequence_discrete_domain.h5'
    )

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

    os.remove('test_reconstruct_DerhamSequence_discrete_domain_2.yml')
    os.remove('test_reconstruct_DerhamSequence_discrete_domain.yml')

@pytest.mark.parametrize( 'dtype', ['float', 'complex'] )
def test_reconstruct_multipatch(dtype):
    bounds1   = (0.5, 1.)
    bounds2_A = (0, np.pi/2)
    bounds2_B = (np.pi/2, np.pi)

    A = Square('A',bounds1=bounds1, bounds2=bounds2_A)
    B = Square('B',bounds1=bounds1, bounds2=bounds2_B)

    connectivity = [((0,1,1),(1,1,-1))]
    patches = [A,B]
    domain = Domain.join(patches, connectivity, 'domain')

    Va = ScalarFunctionSpace('Va', A)
    Va.codomain_type = dtype
    Vb = ScalarFunctionSpace('Vb', B)
    Vb.codomain_type = dtype
    V = ScalarFunctionSpace('V', domain)
    V.codomain_type = dtype
    Vv = VectorFunctionSpace('Vv', domain)
    Vv.codomain_type = dtype
    Vva = VectorFunctionSpace('Vva', A)
    Vva.codomain_type = dtype

    Om = OutputManager('spaces_multipatch.yml', 'fields_multipatch.h5')

    domain_h = discretize(domain, ncells = [15, 15])
    Ah = discretize(A, ncells = [15, 15])
    Bh = discretize(B, ncells = [15, 15])

    Vh = discretize(V, domain_h, degree=[3, 3])
    Vah = discretize(Va, Ah, degree=[3, 3])
    Vbh = discretize(Vb, Bh, degree=[3, 3])

    Vvh = discretize(Vv, domain_h, degree=[3,3])
    Vvah = discretize(Vva, Ah, degree=[3,3])

    Om.add_spaces(V=Vh)
    Om.add_spaces(Va=Vah)
    Om.add_spaces(Vb=Vbh)
    Om.add_spaces(Vv=Vvh)
    Om.add_spaces(Vva=Vvah)

    Om.export_space_info()

    with pytest.warns(UserWarning):
        Pm = PostProcessManager(
            domain=domain,
            space_file='spaces_multipatch.yml',
            fields_file='fields_multipatch.h5'
        )

    Om2 = OutputManager('__.yml', '__.h5')

    Om2.add_spaces(**Pm.spaces)

    os.remove('spaces_multipatch.yml')

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


def test_incorrect_arg_export_to_vtk():
    domain = Square()
    space = ScalarFunctionSpace('V', domain)

    domain_h = discretize(domain, ncells=[4, 4])
    space_h = discretize(space, domain_h, degree=[2, 2])

    u = FemField(space_h)

    Om = OutputManager(
        "test_incorrect_arg_export_to_vtk.yml",
        "test_incorrect_arg_export_to_vtk.h5"
    )

    Om.add_spaces(V=space_h)
    Om.set_static()
    Om.export_fields(u=u)
    Om.add_snapshot(t=0., ts=0)
    Om.export_fields(v=u)
    Om.export_space_info()
    Om.close()

    Pm = PostProcessManager(
        domain=domain,
        space_file="test_incorrect_arg_export_to_vtk.yml",
        fields_file="test_incorrect_arg_export_to_vtk.h5",
    )

    with pytest.raises(ValueError): # needs npts_per_cell or grid
        Pm.export_to_vtk('_', grid=None, npts_per_cell=None)
    with pytest.raises(ValueError): # snapshot = 'none' but no static fields
        Pm.export_to_vtk('_', grid=None, npts_per_cell=2, snapshots='none', fields='f')
    with pytest.warns(UserWarning): # snapshot = 'all' but no static fields
        Pm.export_to_vtk('_', grid=None, npts_per_cell=2, snapshots='all', fields='f')
    with pytest.warns(UserWarning): # snapshot = 'all' but no time dependent field
        Pm.export_to_vtk('_', grid=None, npts_per_cell=2, snapshots='all', fields='u')
    with pytest.raises(ValueError): # empty grid
        Pm.export_to_vtk('_', grid=[[],[]], npts_per_cell=None, snapshots='none', fields='u')

    os.remove('_.static.vtu')
    os.remove("test_incorrect_arg_export_to_vtk.yml")
    os.remove("test_incorrect_arg_export_to_vtk.h5")


@pytest.mark.mpi
@pytest.mark.parametrize('geometry', ['identity_2d.h5',
                                      'identity_3d.h5',
                                    #   'pipe.h5',]) # Doesn't work, see issue #229
                                      'multipatch/magnet.h5',
                                      'multipatch/plate_with_hole_mp_7.h5'])
@pytest.mark.parametrize('kind', ['h1', 'l2', 'hdiv', 'hcurl'])
@pytest.mark.parametrize('space', [ScalarFunctionSpace, VectorFunctionSpace])
@pytest.mark.parametrize( 'dtype', ['float', 'complex'] )
def test_parallel_export_discrete_domain(geometry, kind, space, dtype):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domain_h = discretize(domain, filename=filename, comm=comm)

    dim = domain.dim

    symbolic_space = space('V', domain, kind=kind)
    symbolic_space.codomain_type = dtype

    degree = list(domain_h.mappings.values())[0].space.degree
    space_h = discretize(symbolic_space, domain_h, degree=degree)

    field = FemField(space_h)

    Om = OutputManager(
        "test_parallel_export_discrete_domain.yml",
        "test_parallel_export_discrete_domain.h5",
        comm=comm,
        save_mpi_rank=True,
    )

    Om.add_spaces(space_h=space_h)

    Om.set_static()
    Om.export_fields(field_s=field)
    Om.add_snapshot(t=0., ts=0)
    Om.export_fields(field_t=field)

    Om.export_space_info()
    Om.close()

    Pm = PostProcessManager(
        geometry_file=filename,
        space_file="test_parallel_export_discrete_domain.yml",
        fields_file="test_parallel_export_discrete_domain.h5",
        comm=comm
    )

    grid1 = None
    if 'multipatch' in filename:
        if isinstance(space_h.spaces[0], TensorFemSpace):
            grid2 = {
                i_name: [refine_array_1d(bks[i], 1, False) for i in range(dim)]
                    for i_name, bks in zip(
                    domain.interior_names, [space_j.breaks for space_j in space_h.spaces]
                )
            }
        else:
            grid2 = {
                i_name: [refine_array_1d(bks[i], 1, False) for i in range(dim)]
                    for i_name, bks in zip(
                    domain.interior_names, [space_j.spaces[0].breaks for space_j in space_h.spaces]
                )
            }
    else:
        if isinstance(space_h, TensorFemSpace):
            grid2 = [refine_array_1d(space_h.breaks[i], 1, False) for i in range(dim)]
        else:
            grid2 = [refine_array_1d(space_h.spaces[0].breaks[i], 1, False) for i in range(dim)]

    npts_per_cell = [2] * dim

    # Test grid1
    Pm.export_to_vtk(
        "test_export_None_grid",
        grid=grid1,
        npts_per_cell=npts_per_cell,
        snapshots='all',
        fields=('field_s', 'field_t'),
    )

    # Test grid2
    Pm.export_to_vtk(
        "test_export_regular_grid",
        grid=grid2,
        npts_per_cell=npts_per_cell,
        snapshots='all',
        fields=('field_s', 'field_t'),
    )

    Pm.export_to_vtk(
        "test_export_irregular_grid_1",
        grid=grid2,
        npts_per_cell=None,
        snapshots='all',
        fields=('field_s', 'field_t'),
    )

    Pm.comm.Barrier()
    if rank == 0:
        for f in glob.glob("test_export*grid*.*vtu"):
            os.remove(f)
        os.remove("test_parallel_export_discrete_domain.yml")
        os.remove("test_parallel_export_discrete_domain.h5")


@pytest.mark.mpi
@pytest.mark.parametrize('domain', [
    Square(),
    Cube(),
    build_2_cubes(),
    build_2_mapped_squares(),
    build_2_squares()
])
@pytest.mark.parametrize('mapping', [IdentityMapping, AffineMapping])
@pytest.mark.parametrize('kind', ['h1', 'l2', 'hdiv', 'hcurl'])
@pytest.mark.parametrize('space', [ScalarFunctionSpace, VectorFunctionSpace])
@pytest.mark.parametrize( 'dtype', ['float', 'complex'] )
def test_parallel_export_topological_domain(domain, mapping, kind, space, dtype):
    comm = MPI.COMM_WORLD
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
    if not isinstance(domain.interior, Union):
        F = mapping('F', dim, **dim_params_dict[dim])
        domain = F(domain)

    symbolic_space = space('V', domain, kind=kind)
    symbolic_space.codomain_type = dtype

    degree = [2, 2, 2][:dim]
    domain_h = discretize(domain, ncells=[6] * dim, comm=comm)

    space_h = discretize(symbolic_space, domain_h, degree=degree)

    field = FemField(space_h)

    Om = OutputManager(
        "test_parallel_export_topological_domain.yml",
        "test_parallel_export_topological_domain.h5",
        comm=comm,
    )

    Om.add_spaces(space_h=space_h)

    Om.set_static()
    Om.export_fields(field_s=field)
    Om.add_snapshot(t=0., ts=0)
    Om.export_fields(field_t=field)

    Om.export_space_info()
    Om.close()

    Pm = PostProcessManager(
        domain=domain,
        space_file="test_parallel_export_topological_domain.yml",
        fields_file="test_parallel_export_topological_domain.h5",
        comm=comm,
    )

    grid1 = None
    if isinstance(domain.interior, Union):
        if isinstance(space_h.spaces[0], TensorFemSpace):
            grid2 = {
                i_name: [refine_array_1d(bks[i], 1, False) for i in range(dim)]
                    for i_name, bks in zip(
                    domain.interior_names, [space_j.breaks for space_j in space_h.spaces]
                )
            }
        else:
            grid2 = {
                i_name: [refine_array_1d(bks[i], 1, False) for i in range(dim)]
                    for i_name, bks in zip(
                    domain.interior_names, [space_j.spaces[0].breaks for space_j in space_h.spaces]
                )
            }
    else:
        if isinstance(space_h, TensorFemSpace):
            grid2 = [refine_array_1d(space_h.breaks[i], 1, False) for i in range(dim)]
        else:
            grid2 = [refine_array_1d(space_h.spaces[0].breaks[i], 1, False) for i in range(dim)]

    npts_per_cell = [2] * dim
    # Test grid1
    Pm.export_to_vtk(
        "test_export_None_grid",
        grid=grid1,
        npts_per_cell=npts_per_cell,
        snapshots='all',
        fields=('field_s', 'field_t'),
    )

    # Test grid2
    Pm.export_to_vtk(
        "test_export_regular_grid",
        grid=grid2,
        npts_per_cell=npts_per_cell,
        snapshots='all',
        fields=('field_s', 'field_t'),
    )

    Pm.export_to_vtk(
        "test_export_irregular_grid_1",
        grid=grid2,
        npts_per_cell=None,
        snapshots='all',
        fields=('field_s', 'field_t'),
    )

    Pm.fields_file.close()

    if rank == 0:
        for f in glob.glob("test_export*grid*.*vtu"):
            os.remove(f)
        os.remove("test_parallel_export_topological_domain.yml")
        os.remove("test_parallel_export_topological_domain.h5")
