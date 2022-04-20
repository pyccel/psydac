# coding: utf-8
#
# Copyright 2019 Yaman Güçlü

from operator import mod
from zlib import Z_RLE
import numpy as np
from sympy import N
import yaml
import re
import h5py as h5

from sympde.topology.mapping import Mapping
from sympde.topology import Domain, VectorFunctionSpace, ScalarFunctionSpace

from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkQuad, VtkHexahedron

from psydac.api.discretization import discretize
from psydac.cad.geometry import Geometry
from psydac.mapping.discrete import SplineMapping
from psydac.core.bsplines import cell_index
from psydac.feec.pushforward import Pushforward
from psydac.utilities.utils import refine_array_1d
from psydac.fem.basic import FemSpace, FemField
from psydac.utilities.vtk import writeParallelVTKUnstructuredGrid


#===============================================================================
def get_grid_lines_2d(domain_h, V_h, *, refine=1):
    """
    Get the grid lines (i.e. element boundaries) of a 2D computational domain,
    which can be easily plotted with Matplotlib.

    Parameters
    ----------
    domain_h : psydac.cad.geometry.Geometry
        2D single-patch geometry.

    V_h : psydac.fem.tensor.TensorFemSpace
        Spline space from which the breakpoints are extracted.
                    - TODO: remove this argument -

    refine : int
        Number of segments used to describe a grid curve in each element
        (minimum value is 1, which yields quadrilateral elements).


    Returns
    -------
    isolines_1 : list of dict
        Lines having constant value of 'eta1' parameter;
        each line is a dictionary with three keys:
            - 'eta1' : value of eta1 on the curve
            - 'x'    : x coordinates of N points along the curve
            - 'y'    : y coordinates of N points along the curve

    isolines_2 : list of dict
        Lines having constant value of 'eta2' parameter;
        each line is a dictionary with three keys:
            - 'eta2' : value of eta2 on the curve
            - 'x'    : x coordinates of N points along the curve
            - 'y'    : y coordinates of N points along the curve

    """
    # Check that domain is of correct type and contains only one patch
    assert isinstance(domain_h, Geometry)
    assert domain_h.ldim == 2
    assert domain_h.pdim == 2
    assert len(domain_h) == 1

    # TODO: improve
    # Get mapping over patch (create identity map if needed)
    mapping = list(domain_h.mappings.values())[0]
    if mapping is None:
        mapping = lambda *eta: eta

    # TODO: make this work
    # Get 1D breakpoints in logical domain
    # eta1, eta2 = domain_h.breaks

    # NOTE: temporary solution (V_h should not be passed to this function)
    V1, V2 = V_h.spaces
    eta1 = V1.breaks
    eta2 = V2.breaks

    # Refine logical grid
    eta1_r = refine_array_1d(eta1, refine)
    eta2_r = refine_array_1d(eta2, refine)

    # Compute physical coordinates of lines with eta1=const
    isolines_1 = []
    for e1 in eta1:
        x, y = zip(*[mapping(e1, e2) for e2 in eta2_r])
        isolines_1.append(dict(eta1=e1, x=x, y=y))

    # Compute physical coordinates of lines with eta2=const
    isolines_2 = []
    for e2 in eta2:
        x, y = zip(*[mapping(e1, e2) for e1 in eta1_r])
        isolines_2.append(dict(eta2=e2, x=x, y=y))

    return isolines_1, isolines_2


# ===========================================================================
class OutputManager:
    """A class meant to streamline the exportation of
    output data.

    Parameters
    ----------
    filename_space : str or Path-like
         Name/path of the file in which to save the space information.
         The path is relative to the current working directory.

    filename_fields : str or Path-like
         Name/path of the file in which to save the fields.
         The path is relative to the current working directory.

    Attributes
    ----------
    _spaces_info : dict
        Information about the spaces in a human readable format.
        It is written to ``filename_space`` in yaml.
    _spaces : List
        List of the spaces that were added to an instance of OutputManager.

    filename_space : str or Path-like
        Name of the file in which to save the space information.
    filename_fields : str or Path-like
        Name of the file in which to save the fields.

    _next_snapshot_number : int

    is_static : bool or None
        If None, means that no saving scheme was chosen by the user for the next
        ``export_fields``.

    _current_hdf5_group :  h5py.Group
        Group where the fields will be saved in the next ``export_fields``.
    
    _static_names : list
    """

    space_types_to_str = {
        'H1SpaceType()': 'h1',
        'HcurlSpaceType()': 'hcurl',
        'HdivSpaceType()': 'hdiv',
        'L2SpaceType()': 'l2',
        'UndefinedSpaceType()': 'undefined',
    }

    def __init__(self, filename_space, filename_fields, comm=None, mode='w'):

        self._spaces_info = {}
        self._spaces = []

        if filename_space[-4:] != ".yml" and filename_space[-4:] != ".yaml":
            filename_space += ".yml"
        self.filename_space = filename_space

        if filename_fields[-3:] != ".h5":
            filename_fields += ".h5"
        self.filename_fields = filename_fields
        self._next_snapshot_number = 0
        self.is_static = None
        self._current_hdf5_group = None
        self._static_names = []

        self._space_names = []

        self._mode=mode

        self.comm = comm
        self.fields_file = None
    
    def close(self):
        if not self.fields_file is None:
            self.fields_file.close()

    @property
    def current_hdf5_group(self):
        return self._current_hdf5_group

    @property
    def space_info(self):
        return self._spaces_info

    @property
    def spaces(self):
        return dict([(name, space) for name, space in zip(self._spaces[1::3], self._spaces[0::3])])

    def set_static(self):
        """Sets the export to static mode

        """
        if not self.is_static:
            
            self.is_static = True
            if self.fields_file is None:
                kwargs = {}
                if self.comm is not None and self.comm.size > 1:
                    kwargs.update(driver='mpio', comm=self.comm)
                self.fields_file = h5.File(self.filename_fields, mode=self._mode, **kwargs)

            if 'static' not in self.fields_file.keys():
                static_group = self.fields_file.create_group('static')
                self._current_hdf5_group = static_group
            else:
                self._current_hdf5_group = self.fields_file['static']

    def add_snapshot(self, t, ts):
        """Adds a snapshot to the fields' HDF5 file
        and set the export mode to time dependent.

        Parameters
        ----------
        t : float
            Floating point time of the snapshot
        ts : int
            Time step of the snapshot
        """
 
        self.is_static = False

        if self.fields_file is None:
            kwargs = {}
            if self.comm is not None and self.comm.size > 1:
                kwargs.update(driver='mpio', comm=self.comm)
            self.fields_file = h5.File(self.filename_fields, mode=self._mode, **kwargs)

        i = self._next_snapshot_number
        try:
            snapshot = self.fields_file.create_group(f'snapshot_{i:0>4}')
        except ValueError:
            regexp = re.compile(r'snapshot_(?P<id>\d+)')
            i = max([int(regexp.search(k).group('id')) for k in self.fields_file.keys() if regexp.search(k) is not None]) + 1
            snapshot = self.fields_file.create_group(f'snapshot_{i:0>4}')
        snapshot.attrs.create('t', data=t, dtype=float)
        snapshot.attrs.create('ts', data=ts, dtype=int)

        self._next_snapshot_number = i + 1
        self._current_hdf5_group = snapshot

    def add_spaces(self, **femspaces):
        """Add femspaces to the scope of this instance of OutputManager

        Parameters
        ----------
        femspaces:  psydac.fem.basic.FemSpace dict
            Femspaces to add in the scope of this OutputManager instance.

        """
        assert all(isinstance(femspace, FemSpace) for femspace in femspaces.values())

        for name, femspace in femspaces.items():
            assert name not in self._space_names
            try:
                patches = femspace.symbolic_space.domain.interior.as_tuple()
                for i in range(len(patches)):
                    if self.comm is None or self.comm.rank == 0: 
                        if femspace.spaces[i].is_product:
                            self._add_vector_space(femspace.spaces[i], name=name, patch_name=patches[i].name)
                        else:
                            self._add_scalar_space(femspace.spaces[i], name=name, patch_name=patches[i].name)
                    self._spaces.append(femspace.spaces[i])
                    self._spaces.append(name)
                    self._spaces.append(patches[i].name)

            except AttributeError:
                if self.comm is None or self.comm.rank == 0: 
                    if femspace.is_product:
                        self._add_vector_space(femspace, name=name, patch_name=femspace.symbolic_space.domain.name)
                    else:
                        self._add_scalar_space(femspace, name=name, patch_name=femspace.symbolic_space.domain.name)
                self._spaces.append(femspace)
                self._spaces.append(name)
                self._spaces.append(femspace.symbolic_space.domain.name)
        
            self._space_names.append(name)

    def _add_scalar_space(self, scalar_space, name=None, dim=None, patch_name=None, kind=None):
        """Adds a scalar space to the scope of this instance of OutputManager

        Parameters
        ----------
        scalar_space : psydac.fem.tensor.TensorFemSpace
            Scalar space to add to the scope.

        name : str or None, optional
            Name under which to save the space.

        dim : int or None, optional
            Physical dimension of the related symbolic space.
            Is read directly if not given.

        patch_name : str or None, optional
            Name of the patch in which the symbolic domain belongs.

        kind : str or None, optional
            Kind of the space.

        Returns
        -------
        new_space : dict
            Formatted space info.
        """
        spaces_info = self._spaces_info
        
        scalar_space_name = name
        patch = patch_name

        if dim is None:
            symbolic_space = scalar_space.symbolic_space
            pdim = symbolic_space.domain.dim
            kind = symbolic_space.kind
        else:
            pdim = dim
            kind = kind

        ldim = scalar_space.ldim
        vector_space = scalar_space.vector_space
        dtype = str(vector_space.dtype)
        periodic = scalar_space.periodic
        degree = [scalar_space.spaces[i].degree for i in range(ldim)]
        basis = [scalar_space.spaces[i].basis for i in range(ldim)]
        knots = [scalar_space.spaces[i].knots.tolist() for i in range(ldim)]

        new_space = {'name': scalar_space_name,
                     'ldim': ldim,
                     'kind': self.space_types_to_str[str(kind)],
                     'dtype': dtype,
                     'rational': False,
                     'periodic': periodic,
                     'degree': degree,
                     'basis': basis,
                     'knots': knots
                     }
        if spaces_info == {}:
            spaces_info = {
                           'ndim': pdim,
                           'fields': self.filename_fields,
                           'patches': [{'name': patch,
                                        'scalar_spaces': [new_space]
                                        }]
                           }
        else:
            assert spaces_info['ndim'] == pdim
            try:
                patch_index = [spaces_info['patches'][i]['name']
                               for i in range(len(spaces_info['patches']))].index(patch)
            except ValueError:
                patch_index = -1

            if patch_index != -1:
                assert all(spaces_info['patches'][patch_index]['scalar_spaces'][i]['name'] != scalar_space_name
                           for i in range(len(spaces_info['patches'][patch_index]['scalar_spaces'])))

                spaces_info['patches'][patch_index]['scalar_spaces'].append(new_space)
            else:
                spaces_info['patches'].append({'name': patch, 'scalar_spaces': [new_space]})

        self._spaces_info = spaces_info

        return new_space

    def _add_vector_space(self, vector_space, name=None, patch_name=None):
        """Adds a vector space to the scope of this instance of OutputManager.

        Parameters
        ----------
        vector_space: psydac.fem.vector.VectorFemSpace or psydac.fem.vector.ProductFemSpace
            Vector/Product FemSpace to add to the scope.
        """

        symbolic_space = vector_space.symbolic_space
        dim = symbolic_space.domain.dim
        patch_name = patch_name
        kind = symbolic_space.kind

        scalar_spaces_info = []
        for i, scalar_space in enumerate(vector_space.spaces):
            sc_space_info = self._add_scalar_space(scalar_space,
                                                   name=name + f'[{i}]',
                                                   dim=dim,
                                                   patch_name=patch_name,
                                                   kind='UndefinedSpaceType()')
            scalar_spaces_info.append(sc_space_info)

        spaces_info = self._spaces_info

        new_vector_space = {'name': name,
                            'kind': self.space_types_to_str[str(kind)],
                            'components': scalar_spaces_info,
                            }

        patch_index = [spaces_info['patches'][i]['name']
                       for i in range(len(spaces_info['patches']))].index(patch_name)

        try:
            spaces_info['patches'][patch_index]['vector_spaces'].append(new_vector_space)
        except KeyError:
            spaces_info['patches'][patch_index].update({'vector_spaces': [new_vector_space]})

        self._spaces_info = spaces_info

    def export_fields(self, **fields):
        """
        Exports the fields' coefficients to an HDF5 file.
        They are saved under snapshot/patch/space/field
        or static/patch/space/field depending on the value of
        ``self.is_static``.

        Parameters
        ----------
        fields : psydac.fem.basic.FemField dict
            List of named fields

        Raises
        ------
        ValueError
            When self.is_static is None (no saving scheme specified)
        """
        if self.is_static is None:
            raise ValueError('Saving scheme not specified')
        if self.is_static:
            self._static_names.extend(fields.keys())
        else:
            assert all(not field_name in self._static_names for field_name in fields.keys())

        fh5 = self.fields_file

        if 'spaces' not in fh5.attrs.keys():
            fh5.attrs.create('spaces', self.filename_space)

        saving_group = self._current_hdf5_group

        # Add field coefficients as named datasets
        for name_field, field in fields.items():
            multipatch = hasattr(field.space.symbolic_space.domain.interior, 'as_tuple')
            
            if multipatch:
                for f in field.fields:
                    i = self._spaces.index(f.space)

                    name_space = self._spaces[i+1]
                    name_patch = self._spaces[i+2]

                    if f.space.is_product:  # Vector field case
                        for i, field_coeff in enumerate(f.coeffs):
                            name_field_i = name_field + f'[{i}]'
                            name_space_i = name_space + f'[{i}]'

                            Vi = f.space.vector_space.spaces[i]
                            index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))

                            space_group = saving_group.create_group(f'{name_patch}/{name_space_i}')
                            space_group.attrs.create('parent_space', data=name_space)

                            dset = space_group.create_dataset(f'{name_field_i}',
                                                            shape=Vi.npts, dtype=Vi.dtype)
                            dset.attrs.create('parent_field', data=name_field)
                            dset[index] = field_coeff[index]
                    else:
                        V = f.space.vector_space
                        index = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
                        dset = saving_group.create_dataset(f'{name_patch}/{name_space}/{name_field}', shape=V.npts, dtype=V.dtype)
                        dset[index] = f.coeffs[index]
            else:
                i = self._spaces.index(field.space)

                name_space = self._spaces[i+1]
                name_patch = self._spaces[i+2]

                if field.space.is_product:  # Vector field case
                    for i, field_coeff in enumerate(field.coeffs):
                        name_field_i = name_field + f'[{i}]'
                        name_space_i = name_space + f'[{i}]'

                        Vi = field.space.vector_space.spaces[i]
                        index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))
                        

                        space_group = saving_group.create_group(f'{name_patch}/{name_space_i}')
                        space_group.attrs.create('parent_space', data=name_space)

                        dset = space_group.create_dataset(f'{name_field_i}',
                                                        shape=Vi.npts, dtype=Vi.dtype)
                        dset.attrs.create('parent_field', data=name_field)
                        dset[index] = field_coeff[index]
                else:
                    V = field.space.vector_space
                    index = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
                    dset = saving_group.create_dataset(f'{name_patch}/{name_space}/{name_field}', shape=V.npts, dtype=V.dtype)
                    dset[index] = field.coeffs[index]


    def export_space_info(self):
        """Export the space info to Yaml

        """
        if self.comm is None or self.comm.rank == 0:
            with open(self.filename_space, 'w') as f:
                yaml.dump(data=self._spaces_info, stream=f, default_flow_style=None, sort_keys=False)


class PostProcessManager:
    """A class to read saved information of a previous simulation
    and start post-processing from there.

    Parameters
    ----------
    geometry_file : str or Path-like
        Relative path to the geometry file.
    domain : sympde.topology.basic.Domain
        Symbolic domain, provided alongside ``ncells`` in place of ``geometry_file``.

    space_file : str or Path-like
        Relative path to the file containing the space information.
    fields_file : str or Path-like
        Relative path to the file containing the space information.

    ncells : list of ints
        Number of cells in the domain, provided alongside ``domain`` in place of ``geometry_file``.

    Attributes
    ----------
    geometry_file : str or Path-like
        Relative path to the geometry file
    space_file : str or Path-like
        Relative path to the file containing the space information
    fields_file : str or Path-like
        Relative path to the file containing the space information

    _spaces : dict
        Named spaces

    _domain : sympde.topology.basic.Domain
        Symbolic domain
    _domain_h : psydac.
        Discretized domain

    _ncells : list of ints
        Number of cells of the domain

    _static_fields : dict
        Named static fields
    _snapshot_fields : dict
        Named time dependent fields belonging to the same snapshot

    _loaded_t : float
        Time of the loaded snapshot
    _loaded_ts : int
        Time step of the loaded snapshot

    _snapshot_list : list
        List of all the snapshots
    """

    def __init__(self, geometry_file=None, domain=None, space_file=None, fields_file=None, ncells=None, comm=None):
        if geometry_file is None and domain is None:
            raise ValueError('Domain or geometry file needed')
        if geometry_file is not None and domain is not None:
            raise ValueError("Can't provide both geometry_file and domain")
        if geometry_file is None:
            assert ncells is not None

        self.geometry_filename = geometry_file
        self._domain = domain
        self._domain_h = None

        self.space_filename = space_file
        self.fields_filename = fields_file

        self._ncells = ncells
        self._spaces = {}
        self._static_fields = {}
        self._snapshot_fields = {}
        
        self._last_loaded_fields = None

        self._loaded_t = None
        self._loaded_ts = None
        self._snapshot_list = None

        self.comm = comm
        self.fields_file = None

        self.data_exchanger = None

        self._reconstruct_spaces()
        self.get_snapshot_list()

    @property
    def spaces(self):
        return self._spaces

    @property
    def domain(self):
        return self._domain
    
    @property
    def fields(self):
        fields = {}
        fields.update(self._snapshot_fields)
        fields.update(self._static_fields)
        return fields
    
    def read_space_info(self):
        """Read ``self.space_filename ``.

        Returns
        -------
        dict
            Informations about the spaces.
        """
        return yaml.load(open(self.space_filename), Loader=yaml.SafeLoader)

    def _reconstruct_spaces(self):
        """Reconstructs all of the spaces from reading the files.

        """
        if self.geometry_filename  is not None:
            domain = Domain.from_file(self.geometry_filename)
            domain_h = discretize(domain, filename=self.geometry_filename, comm=self.comm)
        else:
            domain = self._domain
            domain_h = discretize(domain, ncells=self._ncells, comm=self.comm)

        self._domain = domain
        self._domain_h = domain_h
        
        space_info = self.read_space_info()

        pdim = space_info['ndim']

        assert pdim == domain.dim
        assert space_info['fields'] == self.fields_filename 

        # No Multipatch Support for now
        assert len(domain_h.mappings) == 1
        assert not hasattr(domain.interior, 'as_tuple')

        # -------------------------------------------------
        # Space reconstruction
        # -------------------------------------------------
        common_to_all_patches = []
        for patch in space_info['patches']:
            try:
                scalar_spaces = patch['scalar_spaces']
            except KeyError:
                scalar_spaces = {}
            try:
                vector_spaces = patch['vector_spaces']
            except KeyError:
                vector_spaces = {}
            already_used_names = []

            for v_sp in vector_spaces:
                components = v_sp['components']
                temp_v_sp = VectorFunctionSpace(name=v_sp['name'], domain=domain, kind=v_sp['kind'])

                basis = []
                for sc_sp in components:
                    already_used_names.append(sc_sp['name'])
                    basis += sc_sp['basis']

                basis = list(set(basis))
                if len(basis) != 1:
                    raise NotImplementedError("Discretize doesn't support two different bases")

                temp_kwargs_discretization = {
                    'degree': [sc_sp['degree'] for sc_sp in components],
                    'knots': [[np.asarray(sc_sp['knots'][i]) for i in range(sc_sp['ldim'])] for sc_sp in components],
                    'basis': basis[0],
                    'periodic': [sc_sp['periodic'] for sc_sp in components]
                }

                self._spaces[v_sp['name']] = discretize(temp_v_sp, domain_h, **temp_kwargs_discretization)

            for sc_sp in scalar_spaces:
                if sc_sp['name'] not in already_used_names:
                    temp_sc_sp = ScalarFunctionSpace(sc_sp['name'], domain, kind=sc_sp['kind'])

                    basis = list(set(sc_sp['basis']))
                    if len(basis) != 1:
                        raise NotImplementedError("Discretize doesn't support two different bases")

                    temp_kwargs_discretization = {
                        'degree': sc_sp['degree'],
                        'knots': [np.asarray(sc_sp['knots'][i]) for i in range(sc_sp['ldim'])],
                        'basis': basis[0],
                        'periodic': sc_sp['periodic'],
                        'comm': self.comm,
                    }

                    self._spaces[sc_sp['name']] = discretize(temp_sc_sp, domain_h, **temp_kwargs_discretization)

    def get_snapshot_list(self):
        kwargs = {}
        if self.comm is not None and self.comm.size > 1:
            kwargs.update(driver='mpio', comm=self.comm)
        fh5 = h5.File(self.fields_filename, mode='r', **kwargs)
        self._snapshot_list = []
        for k in fh5.keys():
            if k != 'static':
                self._snapshot_list.append(int(k[-4:]))
        self.fields_file = fh5

    def load_static(self, *fields):
        """Reads static fields from file.
        
        Parameters
        ----------
        *fields : tuple of str
            Names of the fields to load
        """
        fh5 = self.fields_file

        static_group = fh5['static']
        temp_space_to_field = {}
        for patch in static_group.keys():
            patch_group = static_group[patch]
            
            for space_name in patch_group.keys():
                space_group = patch_group[space_name]

                if 'parent_space' in space_group.attrs.keys():  # VectorSpace/Field case
                    relevant_space_name = space_group.attrs['parent_space']

                    for field_dset_key in space_group.keys():
                        field_dset = space_group[field_dset_key]
                        relevant_field_name = field_dset.attrs['parent_field']

                        if relevant_field_name in fields:

                            coeff = field_dset

                            # Exceptions to take care of when the dicts are empty
                            try:
                                temp_space_to_field[relevant_space_name][relevant_field_name].append(coeff)
                            except KeyError:
                                try:
                                    temp_space_to_field[relevant_space_name][relevant_field_name] = [coeff]
                                except KeyError:
                                    try:
                                        temp_space_to_field[relevant_space_name] = {relevant_field_name:
                                                                                        [coeff]
                                                                                        }
                                    except KeyError:
                                        temp_space_to_field = {relevant_space_name:
                                                            {relevant_field_name: [coeff]}
                                                                }

                else:  # Scalar case
                    V = self._spaces[space_name].vector_space
                    index = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
                    for field_dset_name in space_group.keys():
                        if field_dset_name in fields:
                            new_field = FemField(self._spaces[space_name])
                            new_field.coeffs[index] = space_group[field_dset_name][index]

                            self._static_fields[field_dset_name] = new_field
 
        for space_name, field_dict in temp_space_to_field.items():
            
            for field_name, list_coeffs in field_dict.items():

                new_field = FemField(self._spaces[space_name])

                for i, coeff in enumerate(list_coeffs):
                    Vi = self._spaces[space_name].vector_space.spaces[i]
                    index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))

                    new_field.coeffs[i][index] = coeff[index]

                self._static_fields[field_name] = new_field

        self._last_loaded_fields = self._static_fields


    def load_snapshot(self, n, *fields):
        """Reads a particular snapshot from file

        Parameters
        ----------
        n : int
            number of the snapshot
        *fields : tuple of str
            Names of the fields to load
        """
        fh5 = self.fields_file

        snapshot_group = fh5[f'snapshot_{n:0>4}']
        temp_space_to_field = {}
        for patch in snapshot_group.keys():
            patch_group = snapshot_group[patch]

            if patch == self._domain.name:
                for space_name in patch_group.keys():
                    space_group = patch_group[space_name]

                    if 'parent_space' in space_group.attrs.keys():  # VectorSpace/Field case
                        relevant_space_name = space_group.attrs['parent_space']

                        for field_dset_key in space_group.keys():
                            field_dset = space_group[field_dset_key]
                            relevant_field_name = field_dset.attrs['parent_field']
                            if relevant_field_name in fields:

                                coeff = field_dset

                                # Exceptions to take care of when the dicts are empty
                                try:
                                    temp_space_to_field[relevant_space_name][relevant_field_name].append(coeff)
                                except KeyError:
                                    try:
                                        temp_space_to_field[relevant_space_name][relevant_field_name] = [coeff]
                                    except KeyError:
                                        try:
                                            temp_space_to_field[relevant_space_name] = {relevant_field_name:
                                                                                            [coeff]
                                                                                            }
                                        except KeyError:
                                            temp_space_to_field = {relevant_space_name:
                                                                {relevant_field_name: [coeff]}
                                                                    }

                    else:  # Scalar case
                            V = self._spaces[space_name].vector_space
                            index = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
                            for field_dset_name in space_group.keys():
                                if field_dset_name in fields:
                                    # Try to reuse memory
                                    try:
                                        self._snapshot_fields[field_dset_name].coeffs[index] = space_group[field_dset_name][index]
                                        
                                        # Ghost regions are not in sync anymore
                                        if self.comm is not None:
                                            self._snapshot_fields[field_dset_name].coeffs.ghost_regions_in_sync = False
                                    except KeyError:
                                        new_field = FemField(self._spaces[space_name])
                                        new_field.coeffs[index] = space_group[field_dset_name][index]

                                        self._snapshot_fields[field_dset_name] = new_field

        for space_name, field_dict in temp_space_to_field.items():
            for field_name, list_coeffs in field_dict.items():
                # Try to reuse memory
                try:
                    for i, coeff in enumerate(list_coeffs):
                        Vi = self._spaces[space_name].vector_space.spaces[i]
                        index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))
                        self._snapshot_fields[field_name].coeffs[i][index] = coeff[index]
                        # Ghost regions are not in sync anymore
                        if self.comm is not None:
                            self._snapshot_fields[field_dset_name].coeffs.ghost_regions_in_sync = False
                except KeyError:
                    new_field = FemField(self._spaces[space_name])

                    for i, coeff in enumerate(list_coeffs):
                        Vi = self._spaces[space_name].vector_space.spaces[i]
                        index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))

                        new_field.coeffs[i][index] = coeff[index]
                    self._snapshot_fields[field_name] = new_field

        self._loaded_t = snapshot_group.attrs['t']
        self._loaded_ts = snapshot_group.attrs['ts']

        for key in list(self._snapshot_fields.keys()):
            if key not in fields:
                del self._snapshot_fields[key]

        self._last_loaded_fields = self._snapshot_fields

    def export_to_vtk(self, 
                      filename_pattern, 
                      grid, 
                      npts_per_cell=None, 
                      snapshots='none', 
                      lz=4, 
                      logical_grid=False, 
                      fields=None, 
                      additional_physical_functions=None,
                      additional_logical_functions=None,
                      color_by_rank=True,
                      debug=False):
        """Exports some fields to vtk. 

        Parameters
        ----------
        filename_pattern: str
            file pattern of the file

        grid: List of ndarray
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        snapshot: list of int or 'all' or 'none', default='none'
            If a list is given, it will export every snapshot present in the list.
            If 'none', only the static fields will be exported.
            Finally, if 'all', will export every time step and the static part.
        
        lz: int, default=4
            Number of leading zeros in the time indexing of the files. 
            Only used if ``snapshot`` is not ``'none'``. 

        fields: dict
            Dictionary with the fields to export as values and the name under which to export them as keys
        
        additional_physical_functions : dict
            Dictionary of callable functions. Those functions will be called on (x_mesh, y_mesh, z_mesh)

        additional_logical_functions : dict
            Dictionary of callable functions. Those functions will be called on the grid.
        
        color_by_rank : bool, default=True
            Adds a cellData attribute that represents the rank of the process that created the file. 

        debug : bool, default=False
            If true, returns ``(mesh, pointData_list)`` where ``mesh`` is ``(x_mesh, y_mesh,  z_mesh)``
            and ``pointData_list`` is the list of all the pointData dictionaries.

        Notes
        -----
        This function only supports regular tensor grid.
        """
        # =================================================
        # Common to everything
        # =================================================
        
        # Get Mappings
        mappings = self._domain_h.mappings

        # Do not support Multipatch domains.
        if len(mappings.values()) != 1:
            raise NotImplementedError("Multipatch not supported yet")

        ldim = self._domain_h.ldim
        # Singular mapping
        mapping = list(mappings.values())[0]
        if isinstance(mapping, SplineMapping):
            local_domain = mapping.space.local_domain
            space_0 = mapping.space

        elif isinstance(mapping, Mapping):
            if self._spaces is not {}:
                space_0 = list(self._spaces.values())[0]
                try:
                    local_domain = space_0.local_domain
                except AttributeError:
                    local_domain = space_0.spaces[0].local_domain
            else:
                space_0s = ScalarFunctionSpace('s', self._domain)
                space_0 = discretize(space_0s, self._domain_h)
                try:
                    local_domain = space_0.local_domain
                except AttributeError:
                    local_domain = space_0.spaces[0].local_domain

        self._pushforward = Pushforward(mapping, grid, npts_per_cell=npts_per_cell)

        # Easy way to ensure everything is 3D
        if ldim == 2:
            slice_3d = (slice(0, None, 1), slice(0, None, 1), None)
        elif ldim == 3:
            slice_3d = (slice(0, None, 1), slice(0, None, 1), slice(0, None, 1))
        else:
            raise NotImplementedError("1D case not supported")

        # Check the grid argument
        assert len(grid) == ldim
        grid_test = [np.asarray(grid[i]) for i in range(ldim)]
        assert all(grid_test[i].ndim == grid_test[i+1].ndim for i in range(len(grid) - 1))
        
        if grid_test[0].ndim == 1 and npts_per_cell is not None:
            # Account for only an int being given
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * ldim

            # Check that the grid is regular
            assert all(grid_test[i].size % npts_per_cell[i] == 0 for i in range(ldim))

            grid_local = []
            for i in range(len(grid_test)):
                grid_local.append(grid_test[i][local_domain[0][i] * npts_per_cell[i]:
                                                (local_domain[1][i] + 1) * npts_per_cell[i]])
                
            cell_indexes = None

        elif grid_test[0].ndim == 1 and npts_per_cell is None:
            cell_indexes = [cell_index(space_0.breaks[i], grid_test[i]) for i in range(ldim)]

            grid_local = []
            for i in range(ldim):
                i_start = np.searchsorted(cell_indexes[i], local_domain[0][i], side='left')
                i_end = np.searchsorted(cell_indexes[i], local_domain[1][i], side='right')
                grid_local.append(grid_test[i][i_start:i_end])

        elif grid_test[0].ndim == ldim:
            raise NotImplementedError("Not Supported Yet")
        else:
            raise ValueError("Wrong input for the grid parameters")

        mesh_grids = np.meshgrid(*grid_local, indexing = 'ij')

        cellData = None
        cellData_info = None    

        if isinstance(mapping, SplineMapping):
            x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid, npts_per_cell=npts_per_cell, overlap=0)
        elif isinstance(mapping, Mapping):
            call_map = mapping.get_callable_mapping
            if ldim == 2:
                fx, fy = call_map._func_eval
                x_mesh = fx(*mesh_grids)[:, :, None]
                y_mesh = fy(*mesh_grids)[:, :, None]
                z_mesh = np.zeros_like(x_mesh)
            elif ldim == 3:
                fx, fy, fz = call_map._func_eval
                x_mesh = fx(*mesh_grids)
                y_mesh = fy(*mesh_grids)
                z_mesh = fz(*mesh_grids)                

        conn, offsets, celltypes, cell_shape = self._compute_unstructured_mesh_info(local_domain, 
                                                                                    npts_per_cell=npts_per_cell, 
                                                                                    cell_indexes=cell_indexes)

        # Check if launched in parallel
        if self.comm is not None and self.comm.size >1:
            # shortcut
            rank = self.comm.Get_rank()
            size = self.comm.Get_size()

            if color_by_rank:
                if ldim == 2:
                    cellData = {'rank': np.full(tuple(cell_shape) + (1,), rank, dtype='i')}
                elif ldim == 3:
                    cellData = {'rank': np.full(tuple(cell_shape), rank, dtype='i')}
                cellData_info = {'rank': (cellData['rank'].dtype, 1)}
            # Filenames
            filename_static = filename_pattern + f'.{rank}.' + 'static'
            filename_time_dependent = filename_pattern + f'.{rank}'

        else:
            # Naming
            filename_static = filename_pattern + ".static"
            filename_time_dependent = filename_pattern

        if debug:
            debug_result = ((x_mesh, y_mesh, z_mesh), [])

        if fields is None:
            fields={}
        
        if additional_physical_functions is None:
            additional_physical_functions = {}
        
        if additional_logical_functions is None:
            additional_logical_functions = {}

        # ============================
        # Static
        # ============================
        if snapshots in ['all', 'none']:
            if self._static_fields == {}:
                self.load_static(*fields.values())
            
            if self.comm is not None and self.comm.rank == 0:
                general_pointData_static_info = {}

            pointData_static = self._export_to_vtk_helper(x_mesh.shape, fields=fields)
           
            if logical_grid:
                for i in range(ldim):
                    pointData_static[f'x_{i}'] = np.reshape(mesh_grids[i], x_mesh.shape)
            
            for name, f in additional_logical_functions.items():
                data = f(*mesh_grids)
                if isinstance(data, tuple):
                    reshaped_tuple = tuple(np.reshape(data[i], x_mesh.shape))
                    pointData_static[name] = reshaped_tuple
                else:
                    pointData_static[name] = np.reshape(data, x_mesh.shape) 

            if ldim == 2:
                for name, f in additional_physical_functions.items():
                    pointData_static[name] = f(x_mesh, y_mesh)
            elif ldim == 3:
                for name, f in additional_physical_functions.items():
                    pointData_static[name] = f(x_mesh, y_mesh, z_mesh)
            
            if debug:
                debug_result[1].append(pointData_static)
            
            if self.comm is not None and self.comm.rank == 0:
                for name, field in pointData_static.items():
                    if isinstance(field, tuple):
                        general_pointData_static_info[name] = (field[0].dtype, 3)
                    else:
                        general_pointData_static_info[name] = (field.dtype, 1)
            
            # Export static fields to VTK
            unstructuredGridToVTK(filename_static, x_mesh, y_mesh, z_mesh,
                                  connectivity=conn,
                                  offsets=offsets,
                                  cell_types=celltypes,
                                  pointData=pointData_static,
                                  cellData=cellData)

            if self.comm is not None and self.comm.Get_size() > 1 and self.comm.Get_rank() == 0:
                writeParallelVTKUnstructuredGrid(filename_pattern + "_static", coordsdtype=x_mesh.dtype,
                                                 sources=[filename_pattern + f".{r}"+'.static.vtu' for r in range(size)],
                                                 ghostlevel=0,
                                                 pointData=general_pointData_static_info,
                                                 cellData=cellData_info)

        # =================================================
        # Time Dependent part
        # =================================================
        # get which snapshots to go through
       
        if snapshots == 'all':
            snapshots = self._snapshot_list
        if isinstance(snapshots, int):
            snapshots = [snapshots]

        if snapshots == 'none':
            snapshots = []
        
        for i, snapshot in enumerate(snapshots):
            self.load_snapshot(snapshot, *fields.values())
            pointData_i = self._export_to_vtk_helper(x_mesh.shape, fields=fields)

            if logical_grid:
                for k in range(ldim):
                    pointData_i[f'x_{k}'] = mesh_grids[k]
            
            for name, f in additional_logical_functions.items():
                data = f(*mesh_grids)
                if isinstance(data, tuple):
                    reshaped_tuple = tuple(np.reshape(data[i], x_mesh.shape))
                    pointData_i[name] = reshaped_tuple
                else:
                    pointData_i[name] = np.reshape(data, x_mesh.shape) 
                
            if ldim == 2:
                for name, f  in additional_physical_functions.items():
                    pointData_i[name] = f(x_mesh, y_mesh)
            elif ldim == 3:
                for name, f  in additional_physical_functions.items():
                    pointData_i[name] = f(x_mesh, y_mesh, z_mesh)

            if debug:
                debug_result[1].append(pointData_i)
            
            if self.comm is not None and self.comm.Get_size() > 1 and self.comm.Get_rank() == 0:
    
                general_pointData_time_info = {}
                for name, data in pointData_i.items():
                    if isinstance(data, tuple):
                        general_pointData_time_info[name] = (data[0].dtype, 3)
                    else:
                        general_pointData_time_info[name] = (data.dtype, 1)
                
                writeParallelVTKUnstructuredGrid(filename_pattern + '.{0:0{1}d}'.format(i, lz),
                                    coordsdtype= x_mesh.dtype,
                                    sources=[filename_pattern + f'.{r}' + '.{0:0{1}d}'.format(i, lz) + '.vtu' for r in range(size)],
                                    ghostlevel=0,
                                    pointData=general_pointData_time_info,
                                    cellData=cellData_info)

            unstructuredGridToVTK(filename_time_dependent + '.{0:0{1}d}'.format(i, lz),
                                  x_mesh, y_mesh, z_mesh,
                                  connectivity=conn,
                                  offsets=offsets,
                                  cell_types=celltypes,
                                  pointData=pointData_i,
                                  cellData=cellData)
    
        if debug:
            return debug_result

    def _export_to_vtk_helper(self, shape, fields=None):
        """
        Helper function to make the proper function easier to read.
        The correct fields are supposed to be already loaded. 

        Parameters
        ----------

        shape : tuple
            Shape of the mesh
        
        fields : dict, optional
            
        """
        fields_relevant = {}
        for vtk_name, f_name in fields.items():
            if f_name in self._last_loaded_fields.keys():
                fields_relevant[vtk_name] = self._last_loaded_fields[f_name]
        pointData_int = self._pushforward(fields=fields_relevant)
        pointData = {}
        if self._domain_h.ldim == 2:
            for (name, field) in pointData_int:
                if isinstance(field, tuple):
                    pointData[name] = (np.reshape(field[0], shape), np.reshape(field[1], shape), np.zeros(shape))
                else:
                    pointData[name] = np.reshape(field, shape)
        else:
            for (name, field) in pointData_int:
                pointData[name] = field

        return pointData

    def _compute_unstructured_mesh_info(self, mapping_local_domain, npts_per_cell=None, cell_indexes=None):
        """
        Computes the connection, offset and celltypes arrays for exportation
        as VTK unstructured grid.

        Parameters
        ----------

        mapping_local_domain : tuple of tuple

        npts_per_cell : tuple of ints or ints, optional

        cell_indexes : tuple of arrays, optional

        Return 
        ------
        connectivity : ndarray
            1D array containing the connectivity between points
        offsets : ndarray 
            1D array containing the index of the last vertex of each cell
        celltypes : ndarray
            1D array containing the type ID of each cell
        cellshape : tuple
            Number of cell in each direction.
        """
        starts, ends = mapping_local_domain
        ldim = len(starts)

        if npts_per_cell is not None:
            n_elem = tuple(ends[i] + 1 - starts[i] for i in range(ldim))
            
            cellshape = np.array(n_elem) * (np.array(npts_per_cell) - 1)
            total_number_cells_vtk = np.prod(cellshape)
            celltypes = np.zeros(total_number_cells_vtk, dtype='i')
            offsets = np.arange(1, total_number_cells_vtk + 1, dtype='i') * (2 ** ldim)
            connectivity = np.zeros(total_number_cells_vtk * 2 ** ldim)
            if ldim == 2: 
                celltypes[:] = VtkQuad.tid
                cellID = 0
                for i_elem in range(n_elem[0]):
                    for j_elem in range(n_elem[1]):
                        for i_intra in range(npts_per_cell[0] - 1):
                            for j_intra in range(npts_per_cell[1] - 1):
                                row_top = i_elem * npts_per_cell[0] + i_intra
                                col_left = j_elem * npts_per_cell[1] + j_intra

                                topleft = row_top * n_elem[1] * npts_per_cell[1] + col_left
                                topright = topleft + 1
                                botleft = topleft + n_elem[1] * npts_per_cell[1] # next row
                                botright = botleft + 1

                                connectivity[4 * cellID: 4 * cellID + 4] = [topleft, topright, botright, botleft]

                                cellID += 1
            
            elif ldim == 3:
                celltypes[:] = VtkHexahedron.tid
                cellID = 0
                n_cols = n_elem[1] * npts_per_cell[1]
                n_layers = n_elem[2] * npts_per_cell[2]
                for i_elem in range(n_elem[0]):
                    for j_elem in range(n_elem[1]):
                        for k_elem in range(n_elem[2]):
                            for i_intra in range(npts_per_cell[0] - 1):
                                for j_intra in range(npts_per_cell[1] - 1):
                                    for k_intra in range(npts_per_cell[2] - 1):
                                        row_top = i_elem * npts_per_cell[0] + i_intra
                                        col_left = j_elem * npts_per_cell[1] + j_intra
                                        layer_front = k_elem * npts_per_cell[2] + k_intra
                                        
                                        top_left_front = row_top * n_cols * n_layers + col_left * n_layers + layer_front
                                        top_left_back = top_left_front + 1
                                        top_right_front = top_left_front + n_layers # next column
                                        top_right_back = top_right_front + 1

                                        bot_left_front = top_left_front + n_layers * n_cols # next row
                                        bot_left_back = bot_left_front + 1
                                        bot_right_front = bot_left_front + n_layers # next column
                                        bot_right_back = bot_right_front + 1


                                        connectivity[8 * cellID: 8 * cellID + 8] = [
                                            top_left_front, top_right_front, bot_right_front, bot_left_front,
                                            top_left_back, top_right_back, bot_right_back, bot_left_back
                                        ]

                                        cellID += 1
        
        elif cell_indexes is not None:
            i_starts = [np.searchsorted(cell_indexes[i], starts[i], side='left') for i in range(ldim)]
            i_ends = [np.searchsorted(cell_indexes[i], ends[i], side='right') for i in range(ldim)]
            n_points = tuple(i_ends[i] - i_starts[i] for i in range(ldim))

            uniques = [len(np.unique(cell_indexes[i][i_starts[i]: i_ends[i]])) for i in range(ldim)]
            cellshape = np.array([n_points[i] - uniques[i] for i in range(ldim)])
            total_number_cells_vtk = np.prod(cellshape)
            celltypes = np.zeros(total_number_cells_vtk, dtype='i')
            offsets = np.arange(1, total_number_cells_vtk + 1, dtype='i') * (2 ** ldim)
            connectivity = np.zeros(total_number_cells_vtk * 2 ** ldim)
            if ldim == 2:
                cellID = 0
                celltypes[:] = VtkQuad.tid
                for i in range(i_ends[0] - 1 - i_starts[0]):
                    if cell_indexes[0][i] == cell_indexes[0][i + 1]:
                        for j in range(i_ends[1] - 1 - i_starts[1]):
                            if cell_indexes[1][j] == cell_indexes[1][j + 1]:
                                row_top = i
                                col_left = j

                                topleft = row_top * n_points[1] + col_left
                                topright = topleft + 1
                                botleft = topleft + n_points[1]
                                botright = botleft +1

                                connectivity[4 * cellID: 4 * cellID + 4] = [topleft, topright, botright, botleft]
                                cellID += 1

            elif ldim == 3:
                cellID = 0
                celltypes[:] = VtkHexahedron.tid
                n_cols = n_points[1]
                n_layers = n_points[2]
                for i in range(i_ends[0] - 1 - i_starts[0]):
                    if cell_indexes[0][i] == cell_indexes[0][i + 1]:
                        for j in range(i_ends[1] - 1 - i_starts[1]):
                            if cell_indexes[1][j] == cell_indexes[1][j + 1]:
                                for k in range(i_ends[2] - 1 - i_starts[2]):
                                    if cell_indexes[2][k] == cell_indexes[2][k + 1]:
                                        row_top = i
                                        col_left = j
                                        layer_front = k

                                        top_left_front = row_top * n_cols * n_layers + col_left * n_layers + layer_front
                                        top_left_back = top_left_front + 1
                                        top_right_front = top_left_front + n_layers # next column
                                        top_right_back = top_right_front + 1

                                        bot_left_front = top_left_front + n_layers * n_cols # next row
                                        bot_left_back = bot_left_front + 1
                                        bot_right_front = bot_left_front + n_layers # next column
                                        bot_right_back = bot_right_front + 1


                                        connectivity[8 * cellID: 8 * cellID + 8] = [
                                            top_left_front, top_right_front, bot_right_front, bot_left_front,
                                            top_left_back, top_right_back, bot_right_back, bot_left_back
                                        ]

                                        cellID += 1

       
        else:
            raise NotImplementedError("Not Supported Yet")
        return connectivity, offsets, celltypes, cellshape
