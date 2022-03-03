# coding: utf-8
#
# Copyright 2019 Yaman Güçlü

import numpy as np
import pyevtk
import yaml
import re
import h5py as h5

from psydac.cad.geometry import Geometry
from psydac.utilities.utils import refine_array_1d
from psydac.fem.basic import FemSpace, FemField


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
    filename_space : str
         Name/path of the file in which to save the space information.
         The path is relative to the current working directory.

    filename_fields : str
         Name/path of the file in which to save the fields.
         The path is relative to the current working directory.

    Attributes
    ----------
    _spaces_info : dict
        Information about the spaces in a human readable format.
        It is written to ``filename_space`` in yaml.
    _spaces : List
        List of the spaces that were added to an instance of OutputManager.

    filename_space : str
        Name of the file in which to save the space information.
    filename_fields : str
        Name of the file in which to save the fields.

    _next_snapshot_number : int

    is_static : bool or None
        If None, means that no saving scheme was chosen by the user for the next
        ``export_fields``.

    _current_hdf5_group :  h5py.Group
        Group where the fields will be saved in the next ``export_fields``.
    """

    space_types_to_str = {
        'H1SpaceType()': 'h1',
        'HcurlSpaceType()': 'hcurl',
        'HdivSpaceType()': 'hdiv',
        'L2SpaceType()': 'l2',
        'UndefinedSpaceType()': 'undefined',
    }

    def __init__(self, filename_space, filename_fields):

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

    @property
    def current_hdf5_group(self):
        return self._current_hdf5_group

    @property
    def space_info(self):
        return self._spaces_info

    @property
    def spaces(self):
        return dict([(name, space) for name, space in zip(self._spaces[1::2], self._spaces[0::2])])

    def set_static(self):
        """Set the export to static mode

        Returns
        -------
        self
        """
        if self.is_static:
            return self

        self.is_static = True

        file_fields = h5.File(self.filename_fields, 'a')

        if 'static' not in file_fields.keys():
            static_group = file_fields.create_group('static')
            self._current_hdf5_group = static_group
        else:
            self._current_hdf5_group = file_fields['static']

        return self

    def add_snapshot(self, t, ts):
        """Add a snapshot to the fields' HDF5 file
        and set the export mode to time dependent.

        Parameters
        ----------
        t : float
            Floating point time of the snapshot
        ts : int
            Time step of the snapshot

        Returns
        -------
        self
        """
        self.is_static = False

        file_fields = h5.File(self.filename_fields, 'a')

        i = self._next_snapshot_number
        try:
            snapshot = file_fields.create_group(f'snapshot_{i:0>4}')
        except ValueError:
            regexp = re.compile(r'snapshot_(?P<id>\d+)')
            i = max([int(regexp.search(k).group('id')) for k in file_fields.keys() if regexp.search(k) is not None]) + 1
            snapshot = file_fields.create_group(f'snapshot_{i:0>4}')
        snapshot.attrs.create('t', data=t, dtype=float)
        snapshot.attrs.create('ts', data=ts, dtype=int)

        self._next_snapshot_number = i + 1
        self._current_hdf5_group = snapshot

        return self

    def add_spaces(self, **femspaces):
        """Add femspaces to the scope of this instance of OutputManager

        Parameters
        ----------
        femspaces:  psydac.fem.basic.FemSpace dict
            Femspaces to add in the scope of this OutputManager instance.

        """
        assert all(isinstance(femspace, FemSpace) for femspace in femspaces.values())
        for name, femspace in femspaces.items():

            if femspace.is_product:
                self._add_vector_space(femspace, name=name)
            else:
                self._add_scalar_space(femspace, name=name)

    def _add_scalar_space(self, scalar_space, name=None, dim=None, patch_name=None, kind=None):
        """Add a scalar space to the scope of this instance of OutputManager

        Parameters
        ----------
        scalar_space : psydac.fem.tensor.TensorFemSpace
            Scalar space to add to the scope.

        name : str or None, optional
            Name under which to save the space. Will be generated
            by looking at the related symbolic space if not given

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

        if dim is None:
            symbolic_space = scalar_space.symbolic_space
            scalar_space_name = name
            pdim = symbolic_space.domain.dim
            patch = symbolic_space.domain.name
            kind = symbolic_space.kind
        else:
            scalar_space_name = name
            pdim = dim
            patch = patch_name
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
            spaces_info = {'ndim': pdim,
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
        self._spaces.append(scalar_space)
        self._spaces.append(name)

        self._spaces_info = spaces_info

        return new_space

    def _add_vector_space(self, vector_space, name=None):
        """Add a vector space to the scope of this instance of OutputManager.

        Parameters
        ----------
        vector_space: psydac.fem.vector.VectorFemSpace
            Vector FemSpace to add to the scope.
        """

        symbolic_space = vector_space.symbolic_space
        dim = symbolic_space.domain.dim
        patch_name = symbolic_space.domain.name
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
        self._spaces.append(vector_space)
        self._spaces.append(name)

    def export_fields(self, **fields):
        """
        A function that exports the fields' coefficients to an HDF5 file. They are
        saved under a snapshot/patch/space/field scheme or static/patch/space/field.

        If the current saving scheme is time dependent, unsets it to avoid writing twice to
        the same snapshot.

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

        # For now, I assume that if something is mpi parallel everything is
        space_test = list(fields.values())[0].space
        if space_test.is_product:
            comm = space_test.spaces[0].vector_space.cart.comm if space_test.vector_space.parallel else None
        else:
            comm = space_test.vector_space.cart.comm if space_test.vector_space.parallel else None

        # Open HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = {}
        if comm is not None and comm.size > 1:
            kwargs.update(driver='mpio', comm=comm)
        fh5 = h5.File(self.filename_fields, mode='a', **kwargs)

        if 'spaces' not in fh5.attrs.keys():
            fh5.attrs.create('spaces', self.filename_space)

        saving_group = self._current_hdf5_group

        # Add field coefficients as named datasets
        for name_field, field in fields.items():

            i = self._spaces.index(field.space)

            name_space = self._spaces[i+1]
            patch = field.space.symbolic_space.domain.name

            if field.space.is_product:  # Vector field case
                for i, field_coeff in enumerate(field.coeffs):
                    name_field_i = name_field + f'[{i}]'
                    name_space_i = name_space + f'[{i}]'

                    Vi = field.space.vector_space.spaces[i]
                    index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))

                    space_group = saving_group.create_group(f'{patch}/{name_space_i}')
                    space_group.attrs.create('parent_space', data=name_space)

                    dset = saving_group.create_dataset(f'{patch}/{name_space_i}/{name_field_i}',
                                                       shape=Vi.npts, dtype=Vi.dtype)
                    dset.attrs.create('parent_field', data=name_field)
                    dset[index] = field_coeff[index]
            else:
                V = field.space.vector_space
                index = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
                dset = saving_group.create_dataset(f'{patch}/{name_space}/{name_field}', shape=V.npts, dtype=V.dtype)
                dset[index] = field.coeffs[index]

        # Close HDF5 file
        fh5.close()

        if self.is_static:
            self.is_static = None

    def export_space_info(self):
        with open(self.filename_space, 'w') as f:
            yaml.dump(data=self._spaces_info, stream=f, default_flow_style=None, sort_keys=False)


class PostProcessManager:

    def __init__(self, geometry_filename, space_filename, fields_filename):
        self.geometry_file = geometry_filename
        self.space_file = space_filename
        self.fields_file = fields_filename

        self._spaces = {}
        self._fields = {}
        self._domain = None

        self._scope_reconstructed = False

    @property
    def spaces(self):
        return self._spaces

    @property
    def fields(self):
        return self._fields

    @property
    def domain(self):
        return self._domain

    def read_space_info(self):
        return yaml.load(open(self.space_file), Loader=yaml.SafeLoader)

    def reconstruct_scope(self):

        from sympde.topology import Domain, VectorFunctionSpace, ScalarFunctionSpace
        from psydac.api.discretization import discretize

        domain = Domain.from_file(self.geometry_file)
        domain_h = discretize(domain, filename=self.geometry_file)

        self._domain = domain_h
        space_info = self.read_space_info()

        pdim = space_info['ndim']

        assert pdim == domain.dim
        assert space_info['fields'] == self.fields_file

        # -------------------------------------------------
        # Space reconstruction
        # -------------------------------------------------
        for patch in space_info['patches']:
            if patch['name'] == domain.name:
                scalar_spaces = patch['scalar_spaces']
                vector_spaces = patch['vector_spaces']

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
                        'knots': [sc_sp['knots'] for sc_sp in components],
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
                            'knots': sc_sp['knots'],
                            'basis': basis[0],
                            'periodic': sc_sp['periodic'],
                        }

                        self._spaces[sc_sp['name']] = discretize(temp_sc_sp, domain_h, **temp_kwargs_discretization)

        # -------------------------------------------------
        # Fields reconstruction
        # -------------------------------------------------
        # TODO: Work out parallel case
        fh5 = h5.File(self.fields_file, 'r')

        # Sanity Check
        if fh5.attrs['spaces'] != self.space_file:
            raise ValueError(f"Unexpected inconsistency between {self.space_file} and {self.fields_file}")

        temp_space_to_field = {}

        for k in fh5.keys():
            global_group = fh5[k]
            if k == 'static':
                for patch in global_group.keys():
                    patch_group = global_group[patch]

                    if patch == domain.name:
                        for space_name in fh5[k][patch].keys():
                            space_group = patch_group[space_name]

                            if 'parent_space' in space_group.attrs.keys():  # VectorSpace/Field case
                                relevant_space_name = space_group.attrs['parent_space']

                                for field_dset_key in space_group.keys():
                                    field_dset = space_group[field_dset_key]
                                    relevant_field_name = field_dset.attrs['parent_field']

                                    coeff = field_dset[...]

                                    # Exceptions to take care of when the dicts are empty
                                    try:
                                        temp_space_to_field[k][relevant_space_name][relevant_field_name].append(coeff)
                                    except KeyError:
                                        try:
                                            temp_space_to_field[k][relevant_space_name][relevant_field_name] = [coeff]
                                        except KeyError:
                                            try:
                                                temp_space_to_field[k][relevant_space_name] = {relevant_field_name:
                                                                                               [coeff]
                                                                                               }
                                            except KeyError:
                                                temp_space_to_field[k] = {relevant_space_name:
                                                                          {relevant_field_name: [coeff]}
                                                                          }

                            else:  # Scalar case
                                V = self._spaces[space_name].vector_space
                                index = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
                                for field_dset_name in space_group.keys():
                                    new_field = FemField(self._spaces[space_name])

                                    new_field.coeffs[index] = space_group[field_dset_name][index]

                                    try:
                                        self._fields[k]['fields'][field_dset_name] = new_field
                                    except KeyError:
                                        self._fields[k] = {'fields': {field_dset_name: new_field}}

            else:
                t = global_group.attrs['t']
                ts = global_group.attrs['ts']
                regexp = re.compile(r'snapshot_(?P<id>\d+)')
                snp_number = int(regexp.search(k).group('id'))

                for patch in global_group.keys():
                    patch_group = global_group[patch]

                    if patch == domain.name:
                        for space_name in fh5[k][patch].keys():
                            space_group = patch_group[space_name]

                            if 'parent_space' in space_group.attrs.keys():  # VectorSpace/Field case
                                relevant_space_name = space_group.attrs['parent_space']

                                for field_dset_key in space_group.keys():
                                    field_dset = space_group[field_dset_key]
                                    relevant_field_name = field_dset.attrs['parent_field']

                                    coeff = field_dset[...]

                                    # Exceptions to take care of when the dicts are empty
                                    try:
                                        temp_space_to_field[snp_number][relevant_space_name][relevant_field_name] \
                                            .append(coeff)

                                    except KeyError:
                                        try:
                                            temp_space_to_field[snp_number][relevant_space_name] \
                                                [relevant_field_name] = [coeff]

                                        except KeyError:
                                            try:
                                                temp_space_to_field[snp_number][relevant_space_name] = {
                                                    relevant_field_name: [coeff]
                                                }
                                            except KeyError:
                                                temp_space_to_field[snp_number] = {relevant_space_name:
                                                                                   {relevant_field_name: [coeff]},
                                                                                   'time': t,
                                                                                   'timestep': ts
                                                                                   }

                            else:  # Scalar case
                                V = self._spaces[space_name].vector_space
                                index = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
                                for field_dset_name in space_group.keys():
                                    new_field = FemField(self._spaces[space_name])

                                    new_field.coeffs[index] = space_group[field_dset_name][index]

                                    try:
                                        self._fields[snp_number]['fields'][field_dset_name] = new_field
                                    except KeyError:
                                        self._fields[snp_number] = {'fields': {field_dset_name: new_field},
                                                                    'time': t,
                                                                    'timestep': ts,
                                                                    }

        for k, space_dict in temp_space_to_field.items():
            for space_name, field_dict in space_dict.items():
                if space_name != 'time' and space_name != 'timestep':
                    for field_name, list_coeffs in field_dict.items():

                        new_field = FemField(self._spaces[space_name])

                        for i, coeff in enumerate(list_coeffs):
                            Vi = self._spaces[space_name].vector_space.spaces[i]
                            index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))

                            new_field.coeffs[i][index] = coeff[index]

                        try:
                            self._fields[k]['fields'][field_name] = new_field
                        except KeyError:
                            if k == 'static':
                                self._fields[k] = {'fields': {field_name: new_field}}
                            else:
                                self._fields[k] = {
                                    'fields': {field_name: new_field},
                                    'time': temp_space_to_field[k]['time'],
                                    'timestep': temp_space_to_field[k]['timestep'],
                                }
        self._scope_reconstructed = True

    def export_to_vtk(self, filename_pattern, grid, npts_per_cell=None, dt=None, export_static=True, lz=4, **fields):
        """Exports all the static fields and some or all of the time dependent fields.

        Parameters
        ----------
        filename_pattern: str
            file pattern of the file

        grid: List of ndarray
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        export_static: bool, default=True
            If True, export the static fields,
            if False don't.

        lz: int, default=4
            Number of leading zeros in the time indexing of the files.

        dt: int or list of int or None, optional
            If an int is given, will export every dt^th snapshot.
            If a list is given instead, will export every snapshot present in the list.
            Finally, if None, will export every time step.

        fields: dict
            Dictionary with the as the fields to export and as values the name under which to export them

        Notes
        -----
        This function only supports regular tensor grid.
        """
        # =================================================
        # Common to everything
        # =================================================
        # Make sure we called reconstruct_scope
        if not self._scope_reconstructed:
            self.reconstruct_scope()

        # Get Mapping
        mappings = self._domain.mappings

        if len(mappings.values()) != 1:
            raise NotImplementedError("Multipatch not supported yet")

        mapping = list(mappings.values())[0]

        ldim = mapping.ldim

        if isinstance(npts_per_cell, int):
            npts_per_cell = (npts_per_cell,) * ldim

        # Only regular tensor product grid is supported for now
        grid_test = [np.asarray(grid[i]) for i in range(ldim)]
        assert grid_test[0].ndim == 1 and npts_per_cell is not None
        assert all(grid_test[i].ndim == grid_test[i+1].ndim for i in range(len(grid) - 1))
        assert all(grid_test[i].size % npts_per_cell[i] == 0 for i in range(ldim))

        # Coordinates of the mesh, C Contiguous arrays
        x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid, npts_per_cell=npts_per_cell)

        if ldim == 2:
            slice_3d = (slice(0, None, 1), slice(0, None, 1), None)
        elif ldim == 3:
            slice_3d = (slice(0, None, 1), slice(0, None, 1), slice(0, None, 1))
        else:
            raise NotImplementedError("1D case not Implemented yet")

        # ============================
        # Static
        # ============================
        if export_static:
            pointData_static = {}
            smart_eval_dict = {}
            if 'static' in self._fields.keys():
                for f_name, field in self._fields['static']['fields'].items():
                    if f_name in fields.keys():
                        try:
                            smart_eval_dict[field.space][0].append(field)
                            smart_eval_dict[field.space][1].append(fields[f_name])
                        except KeyError:
                            smart_eval_dict[field.space] = ([field], [fields[f_name]])

                for space, (field_list_to_eval, name_list) in smart_eval_dict.items():
                    pushed_fields = space.pushforward_fields(grid,
                                                             *field_list_to_eval,
                                                             mapping=mapping,
                                                             npts_per_cell=npts_per_cell)

                    if not isinstance(pushed_fields[0], list):
                        pushed_fields = [[pushed_fields[i]] for i in range(len(pushed_fields))]

                    for i in range(len(name_list)):
                        if len(pushed_fields[i]) == 1:
                            # Means that this is a Scalar space.
                            pointData_static[name_list[i]] = pushed_fields[i][0][slice_3d]
                        else:
                            # Means that this is a vector/product space and that we need to turn the
                            # result into a 3-tuple (x_component, y_component, z_component)
                            tuple_fields = tuple(pushed_fields[i][j][slice_3d] for j in range(ldim)) \
                                                 + (3-ldim) * (np.zeros_like(pushed_fields[i][0])[slice_3d],)
                            pointData_static[name_list[i]] = tuple_fields

                # Export static fields to VTK
                pyevtk.hl.gridToVTK(f'{filename_pattern}_static', x_mesh, y_mesh, z_mesh,
                                    pointData=pointData_static)

        # =================================================
        # Time Dependent part
        # =================================================
        # get which snapshots to go through
        if dt is None:
            dt = [k for k in self._fields.keys() if k != 'static']

        if isinstance(dt, int):
            dt = [dt * i for i in range((len(self._fields.keys()) - 1)//dt)]

        smart_eval_dict = {}
        for i, snapshot in enumerate(dt):
            for f_name, field in self._fields[snapshot]['fields'].items():
                if f_name in fields.keys():
                    try:
                        smart_eval_dict[field.space][0].append(field)
                        smart_eval_dict[field.space][1].append(fields[f_name])
                        smart_eval_dict[field.space][2].append(i)
                    except KeyError:
                        smart_eval_dict[field.space] = ([field], [fields[f_name]], [i])

        pointData_full = {}

        for space, (field_list_to_eval, name_list, time_list) in smart_eval_dict.items():
            pushed_fields = space.pushforward_fields(grid,
                                                     *field_list_to_eval,
                                                     mapping=mapping,
                                                     npts_per_cell=npts_per_cell)

            if not isinstance(pushed_fields[0], list):
                # This is space-dependent, so we only need to check for the first index.
                pushed_fields = [[pushed_fields[i]] for i in range(len(pushed_fields))]


            previous_time = time_list[0]
            pointData_time = {}
            for i, name in enumerate(name_list):

                # Check if we are in the same snapshot as before, if yes do nothing,
                # if not, we save the current pointData under the appropriate timestamp
                current_time = time_list[i]

                if current_time != previous_time:

                    try:
                        pointData_full[previous_time].update(pointData_time.copy())
                    except KeyError:
                        pointData_full[previous_time] = pointData_time.copy()

                    previous_time = current_time

                # Format the results
                if len(pushed_fields[i]) == 1:
                    pointData_time[name_list[i]] = pushed_fields[i][0][slice_3d]
                else:
                    # Means that this is a vector/product space and we need to turn the
                    # result into a 3-tuple (x_component, y_component, z_component)
                    tuple_fields = tuple(pushed_fields[i][j][slice_3d] for j in range(ldim)) \
                                   + (3 - ldim) * (np.zeros_like(pushed_fields[i][0])[slice_3d],)
                    pointData_time[name_list[i]] = tuple_fields

            # Save at the end of the loop
            try:
                pointData_full[time_list[len(name_list) - 1]].update(pointData_time.copy())
            except KeyError:
                pointData_full[time_list[len(name_list) - 1]] = pointData_time.copy()

        for i, pointData_full_i in enumerate(pointData_full.values()):
            pyevtk.hl.gridToVTK(filename_pattern + '_{0:0{1}d}'.format(i, lz),
                                x_mesh, y_mesh, z_mesh, pointData=pointData_full_i)
