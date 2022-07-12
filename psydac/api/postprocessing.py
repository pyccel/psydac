# coding: utf-8
#
# Copyright 2019 Yaman Güçlü

import numpy as np
from requests import patch
import yaml
import re
import h5py as h5

from sympde.topology.mapping import Mapping
from sympde.topology.analytical_mapping import IdentityMapping
from sympde.topology import Domain, VectorFunctionSpace, ScalarFunctionSpace, InteriorDomain, MultiPatchMapping, Mapping
from sympde.topology.datatype import H1SpaceType, HcurlSpaceType, HdivSpaceType, L2SpaceType, UndefinedSpaceType

from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkHexahedron, VtkQuad

from psydac.api.discretization import discretize
from psydac.cad.geometry import Geometry
from psydac.mapping.discrete import SplineMapping
from psydac.core.bsplines import cell_index
from psydac.feec.pushforward import Pushforward
from psydac.utilities.utils import refine_array_1d
from psydac.fem.basic import FemSpace, FemField
from psydac.utilities.vtk import writeParallelVTKUnstructuredGrid
from psydac.core.bsplines import elevate_knots


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
    
    comm : mpi4py.MPI.Intracomm or None, optional
        Communicator
    
    save_mpi_rank : bool
        If True, then the MPI rank are saved alongside the domain.
        i.e. for each patch, there will be an attribute which maps the MPI rank
        to which part of the domain it holds (cell indices).

    mode : str in {'r', 'r+', 'w', 'w-', 'x', 'a'}, default='w'
        Opening mode of the HDF5 file.

    Attributes
    ----------
    _space_info : dict
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
        List of the names of the statically saved fields.
    
    _space_types_to_str : dict

    _spaces_names : list
        List of the names of the saved spaces.

    _comm : mpi4py.MPI.Intracomm or None

    _mode : str in {'r', 'r+', 'w', 'w-', 'x', 'a'}

    _fields_file : h5py.File
    """

    _space_types_to_str = {
        H1SpaceType(): 'h1',
        HcurlSpaceType(): 'hcurl',
        HdivSpaceType(): 'hdiv',
        L2SpaceType(): 'l2',
        UndefinedSpaceType(): 'undefined',
    }

    def __init__(self, filename_space, filename_fields, comm=None, mode='w', save_mpi_rank=True):

        self._space_info = {}
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

        self._mode = mode

        self.comm = comm
        if self.comm is not None and self.comm.size > 1:
            self._save_mpi_rank = save_mpi_rank
        else:
            self._save_mpi_rank = False

        self.fields_file = None
    
    def close(self):
        if not self.fields_file is None:
            self.fields_file.close()

    @property
    def current_hdf5_group(self):
        return self._current_hdf5_group

    @property
    def space_info(self):
        return self._space_info

    @property
    def spaces(self):
        return dict([(name, space) for name, space in zip(self._spaces[1::3], self._spaces[0::3])])

    def set_static(self):
        """
        Sets the export to static mode
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
        """
        Adds a snapshot to the fields' HDF5 file
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
        """
        Add spaces to the scope.

        Parameters
        ----------
        femspaces: dict
            Named femspaces
        
        Note
        ----
        Femspaces are added to ``self._space_info``.
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
        """
        Adds a scalar space to the scope.

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
        spaces_info = self._space_info
        
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
        multiplicity = [int(m) for m in scalar_space.multiplicity]
        
        new_space = {'name': scalar_space_name,
                     'ldim': ldim,
                     'kind': self._space_types_to_str[kind],
                     'dtype': dtype,
                     'rational': False,
                     'periodic': periodic,
                     'degree': degree,
                     'multiplicity': multiplicity,
                     'basis': basis,
                     'knots': knots
                     }
        if spaces_info == {}:
            spaces_info = {
                           'ndim': pdim,
                           'fields': self.filename_fields,
                           'patches': [{'name': patch,
                                        'breakpoints': [scalar_space.breaks[i].tolist() for i in range(ldim)],
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
                assert all(scalar_space.breaks[i].tolist() == spaces_info['patches'][patch_index]['breakpoints'][i] 
                            for i in range(ldim))
                spaces_info['patches'][patch_index]['scalar_spaces'].append(new_space)
            else:
                spaces_info['patches'].append({'name': patch, 
                                               'breakpoints': [scalar_space.breaks[i].tolist() for i in range(ldim)],
                                               'scalar_spaces': [new_space]})

        self._space_info = spaces_info

        return new_space

    def _add_vector_space(self, vector_space, name=None, patch_name=None):
        """
        Adds a vector space to the scope.
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
                                                   kind=UndefinedSpaceType())
            scalar_spaces_info.append(sc_space_info)

        spaces_info = self._space_info

        new_vector_space = {'name': name,
                            'kind': self._space_types_to_str[kind],
                            'components': scalar_spaces_info,
                            }

        patch_index = [spaces_info['patches'][i]['name']
                       for i in range(len(spaces_info['patches']))].index(patch_name)

        try:
            spaces_info['patches'][patch_index]['vector_spaces'].append(new_vector_space)
        except KeyError:
            spaces_info['patches'][patch_index].update({'vector_spaces': [new_vector_space]})

        self._space_info = spaces_info

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
        
        assert len(fields) > 0

        fh5 = self.fields_file

        if self._save_mpi_rank:
            if not 'mpi_dd' in fh5.keys():
                mpi_dd_gp = fh5.create_group('mpi_dd')
            else:
                mpi_dd_gp = fh5['mpi_dd']

        if not 'spaces' in fh5.attrs.keys():
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

                    if self._save_mpi_rank:
                        if not name_patch in mpi_dd_gp.keys():
                            rank = self.comm.Get_rank()
                            size = self.comm.Get_size()


                            if f.space.is_product:
                                sp = f.space.spaces[0]
                            else:
                                sp = f.space
                            try:
                                local_domain = np.array(sp.local_domain)
                            except AttributeError: #empty space
                                local_domain = np.array((0,) * sp.ldim, (-1,) * sp.ldim)
                            mpi_dd_gp.create_dataset(f'{name_patch}', shape=(size, *local_domain.shape))
                            mpi_dd_gp[name_patch][rank] = local_domain 

                    if f.space.is_product:  # Vector field case
                        for i, field_coeff in enumerate(f.coeffs):
                            name_field_i = name_field + f'[{i}]'
                            name_space_i = name_space + f'[{i}]'

                            Vi = f.space.vector_space.spaces[i]
                            index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))

                            try:
                                space_group = saving_group[f'{name_patch}/{name_space_i}']
                            except KeyError:
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

                if self._save_mpi_rank:
                    if not name_patch in mpi_dd_gp.keys():
                        rank = self.comm.Get_rank()
                        size = self.comm.Get_size()

                        if f.space.is_product:
                            sp = f.space.spaces[0]
                        else:
                            sp = f.space
                        try:
                            local_domain = np.array(sp.local_domain)
                        except AttributeError: #empty space
                            local_domain = np.array((0,) * sp.ldim, (-1,) * sp.ldim)
                        mpi_dd_gp.create_dataset(f'{name_patch}', shape=(size, *local_domain.shape))
                        mpi_dd_gp[name_patch][rank] = local_domain

                if field.space.is_product:  # Vector field case
                    for i, field_coeff in enumerate(field.coeffs):
                        name_field_i = name_field + f'[{i}]'
                        name_space_i = name_space + f'[{i}]'

                        Vi = field.space.vector_space.spaces[i]
                        index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))
                        
                        try:
                            space_group = saving_group[f'{name_patch}/{name_space_i}']
                        except KeyError:
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
        """
        Export the space info to Yaml.
        """
        if self.comm is None or self.comm.Get_rank() == 0:
            with open(self.filename_space, 'w') as f:
                yaml.dump(data=self._space_info, stream=f, default_flow_style=None, sort_keys=False)


class PostProcessManager:
    """
    A class to read saved information of a previous simulation
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

    comm : mpi4py.MPI.Intracomm or None, optional
        Communicator

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

    _domain_h : psydac.cad.Geometry
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

    def __init__(self, geometry_file=None, domain=None, space_file=None, fields_file=None, comm=None):
        if geometry_file is None and domain is None:
            raise ValueError('Domain or geometry file needed')
        if geometry_file is not None and domain is not None:
            raise ValueError("Can't provide both geometry_file and domain")
        if domain is not None:
            assert isinstance(domain, Domain)

        self.geometry_filename = geometry_file
        self._domain = domain
        self._domain_h = None

        self.space_filename = space_file
        self.fields_filename = fields_file

        self._spaces = {}
        self._static_fields = {}
        self._snapshot_fields = {}

        self._has_static = False
        
        self._last_loaded_fields = None

        self._loaded_t = None
        self._loaded_ts = None
        self._snapshot_list = None

        self.comm = comm
        self.fields_file = None

        self._multipatch = None
        self._interior_space_index = {}

        self._mappings = {}

        self._last_subdomain = None # List of interior names
        self._last_mesh_info = None # Avoid recomputing mesh in one export call

        self._ncells = {}

        self._pushforwards = {} # One psydac.feec.PUSHFORWARD per Patch

        

        self._reconstruct_spaces()
        self.get_snapshot_list()
        self._mpi_dd = self.get_mpi_dd()

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
        """
        Reconstructs all of the spaces from reading the files.
        """
        domain, domain_h = self._process_domain()

        if domain_h is not None:
            convert_deg_dict = _augment_space_degree_dict(domain_h.ldim)
        else:
            convert_deg_dict = None

        space_info = self.read_space_info()
        pdim = space_info['ndim']

        assert pdim == domain.dim
        assert space_info['fields'] == self.fields_filename 

        # -------------------------------------------------
        # Space reconstruction
        # -------------------------------------------------
        spaces_to_interior_names = {}
        subdomains_to_spaces = {}

        interior_names_to_breaks = {}
        interior_names_to_ncells = {}
        for patch in space_info['patches']:
            
            breaks = patch['breakpoints']
            interior_names_to_breaks[patch['name']] = breaks
            interior_names_to_ncells[patch['name']] = [len(b) - 1 for b in breaks]
            try:
                scalar_spaces = patch['scalar_spaces']
            except KeyError:
                scalar_spaces = {}
            try:
                vector_spaces = patch['vector_spaces']
            except KeyError:
                vector_spaces = {}

            already_used_names = []

            if patch['name'] == domain.name: # Means single patch and thus conforming mesh
                assert len(space_info['patches']) == 1 # Sanity Check

            try:
                temp_ldim = domain.get_subdomain(patch['name']).mapping._ldim
            except AttributeError:
                temp_ldim = domain.get_subdomain(patch['name']).dim

            convert_deg_dict = _augment_space_degree_dict(temp_ldim)

            # Vector Spaces
            for v_sp in vector_spaces:
                components = v_sp['components']

                basis = []
                for sc_sp in components:
                    already_used_names.append(sc_sp['name'])
                    basis += sc_sp['basis']

                basis = list(set(basis))
                if len(basis) != 1:
                    basis = 'M'
                else:
                    basis = basis[0]

                degree = [sc_sp['degree'] for sc_sp in components]
                multiplicity = [sc_sp['multiplicity'] for sc_sp in components]
                
                # Need to get the degree and multiplicity that correspond to the H1 space v_sp was derived
                new_degree, new_mul = convert_deg_dict[v_sp['kind']](degree, multiplicity)
                
                knots = [[np.asarray(sc_sp['knots'][i]) for i in range(sc_sp['ldim'])] for sc_sp in components][0]
                periodic = components[0]['periodic']

                for i in range(components[0]['ldim']):
                    if new_degree[i] != degree[0][i]:
                        for j in range(new_degree[i] - degree[0][i]):
                            knots[i] = elevate_knots(knots[i], degree[0][i], periodic=periodic[i])

                temp_kwargs_discretization = {
                    'degree':[int(new_degree[i]) for i in range(components[0]['ldim'])],
                    'knots': knots,
                    'basis': basis,
                    'periodic':periodic,
                }

                self._space_reconstruct_helper(
                    space_name=v_sp['name'],
                    patch_name=patch['name'],
                    is_vector=True,
                    kind=v_sp['kind'],
                    discrete_kwarg=temp_kwargs_discretization,
                    interior_dict=subdomains_to_spaces,
                    space_name_dict=spaces_to_interior_names
                )  

            # Scalar Spaces
            for sc_sp in scalar_spaces:
                # Check that it's not a component of a vector_space
                if sc_sp['name'] not in already_used_names:

                    basis = list(set(sc_sp['basis']))
                    if len(basis) != 1:
                        basis = 'M'
                    else:
                        basis = basis[0]

                    multiplicity = sc_sp['multiplicity']
                    degree = sc_sp['degree']
                    
                    # Need to get the degree and multiplicity that correspond to the H1 space from which sc_sp was derived
                    new_degree, new_mul = convert_deg_dict[sc_sp['kind']](degree, multiplicity)
                    
                    knots = [np.asarray(sc_sp['knots'][i]) for i in range(sc_sp['ldim'])]
                    periodic = sc_sp['periodic']

                    for i in range(sc_sp['ldim']):
                        if new_degree[i] != degree[i]:
                            for j in range(new_degree[i] - degree[i]):
                                knots[i] = elevate_knots(knots[i], degree[i], periodic=periodic[i])                                    
                    
                    temp_kwargs_discretization = {
                        'degree': [int(new_degree[i]) for i in range(sc_sp['ldim'])],
                        'knots': knots,
                        'basis': basis,
                        'periodic': periodic,
                    }

                    self._space_reconstruct_helper(
                        space_name=sc_sp['name'],
                        patch_name=patch['name'],
                        is_vector=False,
                        kind=sc_sp['kind'],
                        discrete_kwarg=temp_kwargs_discretization,
                        interior_dict=subdomains_to_spaces,
                        space_name_dict=spaces_to_interior_names
                    )

        self._ncells = interior_names_to_ncells

        for subdomain_names, space_dict in subdomains_to_spaces.items():
            ncells_dict = {interior_name: interior_names_to_ncells[interior_name] for interior_name in subdomain_names}
            # No need for a a dict until PR about non-conforming meshes is merged
            # Check for conformity
            ncells =  list(ncells_dict.values())[0]
            assert all(ncells_patch == ncells for ncells_patch in ncells_dict.values())

            subdomain = domain.get_subdomain(subdomain_names)
            if subdomain is domain:
                if domain_h is None:
                    domain_h = discretize(domain, ncells=ncells, comm=self.comm)
                    self._domain_h = domain_h
                subdomain_h = domain_h
            else:
                subdomain_h = discretize(subdomain, ncells=ncells, comm=self.comm)
            
            for space_name, (is_vector, kind, discrete_kwargs) in space_dict.items():
                if is_vector:
                    temp_symbolic_space = VectorFunctionSpace(space_name, subdomain, kind)
                else:
                    temp_symbolic_space = ScalarFunctionSpace(space_name, subdomain, kind)
                
                # Until PR #213 is merged knots as to be set to None
                discrete_kwargs['knots'] = None if len(discrete_kwargs['knots']) !=1 else discrete_kwargs['knots'][subdomain.name]

                space_h = discretize(temp_symbolic_space, subdomain_h, comm=self.comm, **discrete_kwargs)
                self._spaces[space_name] = space_h

                # Document which index in space_h.space corresponds to which interior
                if len(subdomain_names) == 1:
                    self._interior_space_index[space_name] = {subdomain_names[0]: -1}
                else:
                    self._interior_space_index[space_name] = {subdomain.interior_names[i]: i for i in range(len(subdomain_names))}
                self._interior_space_index[space_h] = self._interior_space_index[space_name]

                # Check breakpoints
                if not is_vector and len(subdomain_names) > 1:
                    for interior_name in subdomain_names:
                        assert all(np.allclose(space_h.spaces[i].breaks, interior_names_to_breaks[interior_name]) 
                                    for i in range(len(subdomain_names)) 
                                    if space_h.symbolic_space.domain.interior.as_tuple()[i].name == interior_name)
                elif not is_vector:
                    assert np.allclose(space_h.breaks, interior_names_to_breaks[subdomain_names[0]])

                elif len(subdomain_names) > 1:
                    for interior_name in subdomain_names:
                        assert all(np.allclose(space_h.spaces[i].spaces[0].breaks, interior_names_to_breaks[interior_name]) 
                                    for i in range(len(subdomain_names)) 
                                    if space_h.symbolic_space.domain.interior.as_tuple()[i].name == interior_name)
                else:
                    assert np.allclose(space_h.spaces[0].breaks, interior_names_to_breaks[subdomain_names[0]])

    def _process_domain(self):
        if not self.geometry_filename is None:
            domain = Domain.from_file(self.geometry_filename)
            domain_h = discretize(domain, filename=self.geometry_filename, comm=self.comm)
            
            if not domain_h.mappings is None:
                self._mappings.update(domain_h.mappings)
            if isinstance(domain.interior, InteriorDomain):
                if not domain.mapping is None:
                    self._mappings[domain.name] = domain.mapping
            else:
                if isinstance(domain.mapping, MultiPatchMapping):
                    for interior in domain.interior.as_tuple():
                        self._mappings[interior.name] = domain.mapping.mappings[interior.logical_domain]
                else:
                    for interior in domain.interior.as_tuple():
                        if not self._mappings.get(interior.name, None) is None:
                            self._mappings[interior.name] = interior.mapping

        else:
            domain = self._domain
            if isinstance(domain.interior, InteriorDomain):
                if not domain.mapping is None:
                    self._mappings[domain.name] = domain.mapping
            else:
                if isinstance(domain.mapping, MultiPatchMapping):
                    for interior in domain.interior.as_tuple():
                        self._mappings[interior.name] = domain.mapping.mappings[interior.logical_domain]
                else:
                    for interior in domain.interior.as_tuple():
                            self._mappings[interior.name] = interior.mapping
            domain_h = None

        self._domain = domain
        self._domain_h = domain_h

        return domain, domain_h


    def _space_reconstruct_helper(self,
        *, 
        space_name=None,
        patch_name=None,
        is_vector=True,
        kind='h1',
        discrete_kwarg=None,
        interior_dict=None,
        space_name_dict=None):
        """
        Take cares of adequately filling the dictionnaries to avoid code repetition
        in self._reconstruct_spaces.
        """
        # Check if the discretization kwargs were already retrieved
        try:
            # Get previous partial domain of sc_sp
            patches_where_present = space_name_dict[space_name]

            # remove entry
            _, __, _kwarg = interior_dict[patches_where_present].pop(space_name)

            # Update partial domain
            new_patches_where_present = patches_where_present + (patch_name,)
            space_name_dict[space_name] = new_patches_where_present

            # Check consistency (might need to be changed to allow non conforming meshes)
            assert _kwarg['degree'] == discrete_kwarg['degree']
            assert _kwarg['basis'] == discrete_kwarg['basis']
            assert _kwarg['periodic'] == discrete_kwarg['periodic']

            _kwarg['knots'][patch_name] = discrete_kwarg['knots'] 
            try:
                interior_dict[new_patches_where_present][space_name] = (is_vector, kind, _kwarg)
            except KeyError:
                interior_dict[new_patches_where_present] = {space_name: (is_vector, kind, _kwarg)}

        except KeyError:
            # This is the first patch where we encouter this space so we need to fill the dictionnaries
            space_name_dict[space_name] = (patch_name,)
            # (False for scalar, kind, kwargs for discretization)
            knots = discrete_kwarg['knots']
            discrete_kwarg['knots'] = {patch_name: knots}
            try: 
                interior_dict[(patch_name,)][space_name] = (is_vector, kind, discrete_kwarg)
            except KeyError:
                interior_dict[(patch_name,)] = {space_name: (is_vector, kind, discrete_kwarg)}


    def get_snapshot_list(self):
        kwargs = {}
        if self.comm is not None and self.comm.size > 1:
            kwargs.update(driver='mpio', comm=self.comm)
        fh5 = h5.File(self.fields_filename, mode='r', **kwargs)
        self._snapshot_list = []
        for k in fh5.keys():
            if k != 'static' and k!= 'mpi_dd':
                self._snapshot_list.append(int(k[-4:]))
            elif k == 'static':
                self._has_static = True
        self.fields_file = fh5

    def _get_mpi_dd(self):
        fh5 = self.fields_file
        if 'mpi_dd' in fh5.keys():
            return {k: v for k, v in fh5['mpi_dd'].items()}
        else:
            return None

    def close(self):
        if not self.fields_file is None:
            self.fields_file.close()
            self.fields_file = None
        
    def load_static(self, *fields):
        """Reads static fields from file.
        
        Parameters
        ----------
        *fields : tuple of str
            Names of the fields to load
        """
        if self._snapshot_list is None:
            self.get_snapshot_list()

        fh5 = self.fields_file

        static_group = fh5['static']
        
        self._import_fields_helper(
            static_group, 
            self._static_fields, 
            fields
        )
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
        if self._snapshot_list is None:
            self.get_snapshot_list()
        fh5 = self.fields_file

        snapshot_group = fh5[f'snapshot_{n:0>4}']

        keys_loaded = self._import_fields_helper(
            snapshot_group, 
            self._snapshot_fields, 
            fields
        )

        self._loaded_t = snapshot_group.attrs['t']
        self._loaded_ts = snapshot_group.attrs['ts']

        for v in self._snapshot_fields.values():
            v.coeffs.update_ghost_regions()

        self._snapshot_fields = {k: v for k, v in self._snapshot_fields.items() if k in keys_loaded}

        self._last_loaded_fields = self._snapshot_fields

    def _import_fields_helper(self, hdf5_group, container, keys):
        """
        Helper function that parse through an hdf5 group
        to load the fields whose name appears in ``keys``
        """
        keys_loaded = []
        for patch_name, patch_group in hdf5_group.items():
            temp_space_to_field = {}
            for space_name, space_group in patch_group.items():

                if 'parent_space' not in space_group.attrs.keys(): # Scalar case
                    for field_dset_name, field_dset in space_group.items():
                        if field_dset_name in keys:
                            keys_loaded.append(field_dset_name)
                            self._load_fields_helper(
                                patch_name,
                                space_name,
                                field_dset_name,
                                field_dset,
                                container
                            )

                else: # Vector case
                    relevant_space_name = space_group.attrs['parent_space']

                    for field_dset_key in space_group.keys():
                        field_dset = space_group[field_dset_key]
                        relevant_field_name = field_dset.attrs['parent_field']

                        if relevant_field_name in keys:
                            keys_loaded.append(relevant_field_name)
                            coeff = field_dset

                            # Exceptions to take care of when the dicts are empty
                            try:
                                temp_space_to_field[relevant_space_name][relevant_field_name].append(coeff)
                            except KeyError:
                                try:
                                    temp_space_to_field[relevant_space_name][relevant_field_name] = [coeff]
                                except KeyError:
                                    temp_space_to_field[relevant_space_name] = {relevant_field_name:
                                                                                    [coeff]
                                                                                    }
            for space_name, field_dict in temp_space_to_field.items():
                for field_name, list_coeff in field_dict.items():
                    self._load_fields_helper(
                        patch_name,
                        space_name,
                        field_name,
                        list_coeff,
                        container
                    )
        return keys_loaded

    def _load_fields_helper(self, patch_name, space_name, field_name, coeff, container):
        """
        Creates or loads a scalar or vector field field_name linked to the space 
        space_name on patch name using coeff.
        """
        try:
            field = container[field_name]
        except KeyError:
            field = FemField(self._spaces[space_name])
            container[field_name] = field
        space = self._spaces[space_name]
        index = self._interior_space_index[space_name][patch_name]
        if index != -1:
            space = space.spaces[index]
            field = field.fields[index]
        else:
            space = space
            field = field
        
        if isinstance(coeff, list): # Means vector field
            for i in range(len(coeff)):
                V = space.spaces[i].vector_space
                index_coeff = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
                field.coeffs[i][index_coeff] = coeff[i][index_coeff]
        else:
            V = space.vector_space
            index_coeff = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
            field.coeffs[index_coeff] = coeff[index_coeff]

    def export_to_vtk(self,
                      filename,
                      *,
                      grid=None,
                      npts_per_cell=None,
                      snapshots='none',
                      lz=4,
                      fields=None,
                      additional_logical_functions=None,
                      additional_physical_functions=None,
                      number_by_rank_simu=True,
                      number_by_rank_visu=True,
                      number_by_patch=True,
                      debug=False,
                      ):
        """
        Exports some fields to vtk. 

        Parameters
        ----------
        filename_pattern : str
            file pattern of the file

        grid : List of ndarray
            Grid on which to evaluate the fields

        npts_per_cell : int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        snapshots : list of int or 'all' or 'none', default='none'
            If a list is given, it will export every snapshot present in the list.
            If 'none', only the static fields will be exported.
            Finally, if 'all', will export every time step and the static part.
        
        lz : int, default=4
            Number of leading zeros in the time indexing of the files. 
            Only used if ``snapshot`` is not ``'none'``. 

        fields : tuple
            Names of the fields to export.
        
        additional_physical_functions : dict
            Dictionary of callable functions. Those functions will be called on (x_mesh, y_mesh, z_mesh)

        additional_logical_functions : dict
            Dictionary of callable functions. Those functions will be called on the grid.
        
        number_by_rank : bool, default=True
            Adds a cellData attribute that represents the rank of the process that created the file. 

        debug : bool, default=False
            If true, returns ``(mesh, pointData_list)`` where ``mesh`` is ``(x_mesh, y_mesh,  z_mesh)``
            and ``pointData_list`` is the list of all the pointData dictionaries.

        Notes
        -----
        This function only supports regular and irregular tensor grid.
        L2 and Hdiv push-forward algorithms use the metric determinant and
        not the jacobian determinant. For this reason, sign descrepancies can
        happen when comparing against algorithms which use the latter.
        """
        # Immediatly fail if grid and npts_per_cell are None
        if grid is None and npts_per_cell is None:
            raise ValueError("At least one of 'grid' or 'npts_per_cell' must be provided")
        # Check grid
        if not isinstance(grid, dict):
            grid = {i_name: grid for i_name in self._domain.interior_names}
        
        # Check npts_per_cell
        if not isinstance(npts_per_cell, dict):
            npts_per_cell = {i_name: npts_per_cell for i_name in self._domain.interior_names}
        
        # Check if parallel
        if self.comm is not None:
            rank = self.comm.Get_rank()
            size = self.comm.Get_size()
            # Get ranked filename
            if size > 1:
                filename = filename + f'.{rank}'
            else:
                number_by_rank_visu = False
        else:
            number_by_rank_visu = False
        
        # Check if simu was parallel
        if self._mpi_dd is None:
            number_by_rank_simu = False 
        
        # Check fields
        if fields is None:
            fields = ()
        if isinstance(fields, str):
            fields = (fields,)
        # Check additional_logical_functions
        if additional_logical_functions is None:
            additional_logical_functions = {}

        # Check additional_physical_functions
        if additional_physical_functions is None:
            additional_physical_functions = {}

        # Delete temporary values
        self._last_mesh_info = None
        self._last_subdomain = None
        self._pushforwards = {}

        # Initialize debug mode
        if debug:
            debug_result = {'mesh_info' : [], 'pointData': [], 'cellData': [], 'Exception': []}
        # -----------------------------------------------
        # Static 
        # -----------------------------------------------
        try:
            if (snapshots == 'all' and self._has_static) or snapshots == 'none':
                # Load fields
                self.load_static(*fields)
                # Compute everything
                mesh_info, cell_data, point_data = self._export_to_vtk_helper(
                        grid=grid,
                        npts_per_cell=npts_per_cell,
                        fields=fields,
                        additional_logical_functions=additional_logical_functions,
                        additional_physical_functions=additional_physical_functions,
                        number_by_patch=number_by_patch,
                        number_by_rank_simu=number_by_rank_simu,
                )
                if number_by_rank_visu:
                    cell_data['MPI_RANK'] = np.full_like(mesh_info[1][1], rank)

                # Write .VTU file
                unstructuredGridToVTK(filename+'.static', 
                                    *mesh_info[0], 
                                    connectivity=mesh_info[1][0],
                                    offsets=mesh_info[1][1],
                                    cell_types=mesh_info[1][2],
                                    cellData=cell_data,
                                    pointData=point_data)
                
                # If parallel, Rank 0 writes the .PVTU file
                if self.comm is not None and size > 1 and rank == 0:
                    # Get the dtypes and number of components
                    celldata_info, pointdata_info = self._compute_parallel_info(cell_data, point_data)
                    # Write .PVTU file
                    writeParallelVTKUnstructuredGrid(
                        path=filename[:-2], # Remove ".0"
                        coordsdtype=mesh_info[0][0].dtype,
                        sources=[filename[:-1]+f'{r}.static.vtu' for r in range(size)],
                        ghostlevel=0,
                        cellData=celldata_info,
                        pointData=pointdata_info
                    )
                # Save results for debug mode
                if debug:
                    debug_result['mesh_info'].append(mesh_info)
                    debug_result['pointData'].append(point_data)
                    debug_result['cellData'].append(cell_data)
        except Exception as e_static:
            if debug:
                debug_result['Exception'].append(e_static)
            else:
                raise e_static

        # -----------------------------------------------
        # Time dependent
        # -----------------------------------------------
        # Check snaphots args
        if snapshots == 'all':
            snapshots = self._snapshot_list
        elif snapshots == 'none':
            snapshots = []
        elif isinstance(snapshots, int):
            assert snapshots in self._snapshot_list
            snapshots = [snapshots]
        elif isinstance(snapshots, list):
            assert all(s in self._snapshot_list for s in snapshots)
        # Iterate on the snapshots to avoid memory consumption
        for i, snapshot in enumerate(snapshots):
            try:
            # Load fields
                self.load_snapshot(snapshot, *fields)
                
                # Compute everything
                mesh_info, cell_data, point_data = self._export_to_vtk_helper(
                        grid=grid,
                        npts_per_cell=npts_per_cell,
                        fields=fields,
                        additional_logical_functions=additional_logical_functions,
                        additional_physical_functions=additional_physical_functions,
                        number_by_patch=number_by_patch,
                        number_by_rank_simu=number_by_rank_simu,
                )
                if number_by_rank_visu:
                    cell_data['MPI_RANK'] = np.full_like(mesh_info[1][1], rank)

                # Write .VTU file
                unstructuredGridToVTK(filename + '.{0:0{1}d}'.format(i, lz), 
                                    *mesh_info[0], 
                                    connectivity=mesh_info[1][0],
                                    offsets=mesh_info[1][1],
                                    cell_types=mesh_info[1][2],
                                    cellData=cell_data,
                                    pointData=point_data)

                # If parallel, Rank 0 writes the .PVTU file
                if self.comm is not None and size > 1 and rank == 0:
                    # Get dtypes and number of components
                    celldata_info, pointdata_info = self._compute_parallel_info(cell_data, point_data)
                    # Write .PVTU file
                    writeParallelVTKUnstructuredGrid(
                        path=filename[:-2], # Remove ".0"
                        coordsdtype=mesh_info[0][0].dtype,
                        sources=[filename[:-1]+f'{r}' + '.{0:0{1}d}.vtu'.format(i, lz) for r in range(size)],
                        ghostlevel=0,
                        cellData=celldata_info,
                        pointData=pointdata_info
                    )

                # Save results for debug mode
                if debug:
                    debug_result['mesh_info'].append(mesh_info)
                    debug_result['pointData'].append(point_data)
                    debug_result['cellData'].append(cell_data)
            except Exception as e_i:
                if debug:
                    debug_result['Exception'].append((snapshot, e_i))
                else:
                    raise e_i

        # Delete temporary values
        self._last_mesh_info = None
        self._last_subdomain = None
        self._pushforwards = {}

        # Return debug results
        if debug:
            list_exception = debug_result.get('Exception', [])
            for exception in list_exception:
                if isinstance(exception, tuple):
                    print(
                        f"Failure on snapshot {exception[0]}: {exception[1].args}"
                    )
                else:
                    print(
                        f"Failure on static: {exception.args}"
                    )
            return debug_result

    def _export_to_vtk_helper(
        self,             
        grid=None,
        npts_per_cell=None,
        fields=None,
        additional_logical_functions=None,
        additional_physical_functions=None,
        number_by_patch=True,
        number_by_rank_simu=True):
        """
        Helper function to avoid code repetition.
        This function evaluates and pushforward fields
        on a single patch. The correct fields are assumed to be loaded.
        This function is MPI-unaware.

        This function should not be used directly by the user.

        Parameters
        ----------            
        """
        ldim = self._domain_h.ldim
    
        cell_data = {}
        point_data = {}

        # Not all fields are present in the currently loaded fields
        fields_relevant = tuple(k for k in fields if k in self._last_loaded_fields.keys())

        # Mesh
        needs_mesh = True
        full_mesh = [np.array([])] * ldim
        full_connectivity = np.array([], dtype='i')
        full_offsets = np.array([], dtype='i')
        full_types = np.array([], dtype='i')
        offset = 0

        patch_numbers = np.array([], dtype='i')
        # Get smallest subdomain that contains all fields
        subdomain, interior_to_dict_fields = self._smallest_subdomain()
        if subdomain is self._last_subdomain and not self._last_mesh_info is None:
            
            mesh_info, number_by_patch = self._last_mesh_info 
            needs_mesh = False         
            if number_by_patch:
                cell_data.update(number_by_patch)
                number_by_patch = False
        # No fields -> only build the mesh
        if fields == ():
            interior_to_dict_fields = {
                i_name: {} for i_name in self._domain.interior_names
            }

        for i_patch, (interior_name, space_dict) in enumerate(interior_to_dict_fields.items()):
            mapping = self._mappings[interior_name]
            assert isinstance(mapping, (Mapping, SplineMapping)) or mapping is None

            i_mesh_info, i_point_data = self._compute_single_patch(
                interior_name=interior_name,
                mapping=mapping,
                space_dict=space_dict,
                grid=grid[interior_name],
                npts_per_cell=npts_per_cell[interior_name],
                additional_logical_functions=additional_logical_functions,
                needs_mesh=needs_mesh,
            )

            if needs_mesh:
                i_mesh, i_con, i_off, i_typ = i_mesh_info
                for i in range(ldim):
                    full_mesh[i] = np.concatenate([full_mesh[i], np.ravel(i_mesh[i], 'F')])
                
                patch_numbers = np.concatenate([patch_numbers, np.full_like(i_off, i_patch)])
                    
                full_offsets = np.concatenate([full_offsets,  i_off + full_connectivity.size])
                full_connectivity = np.concatenate([full_connectivity, i_con + offset])
                full_types = np.concatenate([full_types, i_typ])

                offset += i_mesh[0].size

            # Fields
            for name_intra in fields_relevant:
                i_data = i_point_data.get(name_intra, None)
                if isinstance(i_data, tuple):
                    try:
                        assert isinstance(point_data[name_intra], list)
                        for i_dir in range(len(i_data)):
                            point_data[name_intra][i_dir] = np.concatenate([point_data[name_intra][i_dir], 
                                                                        np.ravel(i_data[i_dir], 'F')])
                    except KeyError:
                        point_data[name_intra] = [np.ravel(i_data_i, 'F') for i_data_i in i_data]
                    except AssertionError: # Wrongfully assumed scalar in previous patches
                        point_data[name_intra] = [np.concatenate(point_data[name_intra], np.ravel(i_data_i)) for i_data_i in i_data]
                elif isinstance(i_data, np.ndarray):
                    try:
                        point_data[name_intra] = np.concatenate([point_data[name_intra], np.ravel(i_data, 'F')])
                    except KeyError:
                        point_data[name_intra] = np.ravel(i_data, 'F')
            
                else: # Not all of the fields are present in all patches
                    ref = list(i_point_data.values())[0]
                    if isinstance(ref, tuple):
                        ref = ref[0]
                    try:
                        if isinstance(point_data[name_intra], list):
                            for i_dir in range(len(point_data[name_intra])):
                                point_data[name_intra][i_dir] = np.concatenate([point_data[name_intra][i_dir],
                                                                               np.full(np.prod(ref.shape), np.nan)])
                        else:
                            point_data[name_intra] = np.concatenate([point_data[name_intra], np.full(np.prod(ref.shape), np.nan)])
                    except KeyError:
                        point_data[name_intra] = np.full(np.prod(ref.shape, np.nan))

            # Logical functions
            for name in additional_logical_functions.keys():
                i_data = i_point_data.get(name)
                if isinstance(i_data, tuple):
                    try:
                        for i_dir in range(len(i_data)):
                            point_data[name][i_dir] = np.concatenate([point_data[name][i_dir],
                                                                      np.ravel(i_data[i_dir], 'F')])
                    except KeyError:
                        point_data[name] = [np.ravel(i_data[i_dir], 'F') for i_dir in range(len(i_data))]
                else:
                    try:
                        point_data[name] = np.concatenate([point_data[name],
                                                           np.ravel(i_data, 'F')])
                    except KeyError:
                        point_data[name] = np.ravel(i_data, 'F')

        if number_by_patch:
            cell_data['patch'] = patch_numbers
        
        if needs_mesh:
            mesh_info = full_mesh, (full_connectivity, full_offsets, full_types)

        # physical functions
        for name, lambda_f in additional_physical_functions.items():
            f_result = lambda_f(*mesh_info[0])
            if isinstance(f_result, np.ndarray):
                point_data[name] = np.ravel(f_result, 'F')
            elif isinstance(f_result, tuple):
                point_data[name] = tuple(np.ravel(i_result, 'F') for i_result in f_result)

        self._last_mesh_info = mesh_info, {'patch': patch_numbers}
        self._last_subdomain = subdomain

        # Everything needs to be 3D
        # Arrays are flattened so we don't need to change them.
        # We still need to add a third component to vectors.
        if ldim == 2:
            full_mesh.append(np.zeros_like(full_mesh[0]))
            for name, data in point_data.items():
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    point_data[name] = tuple(data) +(np.zeros_like(data[0]),)
            for name, data in cell_data.items():
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    cell_data[name] = tuple(data) + (np.zeros_like(data[0]),)
            
        else:
            for name, data in point_data.items():
                if isinstance(data, list):
                    point_data[name] = tuple(data)
    
            for name, data in cell_data.items():
                if isinstance(data, list):
                    cell_data[name] = tuple(data)
        
        return mesh_info, cell_data, point_data

    def _compute_parallel_info(self, cell_data, point_data):
        """
        List the dtypes and number of components of 
        the arrays in cell_data and point_data.
        """
        pointData_info = {}
        cellData_info = {}
        for name, data in point_data.items():
            if isinstance(data, tuple):
                pointData_info[name] = (data[0].dtype, 3)
            else:
                pointData_info[name] = (data.dtype, 1)
        for name, data in cell_data.items():
            if isinstance(data, tuple):
                cellData_info[name] = (data[0].dtype, 3)
            else:
                cellData_info[name] = (data.dtype, 1)
        
        return cellData_info, pointData_info
    
    def _smallest_subdomain(self):
        """
        Return the smallest subdomain of self._domain
        that contains all of the loaded fields and

        """
        interior_to_fields = {}
        for f_name, f in self._last_loaded_fields.items():
            space_f = f.space
            interior_index_dict = self._interior_space_index[space_f]

            for interior, i in interior_index_dict.items():
                if i != -1:
                    try:
                        interior_to_fields[interior][space_f.spaces[i]][0].append(f_name)
                        interior_to_fields[interior][space_f.spaces[i]][1].append(f.fields[i])
                    except KeyError:
                        try:
                            interior_to_fields[interior][space_f.spaces[i]] = ([f_name], [f.fields[i]])
                        except KeyError:
                            interior_to_fields[interior] = {space_f.spaces[i]: ([f_name], [f.fields[i]])}
                else:
                    try:
                        interior_to_fields[interior][space_f][0].append(f_name) 
                        interior_to_fields[interior][space_f][1].append(f)
                    except KeyError:
                        try:
                            interior_to_fields[interior][space_f] = ([f_name], [f])
                        except KeyError:
                            interior_to_fields[interior] = {space_f: ([f_name], [f])}
        try:
            subdomain = self._domain.get_subdomain(tuple(interior_to_fields.keys()))
        except IndexError:
            subdomain = None
        except TypeError:
            subdomain = None
        except UnboundLocalError:
            subdomain = None

        return subdomain, interior_to_fields

    def _compute_single_patch(
        self,
        interior_name=None,
        mapping=None,
        space_dict=None,
        grid=None,
        npts_per_cell=None,
        additional_logical_functions=None,
        needs_mesh=True,
        ):
        """
        Evaluates and pushes forward all relevant quantities
        """
        # Shortcut
        ldim = self._domain_h.ldim
        # Get local_info -> local and global_domain + breaks
        local_domain, global_domain, breaks = self._get_local_info(interior_name, mapping, space_dict)
            
        # npts_per_cell
        if isinstance(npts_per_cell, int):
            npts_per_cell = [npts_per_cell] * ldim
        
        # grid
        if grid is None:
            if npts_per_cell is None:
                raise ValueError("At least one of grid or npts_per_cell must be provided")

            grid = [
                np.array(
                refine_array_1d(breaks[i], npts_per_cell[i] - 1, False)
                ) 
                for i in range(len(breaks))
                ]
            grid_local= [grid[i][
                local_domain[0][i] * npts_per_cell[i]:(local_domain[1][i] + 1) * npts_per_cell[i]]
                        for i in range(ldim)]
            grid_as_arrays = [np.reshape(grid[i], (len(grid[i])//npts_per_cell[i], npts_per_cell[i])) 
                        for i in range(ldim)]

            cell_indexes = None
            grid_type = 1
        
        else:
            grid_as_arrays = [np.asarray(grid[i]) for i in range(len(grid))]
            assert all(grid_as_arrays[i].ndim == grid_as_arrays[i+1].ndim for i in range(len(grid) - 1))
            
            # Regular tensor grid
            if grid_as_arrays[0].ndim == 1 and npts_per_cell is not None:
                grid_type = 1
                # Check that the grid is regular
                assert all(grid_as_arrays[i].size % npts_per_cell[i] == 0 for i in range(ldim))
                grid_as_arrays = [np.reshape(grid[i], (len(grid[i])//npts_per_cell[i], npts_per_cell[i])) 
                        for i in range(ldim)]
                grid_local = []
                for i in range(len(grid_as_arrays)):
                    grid_local.append(grid_as_arrays[i][local_domain[0][i] * npts_per_cell[i]:
                                                    (local_domain[1][i] + 1) * npts_per_cell[i]])
                cell_indexes = None

            # Irregular tensor grid
            elif grid_as_arrays[0].ndim == 1 and npts_per_cell is None:
                grid_type = 0
                cell_indexes = [cell_index(breaks[i], grid_as_arrays[i]) for i in range(ldim)]

                grid_local = []
                for i in range(len(grid)):
                    i_start = np.searchsorted(cell_indexes[i], local_domain[0][i], side='left')
                    i_end = np.searchsorted(cell_indexes[i], local_domain[1][i], side='right')
                    grid_local.append(grid_as_arrays[i][i_start:i_end])

            # Unstructured grid
            elif grid_as_arrays[0].ndim == ldim:
                grid_type = 2
                raise NotImplementedError("Unstructured grids are not supported yet")
            # Bad inputs
            else:
                raise ValueError("Wrong input for the grid parameters")

        if needs_mesh:
            partial_mesh_info = self._get_mesh(
                mapping, 
                grid_local, 
                local_domain, 
                npts_per_cell=npts_per_cell,
                cell_indexes=cell_indexes
            )
        else:
            partial_mesh_info = None

        point_data = {}

        pushforward = self._pushforwards.get(interior_name, None)
        if pushforward is None and space_dict != {}:
            pushforward = Pushforward(
                grid_as_arrays, 
                mapping=mapping, 
                npts_per_cell=npts_per_cell,
                cell_indexes=cell_indexes,
                grid_type=grid_type,
                local_domain=local_domain, 
                global_domain=global_domain,
                grid_local=grid_local,
                skip_grid_check=True,
                )
            self._pushforwards[interior_name] = pushforward
        
        for space, (field_names, field_list) in space_dict.items():
            list_pushed_fields = pushforward._dispatch_pushforward(space, *field_list)
            for i, field_name in enumerate(field_names):
                point_data[field_name] = list_pushed_fields[i]

        for name, lambda_f in additional_logical_functions.items():
            f_result = lambda_f(*grid_local)
            point_data[name] = f_result
        return partial_mesh_info, point_data

    def _get_local_info(self, interior_name, mapping, space_dict):
        """
        Returns a local and a global domain and breakpoints
        """
        # Shortcut
        ldim = self._domain_h.ldim
        # Option 1 : mapping is a Spline Mapping -> Use its FemSpace
        if isinstance(mapping, SplineMapping):
            local_domain = mapping.space.local_domain
            global_domain = ((0,) * ldim, tuple(nc_i - 1 for nc_i in list(mapping.space.ncells)))
            breaks = mapping.space.breaks
        # Option 2 : space_dict is not empty -> use the first space encountered there
        elif space_dict != {}:
            space = list(space_dict.keys())[0]
            try:
                local_domain = space.local_domain
                global_domain = ((0,) * ldim, tuple(nc_i - 1 for nc_i in list(space.ncells)))
                breaks = space.breaks
            except AttributeError:
                local_domain = space.spaces[0].local_domain
                global_domain = ((0,) * ldim, tuple(nc_i - 1 for nc_i in list(space.spaces[0].ncells)))
                breaks = space.spaces[0].breaks
        # Option 3 : discretize a ScalarSpace on the patch
        else:
            temp_domain = self._domain.get_subdomain(interior_name)
            temp_sc_space = ScalarFunctionSpace('Space', temp_domain, 'h1')
            temp_domain_h = discretize(temp_domain, ncells=self._ncells[interior_name], comm=self.comm)
            # Degree doesn't matter because there aren't any other spaces
            space = discretize(temp_sc_space, temp_domain_h, degree=[2, 2], comm=self.comm)
            local_domain = space.local_domain
            global_domain = ((0,) * ldim, tuple(nc_i - 1 for nc_i in list(space.ncells)))
            breaks = space.breaks

        return local_domain, global_domain, breaks

    def _get_mesh(self, mapping, grid_local, local_domain, npts_per_cell=None, cell_indexes=None):
        """
        Return the mesh and its informations.

        Parameters
        ----------
        """
        if isinstance(mapping, SplineMapping):
            mesh = mapping.build_mesh(grid_local, npts_per_cell=npts_per_cell)
        else:
            if grid_local[0].ndim == 1:
                mesh = np.meshgrid(*grid_local, indexing='ij')
            else:
                mesh = grid_local
            if isinstance(mapping, Mapping):
                c_m = mapping.get_callable_mapping()
                mesh = c_m(*mesh)
        conn, off, typ, _ = self._compute_unstructured_mesh_info(
            local_domain, 
            npts_per_cell=npts_per_cell,
            cell_indexes=cell_indexes
        )

        return mesh, conn, off, typ

    def _compute_unstructured_mesh_info(self, mapping_local_domain, npts_per_cell=None, cell_indexes=None):
        """
        Computes the connection, offset and celltypes arrays for exportation
        as VTK unstructured grid.

        Parameters
        ----------

        mapping_local_domain : tuple of tuple
            local_domain of the mapping

        npts_per_cell : tuple of ints or ints, optional
            Number of points per cell

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
            cellshape = np.array(n_elem) * (np.array(npts_per_cell)) - 1
            total_number_cells_vtk = np.prod(cellshape)
            celltypes = np.zeros(total_number_cells_vtk, dtype='i')
            offsets = np.arange(1, total_number_cells_vtk + 1, dtype='i') * (2 ** ldim)
            connectivity = np.zeros(total_number_cells_vtk * 2 ** ldim, dtype='i')
            if ldim == 2: 
                celltypes[:] = VtkQuad.tid
                cellID = 0
                for i in range(n_elem[0] * npts_per_cell[0] - 1):
                    for j  in range(n_elem[1] * npts_per_cell[1] - 1):
                        row_top = i 
                        col_left = j

                        # VTK uses Fortran ordering
                        topleft = col_left * npts_per_cell[0] * n_elem[0] + row_top
                        topright = topleft + 1
                        botleft = topleft + n_elem[0] * npts_per_cell[0] # next column
                        botright = botleft + 1

                        connectivity[4 * cellID: 4 * cellID + 4] = [topleft, topright, botright, botleft]

                        cellID += 1
    
            elif ldim == 3:
                celltypes[:] = VtkHexahedron.tid
                cellID = 0
                n_rows = n_elem[0] * npts_per_cell[0]
                n_cols = n_elem[1] * npts_per_cell[1]
                n_layers = n_elem[2] * npts_per_cell[2]
                for i in range(n_rows - 1):
                    for j in range(n_cols - 1):
                        for k in range(n_layers - 1):

                            row_top = i
                            col_left = j 
                            layer_front = k
                            
                            # VTK uses Fortran ordering
                            top_left_front = row_top + col_left * n_rows + layer_front * n_cols * n_rows
                            top_left_back = top_left_front + 1
                            top_right_front = top_left_front + n_rows # next column
                            top_right_back = top_right_front + 1

                            bot_left_front = top_left_front + n_rows * n_cols # next layer
                            bot_left_back = bot_left_front + 1
                            bot_right_front = bot_left_front + n_rows # next column
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

            cellshape = np.array([n_points[i] - 1 for i in range(ldim)])
            total_number_cells_vtk = np.prod(cellshape)
            celltypes = np.zeros(total_number_cells_vtk, dtype='i')
            offsets = np.arange(1, total_number_cells_vtk + 1, dtype='i') * (2 ** ldim)
            connectivity = np.zeros(total_number_cells_vtk * 2 ** ldim, dtype='i')
            if ldim == 2:
                cellID = 0
                celltypes[:] = VtkQuad.tid
                for i in range(i_ends[0] - 1 - i_starts[0]):
                    for j in range(i_ends[1] - 1 - i_starts[1]):

                        row_top = i
                        col_left = j

                        # VTK uses Fortran ordering
                        topleft = row_top + col_left * n_points[0]
                        topright = topleft + 1
                        botleft = topleft + n_points[0]
                        botright = botleft +1

                        connectivity[4 * cellID: 4 * cellID + 4] = [topleft, topright, botright, botleft]
                        cellID += 1

            elif ldim == 3:
                cellID = 0
                celltypes[:] = VtkHexahedron.tid
                n_rows = n_points[0]
                n_cols = n_points[1]
                n_layers = n_points[2]
                for i in range(i_ends[0] - 1 - i_starts[0]):
                    for j in range(i_ends[1] - 1 - i_starts[1]):
                        for k in range(i_ends[2] - 1 - i_starts[2]):
                            row_top = i
                            col_left = j
                            layer_front = k

                            # VTK uses Fortran ordering
                            top_left_front = row_top + col_left * n_rows + layer_front * n_cols * n_rows
                            top_left_back = top_left_front + 1
                            top_right_front = top_left_front + n_rows # next column
                            top_right_back = top_right_front + 1

                            bot_left_front = top_left_front + n_rows * n_cols # next layer
                            bot_left_back = bot_left_front + 1
                            bot_right_front = bot_left_front + n_rows # next column
                            bot_right_back = bot_right_front + 1


                            connectivity[8 * cellID: 8 * cellID + 8] = [
                                top_left_front, top_right_front, bot_right_front, bot_left_front,
                                top_left_back, top_right_back, bot_right_back, bot_left_back
                            ]

                            cellID += 1

        else:
            raise NotImplementedError("Not Supported Yet")

        return connectivity, offsets, celltypes, cellshape


def _augment_space_degree_dict(ldim, sequence='DR'):
    """
    With the 'DR' sequence in 3D, all multiplicies are [r1, r2, r3] and we have
     'H1'   : degree = [p1, p2, p3]
     'Hcurl': degree = [[p1-1, p2, p3], [p1, p2-1, p3], [p1, p2, p3-1]]
     'Hdiv' : degree = [[p1, p2-1, p3-1], [p1-1, p2, p3-1], [p1-1, p2-1, p3]]
     'L2'   : degree = [p1-1, p2-1, p3-1]

    With the 'TH' sequence in 2D we have:
     'H1' : degree = [[p1, p2], [p1, p2]], multiplicity = [[r1, r2], [r1, r2]]
     'L2' : degree = [p1-1, p2-1], multiplicity = [r1-1, r2-1]

    With the 'RT' sequence in 2D we have:
    'H1' : degree = [[p1, p2-1], [p1-1, p2]], multiplicity = [[r1,r2], [r1,r2]]
    'L2' : degree = [p1-1, p2-1], multiplicity = [r1, r2]

    With the 'N' sequence in 2D we have:
    'H1' : degree = [[p1, p2], [p1, p2]], multiplicity = [[r1,r2+1], [r1+1,r2]]
    'L2' : degree = [p1-1, p2-1], multiplicity = [r1, r2]
    """
    assert ldim in [2, 3]
    
    if sequence == 'DR':
        def f_h1(degree, multiplicity):
            if isinstance(degree[0], (list, tuple, np.ndarray)):
                return degree[0], multiplicity
            else:
                return degree, multiplicity
    
        def f_l2(degree, multiplicity):
            if isinstance(degree[0], (list, tuple, np.ndarray)):
                return np.asarray(degree[0]) + 1, multiplicity
            else:
                return np.asarray(degree) + 1, multiplicity

        if ldim == 2:
            def f_hdiv(degree, multiplicity):
                degree = np.asarray(degree[0])
                degree += np.array([0, 1])
                return degree, multiplicity

            def f_hcurl(degree, multiplicity):
                degree = np.asarray(degree[0])
                degree += np.array([1, 0])
                return degree, multiplicity

        else:
            def f_hdiv(degree, multiplicity):
                degree = np.asarray(degree[0])
                degree += np.array([0, 1, 1])
                return degree, multiplicity

            def f_hcurl(degree, multiplicity):
                degree = np.asarray(degree[0])
                degree += np.array([1, 0, 0])
                return degree, multiplicity 

        kind_dict = {
            'h1': f_h1,
            'hcurl': f_hcurl,
            'hdiv': f_hdiv,
            'l2': f_l2,
            'undefined': f_h1, 
        }
        return kind_dict
    else:
        raise NotImplementedError("Only DR sequence is implemented")
