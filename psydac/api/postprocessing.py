# coding: utf-8
#
# Copyright 2019 Yaman Güçlü

import yaml
import re
import h5py as h5

from psydac.cad.geometry    import Geometry
from psydac.utilities.utils import refine_array_1d
from psydac.fem.basic import FemSpace

# ==============================================================================
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
        mapping = lambda eta: eta

    # TODO: make this work
    # Get 1D breakpoints in logical domain
    #eta1, eta2 = domain_h.breaks

    # NOTE: temporary solution (V_h should not be passed to this function)
    V1, V2  = V_h.spaces
    eta1 = V1.breaks
    eta2 = V2.breaks

    # Refine logical grid
    eta1_r = refine_array_1d( eta1, refine )
    eta2_r = refine_array_1d( eta2, refine )

    # Compute physical coordinates of lines with eta1=const
    isolines_1 = []
    for e1 in eta1:
        x, y = zip(*[mapping([e1, e2]) for e2 in eta2_r])
        isolines_1.append( dict(eta1=e1, x=x, y=y) )

    # Compute physical coordinates of lines with eta2=const
    isolines_2 = []
    for e2 in eta2:
        x, y = zip(*[mapping([e1, e2]) for e1 in eta1_r])
        isolines_2.append( dict(eta2=e2, x=x, y=y) )

    return isolines_1, isolines_2


# ===========================================================================
class OutputManager:
    """A class meant to streamline the exportation of
    output data.

    Parameters
    ----------
    filename_space : str
        Name of the file in which to save the space information.
    filename_fields : str
        Name of the file in which to save the fields.
    static : boolean or None, optional
        Whether to initialize the OuputManager in static mode. If None, nothing will be assumed
        and the user will have to specify whether to use the static mode when calling ``export_fields``.

    Attributes
    ----------


    """
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

    def add_spaces(self, *femspaces):
        """Add femspaces to the scope of this instance of OutputManager

        Parameters
        ----------
        femspaces: Tuple of psydac.fem.basic.FemsSpace
            Femspaces to add in the scope of this OutputManager instance.

        """
        assert all(isinstance(femspace, FemSpace) for femspace in femspaces)
        for femspace in femspaces:

            if femspace.is_product:
                self._add_vector_space(femspace)
            else:
                self._add_scalar_space(femspace)

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

        if name is None and dim is None:
            symbolic_space = scalar_space.symbolic_space
            scalar_space_name = symbolic_space.name
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

        new_space = {'name': scalar_space_name, 'ldim': ldim,
                     'kind': str(kind), 'dtype': dtype, 'rational': False,
                     'periodic': periodic,
                     'degree': degree, 'basis': basis, 'knots': knots
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

        self._spaces_info = spaces_info

        return new_space

    def _add_vector_space(self, vector_space):
        """Add a vector space to the scope of this instance of OutputManager.

        Parameters
        ----------
        vector_space: psydac.fem.vector.VectorFemSpace
            Vector FemSpace to add to the scope.
        """

        symbolic_space = vector_space.symbolic_space
        name = symbolic_space.name
        dim = symbolic_space.domain.dim
        patch_name = symbolic_space.domain.name
        kind = symbolic_space.kind

        scalar_spaces_info = []
        for i, scalar_space in enumerate(vector_space.spaces):
            sc_space_info = self._add_scalar_space(scalar_space,
                                                   name=name+f'[{i}]',
                                                   dim=dim,
                                                   patch_name=patch_name,
                                                   kind=kind)
            scalar_spaces_info.append(sc_space_info)

        spaces_info = self._spaces_info

        new_vector_space = {'name': name, 'components': scalar_spaces_info}

        patch_index = [spaces_info['patches'][i]['name']
                       for i in range(len(spaces_info['patches']))].index(patch_name)

        try:
            spaces_info['patches'][patch_index]['vector_spaces'].append(new_vector_space)
        except KeyError:
            spaces_info['patches'][patch_index].update({'vector_spaces': [new_vector_space]})

        self._spaces_info = spaces_info
        self._spaces.append(vector_space)

    def export_fields(self, **fields):
        """
        A function that exports the fields' coefficients to an HDF5 file. They are
        saved under a snapshot/patch/space/field scheme or static/patch/space/field.

        If the saving scheme is time dependent, unsets it to avoid writing twice to
        the same snapshot.

        Parameters
        ----------
        fields : dict
            List of named fields

        Raises
        ------
        ValueError
            When self.is_static is None (no saving scheme specified)
        """
        if self.is_static is None:
            raise ValueError('Saving scheme not specified')

        assert all(f.space in self._spaces for f in fields.values())
        # should we restrict to only fields belonging to the same space ?

        # For now, I assume that if something is mpi parallel everything is
        space_test = list(fields.values())[0].space
        if space_test.is_product:
            comm = space_test.spaces[0].vector_space.cart.comm if space_test.vector_space.parallel else None
        else:
            comm = space_test.vector_space.cart.comm if space_test.vector_space.parallel else None

        # Open HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = {}
        if comm is not None:
            if comm.size > 1:
                kwargs.update(driver='mpio', comm=comm)
        fh5 = h5.File(self.filename_fields, mode='a', **kwargs)

        if not 'spaces' in fh5.attrs.keys():
            fh5.attrs.create('spaces', self.filename_space)

        saving_group = self._current_hdf5_group

        # Add field coefficients as named datasets
        for name_field, field in fields.items():

            name_space = field.space.symbolic_space.name
            patch = field.space.symbolic_space.domain.name

            if field.space.is_product:  # Vector field case
                for i, field_coeff in enumerate(field.coeffs):
                    name_field_i = name_field+f'[{i}]'
                    name_space_i = name_space+f'[{i}]'
                    Vi = field.space.vector_space.spaces[i]
                    index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))

                    dset = saving_group.create_dataset(f'{patch}/{name_space_i}/{name_field_i}',
                                                       shape=Vi.npts, dtype=Vi.dtype)
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
