# coding: utf-8
#
# Copyright 2019 Yaman Güçlü

import yaml
import re
import h5py as h5

from psydac.cad.geometry    import Geometry
from psydac.utilities.utils import refine_array_1d
from psydac.fem.basic import FemSpace

#==============================================================================
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

    Results
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

class OutputManager():
    def __init__(self,filename_space,filename_fields,*spaces):

        self._spaces_info          = {}
        self._spaces               = []
        self.filename_space        = filename_space
        self.filename_fields       = filename_fields
        self._next_snapshot_number = 0
        self.add_spaces(*spaces)

    def add_spaces(self, *femspaces):
        for femspace in femspaces:
            if femspace.is_product:
                self._add_vector_space(femspace)
            else:
                self._add_scalar_space(femspace)

    def _add_scalar_space(self,scalar_space,name = None, dim = None, patch_name = None, kind = None):
        assert isinstance(scalar_space, FemSpace)
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

    def _add_vector_space(self,vector_space):

        assert isinstance(vector_space, FemSpace)



        symbolic_space = vector_space.symbolic_space
        name = symbolic_space.name
        dim = symbolic_space.domain.dim
        patch_name = symbolic_space.domain.name
        kind = symbolic_space.kind

        scalar_spaces_info = []
        for i,scalar_space in enumerate(vector_space.spaces):
            sc_space_info = self._add_scalar_space(scalar_space,
                                                   name = name+f'[{i}]',
                                                   dim = dim,
                                                   patch_name = patch_name,
                                                   kind = kind)
            scalar_spaces_info.append(sc_space_info)

        spaces_info = self._spaces_info

        new_vector_space = {'name': name, 'components': scalar_spaces_info}

        patch_index = [spaces_info['patches'][i]['name']
                       for i in range(len(spaces_info['patches']))].index(patch_name)

        try:
            spaces_info['patches'][patch_index]['vector_spaces'].append(new_vector_space)
        except:
            spaces_info['patches'][patch_index].update({'vector_spaces': [new_vector_space]})

        self._spaces_info = spaces_info
        self._spaces.append(vector_space)


    def export_fields(self,t , ts, **fields):
        '''
        A function that exports the fields' coefficients to an HDF5 file. They are
        saved under a snapshot/patch/space/field scheme. One snapshot corresponds to one call
        to this method.

        Parameters
        ----------
          t: `float`
            Time of the export
          ts: `int`
            Timestep corresponding to the export.
          fields: `dict`
            List of named fields

        '''
        assert all(f.space in self._spaces for f in fields.values())
        # should we restrict to only fields belonging to the same space ?

        # For now, I assume that if something is mpi parallel everything is
        space_test = list(fields.values())[0].space.vector_space
        comm = space_test.cart.comm if space_test.parallel else None

        # Create HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = {}
        if comm is not None:
            if comm.size > 1:
                kwargs.update(driver='mpio', comm=comm)
        fh5 = h5.File(self.filename_fields, mode='a', **kwargs)

        i = self._next_snapshot_number
        try:
            snapshot = fh5.create_group(f'snapshot_{i:0>4}')
        except:
            regexp = re.compile('snapshot_(?P<id>\d+)')
            i = max([int(regexp.search(k).group('id')) for k in fh5.keys() if regexp.search(k) is not None]) + 1
            snapshot = fh5.create_group(f'snapshot_{i:0>4}')
        snapshot.attrs.create('t', data=t, dtype=float)
        snapshot.attrs.create('ts', data=ts, dtype=int)

        self._next_snapshot_number = i + 1

        # Add field coefficients as named datasets
        for name_field, field in fields.items():
            name_space = field.space.symbolic_space.name
            patch = field.space.symbolic_space.domain.name
            if field.space.is_product:
                for i,field_coeff in enumerate(field.coeffs):
                    name_field_i = name_field+f'[{i}]'
                    name_space_i = name_space+f'[{i}]'
                    Vi = field.space.vector_space.spaces[i]
                    index = tuple(slice(s, e + 1) for s, e in zip(Vi.starts, Vi.ends))
                    dset = snapshot.create_dataset(f'{patch}/{name_space_i}/{name_field_i}', shape=Vi.npts, dtype=Vi.dtype)
                    dset[index] = field_coeff[index]
            else:
                V = field.space.vector_space
                index = tuple(slice(s, e + 1) for s, e in zip(V.starts, V.ends))
                dset = snapshot.create_dataset(f'{patch}/{name_space}/{name_field}', shape=V.npts, dtype=V.dtype)
                dset[index] = field.coeffs[index]

        # Close HDF5 file
        fh5.close()

    def export_space_info(self):
        with open(self.filename_space, 'w') as f:
            yaml.dump(self._spaces_info, f, default_flow_style = None, sort_keys = None)