# coding: utf-8
#
# a Geometry class contains the list of patches and additional information about
# the topology i.e. connectivity, boundaries
# For the moment, it is used as a container, that can be loaded from a file
# (hdf5)
from itertools import product
from collections import abc
import numpy as np
import string
import random
import h5py
import yaml
import os
import string
import random


from mpi4py import MPI

from psydac.fem.splines        import SplineSpace
from psydac.fem.tensor         import TensorFemSpace
from psydac.fem.partitioning   import create_cart, construct_connectivity, construct_interface_spaces
from psydac.mapping.discrete   import SplineMapping, NurbsMapping
from psydac.linalg.block       import BlockVectorSpace, BlockVector
from psydac.ddm.cart           import DomainDecomposition, MultiPatchDomainDecomposition

from sympde.topology       import Domain, Interface, Line, Square, Cube, NCubeInterior, Mapping
from sympde.topology.basic import Union

#==============================================================================
class Geometry( object ):
    _ldim     = None
    _pdim     = None
    _patches  = []
    _topology = None

    #--------------------------------------------------------------------------
    # Option [1]: from a (domain, mappings) or a file
    #--------------------------------------------------------------------------
    def __init__( self, domain=None, ncells=None, periodic=None, mappings=None,
                  filename=None, comm=None ):

        # ... read the geometry if the filename is given
        if not( filename is None ):
            self.read(filename, comm=comm)

        elif not( domain is None ):
            assert isinstance( domain, Domain ) 
            assert isinstance( ncells, dict)
            assert isinstance( mappings, dict)
            if periodic is not None:
                assert isinstance( periodic, dict)

            # ... check sanity
            interior_names = sorted(domain.interior_names)
            mappings_keys  = sorted(list(mappings.keys()))

            assert( interior_names == mappings_keys )
            # ...

            if periodic is None:
                periodic = {patch:[False]*len(ncells_i) for patch,ncells_i in ncells.items}

            self._domain   = domain
            self._ldim     = domain.dim
            self._pdim     = domain.dim # TODO must be given => only dim is  defined for a Domain
            self._ncells   = ncells
            self._periodic = periodic
            self._mappings = mappings
            self._cart     = None
            self._is_parallel = comm is not None

            if len(domain) == 1:
                self._ddm = DomainDecomposition(ncells[domain.name], periodic[domain.name], comm=comm)
            else:
                ncells    = [ncells[itr.name] for itr in domain.interior]
                periodic  = [periodic[itr.name] for itr in domain.interior]
                self._ddm = MultiPatchDomainDecomposition(ncells, periodic, comm=comm)

        else:
            raise ValueError('Wrong input')
        # ...

        self._comm = comm

    #--------------------------------------------------------------------------
    # Option [2]: from a discrete mapping
    #--------------------------------------------------------------------------
    @classmethod
    def from_discrete_mapping( cls, mapping, comm=None ):
        """Create a geometry from one discrete mapping."""
        if mapping.ldim in [1]:
            raise NotImplementedError('')

        if mapping.ldim == 2:
            domain = Square(name='Omega')
            mappings = {'Omega': mapping}

            return Geometry(domain=domain, mappings=mappings, comm=comm)

        elif mapping.ldim == 3:
            domain = Cube(name='Omega')
            mappings = {'Omega': mapping}

            return Geometry(domain=domain, mappings=mappings, comm=comm)

    #--------------------------------------------------------------------------
    # Option [3]: discrete topological line/square/cube
    #--------------------------------------------------------------------------
    @classmethod
    def from_topological_domain(cls, domain, ncells, *, periodic=None, comm=None):
        interior = domain.interior
        if not isinstance(interior, Union):
            interior = [interior]

        for itr in interior:
            if not isinstance(itr, NCubeInterior):
                msg = "Topological domain must be an NCube;"\
                      " got {} instead.".format(type(itr))
                raise TypeError(msg)

        mappings = {itr.name: None for itr in interior}
        if isinstance(ncells, (list, tuple)):
            ncells = {itr.name:ncells for itr in interior}

        if periodic is None:
            periodic = [False]*domain.dim

        if isinstance(periodic, (list, tuple)):
            periodic = {itr.name:periodic for itr in interior}

        geo = Geometry(domain=domain, mappings=mappings, ncells=ncells, periodic=periodic, comm=comm)

        return geo

    #--------------------------------------------------------------------------
    @property
    def ldim(self):
        return self._ldim

    @property
    def pdim(self):
        return self._pdim

    @property
    def ncells(self):
        return self._ncells

    @property
    def periodic(self):
        return self._periodic

    @property
    def comm(self):
        return self._comm

    @property
    def domain(self):
        return self._domain

    @property
    def ddm(self):
        return self._ddm

    @property
    def is_parallel(self):
        return self._is_parallel

    @property
    def mappings(self):
        return self._mappings

    def __len__(self):
        return len(self.domain)

    def read( self, filename, comm=None ):
        # ... check extension of the file
        basename, ext = os.path.splitext(filename)
        if not(ext == '.h5'):
            raise ValueError('> Only h5 files are supported')
        # ...

        # read the topological domain
        domain       = Domain.from_file(filename)
        connectivity = construct_connectivity(domain)

        if len(domain)==1:
            interiors  = [domain.interior]
        else:
            interiors  = list(domain.interior.args)

        if not(comm is None):
            kwargs = dict( driver='mpio', comm=comm ) if comm.size > 1 else {}

        else:
            kwargs = {}

        h5  = h5py.File( filename, mode='r', **kwargs )
        yml = yaml.load( h5['geometry.yml'][()], Loader=yaml.SafeLoader )

        ldim = yml['ldim']
        pdim = yml['pdim']

        n_patches = len( yml['patches'] )

        # ...
        if n_patches == 0:

            h5.close()
            raise ValueError( "Input file contains no patches." )
        # ...

        # ... read patchs
        mappings = {}
        ncells   = {}
        periodic = {}
        spaces   = [None]*n_patches
        for i_patch in range( n_patches ):

            item  = yml['patches'][i_patch]
            patch_name = item['name']
            mapping_id = item['mapping_id']
            dtype = item['type']
            patch = h5[mapping_id]
            if dtype in ['SplineMapping', 'NurbsMapping']:

                degree     = [int (p) for p in patch.attrs['degree'  ]]
                periodic_i = [bool(b) for b in patch.attrs['periodic']]
                knots      = [patch['knots_{}'.format(d)][:] for d in range( ldim )]
                space_i    = [SplineSpace( degree=p, knots=k, periodic=P )
                            for p,k,P in zip( degree, knots, periodic_i )]

                spaces[i_patch] = space_i

                ncells  [interiors[i_patch].name] = [sp.ncells for sp in space_i]
                periodic[interiors[i_patch].name] = periodic_i

        print(ncells)
        self._cart = None
        if n_patches == 1:
            self._ddm = DomainDecomposition(ncells[domain.name], periodic[domain.name], comm=comm)
            ddms      = [self._ddm]
        else:
            ncells    = [ncells[itr.name] for itr in interiors]
            periodic  = [periodic[itr.name] for itr in interiors]
            self._ddm = MultiPatchDomainDecomposition(ncells, periodic, comm=comm)
            ddms      = self._ddm.domains

        carts    = create_cart(ddms, spaces)
        g_spaces = {inter:TensorFemSpace( ddms[i], *spaces[i], cart=carts[i]) for i,inter in enumerate(interiors)}

        # ... construct interface spaces
        construct_interface_spaces(self._ddm, g_spaces, carts, interiors, connectivity)

        for i_patch in range( n_patches ):

            item  = yml['patches'][i_patch]
            patch_name = item['name']
            mapping_id = item['mapping_id']
            dtype = item['type']
            patch = h5[mapping_id]
            space_i = spaces[i_patch]
            if dtype in ['SplineMapping', 'NurbsMapping']:

                tensor_space = g_spaces[interiors[i_patch]]

                if dtype == 'SplineMapping':
                    mapping = SplineMapping.from_control_points( tensor_space,
                                                                 patch['points'][..., :pdim] )

                elif dtype == 'NurbsMapping':
                    mapping = NurbsMapping.from_control_points_weights( tensor_space,
                                                                        patch['points'][..., :pdim],
                                                                        patch['weights'] )

                mapping.set_name( item['name'] )
                mappings[patch_name] = mapping

        if n_patches>1:
            coeffs   = [[e._coeffs for e in mapping._fields] for mapping in mappings.values()]
            spaces   = [[coeffs_ij.space for coeffs_ij in coeffs_i] for coeffs_i in coeffs]
            spaces   = [BlockVectorSpace(*space) for space in spaces]
            w_spaces = [sp.spaces[0] for sp in spaces]
            space    = BlockVectorSpace(*spaces, connectivity=connectivity)
            w_space  = BlockVectorSpace(*w_spaces, connectivity=connectivity)
            v  = BlockVector(space)
            w  = BlockVector(w_space)
            mapping_list = list(mappings.values())
            for i in range(n_patches):
                for j in range(len(coeffs[i])):
                    v[i][j] = coeffs[i][j]

                mapping = mapping_list[i]
                if isinstance(mapping, NurbsMapping):
                    w[i] = mapping.weights_field.coeffs
                else:
                    w[i] = v[i][0].space.zeros()

            v.update_ghost_regions()
            w.update_ghost_regions()

        else:
            mapping = list(mappings.values())[0]
            for f in mapping._fields:
                f.coeffs.update_ghost_regions()

            if isinstance(mapping, NurbsMapping):
                mapping.weights_field.coeffs.update_ghost_regions()


        # ... close the h5 file
        h5.close()
        # ...

        # ...
        self._ldim        = ldim
        self._pdim        = pdim
        self._mappings    = mappings
        self._domain      = domain
        self._comm        = comm
        self._ncells      = ncells
        self._periodic    = periodic
        self._is_parallel = comm is not None
        # ...

    def export( self, filename ):
        """
        Parameters
        ----------
        filename : str
          Name of HDF5 output file.

        """

        # ...
        comm  = self.comm
        # ...

        # Create dictionary with geometry metadata
        yml = {}
        yml['ldim'] = self.ldim
        yml['pdim'] = self.pdim

        # ... information about the patches
        if not( self.mappings ):
            raise ValueError('No mappings were found')

        patches_info = []
        i_mapping    = 0
        for patch_name, mapping in self.mappings.items():
            name       = '{}'.format( patch_name )
            mapping_id = 'mapping_{}'.format( i_mapping  )
            dtype      = '{}'.format( type( mapping ).__name__ )

            patches_info += [{'name': name,
                              'mapping_id': mapping_id,
                               'type': dtype}]

            i_mapping += 1

        yml['patches'] = patches_info
        # ...


        # ... topology
        topo_yml = self.domain.todict()
        # ...

        # Create HDF5 file (in parallel mode if MPI communicator size > 1)
        if not(comm is None) and comm.size > 1:
            kwargs = dict( driver='mpio', comm=comm )

        else:
            kwargs = {}

        h5 = h5py.File( filename, mode='w', **kwargs )

        # ...
        # Dump geometry metadata to string in YAML file format
        geo = yaml.dump( data   = yml, sort_keys=False)

        # Write geometry metadata as fixed-length array of ASCII characters
        h5['geometry.yml'] = np.array( geo, dtype='S' )
        # ...

        # ...
        # Dump geometry metadata to string in YAML file format
        geo = yaml.dump( data   = topo_yml, sort_keys=False)
        # Write topology metadata as fixed-length array of ASCII characters
        h5['topology.yml'] = np.array( geo, dtype='S' )
        # ...

        i_mapping    = 0
        for patch_name, mapping in self.mappings.items():
            space = mapping.space

            # Create group for patch 0
            group = h5.create_group( yml['patches'][i_mapping]['mapping_id'] )
            group.attrs['shape'      ] = space.vector_space.npts
            group.attrs['degree'     ] = space.degree
            group.attrs['rational'   ] = False # TODO remove
            group.attrs['periodic'   ] = space.periodic
            for d in range( self.ldim ):
                group['knots_{}'.format( d )] = space.spaces[d].knots

            # Collective: create dataset for control points
            shape = [n for n in space.vector_space.npts] + [self.pdim]
            dtype = space.vector_space.dtype
            dset  = group.create_dataset( 'points', shape=shape, dtype=dtype )

            # Independent: write control points to dataset
            starts = space.vector_space.starts
            ends   = space.vector_space.ends
            index  = [slice(s, e+1) for s, e in zip(starts, ends)] + [slice(None)]
            index  = tuple( index )
            dset[index] = mapping.control_points[index]

            # case of NURBS
            if isinstance(mapping, NurbsMapping):
                # Collective: create dataset for weights
                shape = [n for n in space.vector_space.npts]
                dtype = space.vector_space.dtype
                dset  = group.create_dataset( 'weights', shape=shape, dtype=dtype )

                # Independent: write weights to dataset
                starts = space.vector_space.starts
                ends   = space.vector_space.ends
                index  = [slice(s, e+1) for s, e in zip(starts, ends)]
                index  = tuple( index )
                dset[index] = mapping.weights[index]

            i_mapping += 1

        # Close HDF5 file
        h5.close()

#==============================================================================
def export_nurbs_to_hdf5(filename, nurbs, periodic=None, comm=None ):

    """
    Export a single-patch igakit NURBS object to a Psydac geometry file in HDF5 format

    Parameters
    ----------

    filename : <str>
        Name of output geometry file, e.g. 'geo.h5'

    nurbs   : <igakit.nurbs.NURBS>
        igakit geometry nurbs object

    comm : <MPI.COMM>
        mpi communicator
    """

    import os.path
    import igakit
    assert isinstance(nurbs, igakit.nurbs.NURBS)

    extension = os.path.splitext(filename)[-1]
    if not extension == '.h5':
        raise ValueError('> Only h5 extension is allowed for filename')

    yml = {}
    yml['ldim'] = nurbs.dim
    yml['pdim'] = nurbs.dim

    patches_info = []
    i_mapping    = 0
    i            = 0

    rational = not abs(nurbs.weights-1).sum()<1e-15

    patch_name = 'patch_{}'.format(i)
    name       = '{}'.format( patch_name )
    mapping_id = 'mapping_{}'.format( i_mapping  )
    dtype      = 'NurbsMapping' if rational else 'SplineMapping'

    patches_info += [{'name': name , 'mapping_id':mapping_id, 'type':dtype}]

    yml['patches'] = patches_info
    # ...

    # Create HDF5 file (in parallel mode if MPI communicator size > 1)
    if not(comm is None) and comm.size > 1:
        kwargs = dict( driver='mpio', comm=comm )
    else:
        kwargs = {}

    h5 = h5py.File( filename, mode='w', **kwargs )

    # ...
    # Dump geometry metadata to string in YAML file format
    geom = yaml.dump( data   = yml, sort_keys=False)
    # Write geometry metadata as fixed-length array of ASCII characters
    h5['geometry.yml'] = np.array( geom, dtype='S' )
    # ...

    # ... topology
    if nurbs.dim == 1:
        bounds1 = (float(nurbs.breaks(0)[0]), float(nurbs.breaks(0)[-1]))
        domain  = Line(patch_name, bounds1=bounds1)

    elif nurbs.dim == 2:
        bounds1 = (float(nurbs.breaks(0)[0]), float(nurbs.breaks(0)[-1]))
        bounds2 = (float(nurbs.breaks(1)[0]), float(nurbs.breaks(1)[-1]))
        domain  = Square(patch_name, bounds1=bounds1, bounds2=bounds2)

    elif nurbs.dim == 3:
        bounds1 = (float(nurbs.breaks(0)[0]), float(nurbs.breaks(0)[-1]))
        bounds2 = (float(nurbs.breaks(1)[0]), float(nurbs.breaks(1)[-1]))
        bounds3 = (float(nurbs.breaks(2)[0]), float(nurbs.breaks(2)[-1]))
        domain  = Cube(patch_name, bounds1=bounds1, bounds2=bounds2, bounds3=bounds3)

    mapping = Mapping(mapping_id, dim=nurbs.dim)
    domain  = mapping(domain)
    topo_yml = domain.todict()

    # Dump geometry metadata to string in YAML file format
    geom = yaml.dump( data   = topo_yml, sort_keys=False)
    # Write topology metadata as fixed-length array of ASCII characters
    h5['topology.yml'] = np.array( geom, dtype='S' )

    group = h5.create_group( yml['patches'][i]['mapping_id'] )
    group.attrs['degree'     ] = nurbs.degree
    group.attrs['rational'   ] = rational
    group.attrs['periodic'   ] = tuple( False for d in range( nurbs.dim ) ) if periodic is None else periodic
    for d in range( nurbs.dim ):
        group['knots_{}'.format( d )] = nurbs.knots[d]

    group['points'] = nurbs.points[...,:nurbs.dim]
    if rational:
        group['weights'] = nurbs.weights

    h5.close()

#==============================================================================
def refine_nurbs(nrb, ncells=None, degree=None, multiplicity=None, tol=1e-9):
    """
    This function refines the nurbs object.
    It contructs a new grid based on the new number of cells, and it adds the new break points to the nrb grid,
    such that the total number of cells is equal to the new number of cells.
    We use knot insertion to construct the new knot sequence , so the geometry is identical to the previous one.
    It also elevates the degree of the nrb object based on the new degree.

    Parameters
    ----------

    nrb : <igakit.nurbs.NURBS>
        geometry nurbs object

    ncells   : <list>
        total number of cells in each direction

    degree : <list>
        degree in each direction

    multiplicity : <list>
        multiplicity of each knot in the knot sequence in each direction

    tol : <float>
        Minimum distance between two break points.

    Returns
    -------
    nrb : <igakit.nurbs.NURBS>
        the refined geometry nurbs object

    """

    if multiplicity is None:
        multiplicity = [1]*nrb.dim

    nrb = nrb.clone()
    if ncells is not None:

        for axis in range(0,nrb.dim):
            ub = nrb.breaks(axis)[0]
            ue = nrb.breaks(axis)[-1]
            knots = np.linspace(ub,ue,ncells[axis]+1)
            index = nrb.knots[axis].searchsorted(knots)
            nrb_knots = nrb.knots[axis][index]
            for m,(nrb_k, k) in enumerate(zip(nrb_knots, knots)):
                if abs(k-nrb_k)<tol:
                    knots[m] = np.nan

            knots   = knots[~np.isnan(knots)]
            indices = np.round(np.linspace(0, len(knots) - 1, ncells[axis]+1-len(nrb.breaks(axis)))).astype(int)

            knots = knots[indices]

            if len(knots)>0:
                nrb.refine(axis, knots)

    if degree is not None:
        for axis in range(0,nrb.dim):
            d = degree[axis] - nrb.degree[axis]
            if d<0:
                raise ValueError('The degree {} must be >= {}'.format(degree, nrb.degree))
            nrb.elevate(axis, times=d)

    for axis in range(nrb.dim):
        decimals = abs(np.floor(np.log10(np.abs(tol))).astype(int))
        knots, counts = np.unique(nrb.knots[axis].round(decimals=decimals), return_counts=True)
        counts = multiplicity[axis] - counts
        counts[counts<0] = 0
        knots = np.repeat(knots, counts)
        nrb = nrb.refine(axis, knots)
    return nrb

def refine_knots(knots, ncells, degree, multiplicity=None, tol=1e-9):
    """
    This function refines the knot sequence.
    It contructs a new grid based on the new number of cells, and it adds the new break points to the nrb grid,
    such that the total number of cells is equal to the new number of cells.
    We use knot insertion to construct the new knot sequence , so the geometry is identical to the previous one.
    It also elevates the degree of the nrb object based on the new degree.

    Parameters
    ----------

    knots : <list>
        list of knot sequences in each direction

    ncells   : <list>
        total number of cells in each direction

    degree : <list>
        degree in each direction

    multiplicity : <list>
        multiplicity of each knot in the knot sequence in each direction

    tol : <float>
        Minimum distance between two break points.

    Returns
    -------
    knots : <list>
        the refined knot sequences in each direction
    """
    from igakit.nurbs import NURBS
    dim = len(ncells)

    if multiplicity is None:
        multiplicity = [1]*dim

    assert len(knots) == dim

    nrb = NURBS(knots)
    for axis in range(dim):
        ub = nrb.breaks(axis)[0]
        ue = nrb.breaks(axis)[-1]
        knots = np.linspace(ub,ue,ncells[axis]+1)
        index = nrb.knots[axis].searchsorted(knots)
        nrb_knots = nrb.knots[axis][index]
        for m,(nrb_k, k) in enumerate(zip(nrb_knots, knots)):
            if abs(k-nrb_k)<tol:
                knots[m] = np.nan

        knots   = knots[~np.isnan(knots)]
        indices = np.round(np.linspace(0, len(knots) - 1, ncells[axis]+1-len(nrb.breaks(axis)))).astype(int)

        knots = knots[indices]

        if len(knots)>0:
            nrb.refine(axis, knots)

    for axis in range(dim):
        d = degree[axis] - nrb.degree[axis]
        if d<0:
            raise ValueError('The degree {} must be >= {}'.format(degree, nrb.degree))
        nrb.elevate(axis, times=d)

    for axis in range(dim):
        decimals = abs(np.floor(np.log10(np.abs(tol))).astype(int))
        knots, counts = np.unique(nrb.knots[axis].round(decimals=decimals), return_counts=True)
        counts = multiplicity[axis] - counts
        counts[counts<0] = 0
        knots = np.repeat(knots, counts)
        nrb = nrb.refine(axis, knots)
    return nrb.knots
#==============================================================================
def import_geopdes_to_nurbs(filename):
    """
    This function reads a geopdes geometry file and convert it to igakit nurbs object

    Parameters
    ----------

    filename : <str>
        the filename of the geometry file

    Returns
    -------
    nrb : <igakit.nurbs.NURBS>
        the geometry nurbs object

    """
    extension = os.path.splitext(filename)[-1]
    if not extension == '.txt':
        raise ValueError('> Expected .txt extension')

    f = open(filename)
    lines = f.readlines()
    f.close()

    lines = [line for line in lines if line[0].strip() != "#"]

    data     = _read_header(lines[0])
    n_dim    = data[0]
    r_dim    = data[1]
    n_patchs = data[2]

    n_lines_per_patch = 3*n_dim + 1

    list_begin_line = _get_begin_line(lines, n_patchs)

    nrb = _read_patch(lines, 1, n_lines_per_patch, list_begin_line)

    return nrb

def _read_header(line):
    chars = line.split(" ")
    data  = []
    for c in chars:
        try:
            data.append(int(c))
        except:
            pass
    return data

def _extract_patch_line(lines, i_patch):
    text = "PATCH " + str(i_patch)
    for i_line,line in enumerate(lines):
        r = line.find(text)
        if r != -1:
            return i_line
    return None

def _get_begin_line(lines, n_patchs):
    list_begin_line = []
    for i_patch in range(0, n_patchs):
        r = _extract_patch_line(lines, i_patch+1)
        if r is not None:
            list_begin_line.append(r)
        else:
            raise ValueError(" could not parse the input file")
    return list_begin_line

def _read_line(line):
    chars = line.split(" ")
    data  = []
    for c in chars:
        try:
            data.append(int(c))
        except:
            try:
                data.append(float(c))
            except:
                pass
    return data

def _read_patch(lines, i_patch, n_lines_per_patch, list_begin_line):

    from igakit.nurbs import NURBS

    i_begin_line = list_begin_line[i_patch-1]
    data_patch = []

    for i in range(i_begin_line+1, i_begin_line + n_lines_per_patch+1):
        data_patch.append(_read_line(lines[i]))

    degree = data_patch[0]
    shape  = data_patch[1]

    xl     = [np.array(i) for i in data_patch[2:2+len(degree)] ]
    xp     = [np.array(i) for i in data_patch[2+len(degree):2+2*len(degree)] ]
    w      = np.array(data_patch[2+2*len(degree)])

    X = [i.reshape(shape, order='F') for i in xp]
    W = w.reshape(shape, order='F')

    points = np.zeros((*shape, 3))
    for i in range(len(shape)):
        points[..., i] = X[i]

    knots = xl

    nrb = NURBS(knots, control=points, weights=W)
    return nrb

