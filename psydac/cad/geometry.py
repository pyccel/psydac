#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

# a Geometry class contains the list of patches and additional information about
# the topology i.e. connectivity, boundaries
# For the moment, it is used as a container, that can be loaded from a file
# (hdf5)
import os
from typing import Iterable
from itertools import chain

import numpy as np
import h5py
import yaml
from mpi4py import MPI

from sympde.topology       import Domain, Interface, Line, Square, Cube, NCubeInterior, Mapping, NCube
from sympde.topology.basic import Union
from sympde.topology.callable_mapping import BasicCallableMapping

from psydac.fem.splines        import SplineSpace
from psydac.fem.tensor         import TensorFemSpace
from psydac.fem.partitioning   import create_cart, construct_connectivity, construct_interface_spaces
from psydac.mapping.discrete   import SplineMapping, NurbsMapping
from psydac.linalg.block       import BlockVectorSpace, BlockVector
from psydac.ddm.cart           import DomainDecomposition, MultiPatchDomainDecomposition

__all__ = (
    'Geometry',
    'export_nurbs_to_hdf5',
    'import_geopdes_to_nurbs',
    'refine_knots',
    'refine_nurbs',
)

NoneType = type(None)

#==============================================================================
class Geometry:
    """
    Distributed discrete geometry that works for single and multiple patches.

    The Geometry object can be created in four ways:
    - case 0 : providing a `Domain` to `__init__` with detailed parameters for each patch.
    - case 1 : passing the path to a geometry file to `from_file`.
    - case 2 : passing a `SplineMapping` to `from_discrete_mapping` (single patch).
    - case 3 : passing a `Domain`, ncells, and periodicity to `from_topological_domain` (single or multi-patch).

    Parameters
    ----------
    domain : Sympde.topology.Domain
        The symbolic topological domain to be discretized.

    pdim : int
        Number of physical dimensions of the Geometry object (pdim >= ldim).

    ncells : dict[str, Iterable[int]]
        The number of cells of the discretized domain in each direction.

    periodic : dict[str, Iterable[bool]], optional
        The periodicity of the topological domain in each direction.

    mappings : dict[str, BasicCallableMapping], optional
        The discrete mappings of each patch.

    comm: MPI.Intracomm, optional
        MPI intra-communicator.

    mpi_dims_mask: Iterable[bool], optional
        True if the dimension is to be used in the domain decomposition (=default for each dimension). 
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.
  
    """
    _ldim     = None
    _pdim     = None
    _patches  = []
    _topology = None

    def __init__(self,
                 domain : Domain,
                 *,
                 pdim     : int,
                 ncells   : dict[str, Iterable[int]],
                 mappings : dict[str, SplineMapping | None] = None,
                 periodic : dict[str, Iterable[bool]] = None,
                 comm : MPI.Intracomm = None,
                 mpi_dims_mask : Iterable[bool] = None):

        # Type checks
        assert isinstance(pdim, int)
        assert isinstance(domain, Domain) 
        assert isinstance(ncells, dict)
        assert isinstance(mappings, dict)
        assert isinstance(periodic, (NoneType, dict))
        assert isinstance(comm, (NoneType, MPI.Intracomm))
        assert isinstance(mpi_dims_mask, (NoneType, Iterable))

        # Extract info from domain
        ldim : int = domain.dim
        interior_names : list = domain.interior_names
        set_interior_names = set(interior_names)

        # Check sanity of pdim
        assert pdim >= ldim

        # Check sanity of ncells
        assert set(ncells.keys()) == set_interior_names
        assert all(len(n) == ldim for n in ncells.values())
        assert all(isinstance(ni, int) for ni in chain(*ncells.values()))
        assert all(ni > 0 for ni in chain(*ncells.values()))

        # Check sanity of periodic
        if periodic is None:
            periodic = {patch: [False]*len(ncells_i) for patch, ncells_i in ncells.items()}
        else:
            assert set(periodic.keys()) == set_interior_names
            assert all(len(p) == ldim for p in periodic.values())
            assert all(isinstance(pi, bool) for pi in chain(*periodic.values()))

        # Check sanity of mappings
        if mappings is None:
            mappings = {itr.name : None for itr in domain.interior}
        else:
            assert set(mappings.keys()) == set_interior_names
            assert all(isinstance(m, (BasicCallableMapping, NoneType)) for m in mappings.values())
            assert all(m.pdim == pdim for m in mappings.values() if m is not None)

        # Check sanity of mpi_dims_mask
        if mpi_dims_mask is not None:
            assert len(mpi_dims_mask) == ldim
            assert all(isinstance(mask, bool) for mask in mpi_dims_mask)

        # Create a (multi-patch) domain decomposition
        if len(domain) == 1:
            #name = domain.name
            name = interior_names[0]
            ddm = DomainDecomposition(
                ncells  = ncells[name],
                periods = periodic[name],
                comm    = comm,
                mpi_dims_mask = mpi_dims_mask,
            )
        else:
            ddm = MultiPatchDomainDecomposition(
                ncells  = [  ncells[itr] for itr in interior_names],
                periods = [periodic[itr] for itr in interior_names],
                comm    = comm,
            )

        # Add attributes to the new object
        self._domain   = domain
        self._ldim     = domain.dim
        self._pdim     = pdim
        self._ncells   = ncells
        self._mappings = mappings
        self._periodic = periodic
        self._comm     = comm
        self._ddm      = ddm
        self._cart     = None

    #--------------------------------------------------------------------------
    # Option [1]: from a file
    #--------------------------------------------------------------------------
    @classmethod
    def from_file(cls,
            filename : str,
            *,
            comm : MPI.Intracomm = None,
            mpi_dims_mask : Iterable[bool] = None):

        """
        Create a Geometry instance from an HDF5 input file in Psydac format.

        Parameters
        ----------
        filename: str
            The path to the geometry file.

        comm: MPI.Intracomm, optional
            The MPI intra-communicator.

        mpi_dims_mask: Iterable[bool], optional
            True if the dimension is to be used in the domain decomposition
            (=default for each dimension). If mpi_dims_mask[i]=False, the i-th
            dimension will not be decomposed.
    
        Returns
        -------
        Geometry
            The new instance.
        """
        geo = super().__new__(cls)
        geo.read(filename, comm=comm, mpi_dims_mask=mpi_dims_mask)
        return geo

    #--------------------------------------------------------------------------
    # Option [2]: from a discrete mapping
    #--------------------------------------------------------------------------
    @classmethod
    def from_discrete_mapping(cls, mapping, *, comm=None, mpi_dims_mask=None, name=None):
        """
        Create a single-patch Geometry instance from one discrete mapping.

        Parameters
        ----------
        mapping : BasicCallableMapping
            The mapping from the unit square to the physical domain.

        comm : MPI.Comm
            MPI intra-communicator.
    
        mpi_dims_mask: list of bool
            True if the dimension is to be used in the domain decomposition (=default for each dimension). 
            If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.
    
        name : str
            Optional name for the symbolic Mapping that will be created.
            Needed to avoid conflicts in case several mappings are created.

        Returns
        -------
        Geometry
            The new instance.
        """

        mapping_name = name if name else 'mapping'
        dim      = mapping.ldim
        M        = Mapping(mapping_name, dim = dim)  # this is a symbolic mapping
        domain   = M(NCube(name = 'Omega',
                           dim  = dim,
                           min_coords = [0.] * dim,
                           max_coords = [1.] * dim)) 
        M.set_callable_mapping(mapping)
        pdim     = mapping.pdim
        mappings = {domain.name: mapping}
        ncells   = {domain.name: mapping.space.domain_decomposition.ncells}
        periodic = {domain.name: mapping.space.domain_decomposition.periods}

        return Geometry(domain   = domain,
                        pdim     = pdim,
                        ncells   = ncells,
                        periodic = periodic,
                        mappings = mappings,
                        comm     = comm,
                        mpi_dims_mask = mpi_dims_mask)

    #--------------------------------------------------------------------------
    # Option [3]: discrete topological line/square/cube
    #--------------------------------------------------------------------------
    @classmethod
    def from_topological_domain(cls, domain, ncells, *, periodic=None, comm=None, mpi_dims_mask=None):
        interior = domain.interior
        if not isinstance(interior, Union):
            interior = [interior]

        for itr in interior:
            if not isinstance(itr, NCubeInterior):
                msg = "The topological domain of each patch must be an NCube;"\
                      " got {} instead.".format(type(itr))
                raise TypeError(msg)

        mappings = {itr.name : None for itr in interior}
        pdim = next(iter(interior)).dim

        if isinstance(ncells, (list, tuple)):
            ncells = {itr.name : ncells for itr in interior}

        if periodic is None:
            periodic = [False] * domain.dim
        else:
            if len(interior) > 1 and True in periodic:
                msg = "Discretizing a multipatch domain with a periodic flag is not advised -- continue at your own risk."
                # [MCP 18.12.2025] the following line may be causing a strange error in the CI (MPI tests for macos-14/Python 3.10)
                # warnings.warn(msg, Warning)  
                warnings.warn(msg, UserWarning)


        if isinstance(periodic, (list, tuple)):
            periodic = {itr.name : periodic for itr in interior}

        return Geometry(domain   = domain,
                        pdim     = pdim,
                        ncells   = ncells,
                        periodic = periodic,
                        mappings = mappings,
                        comm     = comm,
                        mpi_dims_mask = mpi_dims_mask)

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
    def mappings(self):
        return self._mappings

    def __len__(self):
        return len(self.domain)

    def read(self, filename, comm=None, mpi_dims_mask=None):
        # ... check extension of the file
        _, ext = os.path.splitext(filename)
        if ext != '.h5':
            raise ValueError('> Only h5 files are supported')
        # ...

        # read the topological domain
        domain       = Domain.from_file(filename)
        connectivity = construct_connectivity(domain)

        if len(domain) == 1:
            interiors = [domain.interior]
        else:
            interiors = list(domain.interior.args)

        if comm is not None:
            kwargs = dict(driver='mpio', comm=comm) if comm.size > 1 else {}
        else:
            kwargs = {}

        h5  = h5py.File(filename, mode='r', **kwargs)
        yml = yaml.load(h5['geometry.yml'][()], Loader=yaml.SafeLoader)

        ldim = yml['ldim']
        pdim = yml['pdim']

        n_patches = len(yml['patches'])

        # ...
        if n_patches == 0:
            h5.close()
            raise ValueError("Input file contains no patches.")
        # ...

        # ... read patches
        mappings = {}
        ncells   = {}
        periodic = {}
        spaces   = [None] * n_patches
        for i_patch in range(n_patches):

            item  = yml['patches'][i_patch]
            patch_name = item['name']
            mapping_id = item['mapping_id']
            dtype = item['type']
            patch = h5[mapping_id]
            if dtype in ['SplineMapping', 'NurbsMapping']:

                degree     = [int (p) for p in patch.attrs['degree'  ]]
                periodic_i = [bool(b) for b in patch.attrs['periodic']]
                knots      = [patch['knots_{}'.format(d)][:] for d in range(ldim)]
                space_i    = [SplineSpace(degree=p, knots=k, periodic=P)
                              for p, k, P in zip(degree, knots, periodic_i)]

                spaces[i_patch] = space_i

                ncells  [interiors[i_patch].name] = [sp.ncells for sp in space_i]
                periodic[interiors[i_patch].name] = periodic_i

        if n_patches == 1:
            ddm  = DomainDecomposition(ncells[domain.name], periodic[domain.name], comm=comm, mpi_dims_mask=mpi_dims_mask)
            ddms = [ddm]
        else:
            ncells_  = [ncells[itr.name] for itr in interiors]
            periodic = [periodic[itr.name] for itr in interiors]
            ddm      = MultiPatchDomainDecomposition(ncells_, periodic, comm=comm)
            ddms     = ddm.domains

        carts    = create_cart(ddms, spaces)
        g_spaces = {inter:TensorFemSpace(ddms[i], *spaces[i], cart=carts[i]) for i,inter in enumerate(interiors)}

        for i, j in connectivity:
            minus = interiors[i]
            plus  = interiors[j]
            max_ncells = [max(ni, nj) for ni, nj in zip(ncells[minus.name], ncells[plus.name])]
            g_spaces[minus].add_refined_space(ncells=max_ncells)
            g_spaces[plus ].add_refined_space(ncells=max_ncells)

        # ... construct interface spaces
        construct_interface_spaces(ddm, g_spaces, carts, interiors, connectivity)

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
                    mapping = SplineMapping.from_control_points(tensor_space,
                                                                patch['points'][..., :pdim])

                elif dtype == 'NurbsMapping':
                    mapping = NurbsMapping.from_control_points_weights(tensor_space,
                                                                       patch['points'][..., :pdim],
                                                                       patch['weights'])

                mapping.set_name(item['name'])
                mappings[patch_name] = mapping

        # ... Update ghost regions within each patch and across interfaces
        if n_patches > 1:
            coeffs         = [[e.coeffs for e in mapping.fields] for mapping in mappings.values()]
            patch_spaces   = [BlockVectorSpace(*[c_ij.space for c_ij in c_i]) for c_i in coeffs]
            patch_spaces_w = [c_i[0].space for c_i in coeffs]
            space          = BlockVectorSpace(*patch_spaces  , connectivity=connectivity)
            space_w        = BlockVectorSpace(*patch_spaces_w, connectivity=connectivity)
            v = BlockVector(space)
            w = BlockVector(space_w)
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
        # ...

        # ... close the h5 file
        h5.close()
        # ...

        # Add spline callable mappings to domain undefined mappings
        # NOTE: We assume that interiors and mappings.values() use the same ordering
        for patch, F in zip(interiors, mappings.values()):
            patch.mapping.set_callable_mapping(F)

        # ...
        self._domain      = domain
        self._ldim        = ldim
        self._pdim        = pdim
        self._ncells      = ncells
        self._mappings    = mappings
        self._periodic    = periodic
        self._comm        = comm
        self._ddm         = ddm
        self._cart        = None
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
            group.attrs['shape'      ] = space.coeff_space.npts
            group.attrs['degree'     ] = space.degree
            group.attrs['rational'   ] = False # TODO remove
            group.attrs['periodic'   ] = space.periodic
            for d in range( self.ldim ):
                group['knots_{}'.format( d )] = space.spaces[d].knots

            # Collective: create dataset for control points
            shape = [n for n in space.coeff_space.npts] + [self.pdim]
            dtype = space.coeff_space.dtype
            dset  = group.create_dataset( 'points', shape=shape, dtype=dtype )

            # Independent: write control points to dataset
            starts = space.coeff_space.starts
            ends   = space.coeff_space.ends
            index  = [slice(s, e+1) for s, e in zip(starts, ends)] + [slice(None)]
            index  = tuple( index )
            dset[index] = mapping.control_points[index]

            # case of NURBS
            if isinstance(mapping, NurbsMapping):
                # Collective: create dataset for weights
                shape = [n for n in space.coeff_space.npts]
                dtype = space.coeff_space.dtype
                dset  = group.create_dataset( 'weights', shape=shape, dtype=dtype )

                # Independent: write weights to dataset
                starts = space.coeff_space.starts
                ends   = space.coeff_space.ends
                index  = [slice(s, e+1) for s, e in zip(starts, ends)]
                index  = tuple( index )
                dset[index] = mapping.weights[index]

            i_mapping += 1

        # Close HDF5 file
        h5.close()

#==============================================================================
def export_nurbs_to_hdf5(filename, nurbs, periodic=None, comm=None ):

    """
    Export a single-patch igakit NURBS object to a PSYDAC geometry file in HDF5 format

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

    else:
        raise NotImplementedError('> nurbs.dim > 3 not implemented')

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
        except ValueError:
            msg = f"WARNING: Cannot convert str '{c}' to int. Moving to next word..."
            print(msg)
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
            i = int(c)
        except ValueError:
            i = None
        else:
            data.append(i)
            continue

        try:
            f = float(c)
        except ValueError:
            f = None
        else:
            data.append(f)
            continue

        if i is None and f is None:
            msg = f"WARNING: Cannot convert str '{c}' to int or float. Moving to next word..."
            print(msg)

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
