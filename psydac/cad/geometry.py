# coding: utf-8
#
# a Geometry class contains the list of patches and additional information about
# the topology i.e. connectivity, boundaries
# For the moment, it is used as a container, that can be loaded from a file
# (hdf5)

from itertools import product
from collections import OrderedDict
from collections import abc
import numpy as np
import string
import random
import h5py
import yaml
import yamlloader
import os
import string
import random
from mpi4py import MPI

from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.mapping.discrete import SplineMapping, NurbsMapping

from sympde.topology import Domain, Line, Square, Cube

#==============================================================================
class Geometry( object ):
    _ldim     = None
    _pdim     = None
    _patches  = []
    _topology = None

    #--------------------------------------------------------------------------
    # Option [1]: from a (domain, mappings) or a file
    #--------------------------------------------------------------------------
    def __init__( self, domain=None, mappings=None,
                  filename=None, comm=MPI.COMM_WORLD ):

        # ... read the geometry if the filename is given
        if not( filename is None ):
            self.read(filename, comm=comm)

        elif not( domain is None ):
            assert( isinstance( domain, Domain ) )
            assert( not( mappings is None ))
            assert( isinstance( mappings, (dict, OrderedDict) ) )

            # ... check sanity
            interior_names = sorted(domain.interior_names)
            mappings_keys  = sorted(list(mappings.keys()))

            assert( interior_names == mappings_keys )
            # ...

            self._domain   = domain
            self._ldim     = domain.dim
            self._pdim     = domain.dim # TODO must be given => only dim is  defined for a Domain
            self._mappings = OrderedDict(mappings.items())

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
    def from_topological_domain(cls, domain, ncells, comm=None):

        if not isinstance(domain, (Line, Square, Cube)):
            msg = "Topological domain must be Line, Square or Cube;"\
                  " got {} instead.".format(type(domain))
            raise TypeError(msg)

        mappings = {domain.interior.name: None}
        geo = Geometry(domain=domain, mappings=mappings, comm=comm)
        geo.ncells = ncells

        return geo

    #--------------------------------------------------------------------------
    @property
    def ldim(self):
        return self._ldim

    @property
    def pdim(self):
        return self._pdim

    @property
    def comm(self):
        return self._comm

    @property
    def domain(self):
        return self._domain

    @property
    def mappings(self):
        return self._mappings

    @property
    def boundaries(self):
        return self.domain.boundaries

    def __len__(self):
        return len(self.domain)

    def read( self, filename, comm=MPI.COMM_WORLD ):
        # ... check extension of the file
        basename, ext = os.path.splitext(filename)
        if not(ext == '.h5'):
            raise ValueError('> Only h5 files are supported')
        # ...

        # read the topological domain
        domain = Domain.from_file(filename)

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
        for i_patch in range( n_patches ):

            item  = yml['patches'][i_patch]
            patch_name = item['name']
            mapping_id = item['mapping_id']
            dtype = item['type']
            patch = h5[mapping_id]
            if dtype in ['SplineMapping', 'NurbsMapping']:

                degree   = [int (p) for p in patch.attrs['degree'  ]]
                periodic = [bool(b) for b in patch.attrs['periodic']]
                knots    = [patch['knots_{}'.format(d)][:] for d in range( ldim )]
                spaces   = [SplineSpace( degree=p, knots=k, periodic=b )
                            for p,k,b in zip( degree, knots, periodic )]

                tensor_space = TensorFemSpace( *spaces, comm=comm )
                if dtype == 'SplineMapping':
                    mapping = SplineMapping.from_control_points( tensor_space,
                                                                 patch['points'] )

                elif dtype == 'NurbsMapping':
                    mapping = NurbsMapping.from_control_points_weights( tensor_space,
                                                                        patch['points'],
                                                                        patch['weights'] )

                mapping.set_name( item['name'] )

                mappings[patch_name] = mapping
        # ...

        # ... close the h5 file
        h5.close()
        # ...

        # ...
        self._ldim       = ldim
        self._pdim       = pdim
        self._mappings   = mappings
        self._domain     = domain
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
        yml = OrderedDict()
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

            patches_info += [OrderedDict( [('name'       , name       ),
                                           ('mapping_id' , mapping_id ),
                                           ('type'       , dtype      )] )]

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
        geo = yaml.dump( data   = yml,
                         Dumper = yamlloader.ordereddict.Dumper )

        # Write geometry metadata as fixed-length array of ASCII characters
        h5['geometry.yml'] = np.array( geo, dtype='S' )
        # ...

        # ...
        # Dump geometry metadata to string in YAML file format
        geo = yaml.dump( data   = topo_yml,
                         Dumper = yamlloader.ordereddict.Dumper )
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
