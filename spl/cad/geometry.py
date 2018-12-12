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
from mpi4py import MPI

from spl.fem.splines      import SplineSpace
from spl.fem.tensor       import TensorFemSpace
from spl.mapping.discrete import SplineMapping

_patches_types = (SplineMapping,)

#==============================================================================
class Edge(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


#==============================================================================
class Topology(abc.Mapping):

    def __init__(self, data=None, filename=None):
        if data is None:
            data = {}

        else:
            assert( isinstance( data, (dict, OrderedDict)) )

        self._data = data

        if not( filename is None ):
            self.read(filename)

    def read(self, filename):
        raise NotImplementedError('TODO')

    def export(self, filename):
        raise NotImplementedError('TODO')

    def get_boundaries(self):
        # TODO treate periodicity of every patch
        # we first inverse the graph, by having patches as keys
        d = {}
        data = OrderedDict(sorted(self._data.items()))
        for edge, pair in data.items():
            p_left  = pair[0].patch
            p_right = pair[1].patch
            for face in pair:
                if not( face.patch in d.keys() ):
                    d[face.patch] = []

                d[face.patch].append(face)

        boundaries = []
        for patch in d.keys():
            ldim = patch.ldim
            for axis in range(ldim):
                for ext in [-1, 1]:
                    bnd = DiscreteBoundary(patch=patch, axis=axis, ext=ext)
                    if not bnd in d[patch]:
                        boundaries.append(bnd)

        return boundaries

    def todict(self):
        # ... create the connectivity
        connectivity = {}
        data = OrderedDict(sorted(self._data.items()))
        for edge, pair in data.items():
            connectivity[edge.name] = [bnd.todict() for bnd in pair]

        connectivity = OrderedDict(sorted(connectivity.items()))
        # ...

        # ...
        boundaries = [bnd.todict() for bnd in self.get_boundaries()]
        # ...

        return {'connectivity': connectivity,
                'boundaries': boundaries}


    def __setitem__(self, key, value):
        assert( isinstance( key, Edge ) )
        assert( isinstance( value, (tuple, list)  ) )
        assert( len(value) in [1, 2] )
        assert( all( [isinstance( P, DiscreteBoundary ) for P in value ] ) )

        self._data[key] = value

    # ==========================================
    #  abstract methods
    # ==========================================
    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)
    # ==========================================


#==============================================================================
class DiscreteBoundary(object):

    def __init__(self, patch, axis, ext):
        assert( isinstance( axis, int ) and axis in [0,1,2] )
        assert( isinstance( ext,  int ) and ext  in [-1,1] )
        assert( isinstance( patch, _patches_types ) )

        self._axis = axis
        self._ext  = ext
        self._patch = patch

    @property
    def axis(self):
        return self._axis

    @property
    def ext(self):
        return self._ext

    @property
    def patch(self):
        return self._patch

    def todict(self):
        axis = self.axis
        ext  = self.ext
        patch = self.patch
        return OrderedDict( [('axis' , axis ),
                             ('ext'  , ext  ),
                             ('patch', patch.name )] )

    def __eq__(self, other):
        cond = ( (self.patch is other.patch) and
                 (self.axis == other.axis)   and
                 (self.ext == other.ext) )
        return cond


#==============================================================================
class Geometry( object ):
    _patches    = []
    _topology   = None
    _ldim       = None
    _pdim       = None

    def __init__( self, filename=None, comm=MPI.COMM_WORLD,
                  patches=None, topology=None ):

        # ... read the geometry if the filename is given
        if not( filename is None ):
            self.read(filename, comm=comm)

        elif not( patches is None ):
            assert( isinstance( patches, (list, tuple) ) )
            assert( not( topology is None ))
            assert( isinstance( topology, Topology ) )

            self._patches = patches
            self._ldim = patches[0].ldim
            self._pdim = patches[0].pdim
            self._topology = topology

        else:
            raise ValueError('Wrong input')
        # ...


    @property
    def ldim(self):
        return self._ldim

    @property
    def pdim(self):
        return self._pdim

    @property
    def patches(self):
        return self._patches

    @property
    def topology(self):
        return self._topology

    @property
    def n_patches(self):
        return len(self.patches)

    def __len__(self):
        return len(self.patches)

    def read( self, filename, comm=MPI.COMM_WORLD ):
        # ... check extension of the file
        basename, ext = os.path.splitext(filename)
        if not(ext == '.h5'):
            raise ValueError('> Only h5 files are supported')
        # ...

        if comm.size > 1:
            kwargs = dict( driver='mpio', comm=comm )
        else:
            kwargs = {}

        h5  = h5py.File( filename, mode='r', **kwargs )
        yml = yaml.load( h5['geometry.yml'].value )

        ldim = yml['ldim']
        pdim = yml['pdim']

        n_patches = len( yml['patches'] )

        # ...
        if n_patches == 0:

            h5.close()
            raise ValueError( "Input file contains no patches." )
        # ...

        # ... read patchs
        patches = []
        for i_patch in range( n_patches ):

            item  = yml['patches'][i_patch]
            dtype = item['type']
            patch = h5[item['name']]
            if dtype == 'SplineMapping':

                degree   = [int (p) for p in patch.attrs['degree'  ]]
                periodic = [bool(b) for b in patch.attrs['periodic']]
                knots    = [patch['knots_{}'.format(d)].value for d in range( ldim )]
                spaces   = [SplineSpace( degree=p, knots=k, periodic=b )
                            for p,k,b in zip( degree, knots, periodic )]

                tensor_space = TensorFemSpace( *spaces, comm=comm )
                mapping      = SplineMapping.from_control_points( tensor_space, patch['points'] )
                mapping.set_name( item['name'] )

                patches.append(mapping)
        # ...

        # ... create a dict of patches, needed for the topology
        d_patches = {}
        for patch in patches:
            d_patches[patch.name] = patch

        d_patches = OrderedDict(sorted(d_patches.items()))
        # ...

        # ... read the topology
        topology = Topology()
        for k,v in yml['topology']['connectivity'].items():
            edge = Edge(k)
            bnds = []
            for desc in v:
                patch = d_patches[desc['patch']]
                axis  = desc['axis']
                ext   = desc['ext']
                bnd = DiscreteBoundary(patch, axis=axis, ext=ext)
                bnds.append(bnd)

            topology[edge] = bnds
        # ...

        # ... close the h5 file
        h5.close()
        # ...

        # ...
        self._ldim     = ldim
        self._pdim     = pdim
        self._patches  = patches
        self._topology = topology
        # ...

    def export( self, filename ):
        """
        Parameters
        ----------
        filename : str
          Name of HDF5 output file.

        """

        # ...
        if not( self.n_patches > 0 ):
            raise ValueError('No patches are found')

        comm  = self.patches[0].space.vector_space.cart.comm
        # ...

        # Create dictionary with geometry metadata
        yml = OrderedDict()
        yml['ldim'] = self.ldim
        yml['pdim'] = self.pdim

        # ... information about the patches
        patches_info = []
        for i_patch, patch in enumerate(self.patches):
            dtype = type( patch ).__name__
            if patch.name is None:
                name = 'patch_{}'.format( i_patch )

            else:
                name = patch.name

            patches_info += [OrderedDict( [('name' , '{}'.format( name )  ),
                                           ('type' , '{}'.format( dtype ) ),
                                           ('color', 'None'               )] )]

        yml['patches'] = patches_info
        # ...

        # ... topology
        yml['topology'] = self.topology.todict()
        # ...

        # Dump geometry metadata to string in YAML file format
        geo = yaml.dump( data   = yml,
                         Dumper = yamlloader.ordereddict.Dumper )
        # ...

        # Create HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = dict( driver='mpio', comm=comm ) if comm.size > 1 else {}
        h5 = h5py.File( filename, mode='w', **kwargs )

        # Write geometry metadata as fixed-length array of ASCII characters
        h5['geometry.yml'] = np.array( geo, dtype='S' )

        for i_patch, patch in enumerate(self.patches):
            space = patch.space

            # Create group for patch 0
            group = h5.create_group( yml['patches'][i_patch]['name'] )
            group.attrs['shape'      ] = space.vector_space.npts
            group.attrs['degree'     ] = space.degree
            group.attrs['rational'   ] = False
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
            dset[index] = patch.control_points[index]

        # Close HDF5 file
        h5.close()

######################################
if __name__ == '__main__':
#    geo = Geometry('square_0.h5')
##    geo.export('square_1.h5')
#
#    geo = Geometry('square_mp.h5')
#
#    topo = Topology()
#
#    B = Edge('B')
#
#    P0 = geo.patches[0]
#    P1 = geo.patches[1]
#
#    B_P0 = DiscreteBoundary(patch=P0, axis=1, ext=-1)
#    B_P1 = DiscreteBoundary(patch=P1, axis=1, ext=-1)
#
#    topo[B] = (B_P0, B_P1)
#
#    newgeo = Geometry(patches=[P0, P1], topology=topo)
#    newgeo.export('square_mp_new.h5')

    geo = Geometry('square_mp_new.h5')
    geo.export('square_mp_newnew.h5')


