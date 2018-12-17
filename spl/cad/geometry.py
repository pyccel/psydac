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

from spl.fem.splines      import SplineSpace
from spl.fem.tensor       import TensorFemSpace
from spl.mapping.discrete import SplineMapping
from spl.cad.basic        import BasicDiscreteDomain
from spl.cad.basic        import DiscreteBoundary
from spl.cad.basic        import Edge, Topology


#==============================================================================
class Geometry( BasicDiscreteDomain ):
    _patches  = []
    _topology = None

    def __init__( self, filename=None, comm=MPI.COMM_WORLD,
                  patches=None, topology=None ):

        # ... read the geometry if the filename is given
        if not( filename is None ):
            self.read(filename, comm=comm)

        elif not( patches is None ):
            assert( isinstance( patches, (list, tuple) ) )
            assert( not( topology is None ))
            assert( isinstance( topology, Topology ) )

            self._patches  = patches
            self._ldim     = patches[0].ldim
            self._pdim     = patches[0].pdim
            self._topology = topology

        else:
            raise ValueError('Wrong input')
        # ...

    @property
    def patches(self):
        return self._patches

    @property
    def topology(self):
        return self._topology

    @property
    def boundaries(self):
        return self.topology.boundaries

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

        # connectivity
        for k,v in yml['topology']['connectivity'].items():
            edge = Edge(k)
            bnds = []
            for desc in v:
                patch = d_patches[desc['patch']]
                axis  = desc['axis']
                ext   = desc['ext']
                name  = desc['name']
                bnd   = DiscreteBoundary(patch, axis=axis, ext=ext, name=name)
                bnds.append(bnd)

            topology[edge] = bnds

        # boundaries
        bnds = []
        for desc in yml['topology']['boundaries']:
            patch = d_patches[desc['patch']]
            axis  = desc['axis']
            ext   = desc['ext']
            name  = desc['name']
            bnd   = DiscreteBoundary(patch, axis=axis, ext=ext, name=name)
            bnds.append(bnd)

        topology.set_boundaries(bnds)
        # ...

        # ... close the h5 file
        h5.close()
        # ...

        # ...
        self._ldim       = ldim
        self._pdim       = pdim
        self._patches    = patches
        self._topology   = topology
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

    geo = Geometry('square_mp_0.h5')
    print(geo.boundaries)
    geo.export('square_mp_1.h5')


