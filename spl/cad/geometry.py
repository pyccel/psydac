# coding: utf-8
#
# a Geometry class contains the list of patches and additional information about
# the topology i.e. connectivity, boundaries
# For the moment, it is used as a container, that can be loaded from a file
# (hdf5)

from itertools import product
from collections import OrderedDict
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

#==============================================================================
class DiscreteBoundary(object):

    def __init__(self, name, axis, ext, parent=None):
        assert( isinstance( name, str ) )
        assert( isinstance( axis, int ) and axis in [0,1,2] )
        assert( isinstance( ext,  int ) and ext  in [-1,1] )
        if not( parent is None ):
            assert( isinstance( parent, str ) )

        self._name = name
        self._axis = axis
        self._ext  = ext
        self._parent = parent

    @property
    def name(self):
        return self._name

    @property
    def axis(self):
        return self._axis

    @property
    def ext(self):
        return self._ext

    @property
    def parent(self):
        return self._parent

    def todict(self):
        name = self.name
        axis = self.axis
        ext  = self.ext
        parent = self.parent
        return OrderedDict( [('name' , name ),
                             ('axis' , axis ),
                             ('ext'  , ext  ),
                             ('parent', parent )] )


#==============================================================================
class Geometry( object ):
    def __init__( self, filename=None, comm=MPI.COMM_WORLD ):
        # ...
        self._patches      = []
        self._boundaries   = []
        self._ldim         = None
        self._pdim         = None
        # ...

        # ... read the geometry if the filename is given
        if not( filename is None ):
            self.read(filename, comm=comm)
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
    def boundaries(self):
        return self._boundaries

    @property
    def n_patches(self):
        return len(self.patches)

    @property
    def n_boundaries(self):
        return len(self.boundaries)

    def __len__(self):
        return len(self.patches)

    # TODO to be removed
    def _create_boundaries(self):
        boundaries = []
        for i_patch, patch in enumerate(self.patches):
            patch_name = 'patch_{}'.format(i_patch)

            i_bnd = 1
            for axis in range(self.ldim):
                for ext in [-1,1]:
                    name = 'B{}'.format( i_bnd )
                    boundaries += [DiscreteBoundary( name, axis, ext,
                                                     parent=patch_name)]

                    i_bnd += 1
        return boundaries

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

        n_patches    = len( yml['patches'] )
        n_boundaries = len( yml['boundaries'] )

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

                patches.append(mapping)
        # ...

        # ... read boundaries
        boundaries = []
        for i_bnd in range( n_boundaries ):
            item  = yml['boundaries'][i_bnd]

            name   = item['name']
            axis   = item['axis']
            ext    = item['ext']
            parent = item['parent']

            if parent == 'None': parent = None

            bnd = DiscreteBoundary(name, axis, ext, parent=parent)
            boundaries += [bnd]
        # ...

        # ... close the h5 file
        h5.close()
        # ...

        # ... default boundaries
        if not boundaries:
            boundaries = self._create_boundaries()
        # ...

        # ...
        self._ldim       = ldim
        self._pdim       = pdim
        self._patches    = patches
        self._boundaries = boundaries
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
            patches_info += [OrderedDict( [('name' , 'patch_{}'.format( i_patch ) ),
                                           ('type' , '{}'.format( dtype )         ),
                                           ('color', 'None'                       )] )]

        yml['patches'] = patches_info
        # ...

        # ... information about the boundaries
        boundaries_info = []

        for bnd in self.boundaries:
            boundaries_info += [bnd.todict()]

        yml['boundaries'] = boundaries_info
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
    geo = Geometry('square_2.h5')
    geo.export('square_3.h5')
