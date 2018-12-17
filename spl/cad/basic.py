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

#==============================================================================
def random_string( n ):
    chars    = string.ascii_uppercase + string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )


#==============================================================================
class BasicDiscreteDomain(object):
    _ldim     = None
    _pdim     = None

    @property
    def ldim(self):
        return self._ldim

    @property
    def pdim(self):
        return self._pdim

#==============================================================================
class ProductDiscreteDomain(BasicDiscreteDomain):
    def __init__(self, *args):
        assert(all([isinstance(i, BasicDiscreteDomain) for i in args]))
        assert(len(args) > 1)

        self._ldim = sum(i.ldim for i in args)
        self._pdim = self.ldim
        self._domains = args

    @property
    def domains(self):
        return self._domains

#==============================================================================
class Edge(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


#==============================================================================
class Topology(abc.Mapping):
    _boundaries = None

    def __init__(self, data=None, boundaries=None, filename=None):
        # ...
        if data is None:
            data = {}

        else:
            assert( isinstance( data, (dict, OrderedDict)) )

        self._data = data
        # ...

        # ...
        if boundaries is None:
            boundaries = {}

        else:
            assert( isinstance( boundaries, (list, tuple) ) )

        self._boundaries = boundaries
        # ...

        if not( filename is None ):
            self.read(filename)

    @property
    def boundaries(self):
        if not self._boundaries:
            self._boundaries = self.find_boundaries()

        return self._boundaries

    def set_boundaries(self, boundaries):
        self._boundaries = boundaries

    def read(self, filename):
        raise NotImplementedError('TODO')

    def export(self, filename):
        raise NotImplementedError('TODO')

    def find_boundaries(self):
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
        boundaries = [bnd.todict() for bnd in self.boundaries]
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

    def __init__(self, patch, axis, ext, name=None):
        assert( isinstance( axis, int ) and axis in [0,1,2] )
        assert( isinstance( ext,  int ) and ext  in [-1,1] )
        assert( isinstance( patch, _patches_types ) )

        if name is None:
            name = 'bnd_{}'.format(random_string( 4 ))

        else:
            assert( isinstance( name, str ) )

        self._axis  = axis
        self._ext   = ext
        self._patch = patch
        self._name  = name

    @property
    def axis(self):
        return self._axis

    @property
    def ext(self):
        return self._ext

    @property
    def patch(self):
        return self._patch

    @property
    def name(self):
        return self._name

    def todict(self):
        axis  = self.axis
        ext   = self.ext
        patch = self.patch
        name  = self.name
        return OrderedDict( [('axis' , axis       ),
                             ('ext'  , ext        ),
                             ('patch', patch.name ),
                             ('name' , name       ) ] )

    def __eq__(self, other):
        cond = ( (self.patch is other.patch) and
                 (self.axis == other.axis)   and
                 (self.ext  == other.ext)    and
                 (self.name == other.name) )
        return cond


#==============================================================================
_patches_types = (SplineMapping, BasicDiscreteDomain)

