#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import h5py
import yaml
import numpy as np

from sympde.topology       import Domain, SymbolicMapping
from sympde.topology.basic import Union

from psydac.mapping.discrete   import SplineCallableMapping, NurbsCallableMapping
from psydac.cad.geometry       import logical_ncube
from psydac.fem.partitioning   import connectivity_to_join_tuples


def export_multipatch_nurbs_to_hdf5(filename:str, nurbs:list, connectivity:dict, comm=None ):

    """
    Create the multipatch geometry file using one Igakit NURBS for each patch with the connectivity between the patches.

    Parameters
    ----------

    filename : <str>
        Name of output geometry file, e.g. 'geo.h5'

    nurbs   : list of igakit.nurbs.NURBS
        Igakit NURBS objects, one for each patch

    connectivity : dict
        Connectivity between the patches.
        It takes the form of {(i, j):((axis_i, ext_i),(axis_j, ext_j))} for each item of the dictionary,
        where i,j represent the patch indices

    comm : <MPI.COMM>|None
        Mpi communicator
    """

    import os.path
    import igakit
    assert all(isinstance(n, igakit.nurbs.NURBS) for n in nurbs) 

    extension = os.path.splitext(filename)[-1]
    if not extension == '.h5':
        raise ValueError('> Only h5 extension is allowed for filename')

    yml = {}
    yml['ldim'] = nurbs[0].dim
    yml['pdim'] = nurbs[0].dim

    patches_info = []

    patch_names = ['patch_{}'.format(i) for i in range(len(nurbs))]
    names       = ['{}'.format( patch_name ) for patch_name in patch_names]
    mapping_ids = ['mapping_{}'.format(i) for i in range(len(nurbs))]
    dtypes      = [NurbsCallableMapping.geometry_dtype if not abs(nurb.weights-1).max()<1e-15 else SplineCallableMapping.geometry_dtype for nurb in nurbs]

    patches_info += [{'name': name , 'mapping_id':mapping_id, 'type':dtype} for name,mapping_id,dtype in zip(names, mapping_ids, dtypes)]

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

    if nurbs[0].dim > 3:
        raise NotImplementedError('> nurbs.dim > 3 not implemented')

    patches = []
    # ... topology
    for i,(nurbsi,patch_name) in enumerate(zip(nurbs, patch_names)):
        bounds = [(float(nurbsi.breaks(d)[0]), float(nurbsi.breaks(d)[-1])) for d in range(nurbsi.dim)]
        domain  = logical_ncube(patch_name, [b[0] for b in bounds], [b[1] for b in bounds])
        mapping = SymbolicMapping(mapping_ids[i], dim=nurbs[0].dim)
        patches.append(mapping(domain))

    interfaces = connectivity_to_join_tuples(connectivity)

    domain = Domain.join(patches, interfaces, filename[:-3])
    topo_yml = domain.todict()

    # Dump geometry metadata to string in YAML file format
    geom = yaml.dump( data   = topo_yml, sort_keys=False)
    # Write topology metadata as fixed-length array of ASCII characters
    h5['topology.yml'] = np.array( geom, dtype='S' )

    for i in range(len(nurbs)):
        nurbsi   = nurbs[i]
        dtype    = dtypes[i]
        rational = dtype == NurbsCallableMapping.geometry_dtype
        group = h5.create_group( yml['patches'][i]['mapping_id'] )
        group.attrs['degree'     ] = nurbsi.degree
        group.attrs['rational'   ] = rational
        group.attrs['periodic'   ] = tuple( False for d in range( nurbsi.dim ) )

        for d in range( nurbsi.dim ):
            group['knots_{}'.format( d )] = nurbsi.knots[d]

        group['points'] = nurbsi.points[...,:nurbsi.dim]
        if rational:
            group['weights'] = nurbsi.weights

    h5.close()
