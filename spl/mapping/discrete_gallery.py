# coding: utf-8
#!/usr/bin/env python

import numpy as np
from mpi4py import MPI

from spl.fem.splines                import SplineSpace
from spl.fem.tensor                 import TensorFemSpace
from spl.mapping.discrete           import SplineMapping
from spl.mapping.analytical         import IdentityMapping
from spl.mapping.analytical_gallery import Annulus, Target, Czarny, Collela
from spl.mapping.analytical_gallery import Collela3D

#==============================================================================
def discrete_mapping(mapping, ncells, degree, return_space=False):
    mapping = mapping.lower()

    dim = len(ncells)
    if not( dim in [2,3] ):
        raise NotImplementedError('only 2d, 3d are available')

    # ...
    if dim == 2:
        # Input parameters
        if mapping == 'identity':
            map_analytic = IdentityMapping( ndim=dim )
            lims1   = (0, 1)
            lims2   = (0, 1)
            period1 = False
            period2 = False

        elif mapping == 'collela':
            map_analytic = Collela( **kwargs )
            lims1   = (0, 1)
            lims2   = (0, 1)
            period1 = False
            period2 = False

        else:
            mapping = mapping.capitalize()
            map_analytic = locals()[mapping]( **kwargs )
            lims1 = (0, 1)
            lims2 = (0, 2*np.pi)
            period1 = False
            period2 = True

        p1 , p2  = degree
        nc1, nc2 = ncells

        # Create tensor spline space, distributed
        V1 = SplineSpace( grid=np.linspace( *lims1, num=nc1+1 ), degree=p1, periodic=period1 )
        V2 = SplineSpace( grid=np.linspace( *lims2, num=nc2+1 ), degree=p2, periodic=period2 )
        space = TensorFemSpace( V1, V2, comm=MPI.COMM_WORLD )

        # Create spline mapping by interpolating analytical one
        map_discrete = SplineMapping.from_mapping( space, map_analytic )

    elif dim == 3:
        # Input parameters
        if mapping == 'identity':
            map_analytic = IdentityMapping( ndim=dim )
            lims1   = (0, 1)
            lims2   = (0, 1)
            lims3   = (0, 1)
            period1 = False
            period2 = False
            period3 = False

        elif mapping == 'collela':
            map_analytic = Collela3D( **kwargs )
            lims1   = (0, 1)
            lims2   = (0, 1)
            lims3   = (0, 1)
            period1 = False
            period2 = False
            period3 = False

        p1 , p2 , p3  = degree
        nc1, nc2, nc3 = ncells

        # Create tensor spline space, distributed
        V1 = SplineSpace( grid=np.linspace( *lims1, num=nc1+1 ), degree=p1, periodic=period1 )
        V2 = SplineSpace( grid=np.linspace( *lims2, num=nc2+1 ), degree=p2, periodic=period2 )
        V3 = SplineSpace( grid=np.linspace( *lims3, num=nc3+1 ), degree=p3, periodic=period3 )
        space = TensorFemSpace( V1, V2, V3, comm=MPI.COMM_WORLD )

        # Create spline mapping by interpolating analytical one
        map_discrete = SplineMapping.from_mapping( space, map_analytic )
    # ...

    if return_space:
        return map_discrete, space

    else:
        return map_discrete
