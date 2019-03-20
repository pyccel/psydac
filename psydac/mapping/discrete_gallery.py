# coding: utf-8
#!/usr/bin/env python

import numpy as np
from mpi4py import MPI

from psydac.fem.splines                import SplineSpace
from psydac.fem.tensor                 import TensorFemSpace
from psydac.mapping.discrete           import SplineMapping
from psydac.mapping.analytical         import IdentityMapping
from psydac.mapping.analytical_gallery import Annulus, Target, Czarny, Collela
from psydac.mapping.analytical_gallery import Collela3D

#==============================================================================
def discrete_mapping(mapping, ncells, degree, **kwargs):

    comm         = kwargs.pop('comm', MPI.COMM_WORLD)
    return_space = kwargs.pop('return_space', False)

    mapping = mapping.lower()

    dim = len(ncells)
    if not( dim in [2,3] ):
        raise NotImplementedError('Only 2D and 3D mappings are available')

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

        elif mapping == 'circle':
            map_analytic = Annulus( **kwargs )
            lims1   = (0, 1)
            lims2   = (0, 2*np.pi)
            period1 = False
            period2 = True

        elif mapping == 'annulus':
            map_analytic = Annulus( **kwargs )
            lims1   = (1, 4)
            lims2   = (0, 2*np.pi)
            period1 = False
            period2 = True

        elif mapping == 'quarter_annulus':
            map_analytic = Annulus( **kwargs )
            lims1   = (1, 4)
            lims2   = (0, np.pi/2)
            period1 = False
            period2 = False

        elif mapping in ['target', 'czarny']:
            mapping = mapping.capitalize()
            map_analytic = globals()[mapping]( **kwargs )
            lims1 = (0, 1)
            lims2 = (0, 2*np.pi)
            period1 = False
            period2 = True

        else:
            raise ValueError("Required 2D mapping not available")

        p1 , p2  = degree
        nc1, nc2 = ncells

        # Create tensor spline space, distributed
        V1 = SplineSpace( grid=np.linspace( *lims1, num=nc1+1 ), degree=p1, periodic=period1 )
        V2 = SplineSpace( grid=np.linspace( *lims2, num=nc2+1 ), degree=p2, periodic=period2 )
        space = TensorFemSpace( V1, V2, comm=comm )

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

        else:
            raise ValueError("Required 3D mapping not available")

        p1 , p2 , p3  = degree
        nc1, nc2, nc3 = ncells

        # Create tensor spline space, distributed
        V1 = SplineSpace( grid=np.linspace( *lims1, num=nc1+1 ), degree=p1, periodic=period1 )
        V2 = SplineSpace( grid=np.linspace( *lims2, num=nc2+1 ), degree=p2, periodic=period2 )
        V3 = SplineSpace( grid=np.linspace( *lims3, num=nc3+1 ), degree=p3, periodic=period3 )
        space = TensorFemSpace( V1, V2, V3, comm=comm )

        # Create spline mapping by interpolating analytical one
        map_discrete = SplineMapping.from_mapping( space, map_analytic )
    # ...

    if return_space:
        return map_discrete, space

    else:
        return map_discrete
