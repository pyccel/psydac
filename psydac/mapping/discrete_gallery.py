# coding: utf-8
#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
from sympde.topology.analytical_mapping import IdentityMapping
from sympde.topology.analytical_mapping import PolarMapping, TargetMapping, CzarnyMapping, CollelaMapping2D
from sympde.topology.mapping            import Mapping

from psydac.fem.splines                 import SplineSpace
from psydac.fem.tensor                  import TensorFemSpace
from psydac.mapping.discrete            import SplineMapping
from psydac.ddm.cart                    import DomainDecomposition


class Collela3D( Mapping ):

    expressions = {'x':'2.*(x1 + 0.1**sin(2.*pi*x1)*sin(2.*pi*x2)) - 1.',
                   'y':'2.*(x2 + 0.1**sin(2.*pi*x1)*sin(2.*pi*x2)) - 1.',
                   'z':'2.*x3  - 1.'}

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
            map_analytic = IdentityMapping( 'M', dim=dim )
            lims1   = (0, 1)
            lims2   = (0, 1)
            period1 = False
            period2 = False

        elif mapping == 'collela':
            default_params = dict( k1=1.0, k2=1.0, eps=0.1 )
            map_analytic = CollelaMapping2D( 'M', dim=dim, **default_params )
            lims1   = (0, 1)
            lims2   = (0, 1)
            period1 = False
            period2 = False

        elif mapping == 'circle':
            default_params = dict( rmin=0.0, rmax=1.0, c1=0.0, c2=0.0)
            map_analytic = PolarMapping( 'M', dim=dim, **default_params )
            lims1   = (0, 1)
            lims2   = (0, 2*np.pi)
            period1 = False
            period2 = True

        elif mapping == 'annulus':
            default_params = dict( rmin=0.0, rmax=1.0, c1=0.0, c2=0.0)
            map_analytic = PolarMapping( 'M', dim=dim, **default_params )
            lims1   = (1, 4)
            lims2   = (0, 2*np.pi)
            period1 = False
            period2 = True

        elif mapping == 'quarter_annulus':
            default_params = dict( rmin=0.0, rmax=1.0, c1=0.0, c2=0.0)
            map_analytic = PolarMapping( 'M', dim=dim, **default_params )
            lims1   = (1, 4)
            lims2   = (0, np.pi/2)
            period1 = False
            period2 = False

        elif mapping == 'target':
            default_params = dict( c1=0, c2=0, k=0.3, D=0.2 )
            map_analytic = TargetMapping( 'M', dim=dim, **default_params )
            lims1   = (0, 1)
            lims2   = (0, 2*np.pi)
            period1 = False
            period2 = True

        elif mapping == 'czarny':
            default_params = dict( c2=0, b=1.4, eps=0.3 )
            map_analytic = CzarnyMapping( 'M', dim=dim, **default_params )
            lims1   = (0, 1)
            lims2   = (0, 2*np.pi)
            period1 = False
            period2 = True
        else:
            raise ValueError("Required 2D mapping not available")

        p1 , p2  = degree
        nc1, nc2 = ncells

        # Create the domain decomposition
        domain_decomposition = DomainDecomposition(ncells=[nc1,nc2], periods=[period1,period2], comm=comm)

        # Create tensor spline space, distributed
        V1    = SplineSpace( grid=np.linspace( *lims1, num=nc1+1 ), degree=p1, periodic=period1 )
        V2    = SplineSpace( grid=np.linspace( *lims2, num=nc2+1 ), degree=p2, periodic=period2 )
        space = TensorFemSpace( domain_decomposition, V1, V2 )

        # Create spline mapping by interpolating analytical one
        map_discrete = SplineMapping.from_mapping( space, map_analytic )

    elif dim == 3:
        # Input parameters
        if mapping == 'identity':
            map_analytic = IdentityMapping( 'M', dim=dim )
            lims1   = (0, 1)
            lims2   = (0, 1)
            lims3   = (0, 1)
            period1 = False
            period2 = False
            period3 = False

        elif mapping == 'collela':
            map_analytic = Collela3D( 'M', dim=dim )
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

        # Create the domain decomposition
        domain_decomposition = DomainDecomposition(ncells=[nc1,nc2,nc3], periods=[period1,period2,period3], comm=comm)

        # Create tensor spline space, distributed
        V1    = SplineSpace( grid=np.linspace( *lims1, num=nc1+1 ), degree=p1, periodic=period1 )
        V2    = SplineSpace( grid=np.linspace( *lims2, num=nc2+1 ), degree=p2, periodic=period2 )
        V3    = SplineSpace( grid=np.linspace( *lims3, num=nc3+1 ), degree=p3, periodic=period3 )
        space = TensorFemSpace( domain_decomposition, V1, V2, V3 )

        # Create spline mapping by interpolating analytical one
        map_discrete = SplineMapping.from_mapping( space, map_analytic )
    # ...

    if return_space:
        return map_discrete, space

    else:
        return map_discrete
