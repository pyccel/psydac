#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from typing import Iterable

import numpy as np
from mpi4py import MPI

from sympde.topology.mapping import Mapping
from sympde.topology.analytical_mapping import (IdentityMapping, PolarMapping,
                                                TargetMapping, CzarnyMapping,
                                                CollelaMapping2D, SphericalMapping)

from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.mapping.discrete import SplineMapping
from psydac.ddm.cart         import DomainDecomposition

#==============================================================================
available_mappings_2d = (
    'identity',
    'collela',
    'circle',
    'annulus',
    'quarter_annulus',
    'target',
    'czarny',
)

available_mappings_3d = (
    'identity',
    'collela',
    'spherical_shell',
)

__all__ = (
    'get_available_mappings',
    'discrete_mapping',
)

class Collela3D( Mapping ):

    _expressions = {'x':'2.*(x1 + 0.1*sin(2.*pi*x1)*sin(2.*pi*x2)) - 1.',
                    'y':'2.*(x2 + 0.1*sin(2.*pi*x1)*sin(2.*pi*x2)) - 1.',
                    'z':'2.*x3  - 1.'}

#==============================================================================
def get_available_mappings(ldim):
    """
    Get a list of `mapping` values accepted as argument to `discrete_mapping`.

    Parameters
    ----------
    ldim : int
        The number of logical dimensions of the topological domain.

    Returns
    -------
    tuple
        All the accepted values for the `mapping` parameter.
    """
    assert isinstance(ldim, int), f'ldim must be int, got {type(ldim).__name__} instead'
    assert ldim > 0, f'ldim must be > 0, got {ldim} instead'

    if ldim == 2:
        return ('identity', 'collela', 'circle', 'annulus', 'quarter_annulus',
                'target', 'czarny')
    elif ldim == 3:
        return ('identity', 'collela', 'spherical_shell')
    else:
        return ()

#==============================================================================
def discrete_mapping(mapping, ncells, degree, *,
                     comm = MPI.COMM_WORLD,
                     return_space = False):
    """
    Create a SplineMapping by interpolating one of the available analytical mappings.

    Parameters
    ----------
    mapping : str
        The name of the mapping. See `available_mappings` to get the options.

    ncells : Iterable[int]
        The number of cells along each logical dimension.

    degree : Iterable[int]
        The spline degree along each logical dimension.

    comm : MPI.Intracomm, optional
        The MPI intracommunicator.

    return_space : bool, optional
        Whether this function should also return the discrete space it creates.

    Returns
    -------
    map_discrete : SplineMapping
        The spline mapping created.

    space : TensorFemSpace
        The space of the components of the spline mapping.
        Only returned if `return_space` is True.
    """
    # Check types
    assert isinstance(mapping, str)
    assert isinstance(ncells, Iterable)
    assert isinstance(degree, Iterable)
    assert isinstance(comm, MPI.Intracomm) or comm is None
    assert isinstance(return_space, bool)

    # Check consistency of ncells and degree
    assert all(isinstance(n, int) and n >= 1 for n in ncells)
    assert all(isinstance(d, int) and d >= 0 for d in degree)
    assert len(ncells) == len(degree)

    mapping = mapping.lower()

    ldim = len(ncells)
    if ldim not in [2, 3]:
        raise NotImplementedError('Only 2D and 3D mappings are available')

    # ...
    if ldim == 2:
        # Input parameters
        if mapping == 'identity':
            map_symbolic = IdentityMapping('M', dim=ldim)
            limits   = ((0, 1), (0, 1))
            periodic =  (False, False)

        elif mapping == 'collela':
            default_params = dict(k1=1.0, k2=1.0, eps=0.1)
            map_symbolic = CollelaMapping2D('M', dim=ldim, **default_params)
            limits   = ((0, 1), (0, 1))
            periodic =  (False, False)

        elif mapping == 'circle':
            default_params = dict(rmin=0.0, rmax=1.0, c1=0.0, c2=0.0)
            map_symbolic = PolarMapping('M', dim=ldim, **default_params)
            limits   = ((0, 1), (0, 2*np.pi))
            periodic =  (False, True)

        elif mapping == 'annulus':
            default_params = dict(rmin=0.0, rmax=1.0, c1=0.0, c2=0.0)
            map_symbolic = PolarMapping('M', dim=ldim, **default_params)
            limits   = ((1, 4), (0, 2*np.pi))
            periodic =  (False, True)

        elif mapping == 'quarter_annulus':
            default_params = dict(rmin=0.0, rmax=1.0, c1=0.0, c2=0.0)
            map_symbolic = PolarMapping('M', dim=ldim, **default_params)
            limits   = ((1, 4), (0, np.pi/2))
            periodic =  (False, False)

        elif mapping == 'target':
            default_params = dict(c1=0, c2=0, k=0.3, D=0.2)
            map_symbolic = TargetMapping('M', dim=ldim, **default_params)
            limits   = ((0, 1), (0, 2*np.pi))
            periodic =  (False, True)

        elif mapping == 'czarny':
            default_params = dict(c2=0, b=1.4, eps=0.3)
            map_symbolic = CzarnyMapping('M', dim=ldim, **default_params)
            limits   = ((0, 1), (0, 2*np.pi))
            periodic =  (False, True)

        else:
            raise ValueError("Required 2D mapping not available")

    elif ldim == 3:
        # Input parameters
        if mapping == 'identity':
            map_symbolic = IdentityMapping('M', dim=ldim)
            limits   = ((0, 1), (0, 1), (0, 1))
            periodic = ( False,  False,  False)

        elif mapping == 'collela':
            map_symbolic = Collela3D('M', dim=ldim)
            limits   = ((0, 1), (0, 1), (0, 1))
            periodic = ( False,  False,  False)

        elif mapping == 'spherical_shell':
            map_symbolic = SphericalMapping('M', dim=ldim)
            limits   = ((1, 4), (0, np.pi), (0, np.pi/2))
            periodic = ( False,  False,  False)

        else:
            raise ValueError("Required 3D mapping not available")

    # ...

    # Create the domain decomposition
    domain_decomposition = DomainDecomposition(ncells=ncells, periods=periodic, comm=comm)

    # Create 1D spline spaces, not distributed
    spaces_1d = [SplineSpace(grid=np.linspace(*lims, num=nc+1), degree=p, periodic=per)
                 for lims, nc, p, per in zip(limits, ncells, degree, periodic)]

    # Create tensor spline space, distributed
    space = TensorFemSpace(domain_decomposition, *spaces_1d)

    # Create spline mapping by interpolating analytical one
    map_analytic = map_symbolic.get_callable_mapping()
    map_discrete = SplineMapping.from_mapping(space, map_analytic)

    if return_space:
        return map_discrete, space
    else:
        return map_discrete
