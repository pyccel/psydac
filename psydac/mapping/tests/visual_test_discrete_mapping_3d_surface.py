# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from mpi4py import MPI

#==============================================================================
# Driver
#==============================================================================
def main( mapping='TwistedTarget', degree=(2,2), ncells=(4,5), **kwargs ):

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from psydac.fem.basic                  import FemField
    from psydac.fem.splines                import SplineSpace
    from psydac.fem.tensor                 import TensorFemSpace
    from psydac.mapping.discrete           import SplineMapping
    from psydac.mapping.analytical_gallery import TwistedTarget, Torus
    from psydac.utilities.utils            import refine_array_1d

    # Input parameters
    map_analytic = locals()[mapping]( **kwargs )
    p1 , p2      = degree
    nc1, nc2     = ncells

    # Limits
    # TODO: read from mapping object
    if mapping == 'TwistedTarget':
        lims1 = (0, 1)
        lims2 = (0, 2*np.pi)
    elif mapping == 'Torus':
        lims1 = (0, 2*np.pi)
        lims2 = (0, 2*np.pi)
    else:
        raise ValueError( 'Unknown limits for given mapping' )

    # Periodicity
    # TODO: read from mapping object
    if mapping == 'TwistedTarget':
        periodic1 = False
        periodic2 = True
    elif mapping == 'Torus':
        periodic1 = True
        periodic2 = True
    else:
        raise ValueError( 'Unknown periodicity for given mapping' )

    # Create tensor spline space, distributed
    V1 = SplineSpace( grid=np.linspace( *lims1, num=nc1+1 ), degree=p1, periodic=periodic1 )
    V2 = SplineSpace( grid=np.linspace( *lims2, num=nc2+1 ), degree=p2, periodic=periodic2 )
    tensor_space = TensorFemSpace( V1, V2, comm=MPI.COMM_WORLD )

    # Create spline mapping by interpolating analytical one
    map_discrete = SplineMapping.from_mapping( tensor_space, map_analytic )

    # Display analytical and spline mapping on refined grid, then plot error
    N = 20
    r = refine_array_1d( V1.breaks, N )
    t = refine_array_1d( V2.breaks, N )

    shape  = len(r), len(t)
    xa, ya, za = [np.array( v ).reshape( shape ) for v in zip( *[map_analytic( [ri,tj] ) for ri in r for tj in t] )]
    xd, yd, zd = [np.array( v ).reshape( shape ) for v in zip( *[map_discrete( [ri,tj] ) for ri in r for tj in t] )]

    figtitle  = 'Mapping: {:s}, Degree: [{:d},{:d}], Ncells: [{:d},{:d}]'.format(
        map_analytic.__class__.__name__, p1, p2, nc1, nc2 )

    fig, ax = plt.subplots( 1, 1, figsize=[5,5], num=figtitle, subplot_kw={'projection':'3d'} )

    ax.set_title( 'Analytical (black) vs. spline (red)' )
    ax.plot_surface  ( xa, ya, za, color='k', alpha=0.2 )
    ax.plot_wireframe( xa, ya, za, color='k', rstride=N, cstride=N )
    ax.plot_wireframe( xd, yd, zd, color='r', rstride=N, cstride=N )

    scaling = np.array( [getattr( ax, 'get_{}lim'.format(dim) )() for dim in 'xyz'] )
    ax.auto_scale_xyz( *[[np.min(scaling), np.max(scaling)]]*3 )

    ax.set_aspect( 'equal' )
    fig.tight_layout()
    fig.show()

    return locals()

#==============================================================================
# Parser
#==============================================================================
def parse_input_arguments():

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = 'Interpolate analytical mapping for 3D surface with'
                          ' 2D tensor-product spline and plot results.'
    )

    parser.add_argument( '-m',
        type    = str,
        choices =('TwistedTarget','Torus'),
        default = 'TwistedTarget',
        dest    = 'mapping',
        help    = 'Analytical mapping'
    )

    parser.add_argument( '-d',
        type    = int,
        nargs   = 2,
        default = [2,2],
        metavar = ('P1','P2'),
        dest    = 'degree',
        help    = 'Spline degree along each dimension'
    )

    parser.add_argument( '-n',
        type    = int,
        nargs   = 2,
        default = [4,5],
        metavar = ('N1','N2'),
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension'
    )

    return parser.parse_args()

#==============================================================================
# Script functionality
#==============================================================================
if __name__ == '__main__':

    args = parse_input_arguments()
    namespace = main( **vars( args ) )

    import __main__
    if hasattr( __main__, '__file__' ):
        try:
           __IPYTHON__
        except NameError:
            import matplotlib.pyplot as plt
            plt.show()
