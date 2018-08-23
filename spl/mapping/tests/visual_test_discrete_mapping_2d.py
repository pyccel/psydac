# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from mpi4py import MPI

#==============================================================================
# Driver
#==============================================================================
def main( mapping='Target', degree=(2,2), ncells=(2,5), **kwargs ):

    import numpy as np
    import matplotlib.pyplot as plt

    from spl.fem.basic                  import FemField
    from spl.fem.splines                import SplineSpace
    from spl.fem.tensor                 import TensorFemSpace
    from spl.mapping.discrete           import SplineMapping
    from spl.mapping.analytical_gallery import Annulus, Target, Czarny
    from spl.utilities.utils            import refine_array_1d

    # Input parameters
    map_analytic = locals()[mapping]( **kwargs )
    p1 , p2      = degree
    nc1, nc2     = ncells

    # Create tensor spline space, distributed
    V1 = SplineSpace( grid=np.linspace( 0, 1,       nc1+1 ), degree=p1, periodic=False )
    V2 = SplineSpace( grid=np.linspace( 0, 2*np.pi, nc2+1 ), degree=p2, periodic=True  )
    tensor_space = TensorFemSpace( V1, V2, comm=MPI.COMM_WORLD )

    # Create spline mapping by interpolating analytical one
    map_discrete = SplineMapping( tensor_space=tensor_space, mapping=map_analytic )

    # Display analytical and spline mapping on refined grid, then plot error
    N = 20
    r = refine_array_1d( V1.breaks, N )
    t = refine_array_1d( V2.breaks, N )

    shape  = len(r), len(t)
    xa, ya = [np.array( v ).reshape( shape ) for v in zip( *[map_analytic( [ri,tj] ) for ri in r for tj in t] )]
    xd, yd = [np.array( v ).reshape( shape ) for v in zip( *[map_discrete( [ri,tj] ) for ri in r for tj in t] )]

    figtitle  = 'Mapping: {:s}, Degree: [{:d},{:d}], Ncells: [{:d},{:d}]'.format(
        map_analytic.__class__.__name__, p1, p2, nc1, nc2 )

    fig, axes = plt.subplots( 1, 2, figsize=[12,5], num=figtitle )
    for ax in axes:
        ax.set_aspect('equal')

    axes[0].set_title( 'Analytical (black) vs. spline (red)' )
    axes[0].plot( xa[::N,:].T, ya[::N,:].T, 'k' ); axes[0].plot( xa[:,::N]  , ya[:,::N]  , 'k' )
    axes[0].plot( xd[::N,:].T, yd[::N,:].T, 'r' ); axes[0].plot( xd[:,::N]  , yd[:,::N]  , 'r' )

    axes[1].set_title( 'Error (distance)' )
    im = axes[1].contourf( xa, ya, np.sqrt( (xa-xd)**2+(ya-yd)**2) )
    fig.colorbar( im )

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
        description     = 'Interpolate 2D analytical mapping with tensor-product'
                          ' spline and plot results.'
    )

    parser.add_argument( '-m',
        type    = str,
        choices =('Annulus', 'Target', 'Czarny'),
        default = 'Annulus',
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
        default = [2,5],
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
