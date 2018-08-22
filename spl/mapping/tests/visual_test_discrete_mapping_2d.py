# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from spl.fem.basic                  import FemField
from spl.fem.splines                import SplineSpace
from spl.fem.tensor                 import TensorFemSpace
from spl.mapping.discrete           import SplineMapping
from spl.mapping.analytical_gallery import *
from spl.utilities.utils            import refine_array_1d

#==============================================================================
# Driver
#==============================================================================
def main( map_analytic=Target(), p1=2, p2=2, nc1=2, nc2=5 ):

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
    # TODO
    return {}

#==============================================================================
# Script functionality
#==============================================================================
if __name__ == '__main__':
    args = parse_input_arguments()
    namespace = main( **args )
