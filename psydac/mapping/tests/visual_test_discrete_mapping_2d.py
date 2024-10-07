# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

#==============================================================================
# Driver
#==============================================================================
def main(*, mapping, degree, ncells, name='F'):

    from mpi4py import MPI
    import numpy as np
    import matplotlib.pyplot as plt

    from sympde.topology import IdentityMapping,TargetMapping, CzarnyMapping, PolarMapping, CollelaMapping2D

    from psydac.mapping.discrete import SplineMapping
    from psydac.fem.basic        import FemField
    from psydac.fem.splines      import SplineSpace
    from psydac.fem.tensor       import TensorFemSpace
    from psydac.ddm.cart         import DomainDecomposition
    from psydac.utilities.utils  import refine_array_1d

    # Input parameters
    if mapping == 'Identity':
        map_analytic = IdentityMapping(name, dim=2)
        lims1   = (0, 1)
        lims2   = (0, 1)
        periodic1 = False
        periodic2 = False

    elif mapping == 'Collela':
        map_analytic = CollelaMapping2D(name, eps=0.05, k1=1, k2=1)
        lims1   = (0, 1)
        lims2   = (0, 1)
        periodic1 = False
        periodic2 = False

    elif mapping == 'Polar':
        map_analytic = PolarMapping(name, c1=0.1, c2=0.1, rmin=0.1, rmax=1)
        lims1 = (0, 1)
        lims2 = (0, 2*np.pi)
        periodic1 = False
        periodic2 = True

    elif mapping == 'Target':
        map_analytic = TargetMapping(name, c1=0.0, c2=0.0, k=0.3, D=0.1)
        lims1 = (0, 1)
        lims2 = (0, 2*np.pi)
        periodic1 = False
        periodic2 = True

    elif mapping == 'Czarny':
        map_analytic = CzarnyMapping(name, eps=0.3, c2=0, b=2)
        lims1 = (0, 1)
        lims2 = (0, 2*np.pi)
        periodic1 = False
        periodic2 = True

    else:
        raise ValueError(f'Test case does not support mapping "{mapping}"')


    p1 , p2  = degree
    nc1, nc2 = ncells

    # Create 1D spline spaces along x1 and x2
    V1 = SplineSpace( grid=np.linspace(*lims1, num=nc1+1), degree=p1, periodic=periodic1 )
    V2 = SplineSpace( grid=np.linspace(*lims2, num=nc2+1), degree=p2, periodic=periodic2 )

    # Create tensor-product 2D spline space, distributed
    domain_decomposition = DomainDecomposition([nc1, nc2], [periodic1, periodic2], comm=MPI.COMM_WORLD)
    tensor_space = TensorFemSpace(domain_decomposition, V1, V2)

    # Create spline mapping by interpolating analytical one
    map_discrete = SplineMapping.from_mapping(tensor_space, map_analytic)

    # Display analytical and spline mapping on refined grid, then plot error
    N = 20
    r = refine_array_1d(V1.breaks, N)
    t = refine_array_1d(V2.breaks, N)

#    shape  = len(r), len(t)
#    xa, ya = [np.array(v).reshape(shape) for v in zip(*[map_analytic(ri, tj) for ri in r for tj in t])]
#    xd, yd = [np.array(v).reshape(shape) for v in zip(*[map_discrete(ri, tj) for ri in r for tj in t])]

    xa, ya = map_analytic(*np.meshgrid(r, t, indexing='ij'))
    xd, yd = map_discrete.build_mesh([r, t])

    figtitle = 'Mapping: {:s}, Degree: [{:d},{:d}], Ncells: [{:d},{:d}]'.format(
        map_analytic.__class__.__name__, p1, p2, nc1, nc2)

    fig, axes = plt.subplots(1, 2, figsize=[12,5], num=figtitle)
    for ax in axes:
        ax.set_aspect('equal')

    axes[0].set_title( 'Analytical (black) vs. spline (red)' )
    axes[0].plot(xa[::N,:].T, ya[::N,:].T, 'k'); axes[0].plot(xa[:,::N], ya[:,::N], 'k')
    axes[0].plot(xd[::N,:].T, yd[::N,:].T, 'r'); axes[0].plot(xd[:,::N], yd[:,::N], 'r')

    axes[1].set_title('Error (distance)')
    im = axes[1].contourf(xa, ya, np.sqrt((xa-xd)**2+(ya-yd)**2))
    fig.colorbar(im)

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

    parser.add_argument('-m',
        type    = str,
        choices =('Identity', 'Polar', 'Target', 'Czarny', 'Collela'),
        default = 'Polar',
        dest    = 'mapping',
        help    = 'Analytical mapping'
    )

    parser.add_argument('-d',
        type    = int,
        nargs   = 2,
        default = [2, 2],
        metavar = ('P1', 'P2'),
        dest    = 'degree',
        help    = 'Spline degree along each dimension'
    )

    parser.add_argument('-n',
        type    = int,
        nargs   = 2,
        default = [2, 5],
        metavar = ('N1', 'N2'),
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension'
    )

    return parser.parse_args()

#==============================================================================
# Script functionality
# Usage:
#     python3 psydac/mapping/tests/visual_test_discrete_mapping_2d.py -m Annulus -d 2 2 -n 4 10
#==============================================================================
if __name__ == '__main__':

    args = parse_input_arguments()
    namespace = main(**vars(args))

    import __main__
    if hasattr(__main__, '__file__'):
        try:
           __IPYTHON__
        except NameError:
            import matplotlib.pyplot as plt
            plt.show()
