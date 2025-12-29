#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt

from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.mapping.discrete import SplineMapping, NurbsMapping
from psydac.utilities.utils  import refine_array_1d

#==============================================================================
def plot_mapping(mapping, N=10):

    V = mapping.space

    assert(isinstance(V, TensorFemSpace))
    assert(V.ldim == 2)

    V1, V2 = V.spaces

    # Compute numerical solution (and error) on refined logical grid
    [sk1, sk2], [ek1, ek2] = V.local_domain

    eta1 = refine_array_1d(V1.breaks[sk1:ek1+2], N)
    eta2 = refine_array_1d(V2.breaks[sk2:ek2+2], N)

    # Compute physical coordinates of logical grid
    pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])
    xx = pcoords[:, :, 0]
    yy = pcoords[:, :, 1]

    # Create figure with 3 subplots:
    fig, ax = plt.subplots(1, 1)

    ax.set_xlabel(r'$x$', rotation='horizontal')
    ax.set_ylabel(r'$y$', rotation='horizontal')
    ax.plot(xx[:, ::N], yy[:, ::N], 'k')
    ax.plot(xx[::N, :].T, yy[::N, :].T, 'k')

    ax.set_aspect('equal')

    # Show figure
    fig.show()
    plt.show()
