#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import pytest
import numpy as np

from sympde.topology import Domain, Square
from sympde.topology import PolarMapping
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace

from psydac.fem.basic              import FemField
from psydac.api.discretization     import discretize
from psydac.fem.plotting_utilities import plot_field_2d as plot_field

#==============================================================================
def plot_some_field(Vh):
    uh = FemField(Vh)

    domain  = Vh.symbolic_space.domain
    if Vh.is_multipatch:
        domain_type = 'multi_patch'
    else:
        domain_type = 'single_patch'    
    plot_types = ['amplitude']
    if Vh.is_vector_valued:
        values_type = 'vector'
        plot_types.append('vector_field')
    else:
        values_type = 'scalar'
    for plot_type in plot_types:
        plot_fn=f'uh_{domain_type}_{values_type}_{plot_type}_test.pdf'
        plot_field(fem_field=uh, Vh=Vh, domain=domain, plot_type=plot_type, title='uh', filename=plot_fn, hide_plot=True)

#==============================================================================
@pytest.mark.parametrize('use_scalar_field', [True, False])
@pytest.mark.parametrize('use_multipatch', [True, False])
def test_plot_field(use_scalar_field, use_multipatch):
    """
    tests that plot_field_2d runs for various types of Fem fields
    (the proper content of the plots is not tested here)
    """

    ncells = [7, 7]
    degree = [2, 2]

    A = Square('A',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    D1     = mapping_1(A)
    
    if use_multipatch:
        B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))
        mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
        D2     = mapping_2(B)
        
        connectivity = [((0,1,1),(1,1,-1))]
        patches = [D1,D2]
        domain = Domain.join(patches, connectivity, 'domain')
    else:
        domain = D1

    if use_scalar_field:
        V = ScalarFunctionSpace('V', domain=domain)
    else:
        V = VectorFunctionSpace('V', domain=domain)

    domain_h = discretize(domain, ncells=ncells)
    Vh       = discretize(V, domain_h, degree=degree)

    plot_some_field(Vh)

if __name__ == '__main__':
    for use_scalar_field in [True, False]:
        for use_multipatch in [True, False]:
            test_plot_field(use_scalar_field, use_multipatch)
