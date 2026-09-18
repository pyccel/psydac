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
from psydac.fem.plotting_utilities import plot_field_2d as plot_field, get_patch_knots_gridlines
from psydac.mapping.discrete       import SplineCallableMapping

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
    mapping_1 = PolarMapping('M1', 2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    D1     = mapping_1(A)
    
    if use_multipatch:
        B = Square('B',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))
        mapping_2 = PolarMapping('M2', 2, c1= 0., c2= 0., rmin = 0., rmax=1.)
        D2 = mapping_2(B)
        
        patches = [D1, D2]
        connectivity = [((0, 1, 1), (1, 1,-1), 1)]
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

#==============================================================================
def test_plot_field_spline_discrete_mapping(tmp_path):
    """
    Regression for WP14a (D5): plot_field_2d on a single-patch domain whose
    mapping is a DiscreteMapping wrapping a SplineCallableMapping used to
    raise a TypeError, because get_patch_knots_gridlines calls the mapping
    on a meshgrid.
    """
    rmin, rmax = 0.3, 1.0
    A = Square('A', bounds1=(0., 1.), bounds2=(0., 0.5 * np.pi))
    F = PolarMapping('F', dim=2, c1=0., c2=0., rmin=rmin, rmax=rmax)

    geo_ncells, geo_degree = (8, 8), (3, 3)
    F_h = SplineCallableMapping.from_mapping(
        None, F.get_callable_mapping(), ncells=geo_ncells, degree=geo_degree,
        bounds=zip(A.min_coords, A.max_coords))

    domain = F_h.to_defined_mapping('F')(A)

    domain_h = discretize(domain, ncells=[8, 8])
    V        = ScalarFunctionSpace('V', domain=domain)
    Vh       = discretize(V, domain_h, degree=[2, 2])
    uh       = FemField(Vh)

    N = 3
    gridlines_x1, gridlines_x2 = get_patch_knots_gridlines(Vh, N, domain.mappings, 0)

    F_callable = F_h
    grid_x1 = Vh.patch_spaces[0].spaces[0].breaks
    grid_x2 = Vh.patch_spaces[0].spaces[1].breaks
    from psydac.fem.plotting_utilities import refine_array_1d
    x1 = refine_array_1d(grid_x1, N)
    x2 = refine_array_1d(grid_x2, N)
    x_loop = np.array([[F_callable(a, b)[0] for b in x2] for a in x1])
    y_loop = np.array([[F_callable(a, b)[1] for b in x2] for a in x1])
    # WP14b: get_patch_knots_gridlines calls the mapping on a dense meshgrid,
    # which is fast-tensor-grid-eligible -- ~1e-15 agreement with the
    # per-point loop above, not bitwise-equal (different summation order).
    np.testing.assert_allclose(gridlines_x1[0], x_loop[:, ::N], rtol=0, atol=5e-14)
    np.testing.assert_allclose(gridlines_x1[1], y_loop[:, ::N], rtol=0, atol=5e-14)

    filename = str(tmp_path / 'uh.png')
    plot_field(fem_field=uh, Vh=Vh, domain=domain, title='uh',
               filename=filename, hide_plot=True)
    assert (tmp_path / 'uh.png').exists()

if __name__ == '__main__':
    for use_scalar_field in [True, False]:
        for use_multipatch in [True, False]:
            test_plot_field(use_scalar_field, use_multipatch)
