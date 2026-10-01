#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
from pathlib import Path

import pytest
import numpy as np
import h5py as h5

from igakit.cad import circle, ruled

from sympde.topology import Domain

from psydac.api.discretization import discretize
from psydac.core.bsplines import cell_index
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.splines import SplineSpace
from psydac.mapping.discrete import NurbsCallableMapping
from psydac.utilities.utils import refine_array_1d
from psydac.ddm.cart        import DomainDecomposition

# Get the mesh directory
import psydac.cad.mesh as mesh_mod
mesh_dir = Path(mesh_mod.__file__).parent

# Tolerance for testing float equality
RTOL = 1e-15
ATOL = 1e-15

#==============================================================================
@pytest.mark.parametrize('geometry_file', ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_per_cell', [2, 3, 4])
def test_build_mesh_reg(geometry_file, npts_per_cell):
    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)
    domain_h = discretize(domain, filename=filename)
    for mapping in domain_h.mappings.values():
        space = mapping.space

        grid = [refine_array_1d(space.breaks[i], npts_per_cell - 1, remove_duplicates=False) for i in range(mapping.ldim)]


        if mapping.ldim == 2:
            x_mesh, y_mesh = mapping.build_mesh(grid, npts_per_cell=npts_per_cell)

            eta1, eta2 = grid

            pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]

        elif mapping.ldim == 3:

            eta1, eta2, eta3 = grid
            x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid, npts_per_cell=npts_per_cell)
            pcoords = np.array([[[mapping(e1, e2, e3) for e3 in eta3] for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]
            z_mesh_l = pcoords[..., 2]

        else:
            assert False

        assert x_mesh.flags['C_CONTIGUOUS'] and y_mesh.flags['C_CONTIGUOUS']

        assert np.allclose(x_mesh, x_mesh_l, atol=ATOL, rtol=RTOL)
        assert np.allclose(y_mesh, y_mesh_l, atol=ATOL, rtol=RTOL)
        if mapping.ldim == 3:
            assert  z_mesh.flags['C_CONTIGUOUS']
            assert np.allclose(z_mesh, z_mesh_l, atol=ATOL, rtol=RTOL)

#==============================================================================
@pytest.mark.parametrize('geometry_file', ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_i', [2, 5, 10, 25])
def test_build_mesh_i(geometry_file, npts_i):
    filename = os.path.join(mesh_dir, geometry_file)

    domain = Domain.from_file(filename)
    domain_h = discretize(domain, filename=filename)

    for mapping in domain_h.mappings.values():
        space = mapping.space

        grid = [np.linspace(space.breaks[i][0], space.breaks[i][-1], npts_i) for i in range(mapping.ldim)]


        if mapping.ldim == 2:
            x_mesh, y_mesh = mapping.build_mesh(grid)

            eta1, eta2 = grid

            pcoords = np.array([[mapping(e1, e2) for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]

        elif mapping.ldim == 3:
            x_mesh, y_mesh, z_mesh = mapping.build_mesh(grid)

            eta1, eta2, eta3 = grid

            pcoords = np.array([[[mapping(e1, e2, e3) for e3 in eta3] for e2 in eta2] for e1 in eta1])

            x_mesh_l = pcoords[..., 0]
            y_mesh_l = pcoords[..., 1]
            z_mesh_l = pcoords[..., 2]

        else:
            assert False

        assert x_mesh.flags['C_CONTIGUOUS'] and y_mesh.flags['C_CONTIGUOUS']

        assert np.allclose(x_mesh, x_mesh_l, atol=ATOL, rtol=RTOL)
        assert np.allclose(y_mesh, y_mesh_l, atol=ATOL, rtol=RTOL)
        if mapping.ldim == 3:
            assert  z_mesh.flags['C_CONTIGUOUS']
            assert np.allclose(z_mesh, z_mesh_l, atol=ATOL, rtol=RTOL)

#==============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize('geometry',  ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_per_cell', [2, 3, 4, 6])
def test_parallel_jacobians_regular(geometry, npts_per_cell):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domain_h = discretize(domain, filename=filename, comm=comm)

    mapping = list(domain_h.mappings.values())[0]

    space = mapping.space

    # Regular tensor grid.
    grid_reg = [refine_array_1d(space.breaks[i], npts_per_cell - 1, False) for i in range(space.ldim)]

    npts_per_cell = [npts_per_cell] * space.ldim

    jacobian_matrix_p = mapping.jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
    inv_jacobian_matrix_p = mapping.inv_jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
    jacobian_determinants_p = mapping.jac_det_grid(grid_reg, npts_per_cell=npts_per_cell)

    starts, ends = space.local_domain

    actual_starts = tuple(npts_per_cell[0] * s for s in starts)
    actual_ends   = tuple(npts_per_cell[0] * (e + 1) for e in ends)

    index = tuple(slice(s, e, 1) for s,e in zip(actual_starts, actual_ends))

    shape_0 = tuple(len(grid_reg[i]) for i in range(space.ldim)) + (space.ldim, space.ldim)
    shape_1 = tuple(len(grid_reg[i]) for i in range(space.ldim))

    # Saving in an hdf5 file to compare on root
    fh5 = h5.File(f'result_parallel.h5', mode='w', driver='mpio', comm=comm)

    fh5.create_dataset('jac_mat', shape=shape_0, dtype=float)
    fh5.create_dataset('inv_jac', shape=shape_0, dtype=float)
    fh5.create_dataset('jac_dets', shape=shape_1, dtype=float)

    fh5['jac_mat'][index] = jacobian_matrix_p
    fh5['inv_jac'][index] = inv_jacobian_matrix_p
    fh5['jac_dets'][index] = jacobian_determinants_p
    fh5.close()

    # Check
    if rank == 0:
        domain_h = discretize(domain, filename=filename, comm=None)
        mapping = list(domain_h.mappings.values())[0]

        space = mapping.space

        jacobian_matrix = mapping.jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
        inv_jacobian_matrix = mapping.inv_jac_mat_grid(grid_reg, npts_per_cell=npts_per_cell)
        jacobian_determinants = mapping.jac_det_grid(grid_reg, npts_per_cell=npts_per_cell)

        fh5 = h5.File(f'result_parallel.h5', mode='r')

        jac_mat_par = fh5['jac_mat'][...]
        inv_jac_mat_par = fh5['inv_jac'][...]
        jac_dets_par = fh5['jac_dets'][...]

        assert np.allclose(jac_mat_par, jacobian_matrix, atol=ATOL, rtol=RTOL)
        assert np.allclose(inv_jac_mat_par, inv_jacobian_matrix, atol=ATOL, rtol=RTOL)
        assert np.allclose(jac_dets_par, jacobian_determinants, atol=ATOL, rtol=RTOL)

        fh5.close()
        os.remove('result_parallel.h5')

#==============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize('geometry',  ['collela_3d.h5', 'collela_2d.h5', 'bent_pipe.h5'])
@pytest.mark.parametrize('npts_irregular', [2, 5, 10, 25])
def test_parallel_jacobians_irregular(geometry, npts_irregular):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domain_h = discretize(domain, filename=filename, comm=comm)

    mapping = list(domain_h.mappings.values())[0]

    space = mapping.space
    # Irregular tensor grid
    grid_i = [np.linspace(space.breaks[i][0], space.breaks[i][-1], npts_irregular, True) for i in range(space.ldim)]

    jacobian_matrix_p = mapping.jac_mat_grid(grid_i)
    inv_jacobian_matrix_p = mapping.inv_jac_mat_grid(grid_i)
    jacobian_determinants_p = mapping.jac_det_grid(grid_i)

    cell_indexes = [cell_index(space.breaks[i], grid_i[i]) for i in range(space.ldim)]

    starts, ends = space.local_domain

    actual_starts = tuple(np.searchsorted(cell_indexes[i], starts[i], side='left')
                          for i in range(space.ldim))
    actual_ends = tuple(np.searchsorted(cell_indexes[i], ends[i], side='right')
                         for i in range(space.ldim))

    index = tuple(slice(s, e, 1) for s,e in zip(actual_starts, actual_ends))

    shape_0 = tuple(len(grid_i[i]) for i in range(space.ldim)) + (space.ldim, space.ldim)
    shape_1 = tuple(len(grid_i[i]) for i in range(space.ldim))

    # Saving in an hdf5 file to compare on root
    fh5 = h5.File(f'result_parallel.h5', mode='w', driver='mpio', comm=comm)

    fh5.create_dataset('jac_mat', shape=shape_0, dtype=float)
    fh5.create_dataset('inv_jac', shape=shape_0, dtype=float)
    fh5.create_dataset('jac_dets', shape=shape_1, dtype=float)

    fh5['jac_mat'][index] = jacobian_matrix_p
    fh5['inv_jac'][index] = inv_jacobian_matrix_p
    fh5['jac_dets'][index] = jacobian_determinants_p
    fh5.close()

    # Check
    if rank == 0:
        domain_h = discretize(domain, filename=filename, comm=None)
        mapping = list(domain_h.mappings.values())[0]

        space = mapping.space

        jacobian_matrix = mapping.jac_mat_grid(grid_i)
        inv_jacobian_matrix = mapping.inv_jac_mat_grid(grid_i)
        jacobian_determinants = mapping.jac_det_grid(grid_i)

        fh5 = h5.File(f'result_parallel.h5', mode='r')

        jac_mat_par = fh5['jac_mat'][...]
        inv_jac_mat_par = fh5['inv_jac'][...]
        jac_dets_par = fh5['jac_dets'][...]

        assert np.allclose(jac_mat_par, jacobian_matrix, atol=ATOL, rtol=RTOL)
        assert np.allclose(inv_jac_mat_par, inv_jacobian_matrix, atol=ATOL, rtol=RTOL)
        assert np.allclose(jac_dets_par, jacobian_determinants, atol=ATOL, rtol=RTOL)

        fh5.close()
        os.remove('result_parallel.h5')

#==============================================================================
def test_nurbs_circle():
    rmin, rmax = 0.2, 1
    c1, c2 = 0, 0

    # Igakit
    c_ext = circle(radius=rmax, center=(c1, c2))
    c_int = circle(radius=rmin, center=(c1, c2))

    disk = ruled(c_ext, c_int).transpose()

    w  = disk.weights
    k = disk.knots
    control = disk.points
    d = disk.degree

    # PSYDAC
    spaces = [SplineSpace(degree, knot) for degree, knot in zip(d, k)]

    ncells = [len(space.breaks)-1 for space in spaces]
    periods = [space.periodic for space in spaces]

    domain_decomposition = DomainDecomposition(ncells=ncells, periods=periods, comm=None)
    T = TensorFemSpace(domain_decomposition, *spaces)
    mapping = NurbsCallableMapping.from_control_points_weights(T, control_points=control[..., :2], weights=w)

    x1_pts = np.linspace(0, 1, 10)
    x2_pts = np.linspace(0, 1, 10)

    for x2 in x2_pts:
        for x1 in x1_pts:
            x_p, y_p = mapping(x1, x2)
            x_i, y_i, z_i = disk(x1, x2)

            assert np.allclose((x_p, y_p), (x_i, y_i), atol=ATOL, rtol=RTOL)

            J_p = mapping.jacobian(x1, x2)
            J_i = disk.gradient(u=x1, v=x2)

            assert np.allclose(J_i[:2], J_p, atol=ATOL, rtol=RTOL)

#==============================================================================
def test_spline_callable_mapping_is_not_a_defined_mapping():
    # WP12/D1: the WP04 registration of SplineCallableMapping/NurbsCallableMapping
    # as virtual subclasses of sympde's DefinedMapping was removed. A spline is a
    # bare BasicCallableMapping (its real base) with no symbolic identity; use
    # to_defined_mapping(name) to wrap it in a DiscreteMapping.
    from sympde.topology.mapping import DefinedMapping, SymbolicMapping, BasicCallableMapping
    from psydac.mapping.discrete import SplineCallableMapping

    assert issubclass(SplineCallableMapping, BasicCallableMapping)
    assert issubclass(NurbsCallableMapping, BasicCallableMapping)
    assert not issubclass(SplineCallableMapping, DefinedMapping)
    assert not issubclass(NurbsCallableMapping, DefinedMapping)

    rmin, rmax = 0.2, 1
    c_ext = circle(radius=rmax, center=(0, 0))
    c_int = circle(radius=rmin, center=(0, 0))
    disk  = ruled(c_ext, c_int).transpose()

    spaces = [SplineSpace(degree, knot) for degree, knot in zip(disk.degree, disk.knots)]
    ncells  = [len(space.breaks) - 1 for space in spaces]
    periods = [space.periodic for space in spaces]
    domain_decomposition = DomainDecomposition(ncells=ncells, periods=periods, comm=None)
    T = TensorFemSpace(domain_decomposition, *spaces)
    mapping = NurbsCallableMapping.from_control_points_weights(
        T, control_points=disk.points[..., :2], weights=disk.weights)

    assert isinstance(mapping, BasicCallableMapping)
    assert not isinstance(mapping, DefinedMapping)
    assert not isinstance(mapping, SymbolicMapping)

    # the supported route to a symbolic identity
    G = mapping.to_defined_mapping('M')
    assert isinstance(G, DefinedMapping)
    assert G.get_callable_mapping() is mapping
    assert G.is_analytical is False

#==============================================================================
def test_psydac_analytic_gallery_classes_are_analytic_mappings():
    # WP06d-1: every analytic `class X(Mapping)` defined in psydac is re-parented
    # onto AnalyticMapping (like WP02b's sympde gallery classes), so
    # get_callable_mapping() returns self and nothing routes through the
    # deprecated CallableMapping wrapper.
    import warnings
    from sympde.topology.mapping import AnalyticMapping
    from psydac.mapping.discrete_gallery import Collela3D, discrete_mapping
    from psydac.feec.multipatch_domain_utilities import TransposedPolarMapping

    for cls in (Collela3D, TransposedPolarMapping):
        assert issubclass(cls, AnalyticMapping)

    F = Collela3D('M', dim=3)
    assert F.get_callable_mapping() is F

    # the 3D collela path used to build a CallableMapping (DeprecationWarning);
    # after re-parenting it must not.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        discrete_mapping('collela', ncells=[6, 6, 6], degree=[2, 2, 2])
    assert not any('CallableMapping' in str(w.message) for w in caught), \
        [str(w.message) for w in caught]

#==============================================================================
def test_basiccallablemapping_name_stays_importable():
    # WP06d-3: BasicCallableMapping is NOT deleted -- SplineCallableMapping's literal
    # base, the base for plain user callables, set_callable_mapping()'s guard,
    # and imported by downstream (struphy). Both it and DefinedMapping must stay
    # importable from both module paths.
    from sympde.topology.mapping import BasicCallableMapping as BCM_m, DefinedMapping as DM_m
    from sympde.topology.callable_mapping import BasicCallableMapping as BCM_cm
    from sympde.topology import BasicCallableMapping as BCM_pkg
    from psydac.mapping.discrete import SplineCallableMapping

    assert BCM_m is BCM_cm is BCM_pkg
    assert issubclass(SplineCallableMapping, BCM_m)          # literal base
    assert not issubclass(SplineCallableMapping, DM_m)       # WP12/D1: registration removed
    assert issubclass(DM_m, BCM_m)

#==============================================================================
def test_legacy_spline_mapping_names_are_deprecated_aliases():
    # WP11: 'SplineMapping'/'NurbsMapping' are kept as module-level identity
    # aliases (PEP 562 __getattr__), not subclasses -- a subclass alias would
    # make isinstance(obj, SplineMapping) False for objects built via the new
    # name (the WP05 revert failure mode).
    import psydac.mapping.discrete
    from psydac.mapping.discrete import SplineCallableMapping

    with pytest.warns(DeprecationWarning, match="deprecated"):
        from psydac.mapping.discrete import SplineMapping
    with pytest.warns(DeprecationWarning, match="deprecated"):
        from psydac.mapping.discrete import NurbsMapping

    assert SplineMapping is SplineCallableMapping
    assert NurbsMapping is NurbsCallableMapping
    assert SplineMapping is not None and SplineMapping.__mro__ == SplineCallableMapping.__mro__

    rmin, rmax = 0.2, 1
    c_ext = circle(radius=rmax, center=(0, 0))
    c_int = circle(radius=rmin, center=(0, 0))
    disk  = ruled(c_ext, c_int).transpose()

    spaces = [SplineSpace(degree, knot) for degree, knot in zip(disk.degree, disk.knots)]
    ncells  = [len(space.breaks) - 1 for space in spaces]
    periods = [space.periodic for space in spaces]
    domain_decomposition = DomainDecomposition(ncells=ncells, periods=periods, comm=None)
    T = TensorFemSpace(domain_decomposition, *spaces)
    mapping = NurbsCallableMapping.from_control_points_weights(
        T, control_points=disk.points[..., :2], weights=disk.weights)
    assert isinstance(mapping, SplineMapping)
    assert isinstance(mapping, NurbsMapping)

    with pytest.raises(AttributeError):
        _ = psydac.mapping.discrete.NoSuchMapping

    assert 'SplineCallableMapping' in psydac.mapping.discrete.__all__
    assert 'SplineMapping' not in psydac.mapping.discrete.__all__

#==============================================================================
def test_from_mapping_builds_tensor_space_from_grid_parameters():
    # from_mapping(mapping, ncells=..., degree=...) builds the
    # TensorFemSpace itself, instead of requiring the caller to hand-assemble
    # a SplineSpace/DomainDecomposition/TensorFemSpace first -- must agree
    # exactly with the manual construction it replaces.
    from sympde.topology.mapping import AnalyticMapping
    from psydac.mapping.discrete import SplineCallableMapping

    class Collela2D(AnalyticMapping):
        _expressions = {'x': 'x1 + 0.1*sin(2*pi*x1)*sin(2*pi*x2)',
                        'y': 'x2 + 0.1*sin(2*pi*x1)*sin(2*pi*x2)'}

    F = Collela2D('M', dim=2).get_callable_mapping()

    ncells   = [6, 6]
    degree   = [3, 3]
    periodic = [False, False]

    domain_decomposition = DomainDecomposition(ncells=ncells, periods=periodic, comm=None)
    spaces = [SplineSpace(degree=p, grid=np.linspace(0, 1, n + 1), periodic=per)
             for n, p, per in zip(ncells, degree, periodic)]
    T = TensorFemSpace(domain_decomposition, *spaces)
    F_h_manual = SplineCallableMapping.from_mapping(F, T)

    F_h_auto = SplineCallableMapping.from_mapping(F, ncells=ncells, degree=degree)

    assert np.allclose(F_h_manual.control_points[...], F_h_auto.control_points[...])

    with pytest.raises(ValueError, match="ncells.*degree"):
        SplineCallableMapping.from_mapping(F, ncells=ncells)

    with pytest.raises(ValueError, match="same length"):
        SplineCallableMapping.from_mapping(F, ncells=[4, 4, 4], degree=[2, 2])

#==============================================================================
def test_from_mapping_accepts_a_defined_mapping():
    # from_mapping unwraps a sympde DefinedMapping itself, so callers need
    # not write `.get_callable_mapping()`. Passing the symbolic mapping and
    # passing its callable must give the identical interpolant.
    from sympde.topology.mapping import AnalyticMapping, DiscreteMapping
    from psydac.mapping.discrete import SplineCallableMapping

    class Collela2D(AnalyticMapping):
        _expressions = {'x': 'x1 + 0.1*sin(2*pi*x1)*sin(2*pi*x2)',
                        'y': 'x2 + 0.1*sin(2*pi*x1)*sin(2*pi*x2)'}

    F        = Collela2D('M', dim=2)
    ncells   = [6, 6]
    degree   = [3, 3]
    kwargs   = dict(ncells=ncells, degree=degree)

    # 1. an AnalyticMapping, vs. its callable (which is `self` since WP06c)
    from_symbolic = SplineCallableMapping.from_mapping(F, **kwargs)
    from_callable = SplineCallableMapping.from_mapping(F.get_callable_mapping(), **kwargs)
    assert np.allclose(from_symbolic.control_points[...],
                       from_callable.control_points[...])

    # 2. a DiscreteMapping, which unwraps to the spline underneath rather
    #    than evaluating through the symbolic wrapper
    G   = DiscreteMapping(from_symbolic, 'G')
    assert isinstance(G, DiscreteMapping)
    G_h = SplineCallableMapping.from_mapping(G, **kwargs)
    assert np.allclose(G_h.control_points[...], from_symbolic.control_points[...])

#==============================================================================
def test_from_mapping_rejects_the_old_argument_order():
    # `mapping` used to come second, after a `tensor_space` that had to be an
    # explicit `None` when unused. Both legacy shapes must fail loudly and
    # name the replacement, rather than binding a space (or None) to
    # `mapping` and failing obscurely further in.
    from sympde.topology.mapping import AnalyticMapping
    from psydac.mapping.discrete import SplineCallableMapping

    class Collela2D(AnalyticMapping):
        _expressions = {'x': 'x1 + 0.1*sin(2*pi*x1)*sin(2*pi*x2)',
                        'y': 'x2 + 0.1*sin(2*pi*x1)*sin(2*pi*x2)'}

    F = Collela2D('M', dim=2)
    ncells, degree = [6, 6], [3, 3]

    # from_mapping(None, F, ncells=..., degree=...)
    with pytest.raises(TypeError, match="reordered"):
        SplineCallableMapping.from_mapping(None, F, ncells=ncells, degree=degree)

    # from_mapping(V, F)
    T = TensorFemSpace(
        DomainDecomposition(ncells=ncells, periods=[False, False], comm=None),
        *[SplineSpace(degree=p, grid=np.linspace(0, 1, n + 1), periodic=False)
          for n, p in zip(ncells, degree)])
    with pytest.raises(TypeError, match="reordered"):
        SplineCallableMapping.from_mapping(T, F)

    # the new order works
    assert SplineCallableMapping.from_mapping(F, T) is not None
