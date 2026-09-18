#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
"""
WP14a (D5): scalar/array point-evaluation contract of `SplineCallableMapping`
and `NurbsCallableMapping`. All array-input cases in this slice go through a
per-point loop reusing the scalar code (`_eval_pointwise`); there is no fast
path yet (that is WP14b, layered on top without changing any shape/type/value
pinned here). Every array-input assertion below therefore uses exact equality
against an independently-written per-point loop.
"""
import os

import numpy as np
import pytest

from sympde.topology import Square, PolarMapping
from sympde.topology.mapping import AnalyticMapping

from psydac.mapping.discrete import (SplineCallableMapping, NurbsCallableMapping,
                                      _tensor_axes_of_meshgrid, _eval_pointwise)
from psydac.cad.geometry     import Geometry
import psydac.cad.mesh as mesh_mod

#==============================================================================
# Fixtures
#==============================================================================
class _SurfaceMapping(AnalyticMapping):
    """ pdim=3, ldim=2 analytic surface, used to exercise non-square Jacobians. """
    _expressions = {'x': 'x1', 'y': 'x2', 'z': 'x1**2+x2**2'}


@pytest.fixture(scope='module')
def spline_mapping():
    """ A PolarMapping-interpolated spline (ldim = pdim = 2). """
    A = Square('A', bounds1=(0., 1.), bounds2=(0., 0.5 * np.pi))
    F = PolarMapping('F', dim=2, c1=0., c2=0., rmin=0.3, rmax=1.0)
    return SplineCallableMapping.from_mapping(
        None, F.get_callable_mapping(), ncells=(8, 8), degree=(3, 3),
        bounds=zip(A.min_coords, A.max_coords))


@pytest.fixture(scope='module')
def nurbs_mapping():
    """ The NURBS mapping from 'quarter_annulus.h5' (ldim = pdim = 2). """
    mesh_dir = os.path.dirname(mesh_mod.__file__)
    filename = os.path.join(mesh_dir, 'quarter_annulus.h5')
    geo = Geometry.from_file(filename)
    mapping = list(geo.mappings.values())[0]
    assert isinstance(mapping, NurbsCallableMapping)
    return mapping


@pytest.fixture(scope='module')
def surface_mapping():
    """ A spline interpolation of `_SurfaceMapping` (pdim=3, ldim=2). """
    F = _SurfaceMapping('Surf', ldim=2, pdim=3)
    return SplineCallableMapping.from_mapping(None, F, ncells=(6, 6), degree=(3, 3))


#==============================================================================
# Reference implementation, independent of the production `_eval_pointwise`
#==============================================================================
def _loop_eval(method, eta, comp_shape):
    """ Reference per-point loop: evaluate `method` scalar-wise on the
    broadcast shape of `eta`, and assemble it with the component axes first
    (mirrors the contract, not the implementation, of `_eval_pointwise`). """
    arrs = np.broadcast_arrays(*(np.asarray(e) for e in eta))
    shape = arrs[0].shape
    n = arrs[0].size

    flat = np.empty((n,) + comp_shape)
    for k in range(n):
        point = [a.flat[k] for a in arrs]
        flat[k] = np.asarray(method(*point))

    out = flat.reshape(shape + comp_shape)
    return np.moveaxis(out, range(len(shape), out.ndim), range(len(comp_shape)))


METHODS = {
    '__call__':     lambda m: (m.pdim,),
    'jacobian':     lambda m: (m.pdim, m.ldim),
    'jacobian_inv': lambda m: (m.ldim, m.pdim),
    'metric':       lambda m: (m.ldim, m.ldim),
    'metric_det':   lambda m: (),
}


def _bounds(mapping):
    return [(b[0], b[-1]) for b in mapping.space.breaks]


#==============================================================================
# Input-kind generators: each returns an eta = (x1, x2) tuple of array_like
#==============================================================================
def _inner(lo, hi, n):
    return np.linspace(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), n)

def _kind_dense_ij(bounds):
    (lo1, hi1), (lo2, hi2) = bounds
    return np.meshgrid(_inner(lo1, hi1, 5), _inner(lo2, hi2, 4), indexing='ij')

def _kind_sparse_ij(bounds):
    (lo1, hi1), (lo2, hi2) = bounds
    return np.meshgrid(_inner(lo1, hi1, 5), _inner(lo2, hi2, 4), indexing='ij', sparse=True)

def _kind_dense_xy(bounds):
    (lo1, hi1), (lo2, hi2) = bounds
    return np.meshgrid(_inner(lo1, hi1, 5), _inner(lo2, hi2, 4), indexing='xy')

def _kind_point_cloud(bounds):
    (lo1, hi1), (lo2, hi2) = bounds
    rng = np.random.default_rng(1234)
    x1 = lo1 + rng.random(7) * (hi1 - lo1)
    x2 = lo2 + rng.random(7) * (hi2 - lo2)
    return (x1, x2)

def _kind_size1(bounds):
    (lo1, hi1), (lo2, hi2) = bounds
    return (np.array([0.5 * (lo1 + hi1)]), np.array([0.5 * (lo2 + hi2)]))

def _kind_single_point_meshgrid(bounds):
    x1, x2 = _kind_size1(bounds)
    return np.meshgrid(x1, x2, indexing='ij')

def _kind_descending(bounds):
    (lo1, hi1), (lo2, hi2) = bounds
    a1 = _inner(lo1, hi1, 5)[::-1]
    a2 = _inner(lo2, hi2, 4)
    return np.meshgrid(a1, a2, indexing='ij')

def _kind_out_of_range(bounds):
    (lo1, hi1), (lo2, hi2) = bounds
    a1 = np.linspace(lo1 - 0.2 * (hi1 - lo1), hi1 + 0.2 * (hi1 - lo1), 5)
    a2 = _inner(lo2, hi2, 4)
    return np.meshgrid(a1, a2, indexing='ij')

def _kind_diagonal(bounds):
    (lo1, hi1), (lo2, hi2) = bounds
    t = np.linspace(0.2, 0.8, 6)
    return (lo1 + t * (hi1 - lo1), lo2 + t * (hi2 - lo2))


INPUT_KINDS = {
    'dense_ij':             _kind_dense_ij,
    'sparse_ij':             _kind_sparse_ij,
    'dense_xy':             _kind_dense_xy,
    'point_cloud':          _kind_point_cloud,
    'size1':                _kind_size1,
    'single_point_meshgrid': _kind_single_point_meshgrid,
    'descending':           _kind_descending,
    'out_of_range':         _kind_out_of_range,
    'diagonal':             _kind_diagonal,
}

# WP14b tolerance rule (A6, applied here since WP14a's own version of this
# check used exact equality unconditionally): the fast tensor-grid path
# (dense/sparse meshgrids) sums in a different order than the per-point loop
# (~1e-15 agreement, never bitwise-equal); every other input kind is
# ineligible for the fast path (non-tensor, out-of-range, single-point, or
# descending) and keeps going through the untouched per-point loop, so it
# still matches the loop exactly.
FAST_PATH_KINDS = {'dense_ij', 'sparse_ij', 'dense_xy'}

def _assert_matches_loop(kind, result, expected):
    if kind in FAST_PATH_KINDS:
        np.testing.assert_allclose(result, expected, rtol=0, atol=5e-14)
    else:
        np.testing.assert_array_equal(result, expected)


#==============================================================================
# Tests
#==============================================================================
@pytest.mark.parametrize('mapping_name', ['spline_mapping', 'nurbs_mapping'])
def test_scalar_return_types_unchanged(mapping_name, request):
    mapping = request.getfixturevalue(mapping_name)

    for x1, x2 in [(0.5, 0.4), (np.float64(0.5), np.float64(0.4)),
                   (np.array(0.5), np.array(0.4))]:
        result = mapping(x1, x2)
        if isinstance(mapping, NurbsCallableMapping):
            assert isinstance(result, np.ndarray) and result.shape == (mapping.pdim,)
        else:
            assert isinstance(result, list) and all(isinstance(v, np.float64) for v in result)

        J = mapping.jacobian(x1, x2)
        assert isinstance(J, np.ndarray) and J.shape == (mapping.pdim, mapping.ldim)

        d = mapping.metric_det(x1, x2)
        assert isinstance(d, np.float64)


@pytest.mark.parametrize('kind', list(INPUT_KINDS))
@pytest.mark.parametrize('method', list(METHODS))
@pytest.mark.parametrize('mapping_name', ['spline_mapping', 'nurbs_mapping'])
def test_array_eval_matches_scalar_loop(mapping_name, method, kind, request):
    mapping = request.getfixturevalue(mapping_name)
    eta = INPUT_KINDS[kind](_bounds(mapping))
    comp_shape = METHODS[method](mapping)

    fn = getattr(mapping, method)
    result = fn(*eta)

    broadcast_shape = np.broadcast_shapes(*(np.shape(e) for e in eta))
    expected = _loop_eval(fn, eta, comp_shape)

    if method == '__call__':
        assert isinstance(result, tuple) and len(result) == mapping.pdim
        result = np.array(result)
    else:
        result = np.asarray(result)

    assert result.shape == comp_shape + broadcast_shape
    _assert_matches_loop(kind, result, expected)


@pytest.mark.parametrize('kind', list(INPUT_KINDS))
@pytest.mark.parametrize('mapping_name', ['spline_mapping', 'nurbs_mapping'])
def test_jacobian_det_array_eval_matches_scalar_loop(mapping_name, kind, request):
    """ `jacobian_det` isn't part of `BasicCallableMapping`'s shared interface
    (so it's kept out of METHODS, which `test_array_shapes_match_analytic_
    mapping` also uses to compare against a sympde AnalyticMapping that has no
    such method), but its array-input support is a direct WP14a /code-review
    fix: np.linalg.det was being applied to jacobian(...)'s component-axes-
    first array output directly, which only "worked" by accident on a square
    broadcast shape and was wrong otherwise (LinAlgError or silently wrong
    values). WP14b later gave it its own fast tensor-grid path too (routed
    through jac_det_grid's dedicated kernel, not jac_mat_grid + det), so this
    uses the same tolerance rule as METHODS' cases: exact for loop-only input
    kinds, ~1e-14 for the fast-path-eligible meshgrid kinds. """
    mapping = request.getfixturevalue(mapping_name)
    eta = INPUT_KINDS[kind](_bounds(mapping))

    result = mapping.jacobian_det(*eta)
    expected = _loop_eval(mapping.jacobian_det, eta, ())

    broadcast_shape = np.broadcast_shapes(*(np.shape(e) for e in eta))
    assert np.shape(result) == broadcast_shape
    _assert_matches_loop(kind, result, expected)


@pytest.mark.parametrize('kind', list(INPUT_KINDS))
def test_array_shapes_match_analytic_mapping(spline_mapping, kind):
    """ Every spline result has the same shape as `PolarMapping`'s on the
    same meshgrid. """
    F = PolarMapping('F', dim=2, c1=0., c2=0., rmin=0.3, rmax=1.0).get_callable_mapping()
    eta = INPUT_KINDS[kind](_bounds(spline_mapping))

    for method in METHODS:
        f_h = getattr(spline_mapping, method)
        f_a = getattr(F, method)
        r_h = f_h(*eta)
        r_a = f_a(*eta)
        shape_h = tuple(np.shape(c) for c in r_h) if method == '__call__' else np.shape(r_h)
        shape_a = tuple(np.shape(c) for c in r_a) if method == '__call__' else np.shape(r_a)
        assert shape_h == shape_a


def test_surface_mapping_array_eval(surface_mapping):
    x1, x2 = np.meshgrid(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 3), indexing='ij')
    S = x1.shape

    r_call = surface_mapping(x1, x2)
    assert isinstance(r_call, tuple) and np.array(r_call).shape == (3,) + S

    r_jac = surface_mapping.jacobian(x1, x2)
    assert r_jac.shape == (3, 2) + S

    r_met = surface_mapping.metric(x1, x2)
    assert r_met.shape == (2, 2) + S

    # __call__ has no square-Jacobian restriction, so this dense meshgrid is
    # fast-path-eligible (WP14b) -- ~1e-15 agreement, not bitwise-equal.
    # jacobian/metric need square=True, which this pdim=3/ldim=2 surface
    # fails, so they stay on the (bitwise-exact) per-point loop.
    np.testing.assert_allclose(np.array(r_call), _loop_eval(surface_mapping.__call__, (x1, x2), (3,)), rtol=0, atol=5e-14)
    np.testing.assert_array_equal(r_jac, _loop_eval(surface_mapping.jacobian, (x1, x2), (3, 2)))
    np.testing.assert_array_equal(r_met, _loop_eval(surface_mapping.metric, (x1, x2), (2, 2)))


def test_discrete_mapping_delegates_array_eval(spline_mapping):
    G = spline_mapping.to_defined_mapping('F')
    X1, X2 = np.meshgrid(_inner(0., 1., 5), _inner(0., 0.5 * np.pi, 4), indexing='ij')

    r_direct = spline_mapping(X1, X2)
    r_via_G  = G(X1, X2)
    for a, b in zip(r_direct, r_via_G):
        np.testing.assert_array_equal(a, b)

    np.testing.assert_array_equal(spline_mapping.jacobian(X1, X2), G.jacobian(X1, X2))


def test_empty_array_eval(spline_mapping):
    x1 = np.array([], dtype=float)
    x2 = np.array([], dtype=float)

    r_call = spline_mapping(x1, x2)
    assert isinstance(r_call, tuple)
    for c in r_call:
        assert c.shape == (0,)

    r_jac = spline_mapping.jacobian(x1, x2)
    assert r_jac.shape == (2, 2, 0)

    r_det = spline_mapping.metric_det(x1, x2)
    assert r_det.shape == (0,)


def test_grid_helper_scalar_branch(spline_mapping):
    """ Regression for A4: the scalar (Case 1) branches of the grid helpers
    used to raise `AttributeError` or return a wrong-shaped `nan`. """
    x1, x2 = 0.3, 0.4

    np.testing.assert_array_equal(
        spline_mapping.jac_mat_grid([x1, x2]), spline_mapping.jacobian(x1, x2))
    np.testing.assert_array_equal(
        spline_mapping.inv_jac_mat_grid([x1, x2]), spline_mapping.jacobian_inv(x1, x2))

    with _no_warning():
        det = spline_mapping.jac_det_grid([x1, x2])
    assert np.ndim(det) == 0
    np.testing.assert_array_equal(det, np.linalg.det(spline_mapping.jacobian(x1, x2)))

    np.testing.assert_array_equal(
        spline_mapping.jacobian_det(x1, x2), np.linalg.det(spline_mapping.jacobian(x1, x2)))


class _no_warning:
    """ Context manager asserting no warning is raised (used in place of
    `pytest.warns(None)`, removed in recent pytest). """
    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings(record=True)
        self._records = self._cm.__enter__()
        warnings.simplefilter('always')
        return self._records

    def __exit__(self, *exc):
        assert not any(issubclass(r.category, RuntimeWarning) for r in self._records)
        self._cm.__exit__(*exc)
        return False


def test_grid_helpers_reject_non_square_jacobian(surface_mapping):
    """ Regression for A5 (grid path) and for a follow-up /code-review finding
    on the scalar (Case 1) path: `inv_jac_mat_grid`/`jac_det_grid`'s Case 1
    needs a square Jacobian too (the inverse/determinant of a non-square
    matrix isn't defined), unlike `jac_mat_grid`'s Case 1, which just returns
    `jacobian(...)`'s raw, possibly non-square matrix -- so the guard must
    gate Case 1 for the first two but not the third. Without this, a scalar
    call on a surface mapping used to hit numpy's own confusing
    `LinAlgError: Last 2 dimensions of the array must be square` instead of
    this method's documented `NotImplementedError`. """
    grid = [np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 4)]
    x1, x2 = 0.3, 0.4

    for name in ('inv_jac_mat_grid', 'jac_det_grid'):
        with pytest.raises(NotImplementedError):
            getattr(surface_mapping, name)(grid)
        with pytest.raises(NotImplementedError):
            getattr(surface_mapping, name)([x1, x2])

    with pytest.raises(NotImplementedError):
        surface_mapping.jac_mat_grid(grid)
    # jac_mat_grid's Case 1 works for any pdim/ldim -- no exception, and it
    # must match jacobian(...) exactly (also covered by
    # test_grid_helper_scalar_branch for the square-mapping case).
    np.testing.assert_array_equal(
        surface_mapping.jac_mat_grid([x1, x2]), surface_mapping.jacobian(x1, x2))

    # jacobian_det/jacobian_inv: not part of the *_grid family, but share the
    # same square-Jacobian requirement -- also a /code-review follow-up.
    # jacobian_inv is checked at both scalar and array eta, since the array
    # path (the loop in _eval_pointwise) is new in this WP.
    with pytest.raises(NotImplementedError):
        surface_mapping.jacobian_det(x1, x2)
    with pytest.raises(NotImplementedError):
        surface_mapping.jacobian_inv(x1, x2)
    with pytest.raises(NotImplementedError):
        X1, X2 = np.meshgrid(np.linspace(0.1, 0.9, 3), np.linspace(0.1, 0.9, 3), indexing='ij')
        surface_mapping.jacobian_inv(X1, X2)

    # build_mesh has no square-Jacobian restriction -- positive control.
    mesh = surface_mapping.build_mesh(grid)
    assert len(mesh) == 3


def test_grid_helpers_accept_mixed_size_axes(spline_mapping):
    """ Regression for a /code-review finding: Case 1's `grid[0].size == 1`
    check only inspected the first axis, so a grid where one axis has a
    single point but another has several was misrouted into Case 1, where the
    new `.item()` conversion (added for A4) crashed on the multi-point axis
    with a confusing `ValueError` instead of falling through to the
    tensor-grid kernels. This is a real, not just theoretical, shape -- e.g.
    evaluating along a boundary line (one axis pinned to a single value). """
    grid = [np.array([0.5]), np.linspace(0.1, 0.9, 5)]

    jac  = spline_mapping.jac_mat_grid(grid)
    ijac = spline_mapping.inv_jac_mat_grid(grid)
    det  = spline_mapping.jac_det_grid(grid)

    assert jac.shape  == (1, 5, 2, 2)
    assert ijac.shape == (1, 5, 2, 2)
    assert det.shape  == (1, 5)

    # must agree with individual scalar calls, not just avoid crashing --
    # jac_mat_grid's family is grid-axes-first/components-last (its own,
    # pre-existing convention, unrelated to _eval_pointwise's component-first
    # one used by the abstract-interface methods). Compares the compiled
    # kernel (eval_jacobians_2d) against the scalar FemField.gradient path,
    # which are not bitwise equal (~1e-14, pre-existing, unrelated to this fix).
    for i, x1 in enumerate(grid[0]):
        for j, x2 in enumerate(grid[1]):
            np.testing.assert_allclose(jac[i, j], spline_mapping.jacobian(x1, x2), rtol=0, atol=5e-14)
            np.testing.assert_allclose(det[i, j], spline_mapping.jacobian_det(x1, x2), rtol=0, atol=5e-14)


#==============================================================================
# WP14b: the tensor-grid fast path
#==============================================================================
class _LineMapping(AnalyticMapping):
    """ ldim=pdim=1 identity, used to exercise the 'no 1D Jacobian kernel' gate. """
    _expressions = {'x': 'x1'}


@pytest.fixture(scope='module')
def line_mapping():
    """ A 1-D spline (ldim = pdim = 1). """
    return SplineCallableMapping.from_mapping(
        None, _LineMapping('L', ldim=1, pdim=1), ncells=[6], degree=[3])


def test_tensor_grid_detection():
    """ `_tensor_axes_of_meshgrid` (B1) against the full case table. """
    x1 = np.linspace(0.2, 0.9, 5)
    x2 = np.linspace(0.1, 0.7, 4)

    # Dense 'ij', dense 'xy', and sparse meshgrids -- all detected.
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    axes, perm = _tensor_axes_of_meshgrid((X1, X2))
    assert perm == [0, 1]
    np.testing.assert_array_equal(axes[0], x1)
    np.testing.assert_array_equal(axes[1], x2)

    X1s, X2s = np.meshgrid(x1, x2, indexing='ij', sparse=True)
    axes, perm = _tensor_axes_of_meshgrid((X1s, X2s))
    assert perm == [0, 1]
    np.testing.assert_array_equal(axes[0], x1)
    np.testing.assert_array_equal(axes[1], x2)

    X1x, X2x = np.meshgrid(x1, x2, indexing='xy')
    axes, perm = _tensor_axes_of_meshgrid((X1x, X2x))
    assert perm == [1, 0]
    np.testing.assert_array_equal(axes[0], x1)
    np.testing.assert_array_equal(axes[1], x2)

    # Constant axis (varies nowhere): detected, takes the free dimension.
    Xc = np.full((5, 4), 3.0)
    axes, perm = _tensor_axes_of_meshgrid((Xc, X2))
    assert perm == [0, 1]
    np.testing.assert_array_equal(axes[0], np.full(5, 3.0))
    np.testing.assert_array_equal(axes[1], x2)

    # Rejected: a 1-D point cloud (broadcast shape has fewer dims than ldim).
    assert _tensor_axes_of_meshgrid((x1[:4], x2)) is None

    # Rejected: a diagonal (two arrays co-varying along a shared index, not
    # independently per array dimension).
    t = np.linspace(0., 1., 6)
    assert _tensor_axes_of_meshgrid((t, t)) is None
    assert _tensor_axes_of_meshgrid((0.2 + 0.5 * t, 0.1 + 0.4 * t)) is None

    # Rejected: a single-point axis (the kernels' grid[0].size == 1 Case 1).
    X1sp, X2sp = np.meshgrid(np.array([0.5]), x2, indexing='ij')
    assert _tensor_axes_of_meshgrid((X1sp, X2sp)) is None

    # Rejected: a descending axis.
    X1d, X2d = np.meshgrid(x1[::-1], x2, indexing='ij')
    assert _tensor_axes_of_meshgrid((X1d, X2d)) is None


FAST_ELIGIBLE_KINDS = ['dense_ij', 'sparse_ij', 'dense_xy']


@pytest.mark.parametrize('kind', FAST_ELIGIBLE_KINDS)
@pytest.mark.parametrize('method', list(METHODS))
@pytest.mark.parametrize('mapping_name', ['spline_mapping', 'nurbs_mapping'])
def test_fast_path_is_taken_and_matches_the_loop(mapping_name, method, kind, request):
    mapping = request.getfixturevalue(mapping_name)
    eta = INPUT_KINDS[kind](_bounds(mapping))
    comp_shape = METHODS[method](mapping)
    square = method != '__call__'

    assert mapping._fast_tensor_grid(eta, square=square) is not None

    fn = getattr(mapping, method)
    result = fn(*eta)
    # Reference: the production per-point loop, called directly (WP14a's
    # code path, untouched by this WP).
    expected = _eval_pointwise(fn, eta, comp_shape)

    result = np.array(result) if method == '__call__' else np.asarray(result)
    np.testing.assert_allclose(result, expected, rtol=0, atol=5e-14)


def test_fast_path_declined(spline_mapping, surface_mapping, line_mapping, monkeypatch):
    bounds = _bounds(spline_mapping)

    assert spline_mapping._fast_tensor_grid(tuple(_kind_descending(bounds))) is None
    assert spline_mapping._fast_tensor_grid(tuple(_kind_out_of_range(bounds))) is None
    assert spline_mapping._fast_tensor_grid(tuple(_kind_single_point_meshgrid(bounds))) is None
    assert spline_mapping._fast_tensor_grid(tuple(_kind_point_cloud(bounds))) is None
    assert spline_mapping._fast_tensor_grid(tuple(_kind_diagonal(bounds))) is None

    # A surface mapping (pdim=3, ldim=2): square=True is declined.
    x1, x2 = np.meshgrid(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 3), indexing='ij')
    assert surface_mapping._fast_tensor_grid((x1, x2), square=True) is None
    # __call__ (square=False) is unaffected by the pdim != ldim restriction.
    assert surface_mapping._fast_tensor_grid((x1, x2)) is not None

    # ldim == 1: square=True is declined (no 1D Jacobian kernel), square=False isn't.
    x = np.linspace(0.1, 0.9, 5)
    assert line_mapping._fast_tensor_grid((x,), square=True) is None
    assert line_mapping._fast_tensor_grid((x,)) is not None

    # Parallel: monkeypatch coeff_space.parallel to True (no MPI needed) and
    # confirm the fast path is declined -- i.e. the loop path is taken.
    X1, X2 = _kind_dense_ij(bounds)
    monkeypatch.setattr(type(spline_mapping.space.coeff_space), 'parallel', True)
    assert spline_mapping._fast_tensor_grid((X1, X2)) is None
    np.testing.assert_array_equal(
        np.array(spline_mapping(X1, X2)), _eval_pointwise(spline_mapping, (X1, X2), (spline_mapping.pdim,)))
