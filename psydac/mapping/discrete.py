#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from itertools import product
import string
import random
import warnings

import yaml
import numpy as np
import h5py

from sympde.topology.callable_mapping import BasicCallableMapping
from sympde.topology.mapping import DefinedMapping, DiscreteMapping

from psydac.fem.basic    import FemField
from psydac.fem.splines  import SplineSpace
from psydac.fem.tensor   import TensorFemSpace
from psydac.ddm.cart     import DomainDecomposition


__all__ = ('SplineCallableMapping', 'NurbsCallableMapping')

#==============================================================================
def _is_scalar_eta(eta):
    """Whether every logical coordinate in `eta` is a single point.

    Used to pick between the two evaluation paths every point-evaluation
    method offers: the original scalar-only body (`FemField.__call__`/
    `.gradient` need a true Python/numpy scalar) below this check, or
    `_eval_pointwise` above it for array-like `eta`.

    Parameters
    ----------
    eta : tuple of array_like
        One entry per logical coordinate, as passed to e.g. `__call__`.

    Returns
    -------
    bool
        `True` when every entry has `ndim == 0` (a `float`, `np.float64`, or
        0-d `ndarray`); `False` if any entry is a higher-dimensional array.

    Examples
    --------
    >>> _is_scalar_eta((0.3, 0.4))
    True
    >>> _is_scalar_eta((np.array(0.3), np.array(0.4)))
    True
    >>> _is_scalar_eta((np.array([0.3, 0.5]), 0.4))
    False
    """
    return all(np.ndim(e) == 0 for e in eta)

def _eval_pointwise(scalar_fn, eta, comp_shape):
    """Apply a scalar-only evaluator over broadcast array coordinates.

    `SplineCallableMapping`'s point evaluation goes through `FemField.__call__`/
    `.gradient`, which read spline coefficients via `find_span`'s `float(x)`
    and are irreducibly scalar. This loops `scalar_fn` over the broadcast
    shape of `eta` and reassembles the results with the component axes first
    (matching `AnalyticMapping`'s convention), so callers can pass array-like
    `eta` without every evaluator having to be rewritten.

    Parameters
    ----------
    scalar_fn : callable
        A method of `self` accepting `ldim` scalar coordinates.
    eta : tuple of array_like
        One array-like per logical coordinate; broadcast together.
    comp_shape : tuple of int
        Shape of one scalar-eta result (e.g. `()`, `(pdim,)`, `(pdim, ldim)`).

    Returns
    -------
    ndarray
        Shape `comp_shape + S`, where `S` is the broadcast shape of `eta`.

    Warns
    -----
    In a distributed (MPI) space, this is the path every array-`eta` call
    takes when `_fast_tensor_grid` declines (which it always does in
    parallel -- see there). It carries over the same locality requirement
    scalar evaluation already had: every point must lie in the calling
    rank's local domain (plus ghost region), on every rank, or the
    underlying `FemField` lookup raises or returns wrong values -- this was
    never checked for scalar `eta` either, so it isn't a new restriction,
    but it now also applies to whatever array `eta` a caller passes.

    Examples
    --------
    >>> X1, X2 = np.meshgrid([0.2, 0.5], [0.3, 0.7], indexing='ij')
    >>> _eval_pointwise(F.metric_det, (X1, X2), ()).shape   # F: a mapping
    (2, 2)
    >>> _eval_pointwise(F.jacobian, (X1, X2), (F.pdim, F.ldim)).shape
    (2, 2, 2, 2)
    """
    arrs = np.broadcast_arrays(*map(np.asarray, eta))
    shape = arrs[0].shape
    n = arrs[0].size

    out = None
    for k in range(n):
        point = [a.flat[k] for a in arrs]
        value = np.asarray(scalar_fn(*point))
        if out is None:
            out = np.empty((n,) + comp_shape, dtype=value.dtype)
        out[k] = value

    if out is None:
        out = np.empty((0,) + comp_shape, dtype=float)

    out = out.reshape(shape + comp_shape)
    return np.moveaxis(out, range(len(shape), out.ndim), range(len(comp_shape)))

#==============================================================================
def _tensor_axes_of_meshgrid(eta):
    """Detect whether `eta` is (a meshgrid of) a tensor-product grid.

    WP14b's fast path routes tensor grids through psydac's compiled kernels
    (`build_mesh`/`jac_mat_grid`) instead of `_eval_pointwise`'s per-point
    loop. Those kernels need one strictly-per-axis 1-D array each; this is a
    pure-numpy structural test for that shape, independent of any mapping, so
    it can be unit-tested on its own and reused by `_fast_tensor_grid`'s
    gates. Diagonal or otherwise non-tensor inputs (e.g. a point cloud, or two
    arrays that co-vary along a shared index) are rejected, not mis-detected:
    every array's variation is checked exhaustively against the data, not
    inferred from `indexing='ij'`/`'xy'` conventions.

    Parameters
    ----------
    eta : tuple of array_like
        One entry per logical coordinate, as passed to e.g. `__call__`.

    Returns
    -------
    (axes, perm) or None
        `axes[i]` is the non-decreasing 1-D `float64` grid along logical axis
        `i`; `perm[i]` is the array dimension that logical axis `i` indexes
        (so `perm` is a permutation of `range(len(eta))`). `None` if `eta`
        is not a tensor grid with at least 2 points along every axis, real
        dtype, and non-decreasing axes.

    Examples
    --------
    >>> X1, X2 = np.meshgrid([0.2, 0.5], [0.3, 0.7, 0.9], indexing='ij')
    >>> axes, perm = _tensor_axes_of_meshgrid((X1, X2))
    >>> [a.tolist() for a in axes], perm
    ([[0.2, 0.5], [0.3, 0.7, 0.9]], [0, 1])
    """
    arrs = np.broadcast_arrays(*(np.asarray(e) for e in eta))
    ldim = len(eta)
    shape = arrs[0].shape

    # Step 1: must be a genuine ldim-D grid, >= 2 points per axis (the
    # kernels dispatch on grid[0].size == 1 and would mis-route a length-1
    # axis into their scalar Case 1).
    if len(shape) != ldim or any(s < 2 for s in shape):
        return None

    # Step 2: real dtype only; work in float64 from here.
    if not all(np.isrealobj(a) for a in arrs):
        return None
    arrs = [a.astype(np.float64, copy=False) for a in arrs]

    # Step 3: per array, the (at most one) array dimension it varies along.
    varying_dim = []
    for a in arrs:
        dims = [d for d in range(ldim)
                if not np.all(a == np.take(a, [0], axis=d))]
        if len(dims) > 1:
            return None
        varying_dim.append(dims[0] if dims else None)

    # Step 4: build the bijection perm from "varies along"; a collision means
    # two logical axes vary along the same array dimension (e.g. a diagonal).
    # Constant arrays (vary along none) take the remaining dimensions, in
    # order -- equivalent by construction since they don't vary anywhere.
    perm = [None] * ldim
    used = set()
    for i, d in enumerate(varying_dim):
        if d is None:
            continue
        if d in used:
            return None
        perm[i] = d
        used.add(d)
    free = iter(d for d in range(ldim) if d not in used)
    for i in range(ldim):
        if perm[i] is None:
            perm[i] = next(free)

    # Step 5-6: extract each axis (contiguous float64) and require it
    # non-decreasing (preprocess_irregular_tensor_grid asserts exactly this).
    axes = []
    for i, a in enumerate(arrs):
        idx = [0] * ldim
        idx[perm[i]] = slice(None)
        axis = np.ascontiguousarray(a[tuple(idx)], dtype=np.float64)
        if np.any(np.diff(axis) < 0):
            return None
        axes.append(axis)

    return axes, perm

def _components_to_front(arr, perm, ncomp):
    """Reorder `arr`'s leading grid axes to match `eta`'s array-dimension
    order, then move the trailing component axes to the front.

    `jac_mat_grid` and friends return grid-axes-first (in logical-axis
    order), components-last; the point-evaluation contract is components-
    first, in the array-dimension order of the original `eta` (matching
    `AnalyticMapping`). `perm`, from `_tensor_axes_of_meshgrid`, maps logical
    axis `i` to the array dimension it indexes.

    Parameters
    ----------
    arr : ndarray
        Shape `(n_0, ..., n_{ldim-1}) + comp_shape`, `comp_shape` of length
        `ncomp`, axis `i` (i < ldim) varying along logical axis `i`.
    perm : list of int
        Permutation of `range(ldim)` from `_tensor_axes_of_meshgrid`.
    ncomp : int
        Number of trailing component axes (0, or 2 for the Jacobian family).

    Returns
    -------
    ndarray
        Shape `comp_shape + S`, `S` the broadcast shape of the original
        `eta` (array-dimension order).

    Examples
    --------
    >>> # eta built with indexing='xy' -> perm = [1, 0] (axes swapped)
    >>> J = jac_mat_grid([x_axis, y_axis])  # shape (n_x, n_y, pdim, ldim)
    >>> _components_to_front(J, perm=[1, 0], ncomp=2).shape
    (pdim, ldim, n_y, n_x)
    """
    ldim = len(perm)
    order = list(np.argsort(perm)) + list(range(ldim, arr.ndim))
    arr = np.transpose(arr, order)
    return np.moveaxis(arr, range(ldim, arr.ndim), range(ncomp))

#==============================================================================
class SplineCallableMapping(BasicCallableMapping):

    #: Tag written into / read back from the 'type' field of a geometry
    #: file's geometry.yml. Frozen at the historical class name so that
    #: renaming the Python class never changes the HDF5 format.
    geometry_dtype = 'SplineMapping'

    def __init__(self, *components, name=None):

        # Sanity checks
        assert len(components) >= 1
        assert all(isinstance(c, FemField) for c in components)
        assert all(isinstance(c.space, TensorFemSpace) for c in components)
        assert all(c.space is components[0].space for c in components)

        # Store spline space and one field for each coordinate X_i
        self._space  = components[0].space
        self._fields = components

        # Store number of logical and physical dimensions
        self._ldim = components[0].space.ldim
        self._pdim = len(components)

        # Create helper object for accessing control points with slicing syntax
        # as if they were stored in a single multi-dimensional array C with
        # indices [i1, ..., i_n, d] where (i1, ..., i_n) are indices of logical
        # coordinates, and d is index of physical component of interest.
        self._control_points = SplineCallableMapping.ControlPoints(self)
        self._name           = name

    @property
    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    #--------------------------------------------------------------------------
    # Option [1]: initialize from TensorFemSpace and pre-existing mapping
    #--------------------------------------------------------------------------
    @classmethod
    def from_mapping(cls, mapping, space=None, *,
                     ncells=None, degree=None, periodic=None, bounds=None, comm=None):
        """
        Interpolate a `BasicCallableMapping` (typically an `AnalyticMapping`'s
        callable) onto a spline space, at that space's Greville points.

        Parameters
        ----------
        mapping : BasicCallableMapping or DefinedMapping
            The mapping to interpolate. Must have the same `ldim` as
            `space` (or as `ncells`/`degree`, when building one).

            A sympde `DefinedMapping` -- the point-evaluable symbolic kind,
            i.e. an `AnalyticMapping` or a `DiscreteMapping` -- is accepted
            directly and unwrapped with `get_callable_mapping()` here, so
            callers don't have to do it themselves.

        space : TensorFemSpace, optional
            The discrete space to interpolate onto. When omitted, one is
            built from `ncells` and `degree` (see below) -- a convenience so
            callers don't need to hand-assemble a `SplineSpace`/
            `DomainDecomposition`/`TensorFemSpace` just to interpolate a
            mapping.

        ncells : Iterable[int], optional
            Number of cells along each logical dimension. Required (together
            with `degree`) when `space` is omitted; ignored otherwise.

        degree : Iterable[int], optional
            Spline degree along each logical dimension. Required (together
            with `ncells`) when `space` is omitted; ignored otherwise.

        periodic : Iterable[bool], optional
            Periodicity along each logical dimension, used only when building
            a `space`. Defaults to non-periodic in every direction.

        bounds : Iterable[tuple[float, float]], optional
            Per-direction `(min, max)` of the logical domain, used only when
            building a `space`. Defaults to `(0, 1)` in every direction.

        comm : MPI.Intracomm, optional
            MPI communicator for the domain decomposition, used only when
            building a `space`. Defaults to serial (`None`).

        Returns
        -------
        SplineCallableMapping

        Raises
        ------
        TypeError
            If called with the pre-2026-09 argument order
            (`from_mapping(space, mapping)`).

        ValueError
            If `space` is omitted and `ncells`/`degree` are not both given,
            or disagree in length.

        Examples
        --------
        >>> F_h = SplineCallableMapping.from_mapping(F, ncells=[8, 8], degree=[3, 3])

        Or onto a space you already have:

        >>> F_h = SplineCallableMapping.from_mapping(F, V)

        `F` may be the symbolic mapping itself -- these are equivalent:

        >>> F_h = SplineCallableMapping.from_mapping(F, V)
        >>> F_h = SplineCallableMapping.from_mapping(F.get_callable_mapping(), V)
        """
        # `mapping` used to come second, after a `tensor_space` that had to be
        # passed as an explicit `None` when unused. Catch that call shape
        # loudly: it would otherwise bind a space (or `None`) to `mapping` and
        # fail further in with a much less obvious message.
        if mapping is None or isinstance(mapping, TensorFemSpace):
            raise TypeError(
                "from_mapping's arguments were reordered: the mapping comes "
                "first now and the space is optional. Replace "
                "`from_mapping(V, F)` with `from_mapping(F, V)`, and "
                "`from_mapping(None, F, ncells=..., degree=...)` with "
                "`from_mapping(F, ncells=..., degree=...)`.")

        # Unwrap a point-evaluable symbolic mapping to the callable it
        # delegates to. For an `AnalyticMapping` this is a no-op
        # (`get_callable_mapping()` returns `self` since WP06c); for a
        # `DiscreteMapping` it reaches the spline underneath instead of
        # evaluating every Greville point through the symbolic wrapper.
        if isinstance(mapping, DefinedMapping):
            mapping = mapping.get_callable_mapping()
        if space is None:
            if ncells is None or degree is None:
                raise ValueError("Provide 'space', or both 'ncells' "
                                 "and 'degree' to build one.")
            ldim = len(ncells)
            if len(degree) != ldim:
                raise ValueError(f"'ncells' and 'degree' must have the same "
                                 f"length, got {len(ncells)} and {len(degree)}.")
            if periodic is None:
                periodic = [False] * ldim
            if bounds is None:
                bounds = [(0., 1.)] * ldim
            domain_decomposition = DomainDecomposition(ncells=ncells, periods=periodic, comm=comm)
            spaces_1d = [SplineSpace(degree=p, grid=np.linspace(*b, num=n + 1), periodic=per)
                        for b, n, p, per in zip(bounds, ncells, degree, periodic)]
            space = TensorFemSpace(domain_decomposition, *spaces_1d)

        assert isinstance(space, TensorFemSpace)
        assert isinstance(mapping, BasicCallableMapping)
        assert space.ldim == mapping.ldim

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        fields = [FemField(space) for d in range(mapping.pdim)]

        V = space.coeff_space
        values = [V.zeros() for d in range(mapping.pdim)]
        ranges = [range(s, e+1) for s, e in zip(V.starts, V.ends)]
        grids  = [sp.greville for sp in space.spaces]

        # Evaluate analytical mapping at Greville points (tensor-product grid)
        # and store vector values in one separate scalar field for each
        # physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        for index in product(*ranges):
            x = [grid[i] for grid, i in zip(grids, index)]
            u = mapping(*x)
            for d, ud in enumerate(u):
                values[d][index] = ud

        # Compute spline coefficients for each coordinate X_i
        for pvals, field in zip(values, fields):
            space.compute_interpolant(pvals, field)

        # Create SplineCallableMapping object
        return cls(*fields)

    #--------------------------------------------------------------------------
    # Option [2]: initialize from TensorFemSpace and spline control points
    #--------------------------------------------------------------------------
    @classmethod
    def from_control_points(cls, tensor_space, control_points):

        assert isinstance(tensor_space, TensorFemSpace)
        assert isinstance(control_points, (np.ndarray, h5py.Dataset))

        assert control_points.ndim       == tensor_space.ldim + 1
        assert control_points.shape[:-1] == tuple(V.nbasis for V in tensor_space.spaces)
        assert control_points.shape[ -1] >= tensor_space.ldim

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        fields = [FemField(tensor_space) for d in range(control_points.shape[-1])]

        # Get spline coefficients for each coordinate X_i
        starts = tensor_space.coeff_space.starts
        ends   = tensor_space.coeff_space.ends

        idx_to = tuple(slice(s, e+1) for s, e in zip(starts, ends))
        for i,field in enumerate(fields):
            idx_from = (*idx_to, i)
            field.coeffs[idx_to] = control_points[idx_from]
            field.coeffs.update_ghost_regions()

        # Create SplineCallableMapping object
        return cls(*fields)

    #--------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    def __call__(self, *eta):
        if not _is_scalar_eta(eta):
            fast = self._fast_tensor_grid(eta)
            if fast is not None:
                axes, perm = fast
                mesh = self.build_mesh(list(axes))
                return tuple(np.transpose(m, np.argsort(perm)) for m in mesh)
            return tuple(_eval_pointwise(self.__call__, eta, (self.pdim,)))
        return [map_Xd(*eta) for map_Xd in self._fields]

    # ...
    def jacobian(self, *eta):
        if not _is_scalar_eta(eta):
            fast = self._fast_tensor_grid(eta, square=True)
            if fast is not None:
                axes, perm = fast
                J = self.jac_mat_grid(list(axes))
                return _components_to_front(J, perm, 2)
            return _eval_pointwise(self.jacobian, eta, (self.pdim, self.ldim))
        return np.array([map_Xd.gradient(*eta) for map_Xd in self._fields])

    # ...
    def jacobian_inv(self, *eta):
        # WP14a follow-up /code-review: np.linalg.inv, like np.linalg.det,
        # needs a square matrix -- a non-square (surface) mapping's inverse
        # Jacobian is undefined regardless of pdim/ldim, so raise the same
        # clear NotImplementedError as jacobian_det/jac_det_grid/
        # inv_jac_mat_grid instead of leaving a raw numpy LinAlgError as the
        # answer for the newly-array-capable case (scalar eta on a surface
        # mapping already raised LinAlgError before this WP; array eta is new
        # here, via the loop below, and would otherwise hit that same error
        # on its first point with no clearer message).
        self._require_square_jacobian()
        if not _is_scalar_eta(eta):
            fast = self._fast_tensor_grid(eta, square=True)
            if fast is not None:
                axes, perm = fast
                J = self.jac_mat_grid(list(axes))
                return _components_to_front(np.linalg.inv(J), perm, 2)
            return _eval_pointwise(self.jacobian_inv, eta, (self.ldim, self.pdim))
        return np.linalg.inv(self.jacobian(*eta))

    # ...
    def metric(self, *eta):
        if not _is_scalar_eta(eta):
            fast = self._fast_tensor_grid(eta, square=True)
            if fast is not None:
                axes, perm = fast
                J = self.jac_mat_grid(list(axes))
                metric_arr = np.einsum('...ki,...kj->...ij', J, J)
                return _components_to_front(metric_arr, perm, 2)
            return _eval_pointwise(self.metric, eta, (self.ldim, self.ldim))
        J = self.jacobian(*eta)
        return np.dot(J.T, J)

    # ...
    def metric_det(self, *eta):
        if not _is_scalar_eta(eta):
            fast = self._fast_tensor_grid(eta, square=True)
            if fast is not None:
                axes, perm = fast
                J = self.jac_mat_grid(list(axes))
                metric_arr = np.einsum('...ki,...kj->...ij', J, J)
                return _components_to_front(np.linalg.det(metric_arr), perm, 0)
            return _eval_pointwise(self.metric_det, eta, ())
        return np.linalg.det(self.metric(*eta))

    @property
    def ldim(self):
        return self._ldim

    @property
    def pdim(self):
        return self._pdim

    #--------------------------------------------------------------------------
    # Symbolic carrier
    #--------------------------------------------------------------------------
    def to_defined_mapping(self, name, *, ldim=None, pdim=None):
        """
        Wrap this spline in a fresh :class:`~sympde.topology.DiscreteMapping`.

        The result is a symbolic :class:`DefinedMapping` (it has a name, is
        callable on a topological domain, and appears as ``domain.mapping``)
        whose ``get_callable_mapping()`` returns this ``SplineCallableMapping``. Use it
        to give a spline geometry a first-class symbolic identity without
        mutating some analytic mapping via ``set_callable_mapping``.

        Parameters
        ----------
        name : str
            Non-empty symbolic name for the mapping (a ``SplineCallableMapping`` built by
            ``from_mapping`` has no name of its own).
        ldim, pdim : int, optional
            If given, must equal this spline's ``ldim`` / ``pdim``.

        Returns
        -------
        DiscreteMapping

        Raises
        ------
        ValueError
            If ``name`` is empty, or ``ldim`` / ``pdim`` contradicts the spline
            (both raised by :class:`~sympde.topology.DiscreteMapping`).

        Examples
        --------
        >>> F_h = SplineCallableMapping.from_mapping(V, F)
        >>> G   = F_h.to_defined_mapping('F')
        >>> G.get_callable_mapping() is F_h
        True
        """
        return DiscreteMapping(self, name, ldim=ldim, pdim=pdim)

    #--------------------------------------------------------------------------
    # Fast evaluation on a grid
    #--------------------------------------------------------------------------
    def _fast_tensor_grid(self, eta, *, square=False):
        """WP14b: is `eta` an array input the compiled-kernel grid machinery
        can evaluate directly, in place of `_eval_pointwise`'s per-point loop?

        `_tensor_axes_of_meshgrid` handles the structural test; this method
        adds the caller-specific gates that make routing through
        `build_mesh`/`jac_mat_grid` safe: serial only (`preprocess_*` slice
        distributed grids to the local domain, incompatible with the "shaped
        like `eta`" contract), a supported `ldim`, a square Jacobian when
        `square=True` (the `jacobian`/`jacobian_inv`/`metric`/`metric_det`
        family, since `jac_mat_grid` itself requires `pdim == ldim`), and
        every axis inside `self.space.breaks` (the kernels raise on
        out-of-range points instead of extrapolating like the scalar path).

        Parameters
        ----------
        eta : tuple of array_like
            One entry per logical coordinate.
        square : bool, optional
            `True` for the Jacobian-derived methods (default `False`, used
            by `__call__`).

        Returns
        -------
        (axes, perm) or None
            See `_tensor_axes_of_meshgrid`. `None` means: fall back to the
            per-point loop.

        Examples
        --------
        >>> X1, X2 = np.meshgrid([0.2, 0.5], [0.3, 0.7, 0.9], indexing='ij')
        >>> spline_mapping._fast_tensor_grid((X1, X2)) is not None  # eligible
        True
        >>> spline_mapping._fast_tensor_grid((0.3, 0.4)) is None  # not a grid
        True
        """
        # Cheap gates first: _tensor_axes_of_meshgrid does an O(ldim^2 * size)
        # scan over the actual data, wasted work if a cheap check below would
        # reject anyway (e.g. every call on a parallel mapping, or every
        # jacobian_inv/metric/metric_det call on a surface mapping).
        if self.space.coeff_space.parallel:
            return None
        if len(eta) != self.ldim or self.ldim not in (1, 2, 3):
            return None
        if square and (self.pdim != self.ldim or self.ldim == 1):
            return None

        result = _tensor_axes_of_meshgrid(eta)
        if result is None:
            return None
        axes, perm = result

        for i, axis in enumerate(axes):
            lo, hi = self.space.breaks[i][0], self.space.breaks[i][-1]
            if axis[0] < lo or axis[-1] > hi:
                return None

        return axes, perm

    def build_mesh(self, grid, npts_per_cell=None, overlap=0):
        """Evaluation of the mapping on the given grid.

        Parameters
        ----------
        grid : List of ndarray
            Grid on which to evaluate the fields.
            Each array in this list corresponds to one logical coordinate.

        npts_per_cell: int, tuple of int or None, optional
            Number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        mesh: tuple
            ldim ldim-D arrays. One for each component.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """

        mesh = self.space.eval_fields(grid, *self._fields, npts_per_cell=npts_per_cell, overlap=overlap)
        return mesh

    # ...
    def _require_square_jacobian(self):
        """Raise if this mapping's Jacobian is not square (`pdim != ldim`).

        Shared by `jac_mat_grid`/`inv_jac_mat_grid`/`jac_det_grid`'s kernel
        paths (`eval_jacobians_*` reads only the first `ldim` fields, silently
        truncating a surface mapping's Jacobian otherwise), by
        `inv_jac_mat_grid`/`jac_det_grid`'s Case-1 scalar path, and by
        `jacobian_inv`/`jacobian_det` directly (the inverse/determinant of a
        non-square matrix isn't defined, regardless of how it's evaluated).
        `jac_mat_grid`'s own Case 1 doesn't need this -- it returns
        `jacobian(...)`'s raw, possibly non-square matrix -- and neither do
        `metric`/`metric_det` (`J.T @ J` is always square).

        Raises
        ------
        NotImplementedError
            If `pdim != ldim`.

        Examples
        --------
        >>> spline_mapping._require_square_jacobian()          # pdim == ldim: no-op
        >>> surface_mapping._require_square_jacobian()          # pdim=3, ldim=2
        Traceback (most recent call last):
            ...
        NotImplementedError: Grid evaluation of the Jacobian needs pdim == ldim, ...
        """
        if self.pdim != self.ldim:
            raise NotImplementedError(
                f'Grid evaluation of the Jacobian needs pdim == ldim, got '
                f'pdim={self.pdim}, ldim={self.ldim}; the kernels read only the '
                f'first {self.ldim} components. Use jacobian(*eta) instead.')

    # ...
    def jac_mat_grid(self, grid, npts_per_cell=None, overlap=0):
        """Evaluates the Jacobian matrix of the mapping at the given location(s) grid.

        Parameters
        ----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        array_like
            Jacobian matrix at the location(s) grid.

        See Also
        --------
        mapping.SplineCallableMapping.inv_jac_mat_grid : Evaluates the inverse
            of the Jacobian matrix of the mapping at the given location(s) grid.
        mapping.SplineCallableMapping.metric_det_grid : Evaluates the metric determinant
            of the mapping at the given location(s) grid.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # --------------------------
        # Case 1. Scalar coordinates -- works for any pdim/ldim: it's just
        # self.jacobian(...)'s raw (pdim, ldim) matrix, no det/inverse needed.
        if all(g.size == 1 for g in grid) or grid[0].ndim == 0:
            return self.jacobian(*(np.asarray(g).item() for g in grid))

        self._require_square_jacobian()

        # Case 2. 1D array of coordinates and no npts_per_cell is given
        # -> grid is tensor-product, but npts_per_cell is not the same in each cell
        if grid[0].ndim == 1 and npts_per_cell is None:
            jac_mats = self.jac_mat_irregular_tensor_grid(grid, overlap=overlap)
            return jac_mats

        # Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer
        # -> grid is tensor-product, and each cell has the same number of evaluation points
        elif grid[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * self.ldim
            for i in range(self.ldim):
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], (ncells_i, npts_per_cell[i]))
            jac_mats = self.jac_mat_regular_tensor_grid(grid, overlap=overlap)
            return jac_mats

        # Case 4. (self.ldim)D arrays of coordinates and no npts_per_cell
        # -> unstructured grid
        elif grid[0].ndim == self.ldim and npts_per_cell is None:
            raise NotImplementedError("Unstructured grids are not supported yet.")

        # Case 5. Nonsensical input
        else:
            raise ValueError("This combination of argument isn't understood. The 4 cases understood are :\n"
                             "Case 1. Scalar coordinates\n"
                             "Case 2. 1D array of coordinates and no npts_per_cell is given\n"
                             "Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer\n"
                             "Case 4. {0}D arrays of coordinates and no npts_per_cell".format(self.ldim))

    # ...
    def jac_mat_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian matrix on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the Jacobian matrix at the location corresponding
            to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jacobians_2d, eval_jacobians_3d

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                            + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_3d(*ncells, *degree, *n_eval_points, *global_basis, 
                              *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                              jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_2d(*ncells, *degree, *n_eval_points, *global_basis, 
                              *global_spans, global_arr_x, global_arr_y, jac_mats)

        else:
            raise NotImplementedError("TODO")

        return jac_mats

    # ...
    def jac_mat_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian matrix on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.
            
        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the Jacobian matrix at the location corresponding
            to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jacobians_irregular_2d, eval_jacobians_irregular_3d

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        jac_mats = np.zeros(tuple(local_shape) + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_irregular_3d(*npts, *degree, *cell_indexes, *global_basis, 
                                        *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                        jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_irregular_2d(*npts, *degree, *cell_indexes, *global_basis, 
                                        *global_spans, global_arr_x, global_arr_y, jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return jac_mats

    # ...
    def inv_jac_mat_grid(self, grid, npts_per_cell=None, overlap=0):
        """Evaluates the inverse of the Jacobian matrix of the mapping at the given location(s) grid.

        Parameters
        ----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        array_like
            Inverse of the Jacobian matrix at the location(s) grid.

        See Also
        --------
        mapping.SplineCallableMapping.jac_mat_grid : Evaluates the Jacobian matrix
            of the mapping at the given location(s) `grid`.
        mapping.SplineCallableMapping.metric_det_grid : Evaluates the metric determinant
            of the mapping at the given location(s) `grid`.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # Unlike jac_mat_grid's Case 1, jacobian_inv is only defined for a
        # square Jacobian -- this must gate Case 1 too, not just the kernel
        # path below, or a surface mapping hits a confusing LinAlgError from
        # np.linalg.inv instead of this clear NotImplementedError.
        self._require_square_jacobian()

        # --------------------------
        # Case 1. Scalar coordinates
        if all(g.size == 1 for g in grid) or grid[0].ndim == 0:
            return self.jacobian_inv(*(np.asarray(g).item() for g in grid))

        # Case 2. 1D array of coordinates and no npts_per_cell is given
        # -> grid is tensor-product, but npts_per_cell is not the same in each cell
        elif grid[0].ndim == 1 and npts_per_cell is None:
            inv_jac_mats = self.inv_jac_mat_irregular_tensor_grid(grid, overlap=overlap)
            return inv_jac_mats

        # Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer
        # -> grid is tensor-product, and each cell has the same number of evaluation points
        elif grid[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * self.ldim
            for i in range(self.ldim):
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], (ncells_i, npts_per_cell[i]))
            inv_jac_mats = self.inv_jac_mat_regular_tensor_grid(grid, overlap=overlap)
            return inv_jac_mats

        # Case 4. (self.ldim)D arrays of coordinates and no npts_per_cell
        # -> unstructured grid
        elif grid[0].ndim == self.ldim and npts_per_cell is None:
            raise NotImplementedError("Unstructured grids are not supported yet.")

        # Case 5. Nonsensical input
        else:
            raise ValueError("This combination of argument isn't understood. The 4 cases understood are :\n"
                             "Case 1. Scalar coordinates\n"
                             "Case 2. 1D array of coordinates and no npts_per_cell is given\n"
                             "Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer\n"
                             "Case 4. {0}D arrays of coordinates and no npts_per_cell".format(self.ldim))

    # ...
    def inv_jac_mat_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the inverse of the Jacobian matrix on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        inv_jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the Jacobian matrix
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jacobians_inv_2d, eval_jacobians_inv_3d

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        inv_jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                                + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_3d(*ncells, *degree, *n_eval_points, *global_basis, 
                                  *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                  inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_2d(*ncells, *degree, *n_eval_points, *global_basis, 
                                  *global_spans, global_arr_x, global_arr_y, inv_jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return inv_jac_mats

    # ...
    def inv_jac_mat_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the inverse of the Jacobian matrix on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        inv_jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the Jacobian matrix
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import (eval_jacobians_inv_irregular_2d,
                                                          eval_jacobians_inv_irregular_3d)

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        inv_jac_mats = np.zeros(tuple(local_shape) + (self.ldim, self.ldim))

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_irregular_3d(*npts, *degree, *cell_indexes, *global_basis, 
                                            *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                            inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_irregular_2d(*npts, *degree, *cell_indexes, *global_basis, 
                                            *global_spans, global_arr_x, global_arr_y, inv_jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return inv_jac_mats

    # ...
    def jac_det_grid(self, grid, npts_per_cell=None, overlap=0):
        """Evaluates the Jacobian determinant of the mapping at the given location(s) grid.

        Parameters
        ----------
        grid : List of array_like
            Grid on which to evaluate the fields

        npts_per_cell: int or tuple of int or None, optional
            number of evaluation points in each cell.
            If an integer is given, then assume that it is the same in every direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        array_like
            Jacobian determinant at the location(s) grid.

        See Also
        --------
        mapping.SplineCallableMapping.jac_mat_grid : Evaluates the Jacobian matrix
            of the mapping at the given location(s) grid.
        mapping.SplineCallableMapping.inv_jac_mat_grid : Evaluates the inverse
            of the Jacobian matrix of the mapping at the given location(s) grid.
        """

        assert len(grid) == self.ldim
        grid = [np.asarray(grid[i]) for i in range(self.ldim)]
        assert all(grid[i].ndim == grid[i + 1].ndim for i in range(self.ldim - 1))

        # Unlike jac_mat_grid's Case 1, a Jacobian determinant is only defined
        # for a square Jacobian -- this must gate Case 1 too, not just the
        # kernel path below, or a surface mapping hits a confusing LinAlgError
        # from np.linalg.det instead of this clear NotImplementedError.
        self._require_square_jacobian()

        # --------------------------
        # Case 1. Scalar coordinates -- delegate to jacobian_det (like
        # inv_jac_mat_grid delegates to jacobian_inv below), so the two
        # det-of-jacobian implementations can't drift apart.
        if all(g.size == 1 for g in grid) or grid[0].ndim == 0:
            return self.jacobian_det(*(np.asarray(g).item() for g in grid))

        # Case 2. 1D array of coordinates and no npts_per_cell is given
        # -> grid is tensor-product, but npts_per_cell is not the same in each cell
        elif grid[0].ndim == 1 and npts_per_cell is None:
            jac_dets = self.jac_det_irregular_tensor_grid(grid, overlap=overlap)
            return jac_dets

        # Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer
        # -> grid is tensor-product, and each cell has the same number of evaluation points
        elif grid[0].ndim == 1 and npts_per_cell is not None:
            if isinstance(npts_per_cell, int):
                npts_per_cell = (npts_per_cell,) * self.ldim
            for i in range(self.ldim):
                ncells_i = len(self.space.breaks[i]) - 1
                grid[i] = np.reshape(grid[i], (ncells_i, npts_per_cell[i]))
            jac_dets = self.jac_det_regular_tensor_grid(grid, overlap=overlap)
            return jac_dets

        # Case 4. (self.ldim)D arrays of coordinates and no npts_per_cell
        # -> unstructured grid
        elif grid[0].ndim == self.ldim and npts_per_cell is None:
            raise NotImplementedError("Unstructured grids are not supported yet.")

        # Case 5. Nonsensical input
        else:
            raise ValueError("This combination of argument isn't understood. The 4 cases understood are :\n"
                             "Case 1. Scalar coordinates\n"
                             "Case 2. 1D array of coordinates and no npts_per_cell is given\n"
                             "Case 3. 1D arrays of coordinates and npts_per_cell is a tuple or an integer\n"
                             "Case 4. {0}D arrays of coordinates and no npts_per_cell".format(self.ldim))

    # ...
    def jac_det_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian determinant on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_dets : ndarray
            ``self.ldim`` D array of shape ``(n_x_1, ..., n_x_ldim)``.
            ``jac_dets[x_1, ..., x_ldim]`` is the Jacobian determinant
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jac_det_3d, eval_jac_det_2d

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, 
                                                                                                    overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        jac_dets = np.zeros(shape=tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim)), dtype=self._fields[0].coeffs.dtype)

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jac_det_3d(*ncells, *degree, *n_eval_points, *global_basis, 
                            *global_spans, global_arr_x, global_arr_y, global_arr_z, jac_dets)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jac_det_2d(*ncells, *degree, *n_eval_points, *global_basis, 
                            *global_spans, global_arr_x, global_arr_y, jac_dets)

        else:
            raise NotImplementedError("TODO")

        return jac_dets

    # ...
    def jac_det_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian determinant on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.
            
        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_dets : ndarray
            ``self.ldim`` D array of shape ``(n_x_1, ..., n_x_ldim)``.
            ``jac_dets[x_1, ..., x_ldim]`` is the Jacobian determinant
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jac_det_irregular_3d, eval_jac_det_irregular_2d

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        jac_dets = np.zeros(local_shape)

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jac_det_irregular_3d(*npts, *degree, *cell_indexes, *global_basis, 
                                      *global_spans, global_arr_x, global_arr_y, global_arr_z, jac_dets)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jac_det_irregular_2d(*npts, *degree, *cell_indexes, *global_basis, 
                                      *global_spans, global_arr_x, global_arr_y, jac_dets)

        else:
            raise NotImplementedError("TODO")

        return jac_dets

    #--------------------------------------------------------------------------
    # Other properties/methods
    #--------------------------------------------------------------------------
    def jacobian_det(self, *eta):
        # WP14a: jacobian(*eta) now returns array results with the component
        # axes first (shape (pdim, ldim) + S), so np.linalg.det -- which needs
        # the matrix axes last -- can't be applied to it directly for array
        # eta. Delegate to the pointwise loop instead, matching metric_det.
        # The square-Jacobian requirement (unlike jacobian_inv/metric, whose
        # pre-existing scalar bodies this WP doesn't touch) is checked here
        # explicitly: before the A4 fix this method always raised
        # AttributeError regardless of pdim/ldim (self.jac_mat never
        # existed), so this is the first time its non-square behaviour is
        # reachable at all -- match jac_det_grid's clear NotImplementedError
        # rather than leaving a fresh numpy LinAlgError as its scalar answer.
        self._require_square_jacobian()
        if not _is_scalar_eta(eta):
            # WP14b: route through jac_det_grid's own dedicated kernel
            # (eval_jac_det_*) rather than jac_mat_grid + np.linalg.det --
            # cheaper, since it never assembles the full Jacobian matrix.
            fast = self._fast_tensor_grid(eta, square=True)
            if fast is not None:
                axes, perm = fast
                return _components_to_front(self.jac_det_grid(list(axes)), perm, 0)
            return _eval_pointwise(self.jacobian_det, eta, ())
        return np.linalg.det(self.jacobian(*eta))

    @property
    def space(self):
        return self._space

    @property
    def fields(self):
        return self._fields

    @property
    def control_points(self):
        return self._control_points

    # TODO: move to 'Geometry' class in 'psydac.cad.geometry' module
    def export(self, filename):
        """
        Export tensor-product spline space and mapping to geometry file in HDF5
        format (single-patch only).

        Parameters
        ----------
        filename : str
          Name of HDF5 output file.

        """
        space = self.space
        comm  = space.coeff_space.cart.comm

        # Create dictionary with geometry metadata
        yml = {}
        yml['ldim'] = self.ldim
        yml['pdim'] = self.pdim
        yml['patches'] = [{ [('name' , 'patch_{}'.format( 0 ) ),
                                        ('type' , 'cad_nurbs'            ),
                                        ('color', 'None'                 )] }]
        yml['internal_faces'] = []
        yml['external_faces'] = [[0,i] for i in range( 2*self.ldim )]
        yml['connectivity'  ] = []

        # Dump geometry metadata to string in YAML file format
        geo = yaml.dump(data = yml, sort_keys = False)

        # Create HDF5 file (in parallel mode if MPI communicator size > 1)
        kwargs = dict( driver='mpio', comm=comm ) if comm.size > 1 else {}
        h5 = h5py.File( filename, mode='w', **kwargs )

        # Write geometry metadata as fixed-length array of ASCII characters
        h5['geometry.yml'] = np.array( geo, dtype='S' )

        # Create group for patch 0
        group = h5.create_group( yml['patches'][0]['name'] )
        group.attrs['shape'      ] = space.coeff_space.npts
        group.attrs['degree'     ] = space.degree
        group.attrs['periodic'   ] = space.periodic
        for d in range( self.pdim ):
            group['knots_{}'.format( d )] = space.spaces[d].knots

        # Collective: create dataset for control points
        shape = [n for n in space.coeff_space.npts] + [self.pdim]
        dtype = space.coeff_space.dtype
        dset  = group.create_dataset( 'points', shape=shape, dtype=dtype )

        # Independent: write control points to dataset
        starts = space.coeff_space.starts
        ends   = space.coeff_space.ends
        index  = [slice(s, e+1) for s, e in zip(starts, ends)] + [slice(None)]
        index  = tuple( index )
        dset[index] = self.control_points[index]

        # Close HDF5 file
        h5.close()

    #==========================================================================
    class ControlPoints:
        """ Convenience object to access control points.

        """
        # TODO: should not allow access to ghost regions

        def __init__(self, mapping):
            assert isinstance(mapping, SplineCallableMapping)
            self._mapping = mapping

        # ...
        @property
        def mapping(self):
            return self._mapping

        # ...
        def __getitem__(self, key):

            m = self._mapping

            if key is Ellipsis:
                key = tuple(slice(None) for i in range(m.ldim + 1))
            elif isinstance(key, tuple):
                assert len(key) == m.ldim + 1
            else:
                raise ValueError(key)

            pnt_idx = key[:-1]
            dim_idx = key[-1]

            if isinstance(dim_idx, slice):
                dim_idx = range(*dim_idx.indices(m.pdim))
                coeffs = np.array([m.fields[d].coeffs[pnt_idx] for d in dim_idx])
                coords = np.moveaxis(coeffs, 0, -1)
            else:
                coords = np.array(m.fields[dim_idx].coeffs[pnt_idx])

            return coords

# SplineCallableMapping implements only BasicCallableMapping (its literal base,
# above): the plain abc.ABC that declares __call__ / jacobian / jacobian_inv /
# metric / metric_det / ldim / pdim, with no sympy in its MRO. It is
# deliberately NOT a DefinedMapping/SymbolicMapping: it has no name and is not
# callable on a topological domain. Use to_defined_mapping(name) to wrap it in
# a DiscreteMapping and get a symbolic identity. It cannot literally subclass
# DefinedMapping: that MRO carries sympy's IndexedBase (via
# SymbolicMapping), whose __new__ would eat SplineCallableMapping's
# (FemField, FemField, ...) constructor args as a symbolic (label, shape)
# pair. The WP04 registration that made isinstance(_, DefinedMapping) True for
# splines was removed in WP12 (D1). NurbsCallableMapping inherits the same
# relationship (real subclass of SplineCallableMapping).

#==============================================================================
class NurbsCallableMapping(SplineCallableMapping):

    #: Tag written into / read back from the 'type' field of a geometry
    #: file's geometry.yml. Frozen at the historical class name so that
    #: renaming the Python class never changes the HDF5 format.
    geometry_dtype = 'NurbsMapping'

    def __init__(self, *components, name=None):

        weights    = components[-1]
        components = components[:-1]

        SplineCallableMapping.__init__(self, *components, name=name)

        self._weights = NurbsCallableMapping.Weights(self)
        self._weights_field = weights

    #--------------------------------------------------------------------------
    # Option [2]: initialize from TensorFemSpace and spline control points
    #--------------------------------------------------------------------------
    @classmethod
    def from_control_points_weights(cls, tensor_space, control_points, weights):

        assert isinstance(tensor_space, TensorFemSpace)
        assert isinstance(control_points, (np.ndarray, h5py.Dataset))
        assert isinstance(weights, (np.ndarray, h5py.Dataset))

        assert control_points.ndim       == tensor_space.ldim + 1
        assert control_points.shape[:-1] == tuple(V.nbasis for V in tensor_space.spaces)
        assert control_points.shape[ -1] >= tensor_space.ldim
        assert weights.shape == tuple(V.nbasis for V in tensor_space.spaces)

        # Create one separate scalar field for each physical dimension
        # TODO: use one unique field belonging to VectorFemSpace
        fields  = [FemField(tensor_space) for d in range(control_points.shape[-1])]
        fields += [FemField(tensor_space)]

        # Get spline coefficients for each coordinate X_i
        # we store w*x where w is the weight and x is the control point
        starts = tensor_space.coeff_space.starts
        ends   = tensor_space.coeff_space.ends
        idx_to = tuple(slice(s, e+1) for s,e in zip(starts, ends))
        for i, field in enumerate(fields[:-1]):
            idx_from = (*idx_to, i)
#            idw_from = tuple(idx_to)
            field.coeffs[idx_to] = control_points[idx_from] #* weights[idw_from]

        # weights
        idx_from = tuple(idx_to)
        fields[-1].coeffs[idx_to] = weights[idx_from]

        # Create SplineCallableMapping object
        return cls(*fields)

    #--------------------------------------------------------------------------
    # Abstract interface
    #--------------------------------------------------------------------------
    def __call__(self, *eta):
        if not _is_scalar_eta(eta):
            fast = self._fast_tensor_grid(eta)
            if fast is not None:
                axes, perm = fast
                mesh = self.build_mesh(list(axes))
                return tuple(np.transpose(m, np.argsort(perm)) for m in mesh)
            return tuple(_eval_pointwise(self.__call__, eta, (self.pdim,)))
        map_W = self._weights_field
        w = map_W(*eta)
        Xd = [map_Xd(*eta , weights=map_W.coeffs) for map_Xd in self._fields]
        return np.asarray(Xd) / w

    # ...
    def jacobian(self, *eta):
        if not _is_scalar_eta(eta):
            fast = self._fast_tensor_grid(eta, square=True)
            if fast is not None:
                axes, perm = fast
                J = self.jac_mat_grid(list(axes))
                return _components_to_front(J, perm, 2)
            return _eval_pointwise(self.jacobian, eta, (self.pdim, self.ldim))
        map_W = self._weights_field
        w = map_W(*eta)
        grad_w = np.array(map_W.gradient(*eta))
        v = np.array([map_Xd(*eta, weights=map_W.coeffs)  for map_Xd in self._fields])
        grad_v = np.array([map_Xd.gradient(*eta, weights=map_W.coeffs) for map_Xd in self._fields])
        return grad_v / w - v[:, None] @ grad_w[None, :] / w**2

    #--------------------------------------------------------------------------
    # Fast evaluation on a grid
    #--------------------------------------------------------------------------
    def build_mesh(self, grid, npts_per_cell=None, overlap=0):
        """Evaluation of the mapping on the given grid.

        Parameters
        ----------
        grid : List of ndarray
            Each array in the list should correspond to a logical coordinate.

        npts_per_cell : int, tuple of int or None, optional

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        mesh: tuple
            ldim ldim-D arrays. One for each component.

        See Also
        --------
        psydac.fem.tensor.TensorFemSpace.eval_fields : More information about the grid parameter.
        """
        mesh = self.space.eval_fields(grid, *self._fields, npts_per_cell=npts_per_cell, weights=self._weights_field, overlap=overlap)
        return mesh

    # ...
    def jac_mat_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian matrix on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the Jacobian matrix at the location corresponding
            to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jacobians_2d_weights, eval_jacobians_3d_weights

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                            + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_3d_weights(ncells[0], ncells[1], ncells[2], degree[0], degree[1],
                                      degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2], global_basis[0],
                                      global_basis[1], global_basis[2], global_spans[0], global_spans[1],
                                      global_spans[2], global_arr_x, global_arr_y, global_arr_z, global_arr_weights,
                                      jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_2d_weights(ncells[0], ncells[1], degree[0], degree[1], n_eval_points[0],
                                      n_eval_points[1], global_basis[0], global_basis[1], global_spans[0],
                                      global_spans[1], global_arr_x, global_arr_y, global_arr_weights, jac_mats)

        else:
            raise NotImplementedError("1D case not Implemented")

        return jac_mats

    # ...
    def jac_mat_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian matrix on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the Jacobian matrix at the location corresponding
            to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import (eval_jacobians_irregular_2d_weights,
                                                          eval_jacobians_irregular_3d_weights)

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        jac_mats = np.zeros(tuple(local_shape) + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_irregular_3d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                                *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                                global_arr_weights, jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_irregular_2d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                                *global_spans, global_arr_x, global_arr_y, 
                                                global_arr_weights, jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return jac_mats

    # ...
    def inv_jac_mat_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the inverse of the Jacobian matrix on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        inv_jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the Jacobian matrix a
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """

        from psydac.core.field_evaluation_kernels import eval_jacobians_inv_2d_weights, eval_jacobians_inv_3d_weights

        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        inv_jac_mats = np.zeros(tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim))
                                + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_3d_weights(ncells[0], ncells[1], ncells[2], degree[0],
                                          degree[1], degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2],
                                          global_basis[0], global_basis[1], global_basis[2], global_spans[0],
                                          global_spans[1], global_spans[2], global_arr_x, global_arr_y, global_arr_z,
                                          global_arr_weights, inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_2d_weights(ncells[0], ncells[1], degree[0], degree[1],
                                          n_eval_points[0], n_eval_points[1], global_basis[0], global_basis[1],
                                          global_spans[0], global_spans[1], global_arr_x, global_arr_y,
                                          global_arr_weights, inv_jac_mats)

        else:
            raise NotImplementedError("1D case not Implemented")

        return inv_jac_mats

    # ...
    def inv_jac_mat_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the inverse of the Jacobian matrix on an irregular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 1D arrays representing each direction of the grid.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        inv_jac_mats : ndarray
            ``self.ldim + 2`` D array of shape ``(n_x_1, ..., n_x_ldim, ldim, ldim)``.
            ``jac_mats[x_1, ..., x_ldim]`` is the inverse of the Jacobian matrix
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import (eval_jacobians_inv_irregular_2d_weights,
                                                          eval_jacobians_inv_irregular_3d_weights)

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        inv_jac_mats = np.zeros(tuple(local_shape) + (self.ldim, self.ldim))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jacobians_inv_irregular_3d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                                    *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                                    global_arr_weights, inv_jac_mats)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jacobians_inv_irregular_2d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                                    *global_spans, global_arr_x, global_arr_y, 
                                                    global_arr_weights, inv_jac_mats)

        else:
            raise NotImplementedError("1D case not supported")

        return inv_jac_mats

    # ...
    def jac_det_regular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian determinant on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_dets : ndarray
            ``self.ldim`` D array of shape ``(n_x_1, ..., n_x_ldim)``.
            ``jac_dets[x_1, ..., x_ldim]`` is the Jacobian determinant
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import eval_jac_det_3d_weights, eval_jac_det_2d_weights
        
        degree, global_basis, global_spans, local_shape = self.space.preprocess_regular_tensor_grid(grid, der=1, overlap=overlap)

        ncells = [local_shape[i][0] for i in range(self.ldim)]
        n_eval_points = [local_shape[i][1] for i in range(self.ldim)]

        jac_dets = np.zeros(shape=tuple(ncells[i] * n_eval_points[i] for i in range(self.ldim)))

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jac_det_3d_weights(ncells[0], ncells[1], ncells[2], degree[0], degree[1],
                                    degree[2], n_eval_points[0], n_eval_points[1], n_eval_points[2], global_basis[0],
                                    global_basis[1], global_basis[2], global_spans[0], global_spans[1],
                                    global_spans[2], global_arr_x, global_arr_y, global_arr_z, global_arr_weights,
                                    jac_dets)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jac_det_2d_weights(ncells[0], ncells[1], degree[0], degree[1], n_eval_points[0],
                                    n_eval_points[1], global_basis[0], global_basis[1], global_spans[0],
                                    global_spans[1], global_arr_x, global_arr_y, global_arr_weights, jac_dets)

        else:
            raise NotImplementedError("1D case not Implemented")

        return jac_dets

    # ...
    def jac_det_irregular_tensor_grid(self, grid, overlap=0):
        """Evaluates the Jacobian determinant on a regular tensor product grid.

        Parameters
        ----------
        grid : List of ndarray
            List of 2D arrays representing each direction of the grid.
            Each of these arrays should have shape (ne_xi, nv_xi) where ne_xi is the
            number of cells in the domain in the direction xi and nv_xi is the number of
            evaluation points in the same direction.

        overlap : int
            How much to overlap. Only used in the distributed context.

        Returns
        -------
        jac_dets : ndarray
            ``self.ldim`` D array of shape ``(n_x_1, ..., n_x_ldim)``.
            ``jac_dets[x_1, ..., x_ldim]`` is the Jacobian determinant
            at the location corresponding to ``(x_1, ..., x_ldim)``.
        """
        from psydac.core.field_evaluation_kernels import (eval_jac_det_irregular_3d_weights,
                                                          eval_jac_det_irregular_2d_weights)

        degree, global_basis, global_spans, cell_indexes, \
        local_shape = self.space.preprocess_irregular_tensor_grid(grid, der=1, overlap=overlap)

        npts = local_shape

        jac_dets = np.zeros(local_shape)

        global_arr_weights = self._weights_field.coeffs._data

        if self.ldim == 3:

            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data
            global_arr_z = self._fields[2].coeffs._data

            eval_jac_det_irregular_3d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                              *global_spans, global_arr_x, global_arr_y, global_arr_z, 
                                              global_arr_weights, jac_dets)

        elif self.ldim == 2:
            global_arr_x = self._fields[0].coeffs._data
            global_arr_y = self._fields[1].coeffs._data

            eval_jac_det_irregular_2d_weights(*npts, *degree, *cell_indexes, *global_basis, 
                                              *global_spans, global_arr_x, global_arr_y, 
                                              global_arr_weights, jac_dets)

        else:
            raise NotImplementedError("TODO")

        return jac_dets

    #--------------------------------------------------------------------------
    # Other properties/methods
    #--------------------------------------------------------------------------
    @property
    def weights_field( self ):
        return self._weights_field

    @property
    def weights( self ):
        return self._weights

    # TODO: move to 'Geometry' class in 'psydac.cad.geometry' module
    def export( self, filename ):
        """
        Export tensor-product spline space and mapping to geometry file in HDF5
        format (single-patch only).

        Parameters
        ----------
        filename : str
          Name of HDF5 output file.

        """
        raise NotImplementedError('')

    #==========================================================================
    class Weights:
        """ Convenience object to access weights.

        """
        # TODO: should not allow access to ghost regions

        def __init__( self, mapping ):
            assert isinstance( mapping, NurbsCallableMapping )
            self._mapping = mapping

        # ...
        @property
        def mapping( self ):
            return self._mapping

        # ...
        def __getitem__( self, key ):

            m = self._mapping

            if key is Ellipsis:
                key = tuple( slice( None ) for i in range( m.ldim ) )
            elif isinstance( key, tuple ):
                assert len( key ) == m.ldim
            else:
                raise ValueError( key )

            pnt_idx = key[:]

            return np.array( m._weights_field.coeffs[pnt_idx] )

#==============================================================================
# Deprecated aliases (pure re-export, PEP 562): 'SplineMapping' /
# 'NurbsMapping' were renamed to 'SplineCallableMapping' /
# 'NurbsCallableMapping'. Kept as module-level identity aliases (not
# subclasses) so isinstance/issubclass stay correct for objects built via the
# new name.
_DEPRECATED_NAMES = {
    'SplineMapping': SplineCallableMapping,
    'NurbsMapping' : NurbsCallableMapping,
}

def __getattr__(name):
    cls = _DEPRECATED_NAMES.get(name)
    if cls is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"psydac.mapping.discrete.{name} is deprecated; "
        f"use {cls.__name__} (it implements sympde's BasicCallableMapping, "
        f"not the symbolic DiscreteMapping).",
        DeprecationWarning, stacklevel=2,
    )
    return cls

def __dir__():
    return sorted([*globals(), *_DEPRECATED_NAMES])
