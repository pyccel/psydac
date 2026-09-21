#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#

# a Geometry class contains the list of patches and additional information about
# the topology i.e. connectivity, boundaries
# For the moment, it is used as a container, that can be loaded from a file
# (hdf5)
import os
import warnings
from typing import Iterable
from itertools import chain

import numpy as np
import h5py
import yaml
from mpi4py import MPI

from sympde.topology       import Domain, Interface, Line, Square, Cube, NCubeInterior, SymbolicMapping, DiscreteMapping, NCube
from sympde.topology.basic import Union
from sympde.topology.callable_mapping import BasicCallableMapping

from psydac.fem.splines        import SplineSpace
from psydac.fem.tensor         import TensorFemSpace
from psydac.fem.partitioning   import create_cart, construct_connectivity, construct_interface_spaces
from psydac.mapping.discrete   import SplineCallableMapping, NurbsCallableMapping
from psydac.linalg.block       import BlockVectorSpace, BlockVector
from psydac.ddm.cart           import DomainDecomposition, MultiPatchDomainDecomposition

__all__ = (
    'Geometry',
    'is_spline_discrete_domain',
    'export_nurbs_to_hdf5',
    'import_geopdes_to_nurbs',
    'refine_knots',
    'refine_nurbs',
)

NoneType = type(None)

#==============================================================================
# Helpers shared by `from_discrete_mapping` (single patch) and
# `from_discrete_domain` (single / multi patch) -- see those methods.
#==============================================================================
def _spline_parametric_box(spline_space):
    """
    Per-direction parametric interval of a spline `TensorFemSpace`, e.g.
    ``([0.0, 0.0], [1.0, pi/2])``. Returns ``(None, None)`` for a callable
    without a `.space` (a non-spline `BasicCallableMapping`), signalling
    "cannot check -- trust the caller".
    """
    if spline_space is None:
        return None, None
    par_min = [float(sp.domain[0]) for sp in spline_space.spaces]
    par_max = [float(sp.domain[1]) for sp in spline_space.spaces]
    return par_min, par_max


def _check_logical_box(min_coords, max_coords, spline_space, *, what):
    """
    Raise ``ValueError`` if the logical (parametric) box ``(min_coords,
    max_coords)`` does not span exactly the spline's own knot span
    (``spline_space.spaces[d].domain``). psydac draws quadrature points from the
    logical box and evaluates the spline there, so a mismatch silently corrupts
    the Jacobian scaling and the boundary identification. A no-op when
    ``spline_space`` is ``None``. ``what`` names the offending object in the
    message.
    """
    par_min, par_max = _spline_parametric_box(spline_space)
    if par_min is None:
        return
    if not (len(min_coords) == len(max_coords) == len(par_min)):
        raise ValueError("{} dimension ({}) does not match the spline ldim ({})"
                         .format(what, len(min_coords), len(par_min)))
    if not (np.allclose(min_coords, par_min, rtol=1e-9, atol=1e-12) and
            np.allclose(max_coords, par_max, rtol=1e-9, atol=1e-12)):
        raise ValueError(
            "{} extent {} does not match the spline's parametric box {}".format(
                what,
                list(zip([float(c) for c in min_coords], [float(c) for c in max_coords])),
                list(zip(par_min, par_max))))


#==============================================================================
# Helpers for `from_discrete_domain` (multipatch rebuild) and its dispatch
# predicate `is_spline_discrete_domain` -- not used by `from_discrete_mapping`
# (single patch has no "is this a spline domain" question, and no interface
# to wire).
#==============================================================================
def _spline_of(M):
    """
    The SplineCallableMapping (or NurbsCallableMapping) `M` wraps, if `M` is a spline-backed
    `DiscreteMapping` with an attached callable.

    Parameters
    ----------
    M : object
        Typically an interior domain's `.mapping` -- anything is accepted;
        non-`DiscreteMapping` values simply return `None`.

    Returns
    -------
    SplineCallableMapping or None
        `None` if `M` is not a `DiscreteMapping`, has no attached callable, or
        wraps a non-spline `BasicCallableMapping`.

    Examples
    --------
    >>> _spline_of(itr.mapping) is None  # itr not mapped by a spline
    True
    """
    if not isinstance(M, DiscreteMapping):
        return None
    if not M.has_callable_mapping():
        return None
    spl = M.get_callable_mapping()
    return spl if isinstance(spl, SplineCallableMapping) else None


def _patch_spline(itr):
    """
    The SplineCallableMapping carried by interior domain `itr`'s `DiscreteMapping` --
    see `_spline_of`. Used by `is_spline_discrete_domain`; `Geometry.
    from_discrete_domain` calls `_spline_of` directly (its own `itr.mapping`
    lookup and this function's would otherwise be two independent lookups that
    could in principle diverge) so its error message reuses the exact mapping
    object classified.

    Parameters
    ----------
    itr : sympde.topology.InteriorDomain
        One patch of a (possibly multipatch) `Domain`.

    Returns
    -------
    SplineCallableMapping or None
        See `_spline_of`.

    Examples
    --------
    >>> _patch_spline(Omega.interior) is not None
    True
    """
    return _spline_of(getattr(itr, 'mapping', None))


def is_spline_discrete_domain(domain):
    """
    True if every patch of ``domain`` is mapped by a `DiscreteMapping` whose
    callable is a psydac `SplineCallableMapping` (or `NurbsCallableMapping`) -- i.e. the domain
    carries its own discrete geometry, so ``discretize(domain)`` (given no
    ``filename`` / ``ncells``) *dispatches* to `Geometry.from_discrete_domain`.
    False for anything else (analytic mapping, no mapping, a bare topological
    `NCube`), so callers' existing behaviour is unchanged.

    A `True` result is not itself a guarantee that `from_discrete_domain` will
    succeed: it can still raise `ValueError` if some patch's logical (symbolic)
    box disagrees with its spline's own parametric knot span -- this predicate
    only checks that every patch is spline-mapped, not the box.
    """
    interior  = domain.interior
    interiors = list(interior.args) if isinstance(interior, Union) else [interior]
    return all(_patch_spline(itr) is not None for itr in interiors)


def _interior_index(coeff_space):
    return tuple(slice(s, e + 1) for s, e in zip(coeff_space.starts, coeff_space.ends))


def _spline_control_points(spline, pdim):
    """``(*nbasis, pdim)`` array of the interior control points of ``spline``."""
    idx = _interior_index(spline.space.coeff_space)
    return np.stack([np.asarray(spline.fields[d].coeffs[idx]) for d in range(pdim)],
                    axis=-1)


def _sync_multipatch_ghost_regions(mappings, connectivity):
    """
    Update the ghost regions of the spline control-point coefficients (and NURBS
    weights) across patch interfaces, so each field's `StencilVector` carries
    the cross-interface `_interface_data` that assembly reads. Mirrors the tail
    of :meth:`Geometry.read`.
    """
    coeffs         = [[f.coeffs for f in m.fields] for m in mappings]
    patch_spaces   = [BlockVectorSpace(*[c.space for c in ci]) for ci in coeffs]
    patch_spaces_w = [ci[0].space for ci in coeffs]
    v = BlockVector(BlockVectorSpace(*patch_spaces,   connectivity=connectivity))
    w = BlockVector(BlockVectorSpace(*patch_spaces_w, connectivity=connectivity))
    for i, m in enumerate(mappings):
        for j in range(len(coeffs[i])):
            v[i][j] = coeffs[i][j]
        w[i] = m.weights_field.coeffs if isinstance(m, NurbsCallableMapping) \
               else v[i][0].space.zeros()
    v.update_ghost_regions()
    w.update_ghost_regions()


#==============================================================================
class _PatchKeyedDict(dict):
    """
    Per-patch dict canonically keyed by interior name, that also resolves a
    fixed set of legacy keys with a ``DeprecationWarning``.

    ``Geometry``'s ``mappings``/``ncells``/``periodic`` dicts used to be keyed
    inconsistently across constructors (WP15). This class lets every
    constructor settle on one canonical key -- the interior name -- while
    still accepting the legacy keys some files/callers relied on, without
    doubling ``len()``/``.values()`` (which would break every positional
    reader that iterates these dicts).

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``dict.__init__``; should already use canonical keys.

    aliases : dict[str | int, str], optional
        Maps a legacy key to its canonical replacement. A legacy key is only
        resolved when it is not itself already a canonical (real) key --
        canonical keys always win and never warn.

    Notes
    -----
    Only ``__getitem__``, ``get`` and ``__contains__`` resolve aliases.
    Iteration, ``len``, ``keys``, ``values``, ``items`` and ``==`` see
    canonical keys only. ``pop``, ``update`` and ``setdefault`` do **not**
    resolve aliases -- they behave like on a plain ``dict``.

    Examples
    --------
    >>> d = _PatchKeyedDict({'Omega': 1}, aliases={'patch_0': 'Omega'})
    >>> d['patch_0']  # doctest: +SKIP
    1
    """
    def __init__(self, *args, aliases=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.aliases = dict(aliases) if aliases else {}

    def _alias_of(self, key):
        """Canonical key `key` aliases to, or None if not a legacy alias."""
        if super().__contains__(key):
            return None
        return self.aliases.get(key)

    def __getitem__(self, key):
        canonical = self._alias_of(key)
        if canonical is None:
            return super().__getitem__(key)
        warnings.warn(
            f"Geometry legacy key {key!r} is deprecated; use {canonical!r} instead.",
            DeprecationWarning, stacklevel=2)
        return super().__getitem__(canonical)

    def get(self, key, default=None):
        canonical = self._alias_of(key)
        if canonical is None:
            return super().get(key, default)
        warnings.warn(
            f"Geometry legacy key {key!r} is deprecated; use {canonical!r} instead.",
            DeprecationWarning, stacklevel=2)
        return super().get(canonical, default)

    def __contains__(self, key):
        canonical = self._alias_of(key)
        if canonical is None:
            return super().__contains__(key)
        warnings.warn(
            f"Geometry legacy key {key!r} is deprecated; use {canonical!r} instead.",
            DeprecationWarning, stacklevel=2)
        # An alias may name a patch this dict has no entry for -- `read()`
        # builds `mappings`' aliases from every on-disk patch name, but only
        # spline/NURBS patches get a mapping. Report what `__getitem__` would
        # actually do, so `k in d` never disagrees with `d[k]`.
        return super().__contains__(canonical)

    def __reduce__(self):
        return (self.__class__, (dict(self),), {'aliases': dict(self.aliases)})

    def copy(self):
        return self.__class__(self, aliases=self.aliases)


#==============================================================================
class Geometry:
    """
    Distributed discrete geometry that works for single and multiple patches.

    The Geometry object can be created in five ways:
    - case 0 : providing a `Domain` to `__init__` with detailed parameters for each patch.
    - case 1 : passing the path to a geometry file to `from_file`; each patch's
      mapping is a spline-backed `DiscreteMapping`, as in case 4.
    - case 2 : passing a `SplineCallableMapping` to `from_discrete_mapping` (single patch).
    - case 3 : passing a `Domain`, ncells, and periodicity to `from_topological_domain` (single or multi-patch).
    - case 4 : passing a `Domain` whose patches are all mapped by a spline `DiscreteMapping` to `from_discrete_domain` (single or multi-patch, serial); this is what `discretize(domain)` uses when given no `filename` / `ncells`.

    Keys
    ----
    `mappings`, `ncells` and `periodic` are always keyed by `domain.
    interior_names`, in interior order, regardless of constructor. A file
    read by `from_file` may have been written with different on-disk patch
    names (`_patch_names` records the mapping); those names still work as
    legacy keys on `mappings` (and, for `periodic`, legacy integer keys), but
    emit a `DeprecationWarning`. `export()` writes the on-disk names back
    unchanged, so the file format is unaffected by this.

    Parameters
    ----------
    domain : Sympde.topology.Domain
        The symbolic topological domain to be discretized.

    pdim : int
        Number of physical dimensions of the Geometry object (pdim >= ldim).

    ncells : dict[str, Iterable[int]]
        The number of cells of the discretized domain in each direction.

    periodic : dict[str, Iterable[bool]], optional
        The periodicity of the topological domain in each direction.

    mappings : dict[str, BasicCallableMapping], optional
        The discrete mappings of each patch.

    comm: MPI.Intracomm, optional
        MPI intra-communicator.

    mpi_dims_mask: Iterable[bool], optional
        True if the dimension is to be used in the domain decomposition (=default for each dimension). 
        If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.
  
    """
    _ldim        = None
    _pdim        = None
    _patches     = []
    _topology    = None
    _patch_names = None

    def __init__(self,
                 domain : Domain,
                 *,
                 pdim     : int,
                 ncells   : dict[str, Iterable[int]],
                 mappings : dict[str, SplineCallableMapping | None] = None,
                 periodic : dict[str, Iterable[bool]] = None,
                 comm : MPI.Intracomm = None,
                 mpi_dims_mask : Iterable[bool] = None):

        # Type checks
        assert isinstance(pdim, int)
        assert isinstance(domain, Domain) 
        assert isinstance(ncells, dict)
        assert isinstance(mappings, dict)
        assert isinstance(periodic, (NoneType, dict))
        assert isinstance(comm, (NoneType, MPI.Intracomm))
        assert isinstance(mpi_dims_mask, (NoneType, Iterable))

        # Extract info from domain
        ldim : int = domain.dim
        interior_names : list = domain.interior_names
        set_interior_names = set(interior_names)

        # Check sanity of pdim
        assert pdim >= ldim

        # Check sanity of ncells
        assert set(ncells.keys()) == set_interior_names
        assert all(len(n) == ldim for n in ncells.values())
        assert all(isinstance(ni, (int, np.integer)) for ni in chain(*ncells.values()))
        assert all(ni > 0 for ni in chain(*ncells.values()))

        # Although we allow the iterable values in ncells to contain NumPy
        # integers, we convert them to lists of Python integers for consistency
        ncells = {patch: [int(ni) for ni in n] for patch, n in ncells.items()}

        # Check sanity of periodic
        if periodic is None:
            periodic = {patch: [False] * len(n) for patch, n in ncells.items()}
        else:
            assert set(periodic.keys()) == set_interior_names
            assert all(len(p) == ldim for p in periodic.values())
            assert all(isinstance(pi, bool) for pi in chain(*periodic.values()))

        # Check sanity of mappings
        if mappings is None:
            mappings = {itr.name : None for itr in domain.interior}
        else:
            assert set(mappings.keys()) == set_interior_names
            assert all(isinstance(m, (BasicCallableMapping, NoneType)) for m in mappings.values())
            assert all(m.pdim == pdim for m in mappings.values() if m is not None)

        # Canonical key order (interior_names), wrapped for legacy-key
        # resolution (see `_PatchKeyedDict`). No aliases here: this Geometry
        # was built directly from these dicts, so there is no separate
        # on-disk name to alias -- `_patch_names` is the identity map.
        ncells   = _PatchKeyedDict({k: ncells  [k] for k in interior_names})
        periodic = _PatchKeyedDict({k: periodic[k] for k in interior_names})
        mappings = _PatchKeyedDict({k: mappings[k] for k in interior_names})
        self._patch_names = {k: k for k in interior_names}

        # Check sanity of mpi_dims_mask
        if mpi_dims_mask is not None:
            assert len(mpi_dims_mask) == ldim
            assert all(isinstance(mask, bool) for mask in mpi_dims_mask)

        # Create a (multi-patch) domain decomposition
        if len(domain) == 1:
            #name = domain.name
            name = interior_names[0]
            ddm = DomainDecomposition(
                ncells  = ncells[name],
                periods = periodic[name],
                comm    = comm,
                mpi_dims_mask = mpi_dims_mask,
            )
        else:
            ddm = MultiPatchDomainDecomposition(
                ncells  = [  ncells[itr] for itr in interior_names],
                periods = [periodic[itr] for itr in interior_names],
                comm    = comm,
            )

        # Add attributes to the new object
        self._domain   = domain
        self._ldim     = domain.dim
        self._pdim     = pdim
        self._ncells   = ncells
        self._mappings = mappings
        self._periodic = periodic
        self._comm     = comm
        self._ddm      = ddm
        self._cart     = None

    #--------------------------------------------------------------------------
    # Option [1]: from a file
    #--------------------------------------------------------------------------
    @classmethod
    def from_file(cls,
            filename : str,
            *,
            comm : MPI.Intracomm = None,
            mpi_dims_mask : Iterable[bool] = None):

        """
        Create a Geometry instance from an HDF5 input file in Psydac format.

        Each patch's `.domain.mapping` is a spline-backed `DiscreteMapping`
        wrapping the `SplineCallableMapping`/`NurbsCallableMapping` loaded from the file --
        the same carrier `from_discrete_domain` builds for an in-memory
        spline domain.

        Parameters
        ----------
        filename: str
            The path to the geometry file.

        comm: MPI.Intracomm, optional
            The MPI intra-communicator.

        mpi_dims_mask: Iterable[bool], optional
            True if the dimension is to be used in the domain decomposition
            (=default for each dimension). If mpi_dims_mask[i]=False, the i-th
            dimension will not be decomposed.
    
        Returns
        -------
        Geometry
            The new instance.
        """
        geo = super().__new__(cls)
        geo.read(filename, comm=comm, mpi_dims_mask=mpi_dims_mask)
        return geo

    #--------------------------------------------------------------------------
    # Option [2]: from a discrete mapping
    #--------------------------------------------------------------------------
    @classmethod
    def from_discrete_mapping(cls, mapping, *, comm=None, mpi_dims_mask=None,
                             name=None, domain_log=None):
        """
        Create a single-patch Geometry instance from one discrete mapping.

        Parameters
        ----------
        mapping : BasicCallableMapping
            The mapping from the logical domain to the physical domain.

        comm : MPI.Comm
            MPI intra-communicator.

        mpi_dims_mask: list of bool
            True if the dimension is to be used in the domain decomposition (=default for each dimension).
            If mpi_dims_mask[i]=False, the i-th dimension will not be decomposed.

        name : str
            Optional name for the symbolic mapping that will be created.
            Needed to avoid conflicts in case several mappings are created.

        domain_log : sympde.topology.NCube, optional
            The logical (parametric) domain the mapping acts on. Defaults to an
            ``NCube`` spanning the spline's own parametric box (identical to
            ``[0, 1]^ldim`` for a ``[0, 1]``-parametrised spline). If given, it
            must be an ``NCube`` (``Line`` / ``Square`` / ``Cube``) of the same
            dimension *and the same per-axis extent* as the spline: psydac draws
            quadrature points from this box and evaluates the spline there, so a
            mismatch would silently corrupt the Jacobian and boundary matching.

        Returns
        -------
        Geometry
            The new instance.

        Raises
        ------
        TypeError
            If ``domain_log`` is not an ``NCube``.
        ValueError
            If ``domain_log``'s dimension or per-axis extent disagrees with the
            spline's parametric box.
        """

        mapping_name = name if name else 'mapping'
        dim      = mapping.ldim

        # The symbolic logical domain must span exactly the spline's own knot
        # span (see `domain_log` above). A non-spline callable has no `.space`,
        # in which case we cannot check and trust the caller (such an input
        # already fails at `mapping.space.domain_decomposition` below).
        spline_space = getattr(mapping, 'space', None)

        if domain_log is None:
            par_min, par_max = _spline_parametric_box(spline_space)
            if par_min is None:
                par_min, par_max = [0.] * dim, [1.] * dim
            domain_log = NCube(name = 'Omega',
                               dim  = dim,
                               min_coords = par_min,
                               max_coords = par_max)
        else:
            if not isinstance(domain_log, NCube):
                raise TypeError("domain_log must be an NCube (Line/Square/Cube);"
                                " got {}".format(type(domain_log).__name__))
            if domain_log.dim != dim:
                raise ValueError("domain_log.dim ({}) does not match mapping.ldim"
                                 " ({})".format(domain_log.dim, dim))
            _check_logical_box(domain_log.min_coords, domain_log.max_coords,
                               spline_space, what='domain_log')

        # A DiscreteMapping: a symbolic carrier whose get_callable_mapping() is
        # `mapping` and whose is_analytical is False, so a domain built from it
        # assembles via grid evaluation of the spline (like Domain.from_file).
        M        = DiscreteMapping(mapping, mapping_name)
        domain   = M(domain_log)
        pdim     = mapping.pdim
        mappings = {domain.name: mapping}
        ncells   = {domain.name: mapping.space.domain_decomposition.ncells}
        periodic = {domain.name: mapping.space.domain_decomposition.periods}

        return Geometry(domain   = domain,
                        pdim     = pdim,
                        ncells   = ncells,
                        periodic = periodic,
                        mappings = mappings,
                        comm     = comm,
                        mpi_dims_mask = mpi_dims_mask)

    #--------------------------------------------------------------------------
    # Option [4]: from a Domain carrying spline DiscreteMappings
    #--------------------------------------------------------------------------
    # `comm` here is whatever `discretize_domain` passed in (already `Dup()`'d
    # there, `Free()`'d on failure) -- `Geometry`'s classmethods do not
    # Dup/Free their own `comm` on any path (`from_file` / `from_topological_
    # domain` included), so a caller invoking this directly with a shared
    # `comm` owns its lifecycle. Making `Geometry` own comm duplication
    # uniformly is a separate design change, not taken up here.
    @classmethod
    def from_discrete_domain(cls, domain, *, comm=None, mpi_dims_mask=None):
        """
        Create a Geometry from a symbolic ``Domain`` whose every patch is mapped
        by a spline ``DiscreteMapping`` (single or multi patch, serial).

        This is the in-memory equivalent of :meth:`from_file`: it takes the
        ``SplineCallableMapping`` carried by each patch's ``DiscreteMapping``
        (``patch.mapping.get_callable_mapping()``) and, for a multipatch domain,
        builds the coefficient-space interface connectivity that assembling an
        interface term (``integral(domain.interfaces, ...)``) requires -- the
        same ``construct_interface_spaces`` + ghost-region sync that
        :meth:`read` runs from HDF5 metadata, which
        :func:`~psydac.api.discretization.discretize_space` does *not* run for a
        discrete geometry. ``discretize(domain)`` dispatches here when neither
        ``filename`` nor ``ncells`` is given and :func:`is_spline_discrete_domain`
        holds.

        For a **multipatch** domain the per-patch ``SplineCallableMapping`` objects are
        *rebuilt* on fresh interface-aware spaces and stored in
        ``geo.mappings``. The originals on the domain are left untouched: unlike
        :meth:`read` (which re-points each ``patch.mapping``'s callable via
        ``set_callable_mapping``), a ``DiscreteMapping``'s callable *cannot* be
        reattached after construction -- ``DiscreteMapping.set_callable_mapping``
        raises ``TypeError`` (it is part of ``_hashable_content``, i.e. identity).
        So

            geo.mappings[name]                                   # rebuilt spline
            domain.interior[i].mapping.get_callable_mapping()    # original spline

        are, and stay, two different-but-equivalent objects: same control
        points, knots and degree, so *point evaluation* -- all any
        post-processing consumer (e.g. ``PostProcessManager``) does per patch --
        is identical. The rebuilt spline differs only in carrying the
        coefficient-space interface connectivity, which matters solely for
        *assembly*, and assembly reads ``geo.mappings``.

        Parameters
        ----------
        domain : sympde.topology.Domain
            Each interior's ``.mapping`` must be a ``DiscreteMapping`` whose
            ``get_callable_mapping()`` is a psydac ``SplineCallableMapping`` /
            ``NurbsCallableMapping``. Each patch's logical box must match its spline's
            parametric knot span.

        comm : MPI.Intracomm, optional
            Serial only. A communicator of size > 1 raises
            ``NotImplementedError`` -- use :meth:`from_file` for parallel
            multipatch.

        mpi_dims_mask : Iterable[bool], optional
            Passed through to the (single-patch) domain decomposition. Note that
            for a single-patch domain ``geo.mappings[name].space`` keeps the
            *incoming* spline's own decomposition, which may differ from
            ``geo.ddm`` if a non-default ``mpi_dims_mask`` (or a different comm)
            is given -- harmless for the supported serial / size-1 case, to be
            revisited with parallel support (see the ``TODO(parallel)`` below).

        Returns
        -------
        Geometry
            The new instance.

        Raises
        ------
        TypeError
            If some patch is not mapped by a spline ``DiscreteMapping``.
        ValueError
            If a patch's logical box disagrees with its spline's parametric box,
            or the patches' splines have inconsistent ``pdim``.
        NotImplementedError
            If ``comm`` has size > 1.

        Examples
        --------
        >>> MA = spl_A.to_defined_mapping('MA')          # a DiscreteMapping
        >>> MB = spl_B.to_defined_mapping('MB')
        >>> Omega = Domain.join([MA(A), MB(B)], connectivity, 'annulus')
        >>> geo = Geometry.from_discrete_domain(Omega)   # == discretize(Omega)
        """
        if comm is not None and getattr(comm, 'size', 1) > 1:
            raise NotImplementedError(
                "Geometry.from_discrete_domain: parallel multipatch is not "
                "supported; use Geometry.from_file")

        interior  = domain.interior
        interiors = list(interior.args) if isinstance(interior, Union) else [interior]

        # Pull the SplineCallableMapping carried by each patch; check the logical box.
        splines = []
        for itr in interiors:
            M   = getattr(itr, 'mapping', None)
            spl = _spline_of(M)
            if spl is None:
                raise TypeError(
                    "Geometry.from_discrete_domain: patch '{}' is not mapped by "
                    "a spline DiscreteMapping (got {})".format(itr.name, type(M).__name__))
            _check_logical_box(itr.min_coords, itr.max_coords, spl.space,
                               what="patch '{}' logical domain".format(itr.name))
            splines.append(spl)

        pdim = splines[0].pdim
        if not all(s.pdim == pdim for s in splines):
            raise ValueError(
                "Geometry.from_discrete_domain: patches have inconsistent pdim "
                "({})".format({itr.name: s.pdim for itr, s in zip(interiors, splines)}))

        ncells   = {itr.name: list(s.space.domain_decomposition.ncells)
                    for itr, s in zip(interiors, splines)}
        periodic = {itr.name: list(s.space.domain_decomposition.periods)
                    for itr, s in zip(interiors, splines)}

        geo = Geometry(domain   = domain,
                       pdim     = pdim,
                       ncells   = ncells,
                       periodic = periodic,
                       mappings = {itr.name: s for itr, s in zip(interiors, splines)},
                       comm     = comm,
                       mpi_dims_mask = mpi_dims_mask)

        connectivity = construct_connectivity(domain)
        if not connectivity:
            # Single patch: no interface wiring needed. The spline keeps its own
            # decomposition here; `geo.ddm` (built by __init__ from comm /
            # mpi_dims_mask) is only consistent with it in the serial / size-1 /
            # mask=None case this method supports.
            # TODO(parallel): rebuild the spline on `geo.ddm` (as `read` does)
            # once comm.size > 1 is supported, or delegate to from_discrete_mapping.
            return geo

        # Multipatch: rebuild the SplineCallableMappings on fresh TensorFemSpaces that
        # carry the interface coefficient spaces (mirrors `read`). The 1D
        # SplineSpaces are DomainDecomposition-independent, so we reuse them.
        ddms      = geo.ddm.domains
        spaces_1d = [list(s.space.spaces) for s in splines]
        carts     = create_cart(ddms, spaces_1d)
        g_spaces  = {itr: TensorFemSpace(ddms[i], *spaces_1d[i], cart=carts[i])
                     for i, itr in enumerate(interiors)}

        for i, j in connectivity:
            max_ncells = [max(ni, nj) for ni, nj in
                          zip(ncells[interiors[i].name], ncells[interiors[j].name])]
            g_spaces[interiors[i]].add_refined_space(ncells=max_ncells)
            g_spaces[interiors[j]].add_refined_space(ncells=max_ncells)

        construct_interface_spaces(geo.ddm, g_spaces, carts, interiors, connectivity)

        new_mappings = {}
        for itr, spl in zip(interiors, splines):
            cp = _spline_control_points(spl, pdim)
            if isinstance(spl, NurbsCallableMapping):
                idx = _interior_index(spl.space.coeff_space)
                w   = np.asarray(spl.weights_field.coeffs[idx])
                m   = NurbsCallableMapping.from_control_points_weights(g_spaces[itr], cp, w)
            else:
                m   = SplineCallableMapping.from_control_points(g_spaces[itr], cp)
            m.set_name(itr.name)
            new_mappings[itr.name] = m

        _sync_multipatch_ghost_regions(list(new_mappings.values()), connectivity)

        geo._mappings = _PatchKeyedDict(new_mappings)
        return geo

    #--------------------------------------------------------------------------
    # Option [3]: discrete topological line/square/cube
    #--------------------------------------------------------------------------
    @classmethod
    def from_topological_domain(cls, domain, ncells, *, periodic=None, comm=None, mpi_dims_mask=None):
        assert isinstance(domain, Domain)

        interior = domain.interior
        if not isinstance(interior, Union):
            interior = [interior]

        for itr in interior:
            if not isinstance(itr, NCubeInterior):
                msg = "The topological domain of each patch must be an NCube;"\
                      " got {} instead.".format(type(itr))
                raise TypeError(msg)

        mappings = {itr.name : None for itr in interior}
        pdim = next(iter(interior)).dim

        if isinstance(ncells, (list, tuple)):
            ncells = {itr.name : ncells for itr in interior}

        if periodic is None:
            periodic = [False] * domain.dim
        else:
            if len(interior) > 1 and True in periodic:
                import warnings
                msg = "Discretizing a multipatch domain with a periodic flag is not advised -- continue at your own risk."
                # [MCP 18.12.2025] the following line may be causing a strange error in the CI (MPI tests for macos-14/Python 3.10)
                # warnings.warn(msg, Warning)  
                warnings.warn(msg, UserWarning)

        if isinstance(periodic, (list, tuple)):
            periodic = {itr.name : periodic for itr in interior}

        return Geometry(domain   = domain,
                        pdim     = pdim,
                        ncells   = ncells,
                        periodic = periodic,
                        mappings = mappings,
                        comm     = comm,
                        mpi_dims_mask = mpi_dims_mask)

    #--------------------------------------------------------------------------
    @property
    def ldim(self):
        return self._ldim

    @property
    def pdim(self):
        return self._pdim

    @property
    def ncells(self):
        """dict[str, list[int]]: per-patch cell count, keyed by interior name."""
        return self._ncells

    @property
    def periodic(self):
        """dict[str, list[bool]]: per-patch periodicity, keyed by interior name."""
        return self._periodic

    @property
    def comm(self):
        return self._comm

    @property
    def domain(self):
        return self._domain

    @property
    def ddm(self):
        return self._ddm

    @property
    def mappings(self):
        """dict[str, BasicCallableMapping | None]: per-patch mapping, keyed by interior name."""
        return self._mappings

    def __len__(self):
        return len(self.domain)

    def read(self, filename, comm=None, mpi_dims_mask=None):
        """
        Populate this instance in place from an HDF5 geometry file.

        This is `from_file`'s implementation, split out so `from_file` can
        `__new__` the instance first (`read` sets every attribute `__init__`
        would, without going through it). Builds each patch's `SplineCallableMapping`/
        `NurbsCallableMapping` from the stored control points, then wraps it in a
        spline-backed `DiscreteMapping` (WP10) -- the same carrier
        `from_discrete_domain` builds for an in-memory spline domain -- so
        `self.domain`'s per-patch `.mapping` is symbolic-and-point-evaluable
        rather than the bare `SymbolicMapping` legs `Domain.from_file` parses
        from the file's topology metadata.

        Parameters
        ----------
        filename : str
            Path to the HDF5 geometry file.

        comm : MPI.Intracomm, optional
            MPI intra-communicator.

        mpi_dims_mask : Iterable[bool], optional
            True if the dimension is to be used in the domain decomposition
            (=default for each dimension). If mpi_dims_mask[i]=False, the i-th
            dimension will not be decomposed.

        Raises
        ------
        ValueError
            If `filename` does not have a `.h5` extension, or the file
            contains no patches.
        """
        # ... check extension of the file
        _, ext = os.path.splitext(filename)
        if ext != '.h5':
            raise ValueError('> Only h5 files are supported')
        # ...

        # read the topological domain
        domain       = Domain.from_file(filename)
        connectivity = construct_connectivity(domain)

        if len(domain) == 1:
            interiors = [domain.interior]
        else:
            interiors = list(domain.interior.args)

        if comm is not None:
            kwargs = dict(driver='mpio', comm=comm) if comm.size > 1 else {}
        else:
            kwargs = {}

        h5  = h5py.File(filename, mode='r', **kwargs)
        yml = yaml.load(h5['geometry.yml'][()], Loader=yaml.SafeLoader)

        ldim = yml['ldim']
        pdim = yml['pdim']

        n_patches = len(yml['patches'])

        # ...
        if n_patches == 0:
            h5.close()
            raise ValueError("Input file contains no patches.")
        # ...

        # Pair each on-disk patch with its interior **by name**, not by
        # position. sympde's `Union` sorts interiors lexicographically
        # (`sorted(set(args), key=str)`, sympde/topology/basic.py), so from
        # 11 patches on the yml order (patch_0, patch_1, patch_2, ...) and
        # the interior order (..., patch_1, patch_10, patch_2, ...) diverge,
        # and a positional pairing binds every per-patch entry to the wrong
        # patch. A yml patch name is either the interior name (files exported
        # from an in-memory Geometry) or the logical-domain name (every
        # committed fixture, and what export_nurbs_to_hdf5 / cad/multipatch.py
        # write), so both are accepted.
        yml_names      = [p['name'] for p in yml['patches']]
        interior_names = [itr.name for itr in interiors]

        by_name = {}
        for j, itr in enumerate(interiors):
            logical = getattr(itr, 'logical_domain', None)
            for nm in {itr.name} | ({logical.name} if logical is not None else set()):
                by_name.setdefault(nm, set()).add(j)

        candidates = [by_name.get(nm, set()) for nm in yml_names]
        if (all(len(c) == 1 for c in candidates)
                and len({next(iter(c)) for c in candidates}) == n_patches):
            patch_to_interior = [next(iter(c)) for c in candidates]
        else:
            # A name that matches no interior, or matches more than one (two
            # patches sharing a logical name), or two patches resolving to
            # the same interior: fall back to the historical positional
            # pairing rather than guess. Correct whenever the two orders
            # agree, which is every readable file in tree.
            patch_to_interior = list(range(n_patches))

        # The interior each on-disk patch belongs to, and the on-disk name of
        # each interior -- `export()` writes these back verbatim, which is
        # what keeps a from_file round trip byte identical.
        patch_interiors = [interiors[j] for j in patch_to_interior]
        patch_names     = {interiors[j].name: yml_names[i]
                           for i, j in enumerate(patch_to_interior)}

        # Legacy `mappings` keys: the on-disk (yml) patch name, when it
        # differs from the canonical interior name, is not itself already a
        # canonical key, and is unique across this file's patches -- so it
        # resolves unambiguously to exactly one interior.
        name_counts = {n: yml_names.count(n) for n in set(yml_names)}
        mapping_aliases = {
            yml_name: itr.name
            for yml_name, itr in zip(yml_names, patch_interiors)
            if yml_name != itr.name
            and yml_name not in interior_names
            and name_counts[yml_name] == 1
        }

        # ... read patches
        mappings = {}
        ncells   = {}
        periodic = {}
        spaces   = [None] * n_patches
        for i_patch in range(n_patches):

            item  = yml['patches'][i_patch]
            patch_name = item['name']
            mapping_id = item['mapping_id']
            dtype = item['type']
            patch = h5[mapping_id]
            if dtype in [SplineCallableMapping.geometry_dtype, NurbsCallableMapping.geometry_dtype]:

                degree     = [int (p) for p in patch.attrs['degree'  ]]
                periodic_i = [bool(b) for b in patch.attrs['periodic']]
                knots      = [patch['knots_{}'.format(d)][:] for d in range(ldim)]
                space_i    = [SplineSpace(degree=p, knots=k, periodic=P)
                              for p, k, P in zip(degree, knots, periodic_i)]

                # indexed/keyed by the patch's *interior*, so that spaces[j],
                # g_spaces[interiors[j]] and ncells[interiors[j].name] below
                # all refer to the same patch (see patch_to_interior above).
                spaces[patch_to_interior[i_patch]] = space_i

                ncells  [patch_interiors[i_patch].name] = [sp.ncells for sp in space_i]
                periodic[patch_interiors[i_patch].name] = periodic_i

        if n_patches == 1:
            ddm  = DomainDecomposition(ncells[domain.name], periodic[domain.name], comm=comm, mpi_dims_mask=mpi_dims_mask)
            ddms = [ddm]
        else:
            ncells_       = [ncells[itr.name] for itr in interiors]
            periodic_list = [periodic[itr.name] for itr in interiors]
            ddm           = MultiPatchDomainDecomposition(ncells_, periodic_list, comm=comm)
            ddms          = ddm.domains

        carts    = create_cart(ddms, spaces)
        g_spaces = {inter:TensorFemSpace(ddms[i], *spaces[i], cart=carts[i]) for i,inter in enumerate(interiors)}

        for i, j in connectivity:
            minus = interiors[i]
            plus  = interiors[j]
            max_ncells = [max(ni, nj) for ni, nj in zip(ncells[minus.name], ncells[plus.name])]
            g_spaces[minus].add_refined_space(ncells=max_ncells)
            g_spaces[plus ].add_refined_space(ncells=max_ncells)

        # ... construct interface spaces
        construct_interface_spaces(ddm, g_spaces, carts, interiors, connectivity)

        for i_patch in range( n_patches ):

            item  = yml['patches'][i_patch]
            mapping_id = item['mapping_id']
            dtype = item['type']
            patch = h5[mapping_id]
            space_i = spaces[patch_to_interior[i_patch]]
            if dtype in [SplineCallableMapping.geometry_dtype, NurbsCallableMapping.geometry_dtype]:
                tensor_space = g_spaces[patch_interiors[i_patch]]

                if dtype == SplineCallableMapping.geometry_dtype:
                    mapping = SplineCallableMapping.from_control_points(tensor_space,
                                                                patch['points'][..., :pdim])

                elif dtype == NurbsCallableMapping.geometry_dtype:
                    mapping = NurbsCallableMapping.from_control_points_weights(tensor_space,
                                                                       patch['points'][..., :pdim],
                                                                       patch['weights'])

                mapping.set_name(item['name'])
                mappings[patch_interiors[i_patch].name] = mapping

        # Canonical order for all three per-patch dicts: interior order, not
        # the on-disk patch order (the two differ from 11 patches on). Both
        # the class docstring's "Keys" section and every caller that walks
        # these dicts positionally -- the domain rebuild below, the
        # ghost-region sync, `Geometry.__len__`, and anything zipping
        # `mappings.values()` against `ncells.values()` -- rely on it.
        mappings = {itr.name: mappings[itr.name]
                    for itr in interiors if itr.name in mappings}
        ncells   = {itr.name: ncells  [itr.name] for itr in interiors}
        periodic = {itr.name: periodic[itr.name] for itr in interiors}

        # ... Update ghost regions within each patch and across interfaces
        if n_patches > 1:
            coeffs         = [[e.coeffs for e in mapping.fields] for mapping in mappings.values()]
            patch_spaces   = [BlockVectorSpace(*[c_ij.space for c_ij in c_i]) for c_i in coeffs]
            patch_spaces_w = [c_i[0].space for c_i in coeffs]
            space          = BlockVectorSpace(*patch_spaces  , connectivity=connectivity)
            space_w        = BlockVectorSpace(*patch_spaces_w, connectivity=connectivity)
            v = BlockVector(space)
            w = BlockVector(space_w)
            mapping_list = list(mappings.values())
            for i in range(n_patches):
                for j in range(len(coeffs[i])):
                    v[i][j] = coeffs[i][j]

                mapping = mapping_list[i]
                if isinstance(mapping, NurbsCallableMapping):
                    w[i] = mapping.weights_field.coeffs
                else:
                    w[i] = v[i][0].space.zeros()

            v.update_ghost_regions()
            w.update_ghost_regions()

        else:
            mapping = list(mappings.values())[0]
            for f in mapping._fields:
                f.coeffs.update_ghost_regions()

            if isinstance(mapping, NurbsCallableMapping):
                mapping.weights_field.coeffs.update_ghost_regions()
        # ...

        # ... close the h5 file
        h5.close()
        # ...

        # Build the domain from spline-backed DiscreteMappings -- consistent
        # with Geometry.from_discrete_domain (WP07c-1) -- instead of mutating
        # the bare SymbolicMapping legs Domain.from_file built via
        # set_callable_mapping (WP10). NOTE: interiors and mappings.values()
        # are guaranteed to share ordering -- the three dicts are rebuilt in
        # interior order above (WP15-1).
        new_legs = []
        for itr, F in zip(interiors, mappings.values()):
            # Reuse itr's own mapping name and logical-domain name verbatim,
            # rather than F.name (the on-disk yml patch name -- kept as F's
            # name by design, see the class docstring's "Keys" section, so
            # export() still writes the same bytes). Since WP15, `mappings`
            # is keyed uniformly by interior name in every constructor, so
            # this is simply itr.name; using F.name here would double-wrap
            # it (verified: reading back a from_discrete_mapping export
            # produced "mapping(mapping(Omega))" instead of "mapping(Omega)").
            # This rebuild is therefore a pure type swap (SymbolicMapping ->
            # DiscreteMapping), never a rename.
            logical_i = NCube(name=itr.logical_domain.name, dim=ldim,
                              min_coords=itr.min_coords, max_coords=itr.max_coords)
            new_legs.append(F.to_defined_mapping(itr.mapping.name)(logical_i))

        if n_patches == 1:
            new_domain = new_legs[0]
        else:
            patch_interfaces = domain.interfaces
            if isinstance(patch_interfaces, Interface):
                patch_interfaces = [patch_interfaces]
            elif isinstance(patch_interfaces, Union):
                patch_interfaces = list(patch_interfaces.args)
            else:
                patch_interfaces = list(patch_interfaces) if patch_interfaces else []
            patch_index = {itr: i for i, itr in enumerate(interiors)}
            join_connectivity = [
                ((patch_index[e.minus.domain], e.minus.axis, e.minus.ext),
                 (patch_index[e.plus.domain],  e.plus.axis,  e.plus.ext),
                 e.ornt)
                for e in patch_interfaces]
            new_domain = Domain.join(new_legs, join_connectivity, domain.name)

        # ...
        # `periodic` is an int-aliased _PatchKeyedDict only for multipatch,
        # where `geo.periodic[i]` used to work (see class docstring's "Keys"
        # section); single-patch files never had an integer-key convention.
        periodic_aliases = {i: itr.name for i, itr in enumerate(interiors)} \
                            if n_patches > 1 else None

        self._domain      = new_domain
        self._ldim        = ldim
        self._pdim        = pdim
        self._ncells      = _PatchKeyedDict(ncells)
        self._mappings    = _PatchKeyedDict(mappings, aliases=mapping_aliases)
        self._periodic    = _PatchKeyedDict(periodic, aliases=periodic_aliases)
        self._comm        = comm
        self._ddm         = ddm
        self._cart        = None
        self._patch_names = patch_names
        # ...

    def _patches_in_file_order(self):
        """
        Per-patch ``(interior key, on-disk name, mapping)``, in the order the
        patches must be written to file.

        ``mappings`` is keyed and ordered by interior name, but a file read by
        :meth:`read` may list its patches in a different order -- the two
        diverge from 11 patches on, because sympde sorts interiors
        lexicographically while the file keeps its own order.
        ``_patch_names`` was built in on-disk order, so iterating it
        reproduces a read file's patch list exactly and keeps the round trip
        byte identical. For every other constructor it is the identity map
        over ``interior_names``, so this is just ``mappings`` order.

        Returns
        -------
        list[tuple[str, str, BasicCallableMapping]]
        """
        names = self._patch_names or {}
        keys  = [k for k in names if k in self.mappings]
        # A patch absent from _patch_names keeps its position in `mappings`.
        keys += [k for k in self.mappings if k not in names]
        return [(k, names.get(k, k), self.mappings[k]) for k in keys]

    def export( self, filename ):
        """
        Parameters
        ----------
        filename : str
          Name of HDF5 output file.

        Notes
        -----
        The written patch names come from `self._patch_names` (identity map
        unless this instance came from `read()`), not from the `mappings`
        keys directly -- this is what keeps a `from_file` round trip byte
        identical even though `mappings` is now keyed by interior name.
        """

        # ...
        comm  = self.comm
        # ...

        # Create dictionary with geometry metadata
        yml = {}
        yml['ldim'] = self.ldim
        yml['pdim'] = self.pdim

        # ... information about the patches
        if not( self.mappings ):
            raise ValueError('No mappings were found')

        patches_info = []
        i_mapping    = 0
        for _, name, mapping in self._patches_in_file_order():
            mapping_id = 'mapping_{}'.format( i_mapping  )
            dtype      = mapping.geometry_dtype

            patches_info += [{'name': name,
                              'mapping_id': mapping_id,
                               'type': dtype}]

            i_mapping += 1

        yml['patches'] = patches_info
        # ...

        # ... topology
        topo_yml = self.domain.todict()
        # ...

        # Create HDF5 file (in parallel mode if MPI communicator size > 1)
        if not(comm is None) and comm.size > 1:
            kwargs = dict( driver='mpio', comm=comm )

        else:
            kwargs = {}

        h5 = h5py.File( filename, mode='w', **kwargs )

        # ...
        # Dump geometry metadata to string in YAML file format
        geo = yaml.dump( data   = yml, sort_keys=False)

        # Write geometry metadata as fixed-length array of ASCII characters
        h5['geometry.yml'] = np.array( geo, dtype='S' )
        # ...

        # ...
        # Dump geometry metadata to string in YAML file format
        geo = yaml.dump( data   = topo_yml, sort_keys=False)
        # Write topology metadata as fixed-length array of ASCII characters
        h5['topology.yml'] = np.array( geo, dtype='S' )
        # ...

        i_mapping    = 0
        for _, _, mapping in self._patches_in_file_order():
            space = mapping.space

            # Create group for patch 0
            group = h5.create_group( yml['patches'][i_mapping]['mapping_id'] )
            group.attrs['shape'      ] = space.coeff_space.npts
            group.attrs['degree'     ] = space.degree
            group.attrs['rational'   ] = False # TODO remove
            group.attrs['periodic'   ] = space.periodic
            for d in range( self.ldim ):
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
            dset[index] = mapping.control_points[index]

            # case of NURBS
            if isinstance(mapping, NurbsCallableMapping):
                # Collective: create dataset for weights
                shape = [n for n in space.coeff_space.npts]
                dtype = space.coeff_space.dtype
                dset  = group.create_dataset( 'weights', shape=shape, dtype=dtype )

                # Independent: write weights to dataset
                starts = space.coeff_space.starts
                ends   = space.coeff_space.ends
                index  = [slice(s, e+1) for s, e in zip(starts, ends)]
                index  = tuple( index )
                dset[index] = mapping.weights[index]

            i_mapping += 1

        # Close HDF5 file
        h5.close()

#==============================================================================
def export_nurbs_to_hdf5(filename, nurbs, periodic=None, comm=None ):

    """
    Export a single-patch igakit NURBS object to a PSYDAC geometry file in HDF5 format

    Parameters
    ----------

    filename : <str>
        Name of output geometry file, e.g. 'geo.h5'

    nurbs   : <igakit.nurbs.NURBS>
        igakit geometry nurbs object

    comm : <MPI.COMM>
        mpi communicator
    """

    import os.path
    import igakit
    assert isinstance(nurbs, igakit.nurbs.NURBS)

    extension = os.path.splitext(filename)[-1]
    if not extension == '.h5':
        raise ValueError('> Only h5 extension is allowed for filename')

    yml = {}
    yml['ldim'] = nurbs.dim
    yml['pdim'] = nurbs.dim

    patches_info = []
    i_mapping    = 0
    i            = 0

    rational = not abs(nurbs.weights-1).sum()<1e-15

    patch_name = 'patch_{}'.format(i)
    name       = '{}'.format( patch_name )
    mapping_id = 'mapping_{}'.format( i_mapping  )
    dtype      = NurbsCallableMapping.geometry_dtype if rational else SplineCallableMapping.geometry_dtype

    patches_info += [{'name': name , 'mapping_id':mapping_id, 'type':dtype}]

    yml['patches'] = patches_info
    # ...

    # Create HDF5 file (in parallel mode if MPI communicator size > 1)
    if not(comm is None) and comm.size > 1:
        kwargs = dict( driver='mpio', comm=comm )
    else:
        kwargs = {}

    h5 = h5py.File( filename, mode='w', **kwargs )

    # ...
    # Dump geometry metadata to string in YAML file format
    geom = yaml.dump( data   = yml, sort_keys=False)
    # Write geometry metadata as fixed-length array of ASCII characters
    h5['geometry.yml'] = np.array( geom, dtype='S' )
    # ...

    # ... topology
    if nurbs.dim == 1:
        bounds1 = (float(nurbs.breaks(0)[0]), float(nurbs.breaks(0)[-1]))
        domain  = Line(patch_name, bounds1=bounds1)

    elif nurbs.dim == 2:
        bounds1 = (float(nurbs.breaks(0)[0]), float(nurbs.breaks(0)[-1]))
        bounds2 = (float(nurbs.breaks(1)[0]), float(nurbs.breaks(1)[-1]))
        domain  = Square(patch_name, bounds1=bounds1, bounds2=bounds2)

    elif nurbs.dim == 3:
        bounds1 = (float(nurbs.breaks(0)[0]), float(nurbs.breaks(0)[-1]))
        bounds2 = (float(nurbs.breaks(1)[0]), float(nurbs.breaks(1)[-1]))
        bounds3 = (float(nurbs.breaks(2)[0]), float(nurbs.breaks(2)[-1]))
        domain  = Cube(patch_name, bounds1=bounds1, bounds2=bounds2, bounds3=bounds3)

    else:
        raise NotImplementedError('> nurbs.dim > 3 not implemented')

    mapping = SymbolicMapping(mapping_id, dim=nurbs.dim)
    domain  = mapping(domain)
    topo_yml = domain.todict()

    # Dump geometry metadata to string in YAML file format
    geom = yaml.dump( data   = topo_yml, sort_keys=False)
    # Write topology metadata as fixed-length array of ASCII characters
    h5['topology.yml'] = np.array( geom, dtype='S' )

    group = h5.create_group( yml['patches'][i]['mapping_id'] )
    group.attrs['degree'     ] = nurbs.degree
    group.attrs['rational'   ] = rational
    group.attrs['periodic'   ] = tuple( False for d in range( nurbs.dim ) ) if periodic is None else periodic
    for d in range( nurbs.dim ):
        group['knots_{}'.format( d )] = nurbs.knots[d]

    group['points'] = nurbs.points[...,:nurbs.dim]
    if rational:
        group['weights'] = nurbs.weights

    h5.close()

#==============================================================================
def refine_nurbs(nrb, ncells=None, degree=None, multiplicity=None, tol=1e-9):
    """
    This function refines the nurbs object.
    It contructs a new grid based on the new number of cells, and it adds the new break points to the nrb grid,
    such that the total number of cells is equal to the new number of cells.
    We use knot insertion to construct the new knot sequence , so the geometry is identical to the previous one.
    It also elevates the degree of the nrb object based on the new degree.

    Parameters
    ----------

    nrb : <igakit.nurbs.NURBS>
        geometry nurbs object

    ncells   : <list>
        total number of cells in each direction

    degree : <list>
        degree in each direction

    multiplicity : <list>
        multiplicity of each knot in the knot sequence in each direction

    tol : <float>
        Minimum distance between two break points.

    Returns
    -------
    nrb : <igakit.nurbs.NURBS>
        the refined geometry nurbs object

    """

    if multiplicity is None:
        multiplicity = [1]*nrb.dim

    nrb = nrb.clone()
    if ncells is not None:

        for axis in range(0,nrb.dim):
            ub = nrb.breaks(axis)[0]
            ue = nrb.breaks(axis)[-1]
            knots = np.linspace(ub,ue,ncells[axis]+1)
            index = nrb.knots[axis].searchsorted(knots)
            nrb_knots = nrb.knots[axis][index]
            for m,(nrb_k, k) in enumerate(zip(nrb_knots, knots)):
                if abs(k-nrb_k)<tol:
                    knots[m] = np.nan

            knots   = knots[~np.isnan(knots)]
            indices = np.round(np.linspace(0, len(knots) - 1, ncells[axis]+1-len(nrb.breaks(axis)))).astype(int)

            knots = knots[indices]

            if len(knots)>0:
                nrb.refine(axis, knots)

    if degree is not None:
        for axis in range(0,nrb.dim):
            d = degree[axis] - nrb.degree[axis]
            if d<0:
                raise ValueError('The degree {} must be >= {}'.format(degree, nrb.degree))
            nrb.elevate(axis, times=d)

    for axis in range(nrb.dim):
        decimals = abs(np.floor(np.log10(np.abs(tol))).astype(int))
        knots, counts = np.unique(nrb.knots[axis].round(decimals=decimals), return_counts=True)
        counts = multiplicity[axis] - counts
        counts[counts<0] = 0
        knots = np.repeat(knots, counts)
        nrb = nrb.refine(axis, knots)
    return nrb

def refine_knots(knots, ncells, degree, multiplicity=None, tol=1e-9):
    """
    This function refines the knot sequence.
    It contructs a new grid based on the new number of cells, and it adds the new break points to the nrb grid,
    such that the total number of cells is equal to the new number of cells.
    We use knot insertion to construct the new knot sequence , so the geometry is identical to the previous one.
    It also elevates the degree of the nrb object based on the new degree.

    Parameters
    ----------

    knots : <list>
        list of knot sequences in each direction

    ncells   : <list>
        total number of cells in each direction

    degree : <list>
        degree in each direction

    multiplicity : <list>
        multiplicity of each knot in the knot sequence in each direction

    tol : <float>
        Minimum distance between two break points.

    Returns
    -------
    knots : <list>
        the refined knot sequences in each direction
    """
    from igakit.nurbs import NURBS
    dim = len(ncells)

    if multiplicity is None:
        multiplicity = [1]*dim

    assert len(knots) == dim

    nrb = NURBS(knots)
    for axis in range(dim):
        ub = nrb.breaks(axis)[0]
        ue = nrb.breaks(axis)[-1]
        knots = np.linspace(ub,ue,ncells[axis]+1)
        index = nrb.knots[axis].searchsorted(knots)
        nrb_knots = nrb.knots[axis][index]
        for m,(nrb_k, k) in enumerate(zip(nrb_knots, knots)):
            if abs(k-nrb_k)<tol:
                knots[m] = np.nan

        knots   = knots[~np.isnan(knots)]
        indices = np.round(np.linspace(0, len(knots) - 1, ncells[axis]+1-len(nrb.breaks(axis)))).astype(int)

        knots = knots[indices]

        if len(knots)>0:
            nrb.refine(axis, knots)

    for axis in range(dim):
        d = degree[axis] - nrb.degree[axis]
        if d<0:
            raise ValueError('The degree {} must be >= {}'.format(degree, nrb.degree))
        nrb.elevate(axis, times=d)

    for axis in range(dim):
        decimals = abs(np.floor(np.log10(np.abs(tol))).astype(int))
        knots, counts = np.unique(nrb.knots[axis].round(decimals=decimals), return_counts=True)
        counts = multiplicity[axis] - counts
        counts[counts<0] = 0
        knots = np.repeat(knots, counts)
        nrb = nrb.refine(axis, knots)
    return nrb.knots

#==============================================================================
def import_geopdes_to_nurbs(filename):
    """
    This function reads a geopdes geometry file and convert it to igakit nurbs object

    Parameters
    ----------

    filename : <str>
        the filename of the geometry file

    Returns
    -------
    nrb : <igakit.nurbs.NURBS>
        the geometry nurbs object

    """
    extension = os.path.splitext(filename)[-1]
    if not extension == '.txt':
        raise ValueError('> Expected .txt extension')

    f = open(filename)
    lines = f.readlines()
    f.close()

    lines = [line for line in lines if line[0].strip() != "#"]

    data     = _read_header(lines[0])
    n_dim    = data[0]
    r_dim    = data[1]
    n_patchs = data[2]

    n_lines_per_patch = 3*n_dim + 1

    list_begin_line = _get_begin_line(lines, n_patchs)

    nrb = _read_patch(lines, 1, n_lines_per_patch, list_begin_line)

    return nrb

def _read_header(line):
    chars = line.split(" ")
    data  = []
    for c in chars:
        try:
            data.append(int(c))
        except ValueError:
            msg = f"WARNING: Cannot convert str '{c}' to int. Moving to next word..."
            print(msg)
    return data

def _extract_patch_line(lines, i_patch):
    text = "PATCH " + str(i_patch)
    for i_line,line in enumerate(lines):
        r = line.find(text)
        if r != -1:
            return i_line
    return None

def _get_begin_line(lines, n_patchs):
    list_begin_line = []
    for i_patch in range(0, n_patchs):
        r = _extract_patch_line(lines, i_patch+1)
        if r is not None:
            list_begin_line.append(r)
        else:
            raise ValueError(" could not parse the input file")
    return list_begin_line

def _read_line(line):
    chars = line.split(" ")
    data  = []
    for c in chars:
        try:
            i = int(c)
        except ValueError:
            i = None
        else:
            data.append(i)
            continue

        try:
            f = float(c)
        except ValueError:
            f = None
        else:
            data.append(f)
            continue

        if i is None and f is None:
            msg = f"WARNING: Cannot convert str '{c}' to int or float. Moving to next word..."
            print(msg)

    return data

def _read_patch(lines, i_patch, n_lines_per_patch, list_begin_line):

    from igakit.nurbs import NURBS

    i_begin_line = list_begin_line[i_patch-1]
    data_patch = []

    for i in range(i_begin_line+1, i_begin_line + n_lines_per_patch+1):
        data_patch.append(_read_line(lines[i]))

    degree = data_patch[0]
    shape  = data_patch[1]

    xl     = [np.array(i) for i in data_patch[2:2+len(degree)] ]
    xp     = [np.array(i) for i in data_patch[2+len(degree):2+2*len(degree)] ]
    w      = np.array(data_patch[2+2*len(degree)])

    X = [i.reshape(shape, order='F') for i in xp]
    W = w.reshape(shape, order='F')

    points = np.zeros((*shape, 3))
    for i in range(len(shape)):
        points[..., i] = X[i]

    knots = xl

    nrb = NURBS(knots, control=points, weights=W)
    return nrb
