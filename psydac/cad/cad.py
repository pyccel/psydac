#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from typing import Callable, Iterable

import numpy as np
from mpi4py import MPI

from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.fem.basic        import FemField
from psydac.linalg.stencil   import StencilVector
from psydac.mapping.discrete import SplineMapping, NurbsMapping
from psydac.ddm.cart         import DomainDecomposition

#==============================================================================
def translate(mapping : SplineMapping, displ : Iterable[float]):
    """
    Translate a CAD geometry by a given vector displacement.

    Translate a single-patch CAD geometry (given as a spline or NURBS mapping)
    by adding a given displacement vector to the control points of the mapping.
    The weights of a NURBS mapping are not changed.

    Parameters
    ----------
    mapping : SplineMapping
        The discrete mapping to be translated, which represents a CAD geometry.

    displ : Iterable[float]
        The vector displacement by which to translate the geometry.

    Returns
    -------
    SplineMapping
        A new discrete mapping representing the translated geometry. It is a
        NurbsMapping if the input mapping is a NurbsMapping.
    """
    assert isinstance(mapping, SplineMapping)
    assert isinstance(displ, Iterable)

    displ = np.array(displ)
    assert mapping.pdim == len(displ)

    pdim           = mapping.pdim
    space          = mapping.space
    control_points = mapping.control_points

    fields = [FemField(space) for d in range(pdim)]

    # Get spline coefficients for each coordinate X_i
    starts = space.coeff_space.starts
    ends   = space.coeff_space.ends
    idx_to = tuple(slice(s, e+1) for s, e in zip(starts, ends))
    for i, field in enumerate(fields):
        idx_from = tuple(list(idx_to)+[i])
        field.coeffs[idx_to] = control_points[idx_from] + displ[i]
        field.coeffs.update_ghost_regions()

    # The control points are Cartesian, hence the weights are not affected
    if isinstance(mapping, NurbsMapping):
        return NurbsMapping(*fields, mapping.weights_field.copy())

    return SplineMapping(*fields)

#==============================================================================
def elevate(mapping, axis, times):
    """
    Elevate the mapping degree times time in the direction axis.

    Note: we are using igakit for the moment, until we implement the elevation
    degree algorithm in psydac
    """
    assert isinstance(mapping, SplineMapping)
    assert isinstance(times, int)
    assert isinstance(axis, int)

    # The decomposition does not change, hence each process only needs the
    # coefficients owned by its neighbours that are stored in its ghost regions
    nrb = _to_igakit(mapping, _local_array).elevate(axis, times)

    return _from_igakit(nrb, mapping, mapping.space.domain_decomposition)


#==============================================================================
# TODO add level
def refine(mapping, axis, values):
    """
    Refine the mapping by inserting values in the direction axis.

    Note: we are using igakit for the moment, until we implement the knot
    insertion algorithm in psydac
    """
    assert isinstance(mapping, SplineMapping)
    assert isinstance(values, (list, tuple))
    assert isinstance(axis, int)

    # The new balanced decomposition may move the subdomain boundaries far
    # away, hence each process needs all the coefficients
    nrb = _to_igakit(mapping, _global_array).refine(axis, values)

    return _from_igakit(nrb, mapping)

#==============================================================================
def _to_igakit(mapping: SplineMapping, to_array: Callable[[StencilVector], np.ndarray]):
    """
    Create an igakit NURBS object from a spline or NURBS mapping.

    Parameters
    ----------
    mapping : SplineMapping
        The input mapping (possibly a NurbsMapping).

    to_array : Callable[[StencilVector], np.ndarray]
        The function which returns the coefficients of a field as an array
        with the global shape, which must be correct where it is used.

    Returns
    -------
    igakit.nurbs.NURBS
        The igakit object with the Cartesian control points and the weights
        of the mapping.
    """
    try:
        from igakit.nurbs import NURBS
    except ImportError as err:
        raise ImportError('Could not find igakit.') from err

    knots = [V.knots  for V in mapping.space.spaces]
    shape = [V.nbasis for V in mapping.space.spaces]

    # igakit requires at least 2 coordinates, the extra one is zero
    points = np.zeros(shape + [max(mapping.pdim, 2)])
    for i, f in enumerate(mapping.fields):
        points[..., i] = to_array(f.coeffs).reshape(shape)

    weights = None
    if isinstance(mapping, NurbsMapping):
        weights = to_array(mapping.weights_field.coeffs).reshape(shape)

    # igakit expects Cartesian control points
    return NURBS(knots, points, weights=weights)

def _from_igakit(nrb,
                 mapping: SplineMapping,
                 domain_decomposition: DomainDecomposition | None = None) -> SplineMapping:
    """
    Create a mapping of the same type and dimension as a given mapping, from
    an igakit NURBS object.

    Parameters
    ----------
    nrb : igakit.nurbs.NURBS
        The igakit object, whose control points and weights are known on
        every process.

    mapping : SplineMapping
        The mapping whose type (SplineMapping or NurbsMapping), physical
        dimension, periodicity, and MPI communicator are used.

    domain_decomposition : DomainDecomposition, optional
        The decomposition of the new mapping's domain. If not given, a new
        balanced decomposition is created for the new number of cells.

    Returns
    -------
    SplineMapping
        The new mapping. It is a NurbsMapping if `mapping` is a NurbsMapping.
    """
    spaces = [SplineSpace(degree=p, knots=u) for p, u in zip(nrb.degree, nrb.knots)]

    if domain_decomposition is not None:
        ddm = domain_decomposition
    else:
        # The number of cells is given by the spaces, because inserting a value
        # which is already a knot only increases its multiplicity (see #619)
        old_ddm = mapping.space.domain_decomposition
        ddm = DomainDecomposition([W.ncells for W in spaces], old_ddm.periods, comm=old_ddm.comm)

    space = TensorFemSpace(ddm, *spaces)

    arrays = [nrb.points[..., i] for i in range(mapping.pdim)]
    is_nurbs = isinstance(mapping, NurbsMapping)
    if is_nurbs:
        arrays.append(nrb.weights)

    # Each process copies the coefficients that it owns
    idx = tuple(slice(s, e + 1) for s, e in zip(space.coeff_space.starts, space.coeff_space.ends))
    fields = [FemField(space) for _ in arrays]
    for field, array in zip(fields, arrays):
        field.coeffs[idx] = array[idx]
        field.coeffs.update_ghost_regions()

    return NurbsMapping(*fields) if is_nurbs else SplineMapping(*fields)

def _local_array(coeffs: StencilVector) -> np.ndarray:
    """Return the array of coefficients owned by the process and its neighbours."""
    coeffs.update_ghost_regions()
    # In parallel, the array is zero outside of the block and the ghost regions
    return coeffs.toarray(with_pads=True)

def _global_array(coeffs: StencilVector) -> np.ndarray:
    """Return the global array of coefficients on every process."""
    array = coeffs.toarray()
    # In parallel, toarray() returns zeros outside of the block owned by the process
    if coeffs.space.parallel:
        coeffs.space.cart.comm.Allreduce(MPI.IN_PLACE, array, op=MPI.SUM)
    return array
