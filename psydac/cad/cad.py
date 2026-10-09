#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from typing import Iterable

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
    try:
        from igakit.nurbs import NURBS
    except:
        raise ImportError('Could not find igakit.')

    assert isinstance(mapping, SplineMapping)
    assert( isinstance(times, int) )
    assert( isinstance(axis, int) )

    space                = mapping.space
    domain_decomposition = space.domain_decomposition
    pdim                 = mapping.pdim

    knots  = [V.knots             for V in space.spaces]
    degree = [V.degree            for V in space.spaces]
    shape  = [V.nbasis            for V in space.spaces]

    # The decomposition does not change, hence each process only needs the
    # coefficients owned by its neighbours that are stored in its ghost regions
    is_nurbs = isinstance(mapping, NurbsMapping)
    input_fields = list(mapping.fields)
    if is_nurbs:
        input_fields.append(mapping.weights_field)
    for f in input_fields:
        f.coeffs.update_ghost_regions()

    points = np.zeros(shape+[mapping.pdim])
    for i, f in enumerate(mapping.fields):
        points[..., i] = f.coeffs.toarray(with_pads=True).reshape(shape)

    weights = None
    if is_nurbs:
        weights = mapping.weights_field.coeffs.toarray(with_pads=True).reshape(shape)

    # degree elevation using igakit, which expects Cartesian control points
    nrb = NURBS(knots, points, weights=weights)
    nrb = nrb.clone().elevate(axis, times)

    spaces = [SplineSpace(degree=p, knots=u) for p,u in zip( nrb.degree, nrb.knots )]
    space  = TensorFemSpace( domain_decomposition, *spaces )
    fields = [FemField( space ) for d in range( pdim )]

    # Get spline coefficients for each coordinate X_i
    starts = space.coeff_space.starts
    ends   = space.coeff_space.ends
    idx_to = tuple( slice( s, e+1 ) for s,e in zip( starts, ends ) )
    for i,field in enumerate( fields ):
        idx_from = tuple(list(idx_to)+[i])
        field.coeffs[idx_to] = nrb.points[idx_from]

        field.coeffs.update_ghost_regions()

    if isinstance(mapping, NurbsMapping):
        weights_field = FemField( space )

        idx_from = idx_to
        weights_field.coeffs[idx_to] = nrb.weights[idx_from]
        weights_field.coeffs.update_ghost_regions()

        fields.append( weights_field )

        return NurbsMapping( *fields )

    return SplineMapping( *fields )


#==============================================================================
# TODO add level
def refine(mapping, axis, values):
    """
    Refine the mapping by inserting values in the direction axis.

    Note: we are using igakit for the moment, until we implement the knot
    insertion algorithm in psydac
    """
    try:
        from igakit.nurbs import NURBS
    except:
        raise ImportError('Could not find igakit.')

    assert isinstance(mapping, SplineMapping)
    assert( isinstance(values, (list, tuple)) )
    assert( isinstance(axis, int) )

    space                = mapping.space
    domain_decomposition = space.domain_decomposition
    pdim                 = mapping.pdim

    knots  = [V.knots             for V in space.spaces]
    degree = [V.degree            for V in space.spaces]
    shape  = [V.nbasis            for V in space.spaces]

    # The new balanced decomposition may move the subdomain boundaries far
    # away, hence each process needs all the coefficients
    points = np.zeros(shape+[mapping.pdim])
    for i, f in enumerate(mapping.fields):
        points[..., i] = _global_array(f.coeffs).reshape(shape)

    weights = None
    if isinstance(mapping, NurbsMapping):
        weights = _global_array(mapping.weights_field.coeffs).reshape(shape)

    # knot insertion using igakit, which expects Cartesian control points
    nrb = NURBS(knots, points, weights=weights)
    nrb = nrb.clone().refine(axis, values)

    spaces = [SplineSpace(degree=p, knots=u) for p,u in zip( nrb.degree, nrb.knots )]

    ncells = list(domain_decomposition.ncells)
    ncells[axis] += len(values)
    domain_decomposition = DomainDecomposition(ncells, domain_decomposition.periods, comm=domain_decomposition.comm)

    space  = TensorFemSpace( domain_decomposition, *spaces )
    fields = [FemField( space ) for d in range( pdim )]

    # Get spline coefficients for each coordinate X_i
    starts = space.coeff_space.starts
    ends   = space.coeff_space.ends
    idx_to = tuple( slice( s, e+1 ) for s,e in zip( starts, ends ) )
    for i,field in enumerate( fields ):
        idx_from = tuple(list(idx_to)+[i])
        field.coeffs[idx_to] = nrb.points[idx_from]
        field.coeffs.update_ghost_regions()

    if isinstance(mapping, NurbsMapping):
        weights_field = FemField( space )

        idx_from = idx_to
        weights_field.coeffs[idx_to] = nrb.weights[idx_from]
        weights_field.coeffs.update_ghost_regions()

        fields.append( weights_field )

        return NurbsMapping( *fields )

    return SplineMapping( *fields )

#==============================================================================
def _global_array(coeffs: StencilVector) -> np.ndarray:
    """Return the global array of coefficients on every process."""
    array = coeffs.toarray()
    # In parallel, toarray() returns zeros outside of the block owned by the process
    if coeffs.space.parallel:
        coeffs.space.cart.comm.Allreduce(MPI.IN_PLACE, array, op=MPI.SUM)
    return array
