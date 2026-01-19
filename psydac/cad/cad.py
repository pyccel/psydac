#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from psydac.fem.splines      import SplineSpace
from psydac.fem.tensor       import TensorFemSpace
from psydac.fem.basic        import FemField
from psydac.mapping.discrete import SplineMapping, NurbsMapping
from psydac.ddm.cart         import DomainDecomposition

#==============================================================================
def translate(mapping, displ):
    displ = np.array(displ)
    assert( mapping.pdim == len(displ) )

    pdim           = mapping.pdim
    space          = mapping.space
    control_points = mapping.control_points

    fields = [FemField( space ) for d in range( pdim )]

    # Get spline coefficients for each coordinate X_i
    starts = space.coeff_space.starts
    ends   = space.coeff_space.ends
    idx_to = tuple( slice( s, e+1 ) for s,e in zip( starts, ends ) )
    for i,field in enumerate( fields ):
        idx_from = tuple(list(idx_to)+[i])
        field.coeffs[idx_to] = control_points[idx_from] + displ[i]
        field.coeffs.update_ghost_regions()

    return SplineMapping( *fields )

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

    assert( isinstance(mapping, (SplineSpace, NurbsMapping)) )
    assert( isinstance(times, int) )
    assert( isinstance(axis, int) )

    space                = mapping.space
    domain_decomposition = space.domain_decomposition
    pdim                 = mapping.pdim

    knots  = [V.knots             for V in space.spaces]
    degree = [V.degree            for V in space.spaces]
    shape  = [V.nbasis            for V in space.spaces]
    points = np.zeros(shape+[mapping.pdim])
    for i,f in enumerate( mapping._fields ):
        points[...,i] = f._coeffs.toarray().reshape(shape)

    weights = None
    if isinstance(mapping, NurbsMapping):
        weights = mapping._weights_field._coeffs.toarray().reshape(shape)

        for i in range(pdim):
            points[...,i] /= weights[...]

    # degree elevation using igakit
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
        idw_from = tuple(idx_to)
        if isinstance(mapping, NurbsMapping):
            field.coeffs[idx_to] = nrb.points[idx_from] * nrb.weights[idw_from]

        else:
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

    assert( isinstance(mapping, (SplineSpace, NurbsMapping)) )
    assert( isinstance(values, (list, tuple)) )
    assert( isinstance(axis, int) )

    space                = mapping.space
    domain_decomposition = space.domain_decomposition
    pdim                 = mapping.pdim

    knots  = [V.knots             for V in space.spaces]
    degree = [V.degree            for V in space.spaces]
    shape  = [V.nbasis            for V in space.spaces]
    points = np.zeros(shape+[mapping.pdim])
    for i,f in enumerate( mapping._fields ):
        points[...,i] = f._coeffs.toarray().reshape(shape)

    weights = None
    if isinstance(mapping, NurbsMapping):
        weights = mapping._weights_field._coeffs.toarray().reshape(shape)

        for i in range(pdim):
            points[...,i] /= weights[...]

    # degree elevation using igakit
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
        idw_from = tuple(idx_to)
        if isinstance(mapping, NurbsMapping):
            field.coeffs[idx_to] = nrb.points[idx_from] * nrb.weights[idw_from]

        else:
            field.coeffs[idx_to] = nrb.points[idx_from]

    if isinstance(mapping, NurbsMapping):
        weights_field = FemField( space )

        idx_from = idx_to
        weights_field.coeffs[idx_to] = nrb.weights[idx_from]
        weights_field.coeffs.update_ghost_regions()

        fields.append( weights_field )

        return NurbsMapping( *fields )

    return SplineMapping( *fields )



######################################
if __name__ == '__main__':
    from psydac.cad.geometry import Geometry

    geo = Geometry('square_0.h5')
    mapping = geo.patches[0]
    new = translate(mapping, [1., 0., 0.])

    geo = Geometry(patches=[mapping, new])
    geo.export('square_mp.h5')
