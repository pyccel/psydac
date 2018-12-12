# coding: utf-8
#
import numpy as np
import string
import random

from spl.fem.splines      import SplineSpace
from spl.fem.tensor       import TensorFemSpace
from spl.fem.basic        import FemField
from spl.mapping.discrete import SplineMapping

#==============================================================================
def random_string( n ):
    chars    = string.ascii_uppercase + string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )


#==============================================================================
def translate(mapping, displ):
    displ = np.array(displ)
    assert( mapping.pdim == len(displ) )

    pdim = mapping.pdim
    space = mapping.space
    control_points = mapping.control_points

    name   = random_string( 8 )
    fields = [FemField( space, 'mapping_{name}_x{d}'.format( name=name, d=d ) )
              for d in range( pdim )]

    # Get spline coefficients for each coordinate X_i
    starts = space.vector_space.starts
    ends   = space.vector_space.ends
    idx_to = tuple( slice( s, e+1 ) for s,e in zip( starts, ends ) )
    for i,field in enumerate( fields ):
        idx_from = tuple(list(idx_to)+[i])
        field.coeffs[idx_to] = control_points[idx_from] + displ[i]
        field.coeffs.update_ghost_regions()

    return SplineMapping( *fields )

######################################
if __name__ == '__main__':
    from spl.cad.geometry import Geometry

    geo = Geometry('square_0.h5')
    mapping = geo.patches[0]
    new = translate(mapping, [1., 0., 0.])

    geo = Geometry(patches=[mapping, new])
    geo.export('square_mp.h5')
