#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np
from igakit.cad import circle, ruled, bilinear, join

from psydac.cad.geometry import export_nurbs_to_hdf5, refine_nurbs

# create pipe geometry
C0      = circle(center=(-1,0),angle=(-np.pi/3,0))
C1      = circle(radius=2,center=(-1,0),angle=(-np.pi/3,0))
annulus = ruled(C0,C1).transpose()
square  = bilinear(np.array([[[0,0],[0,3]],[[1,0],[1,3]]]) )
pipe    = join(annulus, square, axis=1)

# refine the nurbs object
ncells = [2**5,2**5]
degree = [2,2]
multiplicity = [2,2]

new_pipe = refine_nurbs(pipe, ncells=ncells, degree=degree, multiplicity=multiplicity)
filename = "pipe.h5"
export_nurbs_to_hdf5(filename, new_pipe)
