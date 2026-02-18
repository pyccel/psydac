#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from psydac.cad.multipatch import export_multipatch_nurbs_to_hdf5
from igakit.cad import bilinear
from igakit.cad import circle
from igakit.cad import ruled
from igakit.plot import plt

b1 = bilinear([((0.5,-1),(0.5,0.5)),((1,-1),(1,0.5))])
b2 = bilinear([((-1,-1),(-1,0.5)),((-0.5,-1),(-0.5,0.5))])

c1 = circle(radius=0.5,center=(0,0.5), angle=(0,np.pi/2))
c2 = circle(radius=1,center=(0,0.5), angle=(0,np.pi/2))
c3 = circle(radius=0.5,center=(0,0.5), angle=(np.pi/2,np.pi))
c4 = circle(radius=1,center=(0,0.5), angle=(np.pi/2,np.pi))

srf1 = ruled(c1,c2)
srf2 = ruled(c3,c4)

srf1.transpose()
srf2.transpose()

b2.reverse(0)

srf1.elevate(0,1)

srf2.elevate(0,1)

b1.elevate(0,1)
b1.elevate(1,1)

b2.elevate(0,1)
b2.elevate(1,1)

srf1.refine(0,[0.25,0.5,0.75])
srf1.refine(1,[0.25,0.5,0.75])

srf2.refine(0,[0.25,0.5,0.75])
srf2.refine(1,[0.25,0.5,0.75])

b1.refine(0,[0.25,0.5,0.75])
b1.refine(1,[0.25,0.5,0.75])

b2.refine(0,[0.25,0.5,0.75])
b2.refine(1,[0.25,0.5,0.75])

filename     = 'magnet.h5'
nurbs        = [b1, srf1, srf2, b2]
connectivity = {(0,1):((1,1),(1,-1)),(1,2):((1,1),(1,-1)),(2,3):((1,1),(1,1))}

export_multipatch_nurbs_to_hdf5(filename, nurbs, connectivity)

