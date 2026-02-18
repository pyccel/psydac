#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
# todo: this file has a lot of redundant code with psydac/feec/multipatch/multipatch_domain_utilities.py
# it should probably be removed in the future

import numpy as np

from sympde.topology import Square, Domain
from sympde.topology import IdentityMapping, PolarMapping, AffineMapping, Mapping

# remove after sympde PR #155 is merged and call Domain.join instead
from psydac.feec.multipatch_domain_utilities import sympde_Domain_join

#==============================================================================
# small extension to SymPDE:
class TransposedPolarMapping(Mapping):
    """
    Represents a Transposed (x1 <> x2) Polar 2D Mapping object (Annulus).

    Examples

    """
    _expressions = {'x': 'c1 + (rmin*(1-x2)+rmax*x2)*cos(x1)',
                    'y': 'c2 + (rmin*(1-x2)+rmax*x2)*sin(x1)'}

    _ldim        = 2
    _pdim        = 2

# todo: remove this
def create_domain(patches, interfaces, name):
    connectivity = []
    patches_interiors = [D.interior for D in patches]
    for I in interfaces:
        connectivity.append(((patches_interiors.index(I[0].domain),I[0].axis, I[0].ext), (patches_interiors.index(I[1].domain), I[1].axis, I[1].ext), I[2]))
    return Domain.join(patches, connectivity, name)

def get_2D_rotation_mapping(name='no_name', c1=0., c2=0., alpha=np.pi/2):

    # AffineMapping:
    # _expressions = {'x': 'c1 + a11*x1 + a12*x2 + a13*x3',
    #                 'y': 'c2 + a21*x1 + a22*x2 + a23*x3',
    #                 'z': 'c3 + a31*x1 + a32*x2 + a33*x3'}

    return AffineMapping(
        name, 2, c1=c1, c2=c2,
        a11=np.cos(alpha), a12=-np.sin(alpha),
        a21=np.sin(alpha), a22=np.cos(alpha),
    )

def flip_axis(name='no_name', c1=0., c2=0.):

    # AffineMapping:
    # _expressions = {'x': 'c1 + a11*x1 + a12*x2 + a13*x3',
    #                 'y': 'c2 + a21*x1 + a22*x2 + a23*x3',
    #                 'z': 'c3 + a31*x1 + a32*x2 + a33*x3'}

    return AffineMapping(
        name, 2, c1=c1, c2=c2,
        a11=0, a12=1,
        a21=1, a22=0,
    )

#==============================================================================

# todo: use build_multipatch_domain instead
def build_pretzel(domain_name='pretzel', r_min=None, r_max=None):
    """
    design pretzel-like domain
    """

    if r_min is None:
        r_min=1
    if r_max is None:
        r_max=2
 
    assert 0 < r_min
    assert r_min < r_max
    dr = r_max - r_min
    h = dr
    hr = dr/2
    cr = h +(r_max+r_min)/2

    dom_log_1 = Square('dom1',bounds1=(r_min, r_max), bounds2=(0, np.pi/2))
    mapping_1 = PolarMapping('M1',2, c1= h, c2= h, rmin = 0., rmax=1.)
    domain_1  = mapping_1(dom_log_1)

    dom_log_1_1 = Square('dom1_1',bounds1=(r_min, r_max), bounds2=(0, np.pi/4))
    mapping_1_1 = PolarMapping('M1_1',2, c1= h, c2= h, rmin = 0., rmax=1.)
    domain_1_1  = mapping_1_1(dom_log_1_1)

    dom_log_1_2 = Square('dom1_2',bounds1=(r_min, r_max), bounds2=(np.pi/4, np.pi/2))
    mapping_1_2 = PolarMapping('M1_2',2, c1= h, c2= h, rmin = 0., rmax=1.)
    domain_1_2  = mapping_1_2(dom_log_1_2)

    dom_log_2 = Square('dom2',bounds1=(r_min, r_max), bounds2=(np.pi/2, np.pi))
    mapping_2 = PolarMapping('M2',2, c1= -h, c2= h, rmin = 0., rmax=1.)
    domain_2  = mapping_2(dom_log_2)

    dom_log_2_1 = Square('dom2_1',bounds1=(r_min, r_max), bounds2=(np.pi/2, np.pi*3/4))
    mapping_2_1 = PolarMapping('M2_1',2, c1= -h, c2= h, rmin = 0., rmax=1.)
    domain_2_1  = mapping_2_1(dom_log_2_1)

    dom_log_2_2 = Square('dom2_2',bounds1=(r_min, r_max), bounds2=(np.pi*3/4, np.pi))
    mapping_2_2 = PolarMapping('M2_2',2, c1= -h, c2= h, rmin = 0., rmax=1.)
    domain_2_2  = mapping_2_2(dom_log_2_2)

    dom_log_10 = Square('dom10',bounds1=(r_min, r_max), bounds2=(np.pi/2, np.pi))
    mapping_10 = PolarMapping('M10',2, c1= h, c2= h, rmin = 0., rmax=1.)
    domain_10  = mapping_10(dom_log_10)

    dom_log_3 = Square('dom3',bounds1=(r_min, r_max), bounds2=(np.pi, np.pi*3/2))
    mapping_3 = PolarMapping('M3',2, c1= -h, c2= 0, rmin = 0., rmax=1.)
    domain_3  = mapping_3(dom_log_3)

    dom_log_3_1 = Square('dom3_1',bounds1=(r_min, r_max), bounds2=(np.pi, np.pi*5/4))
    mapping_3_1 = PolarMapping('M3_1',2, c1= -h, c2= 0, rmin = 0., rmax=1.)
    domain_3_1  = mapping_3_1(dom_log_3_1)

    dom_log_3_2 = Square('dom3_2',bounds1=(r_min, r_max), bounds2=(np.pi*5/4, np.pi*3/2))
    mapping_3_2 = PolarMapping('M3_2',2, c1= -h, c2= 0, rmin = 0., rmax=1.)
    domain_3_2  = mapping_3_2(dom_log_3_2)

    dom_log_4 = Square('dom4',bounds1=(r_min, r_max), bounds2=(np.pi*3/2, np.pi*2))
    mapping_4 = PolarMapping('M4',2, c1= h, c2= 0, rmin = 0., rmax=1.)
    domain_4  = mapping_4(dom_log_4)

    dom_log_4_1 = Square('dom4_1',bounds1=(r_min, r_max), bounds2=(np.pi*3/2, np.pi*7/4))
    mapping_4_1 = PolarMapping('M4_1',2, c1= h, c2= 0, rmin = 0., rmax=1.)
    domain_4_1  = mapping_4_1(dom_log_4_1)

    dom_log_4_2 = Square('dom4_2',bounds1=(r_min, r_max), bounds2=(np.pi*7/4, np.pi*2))
    mapping_4_2 = PolarMapping('M4_2',2, c1= h, c2= 0, rmin = 0., rmax=1.)
    domain_4_2  = mapping_4_2(dom_log_4_2)

    dom_log_5 = Square('dom5',bounds1=(-hr,hr) , bounds2=(-h/2, h/2))
    mapping_5 = get_2D_rotation_mapping('M5', c1=h/2, c2=cr , alpha=np.pi/2)
    domain_5  = mapping_5(dom_log_5)

    dom_log_6 = Square('dom6',bounds1=(-hr,hr) , bounds2=(-h/2, h/2))
    mapping_6 = flip_axis('M6', c1=-h/2, c2=cr)
    domain_6  = mapping_6(dom_log_6)

    dom_log_7 = Square('dom7',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
    mapping_7 = get_2D_rotation_mapping('M7', c1=-cr, c2=h/2 , alpha=np.pi)
    domain_7  = mapping_7(dom_log_7)

    dom_log_9 = Square('dom9',bounds1=(-hr,hr) , bounds2=(-h, h))
    mapping_9 = get_2D_rotation_mapping('M9', c1=0, c2=h-cr , alpha=np.pi*3/2)
    domain_9  = mapping_9(dom_log_9)

    dom_log_9_1 = Square('dom9_1',bounds1=(-hr,hr) , bounds2=(-h, 0))
    mapping_9_1 = get_2D_rotation_mapping('M9_1', c1=0, c2=h-cr , alpha=np.pi*3/2)
    domain_9_1  = mapping_9_1(dom_log_9_1)

    dom_log_9_2 = Square('dom9_2',bounds1=(-hr,hr) , bounds2=(0, h))
    mapping_9_2 = get_2D_rotation_mapping('M9_2', c1=0, c2=h-cr , alpha=np.pi*3/2)
    domain_9_2  = mapping_9_2(dom_log_9_2)

    dom_log_12 = Square('dom12',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
    mapping_12 = AffineMapping('M12', 2, c1=cr, c2=h/2, a11=1, a22=-1, a21=0, a12=0)
    domain_12  = mapping_12(dom_log_12)

    dom_log_13 = Square('dom13',bounds1=(np.pi*3/2, np.pi*2), bounds2=(r_min, r_max))
    mapping_13 = TransposedPolarMapping('M13',2, c1= -r_min-h, c2= r_min+h, rmin = 0., rmax=1.)
    domain_13  = mapping_13(dom_log_13)

    dom_log_13_1 = Square('dom13_1',bounds1=(np.pi*3/2, np.pi*7/4), bounds2=(r_min, r_max))
    mapping_13_1 = TransposedPolarMapping('M13_1',2, c1= -r_min-h, c2= r_min+h, rmin = 0., rmax=1.)
    domain_13_1  = mapping_13_1(dom_log_13_1)

    dom_log_13_2 = Square('dom13_2',bounds1=(np.pi*7/4, np.pi*2), bounds2=(r_min, r_max))
    mapping_13_2 = TransposedPolarMapping('M13_2',2, c1= -r_min-h, c2= r_min+h, rmin = 0., rmax=1.)
    domain_13_2  = mapping_13_2(dom_log_13_2)

    dom_log_14 = Square('dom14',bounds1=(np.pi, np.pi*3/2), bounds2=(r_min, r_max))
    mapping_14 = TransposedPolarMapping('M14',2, c1= r_min+h, c2= r_min+h, rmin = 0., rmax=1.)
    domain_14  = mapping_14(dom_log_14)

    dom_log_14_1 = Square('dom14_1',bounds1=(np.pi, np.pi*5/4), bounds2=(r_min, r_max))
    mapping_14_1 = TransposedPolarMapping('M14_1',2, c1= r_min+h, c2= r_min+h, rmin = 0., rmax=1.)
    domain_14_1  = mapping_14_1(dom_log_14_1)

    dom_log_14_2 = Square('dom14_2',bounds1=(np.pi*5/4, np.pi*3/2), bounds2=(r_min, r_max))
    mapping_14_2 = TransposedPolarMapping('M14_2',2, c1= r_min+h, c2= r_min+h, rmin = 0., rmax=1.)
    domain_14_2  = mapping_14_2(dom_log_14_2)

    patches = ([
                    domain_1,
                    domain_2,
                    domain_3,
                    domain_4,
                    domain_5,
                    domain_6,
                    domain_7,
                    domain_9,
                    domain_12,
                    domain_13,
                    domain_14,
                    ])

    axis_0 = 0
    axis_1 = 1
    ext_0 = -1
    ext_1 = +1

    connectivity = [
        [(domain_1,  axis_1, ext_1), (domain_5,  axis_1, ext_0), 1],
        [(domain_5,  axis_1, ext_1), (domain_6,  axis_1, ext_1), 1],
        [(domain_6,  axis_1, ext_0), (domain_2,  axis_1, ext_0), 1],
        [(domain_2,  axis_1, ext_1), (domain_7,  axis_1, ext_0), 1],
        [(domain_7,  axis_1, ext_1), (domain_3,  axis_1, ext_0), 1],
        [(domain_3,  axis_1, ext_1), (domain_9,  axis_1, ext_0), 1],
        [(domain_9,  axis_1, ext_1), (domain_4,  axis_1, ext_0), 1],
        [(domain_4,  axis_1, ext_1), (domain_12, axis_1, ext_1), 1],
        [(domain_12, axis_1, ext_0), (domain_1,  axis_1, ext_0), 1],
        [(domain_6,  axis_0, ext_0), (domain_13, axis_0, ext_1), 1],
        [(domain_7,  axis_0, ext_0), (domain_13, axis_0, ext_0), 1],
        [(domain_5,  axis_0, ext_0), (domain_14, axis_0, ext_0), 1],
        [(domain_12, axis_0, ext_0), (domain_14, axis_0, ext_1), 1],
        ]

    # domain = Domain.join(patches, connectivity, name=domain_name)
    domain = sympde_Domain_join(patches, connectivity, name=domain_name)

    return domain

