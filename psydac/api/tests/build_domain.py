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

def build_2_patch_annulus():
    """
    Build a 180º annulus by connecting two 90º annular patches.
    """
    bounds1   = (0.5, 1.)
    bounds2_A = (0, np.pi/2)
    bounds2_B = (np.pi/2, np.pi)

    A = Square('A',bounds1=bounds1, bounds2=bounds2_A)
    B = Square('B',bounds1=bounds1, bounds2=bounds2_B)

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    D1     = mapping_1(A)
    D2     = mapping_2(B)

    connectivity = [((0,1,1), (1,1,-1), 1)]
    patches = [D1, D2]
    
    domain = Domain.join(patches, connectivity, '2_patch_domain')
    
    return domain

def build_11_patch_pretzel(domain_name='pretzel', r_min=None, r_max=None):
    """
    Build a pretzel-like 2D domain by connecting 11 patches through 13 conforming interfaces.
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

    dom_log_2 = Square('dom2',bounds1=(r_min, r_max), bounds2=(np.pi/2, np.pi))
    mapping_2 = PolarMapping('M2',2, c1= -h, c2= h, rmin = 0., rmax=1.)
    domain_2  = mapping_2(dom_log_2)

    dom_log_3 = Square('dom3',bounds1=(r_min, r_max), bounds2=(np.pi, np.pi*3/2))
    mapping_3 = PolarMapping('M3',2, c1= -h, c2= 0, rmin = 0., rmax=1.)
    domain_3  = mapping_3(dom_log_3)

    dom_log_4 = Square('dom4',bounds1=(r_min, r_max), bounds2=(np.pi*3/2, np.pi*2))
    mapping_4 = PolarMapping('M4',2, c1= h, c2= 0, rmin = 0., rmax=1.)
    domain_4  = mapping_4(dom_log_4)

    dom_log_5 = Square('dom5',bounds1=(-hr,hr) , bounds2=(-h/2, h/2))
    mapping_5 = get_2D_rotation_mapping('M5', c1=h/2, c2=cr , alpha=np.pi/2)
    domain_5  = mapping_5(dom_log_5)

    dom_log_6 = Square('dom6',bounds1=(-hr,hr) , bounds2=(-h/2, h/2))
    mapping_6 = flip_axis('M6', c1=-h/2, c2=cr)
    domain_6  = mapping_6(dom_log_6)

    dom_log_7 = Square('dom7',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
    mapping_7 = get_2D_rotation_mapping('M7', c1=-cr, c2=h/2 , alpha=np.pi)
    domain_7  = mapping_7(dom_log_7)

    dom_log_8 = Square('dom8',bounds1=(-hr,hr) , bounds2=(-h, h))
    mapping_8 = get_2D_rotation_mapping('M8', c1=0, c2=h-cr , alpha=np.pi*3/2)
    domain_8  = mapping_8(dom_log_8)

    dom_log_9 = Square('dom9',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
    mapping_9 = AffineMapping('M9', 2, c1=cr, c2=h/2, a11=1, a22=-1, a21=0, a12=0)
    domain_9  = mapping_9(dom_log_9)

    dom_log_10 = Square('dom10',bounds1=(np.pi*3/2, np.pi*2), bounds2=(r_min, r_max))
    mapping_10 = TransposedPolarMapping('M10',2, c1= -r_min-h, c2= r_min+h, rmin = 0., rmax=1.)
    domain_10  = mapping_10(dom_log_10)

    dom_log_11 = Square('dom11',bounds1=(np.pi, np.pi*3/2), bounds2=(r_min, r_max))
    mapping_11 = TransposedPolarMapping('M11',2, c1= r_min+h, c2= r_min+h, rmin = 0., rmax=1.)
    domain_11  = mapping_11(dom_log_11)

    patches = ([
                    domain_1,
                    domain_2,
                    domain_3,
                    domain_4,
                    domain_5,
                    domain_6,
                    domain_7,
                    domain_8,
                    domain_9,
                    domain_10,
                    domain_11,
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
        [(domain_3,  axis_1, ext_1), (domain_8,  axis_1, ext_0), 1],
        [(domain_8,  axis_1, ext_1), (domain_4,  axis_1, ext_0), 1],
        [(domain_4,  axis_1, ext_1), (domain_9, axis_1, ext_1), 1],
        [(domain_9,  axis_1, ext_0), (domain_1,  axis_1, ext_0), 1],
        [(domain_6,  axis_0, ext_0), (domain_10, axis_0, ext_1), 1],
        [(domain_7,  axis_0, ext_0), (domain_10, axis_0, ext_0), 1],
        [(domain_5,  axis_0, ext_0), (domain_11, axis_0, ext_0), 1],
        [(domain_9, axis_0, ext_0),  (domain_11, axis_0, ext_1), 1],
        ]

    domain = Domain.join(patches, connectivity, name=domain_name)
    
    return domain

