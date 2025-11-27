#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from mpi4py import MPI

import numpy as np

from sympde.topology import Square, Domain
# TransposedPolarMapping
from sympde.topology import IdentityMapping, PolarMapping, AffineMapping, Mapping

__all__ = (
    'TransposedPolarMapping',
    'get_2D_rotation_mapping',
    'flip_axis',
    'build_multipatch_domain',
    'build_cartesian_multipatch_domain')

# ==============================================================================
# small extension to SymPDE:


class TransposedPolarMapping(Mapping):
    """
    Represents a Transposed (x1 <> x2) Polar 2D Mapping object (Annulus).

    """
    _expressions = {'x': 'c1 + (rmin*(1-x2)+rmax*x2)*cos(x1)',
                    'y': 'c2 + (rmin*(1-x2)+rmax*x2)*sin(x1)'}

    _ldim = 2
    _pdim = 2


def sympde_Domain_join(patches, connectivity, name):
    """
    temporary fix while sympde PR #155 is not merged
    """        
    connectivity_by_indices = []
    for I in connectivity:           
        connectivity_by_indices.append(
            [(patches.index(I[0][0]), I[0][1], I[0][2]), 
             (patches.index(I[1][0]), I[1][1], I[1][2]),
             I[2]])
    return Domain.join(patches, connectivity_by_indices, name)


def get_2D_rotation_mapping(name='no_name', c1=0., c2=0., alpha=None):

    # AffineMapping:
    # _expressions = {'x': 'c1 + a11*x1 + a12*x2 + a13*x3',
    #                 'y': 'c2 + a21*x1 + a22*x2 + a23*x3',
    #                 'z': 'c3 + a31*x1 + a32*x2 + a33*x3'}    
    if alpha is None: 
        alpha = np.pi/2
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


def build_multipatch_domain(domain_name='square_2', r_min=None, r_max=None):
    """
    Create a 2D multipatch domain among the many available.
    These include fairly complex pretzel-like shapes.

    Parameters
    ----------
    domain_name: <str>
     The name of the chosen domain, which can be one of the following:
      'square_2', 'square_6', 'square_8', 'square_9', 'annulus_3', 'annulus_4',
      'curved_L_shape', 'pretzel', 'pretzel_f', 'pretzel_annulus', 'pretzel_debug'

    Returns
    -------
    domain : <Sympde.topology.Domain>
     The symbolic multipatch domain
    """

    connectivity = None

    # for readability
    axis_0 = 0
    axis_1 = 1
    ext_0 = -1
    ext_1 = +1

    # create the patches
    if domain_name == 'square_2':
        # reference square [0,pi]x[0,pi] with 2 patches
        # mp structure:
        # 2
        # 1
        OmegaLog1 = Square(
            'OmegaLog1', bounds1=(
                0., np.pi), bounds2=(
                0., np.pi / 2))
        mapping_1 = IdentityMapping('M1', 2)
        domain_1 = mapping_1(OmegaLog1)

        OmegaLog2 = Square(
            'OmegaLog2', bounds1=(
                0., np.pi), bounds2=(
                np.pi / 2, np.pi))
        mapping_2 = IdentityMapping('M2', 2)
        domain_2 = mapping_2(OmegaLog2)

        patches = [domain_1, domain_2]

        connectivity = [[(domain_1, axis_1, ext_1), (domain_2, axis_1, ext_0), 1]]
    elif domain_name == 'square_4':
        # C D
        # A B

        A = Square('A', bounds1=(0, np.pi / 2), bounds2=(0, np.pi / 2))
        B = Square('B', bounds1=(np.pi / 2, np.pi), bounds2=(0, np.pi / 2))
        C = Square('C', bounds1=(0, np.pi / 2), bounds2=(np.pi / 2, np.pi))
        D = Square('D', bounds1=(np.pi / 2, np.pi), bounds2=(np.pi / 2, np.pi))
        M1 = IdentityMapping('M1', dim=2)
        M2 = IdentityMapping('M2', dim=2)
        M3 = IdentityMapping('M3', dim=2)
        M4 = IdentityMapping('M4', dim=2)
        A = M1(A)
        B = M2(B)
        C = M3(C)
        D = M4(D)

        patches = [A, B, C, D]

        connectivity = [
            [(A, axis_0, ext_1), (B, axis_0, ext_0), 1],
            [(A, axis_1, ext_1), (C, axis_1, ext_0), 1],
            [(C, axis_0, ext_1), (D, axis_0, ext_0), 1],
            [(B, axis_1, ext_1), (D, axis_1, ext_0), 1],
        ]

    elif domain_name == 'square_6':

        # mp structure:
        # 5 6
        # 3 4
        # 1 2
        OmegaLog1 = Square(
            'OmegaLog1',
            bounds1=(
                0.,
                np.pi / 2),
            bounds2=(
                0.,
                np.pi / 3))
        mapping_1 = IdentityMapping('M1', 2)
        domain_1 = mapping_1(OmegaLog1)

        OmegaLog2 = Square(
            'OmegaLog2',
            bounds1=(
                np.pi / 2,
                np.pi),
            bounds2=(
                0.,
                np.pi / 3))
        mapping_2 = IdentityMapping('M2', 2)
        domain_2 = mapping_2(OmegaLog2)

        OmegaLog3 = Square(
            'OmegaLog3',
            bounds1=(
                0.,
                np.pi / 2),
            bounds2=(
                np.pi / 3,
                np.pi * 2 / 3))
        mapping_3 = IdentityMapping('M3', 2)
        domain_3 = mapping_3(OmegaLog3)

        OmegaLog4 = Square(
            'OmegaLog4',
            bounds1=(
                np.pi / 2,
                np.pi),
            bounds2=(
                np.pi / 3,
                np.pi * 2 / 3))
        mapping_4 = IdentityMapping('M4', 2)
        domain_4 = mapping_4(OmegaLog4)

        OmegaLog5 = Square(
            'OmegaLog5',
            bounds1=(
                0.,
                np.pi / 2),
            bounds2=(
                np.pi * 2 / 3,
                np.pi))
        mapping_5 = IdentityMapping('M5', 2)
        domain_5 = mapping_5(OmegaLog5)

        OmegaLog6 = Square(
            'OmegaLog6',
            bounds1=(
                np.pi / 2,
                np.pi),
            bounds2=(
                np.pi * 2 / 3,
                np.pi))
        mapping_6 = IdentityMapping('M6', 2)
        domain_6 = mapping_6(OmegaLog6)

        patches = [domain_1, domain_2, domain_3, domain_4, domain_5, domain_6]

        connectivity = [
            [(domain_1, axis_0, ext_1), (domain_2, axis_0, ext_0), 1],
            [(domain_3, axis_0, ext_1), (domain_4, axis_0, ext_0), 1],
            [(domain_5, axis_0, ext_1), (domain_6, axis_0, ext_0), 1],
            [(domain_1, axis_1, ext_1), (domain_3, axis_1, ext_0), 1],
            [(domain_3, axis_1, ext_1), (domain_5, axis_1, ext_0), 1],
            [(domain_2, axis_1, ext_1), (domain_4, axis_1, ext_0), 1],
            [(domain_4, axis_1, ext_1), (domain_6, axis_1, ext_0), 1],
        ]

    elif domain_name in ['square_8', 'square_9']:
        # square with third-length patches, with or without a hole:

        OmegaLog1 = Square(
            'OmegaLog1',
            bounds1=(
                0.,
                np.pi / 3),
            bounds2=(
                0.,
                np.pi / 3))
        mapping_1 = IdentityMapping('M1', 2)
        domain_1 = mapping_1(OmegaLog1)

        OmegaLog2 = Square(
            'OmegaLog2',
            bounds1=(
                np.pi / 3,
                np.pi * 2 / 3),
            bounds2=(
                0.,
                np.pi / 3))
        mapping_2 = IdentityMapping('M2', 2)
        domain_2 = mapping_2(OmegaLog2)

        OmegaLog3 = Square(
            'OmegaLog3',
            bounds1=(
                np.pi * 2 / 3,
                np.pi),
            bounds2=(
                0.,
                np.pi / 3))
        mapping_3 = IdentityMapping('M3', 2)
        domain_3 = mapping_3(OmegaLog3)

        OmegaLog4 = Square(
            'OmegaLog4',
            bounds1=(
                0.,
                np.pi / 3),
            bounds2=(
                np.pi / 3,
                np.pi * 2 / 3))
        mapping_4 = IdentityMapping('M4', 2)
        domain_4 = mapping_4(OmegaLog4)

        OmegaLog5 = Square(
            'OmegaLog5',
            bounds1=(
                np.pi * 2 / 3,
                np.pi),
            bounds2=(
                np.pi / 3,
                np.pi * 2 / 3))
        mapping_5 = IdentityMapping('M5', 2)
        domain_5 = mapping_5(OmegaLog5)

        OmegaLog6 = Square(
            'OmegaLog6',
            bounds1=(
                0.,
                np.pi / 3),
            bounds2=(
                np.pi * 2 / 3,
                np.pi))
        mapping_6 = IdentityMapping('M6', 2)
        domain_6 = mapping_6(OmegaLog6)

        OmegaLog7 = Square(
            'OmegaLog7',
            bounds1=(
                np.pi / 3,
                np.pi * 2 / 3),
            bounds2=(
                np.pi * 2 / 3,
                np.pi))
        mapping_7 = IdentityMapping('M7', 2)
        domain_7 = mapping_7(OmegaLog7)

        OmegaLog8 = Square(
            'OmegaLog8',
            bounds1=(
                np.pi * 2 / 3,
                np.pi),
            bounds2=(
                np.pi * 2 / 3,
                np.pi))
        mapping_8 = IdentityMapping('M8', 2)
        domain_8 = mapping_8(OmegaLog8)

        # center domain
        OmegaLog9 = Square(
            'OmegaLog9',
            bounds1=(
                np.pi / 3,
                np.pi * 2 / 3),
            bounds2=(
                np.pi / 3,
                np.pi * 2 / 3))
        mapping_9 = IdentityMapping('M9', 2)
        domain_9 = mapping_9(OmegaLog9)

        if domain_name == 'square_8':
            # square domain with a hole:
            # 6 7 8
            # 4 * 5
            # 1 2 3

            patches = [
                domain_1,
                domain_2,
                domain_3,
                domain_4,
                domain_5,
                domain_6,
                domain_7,
                domain_8]

            connectivity = [
                [(domain_1, axis_0, ext_1), (domain_2, axis_0, ext_0), 1],
                [(domain_2, axis_0, ext_1), (domain_3, axis_0, ext_0), 1],
                [(domain_6, axis_0, ext_1), (domain_7, axis_0, ext_0), 1],
                [(domain_7, axis_0, ext_1), (domain_8, axis_0, ext_0), 1],
                [(domain_1, axis_1, ext_1), (domain_4, axis_1, ext_0), 1],
                [(domain_4, axis_1, ext_1), (domain_6, axis_1, ext_0), 1],
                [(domain_3, axis_1, ext_1), (domain_5, axis_1, ext_0), 1],
                [(domain_5, axis_1, ext_1), (domain_8, axis_1, ext_0), 1],
            ]

        elif domain_name == 'square_9':
            # square domain with no hole:
            # 6 7 8
            # 4 9 5
            # 1 2 3

            patches = [
                domain_1,
                domain_2,
                domain_3,
                domain_4,
                domain_5,
                domain_6,
                domain_7,
                domain_8,
                domain_9]

            connectivity = [
                [(domain_1, axis_0, ext_1), (domain_2, axis_0, ext_0), 1],
                [(domain_2, axis_0, ext_1), (domain_3, axis_0, ext_0), 1],
                [(domain_4, axis_0, ext_1), (domain_9, axis_0, ext_0), 1],
                [(domain_9, axis_0, ext_1), (domain_5, axis_0, ext_0), 1],
                [(domain_6, axis_0, ext_1), (domain_7, axis_0, ext_0), 1],
                [(domain_7, axis_0, ext_1), (domain_8, axis_0, ext_0), 1],
                [(domain_1, axis_1, ext_1), (domain_4, axis_1, ext_0), 1],
                [(domain_4, axis_1, ext_1), (domain_6, axis_1, ext_0), 1],
                [(domain_2, axis_1, ext_1), (domain_9, axis_1, ext_0), 1],
                [(domain_9, axis_1, ext_1), (domain_7, axis_1, ext_0), 1],
                [(domain_3, axis_1, ext_1), (domain_5, axis_1, ext_0), 1],
                [(domain_5, axis_1, ext_1), (domain_8, axis_1, ext_0), 1],
            ]

        else:
            raise ValueError(domain_name)

    elif domain_name in ['pretzel', 'pretzel_f', 'pretzel_annulus', 'pretzel_debug']:
        # pretzel-shaped domain with quarter-annuli and quadrangles -- setting parameters
        # note: 'pretzel_f' is a bit finer than 'pretzel', to have a roughly
        # uniform resolution (patches of approx same size)
        if r_min is None:
            r_min = 1  # smaller radius of quarter-annuli
        if r_max is None:
            r_max = 2  # larger radius of quarter-annuli
        assert 0 < r_min
        assert r_min < r_max
        dr = r_max - r_min
        h = dr  # offset from axes of quarter-annuli
        hr = dr / 2
        cr = h + (r_max + r_min) / 2

        dom_log_1 = Square(
            'dom1', bounds1=(
                r_min, r_max), bounds2=(
                0, np.pi / 2))
        mapping_1 = PolarMapping('M1', 2, c1=h, c2=h, rmin=0., rmax=1.)
        domain_1 = mapping_1(dom_log_1)

        dom_log_1_1 = Square(
            'dom1_1', bounds1=(
                r_min, r_max), bounds2=(
                0, np.pi / 4))
        mapping_1_1 = PolarMapping('M1_1', 2, c1=h, c2=h, rmin=0., rmax=1.)
        domain_1_1 = mapping_1_1(dom_log_1_1)

        dom_log_1_2 = Square(
            'dom1_2', bounds1=(
                r_min, r_max), bounds2=(
                np.pi / 4, np.pi / 2))
        mapping_1_2 = PolarMapping('M1_2', 2, c1=h, c2=h, rmin=0., rmax=1.)
        domain_1_2 = mapping_1_2(dom_log_1_2)

        dom_log_2 = Square(
            'dom2', bounds1=(
                r_min, r_max), bounds2=(
                np.pi / 2, np.pi))
        mapping_2 = PolarMapping('M2', 2, c1=-h, c2=h, rmin=0., rmax=1.)
        domain_2 = mapping_2(dom_log_2)

        dom_log_2_1 = Square(
            'dom2_1', bounds1=(
                r_min, r_max), bounds2=(
                np.pi / 2, np.pi * 3 / 4))
        mapping_2_1 = PolarMapping('M2_1', 2, c1=-h, c2=h, rmin=0., rmax=1.)
        domain_2_1 = mapping_2_1(dom_log_2_1)

        dom_log_2_2 = Square(
            'dom2_2', bounds1=(
                r_min, r_max), bounds2=(
                np.pi * 3 / 4, np.pi))
        mapping_2_2 = PolarMapping('M2_2', 2, c1=-h, c2=h, rmin=0., rmax=1.)
        domain_2_2 = mapping_2_2(dom_log_2_2)

        # for debug:
        dom_log_10 = Square(
            'dom10', bounds1=(
                r_min, r_max), bounds2=(
                np.pi / 2, np.pi))
        mapping_10 = PolarMapping('M10', 2, c1=h, c2=h, rmin=0., rmax=1.)
        domain_10 = mapping_10(dom_log_10)

        dom_log_3 = Square(
            'dom3', bounds1=(
                r_min, r_max), bounds2=(
                np.pi, np.pi * 3 / 2))
        mapping_3 = PolarMapping('M3', 2, c1=-h, c2=0, rmin=0., rmax=1.)
        domain_3 = mapping_3(dom_log_3)

        dom_log_3_1 = Square(
            'dom3_1', bounds1=(
                r_min, r_max), bounds2=(
                np.pi, np.pi * 5 / 4))
        mapping_3_1 = PolarMapping('M3_1', 2, c1=-h, c2=0, rmin=0., rmax=1.)
        domain_3_1 = mapping_3_1(dom_log_3_1)

        dom_log_3_2 = Square(
            'dom3_2', bounds1=(
                r_min, r_max), bounds2=(
                np.pi * 5 / 4, np.pi * 3 / 2))
        mapping_3_2 = PolarMapping('M3_2', 2, c1=-h, c2=0, rmin=0., rmax=1.)
        domain_3_2 = mapping_3_2(dom_log_3_2)

        dom_log_4 = Square(
            'dom4', bounds1=(
                r_min, r_max), bounds2=(
                np.pi * 3 / 2, np.pi * 2))
        mapping_4 = PolarMapping('M4', 2, c1=h, c2=0, rmin=0., rmax=1.)
        domain_4 = mapping_4(dom_log_4)

        dom_log_4_1 = Square(
            'dom4_1', bounds1=(
                r_min, r_max), bounds2=(
                np.pi * 3 / 2, np.pi * 7 / 4))
        mapping_4_1 = PolarMapping('M4_1', 2, c1=h, c2=0, rmin=0., rmax=1.)
        domain_4_1 = mapping_4_1(dom_log_4_1)

        dom_log_4_2 = Square(
            'dom4_2', bounds1=(
                r_min, r_max), bounds2=(
                np.pi * 7 / 4, np.pi * 2))
        mapping_4_2 = PolarMapping('M4_2', 2, c1=h, c2=0, rmin=0., rmax=1.)
        domain_4_2 = mapping_4_2(dom_log_4_2)

        dom_log_5 = Square('dom5', bounds1=(-hr, hr), bounds2=(-h / 2, h / 2))
        mapping_5 = get_2D_rotation_mapping(
            'M5', c1=h / 2, c2=cr, alpha=np.pi / 2)
        domain_5 = mapping_5(dom_log_5)

        dom_log_6 = Square('dom6', bounds1=(-hr, hr), bounds2=(-h / 2, h / 2))
        mapping_6 = flip_axis('M6', c1=-h / 2, c2=cr)
        domain_6 = mapping_6(dom_log_6)

        dom_log_7 = Square('dom7', bounds1=(-hr, hr), bounds2=(-h / 2, h / 2))
        mapping_7 = get_2D_rotation_mapping(
            'M7', c1=-cr, c2=h / 2, alpha=np.pi)
        domain_7 = mapping_7(dom_log_7)

        # dom_log_8 = Square('dom8',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
        # mapping_8 = get_2D_rotation_mapping('M8', c1=-cr, c2=-h/2 , alpha=np.pi)
        # domain_8  = mapping_8(dom_log_8)

        dom_log_9 = Square('dom9', bounds1=(-hr, hr), bounds2=(-h, h))
        mapping_9 = get_2D_rotation_mapping(
            'M9', c1=0, c2=h - cr, alpha=np.pi * 3 / 2)
        domain_9 = mapping_9(dom_log_9)

        dom_log_9_1 = Square('dom9_1', bounds1=(-hr, hr), bounds2=(-h, 0))
        mapping_9_1 = get_2D_rotation_mapping(
            'M9_1', c1=0, c2=h - cr, alpha=np.pi * 3 / 2)
        domain_9_1 = mapping_9_1(dom_log_9_1)

        dom_log_9_2 = Square('dom9_2', bounds1=(-hr, hr), bounds2=(0, h))
        mapping_9_2 = get_2D_rotation_mapping(
            'M9_2', c1=0, c2=h - cr, alpha=np.pi * 3 / 2)
        domain_9_2 = mapping_9_2(dom_log_9_2)

        # dom_log_10 = Square('dom10',bounds1=(-hr,hr) , bounds2=(-h/2, h/2))
        # mapping_10 = get_2D_rotation_mapping('M10', c1=h/2, c2=h-cr , alpha=np.pi*3/2)
        # domain_10  = mapping_10(dom_log_10)
        #
        # dom_log_11 = Square('dom11',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
        # mapping_11 = get_2D_rotation_mapping('M11', c1=cr, c2=-h/2 , alpha=0)
        # domain_11  = mapping_11(dom_log_11)

        dom_log_12 = Square('dom12', bounds1=(-hr, hr),
                            bounds2=(-h / 2, h / 2))
        #mapping_12 = get_2D_rotation_mapping('M12', c1=cr, c2=h/2 , alpha=0)
        mapping_12 = AffineMapping(
            'M12',
            2,
            c1=cr,
            c2=h / 2,
            a11=1,
            a22=-1,
            a21=0,
            a12=0)
        domain_12 = mapping_12(dom_log_12)

        dom_log_13 = Square(
            'dom13',
            bounds1=(
                np.pi * 3 / 2,
                np.pi * 2),
            bounds2=(
                r_min,
                r_max))
        mapping_13 = TransposedPolarMapping(
            'M13', 2, c1=-r_min - h, c2=r_min + h, rmin=0., rmax=1.)
        domain_13 = mapping_13(dom_log_13)

        dom_log_13_1 = Square(
            'dom13_1',
            bounds1=(
                np.pi * 3 / 2,
                np.pi * 7 / 4),
            bounds2=(
                r_min,
                r_max))
        mapping_13_1 = TransposedPolarMapping(
            'M13_1', 2, c1=-r_min - h, c2=r_min + h, rmin=0., rmax=1.)
        domain_13_1 = mapping_13_1(dom_log_13_1)

        dom_log_13_2 = Square(
            'dom13_2',
            bounds1=(
                np.pi * 7 / 4,
                np.pi * 2),
            bounds2=(
                r_min,
                r_max))
        mapping_13_2 = TransposedPolarMapping(
            'M13_2', 2, c1=-r_min - h, c2=r_min + h, rmin=0., rmax=1.)
        domain_13_2 = mapping_13_2(dom_log_13_2)

        dom_log_14 = Square(
            'dom14',
            bounds1=(
                np.pi,
                np.pi * 3 / 2),
            bounds2=(
                r_min,
                r_max))
        mapping_14 = TransposedPolarMapping(
            'M14', 2, c1=r_min + h, c2=r_min + h, rmin=0., rmax=1.)
        domain_14 = mapping_14(dom_log_14)

        dom_log_14_1 = Square(
            'dom14_1',
            bounds1=(
                np.pi,
                np.pi * 5 / 4),
            bounds2=(
                r_min,
                r_max))      # STOP ICI: check domain
        mapping_14_1 = TransposedPolarMapping(
            'M14_1', 2, c1=r_min + h, c2=r_min + h, rmin=0., rmax=1.)
        domain_14_1 = mapping_14_1(dom_log_14_1)

        dom_log_14_2 = Square(
            'dom14_2',
            bounds1=(
                np.pi * 5 / 4,
                np.pi * 3 / 2),
            bounds2=(
                r_min,
                r_max))
        mapping_14_2 = TransposedPolarMapping(
            'M14_2', 2, c1=r_min + h, c2=r_min + h, rmin=0., rmax=1.)
        domain_14_2 = mapping_14_2(dom_log_14_2)

        # dom_log_15 = Square('dom15', bounds1=(-r_min-h, r_min+h), bounds2=(0, h))
        # mapping_15 = IdentityMapping('M15', 2)
        # domain_15  = mapping_15(dom_log_15)

        if domain_name == 'pretzel':
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

            connectivity = [
                [(domain_1, axis_1, ext_1), (domain_5, axis_1, ext_0), 1],
                [(domain_5, axis_1, ext_1), (domain_6, axis_1, ext_1), 1],
                [(domain_6, axis_1, ext_0), (domain_2, axis_1, ext_0), 1],
                [(domain_2, axis_1, ext_1), (domain_7, axis_1, ext_0), 1],
                [(domain_7, axis_1, ext_1), (domain_3, axis_1, ext_0), 1],
                [(domain_3, axis_1, ext_1), (domain_9, axis_1, ext_0), 1],
                [(domain_9, axis_1, ext_1), (domain_4, axis_1, ext_0), 1],
                [(domain_4, axis_1, ext_1), (domain_12, axis_1, ext_1), 1],
                [(domain_12, axis_1, ext_0), (domain_1, axis_1, ext_0), 1],
                [(domain_6, axis_0, ext_0), (domain_13, axis_0, ext_1), 1],
                [(domain_7, axis_0, ext_0), (domain_13, axis_0, ext_0), 1],
                [(domain_5, axis_0, ext_0), (domain_14, axis_0, ext_0), 1],
                [(domain_12, axis_0, ext_0), (domain_14, axis_0, ext_1), 1],
            ]

        elif domain_name == 'pretzel_f':
            patches = ([
                domain_1_1,
                domain_1_2,
                domain_2_1,
                domain_2_2,
                domain_3_1,
                domain_3_2,
                domain_4_1,
                domain_4_2,
                domain_5,
                domain_6,
                domain_7,
                domain_9_1,
                domain_9_2,
                domain_12,
                domain_13_1,
                domain_13_2,
                domain_14_1,
                domain_14_2,
            ])

            # interfaces = [
            #     [domain_1_1.get_boundary(axis=1, ext=+1), domain_1_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_1_2.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=1, ext=-1), 1],
            #     [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=1), 1],
            #     [domain_6.get_boundary(axis=1, ext=-1), domain_2_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_2_1.get_boundary(axis=1, ext=+1), domain_2_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_2_2.get_boundary(axis=1, ext=+1), domain_7.get_boundary(axis=1, ext=-1), 1],
            #     [domain_7.get_boundary(axis=1, ext=+1), domain_3_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_3_1.get_boundary(axis=1, ext=+1), domain_3_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_3_2.get_boundary(axis=1, ext=+1), domain_9_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_9_1.get_boundary(axis=1, ext=+1), domain_9_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_9_2.get_boundary(axis=1, ext=+1), domain_4_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_4_1.get_boundary(axis=1, ext=+1), domain_4_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_4_2.get_boundary(axis=1, ext=+1), domain_12.get_boundary(axis=1, ext=1), 1],
            #     [domain_12.get_boundary(axis=1, ext=-1), domain_1_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_6.get_boundary(axis=0, ext=-1), domain_13_2.get_boundary(axis=0, ext=1), 1],
            #     [domain_13_2.get_boundary(axis=0, ext=-1), domain_13_1.get_boundary(axis=0, ext=1), 1],
            #     [domain_7.get_boundary(axis=0, ext=-1), domain_13_1.get_boundary(axis=0, ext=-1), 1],
            #     [domain_5.get_boundary(axis=0, ext=-1), domain_14_1.get_boundary(axis=0, ext=-1), 1],
            #     [domain_14_1.get_boundary(axis=0, ext=+1), domain_14_2.get_boundary(axis=0, ext=-1), 1],
            #     [domain_12.get_boundary(axis=0, ext=-1), domain_14_2.get_boundary(axis=0, ext=+1), 1],
            # ]

            # interfaces = [
            #     [domain_1_1.get_boundary(axis=1, ext=+1), domain_1_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_1_2.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=1, ext=-1), 1],
            #     [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=1), 1],
            #     [domain_6.get_boundary(axis=1, ext=-1), domain_2_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_2_1.get_boundary(axis=1, ext=+1), domain_2_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_2_2.get_boundary(axis=1, ext=+1), domain_7.get_boundary(axis=1, ext=-1), 1],
            #     [domain_7.get_boundary(axis=1, ext=+1), domain_3_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_3_1.get_boundary(axis=1, ext=+1), domain_3_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_3_2.get_boundary(axis=1, ext=+1), domain_9_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_9_1.get_boundary(axis=1, ext=+1), domain_9_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_9_2.get_boundary(axis=1, ext=+1), domain_4_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_4_1.get_boundary(axis=1, ext=+1), domain_4_2.get_boundary(axis=1, ext=-1), 1],
            #     [domain_4_2.get_boundary(axis=1, ext=+1), domain_12.get_boundary(axis=1, ext=1), 1],
            #     [domain_12.get_boundary(axis=1, ext=-1), domain_1_1.get_boundary(axis=1, ext=-1), 1],
            #     [domain_6, axis_0, ext=-1), domain_13_2, axis_0, ext=1), 1],
            #     [domain_13_2, axis_0, ext=-1), domain_13_1, axis_0, ext=1), 1],
            #     [domain_7, axis_0, ext=-1), domain_13_1, axis_0, ext=-1), 1],
            #     [domain_5, axis_0, ext=-1), domain_14_1, axis_0, ext=-1), 1],
            #     [domain_14_1, axis_0, ext=+1), domain_14_2, axis_0, ext=-1), 1],
            #     [domain_12, axis_0, ext=-1), domain_14_2, axis_0, ext=+1), 1],
            # ]

            connectivity = [
                [(domain_1_1,  axis_1, ext_1), (domain_1_2,  axis_1, ext_0), 1],
                [(domain_1_2,  axis_1, ext_1), (domain_5,    axis_1, ext_0), 1],
                [(domain_5,    axis_1, ext_1), (domain_6,    axis_1, ext_1), 1],
                [(domain_6,    axis_1, ext_0), (domain_2_1,  axis_1, ext_0), 1],
                [(domain_2_1,  axis_1, ext_1), (domain_2_2,  axis_1, ext_0), 1],
                [(domain_2_2,  axis_1, ext_1), (domain_7,    axis_1, ext_0), 1],
                [(domain_7,    axis_1, ext_1), (domain_3_1,  axis_1, ext_0), 1],
                [(domain_3_1,  axis_1, ext_1), (domain_3_2,  axis_1, ext_0), 1],
                [(domain_3_2,  axis_1, ext_1), (domain_9_1,  axis_1, ext_0), 1],
                [(domain_9_1,  axis_1, ext_1), (domain_9_2,  axis_1, ext_0), 1],
                [(domain_9_2,  axis_1, ext_1), (domain_4_1,  axis_1, ext_0), 1],
                [(domain_4_1,  axis_1, ext_1), (domain_4_2,  axis_1, ext_0), 1],
                [(domain_4_2,  axis_1, ext_1), (domain_12,   axis_1, ext_1), 1],
                [(domain_12,   axis_1, ext_0), (domain_1_1,  axis_1, ext_0), 1],
                [(domain_6,    axis_0, ext_0), (domain_13_2, axis_0, ext_1), 1],
                [(domain_13_2, axis_0, ext_0), (domain_13_1, axis_0, ext_1), 1],
                [(domain_7,    axis_0, ext_0), (domain_13_1, axis_0, ext_0), 1],
                [(domain_5,    axis_0, ext_0), (domain_14_1, axis_0, ext_0), 1],
                [(domain_14_1, axis_0, ext_1), (domain_14_2, axis_0, ext_0), 1],
                [(domain_12,   axis_0, ext_0), (domain_14_2, axis_0, ext_1), 1],
            ]

            # reste: 13 et 14

        elif domain_name == 'pretzel_annulus':
            # only the annulus part of the pretzel (not the inner arcs)

            patches = ([
                domain_1,
                domain_5,
                domain_6,
                domain_2,
                domain_7,
                domain_3,
                domain_9,
                domain_4,
                domain_12,
            ])

            connectivity = [
                [(domain_1, axis_1, ext_1), (domain_5, axis_1, ext_0), 1],
                [(domain_5, axis_1, ext_1), (domain_6, axis_1, ext_0), 1],
                [(domain_6, axis_1, ext_1), (domain_2, axis_1, ext_0), 1],
                [(domain_2, axis_1, ext_1), (domain_7, axis_1, ext_0), 1],
                [(domain_7, axis_1, ext_1), (domain_3, axis_1, ext_0), 1],
                [(domain_3, axis_1, ext_1), (domain_9, axis_1, ext_0), 1],
                [(domain_9, axis_1, ext_1), (domain_4, axis_1, ext_0), 1],
                [(domain_4, axis_1, ext_1), (domain_12, axis_1, ext_0), 1],
                [(domain_12, axis_1, ext_1), (domain_1, axis_1, ext_0), 1],
            ]

        elif domain_name == 'pretzel_debug':
            patches = ([
                domain_1,
                domain_10,
            ])

            connectivity = [[(domain_1, axis_1, ext_1), (domain_10, axis_1, ext_0), 1], ]

        else:
            raise NotImplementedError

    elif domain_name == 'curved_L_shape':
        # Curved L-shape benchmark domain of Monique Dauge, see 2DomD in https://perso.univ-rennes1.fr/monique.dauge/core/index.html
        # here with 3 patches
        dom_log_1 = Square('dom1', bounds1=(2, 3), bounds2=(0., np.pi / 8))
        mapping_1 = PolarMapping('M1', 2, c1=0., c2=0., rmin=0., rmax=1.)
        domain_1 = mapping_1(dom_log_1)

        dom_log_2 = Square(
            'dom2', bounds1=(
                2, 3), bounds2=(
                np.pi / 8, np.pi / 4))
        mapping_2 = PolarMapping('M2', 2, c1=0., c2=0., rmin=0., rmax=1.)
        domain_2 = mapping_2(dom_log_2)

        dom_log_3 = Square(
            'dom3', bounds1=(
                1, 2), bounds2=(
                np.pi / 8, np.pi / 4))
        mapping_3 = PolarMapping('M3', 2, c1=0., c2=0., rmin=0., rmax=1.)
        domain_3 = mapping_3(dom_log_3)

        patches = ([
            domain_1,
            domain_2,
            domain_3,
        ])

        connectivity = [
            [(domain_1, axis_1, ext_1), (domain_2, axis_1, ext_0), 1],
            [(domain_3, axis_0, ext_1), (domain_2, axis_0, ext_0), 1],
        ]

    elif domain_name in ['annulus_3', 'annulus_4']:
        # regular annulus
        if r_min is None:
            r_min = 0.5  # smaller radius
        if r_max is None:
            r_max = 1.  # larger radius

        if domain_name == 'annulus_3':
            OmegaLog1 = Square(
                'OmegaLog1', bounds1=(
                    r_min, r_max), bounds2=(
                    0., np.pi / 2))
            mapping_1 = PolarMapping('M1', 2, c1=0., c2=0., rmin=0., rmax=1.)
            domain_1 = mapping_1(OmegaLog1)

            OmegaLog2 = Square(
                'OmegaLog2', bounds1=(
                    r_min, r_max), bounds2=(
                    np.pi / 2, np.pi))
            mapping_2 = PolarMapping('M2', 2, c1=0., c2=0., rmin=0., rmax=1.)
            domain_2 = mapping_2(OmegaLog2)

            OmegaLog3 = Square(
                'OmegaLog3', bounds1=(
                    r_min, r_max), bounds2=(
                    np.pi, 2 * np.pi))
            mapping_3 = PolarMapping('M3', 2, c1=0., c2=0., rmin=0., rmax=1.)
            domain_3 = mapping_3(OmegaLog3)

            patches = [domain_1, domain_2, domain_3]

            connectivity = [
                [(domain_1, axis_1, ext_1), (domain_2, axis_1, ext_0), 1],
                [(domain_2, axis_1, ext_1), (domain_3, axis_1, ext_0), 1],
                [(domain_3, axis_1, ext_1), (domain_1, axis_1, ext_0), 1],
            ]

        elif domain_name == 'annulus_4':
            OmegaLog1 = Square(
                'OmegaLog1', bounds1=(
                    r_min, r_max), bounds2=(
                    0., np.pi / 2))
            mapping_1 = PolarMapping('M1', 2, c1=0., c2=0., rmin=0., rmax=1.)
            domain_1 = mapping_1(OmegaLog1)

            OmegaLog2 = Square(
                'OmegaLog2', bounds1=(
                    r_min, r_max), bounds2=(
                    np.pi / 2, np.pi))
            mapping_2 = PolarMapping('M2', 2, c1=0., c2=0., rmin=0., rmax=1.)
            domain_2 = mapping_2(OmegaLog2)

            OmegaLog3 = Square(
                'OmegaLog3', bounds1=(
                    r_min, r_max), bounds2=(
                    np.pi, np.pi * 3 / 2))
            mapping_3 = PolarMapping('M3', 2, c1=0., c2=0., rmin=0., rmax=1.)
            domain_3 = mapping_3(OmegaLog3)

            OmegaLog4 = Square(
                'OmegaLog4', bounds1=(
                    r_min, r_max), bounds2=(
                    np.pi * 3 / 2, np.pi * 2))
            mapping_4 = PolarMapping('M4', 2, c1=0., c2=0., rmin=0., rmax=1.)
            domain_4 = mapping_4(OmegaLog4)

            patches = [domain_1, domain_2, domain_3, domain_4]

            connectivity = [
                [(domain_1, axis_1, ext_1), (domain_2, axis_1, ext_0), 1],
                [(domain_2, axis_1, ext_1), (domain_3, axis_1, ext_0), 1],
                [(domain_3, axis_1, ext_1), (domain_4, axis_1, ext_0), 1],
                [(domain_4, axis_1, ext_1), (domain_1, axis_1, ext_0), 1],
            ]
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    # domain = Domain.join(patches, connectivity, name='domain')
    domain = sympde_Domain_join(patches, connectivity, name='domain')
    
    return domain


def build_cartesian_multipatch_domain(ncells, log_interval_x, log_interval_y, mapping='identity'):
    """
    Create a 2D multipatch Cartesian domain with the prescribed pattern of patches and with possible mappings.

    Parameters
    ----------
    ncells: <matrix>      
        (Incomplete) Cartesian grid of patches, where some patches may be empty. 
        The pattern of the multipatch domain is defined by the non-None entries of the matrix ncells. 
        (Different numerical values will give rise to the same multipatch decompostion)

        ncells can then be used (afterwards) for discretizing the domain with ncells[i,j] being the number of cells
        (assumed isotropic in each patch) in the patch (i,j), and None for empty patches (removed from domain).

      Example:

      >>> ncells = np.array([[1, None, 5],
      >>>                    [2,    3, 4]])
     
      corresponds to a domain with 5 patches as follows:
      
      >>> |X| |X|
      >>> -------
      >>> |X|X|X|

    log_interval_x: <Tuple>
        The interval in the x direction in the logical domain.
    log_interval_y: <Tuple>
        The interval in the y direction in the logical domain.
    mapping: <String>
        The type of mapping to use. Can be 'identity' or 'polar'.

    Returns
    -------
    domain : <Sympde.topology.Domain>
        The symbolic multipatch domain
    """
    ax, bx = log_interval_x
    ay, by = log_interval_y
    nb_patchx, nb_patchy = np.shape(ncells)

    # equidistant logical patches
    # ensure the following lists have the same shape as ncells
    list_log_patches = [[Square('Log_' + str(j) + '_' + str(i),
                          bounds1=(ax + j / nb_patchx * (bx - ax), ax + (j + 1) / nb_patchx * (bx - ax)),
                          bounds2=(by - (i + 1) / nb_patchy * (by - ay), by - i / nb_patchy * (by - ay)))
                            for i in range(nb_patchy)] for j in range(nb_patchx)]

    # mappings
    if mapping == 'identity':
        list_mapping = [[IdentityMapping('M_' + str(j) + '_' + str(i), 2)
                            for i in range(nb_patchy)] for j in range(nb_patchx)]
    elif mapping == 'polar':
        list_mapping = [[PolarMapping('M_' + str(j) + '_' + str(i), 2, c1=0., c2=0., rmin=0., rmax=1.) 
                            for i in range(nb_patchy)] for j in range(nb_patchx)]

    # list of physical patches
    list_patches = [[list_mapping[j][i](list_log_patches[j][i]) 
                        for i in range(nb_patchy)] for j in range(nb_patchx)]

    # flatten for the join function
    patches = []
    for i in range(nb_patchx):
        for j in range(nb_patchy):
            if ncells[i, j] is not None:
                patches.append(list_patches[j][i])

    axis_0 = 0
    axis_1 = 1
    ext_0 = -1
    ext_1 = +1
    connectivity = []

    # interfaces in y
    for i in range(nb_patchx):
        connectivity.extend([
            [(list_patches[j  ][i], axis_0, ext_1), 
             (list_patches[j+1][i], axis_0, ext_0), 1] 
            for j in range(nb_patchy -1) if ncells[i][j] is not None and ncells[i][j+1] is not None])

    # interfaces in x
    for j in range(nb_patchy):
        connectivity.extend([
            [(list_patches[j][i  ], axis_1, ext_0), 
             (list_patches[j][i+1], axis_1, ext_1), 1] 
            for i in range(nb_patchx -1) if ncells[i][j] is not None and ncells[i+1][j] is not None])

    # domain = Domain.join(patches, connectivity, name='domain')
    domain = sympde_Domain_join(patches, connectivity, name='domain')

    return domain
    