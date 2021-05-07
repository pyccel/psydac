# coding: utf-8

from mpi4py import MPI

import numpy as np

from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping, AffineMapping, Mapping #TransposedPolarMapping


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





def union(domains, name):
    assert len(domains)>1
    domain = domains[0]
    for p in domains[1:]:
        domain = domain.join(p, name=name)
    return domain

def set_interfaces(domain, interfaces):
    for I in interfaces:
        domain = domain.join(domain, domain.name, bnd_minus=I[0], bnd_plus=I[1], direction=I[2])
    return domain

def get_annulus_fourpatches(r_min, r_max):

    dom_log_1 = Square('dom1',bounds1=(r_min, r_max), bounds2=(0, np.pi/2))
    dom_log_2 = Square('dom2',bounds1=(r_min, r_max), bounds2=(np.pi/2, np.pi))
    dom_log_3 = Square('dom3',bounds1=(r_min, r_max), bounds2=(np.pi, np.pi*3/2))
    dom_log_4 = Square('dom4',bounds1=(r_min, r_max), bounds2=(np.pi*3/2, np.pi*2))

    mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_3 = PolarMapping('M3',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
    mapping_4 = PolarMapping('M4',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

    domain_1     = mapping_1(dom_log_1)
    domain_2     = mapping_2(dom_log_2)
    domain_3     = mapping_3(dom_log_3)
    domain_4     = mapping_4(dom_log_4)

    interfaces = [
        [domain_1.get_boundary(axis=1, ext=1), domain_2.get_boundary(axis=1, ext=-1), 1],
        [domain_2.get_boundary(axis=1, ext=1), domain_3.get_boundary(axis=1, ext=-1), 1],
        [domain_3.get_boundary(axis=1, ext=1), domain_4.get_boundary(axis=1, ext=-1), 1],
        [domain_4.get_boundary(axis=1, ext=1), domain_1.get_boundary(axis=1, ext=-1), 1]
        ]
    domain = union([domain_1, domain_2, domain_3, domain_4], name = 'domain')
    domain = set_interfaces(domain, interfaces)

    return domain


def get_2D_rotation_mapping(name='no_name', c1=0, c2=0, alpha=np.pi/2):

    # AffineMapping:
    # _expressions = {'x': 'c1 + a11*x1 + a12*x2 + a13*x3',
    #                 'y': 'c2 + a21*x1 + a22*x2 + a23*x3',
    #                 'z': 'c3 + a31*x1 + a32*x2 + a33*x3'}

    return AffineMapping(
        name, 2, c1=c1, c2=c2,
        a11=np.cos(alpha), a12=-np.sin(alpha),
        a21=np.sin(alpha), a22=np.cos(alpha),
    )

def get_pretzel(h, r_min, r_max, debug_option=1):
    """
    design pretzel-shaped domain with quarter-annuli and quadrangles
    :param h: offset from axes of quarter-annuli
    :param r_min: smaller radius of quarter-annuli
    :param r_max: larger radius of quarter-annuli
    :return: domain, mappings
    """

    assert 0 < r_min
    assert r_min < r_max

    #h == dr
    dr = r_max - r_min
    hr = dr/2
    cr = h +(r_max+r_min)/2

    # print("building domain: hr = ", hr, ", h = ", h)

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
    mapping_6 = get_2D_rotation_mapping('M6', c1=-h/2, c2=cr , alpha=np.pi/2)
    domain_6  = mapping_6(dom_log_6)

    dom_log_7 = Square('dom7',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
    mapping_7 = get_2D_rotation_mapping('M7', c1=-cr, c2=h/2 , alpha=np.pi)
    domain_7  = mapping_7(dom_log_7)

    dom_log_8 = Square('dom8',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
    mapping_8 = get_2D_rotation_mapping('M8', c1=-cr, c2=-h/2 , alpha=np.pi)
    domain_8  = mapping_8(dom_log_8)

    dom_log_9 = Square('dom9',bounds1=(-hr,hr) , bounds2=(-h, h))
    mapping_9 = get_2D_rotation_mapping('M9', c1=0, c2=h-cr , alpha=np.pi*3/2)
    domain_9  = mapping_9(dom_log_9)

    dom_log_10 = Square('dom10',bounds1=(-hr,hr) , bounds2=(-h/2, h/2))
    mapping_10 = get_2D_rotation_mapping('M10', c1=h/2, c2=h-cr , alpha=np.pi*3/2)
    domain_10  = mapping_10(dom_log_10)

    dom_log_11 = Square('dom11',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
    mapping_11 = get_2D_rotation_mapping('M11', c1=cr, c2=-h/2 , alpha=0)
    domain_11  = mapping_11(dom_log_11)

    dom_log_12 = Square('dom12',bounds1=(-hr, hr), bounds2=(-h/2, h/2))
    mapping_12 = get_2D_rotation_mapping('M12', c1=cr, c2=h/2 , alpha=0)
    domain_12  = mapping_12(dom_log_12)

    dom_log_13 = Square('dom13',bounds1=(np.pi*3/2, np.pi*2), bounds2=(r_min, r_max))
    mapping_13 = TransposedPolarMapping('M13',2, c1= -r_max, c2= r_max, rmin = 0., rmax=1.)
    domain_13  = mapping_13(dom_log_13)

    dom_log_14 = Square('dom14',bounds1=(np.pi, np.pi*3/2), bounds2=(r_min, r_max))
    mapping_14 = TransposedPolarMapping('M14',2, c1= r_max, c2= r_max, rmin = 0., rmax=1.)
    #mapping_14 = get_2D_rotation_mapping('M14', c1=-2*np.pi, c2=0 , alpha=0)
    domain_14  = mapping_14(dom_log_14)

    dom_log_15 = Square('dom15', bounds1=(-r_min-h, r_min+h), bounds2=(0, h))
    mapping_15 = IdentityMapping('M15', 2)
    domain_15  = mapping_15(dom_log_15)

    if debug_option == 0:
        domain = union([
                        domain_1,
                        domain_5,
                        domain_6,
                        domain_2,
                        domain_7,
                        domain_3,
                        domain_9,
                        domain_4,
                        domain_12,
                        domain_13,
                        domain_14,
                        ], name = 'domain')

        interfaces = [
            [domain_1.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=1, ext=-1),1],
            [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=-1),1],
            [domain_6.get_boundary(axis=1, ext=+1), domain_2.get_boundary(axis=1, ext=-1),1],
            [domain_2.get_boundary(axis=1, ext=+1), domain_7.get_boundary(axis=1, ext=-1),1],
            [domain_7.get_boundary(axis=1, ext=+1), domain_3.get_boundary(axis=1, ext=-1),1],
            [domain_3.get_boundary(axis=1, ext=+1), domain_9.get_boundary(axis=1, ext=-1),1],
            [domain_9.get_boundary(axis=1, ext=+1), domain_4.get_boundary(axis=1, ext=-1),1],
            [domain_4.get_boundary(axis=1, ext=+1), domain_12.get_boundary(axis=1, ext=-1),1],
            [domain_12.get_boundary(axis=1, ext=+1), domain_1.get_boundary(axis=1, ext=-1),1],
            [domain_6.get_boundary(axis=0, ext=-1), domain_13.get_boundary(axis=0, ext=+1),-1],
            [domain_7.get_boundary(axis=0, ext=-1), domain_13.get_boundary(axis=0, ext=-1),1],
            [domain_5.get_boundary(axis=0, ext=-1), domain_14.get_boundary(axis=0, ext=-1), 1],
            [domain_12.get_boundary(axis=0, ext=-1), domain_14.get_boundary(axis=0, ext=+1), -1],
            ]


    elif debug_option == 1:
        # only the annulus part of the pretzel (not the inner arcs)

        domain = union([
                        domain_1,
                        domain_5,
                        domain_6,
                        domain_2,
                        domain_7,
                        domain_3,
                        domain_9,
                        domain_4,
                        domain_12,
                        ], name = 'domain')

        interfaces = [
            [domain_1.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=1, ext=-1),1],
            [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=-1),1],
            [domain_6.get_boundary(axis=1, ext=+1), domain_2.get_boundary(axis=1, ext=-1),1],
            [domain_2.get_boundary(axis=1, ext=+1), domain_7.get_boundary(axis=1, ext=-1),1],
            [domain_7.get_boundary(axis=1, ext=+1), domain_3.get_boundary(axis=1, ext=-1),1],
            [domain_3.get_boundary(axis=1, ext=+1), domain_9.get_boundary(axis=1, ext=-1),1],
            [domain_9.get_boundary(axis=1, ext=+1), domain_4.get_boundary(axis=1, ext=-1),1],
            [domain_4.get_boundary(axis=1, ext=+1), domain_12.get_boundary(axis=1, ext=-1),1],
            [domain_12.get_boundary(axis=1, ext=+1), domain_1.get_boundary(axis=1, ext=-1),1],
            ]

    elif debug_option == 2:
        domain = union([
                        #domain_1,
                        domain_14,
                        domain_12,
                       # domain_5,
                        ], name = 'domain')

        interfaces = [
             #   [domain_1.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=1, ext=-1),1],
             #   [domain_5.get_boundary(axis=0, ext=-1), domain_14.get_boundary(axis=0, ext=-1), 1],
                [domain_12.get_boundary(axis=0, ext=-1), domain_14.get_boundary(axis=0, ext=+1),-1],
            #    [domain_12.get_boundary(axis=1, ext=+1), domain_1.get_boundary(axis=1, ext=-1),1],
        ]

    elif debug_option == 23:
        domain = union([domain_5, domain_14,
                        ], name = 'domain')

        interfaces = [
            [domain_5.get_boundary(axis=0, ext=-1), domain_14.get_boundary(axis=0, ext=-1)],
            ]

    elif debug_option == 2:
        domain = union([
            domain_5,
            domain_6,
        ], name = 'domain')

        interfaces = [
            [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_5.interior:mapping_5,
            dom_log_6.interior:mapping_6,
        }  # Q (MCP): purpose of a dict ?


    elif debug_option == 3:
        domain = union([
            domain_1,
            domain_5,
            domain_6,
        ], name = 'domain')

        interfaces = [
            [domain_1.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=1, ext=-1)],
            [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_1.interior:mapping_1,
            dom_log_5.interior:mapping_5,
            dom_log_6.interior:mapping_6,
        }  # Q (MCP): purpose of a dict ?

    elif debug_option == 4:
        domain = union([
            domain_1,
            domain_5,
            domain_6,
            domain_2,
        ], name = 'domain')

        interfaces = [
            [domain_1.get_boundary(axis=1, ext=+1), domain_5.get_boundary(axis=1, ext=-1)],
            [domain_5.get_boundary(axis=1, ext=+1), domain_6.get_boundary(axis=1, ext=-1)],
            [domain_6.get_boundary(axis=1, ext=+1), domain_2.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_1.interior:mapping_1,
            dom_log_5.interior:mapping_5,
            dom_log_6.interior:mapping_6,
            dom_log_2.interior:mapping_2,
        }  # Q (MCP): purpose of a dict ?


    elif debug_option == 10:
        domain = union([
            domain_10,
            domain_2,
        ], name = 'domain')

        interfaces = [
            [domain_10.get_boundary(axis=1, ext=+1), domain_2.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_10.interior:mapping_10,
            dom_log_2.interior:mapping_2,
        }  # Q (MCP): purpose of a dict ?

    elif debug_option == 11:
        domain = union([
            domain_6,
            domain_2,
        ], name = 'domain')

        interfaces = [
            [domain_6.get_boundary(axis=1, ext=+1), domain_2.get_boundary(axis=1, ext=-1)],
            ]

        mappings  = {
            dom_log_6.interior:mapping_6,
            dom_log_2.interior:mapping_2,
        }  # Q (MCP): purpose of a dict ?

    else:
        raise NotImplementedError

    domain = set_interfaces(domain, interfaces)

    # print("int: ", domain.interior)
    # print("bound: ", domain.boundary)
    # print("len(bound): ", len(domain.boundary))
    # print("interfaces: ", domain.interfaces)

    return domain
