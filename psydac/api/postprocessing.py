# coding: utf-8
#
# Copyright 2019 Yaman Güçlü

from psydac.cad.geometry    import Geometry
from psydac.utilities.utils import refine_array_1d

#==============================================================================
def get_grid_lines_2d(domain_h, V_h, *, refine=1):
    """
    Get the grid lines (i.e. element boundaries) of a 2D computational domain,
    which can be easily plotted with Matplotlib.

    Parameters
    ----------
    domain_h : psydac.cad.geometry.Geometry
        2D single-patch geometry.

    V_h : psydac.fem.tensor.TensorFemSpace
        Spline space from which the breakpoints are extracted.
                    - TODO: remove this argument -

    refine : int
        Number of segments used to describe a grid curve in each element
        (minimum value is 1, which yields quadrilateral elements).

    Results
    -------
    isolines_1 : list of dict
        Lines having constant value of 'eta1' parameter;
        each line is a dictionary with three keys:
            - 'eta1' : value of eta1 on the curve
            - 'x'    : x coordinates of N points along the curve
            - 'y'    : y coordinates of N points along the curve

    isolines_2 : list of dict
        Lines having constant value of 'eta2' parameter;
        each line is a dictionary with three keys:
            - 'eta2' : value of eta2 on the curve
            - 'x'    : x coordinates of N points along the curve
            - 'y'    : y coordinates of N points along the curve

    """
    # Check that domain is of correct type and contains only one patch
    assert isinstance(domain_h, Geometry)
    assert domain_h.ldim == 2
    assert domain_h.pdim == 2
    assert len(domain_h) == 1

    # TODO: improve
    # Get mapping over patch (create identity map if needed)
    mapping = list(domain_h.mappings.values())[0]
    if mapping is None:
        mapping = lambda eta: eta

    # TODO: make this work
    # Get 1D breakpoints in logical domain
    #eta1, eta2 = domain_h.breaks

    # NOTE: temporary solution (V_h should not be passed to this function)
    V1, V2  = V_h.spaces
    eta1 = V1.breaks
    eta2 = V2.breaks

    # Refine logical grid
    eta1_r = refine_array_1d( eta1, refine )
    eta2_r = refine_array_1d( eta2, refine )

    # Compute physical coordinates of lines with eta1=const
    isolines_1 = []
    for e1 in eta1:
        x, y = zip(*[mapping([e1, e2]) for e2 in eta2_r])
        isolines_1.append( dict(eta1=e1, x=x, y=y) )

    # Compute physical coordinates of lines with eta2=const
    isolines_2 = []
    for e2 in eta2:
        x, y = zip(*[mapping([e1, e2]) for e1 in eta1_r])
        isolines_2.append( dict(eta2=e2, x=x, y=y) )

    return isolines_1, isolines_2
