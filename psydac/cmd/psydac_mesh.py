#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import sys

from psydac.cmd.argparse_helpers import add_help_flag, add_version_flag, exit_with_error_message
from psydac.mapping.discrete_gallery import available_mappings_2d, available_mappings_3d

__all__ = (
    'setup_psydac_mesh_parser',
    'psydac_mesh',
    'PSYDAC_MESH_DESCR',
)

PSYDAC_MESH_DESCR = f"""Generate an HDF5 geometry file with a discrete mapping.

Available 2D mappings: {', '.join(available_mappings_2d)}.
Available 3D mappings: {', '.join(available_mappings_3d)}.
"""
#==============================================================================
def setup_psydac_mesh_parser(parser):
    """
    Add the `psydac mesh` arguments to the parser.

    Add the `psydac mesh` arguments to the parser for command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    group = parser.add_argument_group('Mapping')
    megrp = group.add_mutually_exclusive_group(required=True)
    megrp.add_argument('--map-2d',
        type     = str,
        metavar  = 'MAP-NAME',
        choices  = available_mappings_2d,
        dest     = 'map_2d',
        help     = '2D analytical mapping to be interpolated by a spline.'
    )
    megrp.add_argument('--map-3d',
        type     = str,
        metavar  = 'MAP-NAME',
        choices  = available_mappings_3d,
        dest     = 'map_3d',
        help     = '3D analytical mapping to be interpolated by a spline.'
    )

    group = parser.add_argument_group('Discretization')
    group.add_argument('-n',
        required = True,
        type     = int,
        nargs    = '+',
        dest     = 'ncells',
        metavar  = ('N1','N2'),
        help     = 'Number of grid cells (elements) along each dimension.'
    )
    group.add_argument('-d',
        required = True,
        type     = int,
        nargs    = '+',
        dest     = 'degree',
        metavar  = ('P1','P2'),
        help     = 'Spline degree along each dimension.'
    )

    group = parser.add_argument_group('Other options')
    group.add_argument( '-o',
        type     = str,
        default  = 'out.h5',
        dest     = 'filename',
        help     = 'Name of output geometry file (default: out.h5).'
    )
    add_help_flag(group)
    add_version_flag(group)

#==============================================================================
def psydac_mesh(*, map_2d, map_3d, ncells, degree, filename):
    """
    Generate an HDF5 geometry file with a discrete mapping.
    """
    if map_2d:
        ndim = 2
        error = len(degree) != 2 or len(ncells) != 2
    elif map_3d:
        ndim = 3
        error = len(degree) != 3 or len(ncells) != 3

    if error:
        exit_with_error_message(
            f'ncells and degree must have length {ndim} for a {ndim}D mapping'
        )

    map_name = map_2d or map_3d
    export_analytical_mapping(map_name, ncells, degree, filename)

#==============================================================================
def export_analytical_mapping(mapping, ncells, degree, filename, **kwargs):

    from psydac.cad.geometry             import Geometry
    from psydac.mapping.discrete_gallery import discrete_mapping

    # create the discrete mapping from an analytical one
    mapping = discrete_mapping(mapping, ncells=ncells, degree=degree)

    # create a geometry from a discrete mapping
    geometry = Geometry.from_discrete_mapping(mapping)

    # export the geometry
    geometry.export(filename)

#==============================================================================
# TODO [YG 13.01.2026] fix this using psydac.cad.gallery
#def psydac_mesh(*, filename, geo_name, map_name, degree, ncells):
#
#    if len(degree) != len(ncells):
#        raise ValueError('> ncells and degree must have same dimension')
#
#    if geo_name:
#        export_caid_geometry(geo_name, ncells, degree, filename)
#
#    elif map_name:
#        export_analytical_mapping(map_name, ncells, degree, filename)
#
#==============================================================================
# TODO [YG 13.01.2026] fix this using psydac.cad.gallery
#def export_caid_geometry(name, ncells, degree, filename):
#
#    import os.path
#
#    from caid.cad_geometry import cad_geometry
#    from caid.cad_geometry import line
#    from caid.cad_geometry import square
#    from caid.cad_geometry import circle
#    from caid.cad_geometry import cube
#
#    constructor = eval(name)
#    geo = constructor(n=[i-1 for i in ncells], p=degree)
#    # ...
#
#    extension = os.path.splitext(filename)[1]
#    if not extension == '.h5':
#        raise ValueError('> Only h5 extension is allowed for filename')
#
#    geo.save(filename)
