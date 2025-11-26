#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
# coding: utf-8
#!/usr/bin/env python

# TODO add version
#  --version  show program's version number and exit


#==============================================================================
# TODO add more geoemtries from caid
def export_caid_geometry(name, ncells, degree, filename):

    import os.path

    from caid.cad_geometry import cad_geometry
    from caid.cad_geometry import line
    from caid.cad_geometry import square
    from caid.cad_geometry import circle
    from caid.cad_geometry import cube

    constructor = eval(name)
    geo = constructor(n=[i-1 for i in ncells], p=degree)
    # ...

    extension = os.path.splitext(filename)[1]
    if not extension == '.h5':
        raise ValueError('> Only h5 extension is allowed for filename')

    geo.save(filename)

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
# usage:
#   psydac-mesh --analytical identity -n 8 8 -d 2 2 -o identity_2d.h5
def main():
    """
    pyccel console command.
    """
    import argparse

    parser = argparse.ArgumentParser(
            description="psydac mesh generation command line.",
            epilog = "For more information, visit <http://psydac.readthedocs.io/>.",
            formatter_class = argparse.RawTextHelpFormatter,
            )

    # ...
    group = parser.add_argument_group('Geometry')
    group = group.add_mutually_exclusive_group(required=True)
    group.add_argument('--caid',
        type    = str,
        metavar = 'GEO',
        default = '',
        help    = 'a geometry name from gallery/caid'
    )

    group.add_argument('--analytical',
        type    = str,
        metavar = 'MAP',
        default = '',
        help    = 'analytical mapping from mapping/analytical_gallery'
    )
    # ...

    # ...
    group = parser.add_argument_group('Discretization')
    group.add_argument( '-d',
        required = True,
        type     = int,
        nargs    = '+',
        dest     = 'degree',
        metavar  = ('P1','P2'),
        help     = 'spline degree along each dimension'
    )

    group.add_argument( '-n',
        required = True,
        type     = int,
        nargs    = '+',
        dest     = 'ncells',
        metavar  = ('N1','N2'),
        help     = 'number of grid cells (elements) along each dimension'
    )
    # ...

    # ...
    parser.add_argument( '-o',
        type     = str,
        default  = 'out.h5',
        dest     = 'filename',
        help     = 'Name of output geometry file (default: out.h5)'
    )
    # ...

    # ...
    args = parser.parse_args()
    # ...

    # ...
    filename   = args.filename
    geo_name   = args.caid
    analytical = args.analytical
    degree     = args.degree
    ncells     = args.ncells
    # ...

    # ...
    if len(degree) != len(ncells):
        raise ValueError('> ncells and degree must have same dimension')
    # ...

    if geo_name:
        export_caid_geometry(geo_name, ncells, degree, filename)

    elif analytical:
        export_analytical_mapping(analytical, ncells, degree, filename)
