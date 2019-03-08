# coding: utf-8
#!/usr/bin/env python

import sys
import os
import argparse
import os.path

# TODO add version
#  --version  show program's version number and exit


class MyParser(argparse.ArgumentParser):
    """
    Custom argument parser for printing help message in case of an error.
    See http://stackoverflow.com/questions/4042452/display-help-message-with-python-argparse-when-script-is-called-without-any-argu
    """
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

#==============================================================================
# TODO add more geoemtries from caid
def export_caid_geometry(name, ncells, degree, filename):
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
#   psydac-mesh -n='8,8' -d='2,2' --analytical=identity identity_2d.h5
def main():
    """
    pyccel console command.
    """
    parser = MyParser(
        usage='%(prog)s [OPTIONS] <FILENAME>',
        epilog="For more information, visit <http://psydac.readthedocs.io/>.",
        description="""psydac mesh generation command line.""")

    # ...
    parser.add_argument('filename', metavar='FILENAME',
                        help='output filename')
    # ...

    # ...
    group = parser.add_argument_group('Geometry')
    group.add_argument('--caid', default = '',
                        help='a geometry name from gallery/caid')

    group.add_argument('--analytical', default = '',
                        help='analytical mapping provided by PSYDAC')
    # ...

    # ...
    group = parser.add_argument_group('Discretization')
    group.add_argument( '-d',
        type    = str,
        dest    = 'degree',
        help    = 'Spline degree along each dimension'
    )

    group.add_argument( '-n',
        type    = str,
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension'
    )
    # ...

    # ...
    args = parser.parse_args()
    # ...

    # ...
    filename = args.filename
    geo_name = args.caid
    analytical = args.analytical
    # ...

    # ...
    ncells = args.ncells
    if not ncells: raise ValueError('> ncells must be given')

    ncells = [int(i) for i in  ncells.split(',')]
    # ...

    # ...
    degree = args.degree
    if not degree: raise ValueError('> degree must be given')

    degree = [int(i) for i in  degree.split(',')]
    # ...

    if geo_name:
        export_caid_geometry(geo_name, ncells, degree, filename)

    elif analytical:
        export_analytical_mapping(analytical, ncells, degree, filename)
