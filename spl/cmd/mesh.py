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

# ... TODO add more geoemtries from caid
def export_caid_geometry(name, ncells, degree, filename):
    # ...
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

# ...


# example of usage:
#  spl-mesh -n='16,16' -d='3,3' square mesh.h5
def main():
    """
    pyccel console command.
    """
    parser = MyParser(
        usage='%(prog)s [OPTIONS] <GEOMETRY> <FILENAME>',
        epilog="For more information, visit <http://spl.readthedocs.io/>.",
        description="""spl mesh generation command line.""")

    # ...
    parser.add_argument('geometry', metavar='GEOMETRY',
                        help='a geometry name from gallery/caid')

    parser.add_argument('filename', metavar='FILENAME',
                        help='output filename')
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
    geo_name = args.geometry
#    print('> name = ', geo_name)
    # ...

    # ...
    filename = args.filename
#    print('> filename = ', filename)
    # ...

    # ...
    ncells = args.ncells
    if not ncells: raise ValueError('> ncells must be given')

    ncells = [int(i) for i in  ncells.split(',')]
#    print('> ncells = ', ncells)
    # ...

    # ...
    degree = args.degree
    if not degree: raise ValueError('> degree must be given')

    degree = [int(i) for i in  degree.split(',')]
#    print('> degree = ', degree)
    # ...

    export_caid_geometry(geo_name, ncells, degree, filename)
