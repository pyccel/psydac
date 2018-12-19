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

# ... TODO use export h5 of mapping when available 
def export_analytical_mapping(mapping, ncells, degree, filename, **kwargs):
    import numpy as np
    from spl.mapping.discrete_gallery import discrete_mapping

    map_discrete, space = discrete_mapping(mapping, ncells, degree,
                                           return_space=True)

    # ... TODO remove this later, once we have export h5
    dim    = space.ldim
    knots  = [V.knots for V in space.spaces]
    coeffs = [f.coeffs.toarray() for f in map_discrete._fields]
    shape  = [V.nbasis for V in space.spaces] + [3]
    points = np.zeros(shape)
    for i,c in enumerate(coeffs):
        points[...,i] = c.reshape([V.nbasis for V in space.spaces])

    from caid.cad_geometry import cad_geometry
    from caid.cad_geometry import cad_nurbs

    nrb = cad_nurbs(knots, points)
    # ...
    rdim = 3
    if dim == 2:
        nrb.orientation = [-1,1,1,-1] # not used
        external_faces = [[0,0],[0,1],[0,2],[0,3]]

    elif dim == 3:
        nrb.orientation = [-1,1,1,-1,1,-1] # not used
        external_faces = [[0,0],[0,1],[0,2],[0,3],[0,4],[0,5]]
    # ...

    # ...
    geo = cad_geometry()
    geo._r_dim = rdim
    geo.append(nrb)

    # TODO must be done automaticaly
    geo._internal_faces = []
    geo._external_faces = external_faces
    geo._connectivity   = []
    # ...

    extension = os.path.splitext(filename)[1]
    if not extension == '.h5':
        raise ValueError('> Only h5 extension is allowed for filename')

    geo.save(filename)
    # ...
# ...


# example of usage:
#  spl-mesh -n='16,16' -d='3,3' --geometry=square mesh.h5
def main():
    """
    pyccel console command.
    """
    parser = MyParser(
        usage='%(prog)s [OPTIONS] <FILENAME>',
        epilog="For more information, visit <http://spl.readthedocs.io/>.",
        description="""spl mesh generation command line.""")

    # ...
    parser.add_argument('filename', metavar='FILENAME',
                        help='output filename')
    # ...

    # ...
    group = parser.add_argument_group('Geometry')
    group.add_argument('--caid', default = '',
                        help='a geometry name from gallery/caid')

    group.add_argument('--analytical', default = '',
                        help='analytical mapping provided by SPL')
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
