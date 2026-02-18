#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from mpi4py import MPI

from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace

from psydac.api.discretization     import discretize
from psydac.api.tests.build_domain import build_pretzel
from psydac.api.postprocessing     import OutputManager, PostProcessManager
from psydac.fem.basic import FemField

def save_sample_data(ncells, degree, vector, kind, comm=None):
    domain = build_pretzel()

    if vector or kind in ['hdiv', 'hcurl']:
        space = VectorFunctionSpace('V', domain, kind=kind)
    else:
        space = ScalarFunctionSpace('V', domain, kind=kind)

    domain_h = discretize(domain, ncells=ncells, comm=comm)

    space_h = discretize(space, domain_h, degree=degree)

    f = FemField(space_h)
    for f_f in f.fields:
        if vector or kind in ['hdiv', 'hcurl']:
            f_f.coeffs[0][:] = 4
            f_f.coeffs[1][:] = - 4
        else:
            f_f.coeffs[:] = 4

    Om = OutputManager('sample_data_pretzel.yml', 'sample_data_pretzel.h5', comm=comm)

    Om.add_spaces(V=space_h)
    Om.set_static()
    Om.export_fields(f=f)
    Om.close()

def export_sample_data(npts_per_cell, comm=None):
    Pm = PostProcessManager(
        domain=build_pretzel(),
        space_file='sample_data_pretzel.yml',
        fields_file='sample_data_pretzel.h5',
        comm=comm,
    )

    Pm.export_to_vtk(
        'exported_sample_data',
        grid=None,
        npts_per_cell=npts_per_cell,
        fields='f',
    )


def parse_input_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Saves and export sample data on the pretzel domain"
    )

    parser.add_argument( '-d',
        type    = int,
        nargs   = 2,
        default = [2, 2],
        metavar = ('P1', 'P2'),
        dest    = 'degree',
        help    = 'Spline degree along each dimension'
    )

    parser.add_argument( '-n',
        type    = int,
        nargs   = 2,
        default = [10, 10],
        metavar = ('N1', 'N2'),
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension'
    )

    parser.add_argument('-r',
        action='store',
        default=2,
        type=int,
        metavar='R',
        dest='refinement',
        help='Refinement of the exported model')

    parser.add_argument('-s',
        action='store_true',
        dest='_save',
        help='Save sample data')

    parser.add_argument('-e',
        action='store_true',
        dest='_export',
        help='Export sample data',
    )

    parser.add_argument('-k',
        action='store',
        type=str,
        dest='kind',
        default='h1',
        help='Kind of the space',
    )

    parser.add_argument('-v',
        action='store_true',
        dest='_vector',
        help='Save sample vector data over scalar data',
    )

    return parser.parse_args()

def main(ncells, degree, _save, _export, kind, _vector, refinement):
    comm = MPI.COMM_WORLD
    if _save:
        save_sample_data(ncells, degree, _vector, kind, comm)
    if _export:
        export_sample_data(refinement, comm)


if __name__ == '__main__':
    args = parse_input_arguments()
    main( **vars( args ) )
