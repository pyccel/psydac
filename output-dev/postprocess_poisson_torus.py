from psydac.api.postprocessing import PostProcessManager
from psydac.utilities.utils import refine_array_1d
import numpy as np

def post_process(degree, ncells, npts_per_cell, is_logical):
    p1, p2, p3 = degree
    ne1, ne2, ne3 = ncells
    Pm = PostProcessManager(geometry_file=f'TargetTorusMapping_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}.h5', 
                            space_file=f'spaces_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.yml', 
                            fields_file=f'fields_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.h5')

    grid = [refine_array_1d(Pm.spaces['V'].breaks[i], n =npts_per_cell - 1, remove_duplicates=False) for i in range(3)]
    npts_per_cell = [npts_per_cell] * 3

    u_e_logical = lambda x,y,z: (0.05 - x) * (0.2 - x) * np.sin(y) * np.cos(z) 
    u_e_physical = lambda x,y,z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z) 

    phy_f = {}
    log_f = {}

    if is_logical:
        log_f[u_e_logical] = 'u_e_log' 
    else:
        phy_f[u_e_physical] = 'u_e_phy' 


    Pm.export_to_vtk(f'poisson_3d_target_torus_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{npts_per_cell}_{is_logical}', 
                     grid=grid, npts_per_cell=npts_per_cell, snapshots='none', logical_grid=True, fields={'u':'u'}, 
                     additional_physical_functions=phy_f, additional_logical_functions=log_f)
                     
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        description     = "Solve Poisson's equation on a 3D domain with" +
                          " homogeneous Dirichlet boundary conditions."
    )

    parser.add_argument( '-d',
        type    = int,
        nargs   = 3,
        default = [2, 2, 2],
        metavar = ('P1', 'P2', 'P3'),
        dest    = 'degree',
        help    = 'Spline degree along each dimension'
    )

    parser.add_argument( '-n',
        type    = int,
        nargs   = 3,
        default = [20, 20, 20],
        metavar = ('N1', 'N2', 'N3'),
        dest    = 'ncells',
        help    = 'Number of grid cells (elements) along each dimension'
    )

    parser.add_argument('-p',
        type=int,
        default=3,
        metavar='N',
        dest='npts_per_cell',
        )
    parser.add_argument('-l',
        action='store_true',
        dest='is_logical')
    
    args = parser.parse_args()

    post_process(args.degree, args.ncells, args.npts_per_cell, args.is_logical)