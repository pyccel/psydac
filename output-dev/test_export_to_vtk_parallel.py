import os
import numpy as np
import h5py as h5

from sympde.topology import ScalarFunctionSpace, Domain, VectorFunctionSpace

from psydac.api.postprocessing import PostProcessManager, OutputManager
from psydac.fem.basic import FemField
from psydac.utilities.utils import refine_array_1d
from psydac.api.discretization import discretize

try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1


def test_parallel_export(comm, geometry=None):
    
    filename = os.path.join(mesh_dir, geometry)

    domain = Domain.from_file(filename=filename)
    domainh = discretize(domain, filename=filename, comm=comm)

    V = ScalarFunctionSpace('V', domain, kind='h1')
    Vv = VectorFunctionSpace('Vv', domain, kind = 'h1')
    Vh = discretize(V, domainh, degree=[3] * domainh.ldim)
    Vvh = discretize(Vv, domainh, degree=[3] * domainh.ldim)

    f = FemField(Vh)
    ff = FemField(Vvh)

    Om = OutputManager('space_test.yml', 'fields_test.h5', comm=comm, mode='w')

    Om.add_spaces(Vh=Vh, Vvh=Vvh)
    Om.export_space_info()
    Om.set_static()
    f.coeffs[:] = 15
    Om.export_fields(f=f)
    pi = np.pi
    for i in range(20):
        f.coeffs[:] = i
        ff.coeffs[0][:] = np.cos(pi/ 10 * i)
        ff.coeffs[1][:] = np.sin(pi/ 10 * i)
        Om.add_snapshot(t=float(i), ts=i)
        Om.export_fields(ff=ff, f_t=f)
    Om.export_space_info()
    Om.close()
    comm.Barrier()
    
    grid = [refine_array_1d(Vh.breaks[i], 5, False) for i in range(Vh.ldim)]

    npts_per_cell = [6] * len(grid)

    Vh.eval_fields(grid, f, npts_per_cell=npts_per_cell, overlap=1)

    Pm = PostProcessManager(geometry_file=filename, space_file='space_test.yml', fields_file='fields_test.h5', comm=comm)

    for i in range(20):
        Pm.load_snapshot(i, 'ff')
        new_ff = Pm._last_loaded_fields['ff']

        results = new_ff.space.eval_fields(grid, new_ff, npts_per_cell=npts_per_cell, overlap=1)
        assert np.allclose(results[0][0], np.cos(pi/ 10 * i))
        assert np.allclose(results[0][1], np.sin(pi/ 10 * i))

    Pm.export_to_vtk('test', grid=grid, npts_per_cell=npts_per_cell, 
                     snapshots='all', fields={"circle": "ff"}, 
                     additional_logical_functions={'test_l': lambda X,Y : X[:, :, None]},
                     additional_physical_functions={'test_phy': lambda X,Y : X})


if __name__ == "__main__":
    import sys
    test_parallel_export(comm, sys.argv[1])