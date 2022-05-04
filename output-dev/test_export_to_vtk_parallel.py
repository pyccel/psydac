import os
import numpy as np
import h5py as h5

from sympde.topology import ScalarFunctionSpace, Domain, VectorFunctionSpace, Square, Cube, Mapping
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
if size == 1:
    comm = None


def test_parallel_export(comm):

    class TargetTorusMapping(Mapping):
        """
        3D Torus with a polar cross-section like in the TargetMapping.
        """
        _expressions = {'x' : '(R0 + (1-k)*x1*cos(x2) - D*x1**2) * cos(x3)',
                        'y' : '(R0 + (1-k)*x1*cos(x2) - D*x1**2) * sin(x3)',
                        'z' : '(Z0 + (1+k)*x1*sin(x2))'}

        _ldim = 3
        _pdim = 3

    # Define topological domain
    r_in  = 0.05
    r_out = 0.2
    A       = Cube('A', bounds1=(r_in, r_out), bounds2=(0, 2 * np.pi), bounds3=(0, 2* np.pi))
    mapping = TargetTorusMapping('M', 3, R0=1.0, Z0=0, k=0.3, D=0.2)
    domain  = mapping(A)

    domainh = discretize(domain, ncells=[10, 10, 10], comm=comm)

    V = ScalarFunctionSpace('V', domain, kind='l2')
    Vv = VectorFunctionSpace('Vv', domain, kind = 'hcurl')
    Vh = discretize(V, domainh, degree=[2, 2, 2])
    Vvh = discretize(Vv, domainh, degree=[2, 2, 2])

    f = FemField(Vh)
    ff = FemField(Vvh)

    Om = OutputManager('space_test.yml', 'fields_test.h5', comm=comm, mode='w')

    Om.add_spaces(Vvh=Vvh, Vh=Vh)
    Om.export_space_info()
    Om.set_static()
    f.coeffs[:] = 15
    Om.export_fields(f=f)
    pi = np.pi
    for i in range(1):
        f.coeffs[:] = i
        ff.coeffs[0][:] = np.cos(pi/ 10 * i)
        ff.coeffs[1][:] = np.sin(pi/ 10 * i)
        Om.add_snapshot(t=float(i), ts=i)
        Om.export_fields(ff=ff, f_t=f)
    Om.close()
    
    grid = [refine_array_1d(Vh.breaks[i], 2, False) for i in range(Vh.ldim)]
    npts_per_cell = [3] * len(grid)

    Pm = PostProcessManager(domain=domain, space_file='space_test.yml', fields_file='fields_test.h5', comm=comm)

    Pm.export_to_vtk('test_torus', grid=grid, npts_per_cell=npts_per_cell,
                    snapshots='all', fields={'i': "f"},)
    

if __name__ == "__main__":
    test_parallel_export(comm)