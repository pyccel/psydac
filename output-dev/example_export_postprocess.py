import numpy as np

from sympde.topology import Domain
from sympde.topology import VectorFunctionSpace, ScalarFunctionSpace

from psydac.api.discretization import discretize
from psydac.fem.basic import FemField
from psydac.api.postprocessing import OutputManager, PostProcessManager


def example(n):
    # =================================================================
    # Part 1: Running a simulation
    # =================================================================
    geometry_file = '../../mesh/bent_pipe.h5'

    domain = Domain.from_file(geometry_file)

    V1 = ScalarFunctionSpace('V1', domain, kind='h1')
    V2 = VectorFunctionSpace('V2', domain, kind='hcurl')

    domainh = discretize(domain, filename=geometry_file)

    V1h = discretize(V1, domainh, degree=[4, 3])
    V2h = discretize(V2, domainh, degree=[[3, 2],[2, 3]])

    uh = FemField(V1h)
    vh = FemField(V2h)

    # Output Manager Initialization
    output = OutputManager('space_example.yml', 'fields_example.h5')
    output.add_spaces(V1h=V1h, V2h=V2h)
    output.set_static().export_fields(u=uh, v=vh)

    uh_grids = []
    vh_grids = []

    for i in range(n):
        uh.coeffs[:] = i

        vh.coeffs[0][:] = np.cos(np.pi * 2 * i / n)
        vh.coeffs[1][:] = np.sin(np.pi * 2 * i / n)

        # Export to HDF5
        output.add_snapshot(t=float(i), ts=i).export_fields(u=uh, v=vh)

        # Saving for comparisons
        uh_grid = V1h.eval_fields(uh, refine_factor=2)
        vh_grid_x, vh_grid_y = V2h.eval_fields(vh, refine_factor=2)
        uh_grids.append(uh_grid)
        vh_grids.append((vh_grid_x,vh_grid_y))

    output.export_space_info()

    return uh_grids, vh_grids
    # End of the simulation
    # ---------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
def post_processing(uh_grids, vh_grids):
    # =================================================================================
    # Part 2: Post Processing
    # =================================================================================
    import pyevtk

    geometry_file = '../../mesh/bent_pipe.h5'

    post = PostProcessManager(geometry_filename=geometry_file,
                              space_filename='space_example.yml',
                              fields_filename='fields_example.h5')

    post.reconstruct_scope()

    domain_h = post.domain
    mapping = list(domain_h.mappings.values())[0]
    x_mesh, y_mesh, z_mesh = mapping.build_mesh()

    V1h_new = post.spaces['V1h']
    V2h_new = post.spaces['V2h']

    static_fields = post.fields['static']['fields']

    for i in range(len(uh_grids)):
        snapshot = post.fields[i]
        u_new = snapshot['fields']['u']
        v_new = snapshot['fields']['v']

        print(f"snapshot{i}, time = {snapshot['time']}, timestep = {snapshot['timestep']}")

        uh_grid_new = V1h_new.eval_fields(u_new, refine_factor=2)
        vh_grid_x_new, vh_grid_y_new = V2h_new.eval_fields(v_new, refine_factor=2)

        assert np.allclose(uh_grid_new, uh_grids[i])
        assert np.allclose(vh_grid_x_new, vh_grids[i][0])
        assert np.allclose(vh_grid_y_new, vh_grids[i][1])

        # Export to VTK for visualization in Paraview
        pyevtk.hl.gridToVTK(f'example_{i:0>4}', x_mesh, y_mesh, z_mesh,
                            pointData={'increment': uh_grid_new,
                                       'rotation': (vh_grid_x_new, vh_grid_y_new, np.zeros_like(vh_grid_y_new))},
                            fieldData={'time': np.array([float(i)]), 'time step': np.array([i])})


if __name__ == '__main__':
    uh_grids, vh_grids = example(20)
    post_processing(uh_grids, vh_grids)
