from psydac.mapping.discrete   import SplineMapping
from psydac.fem.splines        import SplineSpace
from psydac.fem.tensor         import TensorFemSpace
from psydac.ddm.cart           import DomainDecomposition
from psydac.api.discretization import discretize
from psydac.cad.geometry       import Geometry
from psydac.api.postprocessing import OutputManager, PostProcessManager

from sympde.topology.analytical_mapping import IdentityMapping
from sympde.topology                  import Cube, Derham

import numpy as np

from sympy      import exp, lambdify

def build_derham_spline_mapping_id_1d(degree=[2], ncells=[10], periodic = [False]):
    
    p1,  = degree
    nc1, = ncells
    periodic1, = periodic

    # Spaces for the SplineMapping
    V1 = SplineSpace( grid=np.linspace( 0, 1, num=nc1+1), degree=p1, periodic=False )

    domain_decomposition = DomainDecomposition([nc1], [periodic1])
    tensor_space = TensorFemSpace(domain_decomposition, V1)

    # Create the mapping
    map_symbolic = IdentityMapping(name = 'Id', dim = 1)
    map_analytic = map_symbolic.get_callable_mapping()
    map_discrete = SplineMapping.from_mapping(tensor_space, map_analytic)
    map_discrete.set_name("map")

    # Create the de Rham sequence
    domain_h = Geometry.from_discrete_mapping(map_discrete)
    derham   = Derham(domain_h.domain)
    derham_h = discretize(derham, domain_h, degree=degree)

    # Create and project a function for export
    x = domain_h.domain.coordinates
    P0, P1 = derham_h.projectors()

    density = exp(-((x-0.6)**2)/0.1)

    f = lambdify(domain_h.domain.coordinates, density)
    f_h = P0(f)

    # Cannot test export : work only in 2/3D
    # Export the discrete function
    #Om = OutputManager("./export_sol.yml",
    #                   "./export_sol.h5")

    #V0h = derham_h.V0
    #Om.add_spaces(V0h=V0h)
    #Om.export_space_info()
    #Om.add_snapshot(0,0)
    #Om.export_fields(f = f_h)  
    #Om.close()

    #Pm = PostProcessManager(domain=domain_h.domain,
    #                        space_file="./export_sol.yml",
    #                        fields_file="./export_sol.h5")
    #Pm.export_to_vtk("./fields_test",
    #                 grid=None, npts_per_cell=2,
    #                 snapshots='all', fields=('f'))     
    #Pm.close()

def build_derham_spline_mapping_id_2d(degree=[2,2], ncells=[10,10], periodic = [False, False]):
    
    p1 , p2 = degree
    nc1, nc2 = ncells
    periodic1, periodic2 = periodic

    # Spaces for the SplineMapping
    V1 = SplineSpace( grid=np.linspace( 0, 1, num=nc1+1), degree=p1, periodic=False )
    V2 = SplineSpace( grid=np.linspace( 0, 1, num=nc2+1), degree=p2, periodic=False )

    domain_decomposition = DomainDecomposition([nc1, nc2], [periodic1, periodic2])
    tensor_space = TensorFemSpace(domain_decomposition, V1, V2)

    # Create the mapping
    map_symbolic = IdentityMapping(name = 'Id', dim = 2)
    map_analytic = map_symbolic.get_callable_mapping()
    map_discrete = SplineMapping.from_mapping(tensor_space, map_analytic)
    map_discrete.set_name("map")

    # Create the de Rham sequence
    domain_h = Geometry.from_discrete_mapping(map_discrete)
    derham   = Derham(domain_h.domain, sequence = ['H1', 'Hdiv', 'L2'])
    derham_h = discretize(derham, domain_h, degree=degree, get_H1vec_space = True)

    # Create and project a function for export
    x,y = domain_h.domain.coordinates
    P0, P1, P2, PX = derham_h.projectors()

    density = exp(-((x-0.6)**2+(y-0.5)**2)/0.1)

    f = lambdify(domain_h.domain.coordinates, density)
    f_h = P0(f)

    # Export the discrete function
    Om = OutputManager("./export_sol.yml",
                       "./export_sol.h5")

    V0h = derham_h.V0
    Om.add_spaces(V0h=V0h)
    Om.export_space_info()
    Om.add_snapshot(0,0)
    Om.export_fields(f = f_h)  
    Om.close()

    Pm = PostProcessManager(domain=domain_h.domain,
                            space_file="./export_sol.yml",
                            fields_file="./export_sol.h5")
    Pm.export_to_vtk("./fields_test",
                     grid=None, npts_per_cell=2,
                     snapshots='all', fields=('f'))     
    Pm.close()

def build_derham_spline_mapping_id_3d(degree=[2,2,2], ncells=[10,10,3], periodic = [False, False, False]):
    
    p1 , p2, p3  = degree
    nc1, nc2, nc3 = ncells
    periodic1, periodic2, periodic3 = periodic

    # Spaces for the SplineMapping
    V1 = SplineSpace( grid=np.linspace( 0, 1, num=nc1+1), degree=p1, periodic=False )
    V2 = SplineSpace( grid=np.linspace( 0, 1, num=nc2+1), degree=p2, periodic=False )
    V3 = SplineSpace( grid=np.linspace( 0, 1, num=nc3+1), degree=p3, periodic=False )

    domain_decomposition = DomainDecomposition([nc1, nc2, nc3], [periodic1, periodic2, periodic3])
    tensor_space = TensorFemSpace(domain_decomposition, V1, V2, V3)

    # Create the mapping
    map_symbolic = IdentityMapping(name = 'Id', dim = 3)
    map_analytic = map_symbolic.get_callable_mapping()
    map_discrete = SplineMapping.from_mapping(tensor_space, map_analytic)
    map_discrete.set_name("map")

    # Create the de Rham sequence
    domain_h = Geometry.from_discrete_mapping(map_discrete)
    derham   = Derham(domain_h.domain)
    derham_h = discretize(derham, domain_h, degree=degree, get_H1vec_space = True)

    # Create and project a function for export
    x,y,z = domain_h.domain.coordinates
    P0, P1, P2, P3, PX = derham_h.projectors()

    density = exp(-((x-0.6)**2+(y-0.5)**2)/0.1)

    f = lambdify(domain_h.domain.coordinates, density)
    f_h = P0(f)

    # Export the discrete function
    Om = OutputManager("./export_sol.yml",
                       "./export_sol.h5")

    V0h = derham_h.V0
    Om.add_spaces(V0h=V0h)
    Om.export_space_info()
    Om.add_snapshot(0,0)
    Om.export_fields(f = f_h)  
    Om.close()

    Pm = PostProcessManager(domain=domain_h.domain,
                            space_file="./export_sol.yml",
                            fields_file="./export_sol.h5")
    Pm.export_to_vtk("./fields_test",
                     grid=None, npts_per_cell=2,
                     snapshots='all', fields=('f'))     
    Pm.close()

if __name__ == '__main__':
    build_derham_spline_mapping_id_1d()    
    build_derham_spline_mapping_id_2d()    
    build_derham_spline_mapping_id_3d()