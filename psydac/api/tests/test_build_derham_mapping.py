#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from psydac.mapping.discrete   import SplineMapping
from psydac.fem.splines        import SplineSpace
from psydac.fem.tensor         import TensorFemSpace
from psydac.ddm.cart           import DomainDecomposition
from psydac.api.discretization import discretize
from psydac.cad.geometry       import Geometry
from psydac.api.postprocessing import OutputManager, PostProcessManager

from sympde.topology.analytical_mapping import IdentityMapping
from sympde.topology                  import Cube, Derham

from sympy      import exp, lambdify

import pytest
import numpy as np
import os


@pytest.mark.parametrize('degree', [[2], [3]])
@pytest.mark.parametrize('ncells', [[4], [10]])
@pytest.mark.parametrize('periodic', [[True], [False]])

def test_build_derham_spline_mapping_id_1d(degree, ncells, periodic):
    
    p1,  = degree
    nc1, = ncells
    periodic1, = periodic

    # Spaces for the SplineMapping
    V1 = SplineSpace( grid=np.linspace( 0, 1, num=nc1+1), degree=p1, periodic=periodic1 )

    domain_decomposition = DomainDecomposition([nc1], [periodic1])
    tensor_space = TensorFemSpace(domain_decomposition, V1)

    # Create the mapping
    map_symbolic = IdentityMapping(name = 'Id', dim = 1)
    map_analytic = map_symbolic.get_callable_mapping()
    map_discrete = SplineMapping.from_mapping(tensor_space, map_analytic)
    map_discrete.set_name("map")

    # Create the de Rham sequence
    name = '_' + str(nc1) + '_' + str(p1)
    domain_h = Geometry.from_discrete_mapping(map_discrete, name = name)
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

@pytest.mark.parametrize('degree', [[2,2], [3,4]])
@pytest.mark.parametrize('ncells', [[5,6], [10,10]])
@pytest.mark.parametrize('periodic', [[True,True], [True,False]])

def test_build_derham_spline_mapping_id_2d(degree, ncells, periodic):

    p1 , p2 = degree
    nc1, nc2 = ncells
    periodic1, periodic2 = periodic

    # Spaces for the SplineMapping
    V1 = SplineSpace( grid=np.linspace( 0, 1, num=nc1+1), degree=p1, periodic=periodic1 )
    V2 = SplineSpace( grid=np.linspace( 0, 1, num=nc2+1), degree=p2, periodic=periodic2 )

    domain_decomposition = DomainDecomposition([nc1, nc2], [periodic1, periodic2])
    tensor_space = TensorFemSpace(domain_decomposition, V1, V2)

    # Create the mapping
    map_symbolic = IdentityMapping(name = 'Id', dim = 2)
    map_analytic = map_symbolic.get_callable_mapping()
    map_discrete = SplineMapping.from_mapping(tensor_space, map_analytic)

    # Create the de Rham sequence
    name = '_' + str(nc1) + '_' + str(nc2) + '_' + str(p1) + '_' + str(p2)
    domain_h = Geometry.from_discrete_mapping(map_discrete, name = name)
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

    #cleanup 
    os.remove("./export_sol.yml")
    os.remove("./export_sol.h5")
    os.remove("./fields_test.0000.vtu")

@pytest.mark.parametrize('degree', [[2,2,2], [2,3,4]])
@pytest.mark.parametrize('ncells', [[4,6,7], [10,10,10]])
@pytest.mark.parametrize('periodic', [[True,True,True], [True,False,False]])

def test_build_derham_spline_mapping_id_3d(degree, ncells, periodic):

    p1 , p2, p3  = degree
    nc1, nc2, nc3 = ncells
    periodic1, periodic2, periodic3 = periodic

    # Spaces for the SplineMapping
    V1 = SplineSpace( grid=np.linspace( 0, 1, num=nc1+1), degree=p1, periodic=periodic1 )
    V2 = SplineSpace( grid=np.linspace( 0, 1, num=nc2+1), degree=p2, periodic=periodic2 )
    V3 = SplineSpace( grid=np.linspace( 0, 1, num=nc3+1), degree=p3, periodic=periodic3 )

    domain_decomposition = DomainDecomposition([nc1, nc2, nc3], [periodic1, periodic2, periodic3])
    tensor_space = TensorFemSpace(domain_decomposition, V1, V2, V3)

    # Create the mapping
    map_symbolic = IdentityMapping(name = 'Id', dim = 3)
    map_analytic = map_symbolic.get_callable_mapping()
    map_discrete = SplineMapping.from_mapping(tensor_space, map_analytic)
    map_discrete.set_name("map")

    # Create the de Rham sequence
    name = '_' + str(nc1) + '_' + str(nc2) + '_' + str(nc3) + '_' + str(p1) + '_' + str(p2) + '_' + str(p3)
    domain_h = Geometry.from_discrete_mapping(map_discrete, name = name)
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

    #cleanup 
    os.remove("./export_sol.yml")
    os.remove("./export_sol.h5")
    os.remove("./fields_test.0000.vtu")

if __name__ == '__main__':
    #test_build_derham_spline_mapping_id_1d([2], [10], [True]) 
    test_build_derham_spline_mapping_id_2d([2,2], [5,10], [True,True])  
    test_build_derham_spline_mapping_id_2d([2,2], [10,10], [True,True])    
    #test_build_derham_spline_mapping_id_3d([2,3,4], [10,11,3], [True,False,False])
