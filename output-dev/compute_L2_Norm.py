from psydac.fem.basic import FemField


def compute_l2_norm(ncells, degree, uh_1, comm=None, is_logical=False):
    import os 
    import numpy as np
    
    from sympy import sin, pi, cos

    from sympde.calculus import laplace, dot, grad
    from sympde.topology import ScalarFunctionSpace, element_of, LogicalExpr, Union, Domain, Cube
    from sympde.topology.mapping import Mapping
    from sympde.expr     import BilinearForm, LinearForm, Norm
    from sympde.expr     import EssentialBC, find 
    from sympde.expr     import integral

    from psydac.api.discretization import discretize
    from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL
    from psydac.fem.splines        import SplineSpace
    from psydac.fem.tensor         import TensorFemSpace
    from psydac.mapping.discrete import SplineMapping
    from psydac.cad.geometry import Geometry

    os.environ['OMP_NUM_THREADS']    = "2"
    # backend to activate multi threading
    PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP           = PSYDAC_BACKEND_GPYCCEL.copy()
    PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP['openmp'] = True

    # Define the Spherical coordinate system
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
    Omega   = mapping(A)


    ne1, ne2, ne3 = ncells
    p1, p2, p3 = degree
    if not os.path.exists(f'TargetTorusMapping_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}.h5')  and comm.rank == 0:
        print("Start creating Spline Mapping", flush=True)
        # Create SplineMapping

        grid_1 = np.linspace(*A.bounds1, num=ne1 + 1)
        grid_2 = np.linspace(*A.bounds2, num=ne2 + 1)
        grid_3 = np.linspace(*A.bounds3, num=ne3 + 1)

        Spl1 = SplineSpace(p1, grid=grid_1, periodic=False)
        Spl2 = SplineSpace(p2, grid=grid_2, periodic=True)
        Spl3 = SplineSpace(p3, grid=grid_3, periodic=True)

        mapping_tensor_space = TensorFemSpace(Spl1, Spl2, Spl3, comm=None)

        map_discrete = SplineMapping.from_mapping(mapping_tensor_space, mapping)
        geometry = Geometry.from_discrete_mapping(map_discrete)

        geometry.export(f'TargetTorusMapping_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}.h5')

    if comm is not None:
        comm.Barrier()

    if not is_logical:
        del Omega, A,
        Omega = Domain.from_file(f'TargetTorusMapping_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}.h5')
    Omega_logical = Omega.logical_domain
    
    if is_logical:
        V = ScalarFunctionSpace('V', Omega_logical, kind='h1')
        v = element_of(V, name='v')

        x1, x2, x3 = Omega.logical_domain.coordinates
        x, y, z = Omega.mapping.expressions
        u_e = (0.05 - x1) * (0.2 - x1) * sin(x2) * cos(x3)
        l2norm_u_e = Norm(u_e - v, Omega_logical, kind='l2')
        
        u = element_of(V, name='u')
        l2norm_e = Norm(u - u_e, Omega_logical, kind='l2')

    else:
        V = ScalarFunctionSpace('V', Omega, kind='h1')
        v = element_of(V, name ='v')

        x, y, z = Omega.coordinates
        u_e = sin(pi * x) * sin(pi * y) * cos(pi * z)
        l2norm_u_e = Norm(u_e - v, Omega, kind='l2')

        u = element_of(V, name='u')
        l2norm_e = Norm(u - u_e, Omega, kind='l2')
    
    print("Start discretization", flush=True)
    # Create computational domain from topological domain
    Omega_h = discretize(Omega, filename=f'TargetTorusMapping_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}.h5', comm=comm)
    Omega_log_h = discretize(Omega_logical, ncells=ncells, comm=comm)

    # Create discrete spline space
    if is_logical:
        Vh = discretize(V, Omega_log_h, degree=degree, periodic=[False, True, True])

    else:
        Vh = discretize(V, Omega_h, degree=degree, periodic=[False, False, False])
    
    # Discretize norms
    if is_logical:
        l2norm_u_e_h = discretize(l2norm_u_e, Omega_log_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP)
        l2norm_e_h = discretize(l2norm_e, Omega_log_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP)
    else:
        l2norm_u_e_h = discretize(l2norm_u_e, Omega_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP)
        l2norm_e_h = discretize(l2norm_e, Omega_h, Vh, backend=PSYDAC_BACKEND_GPYCCEL_WITH_OPENMP)
    
    uh_true = FemField(Vh)
    uh_true.coeffs[:] = uh_1.coeffs[:]
    
    vh = FemField(Vh)
    
    l2_norm_ue = l2norm_u_e_h.assemble(v=vh)
    l2_norm_e  = l2norm_e_h.assemble(u=uh_true)

    print(l2_norm_e/l2_norm_ue)

if __name__ == '__main__':
    import sys
    import os
    from psydac.api.postprocessing import PostProcessManager
    ncells = sys.argv[1:4]
    degree = sys.argv[4:7]
    is_logical = sys.argv[7] == 0

    p1, p2, p3 = degree
    ne1, ne2, ne3 = ncells
    Pm = PostProcessManager(geometry_file=f'TargetTorusMapping_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}.h5', 
                            space_file=f'spaces_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.yml', 
                            fields_file=f'fields_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.h5')
    Pm.load_static('u')
    print(Pm.fields)
    compute_l2_norm(ncells, degree, Pm.fields['u'])
    a = input("Delete Files ?")
    if a == "0":
        del Pm
        os.remove(f'spaces_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.yml')
        os.remove(f'fields_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}_{is_logical}.h5')
        os.remove(f'TargetTorusMapping_{ne1}_{ne2}_{ne3}_{p1}_{p2}_{p3}.h5')