import logging
import numpy as np
from sympy import ImmutableDenseMatrix
import matplotlib.pyplot as plt

from sympde.calculus import grad, dot

from sympde.expr     import Norm
from sympde.expr.expr          import LinearForm, BilinearForm
from sympde.expr.expr          import integral              
from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology.analytical_mapping import Mapping, PolarMapping, TorusMapping
from sympde.topology.mapping import BasicCallableMapping
from sympde.topology.domain import Cube, Domain
from sympde.topology import Derham

from psydac.cad.geometry     import Geometry
from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl
from psydac.feec.derivatives import Gradient_3D
from psydac.feec.pull_push import push_3d_hcurl
from psydac.fem.basic import FemField
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace
from psydac.mapping.discrete import SplineMapping
from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.fem import DiscreteLinearForm, DiscreteBilinearForm
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockLinearOperator

from psydac.api.tests.potential_for_surfaces import find_potential, CylinderMapping, ErrorFunctional

from struphy.geometry.domains import GVECunit
from struphy.fields_background.mhd_equil.equils import GVECequilibrium
from struphy.fields_background.mhd_equil.base import AxisymmMHDequilibrium

from gvec_to_python import GVEC
from gvec_to_python.reader.gvec_reader import create_GVEC_json
from gvec_to_python.geometry.domain import GVEC_domain

from psydac.api.tests.field import DommaschkPotentials


def arrange_from_scalar_potential(f, mapping : Mapping, log_domain : Domain):
    logger = logging.getLogger(name="arrange_from_scalar_potential")
    # log_domain = Cube(name="log_domain", bounds1=(0.1,1), 
    #                 bounds2=(0,2*np.pi), bounds3=(0, np.pi/2))

    domain = mapping(log_domain)
    ncells = [6,6,6]
    domain_h : Geometry = discretize(domain, ncells=ncells, 
                                    periodic=[False, True, False])
    
    derham = Derham(domain, sequence=['H1', 'Hcurl', 'Hdiv', 'L2'])
    degree = [3, 3, 3]
    derham_h : DiscreteDerham = discretize(derham, domain_h, degree=degree)
    projector_h1 = Projector_H1(derham_h.V0)
    f_h = projector_h1(f)
    return derham_h, derham, projector_h1, f_h, domain, domain_h



def test_z_as_scalar_field_with_correct_initial_guess():
    # TODO: Remove duplication in domain and the other stuff
    logger = logging.getLogger(name='test_z_as_scalar_field_with_correct_initial_guess')
    f = lambda x1, x2, x3: x3
    log_domain = Cube(name="log_domain", bounds1=(0.1,1), 
                bounds2=(0,2*np.pi), bounds3=(0, np.pi/2))
    torus_mapping = TorusMapping(name="torus_mapping", R0=2.0)
    derham_h, derham, projector_h1, f_h,  domain, domain_h = arrange_from_scalar_potential(f, 
                                                                                           mapping=torus_mapping,
                                                                                           log_domain=log_domain
    )
    gradient_3D = Gradient_3D(derham_h.V0, derham_h.V1)
    grad_f_h = gradient_3D(f_h)
    alpha0 = lambda x1, x2, x3: 1.0
    alpha0_h = projector_h1(alpha0)
    beta0_h = f_h.copy()
    logger.debug("beta0_h._space:%s", beta0_h.coeffs._space)

    alpha_h, beta_h = find_potential(alpha0_h, beta0_h, grad_f_h, derham_h, derham, domain, domain_h)
    logger.debug("beta_h._space:%s", beta_h.coeffs._space)

    error_functional = ErrorFunctional(domain, domain_h, derham, derham_h, B_h=grad_f_h)
    l2_error = error_functional(alpha_h, beta_h)
    logger.debug("l2_error:%s", l2_error)
    assert l2_error < 1e-3, f"l2_error is {l2_error}"

def test_z_as_scalar_field_with_slightly_perturbed_initial_guess():
    logger = logging.getLogger(name='test_z_as_scalar_field_with_radial_coordinate_as_initial_guess')
    beta0 = lambda x1, x2, x3: 0.01*x1 + x3
    domain, domain_h, derham, derham_h, grad_f_h, alpha0_h, beta0_h = test_z_as_scalar_field_arrange(beta0)
    # initial_l2_error = compute_l2_error(domain, domain_h, derham, derham_h, grad_f_h, alpha0_h, beta0_h)
    alpha_h, beta_h = find_potential(alpha0_h, beta0_h, grad_f_h, derham_h, derham, domain, domain_h)
    error_functional = ErrorFunctional(domain, domain_h, derham, derham_h, grad_f_h)
    error_functional_val = error_functional(alpha_h, beta_h)
    assert error_functional_val < 1e-3, f"l2_error is {error_functional_val}"

def test_z_as_scalar_field_with_radial_coordinate_as_initial_guess():


    logger = logging.getLogger(name='test_z_as_scalar_field_with_radial_coordinate_as_initial_guess')
    beta0 = lambda x1, x2, x3: x1
    domain, domain_h, derham, derham_h, grad_f_h, alpha0_h, beta0_h = test_z_as_scalar_field_arrange(beta0)
    # initial_l2_error = compute_l2_error(domain, domain_h, derham, derham_h, grad_f_h, alpha0_h, beta0_h)
    alpha_h, beta_h = find_potential(alpha0_h, beta0_h, grad_f_h, derham_h, derham, domain, domain_h)
    error_functional = ErrorFunctional(domain, domain_h, derham, derham_h, grad_f_h)
    error_functional_val = error_functional(alpha_h, beta_h)
    assert error_functional_val < 1e-3, f"l2_error is {error_functional_val}"



def test_z_as_scalar_field_arrange(beta0):
    logger = logging.getLogger(name='test_z_as_scalar_field_with_radial_coordinate_as_initial_guess')
    f = lambda x1, x2, x3: x3
    log_domain = Cube(name="log_domain", bounds1=(0.1,1), 
            bounds2=(0,2*np.pi), bounds3=(0, np.pi/2))
    torus_mapping = TorusMapping(name="torus_mapping", R0=2.0)
    derham_h, derham, projector_h1, f_h,  domain, domain_h = arrange_from_scalar_potential(f, 
                                                                                           mapping=torus_mapping,
                                                                                           log_domain=log_domain
    )

    gradient_3D = Gradient_3D(derham_h.V0, derham_h.V1)
    grad_f_h = gradient_3D(f_h)
    alpha0 = lambda x1, x2, x3: 1.0
    alpha0_h = projector_h1(alpha0)
    # beta0 = lambda x1, x2, x3: 0.01*x1 + x3
    beta0_h = projector_h1(beta0)
    return domain, domain_h, derham, derham_h, grad_f_h, alpha0_h, beta0_h



def test_ErrorFunctional():
    # Arrange
    logger = logging.getLogger(name="test_compute_l2_error")
    f = lambda x1, x2, x3: 2/3*x1**3 + x3
    log_domain = Cube(name="log_domain", bounds1=(0.1,1), 
                    bounds2=(0,2*np.pi), bounds3=(0, 1))
    cylinder_mapping = CylinderMapping(name="cylinder_mapping", rmin=0.0, rmax=1.0, c1=0.0, c2=0.0)
    assert isinstance(cylinder_mapping, CylinderMapping)
    derham_h, derham, projector_h1, f_h,  domain, domain_h = arrange_from_scalar_potential(f, 
                                                                                           mapping=cylinder_mapping, 
                                                                                           log_domain=log_domain)
    gradient_3D = Gradient_3D(derham_h.V0, derham_h.V1)
    grad_f_h = gradient_3D(f_h)
    alpha = lambda x1, x2, x3: x1
    alpha_h = projector_h1(alpha)
    beta = lambda x1, x2, x3: x1**2
    beta_h = projector_h1(beta)

    error_functional = ErrorFunctional(domain, domain_h, derham, derham_h, grad_f_h)

    # Act
    error_functional_val = error_functional(alpha_h, beta_h)

    # Assert
    assert np.abs(error_functional_val - (np.pi - 0.01*np.pi)) < 1e-5, f"l2_error:{error_functional_val}"

def test_ErrorFunctional_on_non_gradient():
    """
    Test the ErrorFunctional.__eval__ method on a vector field B which
    is not defined as a gradient.

    We take just a cube as domain
    """
    # Arrange

    domain = Cube()
    domain_h = discretize(domain, ncells=[8,8,4], periodic=[False, False, False])
    derham = Derham(domain, sequence=['H1', 'Hcurl', 'Hdiv', 'L2'])
    derham_h = discretize(derham, domain_h, degree=[2,2,2])
    alpha = lambda x,y,z: x
    beta = lambda x,y,z: y*z
    h1_projector = Projector_H1(derham_h.V0)
    alpha_h = h1_projector(alpha)
    beta_h = h1_projector(beta)
    B1 = lambda x,y,z: x
    B2 = lambda x,y,z: 2*y
    B3 = lambda x,y,z: 3*z

    hcurl_projector = Projector_Hcurl(derham_h.V1)
    B_h = hcurl_projector((B1,B2,B3))

    # Act
    error_functional = ErrorFunctional(domain, domain_h, derham, derham_h, B_h)
    error = error_functional(alpha_h, beta_h)

    # Assert
    assert error == -1, f"error: {error}" # I do not know the error. I just want to know if this runs

def test_ErrorFunctional_with_gvec_mapping():
    # Arrange
    cube = Cube(name='cube', bounds3=(0,0.25))
    cube_discrete = discretize(cube, ncells=[8,8,4], periodic=[False, True, False])
    json_file = 'gvec_run/W7A_GVEC_RUN_State_0001_00040000.json'
    create_GVEC_json(dat_file_in='gvec_run/W7A_GVEC_RUN_State_0001_00040000.dat',
                     json_file_out=json_file
    )

    V0_unit = ScalarFunctionSpace(name='V0_unit', domain=cube, kind='H1')

    V0h_unit = discretize(V0_unit, cube_discrete, degree=[3,3,3])
    # Vh = discretize(V, domain_h, degree=degree)
    gvec = GVEC(json_file=json_file, mapping='unit')
    gvec_mapping = GVECMapping(gvec)
    f_unit_spline = SplineMapping.from_mapping(tensor_space=V0h_unit, mapping=gvec_mapping)
    
    f_unit_mapping = Mapping(name='f_unit_mapping', dim=3)
    f_unit_mapping.set_callable_mapping(f_unit_spline)
    domain = f_unit_mapping(cube)

    
    # # domain_h = Geometry.from_discrete_mapping(f_unit_spline)
    domain_h= discretize(domain, ncells=[8,8,4], periodic = [False, True, False])
    assert isinstance(domain_h, Geometry)
    #                                 periodic=[False, True, False])

    # log_domain = Cube(name="log_domain", bounds1=(0.01,1), 
    #                 bounds2=(0,1), bounds3=(0, np.pi/4))
    # domain_h : Geometry = discretize(domain, ncells=[16,16,4], 
    #                                 periodic=[False, True, False])
    derham = Derham(domain, ['H1', 'Hcurl', 'Hdiv', 'L2'])
    degree = [2, 2, 2]
    derham_h : DiscreteDerham = discretize(derham, domain_h, degree=degree)
    
    projector_hcurl = Projector_Hcurl(derham_h.V1)
    logger.debug('Project the magnetic field')
    B1_R = lambda s, a1, a2: gvec.b1(s, a1, a2)[0]
    B1_th = lambda s, a1, a2: gvec.b1(s, a1, a2)[1]
    B1_ze = lambda s, a1, a2: gvec.b1(s, a1, a2)[2]
    B_h = projector_hcurl((B1_R, B1_th, B1_ze))

    alpha = lambda x,y,z: 0.0
    beta = lambda x,y,z: 0.0
    h1_projector = Projector_H1(derham_h.V0)
    alpha_h = h1_projector(alpha)
    beta_h = h1_projector(beta)

    # Act
    error_functional = ErrorFunctional(domain, domain_h, derham, derham_h, B_h)
    error = error_functional(alpha_h, beta_h)

    # Assert
    assert error == -1, f"error:{error}" # I do not know the error. I just want to know if this runs



class GVECMapping(BasicCallableMapping):
    def __init__(self, gvec: GVEC) -> None:
        super().__init__()
        self._gvec = gvec
    
    def __call__(self, *eta):
        """ Evaluate mapping at location eta. """
        return self._gvec.domain.f_unit(eta[0], eta[1], eta[2])


    def jacobian(self, *eta):
        """ Compute Jacobian matrix at location eta. """
        raise NotImplementedError

    def jacobian_inv(self, *eta):
        """ Compute inverse Jacobian matrix at location eta.
            An exception should be raised if the matrix is singular.
        """
        raise NotImplementedError

    def metric(self, *eta):
        """ Compute components of metric tensor at location eta. """
        raise NotImplementedError

    def metric_det(self, *eta):
        """ Compute determinant of metric tensor at location eta. """
        raise NotImplementedError

    @property
    def ldim(self):
        """ Number of logical/parametric dimensions in mapping
            (= number of eta components).
        """
        return 3
    
    @property
    def pdim(self):
        """ Number of physical dimensions in mapping
            (= number of x components).
        """
        return 3


def compare_with_dommaschk_potential():
    """ 
    Compare the new computed potential with the Dommaschk potential
    
    Plot the difference in degrees between the magnetic field and the Dommaschk
    potential and the new potential respectively to compare them.
    """
    logger = logging.getLogger(name='compare_with_dommaschk_potential')
    logger.debug('Start to compare with Dommaschk')
    # gvec_equil = GVECequilibrium()
    # gvec_unit = GVECunit(gvec_equil)

    # control_points_x = gvec_unit.params_map['cx']
    # control_points_y = gvec_unit.params_map['cy']
    # control_points_z = gvec_unit.params_map['cz']


    # Arrange
    cube = Cube(name='cube', bounds3=(0,0.25))
    cube_discrete = discretize(cube, ncells=[8,8,4], periodic=[False, True, False])
    json_file = 'gvec_run/W7A_GVEC_RUN_State_0001_00040000.json'
    create_GVEC_json(dat_file_in='gvec_run/W7A_GVEC_RUN_State_0001_00040000.dat',
                     json_file_out=json_file
    )

    V0_unit = ScalarFunctionSpace(name='V0_unit', domain=cube, kind='H1')

    V0h_unit = discretize(V0_unit, cube_discrete, degree=[3,3,3])
    # Vh = discretize(V, domain_h, degree=degree)
    gvec = GVEC(json_file=json_file, mapping='unit')
    gvec_mapping = GVECMapping(gvec)
    f_unit_spline = SplineMapping.from_mapping(V0h_unit, gvec_mapping)
    
    f_unit_mapping = Mapping(name='f_unit_mapping', dim=3)
    f_unit_mapping.set_callable_mapping(f_unit_spline)
    domain = f_unit_mapping(cube)

    
    # domain_h = Geometry.from_discrete_mapping(f_unit_spline)
    domain_h= discretize(domain, ncells=[8,8,4], periodic = [False, True, False])
    assert isinstance(domain_h, Geometry)
    #                                 periodic=[False, True, False])

    # log_domain = Cube(name="log_domain", bounds1=(0.01,1), 
    #                 bounds2=(0,1), bounds3=(0, np.pi/4))
    # domain_h : Geometry = discretize(domain, ncells=[16,16,4], 
    #                                 periodic=[False, True, False])
    derham = Derham(domain, ['H1', 'Hcurl', 'Hdiv', 'L2'])
    degree = [2, 2, 2]
    derham_h : DiscreteDerham = discretize(derham, domain_h, degree=degree)
    
    projector_hcurl = Projector_Hcurl(derham_h.V1)
    logger.debug('Project the magnetic field')
    B1_R = lambda s, a1, a2: gvec.b1(s, a1, a2)[0]
    B1_th = lambda s, a1, a2: gvec.b1(s, a1, a2)[1]
    B1_ze = lambda s, a1, a2: gvec.b1(s, a1, a2)[2]
    B_h = projector_hcurl((B1_R, B1_th, B1_ze))

    logger.debug('Compute the Dommaschk potential')
    dcoef= np.load("domm_l2.npy")

    R0_domm = 2.0
    F0_domm = 1.3857888749654599
    B0_domm=F0_domm/R0_domm

    domm = DommaschkPotentials(dcoef,R_0=R0_domm,B_0=B0_domm)
    nfp=5

    def dommaschk_in_gvec_coord(x1, x2, x3):
        R_in, Z_in, phi = AxisymmMHDequilibrium.inverse_map(*gvec.domain.f_unit(x1, x2, x3))
        return domm.Pfunc(R_in, Z_in, phi)


    projector_h1 = Projector_H1(derham_h.V0)
    dommaschk_h = projector_h1(dommaschk_in_gvec_coord)
    beta0 = dommaschk_h.copy()
    alpha0 = lambda x1, x2, x3: 1.0
    alpha0_h : FemField = projector_h1(alpha0)
    assert alpha0_h.space ==  derham_h.V0
    assert beta0.space == derham_h.V0
    assert B_h.space == derham_h.V1
    assert isinstance(B_h.space, VectorFemSpace)

    assert derham_h.callable_mapping == f_unit_spline
    assert derham_h.mapping == f_unit_mapping
    assert derham_h.dim == 3

    assert derham.domain == domain

    assert domain.mapping == f_unit_mapping
    assert domain.mapping.get_callable_mapping() == f_unit_spline
    logger.debug("domain.mapping:%s",domain.mapping)

    assert domain_h.ldim == 3
    assert domain_h.pdim == 3
    assert domain_h.domain == domain
    logger.debug("domain_h.ncells:%s", domain_h.ncells)
    logger.debug("domain_h.mappings:%s", domain_h.mappings)

    # Act
    alpha, beta = find_potential(alpha0_h, beta0, B_h, derham_h, derham, domain, domain_h)

    # Assert

    # R,phi,Z mesh
    n_planes=4
    n_subphi=10
    Rrange=[0.75,1.25]
    Zrange=[-0.25,0.25]

    R1d=R0_domm*np.linspace(Rrange[0],Rrange[1],40)
    Z1d=R0_domm*np.linspace(Zrange[0],Zrange[1],50)
    phi1d=np.linspace(0.,2*np.pi/nfp,n_planes*n_subphi,endpoint=False)
    R,phi,Z=np.meshgrid(R1d,phi1d,Z1d, indexing="ij")

    BR=domm.Bxfunc(R,Z,phi)
    BZ=domm.Bzfunc(R,Z,phi)
    Bphi=domm.Byfunc(R,Z,phi)
    S=domm.Sfunc(R,Z,phi)
    P=domm.Pfunc(R,Z,phi)

    fig = plt.figure(figsize=(13, np.ceil(n_planes/2) * 6.5))

    for n in np.arange(0,n_planes*n_subphi,n_subphi):
        ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n//n_subphi + 1)
        map = ax.contour(R[:,n,:], Z[:,n,:], S[:,n,:], 30)
        #map = ax.contour(R[:,n,:], Z[:,n,:], (BR[:,n,:]**2+BZ[:,n,:]**2)**0.5, 30)
        ax.quiver(R[:,n,:], Z[:,n,:], BR[:,n,:],BZ[:,n,:], scale=5)
        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        ax.axis('equal')
        ax.set_title('Poloidal Bfield at $\\phi$={0:4.3f} $*2\pi /(nfp)$'.format(phi1d[n]/(2*np.pi)*nfp))

    fig.colorbar(map, ax=ax, location='right')

    # print("R,Z of axis gvec:",R_g[0,0,0],Z_g[0,0,0])
    # fig = plt.figure(figsize=(15, 4))


    # ax = fig.add_subplot(1, 3, 1)
    # ax.plot(s_profile,pres_profile)
    # ax.set_title("pressure")
    # ax = fig.add_subplot(1, 3, 2)
    # ax.plot(s_profile,Itor_profile)
    # ax.set_title("Total toroidal current \n(loop integral of oint(B_theta dtheta) at s)\n")
    # ax = fig.add_subplot(1, 3, 3)
    # ax.plot(s_profile,Ipol_profile)
    # ax.set_title("Total poloidal current \n(loop integral of oint(B_zeta dzeta) at s)\n")

    R1d = np.linspace(0.0, 1.0, 50)
    theta1d = np.linspace(0.0, 1.0, 50)
    R, theta = np.meshgrid(R1d, theta1d)


    Bpot_log = alpha(R, theta, 0.0)*beta.gradient(R, theta, 0.0)
    J_inv = f_unit_spline.inv_jac_mat_grid([R1d, theta1d, 0.0])
    Bpot = np.dot(J_inv, f_unit_spline)

    Bvec = gvec.b_cart(R, theta, 0.0)

    bxb_new = ( np.linalg.norm(
                    np.cross(Bvec/np.linalg.norm(Bvec,axis=0),
                             Bpot/np.linalg.norm(Bpot,axis=0),axis=0)
                    ,axis=0
                )
    )
    print(("min |b x b|=  %e , max |b x b|= %e")%(np.amin(bxb_new),np.amax(bxb_new)))
    # n_planes=nzeta//skip_zeta

    # fig = plt.figure(figsize=(13, np.ceil(n_planes/2) * 6.5))
    # fig.suptitle('angle between gvec and dommaschk normalized B-field, $sin^{-1}| b_g  x b_d|$ , in degrees [°]')
    # for n in np.arange(0,n_planes):
    #     ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
    #     map = ax.contourf(R_g[:,:,n], Z_g[:,:,n], np.arcsin(bxb_new[:,:,n])*180/np.pi, 30)
    #     ax.set_title('at $\\phi$={0:4.3f} $*2\\pi /(nfp)$ '.format(phi_g[0,0,n]/(2*np.pi)*nfp))
    #     ax.set_xlabel('R')
    #     ax.set_ylabel('Z')
    #     ax.axis('equal')
    
    # fig.colorbar(map, ax=ax, location='right')

if __name__ == '__main__':
    logging.basicConfig(filename="mylog.log", filemode='w', 
                        level=logging.DEBUG,
                        format='%(name)s\n\t%(message)s',
                        force=True)
    logger = logging.getLogger(name='Test')
    logger.debug("Hello?")
    test_ErrorFunctional_with_gvec_mapping()

