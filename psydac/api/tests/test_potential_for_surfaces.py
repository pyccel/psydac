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
from sympde.topology.domain import Cube, Domain
from sympde.topology import Derham

from psydac.cad.geometry     import Geometry
from psydac.feec.global_projectors import Projector_H1, Projector_Hcurl
from psydac.feec.derivatives import Gradient_3D
from psydac.fem.basic import FemField
from psydac.fem.tensor import TensorFemSpace
from psydac.api.discretization import discretize
from psydac.api.feec import DiscreteDerham
from psydac.api.fem import DiscreteLinearForm, DiscreteBilinearForm
from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockLinearOperator

from psydac.api.tests.potential_for_surfaces import find_potential, CylinderMapping, ErrorFunctional

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

    l2_error = compute_l2_error(domain, domain_h, derham, derham_h, grad_f_h, alpha_h, beta_h)
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






def compare_with_dommaschk_potential():
    """ 
    Compare the new computed potential with the Dommaschk potential
    
    Plot the difference in degrees between the magnetic field and the Dommaschk
    potential and the new potential respectively to compare them.
    """
    # Arrange
    log_domain = Cube(name="log_domain", bounds1=(0.75,1.25), 
                    bounds2=(0,0.5*np.pi), bounds3=(-0.25,0.25))
    cylinder_mapping = CylinderMapping(name="cylinder_mapping", rmin=0.5, rmax=, c1=0, c2=0)
    domain = cylinder_mapping(log_domain)
    domain_h : Geometry = discretize(domain, ncells=ncells, 
                                    periodic=[False, True, False])
    derham = Derham(domain, ['H1', 'Hcurl', 'Hdiv', 'L2'])
    degree = [3, 3, 3]
    derham_h : DiscreteDerham = discretize(derham, domain_h, degree=degree)
    dommaschk_lambda = 

    projector_h1 = Projector_H1(derham_h.V0)
    dommaschk_h = projector_h1(dommaschk_lamdbda)
    beta0 = dommaschk_h.copy()
    alpha0 = lambda x1, x2, x3: 1.0
    alpha0_h = 
    B_h = 

    # Act
    alpha, beta = find_potential(alpha0, beta0, B_h, derham_h, derham, domain, domain_h)

    # Assert
    coordinates = 
    Bpot = alpha(coordinates)*beta.gradient(coordinates)
    bxb_new = ( np.linalg.norm(
                    np.cross(Bvec/np.linalg.norm(Bvec,axis=0),
                             Bpot/np.linalg.norm(Bpot,axis=0),axis=0)
                    ,axis=0
                )
    )
    print(("min |b x b|=  %e , max |b x b|= %e")%(np.amin(bxb_new),np.amax(bxb_new)))
    n_planes=nzeta//skip_zeta

    fig = plt.figure(figsize=(13, np.ceil(n_planes/2) * 6.5))
    fig.suptitle('angle between gvec and dommaschk normalized B-field, $sin^{-1}| b_g  x b_d|$ , in degrees [°]')
    for n in np.arange(0,n_planes):
        ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
        map = ax.contourf(R_g[:,:,n], Z_g[:,:,n], np.arcsin(bxb_new[:,:,n])*180/np.pi, 30)
        ax.set_title('at $\\phi$={0:4.3f} $*2\\pi /(nfp)$ '.format(phi_g[0,0,n]/(2*np.pi)*nfp))
        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        ax.axis('equal')
    
    fig.colorbar(map, ax=ax, location='right')

if __name__ == '__main__':
    logging.basicConfig(filename="mylog.log", filemode='w', 
                        level=logging.DEBUG,
                        format='%(name)s\n\t%(message)s')
    test_z_as_scalar_field_with_slightly_perturbed_initial_guess()

