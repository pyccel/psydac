from pytest import param
from mpi4py import MPI

import os
import numpy as np
import scipy as sp
from collections import OrderedDict
import matplotlib.pyplot as plt

from sympy import lambdify, Matrix

from scipy.sparse.linalg import spsolve
from scipy import special

from sympde.calculus  import dot
from sympde.topology  import element_of
from sympde.expr.expr import LinearForm
from sympde.expr.expr import integral, Norm
from sympde.topology  import Derham

from psydac.api.settings   import PSYDAC_BACKENDS
from psydac.feec.pull_push import pull_2d_hcurl

from psydac.feec.multipatch.api                         import discretize
from psydac.feec.multipatch.fem_linear_operators        import IdLinearOperator
from psydac.feec.multipatch.operators                   import HodgeOperator, get_K0_and_K0_inv, get_K1_and_K1_inv
from psydac.feec.multipatch.plotting_utilities          import plot_field #, write_field_to_diag_grid, 
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain, create_domain
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution_hcurl, get_div_free_pulse, get_curl_free_pulse, get_Delta_phi_pulse, get_Gaussian_beam, get_diag_Gaussian_beam#, get_praxial_Gaussian_beam_E, get_easy_Gaussian_beam_E, get_easy_Gaussian_beam_B,get_easy_Gaussian_beam_E_2, get_easy_Gaussian_beam_B_2
from psydac.feec.multipatch.utils_conga_2d              import DiagGrid, P0_phys, P1_phys, P2_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities                   import time_count #, export_sol, import_sol
from psydac.linalg.utilities                            import array_to_psydac
from psydac.fem.basic                                   import FemField
from psydac.feec.multipatch.non_matching_operators import construct_vector_conforming_projection, construct_scalar_conforming_projection
from psydac.feec.multipatch.non_matching_multipatch_domain_utilities import create_square_domain

from sympde.calculus      import grad, dot, curl, cross
from sympde.topology      import NormalVector
from sympde.expr.expr     import BilinearForm
from sympde.topology      import elements_of
from sympde import Tuple
from sympde.topology import Square, Domain
from sympde.topology import IdentityMapping, PolarMapping, AffineMapping, Mapping #TransposedPolarMapping

from psydac.api.postprocessing import OutputManager, PostProcessManager
from sympy.functions.special.error_functions import erf

from psydac.feec.multipatch.non_matching_operators import get_moment_pres_scalar_extension_restriction
from psydac.fem.splines import SplineSpace

def run_sim():
    ## Minimal example for a PML implementation of the Time-Domain Maxwells equation
    ncells  = [8, 16]
    degree = [3,3]
    plot_dir = "plots/PML/interface_diffusion"
    final_time = 5

    OmegaLog1 = Square('OmegaLog1',bounds1=(0., 2*np.pi), bounds2=(0., np.pi))
    mapping_1 = IdentityMapping('M1',2)
    domain_1     = mapping_1(OmegaLog1)

    OmegaLog2 = Square('OmegaLog2',bounds1=(0., 2*np.pi), bounds2=(np.pi, 2*np.pi))
    mapping_2 = IdentityMapping('M2',2)
    domain_2     = mapping_2(OmegaLog2)

    patches = [domain_1, domain_2]

    interfaces = [
        [domain_1.get_boundary(axis=1, ext=+1), domain_2.get_boundary(axis=1, ext=-1),1]
    ]
    domain = create_domain(patches, interfaces, name='domain')

    ncells_h = {patch.name: [2 * ncells[i], ncells[i]] for (i,patch) in enumerate(domain.interior)}
    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    domain_h = discretize(domain, ncells=ncells_h)
    derham_h = discretize(derham, domain_h, degree=degree)

    nquads = [4*(d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)


    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    I1 = IdLinearOperator(V1h)
    I1_m = I1.to_sparse_matrix()

    I2 = IdLinearOperator(V2h)
    I2_m = I2.to_sparse_matrix()

    I0 = IdLinearOperator(V0h)
    I0_m = I0.to_sparse_matrix()

    backend     = 'pyccel-gcc'

    H0 = HodgeOperator(V0h, domain_h)
    H1 = HodgeOperator(V1h, domain_h)
    H2 = HodgeOperator(V2h, domain_h)

    dH0_m  = H0.to_sparse_matrix()              
    H0_m = H0.get_dual_Hodge_sparse_matrix()  
    dH1_m  = H1.to_sparse_matrix()              
    H1_m = H1.get_dual_Hodge_sparse_matrix()  
    dH2_m = H2.to_sparse_matrix()              
    H2_m = H2.get_dual_Hodge_sparse_matrix()  
    cP0_m = construct_scalar_conforming_projection(V0h, [0,0], [-1,-1], nquads=None, hom_bc=[False,False])
    cP1_m = construct_vector_conforming_projection(V1h, [0,0], [-1,-1], nquads=None, hom_bc=[False,False])

    def patch_extension_restriction(coarse_space, fine_space):
        # coarse patch, fine patch -> Matrix: fine -> coarse
        E_xy = []
        R_xy = []
        for k in range(2):
            cs_k = coarse_space.spaces[k]
            knots = [ (k - cs_k.knots[0])/(cs_k.knots[-1] - cs_k.knots[0]) for k in cs_k.knots]
            css_k = SplineSpace(cs_k.degree, knots = knots, basis=cs_k.basis)

            fs_k = fine_space.spaces[k]
            knots = [ (k - fs_k.knots[0])/(fs_k.knots[-1] - fs_k.knots[0]) for k in fs_k.knots]
            fss_k = SplineSpace(fs_k.degree, knots=knots, basis=fs_k.basis)
            
            matching = (css_k.ncells == fss_k.ncells)
            E_k, R_k, ER_k = get_moment_pres_scalar_extension_restriction(matching, css_k, fss_k, css_k.basis)
            E_xy.append(E_k)
            R_xy.append(R_k)

        E = np.kron(E_xy[0].toarray(), E_xy[1].toarray())
        if matching:
            R = np.kron(R_xy[0].toarray(), R_xy[1].toarray())
        else:
            R = np.kron(R_xy[0], R_xy[1])

        return E, R

    def global_matrices(V0h, V1h, V2h):

        E, R = patch_extension_restriction(V0h.spaces[0], V0h.spaces[1])
        n0 = V0h.spaces[0].nbasis
        n1 = V0h.spaces[1].nbasis
        R0_global = np.block([[np.eye(n0), np.zeros((n0, n1))], 
                [np.zeros((n1, n0)), E@R]])
        I0_global = np.eye(n0+n0)

        # first component
       # E, R = patch_extension_restriction(V1h.spaces[0].spaces[0], V1h.spaces[1].spaces[0])
        n = V1h.spaces[0].nbasis
        #10 = V1h.spaces[1].spaces[0].nbasis
        R00_global = np.eye(n) #np.block([[np.eye(n00), np.zeros((n00, n10))], 
                #[np.zeros((n10, n00)), E@R]])

        #second component
        E, R = patch_extension_restriction(V1h.spaces[0].spaces[0], V1h.spaces[1].spaces[0])
        R0 = E@R
        E, R = patch_extension_restriction(V1h.spaces[0].spaces[1], V1h.spaces[1].spaces[1])
        R1 = E@R

        n01 = V1h.spaces[1].spaces[0].nbasis
        n11 = V1h.spaces[1].spaces[1].nbasis
        
        R11_global = np.block([[R0, np.zeros((n01, n11))], 
                [np.zeros((n11, n01)), R1]])
        
        m11 = n11 + n01
        R1_global = np.block([[R00_global, np.zeros((n, m11))],
                                    [np.zeros((m11, n)), R11_global]])
        I1_global = np.eye(n+n01+m11)

        E, R = patch_extension_restriction(V2h.spaces[0], V2h.spaces[1])
        n0 = V2h.spaces[0].nbasis
        n1 = V2h.spaces[1].nbasis
        R2_global = np.block([[np.eye(n0), np.zeros((n0, n1))], 
                [np.zeros((n1, n0)), E@R]])
        I2_global = np.eye(n0+n0)

        return R0_global, R1_global, R2_global

    R0_global, R1_global, R2_global = global_matrices(V0h, V1h, V2h)
    
    
    ## boundary PML
    u, v     = elements_of(derham.V1, names='u, v')
    x,y = domain.coordinates

    u1 = dot(Tuple(1,0),u)
    u2 = dot(Tuple(0,1),u)
    v1 = dot(Tuple(1,0),v)
    v2 = dot(Tuple(0,1),v)

    def heaviside(x_direction, xmin, xmax, delta, sign, domain, fact):
        x,y = domain.coordinates

        if sign == -1:
            d = xmax - delta    
        else:
            d = xmin + delta

        if x_direction == True:
            return 1/2*(erf(-sign*(x-d) *fact)+1)
        else:
            return 1/2*(erf(-sign*(y-d) *fact)+1)

    def parabola(x_direction, xmin, xmax, delta, sign, domain):
        x,y = domain.coordinates

        if sign == -1:
            d = xmax - delta    
        else:
            d = xmin + delta

        if x_direction == True:
            return ((x - d)/delta)**2
        else:
            return ((y - d)/delta)**2

    def sigma_fun(x, xmin, xmax, delta, sign, sigma_m, domain):
        return sigma_m * heaviside(x, xmin, xmax, delta, sign, domain, 1000) * parabola(x, xmin, xmax, delta, sign, domain)

    def sigma_fun_sym(x, xmin, xmax, delta, sigma_m, domain):
        return sigma_fun(x, xmin, xmax, delta, 1, sigma_m, domain) + sigma_fun(x, xmin, xmax, delta, -1, sigma_m, domain)

    delta = np.pi/6
    xmin = 0
    xmax = 2*np.pi
    ymin = 0
    ymax = 2*np.pi 
    sigma_0 = 20

    sigma_x = sigma_fun_sym(True, xmin, xmax, delta, sigma_0, domain)
    sigma_y = sigma_fun_sym(False, ymin, ymax, delta, sigma_0, domain)
    
    mass = BilinearForm((v,u), integral(domain, u1*v1*sigma_y + u2*v2*sigma_x))
    massh = discretize(mass, domain_h, [V1h, V1h])
    M = massh.assemble().tosparse()

    u, v     = elements_of(derham.V2, names='u, v')
    mass = BilinearForm((v,u), integral(domain, u*v*(sigma_y + sigma_x)))
    massh = discretize(mass, domain_h, [V2h, V2h])
    M2 = massh.assemble().tosparse()

    # interface PML at y = pi

    u, v     = elements_of(derham.V1, names='u, v')
    x,y = domain.coordinates

    u1 = dot(Tuple(1,0),u)
    u2 = dot(Tuple(0,1),u)
    v1 = dot(Tuple(1,0),v)
    v2 = dot(Tuple(0,1),v)

    delta = np.pi/6
    ycenter = np.pi + 3/2*delta
    
    sigma_0 = 0.5
    
    sigma_x = 0#sigma_fun_sym(True, xmin, xmax, delta, sigma_0, domain)
    sigma_y = sigma_0 * heaviside(False, ycenter-delta, ycenter, delta, -1, domain, 10) * heaviside(False, ycenter, ycenter+delta, delta, 1, domain, 10)
    
    mass = BilinearForm((v,u), integral(domain, u1*v1*sigma_y + u2*v2*sigma_x))
    massh = discretize(mass, domain_h, [V1h, V1h])
    M_int = massh.assemble().tosparse()

    u, v     = elements_of(derham.V2, names='u, v')
    mass = BilinearForm((v,u), integral(domain, u*v*(sigma_y + sigma_x)))
    massh = discretize(mass, domain_h, [V2h, V2h])
    M2_int = massh.assemble().tosparse()

    # conf_proj = GSP
    K0, K0_inv = get_K0_and_K0_inv(V0h, uniform_patches=False)
    cP0_m = K0_inv @ cP0_m @ K0
    K1, K1_inv = get_K1_and_K1_inv(V1h, uniform_patches=False)
    cP1_m = K1_inv @ cP1_m @ K1

    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()


    dH1_m = dH1_m.tocsr()
    H2_m = H2_m.tocsr()
    cP1_m = cP1_m.tocsr()
    bD1_m = bD1_m.tocsr()

    C_m = bD1_m @ cP1_m
    dC_m = dH1_m @ C_m.transpose() @ H2_m


    div_m = dH0_m @ cP0_m.transpose() @ bD0_m.transpose() @ H1_m

    jump_penal_m = I1_m - cP1_m
    JP_m = jump_penal_m.transpose() * H1_m * jump_penal_m

    f0_c = np.zeros(V1h.nbasis)


    #E0, B0 = get_Gaussian_beam(x_0=3.14 , y_0=0.5*3.14, domain=domain)
    E0, B0 = get_Gaussian_beam(x_0= 0.5 , y_0= 0.5, domain=domain)
    E0_h = P1_phys(E0, P1, domain, mappings_list)
    E_c = E0_h.coeffs.toarray()
    
    B0_h = P2_phys(B0, P2, domain, mappings_list)
    B_c = B0_h.coeffs.toarray()

    #E_c = dC_m @ B_c
    #B_c[:] = 0

    ###
    phi = np.zeros(H0_m.toarray().shape[0])
    #phi[20] = 1
    phi[342] = 1


    E_c[:] = 0
    #E_c[20] = 1
    E_c[2*169 + 92] = 1

    B_c[:] = 0
    B_c[19] = 1

    plot_field(numpy_coeffs= phi, Vh=V0h, space_kind='h1', domain=domain , filename=plot_dir+"/phi0_{}".format(0))
    plot_field(numpy_coeffs= cP0_m @ phi, Vh=V0h, space_kind='h1', domain=domain , filename=plot_dir+"/phi1_{}".format(0))
    plot_field(numpy_coeffs= R0_global @ phi, Vh=V0h, space_kind='h1', domain=domain , filename=plot_dir+"/phi2_{}".format(0))
    plot_field(numpy_coeffs= R0_global@ cP0_m @ phi, Vh=V0h, space_kind='h1', domain=domain , filename=plot_dir+"/phi3_{}".format(0))
    plot_field(numpy_coeffs= (I0_m.toarray() - R0_global)@ cP0_m @ phi, Vh=V0h, space_kind='h1', domain=domain , filename=plot_dir+"/phi4_{}".format(0))


    plot_field(numpy_coeffs=cP1_m @ E_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, plot_type='components', filename=plot_dir+"/E_{}".format(0))

    plot_field(numpy_coeffs=B_c, Vh=V2h, space_kind='l2', domain=domain, filename=plot_dir+"/B_{}".format(0))

    ###
    exit()

    OM1 = OutputManager(plot_dir+'/spaces1.yml', plot_dir+'/fields1.h5')
    OM1.add_spaces(V1h=V1h)
    OM1.export_space_info()

    OM2 = OutputManager(plot_dir+'/spaces2.yml', plot_dir+'/fields2.h5')
    OM2.add_spaces(V2h=V2h)
    OM2.export_space_info()

    stencil_coeffs_E = array_to_psydac(cP1_m @ E_c, V1h.vector_space)
    Eh = FemField(V1h, coeffs=stencil_coeffs_E)
    OM1.add_snapshot(t=0 , ts=0) 
    OM1.export_fields(Eh=Eh)

    stencil_coeffs_B = array_to_psydac(B_c, V2h.vector_space)
    Bh = FemField(V2h, coeffs=stencil_coeffs_B)
    OM2.add_snapshot(t=0 , ts=0) 
    OM2.export_fields(Bh=Bh)

    dt = compute_stable_dt(C_m=C_m, dC_m=dC_m, cfl_max=0.8, dt_max=None)
    Nt = int(np.ceil(final_time/dt))
    dt = final_time / Nt
    Epml = sp.sparse.linalg.spsolve(H1_m, M)
    Bpml = sp.sparse.linalg.spsolve(H2_m, M2)

    Epml_int = sp.sparse.linalg.spsolve(H1_m, (I1_m - R1_global).transpose()@M_int@(I1_m - R1_global))
    Bpml_int = sp.sparse.linalg.spsolve(H2_m, (I2_m - R2_global).transpose()@M2_int@(I2_m - R2_global))

    #Epml_int = sp.sparse.linalg.spsolve(H1_m, (I1_m - R1_global).transpose()@C_m.transpose()@M2_int@C_m@(I1_m - R1_global))


    f_c = np.copy(f0_c)
    for nt in range(Nt):
        print(' .. nt+1 = {}/{}'.format(nt+1, Nt))

        # 1/2 faraday: Bn -> Bn+1/2
        B_c[:] -= dt/2*(Bpml @ B_c +  Bpml_int@B_c) + (dt/2) * C_m @ E_c

        E_c[:] += -dt*(Epml @ E_c + Epml_int @ E_c)  + dt * (dC_m @ B_c - f_c)
        #E_c[:] = A_eps @ E_c + dt * (dC_m @ B_c - f_c)

        B_c[:] -= dt/2*(Bpml @ B_c +  Bpml_int@B_c) + (dt/2) * C_m @ E_c


        stencil_coeffs_E = array_to_psydac(cP1_m @ E_c, V1h.vector_space)
        Eh = FemField(V1h, coeffs=stencil_coeffs_E)
        OM1.add_snapshot(t=nt*dt, ts=nt) 
        OM1.export_fields(Eh = Eh)

        stencil_coeffs_B = array_to_psydac(B_c, V2h.vector_space)
        Bh = FemField(V2h, coeffs=stencil_coeffs_B)
        OM2.add_snapshot(t=nt*dt, ts=nt) 
        OM2.export_fields(Bh=Bh)

    OM1.close()

    print("Do some PP")
    PM = PostProcessManager(domain=domain, space_file=plot_dir+'/spaces1.yml', fields_file=plot_dir+'/fields1.h5' )
    PM.export_to_vtk(plot_dir+"/Eh",grid=None, npts_per_cell=6,snapshots='all', fields = 'Eh' )
    PM.close()

    PM = PostProcessManager(domain=domain, space_file=plot_dir+'/spaces2.yml', fields_file=plot_dir+'/fields2.h5' )
    PM.export_to_vtk(plot_dir+"/Bh",grid=None, npts_per_cell=6,snapshots='all', fields = 'Bh' )
    PM.close()


#def compute_stable_dt(cfl_max, dt_max, C_m, dC_m, V1_dim):
def compute_stable_dt(*, C_m, dC_m, cfl_max, dt_max=None):
    """
    Compute a stable time step size based on the maximum CFL parameter in the
    domain. To this end we estimate the operator norm of

    `dC_m @ C_m: V1h -> V1h`,

    find the largest stable time step compatible with Strang splitting, and
    rescale it by the provided `cfl_max`. Setting `cfl_max = 1` would run the
    scheme exactly at its stability limit, which is not safe because of the
    unavoidable round-off errors. Hence we require `0 < cfl_max < 1`.

    Optionally the user can provide a maximum time step size in order to
    properly resolve some time scales of interest (e.g. a time-dependent
    current source).

    Parameters
    ----------
    C_m : scipy.sparse.spmatrix
        Matrix of the Curl operator.

    dC_m : scipy.sparse.spmatrix
        Matrix of the dual Curl operator.

    cfl_max : float
        Maximum Courant parameter in the domain, intended as a stability
        parameter (=1 at the stability limit). Must be `0 < cfl_max < 1`.

    dt_max : float, optional
        If not None, restrict the computed dt by this value in order to
        properly resolve time scales of interest. Must be > 0.

    Returns
    -------
    dt : float
        Largest stable dt which satisfies the provided constraints.

    """

    print (" .. compute_stable_dt by estimating the operator norm of ")
    print (" ..     dC_m @ C_m: V1h -> V1h ")
    print (" ..     with dim(V1h) = {}      ...".format(C_m.shape[1]))

    if not (0 < cfl_max < 1):
        print(' ******  ****** ******  ****** ******  ****** ')
        print('         WARNING !!!  cfl = {}  '.format(cfl))
        print(' ******  ****** ******  ****** ******  ****** ')

    def vect_norm_2 (vv):
        return np.sqrt(np.dot(vv,vv))

    t_stamp = time_count()
    vv = np.random.random(C_m.shape[1])
    norm_vv = vect_norm_2(vv)    
    max_ncfl = 500
    ncfl = 0
    spectral_rho = 1
    conv = False
    CC_m = dC_m @ C_m

    while not( conv or ncfl > max_ncfl ):

        vv[:] = (1./norm_vv)*vv
        ncfl += 1
        vv[:] = CC_m.dot(vv)
        
        norm_vv = vect_norm_2(vv)
        old_spectral_rho = spectral_rho
        spectral_rho = vect_norm_2(vv) # approximation
        conv = abs((spectral_rho - old_spectral_rho)/spectral_rho) < 0.001
        print ("    ... spectral radius iteration: spectral_rho( dC_m @ C_m ) ~= {}".format(spectral_rho))
    t_stamp = time_count(t_stamp)
    
    norm_op = np.sqrt(spectral_rho)
    c_dt_max = 2./norm_op    
    
    light_c = 1
    dt = cfl_max * c_dt_max / light_c

    if dt_max is not None:
        dt = min(dt, dt_max)

    print( "  Time step dt computed for Maxwell solver:")
    print(f"     Based on cfl_max = {cfl_max} and dt_max = {dt_max}, we set dt = {dt}")
    print(f"     -- note that c*Dt = {light_c*dt} and c_dt_max = {c_dt_max}, thus c * dt / c_dt_max = {light_c*dt/c_dt_max}")
    print(f"     -- and spectral_radius((c*dt)**2* dC_m @ C_m ) = {(light_c * dt * norm_op)**2} (should be < 4).")

    return dt


if __name__ == '__main__':
    run_sim()