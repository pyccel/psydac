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
from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.examples.ppc_test_cases     import get_source_and_solution_hcurl, get_div_free_pulse, get_curl_free_pulse, get_Delta_phi_pulse, get_Gaussian_beam#, get_praxial_Gaussian_beam_E, get_easy_Gaussian_beam_E, get_easy_Gaussian_beam_B,get_easy_Gaussian_beam_E_2, get_easy_Gaussian_beam_B_2
from psydac.feec.multipatch.utils_conga_2d              import DiagGrid, P0_phys, P1_phys, P2_phys, get_Vh_diags_for
from psydac.feec.multipatch.utilities                   import time_count #, export_sol, import_sol
from psydac.linalg.utilities                            import array_to_psydac
from psydac.fem.basic                                   import FemField
from psydac.feec.multipatch.non_matching_operators import construct_hcurl_conforming_projection, construct_h1_conforming_projection
from psydac.feec.multipatch.multipatch_domain_utilities import build_cartesian_multipatch_domain

from sympde.calculus      import grad, dot, curl, cross
from sympde.topology      import NormalVector
from sympde.expr.expr     import BilinearForm
from sympde.topology      import elements_of
from sympde import Tuple

from psydac.api.postprocessing import OutputManager, PostProcessManager
from sympy.functions.special.error_functions import erf

def run_sim():
    ## Minimal example for a PML implementation of the Time-Domain Maxwells equation
    nc = 20
    # ncells  = np.array([[nc, nc, nc],
    #                     [nc, 2*nc, nc], 
    #                     [nc, nc, nc]])

    ncells  = np.array([[nc, nc, nc],
                        [2*nc, 2*nc,  2*nc], 
                        [nc, nc, nc],
                        [nc, nc, nc]])


    degree = [3,3]
    p_moments= -1
    GSP = False
    plot_dir = "plots/PML2/pml_diag_wave_no_moments_corner"
    bc = 'pml' #'none', 'abc' #'pml'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    x_lim = np.pi
    y_lim = np.pi
    final_time = 3

    domain = build_cartesian_multipatch_domain(ncells, [0, x_lim], [0, y_lim])
    ncells_h = {patch.name: [ncells[int(patch.name[4])][int(patch.name[2])], ncells[int(patch.name[4])][int(patch.name[2])]] for patch in domain.interior}
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

    backend     = 'pyccel-gcc'

    H0 = HodgeOperator(V0h, domain_h)
    H1 = HodgeOperator(V1h, domain_h)
    H2 = HodgeOperator(V2h, domain_h)

    H0_m  = H0.to_sparse_matrix()              
    dH0_m = H0.get_dual_Hodge_sparse_matrix()  
    H1_m  = H1.to_sparse_matrix()              
    dH1_m = H1.get_dual_Hodge_sparse_matrix()  
    H2_m = H2.to_sparse_matrix()              
    dH2_m = H2.get_dual_Hodge_sparse_matrix()  
    cP0_m = construct_h1_conforming_projection(V0h, p_moments=p_moments, hom_bc=False)
    cP1_m = construct_hcurl_conforming_projection(V1h, p_moments=p_moments, hom_bc=False)

    ## PML
    u, v     = elements_of(derham.V1, names='u, v')
    x,y = domain.coordinates

    u1 = dot(Tuple(1,0),u)
    u2 = dot(Tuple(0,1),u)
    v1 = dot(Tuple(1,0),v)
    v2 = dot(Tuple(0,1),v)

    def heaviside(x_direction, xmin, xmax, delta, sign, domain):
        x,y = domain.coordinates

        if sign == -1:
            d = xmax - delta    
        else:
            d = xmin + delta

        if x_direction == True:
            return 1/2*(erf(-sign*(x-d) *1000)+1)
        else:
            return 1/2*(erf(-sign*(y-d) *1000)+1)

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
        return sigma_m * heaviside(x, xmin, xmax, delta, sign, domain) * parabola(x, xmin, xmax, delta, sign, domain)

    def sigma_fun_sym(x, xmin, xmax, delta, sigma_m, domain):
        return sigma_fun(x, xmin, xmax, delta, 1, sigma_m, domain) + sigma_fun(x, xmin, xmax, delta, -1, sigma_m, domain)

    delta = np.pi/6
    xmin = 0
    xmax = x_lim
    ymin = 0
    ymax = y_lim
    sigma_0 = 30

    sigma_x = sigma_fun_sym(True, xmin, xmax, delta, sigma_0, domain)
    sigma_y = sigma_fun_sym(False, ymin, ymax, delta, sigma_0, domain)
    if bc == 'pml':
        mass = BilinearForm((v,u), integral(domain, u1*v1*sigma_y + u2*v2*sigma_x))
        massh = discretize(mass, domain_h, [V1h, V1h])
        M = massh.assemble().tosparse()

        u, v     = elements_of(derham.V2, names='u, v')
        mass = BilinearForm((v,u), integral(domain, u*v*(sigma_y + sigma_x)))
        massh = discretize(mass, domain_h, [V2h, V2h])
        M2 = massh.assemble().tosparse()

    elif bc == 'abc':
        ### Silvermueller ABC
        
        u, v     = elements_of(derham.V1, names='u, v')
        nn       = NormalVector('nn')
        boundary = domain.boundary
        expr_b = cross(nn, u)*cross(nn, v)

        a = BilinearForm((u,v), integral(boundary, expr_b))
        ah = discretize(a, domain_h, [V1h, V1h], backend=PSYDAC_BACKENDS[backend],)
        A_eps = ah.assemble().tosparse()
        ###


    # conf_proj = GSP
    # seems to be bad for non-matching stuff!
    if GSP:
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

    eps=0.1
    E0, B0 = get_Gaussian_beam(x_0=0.4 , y_0=0.4, domain=domain)
    #E0, B0 = get_Gaussian_beam(x_0=np.pi * 1/2 , y_0=np.pi * 1/2, domain=domain)
    #E0, B0 = get_Berenger_wave(x_0=3.14/2 , y_0=3.14/2, domain=domain)
    backend_language = 'pyccel-gcc'
    E_c = dH1_m @ derham_h.get_dual_dofs(space='V1', f=E0, backend_language=backend_language, return_format='numpy_array')
    B_c = dH2_m @ derham_h.get_dual_dofs(space='V2', f=B0, backend_language=backend_language, return_format='numpy_array')

    # E0_h = P1_phys(E0, P1, domain, mappings_list)
    # E_c = E0_h.coeffs.toarray()

    # B0_h = P2_phys(B0, P2, domain, mappings_list)
    # B_c = B0_h.coeffs.toarray()

    #plot_field(numpy_coeffs=E_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, plot_type='amplitude', filename="E_amp_before")

    # plot_field(numpy_coeffs=E_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, plot_type='components', filename="E_comp_before")
    # plot_field(numpy_coeffs=B_c, Vh=V2h, space_kind='l2', domain=domain, filename="B_before")


    # E_c_ = dC_m @ B_c
    # B_c[:] = 0
    # plot_field(numpy_coeffs=E_c_, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, plot_type='components', filename="E_comp_after")
    # plot_field(numpy_coeffs=B_c, Vh=V2h, space_kind='l2', domain=domain, filename="B_after")
    
    # E_c_ = E_c
    #B_c = C_m @ E_c
    # plot_field(numpy_coeffs=E_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, plot_type='components', filename="E_comp_after_after")
    #plot_field(numpy_coeffs=B_c, Vh=V2h, space_kind='l2', domain=domain, filename="B_after_after")
    #B_c[:] = 0
    
    
    #exit()

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
    if bc == 'pml':
        Epml = sp.sparse.linalg.spsolve(H1_m, M)
        Bpml = sp.sparse.linalg.spsolve(H2_m, M2)
    elif bc == 'abc':
        H1A = H1_m + dt * A_eps
        A_eps = sp.sparse.linalg.spsolve(H1A, H1_m)
        dC_m =  sp.sparse.linalg.spsolve(H1A, C_m.transpose() @ H2_m)
    elif bc == 'none':
        A_eps = sp.sparse.linalg.spsolve(H1_m, H1_m)

    f_c = np.copy(f0_c)
    for nt in range(Nt):
        print(' .. nt+1 = {}/{}'.format(nt+1, Nt))

        # 1/2 faraday: Bn -> Bn+1/2
        if bc == 'pml':
            B_c[:] -= dt/2*Bpml@B_c + (dt/2) * C_m @ E_c
            E_c[:] += -dt*Epml @ E_c  + dt * (dC_m @ B_c - f_c)
            B_c[:] -= dt/2*Bpml@B_c + (dt/2) * C_m @ E_c

        else:
            B_c[:] -=  (dt/2) * C_m @ E_c
            E_c[:] = A_eps @ E_c + dt * (dC_m @ B_c - f_c)
            B_c[:] -= (dt/2) * C_m @ E_c

        #plot_field(numpy_coeffs=cP1_m @ E_c, Vh=V1h, space_kind='hcurl', domain=domain, surface_plot=False, plot_type='amplitude', filename=plot_dir+"/E_{}".format(nt))

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
    PM.export_to_vtk(plot_dir+"/Eh",grid=None, npts_per_cell=4,snapshots='all', fields = 'Eh' )
    PM.close()

    PM = PostProcessManager(domain=domain, space_file=plot_dir+'/spaces2.yml', fields_file=plot_dir+'/fields2.h5' )
    PM.export_to_vtk(plot_dir+"/Bh",grid=None, npts_per_cell=4,snapshots='all', fields = 'Bh' )
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