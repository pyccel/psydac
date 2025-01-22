import numpy as np
from collections import OrderedDict

from sympde.topology import Derham

from psydac.feec.multipatch.api import discretize
from psydac.feec.multipatch.utils_conga_2d import P0_phys, P1_phys, P2_phys

from psydac.feec.multipatch.operators import HodgeOperator
from psydac.feec.multipatch.plotting_utilities import plot_field
from psydac.feec.multipatch.multipatch_domain_utilities import build_cartesian_multipatch_domain
from psydac.feec.multipatch.non_matching_operators import construct_h1_conforming_projection, construct_hcurl_conforming_projection

# from psydac.api.postprocessing import OutputManager, PostProcessManager


def get_function_H_div(domain):
    from sympy import pi, cos, sin, Tuple, exp, atan, atan2

    x, y = domain.coordinates

    f = Tuple(cos(pi*x)*sin(pi*y), sin(pi*x)*cos(pi*y))
    divf = -2 * np.pi * sin(pi*x)*sin(pi*y) 
    return f, divf

def get_function_H0_div(domain):
    # on a [0,pi] x [0, pi]
    from sympy import pi, cos, sin, Tuple, exp, atan, atan2

    x, y = domain.coordinates

    k = 3
    f = Tuple(sin(k*x)*cos(k*y), cos(k*x)*sin(k*y))
    divf = 2 * k * cos(k*x)*cos(k*y) 
    return f, divf

def get_function_H_curl(domain):
    from sympy import pi, cos, sin, Tuple, exp, atan, atan2

    x, y = domain.coordinates

    f = cos(pi*x)*cos(pi*y)
    curlf = Tuple(np.pi * cos(pi*x)*sin(pi*y),  -np.pi * sin(pi*x)*cos(pi*y)) 
    return f, curlf

def solve_weak_div(kind='div', h0div = True, 
        ncells=np.array([[None, 8], [8, 16]]), deg=4, mom_pres=True, hom_bc=True, hide_plot=True, verbose=True,
        plot_dir="plots/", backend_language='pyccel-gcc'
        ):

    print("degree: " +str(deg))
    degree = [deg, deg]

    if h0div:
        if ncells[0,0] == None:
            ncells[0,0] = ncells[1,1]
        domain = build_cartesian_multipatch_domain(ncells, [0, np.pi], [0, np.pi], mapping='identity')
        if kind=='div':
            f, divf = get_function_H0_div(domain)
        elif kind=='curl':
            print("not implemented")
    else:
        domain = build_cartesian_multipatch_domain(ncells, [1, 3], [0, np.pi / 4], mapping='polar')
        if kind=='div':
            f, divf = get_function_H_div(domain)
        elif kind=='curl':
            f, curlf = get_function_H_curl(domain)

    mappings = OrderedDict([(P.logical_domain, P.mapping)
                           for P in domain.interior])
    mappings_list = list(mappings.values())

    N = np.sum(ncells[ncells != None])
    print("N: " +str(N))

    if isinstance(ncells, int):
        ncells = [ncells, ncells]
    elif ncells.ndim == 1:
        ncells = {patch.name: [ncells[i], ncells[i]]
                    for (i, patch) in enumerate(domain.interior)}
    elif ncells.ndim == 2:
        ncells = {patch.name: [ncells[int(patch.name[2])][int(patch.name[4])], 
                ncells[int(patch.name[2])][int(patch.name[4])]] for patch in domain.interior}    
    
    domain_h = discretize(domain, ncells=ncells)

    derham  = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    Nbasis = V0h.nbasis
    print("Nbasis: " +str(Nbasis))

    if mom_pres:  
        p_moments = deg
    else: 
        p_moments = -1
        
    cP0_m = construct_h1_conforming_projection(V0h, p_moments=p_moments, hom_bc=hom_bc)
    cP1_m = construct_hcurl_conforming_projection(V1h, p_moments=p_moments, hom_bc=hom_bc)

    nquads = [4 * (d + 1) for d in degree]
    P0, P1, P2 = derham_h.projectors(nquads=nquads)

    H0 = HodgeOperator(V0h, domain_h, backend_language=backend_language)
    H1 = HodgeOperator(V1h, domain_h, backend_language=backend_language)
    H2 = HodgeOperator(V2h, domain_h, backend_language=backend_language)

    H0_m  = H0.to_sparse_matrix()                # = mass matrix of V0
    dH0_m = H0.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V0
    H1_m  = H1.to_sparse_matrix()                # = mass matrix of V1
    dH1_m = H1.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V1
    H2_m  = H2.to_sparse_matrix()                # = mass matrix of V2
    dH2_m = H2.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V2

    bD0, bD1 = derham_h.broken_derivatives_as_operators
    bD0_m = bD0.to_sparse_matrix()
    bD1_m = bD1.to_sparse_matrix()

    if kind=='div':
        divh = -cP0_m.transpose() @ bD0_m.transpose()
        wDiv_m = dH0_m @ divh  @ H1_m

        fh = dH1_m @ derham_h.get_dual_dofs(space='V1', f=f, backend_language=backend_language, return_format='numpy_array')
        l2_fh = P1_phys(f, P1, domain, mappings_list).coeffs.toarray()

        divfh  = dH0_m @ derham_h.get_dual_dofs(space='V0', f=divf, backend_language=backend_language, return_format='numpy_array')
        wdivfh = wDiv_m.dot(fh)

    
        err_fh = l2_fh - fh
        rel_err_fh = np.sqrt(np.dot(err_fh, H1_m.dot(err_fh)))/np.sqrt(np.dot(fh,H1_m.dot(fh)))
        print('relative error fh: ' + str(rel_err_fh))

        err_div = divfh - wdivfh
        rel_err_div = np.sqrt(np.dot(err_div, H0_m.dot(err_div)))/np.sqrt(np.dot(divfh,H0_m.dot(divfh)))
        print('relative error divf: ' + str(rel_err_div))

        if verbose:
            title_text = 'with mom_pres = ' + str(mom_pres) + ', hom_bc = ' + str(hom_bc) + ', h0div = ' + str(h0div)
            plot_field(numpy_coeffs= fh, Vh=V1h, space_kind='hcurl', plot_type='components', domain=domain, surface_plot=False, title='f'+title_text, hide_plot=hide_plot, filename=plot_dir+"fh.png")
            #plot_field(numpy_coeffs= l2_fh, Vh=V1h, space_kind='hcurl', plot_type='components', domain=domain, surface_plot=False, title='L2-fh'+title_text, hide_plot=hide_plot, filename=plot_dir+"L2fh.png")

            plot_field(numpy_coeffs= divfh, Vh=V0h, space_kind='h1', plot_type='components', domain=domain, surface_plot=False, title='divf'+title_text, hide_plot=hide_plot, filename=plot_dir+"divfh.png")
            plot_field(numpy_coeffs= wdivfh, Vh=V0h, space_kind='h1', plot_type='components', domain=domain, surface_plot=False, title='wdivf'+title_text, hide_plot=hide_plot, filename=plot_dir+"pwdivfh.png")

        return N, rel_err_div

    elif kind=='curl':
        curlh = -cP1_m.transpose() @ bD1_m.transpose()
        wCurl_m = dH1_m @ curlh  @ H2_m

        fh = dH2_m @ derham_h.get_dual_dofs(space='V2', f=f, backend_language=backend_language, return_format='numpy_array')
        l2_fh = P2_phys(f, P2, domain, mappings_list).coeffs.toarray()

        curlfh  = dH1_m @ derham_h.get_dual_dofs(space='V1', f=curlf, backend_language=backend_language, return_format='numpy_array')
        wcurlfh = wCurl_m.dot(fh)

        err_fh = l2_fh - fh
        rel_err_fh = np.sqrt(np.dot(err_fh, H2_m.dot(err_fh)))/np.sqrt(np.dot(fh,H2_m.dot(fh)))
        print('relative error fh: ' + str(rel_err_fh))

        err_curl = curlfh - wcurlfh
        rel_err_curl = np.sqrt(np.dot(err_curl, H1_m.dot(err_curl)))/np.sqrt(np.dot(curlfh,H1_m.dot(curlfh)))
        print('relative error curlf: ' + str(rel_err_curl))

        if verbose:
            title_text = 'with mom_pres = ' + str(mom_pres) + ', hom_bc = ' + str(hom_bc) + ', h0curl = ' + str(h0div)
            plot_field(numpy_coeffs= fh, Vh=V2h, space_kind='l2', plot_type='components', domain=domain, surface_plot=False, title='f'+title_text, hide_plot=hide_plot, filename=plot_dir+"fh.png")
            #plot_field(numpy_coeffs= l2_fh, Vh=V1h, space_kind='hcurl', plot_type='components', domain=domain, surface_plot=False, title='L2-fh'+title_text, hide_plot=hide_plot, filename=plot_dir+"L2fh.png")

            plot_field(numpy_coeffs= curlfh, Vh=V1h, space_kind='hcurl', plot_type='components', domain=domain, surface_plot=False, title='curlf'+title_text, hide_plot=hide_plot, filename=plot_dir+"curlfh.png")
            plot_field(numpy_coeffs= wcurlfh, Vh=V1h, space_kind='hcurl', plot_type='components', domain=domain, surface_plot=False, title='wcurlf'+title_text, hide_plot=hide_plot, filename=plot_dir+"pwcurlfh.png")

        return N, rel_err_curl

if __name__ == '__main__':
    plot_dir = "plots/WDIV/"
    # solve_weak_div(h0div = False, ncells=np.array([[None, 8], [16, 8]]), deg=3, mom_pres=True, hom_bc=True, hide_plot=True, verbose=False)

    # deg = 4
    # k = 3
    # fac = 6
    # ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
    # N, rel_err_div = solve_weak_div(kind='curl', h0div = False, ncells=ncells, deg=deg, mom_pres=True, hom_bc=True, hide_plot=True, verbose=True)
    # print(N, rel_err_div)

    fac = 6
    err_dict = dict()
    N_dict = dict()
    for deg in range(4,5):
        err_dict[deg] = []
        N_dict[deg] = []
        for k in range(0,4):
            ncells = np.array([[None, fac * 2**k], [fac * 2**k, fac * 2**(k+1)]])
            N, rel_err_div = solve_weak_div(kind='div', h0div = False, ncells=ncells, deg=deg, mom_pres=False, hom_bc=True, hide_plot=True, verbose=False)

            err_dict[deg].append(rel_err_div)
            N_dict[deg].append(N)

        print(err_dict)    
        print(N_dict)
