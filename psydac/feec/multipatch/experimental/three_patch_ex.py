import numpy as np

from sympde.topology import Derham, Mapping, Square, Domain, IdentityMapping
from psydac.api.postprocessing import OutputManager, PostProcessManager
from psydac.feec.multipatch.api import discretize
from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic import FemField
from sympy import pi, cos, sin, Tuple, exp, atan, atan2, Tuple

from psydac.feec.multipatch.non_matching_operators import construct_h1_conforming_projection, construct_hcurl_conforming_projection
from psydac.feec.multipatch.operators import HodgeOperator

class QuadliteralMapping(Mapping):
    """
        2D quadliteral mapping in the plane (x1, x2):
    """

    _expressions = {'x': 'a1*x1 + b1*x2 + c1*x1*x2 + d1',
                    'y': 'a2*x1 + b2*x2 + c2*x1*x2 + d2'}

    _ldim        = 2
    _pdim        = 2


def get_trapez(corners, name='no_name'):
    # Assume logical domain is the unit square.
    # corners (numpy.ndarray) are counter-clockwise from bottom left corner

    d = corners[0]
    a = corners[1] - d
    b = corners[3] - d
    c = corners[2] - (a + b) - d

    print(a, b, c, d)

    return QuadliteralMapping(
        name, 2,
        a1=a[0], a2=a[1],
        b1=b[0], b2=b[1],
        c1=c[0], c2=c[1],
        d1=d[0], d2=d[1],
    )

def three_patch_domain():
    A = Square('P0', bounds1=(0, 1), bounds2=(0, 1))
    B = Square('P1', bounds1=(0.5, 1), bounds2=(0, 0.5))
    C = Square('P2', bounds1=(0, 1), bounds2=(0, 1))

    M1 = get_trapez(name='M0', corners=np.array([(0, 0), (0.5, 0), (0.5, 0.5), (0, 1)]))
    M2 = IdentityMapping('M1', 2)
    M3 = get_trapez(name='M2', corners=np.array([ (0, 1), (0.5, 0.5), (1, 0.5), (1, 1)]))

    A = M1(A)
    B = M2(B)
    C = M3(C)
    return Domain.join(patches=[A, B, C],
                             connectivity=[((0, 0,  1), (1, 0, -1), 1),
                                          ((2, 1, -1), (0, 1, 1), 1),
                                            ((1, 1, 1), (2, 0, 1), 1),],
                             name='domain')
def quad_domain():
    A = Square('P0', bounds1=(0, 1), bounds2=(0, 1))
    B = Square('P1', bounds1=(0.5, 1), bounds2=(0, 0.5))
    C = Square('P2', bounds1=(0, 1), bounds2=(0, 1))

    M1 = get_trapez(name='M0', corners=np.array([(0, 0), (0.5, 0), (0.5, 0.5), (0, 1)]))
    M2 = IdentityMapping('M1', 2)
    M3 = get_trapez(name='M2', corners=np.array([ (0, 1), (0.5, 0.5), (1, 0.5), (1, 1)]))

    A = M1(A)
    B = M2(B)
    C = M3(C)
    domain = Domain.join(patches=[A, B, C],
                             connectivity=[((0, 0,  1), (1, 0, -1), 1),
                                          ((2, 1, -1), (0, 1, 1), 1),
                                            ((1, 1, 1), (2, 0, 1), 1),],
                             name='domain')

    ncells = [8, 4, 4]
    ncells_h = {patch.name: [ncells[i], ncells[i]]
                    for (i, patch) in enumerate(domain.interior)}
    degree = [3,3]
    
    domain_h = discretize(domain, ncells=ncells_h)
    derham = Derham(domain, ["H1", "Hcurl", "L2"])
    derham_h = discretize(derham, domain_h, degree=degree)
    nquads = [d+1 for d in degree]
    geomP0, geomP1, geomP2 = derham_h.projectors(nquads = nquads)
    V0h, V1h, V2h = derham_h.spaces
    from collections import OrderedDict

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = [m.get_callable_mapping() for m in mappings.values()]
    

    H0 = HodgeOperator(V0h, domain_h)
    H1 = HodgeOperator(V1h, domain_h)
    H0_m = H0.to_sparse_matrix()            # = mass matrix of V0
    dH0_m = H0.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V0
    H1_m = H1.to_sparse_matrix()            # = mass matrix of V1
    dH1_m = H1.get_dual_Hodge_sparse_matrix()  # = inverse mass matrix of V1
    
    #cP0_m = construct_h1_conforming_projection(V0h, hom_bc=True)
    cP1_m = construct_hcurl_conforming_projection(V1h, hom_bc=False)

    x,y = domain.coordinates
    f = Tuple(sin(pi*x)*sin(2*pi*y), 2*sin(2*pi*x)*sin(pi*y))

    f_cc = derham_h.get_dual_dofs(space='V1', f=f, return_format='numpy_array')
    f_c = dH1_m.dot(f_cc)
    cPf_c = cP1_m @ dH1_m.dot(f_cc)

    from psydac.feec.pull_push import pull_2d_h1, pull_2d_hcurl, pull_2d_l2
    from psydac.feec.multipatch.plotting_utilities          import plot_field  
    G = [[lambda xi1, xi2, ii=i : ii for d in [0,1]] for i in range(len(domain))]
    G_log = [pull_2d_hcurl([G[k][0], G[k][1]], m) for (k,m) in enumerate(mappings_list)]
    cPG1h = cP1_m @geomP1(G_log).coeffs.toarray()
    G1h =   geomP1(G_log).coeffs.toarray()

    plot_dir = "./plots"

    OM = OutputManager(plot_dir + '/spaces.yml', plot_dir + '/fields.h5')
    
    OM.add_spaces(V1h=V1h)
    stencil_coeffs = array_to_psydac(f_c, V1h.vector_space)
    cPstencil_coeffs = array_to_psydac(cPf_c, V1h.vector_space)
    vh = FemField(V1h, coeffs=cPstencil_coeffs)

    stencil_coeffs = array_to_psydac(G1h, V1h.vector_space)
    cPstencil_coeffs = array_to_psydac(cPG1h, V1h.vector_space)

    Gh = FemField(V1h, coeffs=cPstencil_coeffs)
   # cPGh = FemField(V1h, coeffs=cPstencil_coeffs)

    OM.set_static()
    OM.export_fields(vh=vh)
    OM.export_fields(Gh=Gh)

    OM.export_space_info()
    OM.close()
    PM = PostProcessManager(
        domain=domain,
        space_file=plot_dir +
        '/spaces.yml',
        fields_file=plot_dir +
        '/fields.h5')


    PM.export_to_vtk(
        plot_dir +
        "/v_h",
        grid=None,
        npts_per_cell=[6] *
        2,
        snapshots='all',
        fields='vh')

    PM.export_to_vtk(
        plot_dir +
        "/G_h",
        grid=None,
        npts_per_cell=[6] *
        2,
        snapshots='all',
        fields='Gh')

    PM.close()

if __name__ == '__main__':
    quad_domain()