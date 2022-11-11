import os
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sympde.topology     import Derham

from psydac.feec.multipatch.api                         import discretize
from psydac.api.settings                                import PSYDAC_BACKENDS
from psydac.feec.multipatch.fem_linear_operators        import IdLinearOperator
from psydac.feec.multipatch.operators                   import HodgeOperator
#from psydac.feec.multipatch.multipatch_domain_utilities import build_multipatch_domain
from psydac.feec.multipatch.plotting_utilities          import plot_field
from psydac.feec.multipatch.utilities                   import time_count, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir, diag_fn
from psydac.feec.multipatch.utils_conga_2d              import write_diags_to_file

from sympde.topology      import Square    
from sympde.topology      import IdentityMapping, PolarMapping
from psydac.fem.vector import ProductFemSpace

from scipy.sparse.linalg import spilu, lgmres
from scipy.sparse.linalg import LinearOperator, eigsh, minres
from scipy.sparse          import csr_matrix
from scipy.linalg        import norm

from psydac.linalg.utilities import array_to_stencil
from psydac.fem.basic        import FemField

#from psydac.feec.multipatch.examples_nc.non_conf_domains_examples import create_square_domain
#from psydac.feec.multipatch.multipatch_non_conf_scipy import construct_V1_conforming_projection

#from psydac.api.postprocessing import OutputManager, PostProcessManager

#from said
from scipy.sparse.linalg import spsolve, inv

from sympde.calculus      import grad, dot, curl, cross
from sympde.calculus      import minus, plus
from sympde.topology      import VectorFunctionSpace
from sympde.topology      import elements_of
from sympde.topology      import NormalVector
from sympde.topology      import Square
from sympde.topology      import IdentityMapping, PolarMapping
from sympde.expr.expr     import LinearForm, BilinearForm
from sympde.expr.expr     import integral
from sympde.expr.expr     import Norm
from sympde.expr.equation import find, EssentialBC

#from psydac.api.tests.build_domain   import build_pretzel
from psydac.fem.basic                import FemField
from psydac.api.settings             import PSYDAC_BACKEND_GPYCCEL
from psydac.feec.pull_push           import pull_2d_hcurl

def hcurl_solve_eigen_pbm_multipatch_nc(ncells=[[2,2], [2,2]], degree=[3,3], domain=[[0, np.pi],[0, np.pi]], domain_name='refined_square', backend_language='pyccel-gcc',
                          sigma=None):

    

    int_x, int_y = domain
    
    if domain_name == 'refined_square' or domain_name =='square_L_shape':
        domain = create_square_domain(ncells, int_x, int_y, mapping='identity')
        ncells_h = {patch.name: [ncells[int(patch.name[2])][int(patch.name[4])], ncells[int(patch.name[2])][int(patch.name[4])]] for patch in domain.interior}
    elif domain_name == 'curved_L_shape':
        domain = create_square_domain(ncells, int_x, int_y, mapping='polar')
        ncells_h = {patch.name: [ncells[int(patch.name[2])][int(patch.name[4])], ncells[int(patch.name[2])][int(patch.name[4])]] for patch in domain.interior}
    elif domain_name == 'pretzel_f':
        domain = build_multipatch_domain(domain_name=domain_name) 
        ncells_h = {patch.name: [ncells[i], ncells[i]] for (i,patch) in enumerate(domain.interior)}

    else:
        ValueError("Domain not defined.")


    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())


    domain_h = discretize(domain, ncells=ncells_h)   # Vh space

    V  = VectorFunctionSpace('V', domain, kind='hcurl')

    u, v, F  = elements_of(V, names='u, v, F')
    nn       = NormalVector('nn')

    I        = domain.interfaces
    boundary = domain.boundary

    kappa   = 10
    k       = 1

    jump = lambda w:plus(w)-minus(w)
    avr  = lambda w:0.5*plus(w) + 0.5*minus(w)

    expr1_I  =  cross(nn, jump(v))*curl(avr(u))\
               +k*cross(nn, jump(u))*curl(avr(v))\
               +kappa*cross(nn, jump(u))*cross(nn, jump(v))

    expr1   = curl(u)*curl(v) 
    expr1_b = -cross(nn, v) * curl(u) -k*cross(nn, u)*curl(v)  + kappa*cross(nn, u)*cross(nn, v)
    #curl curl u = - omega**2 u 

    expr2   = dot(u,v)
    #expr2_I  = kappa*cross(nn, jump(u))*cross(nn, jump(v))
    #expr2_b = -k*cross(nn, u)*curl(v) + kappa * cross(nn, u) * cross(nn, v)

    # Bilinear form a: V x V --> R
    a      = BilinearForm((u,v),  integral(domain, expr1) + integral(I, expr1_I) + integral(boundary, expr1_b))
    
    # Linear form l: V --> R
    b     = BilinearForm((u,v), integral(domain, expr2))# + integral(I, expr2_I) + integral(boundary, expr2_b))

    Vh       = discretize(V, domain_h, degree=degree, basis='M')

    ah = discretize(a, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[backend_language])
    Ah_m = ah.assemble().tosparse()

    bh = discretize(b, domain_h, [Vh, Vh], backend=PSYDAC_BACKENDS[backend_language])
    Bh_m = bh.assemble().tosparse()


    all_eigenvalues_2, all_eigenvectors_transp_2 = get_eigenvalues(10, sigma, Ah_m, Bh_m)
    
    print(all_eigenvalues_2)
   



def get_eigenvalues(nb_eigs, sigma, A_m, M_m):
    print('-----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  ----- ')
    print('computing {0} eigenvalues (and eigenvectors) close to sigma={1} with scipy.sparse.eigsh...'.format(nb_eigs, sigma) )
    mode = 'normal'
    which = 'LM'
    # from eigsh docstring:
    #   ncv = number of Lanczos vectors generated ncv must be greater than k and smaller than n;
    #   it is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))
    ncv = 4*nb_eigs
    print('A_m.shape = ', A_m.shape)
    try_lgmres = True
    max_shape_splu = 24000   # OK for nc=20, deg=6 on pretzel_f
    if A_m.shape[0] < max_shape_splu:
        print('(via sparse LU decomposition)')
        OPinv = None
        tol_eigsh = 0
    else:

        OP_m = A_m - sigma*M_m
        tol_eigsh = 1e-7
        if try_lgmres:
            print('(via SPILU-preconditioned LGMRES iterative solver for A_m - sigma*M1_m)')
            OP_spilu = spilu(OP_m, fill_factor=15, drop_tol=5e-5)
            preconditioner = LinearOperator(OP_m.shape, lambda x: OP_spilu.solve(x) )
            tol = tol_eigsh
            OPinv = LinearOperator(
                matvec=lambda v: lgmres(OP_m, v, x0=None, tol=tol, atol=tol, M=preconditioner,
                                    callback=lambda x: print('cg -- residual = ', norm(OP_m.dot(x)-v))
                                    )[0],
                shape=M_m.shape,
                dtype=M_m.dtype
            )

        else:
            # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html:
            # the user can supply the matrix or operator OPinv, which gives x = OPinv @ b = [A - sigma * M]^-1 @ b.
            # > here, minres: MINimum RESidual iteration to solve Ax=b
            # suggested in https://github.com/scipy/scipy/issues/4170
            print('(with minres iterative solver for A_m - sigma*M1_m)')
            OPinv = LinearOperator(matvec=lambda v: minres(OP_m, v, tol=1e-10)[0], shape=M_m.shape, dtype=M_m.dtype)

    eigenvalues, eigenvectors = eigsh(A_m, k=nb_eigs, M=M_m, sigma=sigma, mode=mode, which=which, ncv=ncv, tol=tol_eigsh, OPinv=OPinv)

    print("done: eigenvalues found: " + repr(eigenvalues))
    return eigenvalues, eigenvectors

def create_square_domain(ncells, interval_x, interval_y, mapping='identity'):
    from mpi4py import MPI
    import numpy as np
    from sympde.topology import Square
    from sympde.topology import IdentityMapping, PolarMapping, AffineMapping, Mapping
    from sympde.topology  import Boundary, Interface, Union

    from scipy.sparse import eye as sparse_eye
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import inv
    from scipy.sparse import coo_matrix, bmat
    from scipy.sparse.linalg import inv as sp_inv

    from psydac.feec.multipatch.utilities import time_count
    from psydac.linalg.utilities          import array_to_stencil
    from psydac.feec.multipatch.api       import discretize
    from psydac.api.settings              import PSYDAC_BACKENDS
    from psydac.fem.splines               import SplineSpace

    from psydac.feec.multipatch.multipatch_domain_utilities import union, set_interfaces, build_multipatch_domain
    """
    Create a 2D multipatch square domain with the prescribed number of patch in each direction.

    Parameters
    ----------
    ncells: <matrix>

    |2|
    _____
    |4|2|

    [[2, None],
     [4, 2]]

     [[2, 2, 0, 0],
      [2, 4, 0, 0],
      [4, 8, 4, 2],
      [4, 4, 2, 2]]
     number of patch in each direction

    Returns
    -------
    domain : <Sympde.topology.Domain>
     The symbolic multipatch domain
    """
    ax, bx = interval_x
    ay, by = interval_y 
    nb_patchx, nb_patchy = np.shape(ncells)

    list_Omega = [[Square('OmegaLog_'+str(i)+'_'+str(j),
                    bounds1 = (ax + i/nb_patchx * (bx-ax),ax + (i+1)/nb_patchx * (bx-ax)),
                    bounds2 = (ay + j/nb_patchy * (by-ay),ay + (j+1)/nb_patchy * (by-ay))) for j in range(nb_patchy)] for i in range(nb_patchx)]
    
        
    if mapping == 'identity':
        list_mapping = [[IdentityMapping('M_'+str(i)+'_'+str(j),2) for j in range(nb_patchy)] for i in range(nb_patchx)]

    elif mapping == 'polar':
        list_mapping = [[PolarMapping('M_'+str(i)+'_'+str(j),2, c1= 0., c2= 0., rmin = 0., rmax=1.) for j in range(nb_patchy)] for i in range(nb_patchx)]

    list_domain = [[list_mapping[i][j](list_Omega[i][j]) for j in range(nb_patchy)] for i in range(nb_patchx)]
    flat_list = []
    for i in range(nb_patchx):
        for j in range(nb_patchy):
            if ncells[i, j] != None:
                flat_list.append(list_domain[i][j])

    domain = union(flat_list, name='domain')
    interfaces = []

    #interfaces in y
    for j in range(nb_patchy):
        interfaces.extend([[list_domain[i][j].get_boundary(axis=0, ext=+1), list_domain[i+1][j].get_boundary(axis=0, ext=-1), 1] for i in range(nb_patchx-1) if ncells[i][j] != None and ncells[i+1][j] != None])

    #interfaces in x
    for i in range(nb_patchx):
        interfaces.extend([[list_domain[i][j].get_boundary(axis=1, ext=+1), list_domain[i][j+1].get_boundary(axis=1, ext=-1), 1] for j in range(nb_patchy-1) if ncells[i][j] != None and ncells[i][j+1] != None])

    domain = set_interfaces(domain, interfaces)

    return domain


if __name__ == '__main__':

    domain=[[0, np.pi],[0, np.pi]] 

    domain_name = 'refined_square' 
    
    ncells = np.array([[16, 8],
                        [8, 8]])
    sigma = 5

    hcurl_solve_eigen_pbm_multipatch_nc(ncells=ncells, degree=[3,3], domain=domain, domain_name=domain_name, backend_language='pyccel-gcc', sigma=sigma)

        