# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigsh

from sympde.topology import Derham
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above

from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, ortho_proj_Hcurl
from psydac.feec.multipatch.operators import ConformingProjection_V0, ConformingProjection_V1
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_patch_knots_gridlines, my_small_plot
from psydac.feec.multipatch.multipatch_domain_utilities import get_annulus_fourpatches, get_pretzel

comm = MPI.COMM_WORLD


def run_maxwell_2d_eigenproblem(nb_eigs, ncells, degree, alpha, cartesian=True, use_pretzel=True, compute_kernel=False, show_all=False):
    """
    Maxwell eigenproblem solver, see eg
    Buffa, Perugia & Warburton, The Mortar-Discontinuous Galerkin Method for the 2D Maxwell Eigenproblem JSC 2009.

    :param nb_eigs: nb of eigenmodes to be computed
    :return: eigenvalues and eigenmodes
    """

    if use_pretzel:
        # domain = get_pretzel(h=0.5, r_min=1, r_max=1.5, debug_option=0)
        domain = get_pretzel(h=0.5, r_min=1, r_max=1.5, debug_option=1)
        # domain = get_annulus_fourpatches(r_min=0.5, r_max=1)

    else:
        if cartesian:
            OmegaLog1 = Square('OmegaLog1',bounds1=(0., np.pi), bounds2=(0., np.pi/2))
            OmegaLog2 = Square('OmegaLog2',bounds1=(0., np.pi), bounds2=(np.pi/2, np.pi))
            mapping_1 = IdentityMapping('M1',2)
            mapping_2 = IdentityMapping('M2',2)
        else:
            OmegaLog1 = Square('OmegaLog1',bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
            OmegaLog2 = Square('OmegaLog2',bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))
            mapping_1 = PolarMapping('M1',2, c1= 0., c2= 0., rmin = 0., rmax=1.)
            mapping_2 = PolarMapping('M2',2, c1= 0., c2= 0., rmin = 0., rmax=1.)

        domain_1     = mapping_1(OmegaLog1)
        domain_2     = mapping_2(OmegaLog2)

        domain = domain_1.join(domain_2, name = 'domain',
                    bnd_minus = domain_1.get_boundary(axis=1, ext=1),
                    bnd_plus  = domain_2.get_boundary(axis=1, ext=-1))

        # mappings  = {OmegaLog1.interior:mapping_1, OmegaLog2.interior:mapping_2}

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    # x,y    = domain.coordinates
    # nquads = [d + 1 for d in degree]

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1
    V2h = derham_h.V2

    # Mass matrices for broken spaces (block-diagonal)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)
    M2 = BrokenMass(V2h, domain_h, is_scalar=True)
    bD0, bD1 = derham_h.broken_derivatives_as_operators
    cP0 = ConformingProjection_V0(V0h, domain_h, hom_bc=True)
    cP1 = ConformingProjection_V1(V1h, domain_h, hom_bc=True)
    D0 = ComposedLinearOperator([bD0,cP0])
    D0_t = ComposedLinearOperator([cP0, bD0.transpose()])
    D1 = ComposedLinearOperator([bD1,cP1])
    D1_t = ComposedLinearOperator([cP1, bD1.transpose()])
    I1 = IdLinearOperator(V1h)

    print("using jump penalization factor alpha = ", alpha )

    # A = ComposedLinearOperator([I1-cP1,I1-cP1]) + ComposedLinearOperator([cP1, bD1.transpose(), M2, bD1, cP1])
    A = ( ComposedLinearOperator([D1_t, M2, D1])
        + alpha*ComposedLinearOperator([I1-cP1,M1, I1-cP1])
        )

        # + M1

        #
        # + 1000*ComposedLinearOperator([M1, D0, D0_t, M1])

    # Find eigenmodes and eigenvalues with scipy.sparse.eigsh (symmetric matrices)
    A = A.to_sparse_matrix()
    M1 = M1.to_sparse_matrix()

    if compute_kernel:
        sigma = 0
        mode = 'normal'
        which = 'LM'
    else:
        sigma = 4
        mode='cayley'
        # mode='buckling'
        which = 'LM'

    # from eigsh docstring:
    #   ncv = number of Lanczos vectors generated ncv must be greater than k and smaller than n;
    #   it is recommended that ncv > 2*k. Default: min(n, max(2*k + 1, 20))
    ncv = 4*nb_eigs
    # search mode: normal and buckling give a lot of zero eigenmodes. Cayley seems best for Maxwell.
    # mode='normal'

    eigenvalues, eigenvectors = eigsh(A, k=nb_eigs, M=M1, sigma=sigma, mode=mode, which=which, ncv=ncv)

    # plotting
    etas, xx, yy = get_plotting_grid(mappings, N=20)

    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')
    grid_vals_hcurl = lambda v: get_grid_vals_vector(v, etas, mappings_list, space_kind='hcurl')

    first_Pemodes_vals = []
    first_Pemodes_titles = []

    for k_eig in range(nb_eigs):
        evalue  = eigenvalues[k_eig]
        emode_sp = np.real(eigenvectors[:,k_eig])
        # normalize mode in L2
        Me = M1.dot(emode_sp)
        norm_emode = np.dot(emode_sp,Me)
        emode_c = array_to_stencil(emode_sp/norm_emode, V1h.vector_space)
        emode = FemField(V1h, coeffs=emode_c)

        cP_emode = cP1(emode)
        curl_emode = D1(emode)

        eh_x_vals, eh_y_vals = grid_vals_hcurl(emode)
        cPeh_x_vals, cPeh_y_vals = grid_vals_hcurl(cP_emode)
        Peh_abs_vals = [np.sqrt(abs(Pex)**2 + abs(Pey)**2) for Pex, Pey in zip(cPeh_x_vals, cPeh_y_vals)]
        jumps_eh_vals = [np.sqrt(abs(ex-Pex)**2 + abs(ey-Pey)**2)
                         for ex, Pex, ey, Pey in zip (eh_x_vals, cPeh_x_vals, eh_y_vals, cPeh_y_vals)]
        curl_eh_vals = grid_vals_h1(curl_emode)

        if show_all:
            my_small_plot(
                title='mode k='+repr(k_eig)+'  --  norm = '+ repr(norm_emode) + '  --  eigenvalue = '+repr(evalue),
                vals=[eh_x_vals, eh_y_vals, Peh_abs_vals, jumps_eh_vals, curl_eh_vals],
                titles=[r'$e^h_{k,x}$', r'$e^h_{k,y}$', r'$|P^1_c e^h_k|$', r'$|(I-P^1_c) e^h_k|$', r'curl$(e^h_k)$'],
                xx=xx,
                yy=yy,
            )

        if k_eig < 8:
            first_Pemodes_vals.append(Peh_abs_vals)
            first_Pemodes_titles.append(r'$\sigma=$'+'{0:0.2f}'.format(np.real(evalue)))
        else:
            print('warning: not plotting eigenmode for k = ' + repr(k_eig))

    my_small_plot(
        title=r'Amplitude $|P^1_c e^h_k|$ of some eigenmodes found',
        vals=first_Pemodes_vals,
        titles=first_Pemodes_titles,
        xx=xx,
        yy=yy,
    )

if __name__ == '__main__':

    nc = 2**3
    h = 1/nc
    deg = 3
    # jump penalization factor from Buffa, Perugia and Warburton
    DG_alpha = 10*(deg+1)**2/h

    # a = [76.753, 7.4143254, 986.654352]
    # title = ' '
    # for i in a:
    #     title += '{0:0.2f}'.format(i)+' '
    # print( 'Got '+title)
    # exit()

    run_maxwell_2d_eigenproblem(nb_eigs=8, ncells=[nc, nc], degree=[deg,deg], alpha=DG_alpha, cartesian=True, use_pretzel=False, show_all=True)