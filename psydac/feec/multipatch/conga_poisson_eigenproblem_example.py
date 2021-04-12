# small script written to test Conga operators on multipatch domains, using the piecewise (broken) de Rham sequences available on every space

from mpi4py import MPI

from collections import OrderedDict
import numpy as np

from scipy.sparse.linalg import eigs, eigsh

from sympde.topology import Derham
from sympde.topology import Square
from sympde.topology import IdentityMapping, PolarMapping

from psydac.feec.multipatch.api import discretize  # TODO: when possible, use line above

from psydac.linalg.utilities import array_to_stencil

from psydac.fem.basic   import FemField

from psydac.feec.multipatch.fem_linear_operators import FemLinearOperator, IdLinearOperator
from psydac.feec.multipatch.fem_linear_operators import SumLinearOperator, MultLinearOperator, ComposedLinearOperator
from psydac.feec.multipatch.operators import BrokenMass, ortho_proj_Hcurl
from psydac.feec.multipatch.operators import ConformingProjection_V0
from psydac.feec.multipatch.plotting_utilities import get_grid_vals_scalar, get_grid_vals_vector
from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_patch_knots_gridlines, my_small_plot
from psydac.feec.multipatch.multipatch_domain_utilities import get_annulus_fourpatches, get_pretzel

comm = MPI.COMM_WORLD


def run_poisson_2d_eigenproblem(nb_eigs, ncells, degree, show_all=False):
    """
    Poisson (ie, Laplace) eigenproblem solver

    :param nb_eigs:  of eigenmodnbes to be computed
    :return: eigenvalues and eigenmodes
    """
    pretzel = True

    if pretzel:
        domain = get_pretzel(h=0.5, r_min=1, r_max=1.5, debug_option=0)
    else:
        OmegaLog1 = Square('OmegaLog1',bounds1=(0., 1.), bounds2=(0., 0.5))
        OmegaLog2 = Square('OmegaLog2',bounds1=(0., 1.), bounds2=(0.5, 1.))
        mapping_1 = IdentityMapping('M1',2)
        mapping_2 = IdentityMapping('M2',2)
        domain_1     = mapping_1(OmegaLog1)
        domain_2     = mapping_2(OmegaLog2)

        domain = domain_1.join(domain_2, name = 'domain',
                    bnd_minus = domain_1.get_boundary(axis=1, ext=1),
                    bnd_plus  = domain_2.get_boundary(axis=1, ext=-1))

        # mappings  = {OmegaLog1.interior:mapping_1, OmegaLog2.interior:mapping_2}

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = list(mappings.values())

    # x,y    = domain.coordinates

    nquads = [d + 1 for d in degree]

    # multipatch de Rham sequence:
    derham  = Derham(domain, ["H1", "Hcurl", "L2"])

    domain_h = discretize(domain, ncells=ncells, comm=comm)
    derham_h = discretize(derham, domain_h, degree=degree)
    V0h = derham_h.V0
    V1h = derham_h.V1
    # V2h = derham_h.V2

    # Mass matrices for broken spaces (block-diagonal)
    M0 = BrokenMass(V0h, domain_h, is_scalar=True)
    M1 = BrokenMass(V1h, domain_h, is_scalar=False)
    bD0, bD1 = derham_h.broken_derivatives_as_operators

    cP0 = ConformingProjection_V0(V0h, domain_h, hom_bc=True)
    I0 = IdLinearOperator(V0h)

    A = ComposedLinearOperator([I0-cP0,I0-cP0]) + ComposedLinearOperator([cP0, bD0.transpose(), M1, bD0, cP0])

    # Find eigenmodes and eigenvalues with scipy.sparse
    A = A.to_sparse_matrix()
    M0 = M0.to_sparse_matrix()
    # eigenvalues, eigenvectors = eigs(A, k=nb_eigs, which='SM' )   # 'SM' = smallest magnitude
    ncv = 4*nb_eigs
    # mode='cayley'
    mode='normal'
    eigenvalues, eigenvectors = eigsh(A, k=nb_eigs, M=M0, sigma=1, mode=mode, which='LM', ncv=ncv)

    print(type(eigenvalues))
    print(type(eigenvectors))
    print(eigenvectors.shape)
    # plotting
    etas, xx, yy = get_plotting_grid(mappings, N=20)
    grid_vals_h1 = lambda v: get_grid_vals_scalar(v, etas, mappings_list, space_kind='h1')

    first_Pemodes_vals = []
    first_Pemodes_titles = []

    for k_eig in range(nb_eigs):
        evalue  = eigenvalues[k_eig]
        emode_c = array_to_stencil(eigenvectors[:,k_eig], V0h.vector_space)
        emode = FemField(V0h, coeffs=emode_c)
        emode = cP0(emode)

        uh_vals = grid_vals_h1(emode)

        if show_all:
            my_small_plot(
                title='mode nb k='+repr(k_eig)+'  --  eigenvalue = '+repr(evalue),
                vals=[uh_vals],
                titles=[r'$u^h_{k}(x,y)$'],
                xx=xx,
                yy=yy,
                cmap='magma',
                surface_plot=True,
            )

        if k_eig < 8:
            first_Pemodes_vals.append(uh_vals)
            first_Pemodes_titles.append(r'$\sigma=$'+'{0:0.2f}'.format(np.real(evalue)))
        else:
            print('warning: not plotting eigenmode for k = ' + repr(k_eig))

    my_small_plot(
        title=r'Amplitude $|P^1_c e^h_k|$ of some eigenmodes found',
        vals=first_Pemodes_vals,
        titles=first_Pemodes_titles,
        xx=xx,
        yy=yy,
        cmap='magma',
        surface_plot=True,
    )

if __name__ == '__main__':

    run_poisson_2d_eigenproblem(nb_eigs=20, ncells=[2**4, 2**4], degree=[2,2], show_all=True)