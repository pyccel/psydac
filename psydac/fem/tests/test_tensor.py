#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
import contextlib
from pathlib import Path

import pytest
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from mpi4py import MPI

from sympde.topology.domain import Square
from sympde.topology.space import ScalarFunctionSpace
from sympde.topology.analytical_mapping import TargetMapping

from psydac.mapping.discrete_gallery import discrete_mapping
from psydac.api.discretization import discretize


#==============================================================================
# Machinery for comparing a PNG image with a reference on the root MPI process
#==============================================================================

def similar_images(file1, file2, tolerance=0.01):
    """
    Compare two PNG images and check if they are similar within tolerance.

    Parameters
    ----------
    file1 : str
        Path to first PNG file.
    file2 : str
        Path to second PNG file.
    tolerance: float
        Maximum allowed average difference between pixel values (0-1).

    Returns
    -------
    bool :
        True if images are similar enough, False otherwise.
    """
    # Load images and convert to numpy arrays
    img1 = np.array(Image.open(file1)).astype(float)
    img2 = np.array(Image.open(file2)).astype(float)

    # Check dimensions match
    if img1.shape != img2.shape:
        return False

    # Normalize pixel values to 0-1
    img1 = img1 / 255.0
    img2 = img2 / 255.0

    # Calculate mean absolute difference
    diff = np.mean(np.abs(img1 - img2))

    return diff <= tolerance


@contextlib.contextmanager
def consistent_png_rendering():
    """
    Context manager for consistent Matplotlib rendering across platforms.
    """
    # Store original settings
    orig_backend = mpl.get_backend()
    orig_settings = {
        'text.usetex': mpl.rcParams['text.usetex'],
        'font.family': mpl.rcParams['font.family'],
    }

    try:
        # Use Agg backend (pure python, no GUI)
        mpl.use('Agg')
        # Configure settings for consistent rendering
        mpl.rcParams.update({
            'text.usetex': False,
            'font.family': 'DejaVu Sans',
        })
        yield  # Control returns to the with block
    finally:
        # Restore original settings
        mpl.use(orig_backend)
        mpl.rcParams.update(orig_settings)


def compare_figure_to_reference(fig, filename, *, dpi, tol, folder, comm, root):
    """
    Compare a matplotlib figure to a reference PNG file on the root MPI process.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to compare with the reference image.
    filename : str
        Name of the PNG file to save and compare.
    dpi : int
        Dots per inch resolution for saving the figure.
    tol : float
        Tolerance for image comparison (between 0 and 1).
    folder : str
        Name of the folder containing reference images.
    comm : mpi4py.MPI.Comm
        MPI communicator object.
    root : int
        Rank of the MPI process that should perform the comparison.

    Returns
    -------
    bool
        True if the images are similar enough within the tolerance,
        False otherwise. The result is broadcast to all MPI processes.

    Notes
    -----
    The function saves the figure to a temporary file, compares it with
    the reference image, and then removes the temporary file. Only the
    root process performs the actual comparison, but the result is
    broadcast to all processes.
    """
    if comm.rank == root:
        test_dir = Path(__file__).parent.absolute()
        file1 = test_dir / filename
        file2 = test_dir / folder / filename
        with consistent_png_rendering():
            fig.savefig(file1, dpi=dpi)
        close_enough = similar_images(file1, file2, tol)
        # Clean up the temporary file
        os.remove(file1)
    else:
        close_enough = None

    # Broadcast the boolean result from the root process to all others
    close_enough = comm.bcast(close_enough, root=root)

    # All MPI processes return the same result
    return close_enough

#==============================================================================
# Unit tests
#==============================================================================
@pytest.mark.mpi
@pytest.mark.parametrize('root', ['first', 'last'])
@pytest.mark.parametrize('kind', ['spline', 'analytical'])
def test_plot_2d_decomposition(kind, root):

    # MPI communicator
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.size
    mpi_rank = mpi_comm.rank

    # MPI rank which should make the plot
    if root == 'first':
        mpi_root = 0
    elif root == 'last':
        mpi_root = mpi_size - 1
    else:
        raise ValueError(f'root argument has wrong value {root}')

    # Parameters of tensor-product 2D spline space
    ncells = (6, 9)
    degree = (2, 2)

    if kind == 'spline':
        # 2D spline mapping and tensor FEM space (distributed)
        F, Vh = discrete_mapping('target', ncells=ncells, degree=degree,
                            comm=mpi_comm, return_space=True)
    elif kind == 'analytical':
        Omega = Square('Omega', bounds1=(0, 1), bounds2=(0, 2 * np.pi))
        params = dict(c1=0, c2=0, k=0.3, D=0.2)
        M = TargetMapping('M', dim=2, **params)
        domain = M(Omega)
        V = ScalarFunctionSpace('V', domain)

        # 2D Geometry object
        domain_h = discretize(domain, ncells=ncells, periodic=(False, True),
                              comm=mpi_comm)

        # 2D spline tensor FEM space (distributed)
        Vh = discretize(V, domain_h, degree=degree)

        # 2D callable mapping (analytical)
        F = M.get_callable_mapping()
    else:
        raise ValueError(f'kind argument has wrong value {kind}')

    # Name of temporary image file to be compared with reference one
    filename = f'decomp_{kind}_{mpi_size}_procs.png'

    # Relative tolerance for image comparison
    RTOL = 0.02

    # Plot 2D decomposition
    # [1] Run without passing (fig, ax)
    fig = Vh.plot_2d_decomposition(F, refine=5, mpi_root=mpi_root)
    assert compare_figure_to_reference(fig, filename, folder='data', dpi=100,
                                       tol=RTOL, comm=mpi_comm, root=mpi_root)

    # [2] Run with given (fig, ax), compatible
    fig2, ax2 = plt.subplots(1, 1) if mpi_rank == mpi_root else (None, None)
    Vh.plot_2d_decomposition(F, refine=5, fig=fig2, ax=ax2, mpi_root=mpi_root)
    assert compare_figure_to_reference(fig2, filename, folder='data', dpi=100,
                                       tol=RTOL, comm=mpi_comm, root=mpi_root)

    # [3] Run with given (fig, ax), incompatible
    if mpi_rank == mpi_root:
        fig3, ax3 = plt.subplots(1, 1)
        with pytest.raises(AssertionError) as excinfo:
            Vh.plot_2d_decomposition(F, refine=5, fig=fig2, ax=ax3, mpi_root=mpi_root)
        assert "Argument `ax` must be in `fig.axes`" in str(excinfo.value)
        plt.close(fig3)
    else:
        Vh.plot_2d_decomposition(F, refine=5, fig=None, ax=None, mpi_root=mpi_root)

#==============================================================================
if __name__ == '__main__':

    test_plot_2d_decomposition('spline', 'first')
    test_plot_2d_decomposition('analytical', 'last')
    plt.show()
