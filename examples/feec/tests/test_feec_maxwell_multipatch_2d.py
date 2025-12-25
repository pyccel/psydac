#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import numpy as np

from examples.feec.hcurl_source_pbms_conga_2d   import solve_hcurl_source_pbm
from examples.feec.hcurl_eigen_pbms_conga_2d    import hcurl_solve_eigen_pbm
from examples.feec.hcurl_eigen_pbms_dg_2d       import hcurl_solve_eigen_pbm_dg
from examples.feec.timedomain_maxwell           import solve_td_maxwell_pbm

def test_time_harmonic_maxwell_pretzel_f():
    nc = 4
    deg = 2

    source_type = 'manu_maxwell_inhom'
    domain_name = 'pretzel_f'

    omega = np.pi
    eta = -omega**2  # source

    err = solve_hcurl_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc')

    assert abs(err - 0.0072015081402929445) < 1e-10

def test_time_harmonic_maxwell_pretzel_f_nc():
    deg = 2
    nc = np.array([8, 8, 8, 8, 8, 4, 4, 4, 4,
                   4, 4, 4, 4, 8, 8, 8, 4, 4])

    source_type = 'manu_maxwell_inhom'
    domain_name = 'pretzel_f'

    omega = np.pi
    eta = -omega**2  # source

    err = solve_hcurl_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc')

    assert abs(err - 0.004849225522124346) < 5e-7

def test_maxwell_eigen_curved_L_shape():
    domain_name = 'curved_L_shape'
    domain = [[1, 3], [0, np.pi / 4]]
    
    ncells = 4
    degree = [2, 2]

    ref_sigmas = [
        0.181857115231E+01,
        0.349057623279E+01,
        0.100656015004E+02,
        0.101118862307E+02,
        0.124355372484E+02,
    ]
    sigma = 7
    nb_eigs_solve = 7
    nb_eigs_plot = 7
    skip_eigs_threshold = 1e-7

    eigenvalues = hcurl_solve_eigen_pbm(
        ncells=ncells, degree=degree,
        gamma_h=0,
        generalized_pbm=True,
        nu=0,
        mu=1,
        sigma=sigma,
        skip_eigs_threshold=skip_eigs_threshold,
        nb_eigs_solve=nb_eigs_solve,
        nb_eigs_plot=nb_eigs_plot,
        domain_name=domain_name, domain=domain,
        backend_language='pyccel-gcc',
    )

    error = 0
    n_errs = min(len(ref_sigmas), len(eigenvalues))
    for k in range(n_errs):
        error += (eigenvalues[k] - ref_sigmas[k])**2
    error = np.sqrt(error)

    assert abs(error - 0.012915398994855902) < 1e-10

def test_maxwell_eigen_curved_L_shape_nc():
    domain_name = 'curved_L_shape'
    domain = [[1, 3], [0, np.pi / 4]]

    ncells = np.array([[None, 4],
                       [4, 8]])

    degree = [2, 2]

    ref_sigmas = [
        0.181857115231E+01,
        0.349057623279E+01,
        0.100656015004E+02,
        0.101118862307E+02,
        0.124355372484E+02,
    ]
    sigma = 7
    nb_eigs_solve = 7
    nb_eigs_plot = 7
    skip_eigs_threshold = 1e-7

    eigenvalues = hcurl_solve_eigen_pbm(
        ncells=ncells, degree=degree,
        gamma_h=0,
        generalized_pbm=True,
        nu=0,
        mu=1,
        sigma=sigma,
        skip_eigs_threshold=skip_eigs_threshold,
        nb_eigs_solve=nb_eigs_solve,
        nb_eigs_plot=nb_eigs_plot,
        domain_name=domain_name, domain=domain,
        backend_language='pyccel-gcc',
    )

    error = 0
    n_errs = min(len(ref_sigmas), len(eigenvalues))
    for k in range(n_errs):
        error += (eigenvalues[k] - ref_sigmas[k])**2
    error = np.sqrt(error)

    assert abs(error - 0.010504876643886937) < 1e-10

def test_maxwell_eigen_curved_L_shape_dg():
    domain_name = 'curved_L_shape'
    domain = [[1, 3], [0, np.pi / 4]]

    ncells = np.array([[None, 4],
                       [4, 8]])

    degree = [2, 2]

    ref_sigmas = [
        0.181857115231E+01,
        0.349057623279E+01,
        0.100656015004E+02,
        0.101118862307E+02,
        0.124355372484E+02,
    ]
    sigma = 7
    nb_eigs_solve = 7
    nb_eigs_plot = 7
    skip_eigs_threshold = 1e-7

    eigenvalues = hcurl_solve_eigen_pbm_dg(
        ncells=ncells, degree=degree,
        nu=0,
        mu=1,
        sigma=sigma,
        skip_eigs_threshold=skip_eigs_threshold,
        nb_eigs_solve=nb_eigs_solve,
        nb_eigs_plot=nb_eigs_plot,
        domain_name=domain_name, domain=domain,
        backend_language='pyccel-gcc',
    )

    error = 0
    n_errs = min(len(ref_sigmas), len(eigenvalues))
    for k in range(n_errs):
        error += (eigenvalues[k] - ref_sigmas[k])**2
    error = np.sqrt(error)

    assert abs(error - 0.035139029534592255) < 1e-10

def test_maxwell_timedomain():
    solve_td_maxwell_pbm(nc = 4, deg = 2, final_time = 2, domain_name = 'square_2')

# ==============================================================================
# CLEAN UP SYMPY NAMESPACE
# ==============================================================================
def teardown_module():
    from sympy.core import cache
    cache.clear_cache()


def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
