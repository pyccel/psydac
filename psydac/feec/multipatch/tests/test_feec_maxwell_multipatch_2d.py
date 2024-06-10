# coding: utf-8

import numpy as np

from psydac.feec.multipatch.examples.hcurl_source_pbms_conga_2d import solve_hcurl_source_pbm
from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d import hcurl_solve_eigen_pbm
from psydac.feec.multipatch.examples.hcurl_eigen_pbms_dg_2d import hcurl_solve_eigen_pbm_dg


def test_time_harmonic_maxwell_pretzel_f():
    nc = 10
    deg = 2

    source_type = 'manu_maxwell_inhom'
    domain_name = 'pretzel_f'
    source_proj = 'tilde_Pi'

    omega = np.pi
    eta = -omega**2  # source

    diags = solve_hcurl_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        source_proj=source_proj,
        backend_language='pyccel-gcc')
    assert abs(diags["err"] - 0.00016729140844149693) < 1e-10


def test_time_harmonic_maxwell_pretzel_f_nc():
    deg = 2
    nc = np.array([20, 20, 20, 20, 20, 10, 10, 10, 10,
                  10, 10, 10, 10, 20, 20, 20, 10, 10])

    source_type = 'manu_maxwell_inhom'
    domain_name = 'pretzel_f'
    source_proj = 'tilde_Pi'

    omega = np.pi
    eta = -omega**2  # source

    diags = solve_hcurl_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        source_proj=source_proj,
        backend_language='pyccel-gcc')

    assert abs(diags["err"] - 0.00012830429612706266) < 1e-10


def test_maxwell_eigen_curved_L_shape():
    domain_name = 'curved_L_shape'
    domain = [[1, 3], [0, np.pi / 4]]
    
    ncells = 10
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

    diags, eigenvalues = hcurl_solve_eigen_pbm(
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
        plot_dir='./plots/eigen_maxell',
    )

    error = 0
    n_errs = min(len(ref_sigmas), len(eigenvalues))
    for k in range(n_errs):
        error += (eigenvalues[k] - ref_sigmas[k])**2
    error = np.sqrt(error)

    assert abs(error - 0.004697863286378944) < 1e-10


def test_maxwell_eigen_curved_L_shape_nc():
    domain_name = 'curved_L_shape'
    domain = [[1, 3], [0, np.pi / 4]]

    ncells = np.array([[None, 10],
                       [10, 20]])

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

    diags, eigenvalues = hcurl_solve_eigen_pbm(
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
        plot_dir='./plots/eigen_maxell_nc',
    )

    error = 0
    n_errs = min(len(ref_sigmas), len(eigenvalues))
    for k in range(n_errs):
        error += (eigenvalues[k] - ref_sigmas[k])**2
    error = np.sqrt(error)

    assert abs(error - 0.004301175400024398) < 1e-10


def test_maxwell_eigen_curved_L_shape_dg():
    domain_name = 'curved_L_shape'
    domain = [[1, 3], [0, np.pi / 4]]

    ncells = np.array([[None, 10],
                       [10, 20]])

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

    diags, eigenvalues = hcurl_solve_eigen_pbm_dg(
        ncells=ncells, degree=degree,
        nu=0,
        mu=1,
        sigma=sigma,
        skip_eigs_threshold=skip_eigs_threshold,
        nb_eigs_solve=nb_eigs_solve,
        nb_eigs_plot=nb_eigs_plot,
        domain_name=domain_name, domain=domain,
        backend_language='pyccel-gcc',
        plot_dir='./plots/eigen_maxell_dg',
    )

    error = 0
    n_errs = min(len(ref_sigmas), len(eigenvalues))
    for k in range(n_errs):
        error += (eigenvalues[k] - ref_sigmas[k])**2
    error = np.sqrt(error)

    assert abs(error - 0.004208158031148591) < 1e-10


# ==============================================================================
# CLEAN UP SYMPY NAMESPACE
# ==============================================================================
def teardown_module():
    from sympy.core import cache
    cache.clear_cache()


def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
