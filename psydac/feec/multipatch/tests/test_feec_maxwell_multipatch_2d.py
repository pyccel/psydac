# coding: utf-8

import numpy as np

from psydac.feec.multipatch.examples.hcurl_source_pbms_conga_2d import solve_hcurl_source_pbm
from psydac.feec.multipatch.examples_nc.hcurl_source_pbms_nc import solve_hcurl_source_pbm_nc

from psydac.feec.multipatch.examples.hcurl_eigen_pbms_conga_2d import hcurl_solve_eigen_pbm
from psydac.feec.multipatch.examples_nc.hcurl_eigen_pbms_nc import hcurl_solve_eigen_pbm_nc
from psydac.feec.multipatch.examples_nc.hcurl_eigen_pbms_dg import hcurl_solve_eigen_pbm_dg


def test_time_harmonic_maxwell_pretzel_f():
    nc, deg = 10, 2
    source_type = 'manu_maxwell'
    domain_name = 'pretzel_f'

    eta = -170.0  # source

    l2_error = solve_hcurl_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc')

    assert abs(l2_error - 0.06247745643640749) < 1e-10


def test_time_harmonic_maxwell_pretzel_f_nc():
    deg = 2
    nc = np.array([20, 20, 20, 20, 20, 10, 10, 10, 10,
                  10, 10, 10, 10, 20, 20, 20, 10, 10])

    source_type = 'manu_maxwell'
    domain_name = 'pretzel_f'
    source_proj = 'tilde_Pi'

    eta = -170.0

    l2_error = solve_hcurl_source_pbm_nc(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        source_proj=source_proj,
        plot_dir='./plots/th_maxell_nc',
        backend_language='pyccel-gcc',
        test=True)

    assert abs(l2_error - 0.04753613858909066) < 1e-10


def test_maxwell_eigen_curved_L_shape():
    domain_name = 'curved_L_shape'

    nc = 10
    deg = 2

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

    eigenvalues, eigenvectors = hcurl_solve_eigen_pbm(
        nc=nc, deg=deg,
        gamma_h=0,
        nu=0,
        mu=1,
        sigma=sigma,
        skip_eigs_threshold=skip_eigs_threshold,
        nb_eigs=nb_eigs_solve,
        nb_eigs_plot=nb_eigs_plot,
        domain_name=domain_name,
        backend_language='pyccel-gcc',
        plot_dir='./plots/eigen_maxell',
    )

    error = 0
    n_errs = min(len(ref_sigmas), len(eigenvalues))
    for k in range(n_errs):
        error += (eigenvalues[k] - ref_sigmas[k])**2
    error = np.sqrt(error)

    assert abs(error - 0.023395836648441557) < 1e-10


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

    diags, eigenvalues = hcurl_solve_eigen_pbm_nc(
        ncells=ncells, degree=degree,
        gamma_h=0,
        generalized_pbm=True,
        nu=0,
        mu=1,
        sigma=sigma,
        ref_sigmas=ref_sigmas,
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
        gamma_h=0,
        generalized_pbm=True,
        nu=0,
        mu=1,
        sigma=sigma,
        ref_sigmas=ref_sigmas,
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
