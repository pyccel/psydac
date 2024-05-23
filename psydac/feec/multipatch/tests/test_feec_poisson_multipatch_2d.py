import numpy as np

from psydac.feec.multipatch.examples.h1_source_pbms_conga_2d import solve_h1_source_pbm
from psydac.feec.multipatch.examples_nc.h1_source_pbms_nc import solve_h1_source_pbm_nc


def test_poisson_pretzel_f():

    source_type = 'manu_poisson_2'
    domain_name = 'pretzel_f'
    nc = 10
    deg = 2
    run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
    l2_error = solve_h1_source_pbm(
        nc=nc, deg=deg,
        eta=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_source=False,
        plot_dir='./plots/h1_tests_source_february/' + run_dir)

    assert abs(l2_error - 8.054935880166114e-05) < 1e-10


def test_poisson_pretzel_f_nc():

    source_type = 'manu_poisson_2'
    domain_name = 'pretzel_f'
    nc = np.array([20, 20, 20, 20, 20, 10, 10, 10, 10,
                  10, 10, 10, 10, 20, 20, 20, 10, 10])
    deg = 2
    run_dir = '{}_{}_nc={}_deg={}/'.format(domain_name, source_type, nc, deg)
    l2_error = solve_h1_source_pbm_nc(
        nc=nc, deg=deg,
        eta=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_source=False,
        plot_dir='./plots/h1_tests_source_february/' + run_dir)

    assert abs(l2_error - 4.6086851224995065e-05) < 1e-10
# ==============================================================================
# CLEAN UP SYMPY NAMESPACE
# ==============================================================================


def teardown_module():
    from sympy.core import cache
    cache.clear_cache()


def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
