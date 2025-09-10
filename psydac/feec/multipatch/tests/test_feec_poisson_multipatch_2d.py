import numpy as np

from psydac.feec.multipatch.examples.h1_source_pbms_conga_2d import solve_h1_source_pbm


def test_poisson_pretzel_f():

    source_type = 'manu_poisson_2'
    domain_name = 'pretzel_f'
    nc = 2
    deg = 2

    l2_error = solve_h1_source_pbm(
        nc=nc, deg=deg,
        eta=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_dir=None)

    assert abs(l2_error) < 8


def test_poisson_pretzel_f_nc():

    source_type = 'manu_poisson_2'
    domain_name = 'pretzel_f'
    nc = np.array([4, 4, 4, 4, 4, 2, 2, 2, 2,
                   2, 2, 2, 2, 4, 4, 4, 2, 2])
    deg = 2

    l2_error = solve_h1_source_pbm(
        nc=nc, deg=deg,
        eta=0,
        mu=1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc',
        plot_dir=None)

    assert abs(l2_error) < 5


# ==============================================================================
# CLEAN UP SYMPY NAMESPACE
# ==============================================================================
def teardown_module():
    from sympy.core import cache
    cache.clear_cache()


def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
