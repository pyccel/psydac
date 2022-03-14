# coding: utf-8

import numpy as np

from psydac.feec.multipatch.examples.hcurl_source_pbms_conga_2d import solve_hcurl_source_pbm

def test_time_harmonic_maxwell_pretzel_f():
    nc,deg      = 10,2
    source_type = 'manu_maxwell'
    domain_name = 'pretzel_f'

    omega = np.sqrt(170) # source
    roundoff = 1e4
    eta = int(-omega**2 * roundoff)/roundoff

    l2_error = solve_hcurl_source_pbm(
        nc=nc, deg=deg,
        eta=eta,
        nu=0,
        mu=1, #1,
        domain_name=domain_name,
        source_type=source_type,
        backend_language='pyccel-gcc')

    assert abs(l2_error - 0.06246693595198972)<1e-10


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()
