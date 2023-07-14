from psydac.feec.multipatch.examples.h1_source_pbms_conga_2d import solve_h1_source_pbm

def test_poisson_pretzel_f():

    source_type = 'manu_poisson_2'
    domain_name = 'pretzel_f'
    nc  = 10
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
                plot_dir='./plots/h1_tests_source_february/'+run_dir)
    print(l2_error)
    assert abs(l2_error-8.054935880021907e-05)<1e-10

#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy.core import cache
    cache.clear_cache()

def teardown_function():
    from sympy.core import cache
    cache.clear_cache()

if __name__ == '__main__':
    test_poisson_pretzel_f()
