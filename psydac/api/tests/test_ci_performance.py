# 10 slowest tests
# from psydac.api.tests.test_2d_complex import (
    # test_maxwell_2d_2_patch_dirichlet_2,
# )
from psydac.api.tests.test_2d_multipatch_mapping_maxwell import (
    # test_maxwell_2d_2_patch_dirichlet_0,
    test_maxwell_2d_2_patch_dirichlet_1,
    )

# from psydac.api.tests.test_api_feec_3d import (
    # test_maxwell_3d_1,
    # test_maxwell_3d_2_mult,
# )

from psydac.api.tests.test_2d_navier_stokes import (
    test_navier_stokes_2d,
)

from psydac.feec.multipatch.tests.test_feec_poisson_multipatch_2d import (
    test_poisson_pretzel_f,
    test_poisson_pretzel_f_nc,
)
from psydac.feec.multipatch.tests.test_feec_maxwell_multipatch_2d import (
    test_time_harmonic_maxwell_pretzel_f,
    test_time_harmonic_maxwell_pretzel_f_nc,)