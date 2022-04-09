import numpy as np
from psydac.feec.multipatch.examples.h1_source_pbms_conga_2d import solve_h1_source_pbm
from psydac.feec.multipatch.utilities                   import time_count

t_stamp_full = time_count()

homogeneous = True
domain_name = 'pretzel_f'
# domain_name = 'curved_L_shape'

nc = 4
deg = 2

if homogeneous:
    source_type = 'manu_poisson'
else:
    raise NotImplementedError

backend_language = 'numba'
# backend_language='pyccel-gcc'

# solve_h1_source_pbm(): solver for
# find u in H1, s.t.
#       A u = f             on \Omega
#         u = u_bc          on \partial \Omega
# with
#       A u := eta * u  -  mu * div grad u

run_dir = '{}_{}_nc={}_deg={}'.format(domain_name, source_type, nc, deg)
plot_dir = './plots/poisson_hom/'+run_dir
m_load_dir = './matrices_{}_nc={}_deg={}'.format(domain_name, nc, deg)
solve_h1_source_pbm(
    nc=nc, deg=deg,
    eta=0,
    mu=1,
    domain_name=domain_name,
    source_type=source_type,
    backend_language=backend_language,
    plot_source=True,
    plot_dir=plot_dir,
    hide_plots=True,
    m_load_dir=m_load_dir,
)

time_count(t_stamp_full, msg='full program')