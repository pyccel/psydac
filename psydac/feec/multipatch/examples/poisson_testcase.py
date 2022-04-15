import os
import numpy as np
from psydac.feec.multipatch.examples.h1_source_pbms_conga_2d import solve_h1_source_pbm
from psydac.feec.multipatch.utilities                   import time_count, FEM_sol_fn

t_stamp_full = time_count()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
# test-case and numerical parameters:

homogeneous = False # True # 

nc = 16
deg = 4

if homogeneous:
    case_dir = 'poisson_hom'
    source_type = 'manu_poisson_elliptic'
    domain_name = 'pretzel_f'
    # domain_name = 'curved_L_shape'
else:
    case_dir = 'poisson_inhom'
    source_type = 'manu_poisson_sincos' # 'manu_poisson_2'
    domain_name = 'pretzel_f'
    # domain_name = 'curved_L_shape'
    # raise NotImplementedError

# source_proj='P_L2'
source_proj='P_geom' # geom proj (interpolation) of source: quicker but not standard

# ref solution (if no exact solution)
ref_nc = 20
ref_deg = 6

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


# backend_language = 'numba'
backend_language='pyccel-gcc'

run_dir = '{}_{}_nc={}_deg={}'.format(domain_name, source_type, nc, deg)
plot_dir = './plots/'+case_dir+'/'+run_dir
# to save and load matrices
m_load_dir = './saved_matrices/matrices_{}_nc={}_deg={}'.format(domain_name, nc, deg)
# to save the FEM sol
sol_dir = './saved_solutions/'+case_dir+'/solutions_{}_nc={}_deg={}'.format(domain_name, nc, deg)
sol_filename = sol_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)
if not os.path.exists(sol_dir):
    os.makedirs(sol_dir)

# to load the ref FEM sol (no need for now)
sol_ref_dir = './saved_solutions/'+case_dir+'/solutions_{}_nc={}_deg={}'.format(domain_name, ref_nc, ref_deg)
sol_ref_filename = sol_ref_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
# calling solver for:
# 
# find u in H1, s.t.
#       A u = f             on \Omega
#         u = u_bc          on \partial \Omega
# with
#       A u := eta * u  -  mu * div grad u

solve_h1_source_pbm(
    nc=nc, deg=deg,
    eta=0,
    mu=1,
    domain_name=domain_name,
    source_type=source_type,
    source_proj=source_proj,   
    backend_language=backend_language,
    plot_source=True,
    plot_dir=plot_dir,
    hide_plots=True,
    m_load_dir=m_load_dir,
    sol_filename=sol_filename,
    sol_ref_filename=sol_ref_filename,
    ref_nc=ref_nc,
    ref_deg=ref_deg,    
)

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

time_count(t_stamp_full, msg='full program')