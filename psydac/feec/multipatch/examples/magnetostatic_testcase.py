from cProfile import run
import os
from unittest import case
import numpy as np
from psydac.feec.multipatch.examples.mixed_source_pbms_conga_2d import solve_magnetostatic_pbm
from psydac.feec.multipatch.utilities                   import time_count, FEM_sol_fn, get_run_dir, get_plot_dir, get_mat_dir, get_sol_dir

t_stamp_full = time_count()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
# test-case and numerical parameters:

bc_type = 'pseudo-vacuum' # 'metallic' # 

source_type = 'dipole_J'
source_proj = 'P_L2_wcurl_J'
assert source_proj in ['P_geom', 'P_L2', 'P_L2_wcurl_J']

domain_name = 'pretzel_f'
dim_harmonic_space = 3

nc = 16
deg = 4

if bc_type == 'metallic':
    case_dir = 'magnetostatic_metal'

elif bc_type == 'pseudo-vacuum':
    case_dir = 'magnetostatic_vacuum'

else:
    raise ValueError(bc_type)
    # domain_name = 'curved_L_shape'

# ref solution (if no exact solution)
ref_nc = 20
ref_deg = 6

#
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


# backend_language = 'numba'
backend_language='pyccel-gcc'

run_dir = get_run_dir(domain_name, source_type, nc, deg)
plot_dir = get_plot_dir(case_dir, run_dir)

# to save and load matrices
m_load_dir = get_mat_dir(domain_name, nc, deg)
# to save the FEM sol
sol_dir = get_sol_dir(case_dir, domain_name, nc, deg)
sol_filename = sol_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)
if not os.path.exists(sol_dir):
    os.makedirs(sol_dir)
# to load the ref FEM sol
sol_ref_dir = get_sol_dir(case_dir, domain_name, ref_nc, ref_deg)
sol_ref_filename = sol_ref_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
# calling ms solver

solve_magnetostatic_pbm(
    nc=nc, deg=deg,
    domain_name=domain_name,
    source_type=source_type,
    source_proj=source_proj,
    bc_type=bc_type,
    backend_language=backend_language,
    dim_harmonic_space=dim_harmonic_space,
    plot_source=True,
    plot_dir=plot_dir,
    # plot_dir='./plots/magnetostatic_runs/'+run_dir,
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