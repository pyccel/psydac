import os
import numpy as np
from psydac.feec.multipatch.examples.hcurl_source_pbms_conga_2d import solve_hcurl_source_pbm
from psydac.feec.multipatch.utilities                   import time_count, FEM_sol_fn

t_stamp_full = time_count()



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
#
# test-case and numerical parameters:

homogeneous = True # False # 

omega = np.sqrt(170) # source time pulsation
roundoff = 1e4
eta = int(-omega**2 * roundoff)/roundoff

# nc = 16
# deg = 4
# nc = 4
nc = 16
deg = 4

# ref solution
ref_nc = 20
ref_deg = 6

if homogeneous:
    case_dir = 'maxwell_hom'
    source_type = 'elliptic_J'
    domain_name = 'pretzel_f'

else:
    case_dir = 'maxwell_inhom'
    source_type = 'manu_maxwell_inhom'
    domain_name = 'pretzel_f'
    # domain_name = 'curved_L_shape'

# domain_name = 'annulus_4'
# source_proj='P_L2'
source_proj='P_geom' # geom proj

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

# to load the ref FEM sol
sol_ref_dir = './saved_solutions/'+case_dir+'/solutions_{}_nc={}_deg={}'.format(domain_name, ref_nc, ref_deg)
sol_ref_filename = sol_ref_dir+'/'+FEM_sol_fn(source_type=source_type, source_proj=source_proj)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
# calling solver for:
# 
# find u in H(curl), s.t.
#   A u = f             on \Omega
#   n x u = n x u_bc    on \partial \Omega
# with
#   A u := eta * u  +  mu * curl curl u  -  nu * grad div u

solve_hcurl_source_pbm(
    nc=nc, deg=deg,
    eta=eta,
    nu=0,
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