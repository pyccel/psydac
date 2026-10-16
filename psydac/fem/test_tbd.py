import numpy as np
from sympy import sin, cos, pi, Tuple, exp
import matplotlib.pyplot as plt

from sympde.topology import Square, Domain, PolarMapping, ScalarFunctionSpace, elements_of, VectorFunctionSpace
from sympde.expr import BilinearForm, LinearForm, integral
from sympde.calculus import inner

from psydac.api.discretization import discretize
from psydac.fem.plotting_utilities2 import plot_2d
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.linalg.solvers import inverse
from psydac.fem.basic import FemField

# This file can be used for testing during development
# Eventually it should be made into a proper showcase / test file



# The two functions below grant access to 2 scalar-valued FemFields (singlepatch and multipatch) and 1 vector-valued FemField.

backend = PSYDAC_BACKEND_GPYCCEL

A       = Square('A', bounds1=(0.5, 1.), bounds2=(0, np.pi/2))
B       = Square('B', bounds1=(0.5, 1.), bounds2=(np.pi/2, np.pi))
C       = Square('C', bounds1=(0.5, 1.), bounds2=(np.pi, 3*np.pi/2))
D       = Square('D', bounds1=(0.5, 1.), bounds2=(3*np.pi/2, 2*np.pi))

mapping = PolarMapping('M', 2, c1=0., c2=0., rmin=0., rmax=1.)

D1      = mapping(A)
D2      = mapping(B)
D3      = mapping(C)
D4      = mapping(D)

connectivity1 = [((0,1,1),(1,1,-1))]
connectivity2 = [((0,1,1),(1,1,-1)), ((1,1,1),(2,1,-1)), ((2,1,1),(3,1,-1)), ((3,1,1),(0,1,-1))]

patches1 = [D1,D2]
patches2 = [D1,D2,D3,D4]

domain1 = Domain.join(patches1, connectivity1, 'domain')
domain2 = Domain.join(patches2, connectivity2, 'domain')
domain3 = D1

def get_scalar_FemField(singlepatch=True):

    ncells = [8, 8]
    degree = [3, 3]

    x, y = domain1.coordinates

    if singlepatch:
        f0 = sin(2*pi*x)
        domain = domain3
    else:
        f0 = x**2 + y**2
        domain = domain1

    domain_h = discretize(domain, ncells=ncells)

    V = ScalarFunctionSpace('V', domain, kind='h1')
    Vh = discretize(V, domain_h, degree=degree)

    u, v = elements_of(V, names='u, v')
    m = BilinearForm((u, v), integral(domain1, u*v))
    M = discretize(m, domain_h, (Vh, Vh), backend=backend).assemble()

    l = LinearForm(v, integral(domain1, f0*v))
    L = discretize(l, domain_h, Vh, backend=backend).assemble()

    M_inv = inverse(M, 'CG', maxiter=1000, tol=1e-10)

    f0_coeffs = M_inv @ L
    F0 = FemField(Vh, f0_coeffs)

    return F0

def get_vector_FemField():

    ncells = [8, 8]
    degree = [3, 3]

    x, y = domain2.coordinates

    r2 = x**2 + y**2
    f1_1 = (1/r2) * (-y)
    f1_2 = (1/r2) * ( x)
    f1 = Tuple(f1_1, f1_2)
    domain = domain2

    domain_h = discretize(domain, ncells=ncells)

    V = VectorFunctionSpace('V', domain, kind='h1')
    Vh = discretize(V, domain_h, degree=degree)

    u, v = elements_of(V, names='u, v')
    m = BilinearForm((u, v), integral(domain, inner(u, v)))
    M = discretize(m, domain_h, (Vh, Vh), backend=backend).assemble()

    l = LinearForm(v, integral(domain, inner(f1, v)))
    L = discretize(l, domain_h, Vh, backend=backend).assemble()

    M_inv = inverse(M, 'CG', maxiter=1000, tol=1e-10)

    f1_coeffs = M_inv @ L
    F1 = FemField(Vh, f1_coeffs)

    return F1



#####
##### 
#####



F0s = get_scalar_FemField()
F0m = get_scalar_FemField(singlepatch=False)
F1  = get_vector_FemField()



funs = (F0s, F0m, F1, {'fem_field':F0s, 'magnitude':True}, {'fem_field':F0m, 'magnitude':True}, {'fem_field':F1, 'magnitude':True})
#plot_2d(funs, layout=(3, 4), plot_patch_boundaries=True)

funs = (F1, {'fem_field':F1, 'plot_type':'vector_field'}, 
        {'fem_field':F1, 'magnitude':True}, {'fem_field':F1, 'plot_type':'vector_field', 'contourf':True},
        {'fem_field':F1, 'components':'x'}, {'fem_field':F1, 'components':'y'}, {'fem_field':F1, 'components':False, 'magnitude':True, 'plot_type':'surface_plot'})
#plot_2d(funs, plot_patch_boundaries=True)

from psydac.fem.plotting_utilities2 import get_plotting_grid, get_grid_vals
N_vis = (100, 100)
etas1, xx1, yy1 = get_plotting_grid(domain1.mappings, N_vis)
etas2, xx2, yy2 = get_plotting_grid(domain2.mappings, N_vis)
etas3, xx3, yy3 = get_plotting_grid(domain3.mappings, N_vis)

vals0s = get_grid_vals(F0s, etas3, list([M for M in domain3.mappings.values()]), space_kind='h1')
vals0m = get_grid_vals(F0m, etas1, list([M for M in domain1.mappings.values()]), space_kind='h1')
vals1  = get_grid_vals(F1,  etas2, list([M for M in domain2.mappings.values()]), space_kind='h1')

#plot_2d(vals0s, xx=xx3, yy=yy3, plot_spline_grid=True, spline_grid=F0s.space)
#plot_2d(vals0s, xx=xx3, yy=yy3,                        spline_grid=F0s.space)
#plot_2d(F0s,                    verbose=True)
#plot_2d(vals0m, xx=xx1, yy=yy1, plot_spline_grid=(0,), spline_grid=F0m.space)
#plot_2d(vals0m, xx=xx1, yy=yy1, plot_spline_grid=(1,), spline_grid=F0m.space)
#plot_2d(vals0m, xx=xx1, yy=yy1, plot_spline_grid=True, spline_grid=F0m.space)
#plot_2d(vals0m, xx=xx1, yy=yy1,                        spline_grid=F0m.space)
#plot_2d(F0m,                    verbose=True)
#plot_2d(vals1,  xx=xx2, yy=yy2, verbose=True)
#plot_2d(F1,                     verbose=True)
#print('-----')
#plot_2d((vals0s, vals0s), xx=xx3, yy=yy3, verbose=True)
#plot_2d((F0s, F0s),                    verbose=True)
#plot_2d((vals0m, vals0m), xx=xx1, yy=yy1, verbose=True)
#plot_2d((F0m, F0m),                    verbose=True)
#plot_2d((vals1, vals1),  xx=xx2, yy=yy2, verbose=True)
#plot_2d((F1, F0s, F0m, 
#         {'vals':vals1,  'xx':xx2, 'yy':yy2, 'spline_grid':F1.space,  'patch_boundaries':F1.space}, 
#         {'vals':vals0s, 'xx':xx3, 'yy':yy3, 'spline_grid':F0s.space, 'patch_boundaries':F0s.space}, 
#         {'vals':vals0m, 'xx':xx1, 'yy':yy1, 'spline_grid':F0m.space, 'patch_boundaries':F0m.space}), 
#         layout=(2, 4), patch_boundaries_color='darkviolet', plot_spline_grid=(0, ), plot_patch_boundaries=(1, 2))
#



# This part of the code (sort of) shows how to read the coefficient data stored in generate_plot_data_13Okt/field_info.h5

from psydac.linalg.block import BlockVector
import h5py
from sympde.topology import Mapping, Derham
from anisotropic_maxwell_to_be_deleted import get_mapping

def get_E_coefficients_from_data(V1_vs, data):
    V1_0 = V1_vs.spaces[0]
    V1_1 = V1_vs.spaces[1]

    v0 = V1_0.zeros()
    v1 = V1_1.zeros()

    pads0_0, pads0_1 = V1_0.pads
    pads1_0, pads1_1 = V1_1.pads

    v0._data[pads0_0:-pads0_0, pads0_1:-pads0_1] = data[0]
    v1._data[pads1_0:-pads1_0, pads1_1:-pads1_1] = data[1]

    coeffs = BlockVector(V1_vs, (v0, v1))

    return coeffs

def get_B_coefficients_from_data(V2_vs, data):

    v = V2_vs.zeros()

    pads0_0, pads0_1 = V2_vs.pads

    v._data[pads0_0:-pads0_0, pads0_1:-pads0_1] = data

    coeffs = v

    return coeffs

file_path = 'generate_plot_data_13Okt/field_info.h5' # this is where I stored 200 snapshots (200 FemFields, 100 of each E (vector-valued) and B (scalar-valued))

# -----
# The following code does what we want to avoid in the future: Load all coefficients.
# Better: Load coefficients, update the Figure, load the next coefficients

E_coeffs_list        = []
B_coeffs_list        = []
simulation_time_list = []
time_step_list       = []

with h5py.File(file_path, 'r') as f:
    for i, snapshot in enumerate(f.values()):
        simulation_time = snapshot.attrs['t']   # simulation from t=0 to t=1 (not seconds, artificial time unit)
        time_step       = snapshot.attrs['ts']  # 20000 time steps, but coefficients were only saved every 200 time steps (0, 200, 400, ..., 20000)

        domain_keys = list(snapshot.keys())     # snapshot could correspond to coefficients defining functions on various domains
                                                # here: only 1 domain 'WHA(S)' (Wiggly Half Annulus - mapping defined on a Square)
        domain_key  = domain_keys[0]            # = 'WHA(S)'

        space_keys  = list(snapshot[domain_key].keys()) # snapshots corresponding to coefficients defining functions defined on WHA(S) could 
                                                        # correspond to functions defined on different function spaces (H1, Hcurl, Hdiv, L2)
                                                        # here: Indeed, I saved coefficients of 2 kinds of functions:
                                                        #       E in V1 (Hcurl - vector-valued) and B in V2 (L2 - scalar-valued)
        V1_0_key, V1_1_key, V2_key = space_keys         # = 'V1[0]', 'V1[1]', 'V2'

        # snapshot[domain_key][V1_0_key] is again a dictionary, as we could have potentially stored various function coefficients
        # corresponding to functions defined on V1 (both components V1[0] and V1[1]). Here, this dict only contains 1 value: 'E[0]'
        # similarly snapshot[domain_key][V1_1_key] and snapshot[domain_key][V2_key] are dicts containing only 1 key each: 'E[1]' and 'B' respectively.
        E_0_key = list(snapshot[domain_key][V1_0_key].keys())[0] # 'E[0]'
        E_1_key = list(snapshot[domain_key][V1_1_key].keys())[0] # 'E[1]'
        B_key   = list(snapshot[domain_key][V2_key].keys())  [0] # 'B'
        # finally, the actually coefficients can be obtained via [()]:
        E_0_coeffs = snapshot[domain_key][V1_0_key][E_0_key][()]
        E_1_coeffs = snapshot[domain_key][V1_1_key][E_1_key][()]
        E_coeffs   = [E_0_coeffs, E_1_coeffs]
        B_coeffs   = snapshot[domain_key][V2_key]  [B_key]  [()]

        simulation_time_list.append(simulation_time)
        time_step_list.append(time_step)
        E_coeffs_list.append(E_coeffs)
        B_coeffs_list.append(B_coeffs)
# -----

# -----
# What this should actually look like (approximately)

# Given: file_path (to an .h5 file), domain_key, space_key(s), field_key(s)
#with h5py.File(file_path, 'r') as f:
#    for i, snapshot in enumerate(f.values()):
#        #simulation_time = snapshot.attrs['t']  # might be useful!
#        #time_step       = snapshot.attrs['ts'] # might be useful!
#
#        data = []
#        for (space_key, field_key) in zip(space_keys, field_keys):
#            data.append(snapshot[domain_key][space_key][field_key][()])
#        coeffs = coeffs_from_data(data)
#        FemField = FemField_from_coeffs(FemSpace, coeffs)
#        update_plot(FemField, ...)
# -----

# -----
# In order to obtain functions from these coefficients, we need to assign them a FemSpace.
# Compare with code in anisotropic_maxwell_to_be_deleted.py

# compare lines 397-399
R0 = 2
R1 = 4

class WigglyHalfAnnulus(Mapping):
    _expressions = {'x': '(R0 + (R1-R0)*x1 + (R0/2)*cos(4*pi*x2)) * cos(pi*x2)',
                    'y': '(R0 + (R1-R0)*x1 + (R0/2)*cos(4*pi*x2)) * sin(pi*x2)'}
    _ldim = 2
    _pdim = 2

# compare lines 116-121
logical_domain = Square('S', bounds1=(0, 1), bounds2=(0, 1)) # Square('S', bounds1=(r, R), bounds2=(0,np.pi))
#mapping = WigglyHalfAnnulus('WHA', R0=R0, R1=R1) # get_mapping('WigglyHalfAnnulus', R0=R0, R1=R1)
mapping = get_mapping('WigglyHalfAnnulus', R0=R0, R1=R1)
domain = mapping(logical_domain)

# spacial discretization paramaters
ncells = [32, 256]
degree = [3, 3]

sequence       = ['h1', 'hcurl', 'l2']
derham         = Derham(domain, sequence=sequence)

V0, V1, V2 = derham.spaces

domain_h = discretize(domain, ncells=ncells)
derham_h = discretize(derham, domain_h, degree=degree)

# These are our FemSpaces
V0h, V1h, V2h = derham_h.spaces
# -----

# -----
# Now we can plot!
E_FemField_snapshots_to_plot = [0, 20, 40, 60, 80, 100]
E_FemFields_to_plot = []
for ss in E_FemField_snapshots_to_plot:
    coeffs = E_coeffs_list[ss]
    psydac_coeffs = get_E_coefficients_from_data(V1h.coeff_space, coeffs)
    E_ss = FemField(V1h, psydac_coeffs)
    E_FemFields_to_plot.append(E_ss)
# title = r'$\|\boldsymbol{E}\|$' # currently doesn't work, please fix
titles = [f'Time Step {time_step_list[i]} : t = {simulation_time_list[i]:.3g}' for i in E_FemField_snapshots_to_plot]
plot_2d(E_FemFields_to_plot, titles=titles, magnitude=True, components=False, cmap='plasma', N_vis=(100, 1000)) # , title=title)

B_FemField_snapshots_to_plot = [0, 20, 40, 60, 80, 100]
B_FemFields_to_plot = []
for ss in B_FemField_snapshots_to_plot:
    coeffs = B_coeffs_list[ss]
    psydac_coeffs = get_B_coefficients_from_data(V2h.coeff_space, coeffs)
    B_ss = FemField(V2h, psydac_coeffs)
    B_FemFields_to_plot.append(B_ss)
# title = r'$B$' # currently doesn't work, please fix
titles = [f'Time Step {time_step_list[i]} : t = {simulation_time_list[i]:.3g}' for i in B_FemField_snapshots_to_plot]
plot_2d(B_FemFields_to_plot, titles=titles, cmap='plasma', N_vis=(100, 1000)) # , title=title)
