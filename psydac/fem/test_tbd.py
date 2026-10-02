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
plot_2d((F1, F0s, F0m, 
         {'vals':vals1,  'xx':xx2, 'yy':yy2, 'spline_grid':F1.space,  'patch_boundaries':F1.space}, 
         {'vals':vals0s, 'xx':xx3, 'yy':yy3, 'spline_grid':F0s.space, 'patch_boundaries':F0s.space}, 
         {'vals':vals0m, 'xx':xx1, 'yy':yy1, 'spline_grid':F0m.space, 'patch_boundaries':F0m.space}), 
         layout=(2, 4), patch_boundaries_color='darkviolet', plot_spline_grid=(0, ), plot_patch_boundaries=(1, 2))
#