#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import os
import shutil
import time
from   pathlib  import Path
from   datetime import datetime

from   mpi4py  import  MPI
import numpy   as      np
#import matplotlib.pyplot as plt
from   sympy   import  sin

from   sympde.calculus             import dot, cross, grad, curl
from   sympde.expr                 import BilinearForm, integral
from   sympde.topology             import element_of, elements_of, Cube, Mapping, ScalarFunctionSpace, Domain, Derham

from   psydac.api.discretization   import discretize
from   psydac.api.settings         import PSYDAC_BACKEND_GPYCCEL
from   psydac.cad.geometry         import Geometry
from   psydac.fem.basic            import FemField
from   psydac.mapping.discrete     import SplineMapping

datetime_md = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
datetime_file = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')

"""
Executing this file will add new tables to the matrix_assembly_speed_log.md file in which we keep track of assembly and discretization speed.
Estimated runtime: ~4 min.

"""

comm = MPI.COMM_WORLD
backend = PSYDAC_BACKEND_GPYCCEL
mpi_rank = comm.rank

if mpi_rank == 0:
    print('Expected runtime: 4 min.')
    print()

class SquareTorus(Mapping):

    _expressions = {'x': 'x1 * cos(x2)',
                    'y': 'x1 * sin(x2)',
                    'z': 'x3'}
    
    _ldim        = 3
    _pdim        = 3

def make_square_torus_geometry_3d(ncells, degree, comm=None):

    if comm is not None:
        mpi_rank = comm.Get_rank()
    else:
        mpi_rank = 0

    if (ncells[0] == ncells[1]) and (ncells[0] == ncells[2]):
        nc = f'{ncells[0]}'
    else:
        nc = f'{ncells[0]}_{ncells[1]}_{ncells[2]}'
    
    if (degree[0] == degree[1]) and (degree[0] == degree[2]):
        de = f'{degree[0]}'
    else:
        de = f'{degree[0]}_{degree[1]}_{degree[2]}'

    name = f'st_3d_nc_{nc}_d_{de}'

    r = 0.5
    R = 1.
    logical_domain = Cube('C', bounds1=(r, R), bounds2=(0, 2*np.pi), bounds3=(0, 1))
    domain_h = discretize(logical_domain, ncells=ncells, comm=comm)

    V = ScalarFunctionSpace('V', logical_domain)
    V_h = discretize(V, domain_h, degree=degree)

    mapping = SquareTorus('S')
    map_discrete = SplineMapping.from_mapping(V_h, mapping.get_callable_mapping())

    geometry = Geometry.from_discrete_mapping(map_discrete, comm=comm)

    if mpi_rank == 0:
        if not os.path.isdir('geometry'):
            os.makedirs('geometry')
            os.makedirs('geometry/files')
        else:
            if not os.path.isdir('geometry/files'):
                os.makedirs('geometry/files')

    geometry.export(f'geometry/files/{name}.h5')

    return f'geometry/files/{name}.h5'

# ---------- 1 ----------
#t0_1_glob = time.time()
#
#ncells          = [32, 32, 32]
#degree_list     = [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
#degree_list2    = [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
#periodic        = [False, False, False]
#
##mapping = HalfHollowTorusMapping3D('M', R=2, r=1)
#mapping = SquareTorus('S')
##logical_domain = Cube('C', bounds1=(0,1), bounds2=(0,1), bounds3=(0,1))
#r = 0.5
#R = 1.
#logical_domain = Cube('C', bounds1=(r, R), bounds2=(0, 2*np.pi), bounds3=(0, 1))
#
#domain = mapping(logical_domain)
#derham = Derham(domain)
#plot_domain(domain, draw=True, isolines=True)
#
#ass_time_H1_old = [[] for _ in degree_list]
#ass_time_H1_new = [[] for _ in degree_list]
#
#ass_time_Hcurl_old = [[] for _ in degree_list2]
#ass_time_Hcurl_new = [[] for _ in degree_list2]
#
#for i, degree in enumerate(degree_list):
#
#    domain_h = discretize(domain, ncells=ncells, periodic=periodic, comm=comm)
#    derham_h = discretize(derham, domain_h, degree=degree)
#
#    V0  = derham.V0
#    V0h = derham_h.V0
#
#    u, v = elements_of(V0, names='u, v')
#
#    a = BilinearForm((u, v), integral(domain, u*v))
#
#    a_h = discretize(a, domain_h, (V0h, V0h), backend=backend, sum_factorization=False)
#    
#    t0_ao = time.time()
#    M_o = a_h.assemble()
#    t1_ao = time.time()
#
#    a_h = discretize(a, domain_h, (V0h, V0h), backend=backend)
#    
#    t0_an = time.time()
#    M_n = a_h.assemble()
#    t1_an = time.time()
#
#    ass_time_H1_old[i].append(t1_ao - t0_ao)
#    ass_time_H1_new[i].append(t1_an - t0_an)
#
#    if degree != [5, 5, 5]:
#
#        V1  = derham.V1
#        V1h = derham_h.V1
#
#        u, v = elements_of(V1, names='u, v')
#
#        a = BilinearForm((u, v), integral(domain, dot(u, v)))
#
#        a_h = discretize(a, domain_h, (V1h, V1h), backend=backend, sum_factorization=False)
#        
#        t0_ao = time.time()
#        M_o = a_h.assemble()
#        t1_ao = time.time()
#
#        a_h = discretize(a, domain_h, (V1h, V1h), backend=backend)
#        
#        t0_an = time.time()
#        M_n = a_h.assemble()
#        t1_an = time.time()
#
#        ass_time_Hcurl_old[i].append(t1_ao - t0_ao)
#        ass_time_Hcurl_new[i].append(t1_an - t0_an)
#
#d = [degree[0] for degree in degree_list]
#d2 = [degree[0] for degree in degree_list2]
#
#ass_time_H1_old_d = [ass_time_H1_old[i][0] for i, _ in enumerate(degree_list)]
#ass_time_H1_new_d = [ass_time_H1_new[i][0] for i, _ in enumerate(degree_list)]
#
#ass_time_Hcurl_old_d = [ass_time_Hcurl_old[i][0] for i, _ in enumerate(degree_list2)]
#ass_time_Hcurl_new_d = [ass_time_Hcurl_new[i][0] for i, _ in enumerate(degree_list2)]
#
#if mpi_rank == 0:
#    plt.plot(d, ass_time_H1_old_d, '--.', label=f'old Algorithm')
#    plt.plot(d, ass_time_H1_new_d, '--.', label=f'new Algorithm')
#    plt.title(r'Assembly times for the $H^1(\Omega)$ mass matrix')
#    plt.legend()
#    plt.yscale('log')
#    plt.ylabel('Wallclock Time [s]')
#    plt.xlabel('d - [d, d, d] Bspline degrees')
#    plt.xticks(d)
#    #plt.savefig(f'figures/H1_{datetime_file}.png')
#    plt.show()
#    plt.clf()
#
#    plt.plot(d2, ass_time_Hcurl_old_d, '--.', label=f'old Algorithm')
#    plt.plot(d2, ass_time_Hcurl_new_d, '--.', label=f'new Algorithm')
#    plt.title(r'Assembly times for the $H(curl;\Omega)$ mass matrix')
#    plt.legend()
#    plt.yscale('log')
#    plt.ylabel('Wallclock Time [s]')
#    plt.xlabel('d - [d, d, d] Bspline degrees')
#    plt.xticks(d2)
#    #plt.savefig(f'figures/Hcurl_{datetime_file}.png')
#    plt.show()
#
#t1_1_glob = time.time()
#if mpi_rank == 0:
#    print(f'Part 1 out of 3 done after {(t1_1_glob-t0_1_glob)/60:.2g}min')
# -----------------------

# ---------- 2 ----------
t0_2_glob = time.time()

ncells      = [32, 32, 32]
degree      = [3, 3, 3]
periodic    = [False, False, False]

r = 0.5
R = 1.
logical_domain = Cube('C', bounds1=(r, R), bounds2=(0, 2*np.pi), bounds3=(0, 1))
logical_derham = Derham(logical_domain)

logical_domain_h = discretize(logical_domain, ncells=ncells, periodic=periodic, comm=comm)
logical_derham_h = discretize(logical_derham, logical_domain_h, degree=degree)

filename = make_square_torus_geometry_3d(ncells, degree, comm=comm)

bspline_domain = Domain.from_file(filename)
bspline_derham = Derham(bspline_domain)

bspline_domain_h = discretize(bspline_domain, filename=filename, comm=comm)
bspline_derham_h = discretize(bspline_derham, bspline_domain_h, degree=bspline_domain.mapping.get_callable_mapping().space.degree)

mapping = SquareTorus('S')

analytical_domain = mapping(logical_domain)
analytical_derham = Derham(analytical_domain)

analytical_domain_h = discretize(analytical_domain, ncells=ncells, periodic=periodic, comm=comm)
analytical_derham_h = discretize(analytical_derham, analytical_domain_h, degree=degree)

ax, ay, az         = analytical_domain.coordinates
agamma           = ax*ay*az + sin(ax*ay+az)**2

# 2.1
dom     = logical_domain
domh    = logical_domain_h
V       = logical_derham.V1
Vh      = logical_derham_h.V1

F = element_of(V, name='F')
f = Vh.coeff_space.zeros()
f[0]._data = np.ones(f[0]._data.shape)
f[1]._data = np.ones(f[1]._data.shape)
f[2]._data = np.ones(f[2]._data.shape)
f_field = FemField(Vh, f)

u, v    = elements_of(V, names='u, v')
a = BilinearForm((u, v), integral(dom, dot(cross(F, u), cross(F, v))))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_21 = round(t1-t0, 3)
t0 = time.time()
M  = ah.assemble(F=f_field)
t1 = time.time()
old_ass_21 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_21 = round(t1-t0, 3)
t0 = time.time()
M  = ah.assemble(F=f_field)
t1 = time.time()
new_ass_21 = round(t1-t0, 3)

# 2.2
dom     = analytical_domain
domh    = analytical_domain_h
V       = analytical_derham.V2
Vh      = analytical_derham_h.V2

u, v    = elements_of(V, names='u, v')
a = BilinearForm((u, v), integral(dom, dot(u, v)*agamma))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_22 = round(t1-t0, 3)
t0 = time.time()
M  = ah.assemble()
t1 = time.time()
old_ass_22 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_22 = round(t1-t0, 3)
t0 = time.time()
M  = ah.assemble()
t1 = time.time()
new_ass_22 = round(t1-t0, 3)

# 2.3
dom     = bspline_domain
domh    = bspline_domain_h
V       = bspline_derham.V1
Vh      = bspline_derham_h.V1

u, v    = elements_of(V, names='u, v')
a = BilinearForm((u, v), integral(dom, dot(curl(u), curl(v))))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_23 = round(t1-t0, 3)
t0 = time.time()
M  = ah.assemble()
t1 = time.time()
old_ass_23 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_23 = round(t1-t0, 3)
t0 = time.time()
M  = ah.assemble()
t1 = time.time()
new_ass_23 = round(t1-t0, 3)

t1_2_glob = time.time()
if mpi_rank == 0:
    print(f'Part 2 out of 3 done after {(t1_2_glob-t0_2_glob)/60:.2g}min')
# -----------------------

# ---------- 3 ----------
t0_3_glob = time.time()

ncells      = [16, 8, 32]
degree      = [2, 4, 3]
periodic    = [False, True, False]

r = 0.5
R = 1.
logical_domain = Cube('C', bounds1=(r, R), bounds2=(0, 2*np.pi), bounds3=(0, 1))
logical_derham = Derham(logical_domain)

logical_domain_h = discretize(logical_domain, ncells=ncells, periodic=periodic, comm=comm)
logical_derham_h = discretize(logical_derham, logical_domain_h, degree=degree)

filename = make_square_torus_geometry_3d(ncells, degree, comm=comm)

bspline_domain = Domain.from_file(filename)
bspline_derham = Derham(bspline_domain)

bspline_domain_h = discretize(bspline_domain, filename=filename, comm=comm)
bspline_derham_h = discretize(bspline_derham, bspline_domain_h, degree=bspline_domain.mapping.get_callable_mapping().space.degree)

mapping = SquareTorus('S')

analytical_domain = mapping(logical_domain)
analytical_derham = Derham(analytical_domain)

analytical_domain_h = discretize(analytical_domain, ncells=ncells, periodic=periodic, comm=comm)
analytical_derham_h = discretize(analytical_derham, analytical_domain_h, degree=degree)

ax, ay, az         = analytical_domain.coordinates
agamma           = ax*ay*az + sin(ax*ay+az)**2

# 3.1.1
dom     = logical_domain
domh    = logical_domain_h
V       = logical_derham.V0
Vh      = logical_derham_h.V0

u, v    = elements_of(V, names='u, v')
a       = BilinearForm((u, v), integral(dom, dot(grad(u), grad(v))))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_311 = round(t1-t0, 3)
t0 = time.time()
M  = ah.assemble()
t1 = time.time()
old_ass_311 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_311 = round(t1-t0, 3)
t0 = time.time()
M  = ah.assemble()
t1 = time.time()
new_ass_311 = round(t1-t0, 3)

# 3.1.2
dom     = analytical_domain
domh    = analytical_domain_h
V       = analytical_derham.V0
Vh      = analytical_derham_h.V0

u, v    = elements_of(V, names='u, v')
a       = BilinearForm((u, v), integral(dom, dot(grad(u), grad(v))))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_312 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
old_ass_312 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_312 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
new_ass_312 = round(t1-t0, 3)

# 3.1.3
dom     = bspline_domain
domh    = bspline_domain_h
V       = bspline_derham.V0
Vh      = bspline_derham_h.V0

u, v    = elements_of(V, names='u, v')
a       = BilinearForm((u, v), integral(dom, dot(grad(u), grad(v))))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_313 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
old_ass_313 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_313 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
new_ass_313 = round(t1-t0, 3)

# 3.2
dom     = analytical_domain
domh    = analytical_domain_h
V       = ScalarFunctionSpace('V', analytical_domain)
Vh      = discretize(V, analytical_domain_h, degree=degree)

u, v    = elements_of(V, names='u, v')
a       = BilinearForm((u, v), integral(dom, dot(grad(u), grad(v))))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_32 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
old_ass_32 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_32 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
new_ass_32 = round(t1-t0, 3)

# 3.3
dom     = analytical_domain
domh    = analytical_domain_h
V       = ScalarFunctionSpace('V', analytical_domain)
Vh      = discretize(V, analytical_domain_h, degree=degree)
W       = analytical_derham.V0
Wh      = analytical_derham_h.V0

u       = element_of(V, name='u')
v       = element_of(W, name='v')
a       = BilinearForm((u, v), integral(dom, dot(grad(u), grad(v))))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_33 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
old_ass_33 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_33 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
new_ass_33 = round(t1-t0, 3)

# ------------------
dom     = analytical_domain
domh    = analytical_domain_h
V       = analytical_derham.V0
Vh      = analytical_derham_h.V0

u, v    = elements_of(V, names='u, v')
# ------------------

# 3.4
a       = BilinearForm((u, v), integral(dom, dot(grad(u), grad(v)) * agamma))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_34 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
old_ass_34 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_34 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble()
t1 = time.time()
new_ass_34 = round(t1-t0, 3)

# 3.5
F = element_of(V, name='F')
f = Vh.coeff_space.zeros()
f._data = np.ones(f._data.shape)
f_field = FemField(Vh, f)

a       = BilinearForm((u, v), integral(dom, dot(grad(u), grad(v)) * F))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_35 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble(F=f_field)
t1 = time.time()
old_ass_35 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_35 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble(F=f_field)
t1 = time.time()
new_ass_35 = round(t1-t0, 3)

# 3.6
mult                = [1, 3, 2]
analytical_derham_h = discretize(analytical_derham, analytical_domain_h, degree=degree, multiplicity=mult)
Vh                  = analytical_derham_h.V0

a       = BilinearForm((u, v), integral(dom, dot(grad(u), grad(v)) * F))

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend, sum_factorization=False)
t1 = time.time()
old_disc_36 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble(F=f_field)
t1 = time.time()
old_ass_36 = round(t1-t0, 3)

t0 = time.time()
ah = discretize(a, domh, (Vh, Vh), backend=backend)
t1 = time.time()
new_disc_36 = round(t1-t0, 3)
t0 = time.time()
M = ah.assemble(F=f_field)
t1 = time.time()
new_ass_36 = round(t1-t0, 3)

t1_3_glob = time.time()
if mpi_rank == 0:
    print(f'Part 3 out of 3 done after {(t1_3_glob-t0_3_glob)/60:.2g}min')
# -----------------------

template = '''| Test case | old assembly | new assembly | old discretization | new discretization |
| --- | --- | --- | --- | --- |
| 2.1 | {old_ass_21} | {new_ass_21} | {old_disc_21} | {new_disc_21} |
| 2.2 | {old_ass_22} | {new_ass_22} | {old_disc_22} | {new_disc_22} |
| 2.3 | {old_ass_23} | {new_ass_23} | {old_disc_23} | {new_disc_23} |
| 3.1.1 | {old_ass_311} | {new_ass_311} | {old_disc_311} | {new_disc_311} |
| 3.1.2 | {old_ass_312} | {new_ass_312} | {old_disc_312} | {new_disc_312} |
| 3.1.3 | {old_ass_313} | {new_ass_313} | {old_disc_313} | {new_disc_313} |
| 3.2 | {old_ass_32} | {new_ass_32} | {old_disc_32} | {new_disc_32} |
| 3.3 | {old_ass_33} | {new_ass_33} | {old_disc_33} | {new_disc_33} |
| 3.4 | {old_ass_34} | {new_ass_34} | {old_disc_34} | {new_disc_34} |
| 3.5 | {old_ass_35} | {new_ass_35} | {old_disc_35} | {new_disc_35} |
| 3.6 | {old_ass_36} | {new_ass_36} | {old_disc_36} | {new_disc_36} |'''

txt = ''
txt += f'{datetime_md}\n'
txt += f'----------\n\n'
#txt += f'![](tests/figures/H1_{datetime_file}.png)\n'
#txt += f'![](tests/figures/Hcurl_{datetime_file}.png)\n\n'
txt += template.format(old_ass_21=old_ass_21, new_ass_21=new_ass_21, old_disc_21=old_disc_21, new_disc_21=new_disc_21,
                       old_ass_22=old_ass_22, new_ass_22=new_ass_22, old_disc_22=old_disc_22, new_disc_22=new_disc_22,
                       old_ass_23=old_ass_23, new_ass_23=new_ass_23, old_disc_23=old_disc_23, new_disc_23=new_disc_23,
                       old_ass_311=old_ass_311, new_ass_311=new_ass_311, old_disc_311=old_disc_311, new_disc_311=new_disc_311,
                       old_ass_312=old_ass_312, new_ass_312=new_ass_312, old_disc_312=old_disc_312, new_disc_312=new_disc_312,
                       old_ass_313=old_ass_313, new_ass_313=new_ass_313, old_disc_313=old_disc_313, new_disc_313=new_disc_313,
                       old_ass_32=old_ass_32, new_ass_32=new_ass_32, old_disc_32=old_disc_32, new_disc_32=new_disc_32,
                       old_ass_33=old_ass_33, new_ass_33=new_ass_33, old_disc_33=old_disc_33, new_disc_33=new_disc_33,
                       old_ass_34=old_ass_34, new_ass_34=new_ass_34, old_disc_34=old_disc_34, new_disc_34=new_disc_34,
                       old_ass_35=old_ass_35, new_ass_35=new_ass_35, old_disc_35=old_disc_35, new_disc_35=new_disc_35,
                       old_ass_36=old_ass_36, new_ass_36=new_ass_36, old_disc_36=old_disc_36, new_disc_36=new_disc_36)
txt += '\n\n'

if mpi_rank == 0:
    # Write performance table to MarkDown file
    with open('matrix_assembly_speed_log.md', 'a') as f:
        f.write(txt)

    # Remove temporary folders
    base_dir = Path(__file__).parent
    dirs_to_remove = [
        "geometry",
        "__psydac__",
        "__pycache__",
        "__epyccel__",
        "__gpyccel__"
    ]
    for d in dirs_to_remove:
        path = base_dir / d
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Failed to remove {path}: {e}")
