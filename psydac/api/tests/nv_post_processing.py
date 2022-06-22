import os
import pytest
import numpy as np

from sympde.topology import ScalarFunctionSpace, VectorFunctionSpace
from sympde.topology import ProductSpace
from sympde.topology import Domain, Union
from sympde.expr     import LinearForm, integral
from sympde.topology import element_of, elements_of
from sympde.topology import NormalVector
from sympde.calculus import grad, dot

from sympy import ImmutableDenseMatrix as Matrix

from psydac.api.discretization import discretize
from psydac.linalg.utilities   import array_to_stencil
from psydac.fem.basic          import FemField
from psydac.api.settings       import PSYDAC_BACKEND_GPYCCEL

from psydac.feec.multipatch.plotting_utilities import get_plotting_grid, get_grid_vals
from psydac.feec.multipatch.plotting_utilities import get_patch_knots_gridlines, my_small_plot

import matplotlib.pyplot as plt
from matplotlib import animation
import tqdm

# ... get the mesh directory
try:
    mesh_dir = os.environ['PSYDAC_MESH_DIR']

except:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(base_dir, '..', '..', '..')
    mesh_dir = os.path.join(base_dir, 'mesh')

#..................................................................................................
def animate_field(fields_vals, xx, yy, n_patches, titles=None, vrange=None, cmap=None, interval=35, figsize=(14,4)):
    """Animate a sequence of scalar fields over a geometry."""
    from matplotlib import animation
    import matplotlib.pyplot as plt

    fields_vals = list(fields_vals)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    vrange   = (np.array(fields_vals).min(), np.array(fields_vals).max())
    quadmeshs = []
    for k in range(n_patches):
        quadmeshs.append(plt.pcolormesh(xx[k], yy[k], fields_vals[0][k], shading='gouraud', cmap=cmap,
                    vmin=vrange[0], vmax=vrange[1], axes=ax))
    fig.colorbar(quadmeshs[0], ax=ax)

    pbar = tqdm.tqdm(total=len(fields_vals))
    def anim_func(i):
        C = fields_vals[i]
        for k in range(n_patches):
            quadmeshs[k].set_array(C[k])
        pbar.update()
        if titles:
            ax.set_title(titles[i])
        if i == len(fields_vals) - 1:
            pbar.close()

    return animation.FuncAnimation(fig, anim_func, frames=len(fields_vals), interval=interval)

#..................................................................................................
filename = os.path.join(mesh_dir, 'multipatch/plate_with_hole_mp_32_2.h5')
dt_h     = 0.05
mu       = 0.001
#..................................................................................................
domain = Domain.from_file(filename)
V1 = VectorFunctionSpace('V1', domain, kind='H1')
V2 = ScalarFunctionSpace('V2', domain, kind='L2')
X  = ProductSpace(V1, V2)

#v  = element_of(V1, name='v')
#q  = element_of(V2, name='q')
#nn = NormalVector('nn')

#patches = domain.interior.args
#boundary_circle = Union(*[patches[0].get_boundary(axis=1, ext=-1), 
#                          patches[1].get_boundary(axis=1, ext=-1),
#                          patches[2].get_boundary(axis=1, ext=-1),
#                          patches[3].get_boundary(axis=1, ext=-1)])


#drag = LinearForm((v, q),  integral(boundary_circle,  dot(Matrix([1,0]),mu*grad(v)*nn-q*nn)))
#lift = LinearForm((v, q),  integral(boundary_circle,  dot(Matrix([0,1]),mu*grad(v)*nn-q*nn)))

# ... discretization

domain_h = discretize(domain, filename=filename)
V1h      = discretize(V1, domain_h)
V2h      = discretize(V2, domain_h)
Xh       = discretize(X, domain_h)

#drag_h = discretize(drag, domain_h, Xh, backend=PSYDAC_BACKEND_GPYCCEL)
#lift_h = discretize(lift, domain_h, Xh, backend=PSYDAC_BACKEND_GPYCCEL)

#drag_h = drag_h.assemble()
#lift_h = lift_h.assemble()
#......................................................................................................
N1 = 1
N2 = 141
fields_folder = 'fields_p=2_ncells=32_2_dt_h=0.05'
fields_list = [np.load(fields_folder+"/u_p_{}.npy".format(k)) for k in range(N1,N2+1)]
fields_list = [array_to_stencil(f, Xh.vector_space) for f in fields_list]
fields_list = [FemField(Xh, coeffs=f) for f in fields_list]

velocity_fields = []
pressure_fields = []
drag_vals       = []
lift_vals       = []
times           = []
for i in range(len(fields_list)):
    u_h = FemField(V1h)
    p_h = FemField(V2h)
    xh  = fields_list[i]
    for k in range(len(domain)):
        u_h[k][0].coeffs[:,:] = xh[k][0].coeffs[:,:]
        u_h[k][1].coeffs[:,:] = xh[k][1].coeffs[:,:]
        p_h[k].coeffs[:,:]    = xh[k][2].coeffs[:,:]

    times.append(i*dt_h)
    velocity_fields.append(u_h)
    pressure_fields.append(p_h)
#    drag_vals.append(drag_h.dot(xh.coeffs))
#    lift_vals.append(lift_h.dot(xh.coeffs))

domains  = domain.logical_domain.interior
mappings = list(domain_h.mappings.values())

etas, xx, yy   = get_plotting_grid({I:M for I,M in zip(domains, mappings)}, N=50)

#gridlines_x11, gridlines_x21 = get_patch_knots_gridlines(V1h, 50, {I:M for I,M in zip(domains, mappings)}, plotted_patch=0)
#gridlines_x12, gridlines_x22 = get_patch_knots_gridlines(V1h, 50, {I:M for I,M in zip(domains, mappings)}, plotted_patch=1)
#gridlines_x13, gridlines_x23 = get_patch_knots_gridlines(V1h, 50, {I:M for I,M in zip(domains, mappings)}, plotted_patch=2)
#gridlines_x14, gridlines_x24 = get_patch_knots_gridlines(V1h, 50, {I:M for I,M in zip(domains, mappings)}, plotted_patch=3)
#gridlines_x15, gridlines_x25 = get_patch_knots_gridlines(V1h, 50, {I:M for I,M in zip(domains, mappings)}, plotted_patch=4)

grid_vals_h1   = lambda v: get_grid_vals(v, etas, mappings, space_kind='h1')

t    = 0
ts   = [1.,1.5,2.,2.5, 3.,3.5, 4.]
for i,f in enumerate(velocity_fields):
    t = (i+N1)*dt_h
    t = int(t) + (int(10*t)-10*int(t))/10
    if len(ts)==0:break
    elif not t in ts:continue
    else:ts.remove(t)
    uh_x_vals, uh_y_vals = grid_vals_h1(f)
    my_small_plot(
        title=r'$t=%0.1f$'%(t),
        vals=[np.sqrt(uh_x_vals**2+uh_y_vals**2)],
        xx=xx,
        yy=yy,
#        gridlines_x1=[gridlines_x11, gridlines_x12, gridlines_x13, gridlines_x14, gridlines_x15],
#        gridlines_x2=[gridlines_x21, gridlines_x22, gridlines_x23, gridlines_x24, gridlines_x25],
        save_fig = 'figures/u_n{}.pdf'.format(i+N1),
        hide_plot=True,
        )
raise

plt.plot(times, 20*abs(np.array(drag_vals)))
plt.show()
plt.plot(times, 20*np.array(lift_vals))
plt.show()

amp = lambda x,y:np.sqrt(x**2+y**2)
velocity_fields_vals = [amp(*grid_vals_h1(f)) for f in tqdm.tqdm(velocity_fields)]

titles = [r'approximation of solution $u$ at $t=%0.1f$'%(i*dt_h) for i in range(len(velocity_fields))]
anim = animate_field(velocity_fields_vals, xx, yy, len(domain), titles=titles)
anim.save('animated_fields.mp4', writer=animation.FFMpegWriter(fps=3))
