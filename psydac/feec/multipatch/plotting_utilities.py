# coding: utf-8

# Conga operators on piecewise (broken) de Rham sequences

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from psydac.fem.basic   import FemField


#==============================================================================
# some plotting utilities

from psydac.feec.pull_push     import push_2d_h1, push_2d_hcurl, push_2d_hdiv, push_2d_l2

# todo (MCP, april 12): merge get_grid_vals_scalar and get_grid_vals_vector into a single function

def get_grid_vals_scalar(u, etas, mappings_list, space_kind='h1'):  #_obj):
    # get the physical field values, given the logical field and the logical grid
    # n_patches = len(domain)
    n_patches = len(mappings_list)
    u_vals    = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        u_vals[k] = np.empty_like(eta_1)
        if isinstance(u,FemField):
            uk_field = u.fields[k]   # todo (MCP): try with u[k].fields?
        else:
            # then field is just callable
            uk_field = u[k]

        if space_kind == 'h1':
            # todo (MCP): add 2d_hcurl_vector
            push_field = lambda eta1, eta2: push_2d_h1(uk_field, eta1, eta2)
        else:
            push_field = lambda eta1, eta2: push_2d_l2(uk_field, eta1, eta2, mapping=mappings_list[k])
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                u_vals[k][i, j] = push_field(x1i, x2j)

    # u_vals  = np.concatenate(u_vals, axis=1)

    return u_vals


def get_grid_vals_vector(E, etas, mappings_list, space_kind='hcurl'):
    # get the physical field values, given the logical field and logical grid
    n_patches = len(mappings_list)
    # mappings_list = list(mappings.values())
    E_x_vals = n_patches*[None]
    E_y_vals = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        E_x_vals[k] = np.empty_like(eta_1)
        E_y_vals[k] = np.empty_like(eta_1)
        if isinstance(E,FemField):
            Ek_field_0 = E[k].fields[0]   # or E.fields[k][0] ?
            Ek_field_1 = E[k].fields[1]
        else:
            # then E field is just callable
            Ek_field_0 = E[k][0]
            Ek_field_1 = E[k][1]
        if space_kind == 'hcurl':
            # todo (MCP): specify 2d_hcurl_scalar in push functions
            push_field = lambda eta1, eta2: push_2d_hcurl(Ek_field_0, Ek_field_1, eta1, eta2, mapping=mappings_list[k])
        else:
            push_field = lambda eta1, eta2: push_2d_hdiv(Ek_field_0, Ek_field_1, eta1, eta2, mapping=mappings_list[k])

        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                E_x_vals[k][i, j], E_y_vals[k][i, j] = push_field(x1i, x2j)
    # E_x_vals = np.concatenate(E_x_vals, axis=1)
    # E_y_vals = np.concatenate(E_y_vals, axis=1)
    return E_x_vals, E_y_vals

def get_grid_quad_weights(etas, patch_logvols, mappings_list):  #_obj):
    # get approximate weights for a physical quadrature, namely
    #  |J_F(xi1, xi2)| * log_weight   with uniform log_weight = h1*h2     for (xi1, xi2) in the logical grid,
    # in the same format as the fields value in get_grid_vals_scalar and get_grid_vals_vector

    n_patches = len(mappings_list)
    quad_weights    = n_patches*[None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        quad_weights[k] = np.empty_like(eta_1)
        one_field = lambda xi1, xi2: 1

        N0 = eta_1.shape[0]
        N1 = eta_1.shape[1]

        log_weight = patch_logvols[k]/(N0*N1)
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                quad_weights[k][i, j] = push_2d_l2(one_field, x1i, x2j, mapping=mappings_list[k]) * log_weight

    return quad_weights

from psydac.utilities.utils    import refine_array_1d
from sympy import lambdify


def get_plotting_grid(mappings, N, centered_nodes=False, return_patch_logvols=False):
    # if centered_nodes == False, returns a regular grid with (N+1)x(N+1) nodes, starting and ending at patch boundaries
    # (useful for plotting the full patches)
    # if centered_nodes == True, returns the grid consisting of the NxN centers of the latter
    # (useful for quadratures and to avoid evaluating at patch boundaries)
    # if return_patch_logvols == True, return the logival volume (area) of the patches
    nb_patches = len(mappings)
    grid_min_coords = [np.array(D.min_coords) for D in mappings]
    grid_max_coords = [np.array(D.max_coords) for D in mappings]
    if return_patch_logvols:
        patch_logvols = [(D.max_coords[1]-D.min_coords[1])*(D.max_coords[0]-D.min_coords[0]) for D in mappings]
    else:
        patch_logvols = None
    if centered_nodes:
        for k in range(nb_patches):
            for dim in range(2):
                h_grid = (grid_max_coords[k][dim] - grid_min_coords[k][dim])/N
                grid_max_coords[k][dim] -= h_grid/2
                grid_min_coords[k][dim] += h_grid/2
        N_cells = N-1
    else:
        N_cells = N
    # etas     = [[refine_array_1d( bounds, N ) for bounds in zip(D.min_coords, D.max_coords)] for D in mappings]
    etas = [[refine_array_1d( bounds, N_cells ) for bounds in zip(grid_min_coords[k], grid_max_coords[k])] for k in range(nb_patches)]
    mappings_lambda = [lambdify(M.logical_coordinates, M.expressions) for d,M in mappings.items()]

    pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings_lambda, etas)]

    xx = [pcoords[k][:,:,0] for k in range(nb_patches)]
    yy = [pcoords[k][:,:,1] for k in range(nb_patches)]

    if return_patch_logvols:
        return etas, xx, yy, patch_logvols
    else:
        return etas, xx, yy

def get_diag_grid(mappings, N):
    nb_patches = len(mappings)
    etas     = [[refine_array_1d( bounds, N ) for bounds in zip(D.min_coords, D.max_coords)] for D in mappings]
    mappings_lambda = [lambdify(M.logical_coordinates, M.expressions) for d,M in mappings.items()]

    pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings_lambda, etas)]

    # pcoords  = np.concatenate(pcoords, axis=1)
    # xx = pcoords[:,:,0]
    # yy = pcoords[:,:,1]

    xx = [pcoords[k][:,:,0] for k in range(nb_patches)]
    yy = [pcoords[k][:,:,1] for k in range(nb_patches)]

    return etas, xx, yy

def get_patch_knots_gridlines(Vh, N, mappings, plotted_patch=-1):
    # get gridlines for one patch grid

    F = [M.get_callable_mapping() for d,M in mappings.items()]

    if plotted_patch in range(len(mappings)):
        grid_x1 = Vh.spaces[plotted_patch].breaks[0]
        grid_x2 = Vh.spaces[plotted_patch].breaks[1]

        x1 = refine_array_1d(grid_x1, N)
        x2 = refine_array_1d(grid_x2, N)

        x1, x2 = np.meshgrid(x1, x2, indexing='ij')
        x, y = F[plotted_patch](x1, x2)

        gridlines_x1 = (x[:, ::N],   y[:, ::N]  )
        gridlines_x2 = (x[::N, :].T, y[::N, :].T)
        # gridlines = (gridlines_x1, gridlines_x2)
    else:
        gridlines_x1 = None
        gridlines_x2 = None

    return gridlines_x1, gridlines_x2

from matplotlib import colors

def my_small_plot(
        title, vals, titles=None,
        xx=None, yy=None,
        gridlines_x1=None,
        gridlines_x2=None,
        surface_plot=False,
        cmap='viridis',
        save_fig=None,
        hide_plot=False,
        dpi='figure',
        show_xylabel=True,
):
    # titles is discarded if only one plot
    # cmap = 'jet' is nice too, but not so uniform. 'plasma' or 'magma' are uniform also.
    # cmap = 'hsv' is good for singular fields, for its rapid color change
    assert xx and yy
    n_plots = len(vals)
    if n_plots > 1:
        assert n_plots == len(titles)
    else:
        if titles:
            print('Warning [my_small_plot]: will discard argument titles for a single plot')

    n_patches = len(xx)
    assert n_patches == len(yy)

    fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
    fig.suptitle(title, fontsize=14)

    for i in range(n_plots):
        vmin = np.min(vals[i])
        vmax = np.max(vals[i])
        cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
        assert n_patches == len(vals[i])
        ax = fig.add_subplot(1, n_plots, i+1)
        for k in range(n_patches):
            ax.contourf(xx[k], yy[k], vals[i][k], 50, norm=cnorm, cmap=cmap) #, extend='both')
        cbar = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap), ax=ax,  pad=0.05)
        if gridlines_x1 is not None:
            ax.plot(*gridlines_x1, color='k')
            ax.plot(*gridlines_x2, color='k')
        if show_xylabel:
            ax.set_xlabel( r'$x$', rotation='horizontal' )
            ax.set_ylabel( r'$y$', rotation='horizontal' )
        if n_plots > 1:
            ax.set_title ( titles[i] )

    if save_fig:
        print('saving contour plot in file '+save_fig)
        plt.savefig(save_fig, bbox_inches='tight',dpi=dpi)

    if not hide_plot:
        plt.show()

    if surface_plot:
        fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
        fig.suptitle(title+' -- surface', fontsize=14)

        for i in range(n_plots):
            vmin = np.min(vals[i])
            vmax = np.max(vals[i])
            cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
            assert n_patches == len(vals[i])
            ax = fig.add_subplot(1, n_plots, i+1, projection='3d')
            for k in range(n_patches):
                ax.plot_surface(xx[k], yy[k], vals[i][k], norm=cnorm, rstride=10, cstride=10, cmap=cmap,
                           linewidth=0, antialiased=False)
            cbar = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap), ax=ax,  pad=0.05)
            if show_xylabel:
                ax.set_xlabel( r'$x$', rotation='horizontal' )
                ax.set_ylabel( r'$y$', rotation='horizontal' )
            ax.set_title ( titles[i] )

        if save_fig:
            ext = save_fig[-4:]
            if ext[0] != '.':
                print('WARNING: extension unclear for file_name '+save_fig)
            save_fig_surf = save_fig[:-4]+'_surf'+ext
            print('saving surface plot in file '+save_fig_surf)
            plt.savefig(save_fig_surf, bbox_inches='tight', dpi=dpi)
        else:
            plt.show()



def my_small_streamplot(
        title, vals_x, vals_y,
        xx, yy, skip=2,
        amp_factor=1,
        save_fig=None,
        hide_plot=False,
        dpi='figure',
):
    """
    :param skip: every skip-th data point will be skipped
    """
    n_patches = len(xx)
    assert n_patches == len(yy)

    fig = plt.figure(figsize=(2.6+4.8, 4.8))
    fig.suptitle(title, fontsize=14)

    delta = 0.25
    # x = y = np.arange(-3.0, 3.01, delta)
    # X, Y = np.meshgrid(x, y)
    max_val = max(np.max(vals_x), np.max(vals_y))
    #print('max_val = {}'.format(max_val))
    vf_amp = amp_factor/max_val
    for k in range(n_patches):
        plt.quiver(xx[k][::skip, ::skip], yy[k][::skip, ::skip], vals_x[k][::skip, ::skip], vals_y[k][::skip, ::skip],
                   scale=1/(vf_amp*0.05), width=0.002) # width=) units='width', pivot='mid',

    if save_fig:
        print('saving vector field (stream) plot in file '+save_fig)
        plt.savefig(save_fig, bbox_inches='tight', dpi=dpi)

    if not hide_plot:
        plt.show()

