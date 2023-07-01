# coding: utf-8

from mpi4py import MPI
from sympy  import lambdify

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib  import cm, colors
from mpl_toolkits import mplot3d
from collections import OrderedDict

from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic        import FemField
from psydac.fem.vector       import ProductFemSpace, VectorFemSpace
from psydac.utilities.utils  import refine_array_1d
from psydac.feec.pull_push   import push_2d_h1, push_2d_hcurl, push_2d_hdiv, push_2d_l2

matplotlib.rcParams['font.size'] = 15

#==============================================================================
def is_vector_valued(u):
    # small utility function, only tested for FemFields in multi-patch spaces of the 2D grad-curl sequence
    # todo: a proper interface returning the number of components of a general FemField would be nice
    return u.fields[0].space.is_product

#------------------------------------------------------------------------------
def get_grid_vals(u, etas, mappings_list, space_kind='hcurl'):
    """
    get the physical field values, given the logical field and the logical grid
    :param u: FemField
    :param etas: logical grid
    :param space_kind: specifies the push-forward for the physical values
    """
    n_patches = len(mappings_list)
    vector_valued = is_vector_valued(u) if isinstance(u, FemField) else isinstance(u[0],(list, tuple))
    if vector_valued:
        # WARNING: here we assume 2D !
        u_vals_components = [n_patches*[np.nan], n_patches*[np.nan]]
    else:
        u_vals_components = [n_patches*[np.nan]]

    for k in range(n_patches):        
        if etas[k][0][0] is not None:
            # print("etas[k][0] = ", etas[k][0])
            eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
            for vals in u_vals_components:
                vals[k] = np.empty_like(eta_1)
            uk_field_1 = None
            if isinstance(u,FemField):
                if vector_valued:
                    uk_field_0 = u[k].fields[0]
                    uk_field_1 = u[k].fields[1]
                else:
                    uk_field_0 = u.fields[k]   # it would be nice to just write u[k].fields[0] here...
            else:
                # then u should be callable
                if vector_valued:
                    uk_field_0 = u[k][0]
                    uk_field_1 = u[k][1]
                else:
                    uk_field_0 = u[k]

            # computing the pushed-fwd values on the grid
            if space_kind == 'h1':
                assert not vector_valued
                # todo (MCP): add 2d_hcurl_vector
                push_field = lambda eta1, eta2: push_2d_h1(uk_field_0, eta1, eta2)
            elif space_kind == 'hcurl':
                # todo (MCP): specify 2d_hcurl_scalar in push functions
                push_field = lambda eta1, eta2: push_2d_hcurl(uk_field_0, uk_field_1, eta1, eta2, F=mappings_list[k])
            elif space_kind == 'hdiv':
                push_field = lambda eta1, eta2: push_2d_hdiv(uk_field_0, uk_field_1, eta1, eta2, F=mappings_list[k])
            elif space_kind == 'l2':
                assert not vector_valued
                push_field = lambda eta1, eta2: push_2d_l2(uk_field_0, eta1, eta2, F=mappings_list[k])
            else:
                raise ValueError('unknown value for space_kind = {}'.format(space_kind))

            for i, x1i in enumerate(eta_1[:, 0]):
                for j, x2j in enumerate(eta_2[0, :]):
                    if vector_valued:
                        u_vals_components[0][k][i, j], u_vals_components[1][k][i, j] = push_field(x1i, x2j)
                    else:
                        u_vals_components[0][k][i, j] = push_field(x1i, x2j)

    # always return a list, even for scalar-valued functions ?
    if not vector_valued:
        return u_vals_components[0]
    else:
        return u_vals_components

#------------------------------------------------------------------------------
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
        Fk = mappings_list[k].get_callable_mapping()
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                det_Fk_ij = Fk.metric_det(x1i, x2j)**0.5
                quad_weights[k][i, j] = det_Fk_ij * log_weight

    return quad_weights

#------------------------------------------------------------------------------
def get_plotting_grid(mappings, N, centered_nodes=False, return_patch_logvols=False, eta_crop=None):
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
    if eta_crop is None:
        etas = [[refine_array_1d( bounds, N_cells ) for bounds in zip(grid_min_coords[k], grid_max_coords[k])] for k in range(nb_patches)]
    else:
        etas = [[[None,], [None,]] for k in range(nb_patches)]
        for k in range(nb_patches):
            nonempty_patch_crop = True
            for dim in range(2):
                nonempty_patch_crop = ( 
                    nonempty_patch_crop
                    and eta_crop[dim][1] > grid_min_coords[k][dim]
                    and eta_crop[dim][0] < grid_max_coords[k][dim]
                    )
            if nonempty_patch_crop:           
                for dim in range(2):
                    grid_min_coords[k][dim] = max(grid_min_coords[k][dim], eta_crop[dim][0])
                    grid_max_coords[k][dim] = min(grid_max_coords[k][dim], eta_crop[dim][1])
                etas[k] = [refine_array_1d( bounds, N_cells ) for bounds in zip(grid_min_coords[k], grid_max_coords[k])]
    mappings_lambda = [lambdify(M.logical_coordinates, M.expressions) for d,M in mappings.items()]

    pcoords = [np.array( [[f(e1,e2) for e2 in eta[1]] for e1 in eta[0]] ) for f,eta in zip(mappings_lambda, etas)]

    xx = [pcoords[k][:,:,0] for k in range(nb_patches)]
    yy = [pcoords[k][:,:,1] for k in range(nb_patches)]

    if return_patch_logvols:
        return etas, xx, yy, patch_logvols
    else:
        return etas, xx, yy

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
def get_patch_knots_gridlines(Vh, N, mappings, plotted_patch=-1):
    # get gridlines for one patch grid

    F = [M.get_callable_mapping() for d,M in mappings.items()]

    if plotted_patch in range(len(mappings)):
        space   = Vh.spaces[plotted_patch]
        if isinstance(space, (VectorFemSpace, ProductFemSpace)):
            space = space.spaces[0]

        grid_x1 = space.breaks[0]
        grid_x2 = space.breaks[1]

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

#------------------------------------------------------------------------------
def plot_field(
    fem_field=None, stencil_coeffs=None, numpy_coeffs=None, Vh=None, domain=None, surface_plot=False, cb_min=None, cb_max=None,
    cf_levels=50, plot_type='amplitude', show_grid=False, cmap='hsv', space_kind=None, title=None, filename='dummy_plot.png', subtitles=None, 
    N_vis=20, vf_skip=2, hide_plot=True, eta_crop=None):
    """
    plot a discrete field (given as a FemField or by its coeffs in numpy or stencil format) on the given domain

    :param Vh: Fem space needed if v is given by its coeffs
    :param space_kind: type of the push-forward defining the physical Fem Space
    :param subtitles: in case one would like to have several subplots # todo: then v should be given as a list of fields...
    :param N_vis: nb of visualization points per patch (per dimension)
    :param cf_levels: nb of levels for the contourf plots
    """
    if not space_kind in ['h1', 'hcurl', 'hdiv', 'l2']:
        raise ValueError('invalid value for space_kind = {}'.format(space_kind))

    vh = fem_field
    if vh is None:
        if numpy_coeffs is not None:
            assert stencil_coeffs is None
            stencil_coeffs = array_to_psydac(numpy_coeffs, Vh.vector_space)
        vh = FemField(Vh, coeffs=stencil_coeffs)

    mappings = OrderedDict([(P.logical_domain, P.mapping) for P in domain.interior])
    mappings_list = [m.get_callable_mapping() for m in mappings.values()]
    # mappings_list = list(mappings.values())
    etas, xx, yy    = get_plotting_grid(mappings, N=N_vis, eta_crop=eta_crop)
    grid_vals = lambda v: get_grid_vals(v, etas, mappings_list, space_kind=space_kind)

    vh_vals = grid_vals(vh)
    if plot_type == 'vector_field' and not is_vector_valued(vh):
        print("WARNING [plot_field]: vector_field plot is not possible with a scalar field, plotting the amplitude instead")
        plot_type = 'amplitude'

    if plot_type == 'vector_field':
        if is_vector_valued(vh):
            my_small_streamplot(
                title=title,
                vals_x=vh_vals[0],
                vals_y=vh_vals[1],
                skip=vf_skip,
                xx=xx,
                yy=yy,
                amp_factor=2,
                save_fig=filename,
                hide_plot=hide_plot,
                dpi = 200,
            )

    else:
        # computing plot_vals_list: may have several elements for several plots
        if plot_type=='amplitude':

            if is_vector_valued(vh):
                # then vh_vals[d] contains the values of the d-component of vh (as a patch-indexed list)
                plot_vals = [np.sqrt(abs(v[0])**2 + abs(v[1])**2) for v in zip(vh_vals[0],vh_vals[1])]
            else:
                # then vh_vals just contains the values of vh (as a patch-indexed list)
                plot_vals = np.abs(vh_vals)
            plot_vals_list = [plot_vals]

        elif plot_type=='components':
            if is_vector_valued(vh):
                # then vh_vals[d] contains the values of the d-component of vh (as a patch-indexed list)
                plot_vals_list = vh_vals
                if subtitles is None:
                    subtitles = ['x-component', 'y-component']
            else:
                # then vh_vals just contains the values of vh (as a patch-indexed list)
                plot_vals_list = [vh_vals]
        else:
            raise ValueError(plot_type)

        gridlines_x1_list = []
        gridlines_x2_list = []
        if show_grid:
            for k in range(len(domain)):
                gl_x1, gl_x2 = get_patch_knots_gridlines(Vh, N=N_vis, mappings=mappings, plotted_patch=k)
                gridlines_x1_list.append(gl_x1)
                gridlines_x2_list.append(gl_x2)

        # if eta_crop is not None:
        #     # otherwise may bug
        #     cb_min = 0
        #     cb_max = 1

        my_small_plot(
            title=title,
            vals=plot_vals_list,
            titles=subtitles,
            xx=xx,
            yy=yy,
            surface_plot=surface_plot,
            gridlines_x1_list=gridlines_x1_list,
            gridlines_x2_list=gridlines_x2_list,
            cb_min=cb_min,
            cb_max=cb_max,
            levels=cf_levels,
            save_fig=filename,
            save_vals = False,
            hide_plot=hide_plot,
            cmap=cmap, 
            dpi = 300,
        )

    # if is_vector_valued(vh):
    #     # then vh_vals[d] contains the values of the d-component of vh (as a patch-indexed list)
    #     vh_abs_vals = [np.sqrt(abs(v[0])**2 + abs(v[1])**2) for v in zip(vh_vals[0],vh_vals[1])]
    # else:
    #     # then vh_vals just contains the values of vh (as a patch-indexed list)
    #     vh_abs_vals = np.abs(vh_vals)

    # my_small_plot(
    #     title=title,
    #     vals=[vh_abs_vals],
    #     titles=subtitles,
    #     xx=xx,
    #     yy=yy,
    #     surface_plot=False,
    #     save_fig=filename,
    #     save_vals=False,
    #     hide_plot=hide_plot,
    #     cmap='hsv',
    #     dpi = 400,
    # )

#------------------------------------------------------------------------------
def my_small_plot(
        title, vals, titles=None,
        xx=None, yy=None,
        gridlines_x1_list=[],
        gridlines_x2_list=[],
        surface_plot=False,
        cmap='viridis',
        cb_min=None,
        cb_max=None,
        levels=100,
        save_fig=None,
        save_vals = False,
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
        if titles is None or n_plots != len(titles):
            titles = n_plots*[title]
    else:
        if titles:
            print('Warning [my_small_plot]: will discard argument titles for a single plot')
        titles = [title]

    # if xx is not None
    n_patches = len(xx)
    assert n_patches == len(yy)

    if save_vals:
        np.savez('vals', xx=xx, yy=yy, vals=vals)
        
    fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
    fig.suptitle(title, fontsize=14)

    vmin = [None for i in range(n_plots)]
    vmax = [None for i in range(n_plots)]

    for i in range(n_plots):
        if cb_min is None:
            # print("vals[i][k] = ", vals[i])
            vmin[i] = np.nanmin([np.min(vals_ik) for vals_ik in vals[i]])
        else:
            vmin[i] = cb_min
        if cb_max is None:
            vmax[i] = np.nanmax([np.max(vals_ik) for vals_ik in vals[i]])
        else:
            vmax[i] = cb_max
        
        cnorm = colors.Normalize(vmin=vmin[i], vmax=vmax[i])
        assert n_patches == len(vals[i])

        ax = fig.add_subplot(1, n_plots, i+1)
        for k in range(n_patches):
            if xx[k][0][0] is not None and yy[k][0][0] is not None:
                ax.contourf(xx[k], yy[k], vals[i][k], levels, norm=cnorm, cmap=cmap, zorder=-10) #, extend='both')
        
        ax.plot(yy[k], vals[i][k], '--', color='k', label='curve', ms=5)

        ax.set_rasterization_zorder(0)
        cbar = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap), ax=ax,  pad=0.05)
        for gl_x1 in gridlines_x1_list:
            ax.plot(*gl_x1, color='k')
        for gl_x2 in gridlines_x2_list:
            ax.plot(*gl_x2, color='k')
        if show_xylabel:
            ax.set_xlabel( r'$x$', rotation='horizontal' )
            ax.set_ylabel( r'$y$', rotation='horizontal' )
        if n_plots > 1:
            ax.set_title ( titles[i] )
        ax.set_aspect('equal')

    if save_fig:
        print('saving contour plot in file '+save_fig)
        plt.savefig(save_fig, bbox_inches='tight',dpi=dpi)

    if not hide_plot:
        plt.show()

    if surface_plot:
        fig = plt.figure(figsize=(2.6+4.8*n_plots, 4.8))
        fig.suptitle(title+' -- surface', fontsize=14)

        for i in range(n_plots):
            cnorm = colors.Normalize(vmin=vmin[i], vmax=vmax[i])
            assert n_patches == len(vals[i])
            ax = fig.add_subplot(1, n_plots, i+1, projection='3d')
            for k in range(n_patches):
                if xx[k][0][0] is not None and yy[k][0][0] is not None:
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
        
        if not hide_plot:
            plt.show()

#------------------------------------------------------------------------------
def my_small_streamplot(
        title, vals_x, vals_y,
        xx, yy, skip=2,
        amp_factor=1,
        save_fig=None,
        hide_plot=False,
        show_xylabel=True,
        dpi='figure',
):
    """
    :param skip: every skip-th data point will be skipped
    """
    n_patches = len(xx)
    assert n_patches == len(yy)

    # fig = plt.figure(figsize=(2.6+4.8, 4.8))
    
    fig, ax = plt.subplots(1,1, figsize=(2.6+4.8, 4.8))
    
    fig.suptitle(title, fontsize=14)

    delta = 0.25
    # x = y = np.arange(-3.0, 3.01, delta)
    # X, Y = np.meshgrid(x, y)
    max_val = max(np.max(vals_x), np.max(vals_y))
    #print('max_val = {}'.format(max_val))
    vf_amp = amp_factor/max_val
    for k in range(n_patches):
        ax.quiver(xx[k][::skip, ::skip], yy[k][::skip, ::skip], vals_x[k][::skip, ::skip], vals_y[k][::skip, ::skip],
                   scale=1/(vf_amp*0.05), width=0.002) # width=) units='width', pivot='mid',

    if show_xylabel:
        ax.set_xlabel( r'$x$', rotation='horizontal' )
        ax.set_ylabel( r'$y$', rotation='horizontal' )

    ax.set_aspect('equal')

    if save_fig:
        print('saving vector field (stream) plot in file '+save_fig)
        plt.savefig(save_fig, bbox_inches='tight', dpi=dpi)

    if not hide_plot:
        plt.show()
