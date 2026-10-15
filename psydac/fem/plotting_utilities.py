#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from collections import OrderedDict

from mpi4py import MPI
from sympy import lambdify
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from psydac.linalg.utilities import array_to_psydac
from psydac.fem.basic import FemField, FemSpace
from psydac.utilities.utils import refine_array_1d
from psydac.feec.pull_push import push_2d_h1_vec, push_2d_h1, push_2d_hcurl, push_2d_hdiv, push_2d_l2


__all__ = (
    'get_grid_vals',
    'get_grid_quad_weights',
    'get_plotting_grid',
    'get_diag_grid',
    'get_patch_knots_gridlines',
    'plot_field_2d',
    'my_small_plot',
    'my_small_streamplot')

# ==============================================================================

def get_grid_vals(u, etas, mappings_list, space_kind=None):
    """
    get the physical field values, given the logical field and the logical grid
    :param u: FemField or callable function
    :param etas: logical grid
    :param space_kind: specifies the push-forward for the physical values
    """
    n_patches = len(mappings_list)
    if isinstance(u, FemField):
        vector_valued = u.space.is_vector_valued 
    else:
        # then u should be callable
        if len(mappings_list) == 1:
            # single patch
            u_single_patch = u
        else:
            # multiple patches
            u_single_patch = u[0]
        vector_valued = isinstance(u_single_patch, (list, tuple)) # [MCP 04.03.25]: this needs to be tested

    if space_kind is None:
        # use a simple change of variable
        # todo [MCP 26.03.2025]: this information should be stored in the FemSpace object!
        space_kind = 'h1'
            
    if vector_valued:
        # WARNING: here we assume 2D !
        u_vals_components = [n_patches * [None], n_patches * [None]]
    else:
        u_vals_components = [n_patches * [None]]

    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        for vals in u_vals_components:
            vals[k] = np.empty_like(eta_1)
        uk_field_1 = None
        if isinstance(u, FemField):
            if vector_valued:
                uk_field_0 = u.patch_fields[k].fields[0]
                uk_field_1 = u.patch_fields[k].fields[1]
            else:
                # it would be nice to just write u.patch_fields[k].fields[0] here...
                uk_field_0 = u.patch_fields[k]
        else:
            # then u should be callable
            if vector_valued:
                uk_field_0 = u[k][0]
                uk_field_1 = u[k][1]
            else:
                uk_field_0 = u[k]

        # computing the pushed-fwd values on the grid
        if space_kind == 'h1' or space_kind == 'V0':

            if vector_valued:
                def push_field(
                    eta1, eta2): return push_2d_h1_vec(
                    uk_field_0, uk_field_1, eta1, eta2)
            
            else:
                def push_field(
                    eta1, eta2): return push_2d_h1(
                    uk_field_0, eta1, eta2)
        elif space_kind == 'hcurl' or space_kind == 'V1':
            # todo (MCP): specify 2d_hcurl_scalar in push functions
            def push_field(
                eta1,
                eta2): return push_2d_hcurl(
                uk_field_0,
                uk_field_1,
                eta1,
                eta2,
                mappings_list[k].get_callable_mapping())
        elif space_kind == 'hdiv' or space_kind == 'V2':
            def push_field(
                eta1,
                eta2): return push_2d_hdiv(
                uk_field_0,
                uk_field_1,
                eta1,
                eta2,
                mappings_list[k].get_callable_mapping())
        elif space_kind == 'l2':
            assert not vector_valued

            def push_field(
                eta1,
                eta2): return push_2d_l2(
                uk_field_0,
                eta1,
                eta2,
                mappings_list[k].get_callable_mapping())
        else:
            raise ValueError(
                'unknown value for space_kind = {}'.format(space_kind))

        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                if vector_valued:
                    u_vals_components[0][k][i, j], u_vals_components[1][k][i, j] = push_field(
                        x1i, x2j)
                else:
                    u_vals_components[0][k][i, j] = push_field(x1i, x2j)

    # always return a list, even for scalar-valued functions ?
    if not vector_valued:
        return u_vals_components[0]
    else:
        return u_vals_components

# ------------------------------------------------------------------------------


def get_grid_quad_weights(etas, patch_logvols, mappings_list):  # _obj):
    # get approximate weights for a physical quadrature, namely
    #  |J_F(xi1, xi2)| * log_weight   with uniform log_weight = h1*h2     for (xi1, xi2) in the logical grid,
    # in the same format as the fields value in get_grid_vals_scalar and
    # get_grid_vals_vector

    n_patches = len(mappings_list)
    quad_weights = n_patches * [None]
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')
        quad_weights[k] = np.empty_like(eta_1)
        def one_field(xi1, xi2): return 1

        N0 = eta_1.shape[0]
        N1 = eta_1.shape[1]

        log_weight = patch_logvols[k] / (N0 * N1)
        Fk = mappings_list[k].get_callable_mapping()
        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                det_Fk_ij = Fk.metric_det(x1i, x2j)**0.5
                quad_weights[k][i, j] = det_Fk_ij * log_weight

    return quad_weights

# ------------------------------------------------------------------------------


def get_plotting_grid(
        mappings,
        N,
        centered_nodes=False,
        return_patch_logvols=False):
    # if centered_nodes == False, returns a regular grid with (N+1)x(N+1) nodes, starting and ending at patch boundaries
    # (useful for plotting the full patches)
    # if centered_nodes == True, returns the grid consisting of the NxN centers of the latter
    # (useful for quadratures and to avoid evaluating at patch boundaries)
    # if return_patch_logvols == True, return the logival volume (area) of the
    # patches
    nb_patches = len(mappings)
    grid_min_coords = [np.array(D.min_coords) for D in mappings]
    grid_max_coords = [np.array(D.max_coords) for D in mappings]
    if return_patch_logvols:
        patch_logvols = [(D.max_coords[1] - D.min_coords[1]) *
                         (D.max_coords[0] - D.min_coords[0]) for D in mappings]
    else:
        patch_logvols = None
    if centered_nodes:
        for k in range(nb_patches):
            for dim in range(2):
                h_grid = (grid_max_coords[k][dim] -
                          grid_min_coords[k][dim]) / N
                grid_max_coords[k][dim] -= h_grid / 2
                grid_min_coords[k][dim] += h_grid / 2
        N_cells = N - 1
    else:
        N_cells = N
    # etas     = [[refine_array_1d( bounds, N ) for bounds in zip(D.min_coords, D.max_coords)] for D in mappings]
    etas = [[refine_array_1d(bounds, N_cells) for bounds in zip(
        grid_min_coords[k], grid_max_coords[k])] for k in range(nb_patches)]
    callable_mappings = [M.get_callable_mapping() for d, M in mappings.items()]

    pcoords = [np.array([[f(e1, e2) for e2 in eta[1]] for e1 in eta[0]])
               for f, eta in zip(callable_mappings, etas)]
    
    xx = [pcoords[k][:, :, 0] for k in range(nb_patches)]
    yy = [pcoords[k][:, :, 1] for k in range(nb_patches)]

    if return_patch_logvols:
        return etas, xx, yy, patch_logvols
    else:
        return etas, xx, yy

# ------------------------------------------------------------------------------


def get_diag_grid(mappings, N):
    nb_patches = len(mappings)
    etas = [[refine_array_1d(bounds, N) for bounds in zip(
        D.min_coords, D.max_coords)] for D in mappings]
    callable_mappings = [M.get_callable_mapping() for d, M in mappings.items()]
    pcoords = [np.array([[f(e1, e2) for e2 in eta[1]] for e1 in eta[0]])
               for f, eta in zip(callable_mappings, etas)]

    xx = [pcoords[k][:, :, 0] for k in range(nb_patches)]
    yy = [pcoords[k][:, :, 1] for k in range(nb_patches)]

    return etas, xx, yy

# ------------------------------------------------------------------------------


def get_patch_knots_gridlines(Vh, N, mappings, plotted_patch=-1):
    # get gridlines for one patch grid

    F = [M.get_callable_mapping() for d, M in mappings.items()]

    if plotted_patch in range(len(mappings)):
        grid_x1 = Vh.patch_spaces[plotted_patch].spaces[0].breaks
        grid_x2 = Vh.patch_spaces[plotted_patch].spaces[1].breaks

        x1 = refine_array_1d(grid_x1, N)
        x2 = refine_array_1d(grid_x2, N)

        x1, x2 = np.meshgrid(x1, x2, indexing='ij')
        x, y = F[plotted_patch](x1, x2)

        gridlines_x1 = (x[:, ::N], y[:, ::N])
        gridlines_x2 = (x[::N, :].T, y[::N, :].T)
        # gridlines = (gridlines_x1, gridlines_x2)
    else:
        gridlines_x1 = None
        gridlines_x2 = None

    return gridlines_x1, gridlines_x2

# ------------------------------------------------------------------------------


def plot_field_2d(
        fem_field=None,
        stencil_coeffs=None,
        numpy_coeffs=None,
        Vh=None,
        domain=None,
        surface_plot=False,
        cb_min=None,
        cb_max=None,
        plot_type='amplitude',
        cmap='hsv',
        space_kind=None,
        title=None,
        filename='dummy_plot.png',
        subtitles=None,
        N_vis=20,
        vf_skip=2,
        hide_plot=True):
    """
    plot a discrete field (given as a FemField or by its coeffs in numpy or stencil format) on the given domain

    Parameters
    ----------
    numpy_coeffs : (np.ndarray)
        Coefficients of the field to plot

    Vh : TensorFemSpace
        Fem space needed if v is given by its coeffs

    space_kind : (str)
        type of the push-forward defining the physical Fem Space
        ## todo [MCP 13.03.2025]: rename this argument to something like push_kind and check other similar arguments in the code

    N_vis : (int)
        nb of visualization points per patch (per dimension)
    """

    vh = fem_field
    if vh is None:
        if numpy_coeffs is not None:
            assert stencil_coeffs is None
            stencil_coeffs = array_to_psydac(numpy_coeffs, Vh.coeff_space)
        vh = FemField(Vh, coeffs=stencil_coeffs)

    mappings = domain.mappings
    mappings_list = list(mappings.values())
    etas, xx, yy = get_plotting_grid(mappings, N=N_vis)

    def grid_vals(v): return get_grid_vals(
        v, etas, mappings_list, space_kind=space_kind)

    vh_vals = grid_vals(vh)
    if plot_type == 'vector_field' and not vh.space.is_vector_valued:
        print(
            "WARNING [plot_field_2d]: vector_field plot is not possible with a scalar field, plotting the amplitude instead")
        plot_type = 'amplitude'

    if plot_type == 'vector_field':
        if vh.space.is_vector_valued:
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
                dpi=200,
            )

    else:
        # computing plot_vals_list: may have several elements for several plots
        if plot_type == 'amplitude':

            if vh.space.is_vector_valued:
                # then vh_vals[d] contains the values of the d-component of vh
                # (as a patch-indexed list)
                plot_vals = [np.sqrt(abs(v[0])**2 + abs(v[1])**2)
                             for v in zip(vh_vals[0], vh_vals[1])]
            else:
                # then vh_vals just contains the values of vh (as a
                # patch-indexed list)
                plot_vals = np.abs(vh_vals)
            plot_vals_list = [plot_vals]

        elif plot_type == 'components':
            if vh.space.is_vector_valued:
                # then vh_vals[d] contains the values of the d-component of vh
                # (as a patch-indexed list)
                plot_vals_list = vh_vals
                if subtitles is None:
                    subtitles = ['x-component', 'y-component']
            else:
                # then vh_vals just contains the values of vh (as a
                # patch-indexed list)
                plot_vals_list = [vh_vals]
        else:
            raise ValueError(plot_type)

        # If there is just one patch, also plot the grid
        if not vh.space.is_multipatch:
            ncells = min(vh.space.component_spaces[0].axis_spaces[i].ncells for i in [0, 1])
            N = N_vis // ncells
            gridlines_x1, gridlines_x2 = get_patch_knots_gridlines(vh.space, N, mappings, 0)
        else:
            gridlines_x1 = None
            gridlines_x2 = None

        my_small_plot(
            title=title,
            vals=plot_vals_list,
            titles=subtitles,
            xx=xx,
            yy=yy,
            gridlines_x1 = gridlines_x1,
            gridlines_x2 = gridlines_x2,
            surface_plot=surface_plot,
            cb_min=cb_min,
            cb_max=cb_max,
            save_fig=filename,
            save_vals=False,
            hide_plot=hide_plot,
            cmap=cmap,
            dpi=300,
        )

# ------------------------------------------------------------------------------


def my_small_plot(
        title, vals, titles=None,
        xx=None, yy=None,
        gridlines_x1=None,
        gridlines_x2=None,
        surface_plot=False,
        cmap='viridis',
        cb_min=None,
        cb_max=None,
        save_fig=None,
        save_vals=False,
        hide_plot=False,
        dpi='figure',
        show_xylabel=True,
):
    """
        plot a list of scalar fields on a list of patches

        Parameters
        ----------
        title : (str)
            title of the plot

        vals : (list)
            list of scalar fields to plot

        titles : (list)
            list of titles for each plot

        xx : (list)
            list of x-coordinates of the grid points

        yy : (list)
            list of y-coordinates of the grid points
    """
    # titles is discarded if only one plot
    # cmap = 'jet' is nice too, but not so uniform. 'plasma' or 'magma' are uniform also.
    # cmap = 'hsv' is good for singular fields, for its rapid color change
    assert xx and yy
    n_plots = len(vals)
    if n_plots > 1:
        if titles is None or n_plots != len(titles):
            titles = n_plots * [title]
    else:
        if titles:
            print(
                'Warning [my_small_plot]: will discard argument titles for a single plot')
        titles = [title]

    n_patches = len(xx)
    assert n_patches == len(yy)

    if save_vals:
        # saving as vals.npz
        np.savez('vals', xx=xx, yy=yy, vals=vals)

    fig = plt.figure(figsize=(2.6 + 4.8 * n_plots, 4.8))
    fig.suptitle(title, fontsize=14)

    for i in range(n_plots):
        if cb_min is None:
            vmin = np.min(vals[i])
        else:
            vmin = cb_min
        if cb_max is None:
            vmax = np.max(vals[i])
        else:
            vmax = cb_max
        cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
        assert n_patches == len(vals[i])

        ax = fig.add_subplot(1, n_plots, i + 1)
        for k in range(n_patches):
            ax.contourf(
                xx[k],
                yy[k],
                vals[i][k],
                50,
                norm=cnorm,
                cmap=cmap,
                zorder=-
                10)  # , extend='both')
        ax.set_rasterization_zorder(0)
        cbar = fig.colorbar(
            cm.ScalarMappable(
                norm=cnorm,
                cmap=cmap),
            ax=ax,
            pad=0.05)
        if gridlines_x1 is not None:
            ax.plot(*gridlines_x1, color='k', linewidth=1)
        if gridlines_x2 is not None:
            ax.plot(*gridlines_x2, color='k', linewidth=1)
        if show_xylabel:
            ax.set_xlabel(r'$x$', rotation='horizontal')
            ax.set_ylabel(r'$y$', rotation='horizontal')
        if n_plots > 1:
            ax.set_title(titles[i])
        ax.set_aspect('equal')

    if save_fig:
        print('saving contour plot in file ' + save_fig)
        plt.savefig(save_fig, bbox_inches='tight', dpi=dpi)

    if not hide_plot:
        plt.show()

    if surface_plot:
        fig = plt.figure(figsize=(2.6 + 4.8 * n_plots, 4.8))
        fig.suptitle(title + ' -- surface', fontsize=14)

        for i in range(n_plots):
            if cb_min is None:
                vmin = np.min(vals[i])
            else:
                vmin = cb_min
            if cb_max is None:
                vmax = np.max(vals[i])
            else:
                vmax = cb_max
            cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
            assert n_patches == len(vals[i])
            ax = fig.add_subplot(1, n_plots, i + 1, projection='3d')
            for k in range(n_patches):
                ax.plot_surface(
                    xx[k],
                    yy[k],
                    vals[i][k],
                    norm=cnorm,
                    rstride=10,
                    cstride=10,
                    cmap=cmap,
                    linewidth=0,
                    antialiased=False)
            cbar = fig.colorbar(
                cm.ScalarMappable(
                    norm=cnorm,
                    cmap=cmap),
                ax=ax,
                pad=0.05)
            if show_xylabel:
                ax.set_xlabel(r'$x$', rotation='horizontal')
                ax.set_ylabel(r'$y$', rotation='horizontal')
            ax.set_title(titles[i])

        if save_fig:
            ext = save_fig[-4:]
            if ext[0] != '.':
                print('WARNING: extension unclear for file_name ' + save_fig)
            save_fig_surf = save_fig[:-4] + '_surf' + ext
            print('saving surface plot in file ' + save_fig_surf)
            plt.savefig(save_fig_surf, bbox_inches='tight', dpi=dpi)

        if not hide_plot:
            plt.show()

# ------------------------------------------------------------------------------


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

    fig, ax = plt.subplots(1, 1, figsize=(2.6 + 4.8, 4.8))

    fig.suptitle(title, fontsize=14)

    delta = 0.25
    # x = y = np.arange(-3.0, 3.01, delta)
    # X, Y = np.meshgrid(x, y)
    max_val = max(np.max(vals_x), np.max(vals_y))
    # print('max_val = {}'.format(max_val))
    vf_amp = amp_factor / (max_val + 1e-20)
    for k in range(n_patches):
        ax.quiver(xx[k][::skip,
                        ::skip],
                  yy[k][::skip,
                        ::skip],
                  vals_x[k][::skip,
                            ::skip],
                  vals_y[k][::skip,
                            ::skip],
                  scale=1 / (vf_amp * 0.05),
                  width=0.002)  # width=) units='width', pivot='mid',

    if show_xylabel:
        ax.set_xlabel(r'$x$', rotation='horizontal')
        ax.set_ylabel(r'$y$', rotation='horizontal')

    ax.set_aspect('equal')

    if save_fig:
        print('saving vector field (stream) plot in file ' + save_fig)
        plt.savefig(save_fig, bbox_inches='tight', dpi=dpi)

    if not hide_plot:
        plt.show()
