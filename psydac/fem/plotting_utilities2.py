#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from    collections.abc         import Iterable
import  numpy                   as np
import  matplotlib.pyplot       as plt
from    matplotlib              import cm, colors

from    psydac.utilities.utils  import refine_array_1d
from    psydac.feec.pull_push   import push_2d_h1_vec, push_2d_h1, push_2d_hcurl, push_2d_hdiv, push_2d_l2
from    psydac.fem.basic        import FemField

__all__ = (
    'get_grid_vals',
    'get_plotting_grid',
    'get_patch_knots_gridlines',
    'plot_field_2d',
    'my_small_plot',
    'my_small_streamplot')

# ==============================================================================

def get_grid_vals(u, etas, mappings_list=None, space_kind=None):
    """
    Get the physical field values given the logical field and the logical grid.

    Parameters
    ----------
    u : psydac.fem.basic.FemField | Callable Function
        The logical function.

    etas : list | tuple
        Logical grid as returned by get_plotting_grid in this file.
    
    mappings_list : sympde.topology.Mapping | None
        Required for the pushforward. Must be passed only if u is not a FemField.

    space_kind : str | None
        Either 'h1', 'hcurl', 'hdiv', 'l2', 'undefined' or None. Required for the pushforward. Must be passed only if u is not a FemField.

    Returns
    -------
    u_component_patch_vals : list
        List of lists corresponding to components of u. Each such list contains lists corresponding to patches. 
        Each such list contains function values to be plotted over the pysical domain.

    """
    is_femfield = isinstance(u, FemField)

    if is_femfield:
        Vh            = u.space
        V             = Vh.symbolic_space
        domain        = V.domain
        mappings      = domain.mappings
        if mappings_list is None:
            mappings_list = list(mappings.values())
        if space_kind is None:
            space_kind = V.kind.name
        vector_valued = Vh.is_vector_valued
    else:
        assert mappings_list is not None
        assert space_kind is not None
        if len(mappings_list) == 1:
            u_single_patch = u
        else:
            u_single_patch = u[0]
        vector_valued = isinstance(u_single_patch, (list, tuple)) 

    if space_kind == 'undefined':
        space_kind = 'h1'

    n_patches     = len(mappings_list)
            
    if vector_valued:
        # WARNING: here we assume 2D !
        u_component_patch_vals = [n_patches * [None], n_patches * [None]]
    else:
        u_component_patch_vals = [n_patches * [None]]

    #print(f'{n_patches=}, {u.patch_fields=}')
    for k in range(n_patches):
        eta_1, eta_2 = np.meshgrid(etas[k][0], etas[k][1], indexing='ij')

        for vals in u_component_patch_vals:
            vals[k] = np.empty_like(eta_1)

        if isinstance(u, FemField):
            if vector_valued:
                uk_field_0 = u.patch_fields[k].fields[0]
                uk_field_1 = u.patch_fields[k].fields[1]
            else:
                uk_field_0 = u.patch_fields[k]
                uk_field_1 = None
        else:
            if vector_valued:
                uk_field_0 = u[k][0]
                uk_field_1 = u[k][1]
            else:
                uk_field_0 = u[k]
                uk_field_1 = None

        # computing the pushed-fwd values on the grid
        if space_kind == 'h1':
            if vector_valued:
                def push_field(eta1, eta2): 
                    return push_2d_h1_vec(uk_field_0, uk_field_1, eta1, eta2)
            else:
                def push_field(eta1, eta2): 
                    return push_2d_h1(uk_field_0, eta1, eta2)
                
        elif space_kind == 'hcurl':
            def push_field(eta1, eta2): 
                return push_2d_hcurl(uk_field_0, uk_field_1, eta1, eta2, mappings_list[k].get_callable_mapping())
            
        elif space_kind == 'hdiv':
            def push_field(eta1, eta2): 
                return push_2d_hdiv(uk_field_0, uk_field_1, eta1, eta2, mappings_list[k].get_callable_mapping())
            
        elif space_kind == 'l2':
            def push_field(eta1, eta2): 
                return push_2d_l2(uk_field_0, eta1, eta2, mappings_list[k].get_callable_mapping())
        else:
            raise ValueError(
                'unknown value for space_kind = {}'.format(space_kind))

        for i, x1i in enumerate(eta_1[:, 0]):
            for j, x2j in enumerate(eta_2[0, :]):
                if vector_valued:
                    u_component_patch_vals[0][k][i, j], u_component_patch_vals[1][k][i, j] = push_field(x1i, x2j)
                else:
                    u_component_patch_vals[0][k][i, j]                                     = push_field(x1i, x2j)

    # always return a list, even for scalar-valued functions
    #return u_component_patch_vals
    if not vector_valued:
        return u_component_patch_vals[0]
    else:
        return u_component_patch_vals

# ------------------------------------------------------------------------------

def get_plotting_grid(mappings, N, centered_nodes=False):
    # if centered_nodes == False, returns a regular grid with (N+1)x(N+1) nodes, starting and ending at patch boundaries
    # (useful for plotting the full patches)
    # if centered_nodes == True, returns the grid consisting of the NxN centers of the latter
    # (useful for quadratures and to avoid evaluating at patch boundaries)
    # if return_patch_logvols == True, return the logival volume (area) of the
    # patches

    nb_patches = len(mappings)

    grid_min_coords = [np.array(D.min_coords) for D in mappings]
    grid_max_coords = [np.array(D.max_coords) for D in mappings]

    if centered_nodes:
        for k in range(nb_patches):
            for dim in range(2):
                h_grid = (grid_max_coords[k][dim] -
                          grid_min_coords[k][dim]) / N
                grid_max_coords[k][dim] -= h_grid / 2
                grid_min_coords[k][dim] += h_grid / 2
        N_cells = (N[0] - 1, N[1] - 1)
    else:
        N_cells = N

    etas = [[refine_array_1d(bounds, N_cells[i]) for i, bounds in enumerate(zip(
        grid_min_coords[k], grid_max_coords[k]))] for k in range(nb_patches)]
    
    callable_mappings = [M.get_callable_mapping() for M in mappings.values()]

    pcoords = [np.array([[f(e1, e2) for e2 in eta[1]] for e1 in eta[0]])
               for f, eta in zip(callable_mappings, etas)]
    
    xx = [pcoords[k][:, :, 0] for k in range(nb_patches)]
    yy = [pcoords[k][:, :, 1] for k in range(nb_patches)]

    return etas, xx, yy

# ------------------------------------------------------------------------------

def get_patch_knots_gridlines(Vh, N, plotted_patch=-1):

    is_vector_valued = Vh.is_vector_valued
    V = Vh.symbolic_space
    domain = V.domain
    mappings = domain.mappings

    F = [M.get_callable_mapping() for M in mappings.values()]

    if plotted_patch in range(len(mappings)):
        if is_vector_valued:
            grid_x1 = Vh.patch_spaces[plotted_patch].spaces[0].breaks[0]
            grid_x2 = Vh.patch_spaces[plotted_patch].spaces[1].breaks[1]
        else:
            grid_x1 = Vh.patch_spaces[plotted_patch].spaces[0].breaks
            grid_x2 = Vh.patch_spaces[plotted_patch].spaces[1].breaks

        x1 = refine_array_1d(grid_x1, N)
        x2 = refine_array_1d(grid_x2, N)

        x1, x2 = np.meshgrid(x1, x2, indexing='ij')
        x, y = F[plotted_patch](x1, x2)

        gridlines_x1 = (x[:, ::N], y[:, ::N])
        gridlines_x2 = (x[::N, :].T, y[::N, :].T)
    else:
        gridlines_x1 = None
        gridlines_x2 = None

    return gridlines_x1, gridlines_x2

# ------------------------------------------------------------------------------

def get_patch_boundary_gridlines(Vh, N, plotted_patch=-1):

    is_vector_valued = Vh.is_vector_valued
    V = Vh.symbolic_space
    domain = V.domain
    mappings = domain.mappings

    F = [M.get_callable_mapping() for M in mappings.values()]

    if plotted_patch in range(len(mappings)):
        if is_vector_valued:
            grid_x1 = Vh.patch_spaces[plotted_patch].spaces[0].breaks[0]
            grid_x2 = Vh.patch_spaces[plotted_patch].spaces[1].breaks[1]
        else:
            grid_x1 = Vh.patch_spaces[plotted_patch].spaces[0].breaks
            grid_x2 = Vh.patch_spaces[plotted_patch].spaces[1].breaks

        x1 = refine_array_1d(grid_x1, N)
        x2 = refine_array_1d(grid_x2, N)

        line1x = np.array([F[plotted_patch](xi, grid_x2[0]) for xi in x1]).T
        line2x = np.array([F[plotted_patch](xi, grid_x2[-1]) for xi in x1]).T
        line1y = np.array([F[plotted_patch](grid_x1[0], yi) for yi in x2]).T
        line2y = np.array([F[plotted_patch](grid_x1[-1], yi) for yi in x2]).T

    else:
        line1x = None
        line2x = None
        line1y = None
        line2y = None

    return line1x, line2x, line1y, line2y

# ------------------------------------------------------------------------------

def plot_2d(funs, titles=None, suptitle=None, xx=None, yy=None,
        plot_type='contourf', components=True, magnitude=False,
        N_vis=(100,100),
        plot_spline_grid=False, spline_grid=None, plot_patch_boundaries=False, patch_boundaries=None,
        cmap='jet', layout=None,
        show_plot=True, filename=None,
        **kwargs
        ):
    """
    Plot one or many FemFields (with a 2d domain) or grid values (on a 2d grid).

    FemFields will always be plotted over their own domain (regardless of xx & yy passed).
    Grid values will by default be plotted over the grid specified by xx & yy, 
    unless a plot specific grid is passed in the corresponding plot specific dictionary.
    Similarly, other settings to be applied to all plots in the Figure are to be passed to this function as kwargs, 
    settings for individual plots are to be passed in the plot specific dictionary, see example usage below.
    Many additional, not listed, kwargs can be passed, see example usage below.

    Parameters
    ----------
    funs : FemField | dict | list | tuple
        Either a FemField (plot settings as passed to this function), 
        or a list of grid values (plot settings as passed to this function, requires xx & yy),
        or a dictionary, 
            either correspondong to a FemField, containing 'fem_field' as key,
            or corresponding to grid values, containing at least 'vals' as key,
            possibly with additional keys corresponding to plot settings overwriting the global plot settings passed to this function,
        or a list or tuple of the above.

    titles : list | tuple | None
        (optional) list of strings (titles) for each plot. Vector-valued FemFields might require 2 titles (if components==True).

    suptitle : str | None
        Suptitle of the Figure.

    xx : list | None
        numpy meshgrid like, x-values of the grid. Ignored by FemFields. Used for grid values unless locally overwritten.

    yy : list | None
        numpy meshgrid like, y-values of the grid. Ignored by FemFields. Used for grid values unless locally overwritten.

    plot_type : str
        Either 'contourf', 'surface_plot' or 'vector_field'. Determines which mpl function is used for the individual plots.

    components : str | bool
        Relevant only for vector-valued FemFields. Determines whether both components (True) are plotted individually (2 plots),
        or only one component ('x' or 'y') (1 plot).
        If False, we assert magnitude and plot its magnitude (1 plot).

    magnitude : bool
        If True, the absolute values of the FemFields are plotted. In the vector-valued case, depending on components, 
        either the magnitude of both components (2 plots) or the magnitude (\\ell^2 norm) of the entire function (1 plot).

    N_vis : list | tuple
        of two ints. Determines the interpolation points (per patch) used for the plot.

    plot_spline_grid : bool
        If True, the spline cells are visualized using line plots. 
        Grid value plots require in addition spline_grid, whereas FemField plots don't.

    spline_grid : list | None
        list of lists corresponding to spline grid lines. Required for grid values plots, ignored by FemField plots.

    plot_patch_boundaries : bool
        If True, the patch boundaries are visualized using line plots.
        Grid value plots require in addition patch_boundaries, whereas FemField plots don't.

    patch_boundaries : list | None
        list of lists corresponding to patch boundary lines. Required for grid value plots, ignored by FemField plots.

    layout : list | tuple
        of two ints. Determines the amount of rows and columns of the Figure respectively. Optional as "good" layout is chosen automatically.

    cmap : matplotlib.colormap
        Choose among 'viridis', 'plasma', 'inferno', 'magma', 'cividis' and many more. (See mpl docs)

    show_plot : bool
        Determines whether we call plt.show(). Must be set to False if one wishes to make additional changes to the returned Figure.

    filename : str | None
        The Figure will be saved if a filename is passed.

    **kwargs : dict
        Extra settings. See this function, and the plot function below, for usage. Currently implemented: 
        vf_skip, amp_factor, dpi, tight_layout, figsize, save_vals

    Returns
    -------
    fig : matplotlib.Figure
        Set show_plot to False in order to manipulate the returned Figure.

    Examples
    --------
    >>> plot_fields_2d((F, {'fem_field':G, 'plot_type':'surface_plot'}), titles=('F', 'G - Surface Plot'), cbar='magma', patch_boundary_linewidth=1)

    F and G are FemFields. The code produces a Figure with 2 to 4 plots depending on whether F and G are scalar- or vector-valued.
    The components (1 or 2) of F are visualized using a contourf plot (default plot_type).
    The components (1 or 2) of G are visualized using a surface plot. This G-specific setting is passed to the function by changing the fem_fields arg 
    from the expected (F, G) to (F, {'fem_field':G, 'plot_type':'surface_plot'}).
    Titles for each plot are passed ('F' and 'G - Surface Plot'). No Figure title (suptitle) is passed.
    The default colorbar is 'viridis'. This default is overwritten for all 2 to 4 plots by passing cbar='magma'.
    An additional kwarg is passed: patch_boundary_linewidth=1. It does not appear in the documentation for this function.
    It is on of many additional kwargs that can be passed for more subtle changes.
    In the plot function below, instead of hardcoding these vast options, e.g. patch_boundary_linewidth = 2, we write 
    patch_boundary_linewidth = kwargs.get('patch_boundary_linewidth', 2).
    
    """

    # Store global settings (used for all plots unless overwritten)
    global_kwargs = {'plot_type':plot_type, 'components':components, 'magnitude':magnitude, 
                     'N_vis':N_vis, 'xx':xx, 'yy':yy,
                     'plot_spline_grid':plot_spline_grid, 'spline_grid':spline_grid, 
                     'plot_patch_boundaries':plot_patch_boundaries, 'patch_boundaries':patch_boundaries,
                     'cmap':cmap,}
    global_kwargs.update(kwargs)

    # Handle case in which a single FemField or grid-value instance is passed
    if isinstance (funs, (list)):
        try:
            # Check if only grid values are passed.
            # This only works if only vector valued grid data or only scalar valued grid data is passed as else the dimensions don't match
            funs_np = np.array(funs)
            # If single instance of scalar-valued grid data -> make tuple
            if len(funs_np.shape) == 3:
                funs = (funs, )
                if titles is not None:
                    assert isinstance(titles, str)
                    titles = (titles, )
            # If either several instances of scalar-valued grid data, or vector-valued grid data
            if len(funs_np.shape) == 4:
                # If either 2 instances of scalar-valued grid data or single instance of vector-valued grid data:
                # Assume it's single instance of vector-valued grid data. There's no way of knowing, also it shouldn't make a difference
                # -> make tuple
                if funs_np.shape[0] == 2:
                    funs = (funs, )
                if titles is not None:
                    assert isinstance(titles, (list, tuple))
        except:
            pass
    if isinstance(funs, FemField) or isinstance(funs, dict):
        funs = (funs, )
        assert titles is None or isinstance(titles, str)
        if titles is not None:
            titles = (titles, )

    # Gather information on the amount of plots (per fem_field) to generate
    plots_per_fem_field = []
    for fun in funs:
        if isinstance(fun, dict):
            # If the dict corresponds to a plot of a FemField:
            if fun.get('fem_field', None) is not None:
                vh = fun['fem_field']
                is_vector_valued = True if vh.space.is_vector_valued else False
                if fun.get('components', None) is not None:
                    is_components_plot = True if fun.get('components', False) == True else False
                else:
                    is_components_plot = True if global_kwargs.get('components', False) == True else False
            # Else the dict fun corresponds to a grid value plot
            else:
                vals = fun.get('vals', None)
                assert vals is not None
                is_components_plot = True # pass only component data if you want to plot individual components
                is_vector_valued = len(vals) == 2
        else:
            # Else fun is a FemField or a list of grid value data
            if isinstance(fun, FemField):
                vh = fun
                is_vector_valued = True if vh.space.is_vector_valued else False
                is_components_plot = True if global_kwargs['components'] else False
            else:
                assert isinstance(fun, list)
                vals = fun
                is_components_plot = True # pass only component data if you want to plot individual components
                is_vector_valued = len(vals) == 2
        if is_vector_valued and is_components_plot:
            plots_per_fem_field.append(2)
        else:
            plots_per_fem_field.append(1)
    total_nb_plots = sum(plots_per_fem_field)

    # Use above information to create layout if not already passed
    if layout is not None:
        assert layout[0]*layout[1] >= total_nb_plots
        nb_rows = layout[0]
        nb_cols = layout[1]
    else:
        nb_rows = int(np.floor(np.sqrt(total_nb_plots)))
        nb_cols = int(np.ceil(total_nb_plots/nb_rows))
        layout = (nb_rows, nb_cols)

    # Create Figure and set Figure title
    figsize = global_kwargs.pop('figsize', (2.6 + 4.8 * layout[1], 4.8 * layout[0]))
    fig = plt.figure(figsize=figsize)

    if suptitle is not None:
        suptitle_size = kwargs.get('suptitle_size', 14)
        fig.suptitle(suptitle, fontsize=suptitle_size)

    # Generate the individual plots
    count = 0
    for i, fun in enumerate(funs):

        # Check if fun corresponds to a FemField or grid & grid-values
        is_fem_field = isinstance(fun, FemField) or (isinstance(fun, dict) and (fun.get('fem_field', None) is not None))

        if is_fem_field:
            # Update kwargs
            if isinstance(fun, dict):
                vh = fun.pop('fem_field')
                local_kwargs = global_kwargs.copy()
                local_kwargs.update(fun)
            else:
                vh = fun
                local_kwargs = global_kwargs.copy()

            local_kwargs.pop('xx')
            local_kwargs.pop('yy')

            # Get grid and vals
            Vh            = vh.space
            V             = Vh.symbolic_space
            domain        = V.domain
            mappings      = domain.mappings
            mappings_list = list(mappings.values())

            local_N_vis   = local_kwargs.pop('N_vis')
            etas, xx, yy  = get_plotting_grid(mappings, N=local_N_vis)
            vh_vals       = get_grid_vals(vh, etas, mappings_list)

            # Create plot_vals from vh_vals based on values of 'components' and 'magnitude'
            local_components = local_kwargs.pop('components')
            local_magnitude  = local_kwargs.pop('magnitude')
            is_vector_valued = Vh.is_vector_valued

            if is_vector_valued:
                if local_components == True:
                    if local_magnitude:
                        plot_vals = np.abs(vh_vals)
                    else:
                        plot_vals = vh_vals
                elif local_components in ('x', 'y'):
                    if local_magnitude:
                        plot_vals = (np.abs(vh_vals[0]), ) if local_components == 'x' else (np.abs(vh_vals[1]), )
                    else:
                        plot_vals = (vh_vals[0], ) if local_components == 'x' else (vh_vals[1], )
                else:
                    if local_kwargs['vector_field']:
                        plot_vals = vh_vals
                    else:
                        assert local_magnitude
                        plot_vals = [np.sqrt(abs(v[0])**2 + abs(v[1])**2)
                                    for v in zip(vh_vals[0], vh_vals[1])]
            else:
                if local_magnitude:
                    plot_vals = [np.abs(vh_vals)]
                else:
                    plot_vals = [vh_vals]

            # Obtain spline grid
            local_plot_spline_grid = local_kwargs.pop('plot_spline_grid')
            if local_plot_spline_grid or local_kwargs.get('spline_grid', None) is not None:
                if local_kwargs.get('spline_grid', None) is None:
                    local_kwargs.pop('spline_grid')
                    spline_grid_on_patches = local_kwargs.pop('spline_grid_on_patches', range(0, len(mappings)))
                    spline_grid = [get_patch_knots_gridlines(Vh, 100, k) if k in spline_grid_on_patches else None for k in range(len(mappings))]
                else:
                    spline_grid_on_patches = local_kwargs.pop('spline_grid_on_patches', range(0, len(mappings)))
                    spline_grid = local_kwargs.pop('spline_grid')
                    spline_grid = [spline_grid[k] if k in spline_grid_on_patches else None for k in range(len(mappings))]
            else:
                local_kwargs.pop('spline_grid')
                spline_grid = None

            # Obtain patch boundaries
            local_plot_patch_boundaries = local_kwargs.pop('plot_patch_boundaries')
            if local_plot_patch_boundaries or local_kwargs.get('patch_boundaries', None) is not None:
                if local_kwargs.get('patch_boundaries', None) is None:
                    local_kwargs.pop('patch_boundaries')
                    patch_boundaries_on_patches = local_kwargs.pop('patch_boundaries_on_patches', range(0, len(mappings)))
                    patch_boundaries = [get_patch_boundary_gridlines(Vh, 100, k) if k in patch_boundaries_on_patches else None for k in range(len(mappings))]
                else:
                    patch_boundaries_on_patches = local_kwargs.pop('patch_boundaries_on_patches', range(0, len(mappings)))
                    patch_boundaries = local_kwargs.pop('patch_boundaries')
                    patch_boundaries = [patch_boundaries[k] if k in patch_boundaries_on_patches else None for k in range(len(mappings))]

            else:
                local_kwargs.pop('patch_boundaries')
                patch_boundaries = None

        # else fun is a dict corresponding to grid & grid-values data
        else:
            #xx = fun.pop('xx')
            #yy = fun.pop('yy')
            if isinstance(fun, dict):
                vals = fun.pop('vals')
                # Update kwargs
                local_kwargs = global_kwargs.copy()
                local_kwargs.update(fun)
            else:
                vals = fun
                local_kwargs = global_kwargs.copy()

            plot_vals = vals
            
            #plot_vals = fun.pop('vals')
            plot_vals_np = np.array(plot_vals)
            if len(plot_vals_np.shape) == 3:
                plot_vals = (plot_vals, )
            else:
                assert len(plot_vals_np.shape) == 4
            #if not isinstance(plot_vals[0][0, 0], Iterable):
            #    plot_vals = (plot_vals, )

            xx = local_kwargs.pop('xx')
            yy = local_kwargs.pop('yy')

            spline_grid = local_kwargs.pop('spline_grid', None)
            if spline_grid is not None:
                spline_grid_on_patches = local_kwargs.pop('spline_grid_on_patches', range(0, len(spline_grid)))
                spline_grid = [spline_grid[k] if k in spline_grid_on_patches else None for k in range(len(spline_grid))]
            
            patch_boundaries = local_kwargs.pop('patch_boundaries', None)
            if patch_boundaries is not None:
                patch_boundaries_on_patches = local_kwargs.pop('patch_boundaries_on_patches', range(0, len(patch_boundaries)))
                patch_boundaries = [patch_boundaries[k] if k in patch_boundaries_on_patches else None for k in range(len(patch_boundaries))]

        # Generate the plot(s)
        for j in range(plots_per_fem_field[i]):

            # Get title if given
            if titles is not None:
                title = titles[count]
            else:
                title = None

            # Add 2d or 3d axes to the Figure
            if local_kwargs['plot_type'] == 'surface_plot':
                ax = fig.add_subplot(*layout, count+1, projection='3d')
            else:
                ax = fig.add_subplot(*layout, count+1)

            plot(ax, xx, yy, plot_vals[j], 
                 title=title, spline_grid=spline_grid, patch_boundaries=patch_boundaries, index=count, **local_kwargs)
            count += 1

    # Missing code that removes empty axes
    #axs[1, 2].remove()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight') # , dpi=dpi)

    if global_kwargs.get('tight_layout', True):
        fig.tight_layout()

    if show_plot:
        plt.show()

    return fig

# ------------------------------------------------------------------------------

def plot(ax, xx, yy, vals, 
        title=None, plot_type='contourf', cmap='viridis', spline_grid=None, patch_boundaries=None,
        vf_skip=2,
        amp_factor=1,
        **kwargs
):
    """
    kwargs can be: title_size, save_vals, index, vmin, vmax, contourf_levels, contourf_zorder, contourf_extend,
    spline_grid_color, spline_grid_linewidth, patch_boundaries_linewidth, patch_boundaries_color, rastarization_zorder,
    show_xylabel, xlabel, ylabel, xlabel_rotation, ylabel_rotation, aspect, cbar,
    contour, contour_levels, contour_zorder, contour_cmap, contour_colors
    """
    
    # Save vals as f'vals{index}.npz'
    save_vals = kwargs.get('save_vals', False)
    if save_vals:
        index = kwargs.get('index', '')
        np.savez(f'vals{index}', xx=xx, yy=yy, vals=vals)

    # Set title
    if title is not None:
        title_size = kwargs.get('title_size', 14)
        ax.set_title(title, fontsize=title_size)

    # Essential to guarantee continuous colors along patch interfaces
    vmin  = kwargs.get('vmin', np.min(vals))
    vmax  = kwargs.get('vmax', np.max(vals))
    cnorm = colors.Normalize(vmin=vmin, vmax=vmax)

    n_patches = len(xx)

    if plot_type == 'contourf':
        contourf_levels            = kwargs.get('contourf_levels', 50)
        contourf_zorder            = kwargs.get('contourf_zoder', -10)
        contourf_extend            = kwargs.get('contourf_extend', 'neither')
        spline_grid_color          = kwargs.get('spline_grid_color', 'k')
        spline_grid_linewidth      = kwargs.get('spline_grid_linewidth', 1)
        patch_boundaries_color     = kwargs.get('patch_boundaries_color', 'blueviolet')
        patch_boundaries_linewidth = kwargs.get('patch_boundaries_linewidth', 2)

        contour = kwargs.get('contour', False)
        if contour:
            contour_levels = kwargs.get('contour_levels', 10)
            contour_zorder = kwargs.get('contour_zorder', contourf_zorder+1)
            contour_cmap   = kwargs.get('contour_cmap', None)
            contour_colors = kwargs.get('contour_colors', 'k')

        for k in range(n_patches):
            ax.contourf(xx[k], yy[k], vals[k], levels=contourf_levels, norm=cnorm, cmap=cmap, zorder=contourf_zorder, extend=contourf_extend)
            if contour:
                ax.contour(xx[k], yy[k], vals[k], levels=contour_levels, norm=cnorm, cmap=contour_cmap, zorder=contour_zorder, colors=contour_colors)
            
            if spline_grid is not None:
                if spline_grid[k] is not None:
                    ax.plot(*spline_grid[k][0], color=spline_grid_color, linewidth=spline_grid_linewidth)
                    ax.plot(*spline_grid[k][1], color=spline_grid_color, linewidth=spline_grid_linewidth)

            if patch_boundaries is not None:
                if patch_boundaries[k] is not None:
                    ax.plot(*patch_boundaries[k][0], color=patch_boundaries_color, linewidth=patch_boundaries_linewidth)
                    ax.plot(*patch_boundaries[k][1], color=patch_boundaries_color, linewidth=patch_boundaries_linewidth)
                    ax.plot(*patch_boundaries[k][2], color=patch_boundaries_color, linewidth=patch_boundaries_linewidth)
                    ax.plot(*patch_boundaries[k][3], color=patch_boundaries_color, linewidth=patch_boundaries_linewidth)

        rastarization_zorder = kwargs.get('rastarization_zorder', 0)
        ax.set_rasterization_zorder(rastarization_zorder)

    elif plot_type == 'surface_plot':

        for k in range(n_patches):
            ax.plot_surface(
                xx[k],
                yy[k],
                vals[k],
                norm=cnorm,
                rstride=10,
                cstride=10,
                cmap=cmap,
                linewidth=0,
                antialiased=False,
                #levels=50
                )

    # Add the colorbar
    cbar = kwargs.get('cbar', True)
    if cbar:
        plt.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap), ax=ax, pad=0.05)

    # Set x & ylabel
    show_xylabel = kwargs.get('show_xylabel', True) 
    if show_xylabel:
        xlabel          = kwargs.get('xlabel', r'$x$')
        ylabel          = kwargs.get('ylabel', r'$y$')
        xlabel_rotation = kwargs.get('xlabel_rotation', 'horizontal')
        ylabel_rotation = kwargs.get('ylabel_rotation', 'horizontal')
        ax.set_xlabel(xlabel, rotation=xlabel_rotation)
        ax.set_ylabel(ylabel, rotation=ylabel_rotation)

    aspect = kwargs.get('aspect', 'equal')
    ax.set_aspect(aspect)

# ------------------------------------------------------------------------------

def my_small_streamplot(
        title, vals_x, vals_y,
        xx, yy, skip=2,
        amp_factor=1,
        save_fig=None,
        show_plot=True,
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

    if show_plot:
        plt.show()
