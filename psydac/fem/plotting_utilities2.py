#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
import  numpy                   as np
import  matplotlib.pyplot       as plt
from    matplotlib              import cm, colors

from    psydac.utilities.utils  import refine_array_1d
from    psydac.feec.pull_push   import push_2d_h1_vec, push_2d_h1, push_2d_hcurl, push_2d_hdiv, push_2d_l2
from    psydac.fem.basic        import FemField, FemSpace

__all__ = (
    'get_grid_vals',
    'get_plotting_grid',
    'get_patch_knots_gridlines',
    'get_patch_boundary_gridlines',
    'plot_2d',
    'fill_axes')

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
        u_component_patch_vals = [n_patches * [None], n_patches * [None]]
    else:
        u_component_patch_vals = [n_patches * [None]]

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
        verbose=False, # to be deleted
        **kwargs
        ):
    """
    Plot one or many FemFields (with a 2d domain) or grid values (on a 2d grid).

    FemFields will always be plotted over their own domain (regardless of xx & yy passed).
    Grid values will by default be plotted over the grid specified by xx & yy, 
    unless a plot specific grid is passed in the corresponding plot specific dictionary.
    Similarly, other settings to be applied to all plots in the Figure are to be passed to this function as kwargs, 
    settings for individual plots are to be passed in the plot specific dictionary, see example usage below.
    Many additional, not listed, kwargs can be passed.

    Parameters
    ----------
    funs : FemField | dict | list | tuple
        Either a FemField, 
        or a list of grid values (shape: (#components, #patches, #x-grid points, #y-grid points), requires xx & yy),
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
        Relevant only for vector-valued FemFields. Determines whether both components (True) are plotted individually (2 plots, unless plot_type='vector_plot', then 1 plot),
        or only one component ('x' or 'y') (1 plot).
        If False, we assert magnitude and plot its magnitude (1 plot).

    magnitude : bool
        If True, the absolute values of the FemFields are plotted. In the vector-valued case, depending on components, 
        either the magnitude of both components (2 plots) or the magnitude (\\ell^2 norm) of the entire function (1 plot).

    N_vis : list | tuple
        of two ints. Determines the interpolation points (per patch) used for the plot.

    plot_spline_grid : bool | tuple | list
        If True, the spline cells are visualized (on all patches) using line plots. 
        Grid value plots require in addition spline_grid, whereas FemField plots don't. If list or tuple of patch indices, plot on these patches only.

    spline_grid : list | psydac.fem.basic.FemSpace | None
        list of lists corresponding to spline grid lines (per patch, for all patches) as (patch-wise) returned by get_patch_knots_gridlines, 
        or FemSpace from which the spline grid is to be obtained from.
        Required for grid value plots in case of plot_spline_grid. Overwrites plot_spline_grid=False in that case. Ignored by FemField plots.

    plot_patch_boundaries : bool | tuple | list
        If True, the patch boundaries are visualized (on all patches) using line plots.
        Grid value plots require in addition patch_boundaries, whereas FemField plots don't. If list or tuple of patch indices, plot on these patches only.

    patch_boundaries : list | psydac.fem.basic.FemSpace | None
        list of lists corresponding to patch boundary lines as returned (patch-wise) by get_patch_boundary_gridlines, 
        or FemSpace from which the patch boundaries are to be obtained from. 
        Required for grid value plots in case of plot_patch_boundaries. Overwrites plot_patch_boundaries=False in that case. Ignored by FemField plots.

    layout : list | tuple
        of two ints. Determines the amount of rows and columns of the Figure respectively. Optional as "good" layout is chosen automatically.

    cmap : matplotlib.colormap
        Choose among 'viridis', 'plasma', 'inferno', 'magma', 'cividis' and many more. (See mpl docs)

    show_plot : bool
        Determines whether we call plt.show(). Must be set to False if one wishes to make additional changes to the returned Figure.

    filename : str | None
        The Figure will be saved if a filename is passed.

    **kwargs : dict
        Extra settings to be applied to all plots. Currently implemented: 
        figsize, suptitle_size, tight_layout & 
        all additional kwargs implemented for fil_axes in this file.

    Returns
    -------
    fig : matplotlib.Figure
        Set show_plot to False in order to manipulate the returned Figure.

    Examples
    --------
    >>> plot_fields_2d(funs   = (F, {'fem_field':G, 'plot_type':'surface_plot'}, vals_H, {'vals':vals_I, 'cmap':'jet'}), 
                       titles = ('F', 'G - Surface Plot', 'H', 'I'), 
                       xx     = xx, 
                       yy     = yy,
                       cmap   = 'magma', 
                       plot_patch_boundaries = True,
                       patch_boundary_linewidth = 1)

    F and G are FemFields. vals_H and vals_I are grid data.
    The code produces a Figure with 4 to 8 plots depending on whether F, G, H and I are scalar- or vector-valued.
    The components (1 or 2) of F, H and I are visualized using a contourf plot (default plot_type).
    The components (1 or 2) of G are visualized using a surface plot. This G-specific setting is passed to the function 
    by passing {'fem_field':G, 'plot_type':'surface_plot'} rather than only G.
    The FemFields F and G are plotted over their own respective domain, vals_H and vals_I are plotted over xx & yy.
    Titles for each plot are specified. No Figure title (suptitle) is specified.
    The default colormap is 'jet'. This default is overwritten for all 4 to 8 plots by setting cmap='magma'.
    A plot-specific colormap is chosen for vals_I by locally overwriting the global setting.
    The plot_patch_boundaries setting will automatically add patch boundaries to the FemField plots.
    Grid value plots would require additional patch_boundaries information.
    An additional kwarg is passed: patch_boundary_linewidth=1. It does not appear in the documentation for this function.
    It is on of many additional kwargs that can be passed for more subtle changes to fill_axes in this file.
    In the fill_axes function below, instead of hardcoding these vast options, e.g. patch_boundary_linewidth = 2, we write 
    patch_boundary_linewidth = kwargs.get('patch_boundary_linewidth', 2).
    
    """

    # Store global settings (used for all plots unless overwritten)
    global_kwargs = {'plot_type':plot_type, 'components':components, 'magnitude':magnitude, 
                     'N_vis':N_vis, 'xx':xx, 'yy':yy,
                     'plot_spline_grid':plot_spline_grid, 'spline_grid':spline_grid, 
                     'plot_patch_boundaries':plot_patch_boundaries, 'patch_boundaries':patch_boundaries,
                     'cmap':cmap,}
    global_kwargs.update(kwargs)

    # -----
    # Handle case in which funs is not a list or tuple of functions, but rather a FemField or grid data. 
    # In the latter case, write funs = (funs, ), and possibly do the same to titles.
    if isinstance (funs, (list, tuple)):
        try:
            # Check if only grid values are passed.
            # This only works if only vector valued grid data or only scalar valued grid data is passed as else the dimensions don't match
            funs_np    = np.array(funs)
            funs_shape = funs_np.shape

            # (patches, x, y) -> 3
            is_single_scalar_valued_grid_data = True if len(funs_shape) == 3 else False
            # (2 components, patches, x, y) -> 4 & [0]==2 (could also be a list/tuple of exactly 2 scalar-avlued function grid data)
            is_single_vector_valued_grid_data = True if len(funs_shape) == 4 and funs_shape[0] == 2 else False # or 2 scalar-valued grid data

            if verbose: # to be deleted
                if is_single_scalar_valued_grid_data:
                    print('Single scalar-valued grid data')
                if is_single_vector_valued_grid_data:
                    print('Single vector-valued grid data')

            if is_single_scalar_valued_grid_data or is_single_vector_valued_grid_data:
                funs = (funs, )
                if titles is not None:
                    assert isinstance(titles, str)
                    titles = (titles, )
        except:
            pass
    elif isinstance(funs, FemField):
        funs = (funs, )
        if titles is not None:
            assert isinstance(titles, str)
            titles = (titles, )
        if verbose: # to be deleted
            print('Single FemField')
    elif isinstance(funs, dict):
        funs = (funs, )
        if titles is not None:
            assert isinstance(titles, str)
            titles = (titles, )
        if verbose: # to be deleted
            print('Single dict')
    else:
        raise ValueError(f'funs not understood.')
    # -----

    # -----
    # Gather information on the amount of plots (per fun) to generate
    plots_per_fun = []
    for fun in funs:

        is_fem_field = isinstance(fun, FemField) or (isinstance(fun, dict) and fun.get('fem_field', None) is not None)
        is_grid_data = isinstance(fun, (list, tuple)) or (isinstance(fun, dict) and fun.get('vals', None) is not None)
        is_dict      = isinstance(fun, dict)

        if is_fem_field:
            vh = fun['fem_field'] if is_dict else fun
            is_vector_valued = True if vh.space.is_vector_valued else False
            comp = fun.get('components', global_kwargs.get('components')) if is_dict else global_kwargs.get('components')
            pt   = fun.get('plot_type', global_kwargs.get('plot_type'))   if is_dict else global_kwargs.get('plot_type')
            # A vector-valued FemField will generate 2 plots unless (components==False and magnitude==True) or (plot_type=='vector_field')
            if is_vector_valued:
                if comp == True and pt != 'vector_field':
                    plots_per_fun.append(2)
                else:
                    if comp == False:
                        mag = fun.get('magnitude', global_kwargs.get('magnitude')) if is_dict else global_kwargs.get('magnitude')
                        assert mag == True, f'Components=False must be acompanied by magnitude=True for vector-valued functions'
                    plots_per_fun.append(1)
            else:
                plots_per_fun.append(1)
        else:
            assert is_grid_data
            vals = fun.get('vals') if is_dict else fun
            vals_np = np.array(vals)
            vals_shape = vals_np.shape
            # (2 components, patches, x, y) -> 4
            is_vector_valued = True if len(vals_shape) == 4 else False
            # comp = True (in case of grid data, convert to abs value manually if required)
            pt = fun.get('plot_type', global_kwargs.get('plot_type')) if is_dict else global_kwargs.get('plot_type')
            # Hence, the only exception of vector-valued data implying 2 plots: A vector_field plot!
            if is_vector_valued:
                if pt != 'vector_field':
                    plots_per_fun.append(2)
                else:
                    plots_per_fun.append(1)
            else:
                plots_per_fun.append(1)
    if verbose: # to be deleted
        print(f'{plots_per_fun = }')
    total_nb_plots = sum(plots_per_fun)
    # -----

    # -----
    # Use above information to create layout if not already passed
    if layout is not None:
        assert layout[0]*layout[1] >= total_nb_plots
        nb_rows = layout[0]
        nb_cols = layout[1]
    else:
        nb_rows = int(np.floor(np.sqrt(total_nb_plots)))
        nb_cols = int(np.ceil(total_nb_plots/nb_rows))
        layout = (nb_rows, nb_cols)
    # -----

    # -----
    # Create Figure, set figsize based on layout, and set Figure title
    figsize_default = (2.6 + 4.8 * layout[1], 4.8 * layout[0])
    figsize         = global_kwargs.pop('figsize', figsize_default)
    fig             = plt.figure(figsize=figsize)

    if suptitle is not None:
        suptitle_size = kwargs.get('suptitle_size', 14)
        fig.suptitle(suptitle, fontsize=suptitle_size)
    # -----

    # -----
    # Generate the individual plots
    count = 0

    for i, fun in enumerate(funs):

        # ---
        # Check if fun corresponds to a FemField or grid values
        is_fem_field = isinstance(fun, FemField) or (isinstance(fun, dict) and (fun.get('fem_field', None) is not None))
        # ---

        # ---
        # Get all plotting relevant data
        if is_fem_field:
            # -
            # Update kwargs
            if isinstance(fun, dict):
                vh           = fun.pop('fem_field')
                local_kwargs = global_kwargs.copy()
                local_kwargs.update(fun)
            else:
                vh           = fun
                local_kwargs = global_kwargs.copy()

            # xx and yy are ignored by FemFields
            local_kwargs.pop('xx')
            local_kwargs.pop('yy')
            # -

            # -
            # Get grid and vals
            Vh            = vh.space
            V             = Vh.symbolic_space
            domain        = V.domain
            mappings      = domain.mappings
            mappings_list = list(mappings.values())

            N_vis         = local_kwargs.pop('N_vis')
            etas, xx, yy  = get_plotting_grid(mappings, N=N_vis)
            vh_vals       = get_grid_vals(vh, etas, mappings_list)
            # -

            # -
            # Create plot_vals from vh_vals based on values of 'components', 'magnitude' and 'plot_type'
            components           = local_kwargs.pop('components')
            magnitude            = local_kwargs.pop('magnitude')
            is_vector_field_plot = local_kwargs.get('plot_type') == 'vector_field'
            is_vector_valued     = Vh.is_vector_valued

            if is_vector_field_plot:
                if magnitude:                           # vector field plot of v = ( |v_x|, |v_y| ) - probably rarely used
                    plot_vals = (np.abs(vh_vals), )
                else:                                   # vector field plot of v = (  v_x ,  v_y  )
                    plot_vals = (vh_vals, )
            else:
                if is_vector_valued:
                    if components == True:
                        if magnitude:                   # 2 plots corresponding to |v_x| and |v_y|
                            plot_vals = np.abs(vh_vals)
                        else:                           # 2 plots corresponding to  v_x  and  v_y
                            plot_vals = vh_vals
                    elif components in ('x', 'y'):      # 1 plot only of either v_x, v_y, |v_x| or |v_y|
                        if magnitude:
                            plot_vals = (np.abs(vh_vals[0]), ) if components == 'x' else (np.abs(vh_vals[1]), )
                        else:
                            plot_vals = (       vh_vals[0] , ) if components == 'x' else (       vh_vals[1] , )
                    else:                               # 1 plot of || (v_x, v_y) ||
                        assert magnitude
                        plot_vals = ([np.sqrt(abs(v[0])**2 + abs(v[1])**2) for v in zip(vh_vals[0], vh_vals[1])], )
                else:                                   # v is scalar-valued
                    if magnitude:                       # 1 plot of |v|
                        plot_vals = (np.abs(vh_vals), )
                    else:                               # 1 plot of  v
                        plot_vals = (vh_vals, )
            # -

            # -
            # Obtain spline grid
            plot_spline_grid = local_kwargs.pop('plot_spline_grid')
            local_kwargs.pop('spline_grid')
            if plot_spline_grid is not False:
                spline_grid_on_patches = plot_spline_grid if isinstance(plot_spline_grid, (list, tuple)) else range(len(mappings))
                spline_grid = [get_patch_knots_gridlines(Vh, 100, k) if k in spline_grid_on_patches else None for k in range(len(mappings))]
            else:
                spline_grid = None
            # -

            # -
            # Obtain patch boundaries
            plot_patch_boundaries = local_kwargs.pop('plot_patch_boundaries')
            local_kwargs.pop('patch_boundaries')
            if plot_patch_boundaries is not False: #  or local_kwargs.get('patch_boundaries', None) is not None:
                patch_boundaries_on_patches = plot_patch_boundaries if isinstance(plot_patch_boundaries, (list, tuple)) else range(len(mappings))
                patch_boundaries = [get_patch_boundary_gridlines(Vh, 100, k) if k in patch_boundaries_on_patches else None for k in range(len(mappings))]
            else:
                patch_boundaries = None
            # -

        else: # fun corresponds to grid values
            # -
            # Update kwargs
            if isinstance(fun, dict):
                vals         = fun.pop('vals')
                local_kwargs = global_kwargs.copy()
                local_kwargs.update(fun)
            else:
                vals         = fun
                local_kwargs = global_kwargs.copy()
            # -

            # -
            # Get plot_vals from vals --- Pass absolute values manually if required
            vals_np    = np.array(vals)
            vals_shape = vals_np.shape
            # (patches, x, y) -> 3
            is_scalar_valued = len(vals_shape) == 3
            # (2 components, patches, x, y) -> 4
            is_vector_valued = len(vals_shape) == 4
            if is_scalar_valued:
                plot_vals = (vals, )
            else:
                assert is_vector_valued
                plot_vals = vals
            # -

            # -
            # Get grid
            xx = local_kwargs.pop('xx')
            yy = local_kwargs.pop('yy')
            # -

            # -
            # Obtaine spline grid
            plot_spline_grid = local_kwargs.pop('plot_spline_grid')
            spline_grid      = local_kwargs.pop('spline_grid')
            # if spline_grid is None, no spline grid is plotted, even if plot_spline_grid, 
            # because in the case of grid values the "source" of the spline grid must be specified
            if spline_grid is not None:
                if isinstance(spline_grid, FemSpace):
                    Vh                     = spline_grid
                    mappings               = Vh.symbolic_space.domain.mappings
                    spline_grid_on_patches = plot_spline_grid if isinstance(plot_spline_grid, (list, tuple)) else range(0, len(mappings))
                    spline_grid            = [get_patch_knots_gridlines(Vh, 100, k) if k in spline_grid_on_patches else None for k in range(len(mappings))]
                else:
                    spline_grid_on_patches = plot_spline_grid if isinstance(plot_spline_grid, (list, tuple)) else range(0, len(spline_grid))
                    spline_grid            = [sg if k in spline_grid_on_patches else None for k, sg in enumerate(spline_grid)]
            # -
            
            # -
            # Obtain patch boundaries
            plot_patch_boundaries = local_kwargs.pop('plot_patch_boundaries')
            patch_boundaries = local_kwargs.pop('patch_boundaries')
            # if patch_boundaries is None, no patch boundary is plotted, even if plot_patch_boundaries, 
            # because in the case of grid values the "source" of the patch boundaries must be specified
            if patch_boundaries is not None:
                if isinstance(patch_boundaries, FemSpace):
                    Vh                     = patch_boundaries
                    mappings               = Vh.symbolic_space.domain.mappings
                    patch_boundaries_on_patches = plot_patch_boundaries if isinstance(plot_patch_boundaries, (list, tuple)) else range(0, len(mappings))
                    patch_boundaries            = [get_patch_boundary_gridlines(Vh, 100, k) if k in patch_boundaries_on_patches else None for k in range(len(mappings))]
                else:
                    patch_boundaries_on_patches = plot_patch_boundaries if isinstance(plot_patch_boundaries, (list, tuple)) else range(0, len(patch_boundaries))
                    patch_boundaries            = [pb if k in patch_boundaries_on_patches else None for k, pb in enumerate(patch_boundaries)]
            # -
        # ---

        # ---
        # Generate the plot(s)
        for j in range(plots_per_fun[i]):
            title = titles[count] if titles is not None else None

            # Add 2d or 3d axes to the Figure
            if local_kwargs['plot_type'] == 'surface_plot':
                ax = fig.add_subplot(*layout, count+1, projection='3d')
            else:
                ax = fig.add_subplot(*layout, count+1)

            fill_axes(ax, xx, yy, plot_vals[j], 
                 title=title, spline_grid=spline_grid, patch_boundaries=patch_boundaries, index=count, **local_kwargs)
            count += 1
        # ---

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight') # look up dpi keyword and implement, look up what bbox_inches does

    if global_kwargs.get('tight_layout', True):
        fig.tight_layout()

    if show_plot:
        plt.show() # look into fig.show() / fig.canvas.draw() ? 

    return fig

# ------------------------------------------------------------------------------

def fill_axes(ax, xx, yy, vals, 
        title=None, plot_type='contourf', cmap='jet', spline_grid=None, patch_boundaries=None,
        **kwargs
):
    """
    Fill a matplotlib.axes.Axes instance with a plot (vals over the meshgrid xx & yy). Primaritly a tool for plot_2d in this file.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Location of the plot within a Figure.

    xx : list
        numpy meshgrid like, x-values of the grid.

    yy : list
        numpy meshgrid like, x-values of the grid.

    vals : list
        function values to be plotted over xx & yy as returned by get_grid_vals in this file.

    title : str | None
        Title of this plot.

    plot_type : str
        Either contourf, surface_plot or vector_field

    cmap : matplotlib.colormap
        Choose among 'viridis', 'plasma', 'inferno', 'magma', 'cividis' and many more. (See mpl docs)

    spline_grid : list | None
        list of lists corresponding to spline grid lines (per patch) as returned by get_patch_knots_gridlines. May include None to exclude patches.

    patch_boundaries : list | None
        list of lists corresponding to patch boundaries as returned by get_patch_boundary_gridlines. May include None to exclude patches.

    **kwargs : dict
        Extra settings. Currently implemented:
        title_size, 
        save_vals, index, 
        vmin, vmax, 
        contourf_levels, contourf_zorder, contourf_extend,
        spline_grid_color, spline_grid_linewidth, 
        patch_boundaries_color, patch_boundaries_linewidth, 
        rastarization_zorder,
        show_xylabel, xlabel, ylabel, xlabel_rotation, ylabel_rotation, 
        aspect, 
        cbar,
        contour (in addition to contourf), contour_levels, contour_zorder, contour_cmap, contour_colors,
        rstride, cstride,
        vf_skip, vf_skip_x, vf_skip_y, amp_factor, scale, vector_width,
        contourf (in addition to a quiver/vector_field plot)

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axes that was passed to this function.

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

    rastarization_zorder = kwargs.get('rastarization_zorder', 0)
    ax.set_rasterization_zorder(rastarization_zorder)

    n_patches = len(xx)

    if plot_type == 'contourf':
        # Essential to guarantee continuous colors along patch interfaces
        vmin  = kwargs.get('vmin', np.min(vals))
        vmax  = kwargs.get('vmax', np.max(vals))
        cnorm = colors.Normalize(vmin=vmin, vmax=vmax)

        contourf_levels            = kwargs.get('contourf_levels', 50)
        contourf_zorder            = kwargs.get('contourf_zorder', -10)
        contourf_extend            = kwargs.get('contourf_extend', 'neither')
        spline_grid_color          = kwargs.get('spline_grid_color', 'k')
        spline_grid_linewidth      = kwargs.get('spline_grid_linewidth', 1)
        patch_boundaries_color     = kwargs.get('patch_boundaries_color', 'k')
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

    elif plot_type == 'surface_plot':
        # Essential to guarantee continuous colors along patch interfaces
        vmin  = kwargs.get('vmin', np.min(vals))
        vmax  = kwargs.get('vmax', np.max(vals))
        cnorm = colors.Normalize(vmin=vmin, vmax=vmax)

        rstride = kwargs.get('rstride', 2)
        cstride = kwargs.get('cstride', 2)

        for k in range(n_patches):
            ax.plot_surface(
                xx[k],
                yy[k],
                vals[k],
                norm=cnorm,
                rstride=rstride,
                cstride=cstride,
                cmap=cmap,
                linewidth=0,
                antialiased=False,
                )

    elif plot_type == 'vector_field':
        vals_x      = vals[0]
        vals_y      = vals[1]
        abs_vals    = [np.sqrt(abs(v_x)**2 + abs(v_y)**2) for v_x, v_y in zip(vals_x, vals_y)]
        max_val     = np.max(abs_vals)

        vf_skip_x_default          = max(10, int(np.floor(len(vals_x[0])/10)))
        vf_skip_y_default          = max(10, int(np.floor(len(vals_y[0])/10)))

        vf_skip_x                  = kwargs.get('vf_skip_x', kwargs.get('vf_skip', vf_skip_x_default))
        vf_skip_y                  = kwargs.get('vf_skip_y', kwargs.get('vf_skip', vf_skip_y_default))
        amp_factor                 = kwargs.get('amp_factor', 10)
        scale                      = kwargs.get('scale', amp_factor * max_val)
        vector_width               = kwargs.get('vector_width', 0.005)
        if kwargs.get('scale', None) is not None and kwargs.get('amp_factor', None) is not None:
            print(f'Warning: scale overwrites amp_factor.')

        patch_boundaries_color     = kwargs.get('patch_boundaries_color', 'k')
        patch_boundaries_linewidth = kwargs.get('patch_boundaries_linewidth', 2)
        spline_grid_color          = kwargs.get('spline_grid_color', 'k')
        spline_grid_linewidth      = kwargs.get('spline_grid_linewidth', 1)

        for k in range(n_patches):
            ax.quiver(xx[k][::vf_skip_x, ::vf_skip_x],
                      yy[k][::vf_skip_y, ::vf_skip_y],
                      vals_x[k][::vf_skip_x, ::vf_skip_x],
                      vals_y[k][::vf_skip_y, ::vf_skip_y],
                      scale=scale,
                      width=vector_width)

            contourf = kwargs.get('contourf', False)
            if contourf:
                # Essential to guarantee continuous colors along patch interfaces
                vmin  = kwargs.get('vmin', np.min(abs_vals))
                vmax  = kwargs.get('vmax', np.max(abs_vals))
                cnorm = colors.Normalize(vmin=vmin, vmax=vmax)

                contourf_levels = kwargs.get('contourf_levels', 50)
                contourf_zorder = kwargs.get('contourf_zoder', -10)
                contourf_extend = kwargs.get('contourf_extend', 'neither')

                ax.contourf(xx[k], yy[k], abs_vals[k], alpha=0.5, levels=contourf_levels, norm=cnorm, cmap=cmap, zorder=contourf_zorder, extend=contourf_extend)

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

    # Add the colorbar
    cbar_default = True if (plot_type == 'contourf') or (plot_type == 'vector_field' and kwargs.get('contourf', False)) else False
    cbar = kwargs.get('cbar', cbar_default)
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

    aspect_default = 'equal' if plot_type in ('contourf', 'vector_field') else 'auto'
    aspect = kwargs.get('aspect', aspect_default)
    ax.set_aspect(aspect)

    return ax
