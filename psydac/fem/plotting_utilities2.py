#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from __future__ import annotations

import  os
from    typing                  import NamedTuple

import  h5py
import  numpy                   as np
import  matplotlib.pyplot       as plt
from    matplotlib              import cm, colors
from    PIL                     import Image

from    psydac.utilities.utils  import refine_array_1d
from    psydac.feec.pull_push   import push_2d_h1_vec, push_2d_h1, push_2d_hcurl, push_2d_hdiv, push_2d_l2
from    psydac.fem.basic        import FemField, FemSpace
from    psydac.api.postprocessing import PostProcessManager

__all__ = (
    'get_grid_vals',
    'get_plotting_grid',
    'get_patch_knots_gridlines',
    'get_patch_boundary_gridlines',
    'plot_2d',
    'plot_2d_time',
    'fill_axes'
)

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
            # The whole patch is evaluated at once (compiled kernel). The push-forwards below only call the field, 
            # so the values are handed to them as functions returning these arrays.
            values = Vh.patch_spaces[k].eval_fields(list(etas[k]), u.patch_fields[k])[0]
            if vector_valued:
                uk_field_0 = lambda *eta, v=values[0]: v
                uk_field_1 = lambda *eta, v=values[1]: v
            else:
                uk_field_0 = lambda *eta, v=values: v
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

        if is_femfield:
            # All points at once (the push-forwards work on arrays)
            result = push_field(eta_1, eta_2)
            for c, vals in enumerate(u_component_patch_vals):
                vals[k] = result[c] if vector_valued else result
        else:
            # Arbitrary Python functions are not necessarily vectorized: point by point
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
# ------------------------------------------------------------------------------

def _grid_ndim(x):
    """Number of array dimensions of (nested lists/tuples of) arrays, without converting anything to a numpy array."""
    n = 0
    while isinstance(x, (list, tuple)) and len(x) > 0:
        x, n = x[0], n + 1
    return n + (x.ndim if isinstance(x, np.ndarray) else 0)


def _is_grid_data(x):
    """True for a single scalar-valued (patches, x, y) or vector-valued (2, patches, x, y) grid data."""
    n = _grid_ndim(x)
    return n == 3 or (n == 4 and len(x) == 2)


def _patch_lines(source, selection, getter, N=100, cache=None):
    """
    Lines to be drawn on top of the plot (spline grid or patch boundaries), one entry per patch.

    Parameters
    ----------
    source : FemSpace | list | None
        Where the lines come from.
        FemSpace: the lines are computed with `getter` (get_patch_knots_gridlines or get_patch_boundary_gridlines).
        list:     the lines are already computed, one entry per patch.
        None:     nothing is drawn.

    selection : bool | list | tuple
        Patches to draw the lines on: a list/tuple of patch indices, otherwise (True or False) all patches.
        Whether to draw at all is decided by the caller, by passing source=None.

    getter : Callable
        Function (Vh, N, patch_index) -> lines of that patch. Only used if source is a FemSpace.

    N : int
        Resolution of the lines. Only used if source is a FemSpace.

    cache : dict | None
        Computed lines are stored here and reused, as long as the source is the same FemSpace object.

    Returns
    -------
    list | None
        One entry per patch: the lines of the patch, or None if the patch is not selected.
        None if source is None.
    """
    if source is None:
        return None

    is_space  = isinstance(source, FemSpace)
    n_patches = len(source.symbolic_space.domain.mappings) if is_space else len(source)
    selected  = selection if isinstance(selection, (list, tuple)) else range(n_patches)

    if not is_space:
        return [lines if k in selected else None for k, lines in enumerate(source)]

    cache = {} if cache is None else cache
    lines = []
    for k in range(n_patches):
        if k not in selected:
            lines.append(None)
            continue
        key = ('lines', id(source), getter.__name__, N, k)
        if key not in cache or cache[key][0] is not source:     # the stored source guards against a reused id
            cache[key] = (source, getter(source, N, k))
        lines.append(cache[key][1])
    return lines

# ------------------------------------------------------------------------------

def _abs(vals):
    """Absolute values of scalar-valued grid values (a list with one array per patch)."""
    return [np.abs(v) for v in vals]


def _fem_field_arrays(vals, vector_valued, o):
    """
    The grid values to plot for a FemField, one entry per plot (axes), 
    depending on the options 'plot_type', 'components' and 'magnitude' in o.
    """
    plot_type, components, magnitude = o['plot_type'], o['components'], o['magnitude']

    if plot_type == 'vector_field':                 # 1 plot of the vector (v_x, v_y)
        if not vector_valued:
            raise ValueError("plot_type='vector_field' requires a vector-valued function.")
        return [[_abs(v) for v in vals] if magnitude else vals]

    if not vector_valued:                           # 1 plot of v or |v|
        return [_abs(vals) if magnitude else vals]

    vx, vy = vals
    if components is True:                          # 2 plots: v_x, v_y (or their absolute values)
        arrays = [vx, vy]
    elif components in ('x', 'y'):                  # 1 plot: v_x or v_y (or its absolute value)
        arrays = [vx if components == 'x' else vy]
    elif magnitude:                                 # 1 plot of || (v_x, v_y) ||
        return [[np.hypot(a, b) for a, b in zip(vx, vy)]]
    else:
        raise ValueError('components=False must be accompanied by magnitude=True for vector-valued functions.')

    return [_abs(a) for a in arrays] if magnitude else arrays


def _from_fem_field(vh, o, cache):
    """Grid values (one entry per plot) and physical grid xx, yy of a FemField, over its own domain."""
    Vh       = vh.space
    mappings = Vh.symbolic_space.domain.mappings

    # The grid only depends on the domain and N_vis, so it is computed once and then reused
    key = ('grid', id(mappings), tuple(o['N_vis']))
    if key not in cache or cache[key][0] is not mappings:       # the stored mappings guard against a reused id
        cache[key] = (mappings, *get_plotting_grid(mappings, N=o['N_vis']))
    _, etas, xx, yy = cache[key]

    vals         = get_grid_vals(vh, etas, list(mappings.values()))
    return _fem_field_arrays(vals, Vh.is_vector_valued, o), xx, yy


def _from_grid_vals(vals, o):
    """Grid values (one entry per plot) and physical grid xx, yy of grid values. Absolute values must be passed manually."""
    if o['xx'] is None or o['yy'] is None:
        raise ValueError('Grid values require xx and yy.')
    ndim = _grid_ndim(vals)
    if ndim not in (3, 4):
        raise ValueError(f'Grid values must have shape (patches, x, y) or (2, patches, x, y), got {ndim} dimensions.')

    # Vector-valued grid values give 2 plots (one per component), except for a vector field plot
    one_plot = ndim == 3 or o['plot_type'] == 'vector_field'
    return ([vals] if one_plot else list(vals)), o['xx'], o['yy']

# ------------------------------------------------------------------------------

class _Prepared(NamedTuple):
    """Everything needed to draw the plot(s) of one entry of funs."""
    arrays           : list          # one entry per plot (axes)
    xx               : list
    yy               : list
    spline_grid      : list | None   # per patch, see _patch_lines
    patch_boundaries : list | None   # per patch, see _patch_lines
    style            : dict          # options passed on to fill_axes

# Options that plot_2d uses itself, they are not passed on to fill_axes
_PLOT2D_ONLY = {'xx', 'yy', 'N_vis', 'N_lines', 'components', 'magnitude', 
                'plot_spline_grid', 'spline_grid', 'plot_patch_boundaries', 'patch_boundaries'}


def _split_fun(fun):
    """
    Split one entry of funs into (kind, data, local options), where kind is 'fem_field' or 'vals'.
    The local options (only possible for dictionaries) overwrite the global options for this entry.
    """
    if isinstance(fun, FemField):
        return 'fem_field', fun, {}
    if not isinstance(fun, dict):
        return 'vals', fun, {}

    options = {k: v for k, v in fun.items() if k not in ('fem_field', 'vals')}   # the user's dict stays untouched
    for kind in ('fem_field', 'vals'):
        if fun.get(kind) is not None:
            return kind, fun[kind], options
    raise ValueError("A dictionary in funs needs the key 'fem_field' or 'vals'.")


def _prepare(fun, global_opts, cache):
    """Compute everything needed to draw one entry of funs: the data of its plot(s) and its lines."""
    kind, data, local_opts = _split_fun(fun)
    o = {**global_opts, **local_opts}

    if kind == 'fem_field':
        arrays, xx, yy = _from_fem_field(data, o, cache)
        # A FemField is its own source of the lines, they are drawn if requested
        spline_source   = data.space if o['plot_spline_grid']      is not False else None
        boundary_source = data.space if o['plot_patch_boundaries'] is not False else None
    else:
        arrays, xx, yy = _from_grid_vals(data, o)
        # For grid values the source of the lines must be passed, and the lines are drawn if it is
        spline_source, boundary_source = o['spline_grid'], o['patch_boundaries']

    return _Prepared(
        arrays           = arrays, 
        xx               = xx, 
        yy               = yy,
        spline_grid      = _patch_lines(spline_source,   o['plot_spline_grid'],      get_patch_knots_gridlines,    o['N_lines'], cache),
        patch_boundaries = _patch_lines(boundary_source, o['plot_patch_boundaries'], get_patch_boundary_gridlines, o['N_lines'], cache),
        style            = {k: v for k, v in o.items() if k not in _PLOT2D_ONLY},
    )


def _choose_layout(n_plots, layout=None):
    """(rows, columns) of the Figure. Chosen automatically (roughly square) if not passed."""
    if n_plots == 0:
        raise ValueError('Nothing to plot.')
    if layout is None:
        n_rows = int(np.floor(np.sqrt(n_plots)))
        return n_rows, int(np.ceil(n_plots / n_rows))
    if layout[0] * layout[1] < n_plots:
        raise ValueError(f'layout {tuple(layout)} has too few cells for {n_plots} plots.')
    return layout

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def plot_2d(funs, titles=None, suptitle=None, xx=None, yy=None,
        plot_type='contourf', components=True, magnitude=False,
        N_vis=(100,100),
        plot_spline_grid=False, spline_grid=None, plot_patch_boundaries=False, patch_boundaries=None,
        cmap='jet', layout=None,
        show_plot=True, filename=None, fig=None, scales=None,
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

    fig : matplotlib.Figure | None
        If passed, this Figure is cleared and reused instead of creating a new one (used by plot_2d_time).

    scales : list | None
        One dictionary per plot, with settings overwriting all others for that plot (used by plot_2d_time to fix the scales).

    **kwargs : dict
        Extra settings to be applied to all plots. Currently implemented: 
        figsize, suptitle_size, tight_layout, 
        N_lines (resolution of the spline grid and patch boundary lines, default 100) & 
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
    return _plot_2d(funs, titles=titles, suptitle=suptitle, xx=xx, yy=yy, 
                    plot_type=plot_type, components=components, magnitude=magnitude, N_vis=N_vis,
                    plot_spline_grid=plot_spline_grid, spline_grid=spline_grid, 
                    plot_patch_boundaries=plot_patch_boundaries, patch_boundaries=patch_boundaries,
                    cmap=cmap, layout=layout, show_plot=show_plot, filename=filename, fig=fig, scales=scales, 
                    **kwargs)


def _plot_2d(funs, titles=None, suptitle=None, layout=None, show_plot=True, filename=None, fig=None, scales=None,
             prepare_only=False, cache=None, **options):
    """
    The work of plot_2d, with the additional arguments used by plot_2d_time.
    The options of plot_2d not listed here (plot_type, N_vis, xx, ...) are passed as options.

    prepare_only : bool
        If True, nothing is drawn and the prepared data of every entry of funs is returned.

    cache : dict | None
        Plotting grid and lines are stored here and reused by later calls passing the same dictionary.
    """
    # Global options (used for all plots unless overwritten locally)
    global_opts = {'plot_type':'contourf', 'components':True, 'magnitude':False, 'N_vis':(100, 100), 'N_lines':100, 
                   'xx':None, 'yy':None, 'cmap':'jet', 'plot_spline_grid':False, 'spline_grid':None, 
                   'plot_patch_boundaries':False, 'patch_boundaries':None,
                   **options}
    kwargs = options

    if isinstance(titles, str):
        titles = (titles, )

    # A single FemField, dict or grid data is turned into a tuple with one entry (and so is a single title).
    if isinstance(funs, (FemField, dict)) or (isinstance(funs, (list, tuple)) and _is_grid_data(funs)):
        funs = (funs, )
    elif not isinstance(funs, (list, tuple)):
        raise ValueError('funs must be a FemField, grid values, a dictionary, or a list/tuple of these.')

    # Everything that needs computing, once per entry of funs. Then we know how many plots there are.
    cache    = {} if cache is None else cache
    prepared = [_prepare(fun, global_opts, cache) for fun in funs]
    if prepare_only:
        return prepared
    layout   = _choose_layout(sum(len(p.arrays) for p in prepared), layout)

    # Figure
    if fig is None:
        fig = plt.figure(figsize=kwargs.get('figsize', (2.6 + 4.8 * layout[1], 4.8 * layout[0])))
    else:
        fig.clf()
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=kwargs.get('suptitle_size', 14))

    # Individual plots
    count = 0
    for p in prepared:
        projection = '3d' if p.style['plot_type'] == 'surface_plot' else None
        for arr in p.arrays:
            title = titles[count] if titles is not None and count < len(titles) else None
            style = {**p.style, **scales[count]} if scales is not None else p.style
            ax    = fig.add_subplot(*layout, count + 1, projection=projection)
            fill_axes(ax, p.xx, p.yy, arr, title=title, 
                      spline_grid=p.spline_grid, patch_boundaries=p.patch_boundaries, index=count, **style)
            count += 1

    if kwargs.get('tight_layout', True):
        fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()

    return fig

# ------------------------------------------------------------------------------

def _h5_frames(f, domain, fields):
    """
    Frames of a fields file written by psydac.api.postprocessing.OutputManager.
    Returns (get_frame, n_frames, times), where get_frame(i) loads snapshot i only.
    """

    if domain is None:
        raise ValueError('domain is required for an h5py.File, the mapping is not stored in the file.')

    # The spaces are rebuilt from the space file the fields file points to; the coefficients are read from f.
    space_file = os.path.join(os.path.dirname(f.filename), f.attrs['spaces'])
    ppm = PostProcessManager(domain=domain, space_file=space_file, fields_file=f.filename)
    ppm.close()
    ppm.fields_file = f

    snapshots = sorted(k for k in f.keys() if k.startswith('snapshot_'))

    if fields is None:
        fields = []
        for patch in f[snapshots[0]].values():
            for space in patch.values():
                for name, dset in space.items():
                    # Components of a vector-valued field (E[0], E[1]) carry the name of the field (E)
                    name = dset.attrs.get('parent_field', name)
                    if name not in fields:
                        fields.append(name)
    fields = fields if isinstance(fields, (list, tuple)) else [fields]
    names  = [x if isinstance(x, str) else x['fem_field'] for x in fields]

    def get_frame(i):
        ppm.load_snapshot(int(snapshots[i].split('_')[1]), *names)
        return [ppm.fields[x] if isinstance(x, str) else {**x, 'fem_field': ppm.fields[x['fem_field']]} for x in fields]

    return get_frame, len(snapshots), [f[s].attrs['t'] for s in snapshots]


def _value_range(arr, plot_type):
    """(min, max) of the values of one plot, of the magnitude for a vector field plot."""
    vals = [np.hypot(a, b) for a, b in zip(*arr)] if plot_type == 'vector_field' else arr
    return min(np.min(v) for v in vals), max(np.max(v) for v in vals)


def _fixed_scales(style, vmin, vmax):
    """Settings that keep every scale of one plot the same in all frames. Settings passed by the user win."""
    vmin = style.get('vmin', vmin)
    vmax = style.get('vmax', vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-12     # contour levels must be increasing, even for a constant function
    opts = {'vmin': vmin, 'vmax': vmax}
    for key, n in (('contourf_levels', 50), ('contour_levels', 10)):
        levels = style.get(key, n)
        if np.isscalar(levels):
            opts[key] = np.linspace(vmin, vmax, levels + 1)
    if style['plot_type'] == 'vector_field' and 'scale' not in style:
        opts['scale'] = style.get('amp_factor', 10) * vmax
    return opts


def _to_palette(rgb, palette):
    """256-color image of an RGB array, every color replaced by the nearest color of palette (n x 3)."""
    rgb             = np.ascontiguousarray(rgb)
    packed          = rgb.reshape(-1, 3).astype(np.uint32) @ np.array([65536, 256, 1], dtype=np.uint32)     # one integer per pixel (fast unique)
    unique, inverse = np.unique(packed, return_inverse=True)
    colors          = np.stack([unique >> 16, (unique >> 8) & 255, unique & 255], axis=1)
    nearest = ((colors[:, None, :].astype(np.int32) - palette[None, :, :]) ** 2).sum(-1).argmin(1)
    image   = Image.fromarray(nearest[inverse.ravel()].reshape(rgb.shape[:2]).astype(np.uint8), 'P')
    image.putpalette(palette.ravel().tolist())
    return image


def plot_2d_time(frames, filename, domain=None, fields=None, suptitles=None, fps=10, N_vis=(20, 20), **kwargs):
    """
    Save an animated GIF of plot_2d over time. Nothing is shown, the frames are drawn one at a time.

    All scales (color ranges, contour levels, arrow lengths, z-axes) are the same in every frame.
    To that end the frames are gone through twice: first to find the value range of every plot, then to draw them.
    Evaluating a frame is cheap, so it is simply done in both passes.
    Only one frame is held in memory at a time (plus the finished GIF frames, as 256-color images).
    All frames must have the same structure (same plots, grids and settings).

    Parameters
    ----------
    frames : list | tuple | h5py.File
        One entry per time step, each one exactly what plot_2d accepts as funs
        (a FemField, grid values, a dictionary, or a list/tuple of those).
        Or a fields file written by psydac.api.postprocessing.OutputManager (requires domain).
        In that case one snapshot at a time is loaded, only when it is needed.

    filename : str
        Name of the GIF, e.g. 'solution.gif' ('.gif' is appended if missing).

    domain : sympde.topology.Domain | None
        Symbolic domain of the simulation. Required if frames is an h5py.File.

    fields : str | dict | list | tuple | None
        Only used if frames is an h5py.File. Names of the fields to plot, e.g. ('E', 'B'), all fields if None.
        A dictionary as in plot_2d, with the name instead of the FemField, gives field specific settings,
        e.g. ({'fem_field':'E', 'magnitude':True, 'components':False}, 'B').

    suptitles : list | tuple | None
        One Figure title per time step, e.g. the simulation time. Overwrites suptitle.
        For an h5py.File the simulation time is used if None.

    fps : float
        Speed of the GIF in frames per second.

    N_vis : list | tuple
        of two ints. Number of plotting points per patch and direction, see plot_2d. Low by default to keep the GIF small.

    **kwargs : dict
        Passed on to plot_2d (titles, plot_type, xx, yy, ...). 
        vmin and vmax passed here or in a dictionary of a frame overwrite the value range found over all frames.

    Returns
    -------
    str
        Name of the saved GIF.
    """
    if not filename.endswith('.gif'):
        filename += '.gif'

    if isinstance(frames, h5py.File):
        get_frame, n_frames, times = _h5_frames(frames, domain, fields)
        if suptitles is None:
            suptitles = [f't = {t:.3g}' for t in times]
    else:
        get_frame, n_frames = frames.__getitem__, len(frames)

    # First pass: evaluate every frame (one at a time) and keep track of the value range of every plot over all frames.
    # The grid and the lines are the same in every frame, they are only computed once (cache).
    kwargs['N_vis']   = N_vis
    cache, vmin, vmax = {}, None, None
    for i in range(n_frames):
        prepared = _plot_2d(get_frame(i), prepare_only=True, cache=cache, **kwargs)
        plots    = [(arr, p.style) for p in prepared for arr in p.arrays]
        lo, hi   = np.array([_value_range(arr, style['plot_type']) for arr, style in plots]).T
        vmin     = lo if vmin is None else np.minimum(vmin, lo)
        vmax     = hi if vmax is None else np.maximum(vmax, hi)
    scales = [_fixed_scales(style, a, b) for (_, style), a, b in zip(plots, vmin, vmax)]

    # Second pass: draw every frame into the same Figure and keep it as a 256-color image.
    # All frames use the palette of the first one, which contains the full colorbars, so colors don't flicker.
    fig, palette, images = None, None, []
    for i in range(n_frames):
        if suptitles is not None:
            kwargs['suptitle'] = suptitles[i]
        fig = _plot_2d(get_frame(i), cache=cache, fig=fig, scales=scales, show_plot=False, **kwargs)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        if palette is None:
            palette = np.array(Image.fromarray(rgb).quantize().getpalette()[:768]).reshape(-1, 3)
        images.append(_to_palette(rgb, palette))
    plt.close(fig)

    images[0].save(filename, save_all=True, append_images=images[1:], duration=1000 / fps, loop=0)

    return filename


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
        if 'vmin' in kwargs or 'vmax' in kwargs:
            ax.set_zlim(vmin, vmax)

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
