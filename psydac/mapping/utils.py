import numpy as np
import itertools as it
from sympy import lambdify

from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt

from sympde.topology import IdentityMapping, InteriorDomain, MultiPatchMapping

def lambdify_sympde(variables, expr):
    """
    Custom lambify function that covers the
    shortcomings of sympy's lambdify. Most notably,
    this function uses numpy broadcasting rules to
    compute the shape of the output.

    Parameters
    ----------
    variables : sympy.core.symbol.Symbol or list of sympy.core.symbol.Symbol
        variables that appear in the expression
    expr :
        Sympy expression

    Returns
    -------
    lambda_f : callable
        Lambdified function built using numpy.

    Notes
    -----
    Compared to Sympy's lambdify, this function
    is capable of properly handling constant values,
    and array_like structures where not all components
    depend on all variables. See below.

    Examples
    --------
    >>> import numpy as np
    >>> from sympy import symbols,  Matrix
    >>> from sympde.utilities.utils import lambdify_sympde
    >>> x, y = symbols("x,y")
    >>> expr = Matrix([[x, x + y], [0, y]])
    >>> f = lambdify_sympde([x,y], expr)
    >>> f(np.array([[0, 1]]), np.array([[2], [3]]))
    array([[[[0., 1.],
             [0., 1.]],

            [[2., 3.],
             [3., 4.]]],


           [[[0., 0.],
             [0., 0.]],

            [[2., 2.],
             [3., 3.]]]])
    """
    array_expr = np.asarray(expr)
    scalar_shape = array_expr.shape
    if scalar_shape == ():
        f = lambdify(variables, expr, 'numpy')
        def f_vec_sc(*XYZ):
            b = np.broadcast(*XYZ)
            if b.ndim == 0:
                return f(*XYZ)
            temp = np.asarray(f(*XYZ))
            if b.shape == temp.shape:
                return temp

            result = np.zeros(b.shape)
            result[...] = temp
            return result
        return f_vec_sc

    else:
        scalar_functions = {}
        for multi_index in it.product(*tuple(range(s) for s in scalar_shape)):
            scalar_functions[multi_index] = lambdify(variables, array_expr[multi_index], 'numpy')

        def f_vec_v(*XYZ):
            b = np.broadcast(*XYZ)
            result = np.zeros(scalar_shape + b.shape)
            for multi_index in it.product(*tuple(range(s) for s in scalar_shape)):
                result[multi_index] = scalar_functions[multi_index](*XYZ)
            return result
        return f_vec_v


def plot_domain(domain, draw=True, isolines=False, refinement=None):
    """
    Plots a 2D  or 3D domain using matplotlib

    Parameters
    ----------
    domain : sympde.topology.Domain
        Domain to plot

    draw : bool, default=True
        If true, plt.show() will be called.

    isolines : bool, default=False
        If true and the domain is 2D, also plots iso-lines.

    refinement : int or None
        Number of straight line segments used to approximate each boundary edge.
        If None, uses 15 for 3D domains and 40 for 2D domains
    """
    pdim = domain.dim if domain.mapping is None else domain.mapping.pdim
    if pdim == 2:
        if refinement is None:
            plot_2d(domain, draw=draw, isolines=isolines)
        else:
            plot_2d(domain, draw=draw, isolines=isolines, refinement=refinement)
    elif pdim ==3:
        if refinement is None:
            plot_3d(domain, draw=draw)
        else:
            plot_3d(domain, draw=draw, refinement=refinement)


def plot_2d(domain, draw=True, isolines=False, refinement=40):
    """
    Plot a 2D domain

    Parameters
    ----------
    domain : sympde.topology.Domain
        Domain to plot

    draw : bool
        if true, plt.show() will be called.

    refinement : int
        Number of straight line segments used to approximate each boundary edge.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if isinstance(domain.interior, InteriorDomain):
        plot_2d_single_patch(domain.interior, domain.mapping, ax, isolines=isolines, refinement=refinement)
    else:
        if isinstance(domain.mapping, MultiPatchMapping):
            for patch, mapping in domain.mapping.mappings.items():
                plot_2d_single_patch(patch, mapping, ax, isolines=isolines, refinement=refinement)
        else:
            for interior in domain.interior.as_tuple():
                plot_2d_single_patch(interior, interior.mapping, ax, isolines=isolines, refinement=refinement)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y', rotation='horizontal')
    if draw:
        plt.show()

def plot_3d(domain, draw=True, refinement=15):
    """
    Plot a 3D domain

    Parameters
    ----------
    domain : sympde.topology.Domain
        Domain to plot

    draw : bool
        if true, plt.show() will be called.

    refinement : int
        Number of straight line segments used to approximate each boundary edge.
    """
    mapping = domain.mapping

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if isinstance(domain.interior, InteriorDomain):
        plot_3d_single_patch(domain.interior, domain.mapping, ax, refinement=refinement)
    else:
        if isinstance(domain.mapping, MultiPatchMapping):
            for patch, mapping in domain.mapping.mappings.items():
                plot_3d_single_patch(patch, mapping, ax, refinement=refinement)
        else:
            for interior in domain.interior.as_tuple():
                plot_3d_single_patch(interior, interior.mapping, ax, refinement=refinement)

    ax.set_xlabel('X')
    ax.set_ylabel('Y', rotation='horizontal')
    ax.set_zlabel('Z')
    if draw:
        plt.show()

def plot_3d_single_patch(patch, mapping, ax, refinement=15):
    """
    Plot a singe patch in a 3D domain

    Parameters
    ----------
    patch : sympde.topology.InteriorDomain

    mapping : sympde.topology.mapping

    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Axes object on which the patch is drawn.

    refinement : int, default=15
        Number of straight line segments used to approximate each boundary edge.
    """
    if mapping is None:
        mapping = IdentityMapping('Id', dim=3)


    refinement += 1

    linspace_0 = np.linspace(patch.min_coords[0], patch.max_coords[0], refinement, endpoint=True)
    linspace_1 = np.linspace(patch.min_coords[1], patch.max_coords[1], refinement, endpoint=True)
    linspace_2 = np.linspace(patch.min_coords[2], patch.max_coords[2], refinement, endpoint=True)

    grid_01 = np.meshgrid(linspace_0, linspace_1, indexing='ij')
    grid_02 = np.meshgrid(linspace_0, linspace_2, indexing='ij')
    grid_12 = np.meshgrid(linspace_1, linspace_2, indexing='ij')

    full_00 = np.full((refinement, refinement), linspace_0[0])
    full_01 = np.full((refinement, refinement), linspace_0[-1])
    full_10 = np.full((refinement, refinement), linspace_1[0])
    full_11 = np.full((refinement, refinement), linspace_1[-1])
    full_20 = np.full((refinement, refinement), linspace_2[0])
    full_21 = np.full((refinement, refinement), linspace_2[-1])

    mesh_01_0 = mapping(*grid_01, full_20)
    mesh_01_1 = mapping(*grid_01, full_21)

    mesh_02_0 = mapping(grid_02[0], full_10, grid_02[1])
    mesh_02_1 = mapping(grid_02[0], full_11, grid_02[1])

    mesh_12_0 = mapping(full_00, *grid_12)
    mesh_12_1 = mapping(full_01, *grid_12)

    kwargs_plot = {'color': 'c', 'alpha': 0.7}

    ax.plot_surface(*mesh_01_0, **kwargs_plot)
    ax.plot_surface(*mesh_01_1, **kwargs_plot)
    ax.plot_surface(*mesh_02_0, **kwargs_plot)
    ax.plot_surface(*mesh_02_1, **kwargs_plot)
    ax.plot_surface(*mesh_12_0, **kwargs_plot)
    ax.plot_surface(*mesh_12_1, **kwargs_plot)


def plot_2d_single_patch(patch, mapping, ax, isolines=False, refinement=40):
    """
    Plots a singe patch in a 2D domain

    Parameters
    ----------
    patch : sympde.topology.InteriorDomain

    mapping : sympde.topology.mapping

    ax : matplotlib.axes.Axes
        Axes object on which the patch is drawn.

    isolines : bool, default=False
        If true also plots some iso-lines

    refinement : int, default=40
        Number of straight line segments used to approximate each boundary edge.
    """
    if mapping is None:
        mapping = IdentityMapping('Id', dim=3)

    refinement+=1
    linspace_0 = np.linspace(patch.min_coords[0], patch.max_coords[0], refinement, endpoint=True)
    linspace_1 = np.linspace(patch.min_coords[1], patch.max_coords[1], refinement, endpoint=True)

    if isolines:
        mesh_grid = np.meshgrid(linspace_0, linspace_1, indexing='ij')

        XX, YY = mapping(*mesh_grid)

        ax.plot(XX[:, ::5], YY[:, ::5], color='darkgrey')
        ax.plot(XX[::5, :].T, YY[::5, :].T, color='darkgrey')

    X_00, Y_00 = mapping(linspace_0, np.full(refinement, linspace_1[0]))
    X_01, Y_01 = mapping(linspace_0, np.full(refinement, linspace_1[-1]))
    X_10, Y_10 = mapping(np.full(refinement, linspace_0[0]), linspace_1)
    X_11, Y_11 = mapping(np.full(refinement, linspace_0[-1]), linspace_1)

    ax.plot(X_00, Y_00, 'k')
    ax.plot(X_01, Y_01, 'k')
    ax.plot(X_10, Y_10, 'k')
    ax.plot(X_11, Y_11, 'k')

if __name__ == '__main__':
    from sympde.topology import Square, PolarMapping
    A = Square('A', bounds1=(0, 1), bounds2=(0, np.pi/2))
    F = PolarMapping('F', c1=0, c2=0, rmin=0.5, rmax=1)
    Omega = F(A)

    plot_domain(Omega, draw=True, isolines=True)
