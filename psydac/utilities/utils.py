#---------------------------------------------------------------------------#
# This file is part of PSYDAC which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/psydac/blob/devel/LICENSE #
# for full license details.                                                 #
#---------------------------------------------------------------------------#
from numbers import Number

import numpy as np

__all__ = (
    'refine_array_1d',
    'unroll_edges',
    'split_space',
    'split_field',
    'animate_field'
)

#==============================================================================
def is_real(x):
    """Determine whether the given input represents a real number.

    Parameters
    ----------
    x : Any

    Returns
    -------
    bool
        True if x is real, False otherwise.

    """
    return isinstance(x, Number) and np.isrealobj(x) and not isinstance(x, bool)

#===============================================================================
def refine_array_1d(x, n, remove_duplicates=True):
    """
    Refines a 1D array by subdividing each interval (x[i], x[i+1]) into n identical parts.

    Parameters
    ----------
    x : ndarray
        1D array to be refined.
    
    n : int
        Number of subdivisions to be created in each interval (x[i], x[i+1]).

    remove_duplicates : bool, default=True
        If True, the refined array will not contain any duplicate points.
        If False, the original internal grid points x[1:-1] will appear twice: this may
        be useful to visualize fields that are discontinuous across cell boundaries.

    Returns
    -------
    ndarray
        Refined 1D array. The length of this array is `n * (len(x) - 1) + 1`
        if remove_duplicates and `(n + 1) * (len(x) - 1)` if not.
    """
    xr = []
    if not remove_duplicates:
        n += 1
    for (a, b) in zip(x[:-1], x[1:]):
        xr.extend(np.linspace(a, b, n, endpoint=not remove_duplicates))
    if remove_duplicates:
        xr.append(x[-1])
    return np.array(xr)

#===============================================================================
def unroll_edges(domain, xgrid):
    """If necessary, "unroll" intervals that cross boundary of periodic domain.
    """

    xA, xB = domain

    assert all(np.diff(xgrid) >= 0)
    assert xA < xB
    assert xA <= xgrid[0]
    assert xgrid[-1] <= xB

    if xgrid[0] == xA and xgrid[-1] == xB:
        return xgrid

    elif xgrid[0] != xA:
        return np.array([xgrid[-1] - (xB-xA), *xgrid])

    elif xgrid[-1] != xB:
        return np.array([*xgrid, xgrid[0] + (xB-xA)])

#===============================================================================
def roll_edges(domain, points):
    """If necessary, "roll" back intervals that cross boundary of periodic domain.
    Changes are made in place to avoid duplicating the array
    """
    xA, xB = domain
    assert xA < xB
    points -=xA
    points %=(xB-xA)
    points +=xA
#===============================================================================
def split_space(Xh):
    """Split the flattened fem spaces into
       a list of spaces that corresponds to the symbolic function spaces.

    Parameters
    ----------
    Xh : MultipatchFemSpace
        The discrete space.

    Returns
    -------
    Vh : <list, FemSpace>
         List of fem spaces.
    """
    from sympde.topology.space import VectorFunctionSpace
    from psydac.fem.vector     import VectorFemSpace
    V = Xh.symbolic_space
    spaces = Xh.spaces
    Vh    = []
    for Vi in V.spaces:
        if isinstance(Vi, VectorFunctionSpace):
            Vh.append(VectorFemSpace(*spaces[:Vi.ldim]))
            Vh[-1].symbolic_space = Vi
            spaces = spaces[Vi.ldim:]
        else:
            Vh.append(spaces[0])
            Vh[-1].symbolic_space = Vi
            spaces = spaces[1:]
    return Vh

#===============================================================================
def split_field(uh, spaces, out=None):
    """Split a field into a list of fields that corresponds to the spaces.
       The split field function will allocate new memory if out is not passed.

    Parameters
    ----------
    uh : FemField
        The fem field.

    spaces: <list, FemSpace>
        List of spaces that split the field.

    out: optional, <list, FemField>
        List of fields to write the results to.
 
    Returns
    -------
    out : <list, FemField>
         List of fem fields.
    """
    from psydac.fem.basic import FemField
    if out is None:
        out = [FemField(S) for S in spaces]

    flattened_fields = [f.fields if f.fields else [f] for f in out]
    flattened_fields = [f for l in flattened_fields for f in l]
    for f1,f2 in zip(flattened_fields, uh.fields):
        assert f1.space is f2.space
        f1.coeffs[:] = f2.coeffs[:]

    return out

#===============================================================================
def animate_field(fields, domain, mapping, res=(150,150), vrange=None, cmap=None, interval=35, progress=False, figsize=(14,4)):
    """Animate a sequence of scalar fields over a geometry."""
    from matplotlib import animation
    import matplotlib.pyplot as plt
    import tqdm

    fields = list(fields)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    etas    = [refine_array_1d( bounds, r ) for r,bounds in zip(res, zip(domain.min_coords, domain.max_coords))]
    pcoords = np.array( [[mapping( e1,e2 ) for e2 in etas[1]] for e1 in etas[0]] )
    xx      = pcoords[:,:,0]
    yy      = pcoords[:,:,1]

    # determine range of values from first field
    num1     = np.array( [[fields[0].fields[0]( e1,e2 ) for e2 in etas[1]] for e1 in etas[0]] )
    num2     = np.array( [[fields[0].fields[1]( e1,e2 ) for e2 in etas[1]] for e1 in etas[0]] )
    num      = np.hypot(num1, num2)
    vrange   = (num.min(), num.max())

    quadmesh = plt.pcolormesh(xx, yy, num, shading='gouraud', cmap=cmap,
                vmin=vrange[0], vmax=vrange[1], axes=ax)
    fig.colorbar(quadmesh, ax=ax)

    pbar = tqdm.tqdm(total=len(fields))
    def anim_func(i):
        num1     = np.array( [[fields[i].fields[0]( e1,e2 ) for e2 in etas[1]] for e1 in etas[0]] )
        num2     = np.array( [[fields[i].fields[1]( e1,e2 ) for e2 in etas[1]] for e1 in etas[0]] )
        C        = np.hypot(num1, num2)
        quadmesh.set_array(C)
        pbar.update()
        if i == len(fields) - 1:
            pbar.close()

    return animation.FuncAnimation(fig, anim_func, frames=len(fields), interval=interval)
