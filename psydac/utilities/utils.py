# coding: utf-8
#
# Copyright 2018 Yaman Güçlü

import numpy as np

#===============================================================================
def refine_array_1d( x, n ):

    xr = []
    for (a,b) in zip(x[:-1],x[1:]):
        xr.extend( np.linspace( a, b, n, endpoint=False ) )
    xr.append( x[-1] )

    return xr

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
def decompose_spaces(Xh):
    from sympde.topology.space import VectorFunctionSpace
    from psydac.fem.vector     import ProductFemSpace
    V = Xh.symbolic_space
    spaces = Xh.spaces
    Vh    = []
    for Vi in V.spaces:
        if isinstance(Vi, VectorFunctionSpace):
            Vh.append(ProductFemSpace(*spaces[:Vi.ldim]))
            Vh[-1].symbolic_space = Vi
            spaces = spaces[Vi.ldim:]
        else:
            Vh.append(spaces[0])
            Vh[-1].symbolic_space = Vi
            spaces = spaces[1:]
    return Vh
#===============================================================================
def animate_field(fields, domain, mapping, res=(150,150), vrange=None, cmap=None, interval=35, progress=False, figsize=(14,4)):
    """Animate a sequence of scalar fields over a geometry."""
    from matplotlib import animation
    from psydac.utilities.utils import refine_array_1d
    import matplotlib.pyplot as plt
    import tqdm

    fields = list(fields)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    etas    = [refine_array_1d( bounds, r ) for r,bounds in zip(res, zip(domain.min_coords, domain.max_coords))]
    pcoords = np.array( [[mapping( [e1,e2] ) for e2 in etas[1]] for e1 in etas[0]] )
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
