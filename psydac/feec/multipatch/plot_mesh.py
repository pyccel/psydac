def surface(nurbs, resolution):

    lines = []
    for axis in range(nurbs.dim):
        for side in range(2):
            nrb = nurbs.boundary(axis, side)
            lines.append(nrb)

    for i in range(len(lines)):
        uvw = [np.linspace(U[p], U[-p-1], resolution)
               for (p, U) in zip(lines[i].degree, lines[i].knots)]
        C = lines[i](*uvw)
        lines[i] = tuple(C.T)

    uvw = [np.linspace(U[p], U[-p-1], resolution)
           for (p, U) in zip(nurbs.degree, nurbs.knots)]
    C = nurbs(*uvw)

    surfs = tuple(C.T)
    return surfs, lines

def plot(nurbs, resolution):
    surfs = [surface(nurb, resolution) for nurb in nurbs]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for (x,y,z),lines in surfs:
        ax.plot(x,y,color='k')
        for x,y,z in lines:
            ax.plot(x,y,color='k')

    ax.set_aspect('equal')        
    return fig
