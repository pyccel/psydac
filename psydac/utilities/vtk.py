import os
from pyevtk.vtk import (VtkPUnstructuredGrid,
                        VtkParallelFile,
                        )
from pyevtk.hl import _addDataToParallelFile

def writeParallelVTKUnstructuredGrid(
    path, coordsdtype, sources, ghostlevel=0, cellData=None, pointData=None
):
    """
    Writes a parallel vtk file from grid-like data:
    VTKStructuredGrid or VTKRectilinearGrid

    Parameters
    ----------
    path : str
        name of the file without extension.
    coordsData : tuple
        2-tuple (shape, dtype) where shape is the
        shape of the coordinates of the full mesh
        and dtype is the dtype of the coordinates.
    starts : list
        list of 3-tuple representing where each source file starts
        in each dimension
    source : list
        list of the relative paths of the source files where the actual data is found
    ghostlevel : int, optional
        Number of ghost-levels by which
        the extents in the individual source files overlap.
    pointData : dict
        dictionnary containing the information about the arrays
        containing node centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)
    cellData :
        dictionnary containing the information about the arrays
        containing cell centered data.
        Keys shoud be the names of the arrays.
        Values are (dtype, number of components)
    """
    # Get the extension + check that it's consistent accros all source files
    common_ext = sources[0].split(".")[-1]
    assert all(s.split(".")[-1] == common_ext for s in sources)

    assert common_ext == 'vtu'
    ftype = VtkPUnstructuredGrid
    w = VtkParallelFile(path, ftype)
    w.openGrid(ghostlevel=ghostlevel)

    _addDataToParallelFile(w, cellData=cellData, pointData=pointData)

    w.openElement("PPoints")
    w.addHeader("points", dtype=coordsdtype, ncomp=3)
    w.closeElement("PPoints")

    for source in sources:
        w.addPiece(source=source)

    w.closeGrid()
    w.save()
    return w.getFileName()
