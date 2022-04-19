# ***********************************************************************************
# * Copyright 2010 - 2016 Paulo A. Herrera. All rights reserved.                    *
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ***********************************************************************************
import os
from pyevtk.vtk import (VtkUnstructuredGrid,
                        VtkImageData,
                        VtkRectilinearGrid,
                        VtkStructuredGrid,
                        VtkPolyData,
                        _get_byte_order,
                        _mix_extents,
                        _array_to_string,
                        np_to_vtk)
from pyevtk.xml import XmlWriter

# -----------------------------------------------------------------------------
# Parallel file
# -----------------------------------------------------------------------------
class VtkParallelFileType:
    """
    A wrapper class for parallel vtk file types.
    """
    def __init__(self, vtkftype):
        self.name ="P"+vtkftype.name
        ext = vtkftype.ext    
        self.ext = ext[0]+"p"+ext[1:]

    
VtkPUnstructuredGrid =VtkParallelFileType(VtkUnstructuredGrid)
VtkPImageData = VtkParallelFileType(VtkImageData)
VtkPRectilinearGrid = VtkParallelFileType(VtkRectilinearGrid)
VtkPStructuredGrid = VtkParallelFileType(VtkStructuredGrid)
VtkPPolyData = VtkParallelFileType(VtkPolyData)


class VtkParallelFile:
    """
    Class for a VTK parallel file.

    Parameters
    ----------
    filepath : str
        filename without extension
    ftype : VtkParallelFileType
    """
    def __init__(self, filepath, ftype):
        assert isinstance(ftype, VtkParallelFileType)
        self.ftype = ftype
        self.filename = filepath + ftype.ext
        self.xml = XmlWriter(self.filename)
        self.xml.openElement("VTKFile").addAttributes(
            type=ftype.name,
            version="1.0",
            byte_order=_get_byte_order(),
            header_type="UInt64",
        )
    
    def getFileName(self):
        """Return absolute path to this file."""
        return os.path.abspath(self.filename)

    def addPiece(
            self,
            start=None,
            end=None,
            source=None,
        ):
        """
        Add piece section with extent and source.
        Parameters
        ----------
        start : array-like, optional
            array or list with start indexes in each direction.
            Must be given with end.
        end : array-like, optional
            array or list with end indexes in each direction.
            Must be given with start.
        source : str
            Source of this piece
        Returns
        -------
        VtkParallelFile
            This VtkFile to allow chained calls.
        """
        # Check Source
        assert source is not None
        assert source.split('.')[-1] == self.ftype.ext[2:]

        self.xml.openElement("Piece")
        if start and end:
            ext = _mix_extents(start, end)
            self.xml.addAttributes(Extent=ext)
        self.xml.addAttributes(Source=source)
        self.xml.closeElement()
        return self


    def openData(self, nodeType, scalars=None, vectors=None, normals=None, tensors=None, tcoords=None):
        """
        Open data section.
        Parameters
        ----------
        nodeType : str
            Either "Point", "Cell" or "Field".
        scalars : str, optional
            default data array name for scalar data.
        vectors : str, optional
            default data array name for vector data.
        normals : str, optional
            default data array name for normals data.
        tensors : str, optional
            default data array name for tensors data.
        tcoords : str, optional
            default data array name for tcoords data.
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.openElement(nodeType + "Data")
        if scalars:
            self.xml.addAttributes(Scalars=scalars)
        if vectors:
            self.xml.addAttributes(Vectors=vectors)
        if normals:
            self.xml.addAttributes(Normals=normals)
        if tensors:
            self.xml.addAttributes(Tensors=tensors)
        if tcoords:
            self.xml.addAttributes(TCoords=tcoords)

        return self
    
    def closeData(self, nodeType):
        """
        Close data section.
        Parameters
        ----------
        nodeType : str
            "Point", "Cell" or "Field".
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.closeElement(nodeType + "Data")

    
    def openGrid(self, start=None, end=None, origin=None, spacing=None, ghostlevel=0):
        """
        Open grid section.

        Parameters
        ----------
        start : array-like, optional
            array or list of start indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        end : array-like, optional
            array or list of end indexes.
            Required for Structured, Rectilinear and ImageData grids.
            The default is None.
        origin : array-like, optional
            3D array or list with grid origin.
            Only required for ImageData grids.
            The default is None.
        spacing : array-like, optional
            3D array or list with grid spacing.
            Only required for ImageData grids.
            The default is None.
        ghostlevel : int
            Number of ghost-levels by which 
            the extents in the individual pieces overlap.
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        gType = self.ftype.name
        self.xml.openElement(gType)

        if gType == VtkPImageData.name:
            if not start or not end or not origin or not spacing:
                raise ValueError(f"start, end, origin and spacing required for {gType}")
            ext = _mix_extents(start, end)
            self.xml.addAttributes(
                WholeExtent=ext,
                Origin=_array_to_string(origin),
                Spacing=_array_to_string(spacing),
            )

        elif gType in [VtkPStructuredGrid.name, VtkPRectilinearGrid.name]:
            if not start or not end:
                raise ValueError(f"start and end required for {gType}.")
            ext = _mix_extents(start, end)
            self.xml.addAttributes(WholeExtent=ext)
        
        # Ghostlevel
        self.xml.addAttributes(Ghostlevel=ghostlevel)
        return self
    
    def closeGrid(self):
        """
        Close grid element.
        Returns
        -------
        VtkFile
            This VtkFile to allow chained calls.
        """
        self.xml.closeElement(self.ftype.name)        
    
    def addHeader(self, name, dtype, ncomp):
        """
        Add data array description to xml header section.
        Parameters
        ----------
        name : str
            data array name.
        dtype : str
            data type.
        ncomp : int
            number of components, 1 (=scalar) and 3 (=vector).
        Returns
        -------

        VtkFile
            This VtkFile to allow chained calls.
        Notes
        -----

        This is a low level function.
        Use addData if you want to add a numpy array.
        """
        dtype = np_to_vtk[dtype.name]

        self.xml.openElement("DataArray")
        self.xml.addAttributes(
            Name=name,
            NumberOfComponents=ncomp,
            type=dtype.name,
        )
        self.xml.closeElement()        

    def openElement(self, tagName):
        """
        Open an element.
        Useful to add elements such as: Coordinates, Points, Verts, etc.
        """
        self.xml.openElement(tagName)

    def closeElement(self, tagName):
        self.xml.closeElement(tagName)

    def save(self):
        """Close file."""
        self.xml.closeElement("VTKFile")
        self.xml.close()

# ==================================================================
def _addDataToParallelFile(vtkParallelFile, cellData, pointData):
    assert isinstance(vtkParallelFile, VtkParallelFile)
    # Point data
    if pointData:
        keys = list(pointData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next((key for key in keys if pointData[key][1] == 1), None)
        vectors = next((key for key in keys if pointData[key][1] == 3), None)
        vtkParallelFile.openData("PPoint", scalars=scalars, vectors=vectors)
        for key in keys:
            dtype, ncomp = pointData[key]
            vtkParallelFile.addHeader(key, dtype=dtype, ncomp=ncomp)
        vtkParallelFile.closeData("PPoint")
    
    # Cell data
    if cellData:
        keys = list(cellData.keys())
        # find first scalar and vector data key to set it as attribute
        scalars = next((key for key in keys if cellData[key][1] == 1), None)
        vectors = next((key for key in keys if cellData[key][1] == 3), None)
        vtkParallelFile.openData("PCell", scalars=scalars, vectors=vectors)
        for key in keys:
            dtype, ncomp = cellData[key]
            vtkParallelFile.addHeader(key, dtype=dtype, ncomp=ncomp)
        vtkParallelFile.closeData("PCell")


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
