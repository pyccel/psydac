# Mesh Generation

After installation, the command `psydac mesh` will be available.

Currently, this command interpolates an analytical 2D or 3D mapping with a discrete (spline) mapping.
The knot sequences and control points of the discrete mapping are stored into an HDF5 geometry file which can be loaded from the PSYDAC library.

## Example of usage

```bash
psydac mesh --map-2d circle -n 10 20 -d 3 3 -o circle.h5
```