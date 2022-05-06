# CAD functions

We refer to The NURBS Book for the description of the different algorithms, implemented here.

## B-Splines case


| Function                            | Description | Dim | Implemented | Unitary Test |
| ----------------------------------- | ----------- | --- | ----------- | ------------ |
| point_on_bspline_curve              |             |  1  | Yes         | Yes          |
| insert_knot_bspline_curve           |             |  1  | Yes         | Yes          |
| elevate_degree_bspline_curve        |             |  1  | Yes         | No           |
| translate_bspline_curve             |             |  1  | Yes         | No           |
| rotate_bspline_curve                |             |  1  | Yes         | No           |
| homothetic_bspline_curve            |             |  1  | Yes         | No           |
| point_on_bspline_surface            |             |  2  | Yes         | Yes          |
| insert_knot_bspline_surface         |             |  2  | Yes         | No           |
| elevate_degree_bspline_surface      |             |  2  | Yes         | No           |
| translate_bspline_surface           |             |  2  | Yes         | No           |
| rotate_bspline_surface              |             |  2  | Yes         | No           |
| homothetic_bspline_surface          |             |  2  | Yes         | No           |
|                                     |             |     |             |              |
|                                     |             |     |             |              |

## NURBS case 

| Function                          | Description | Dim | Implemented | Unitary Test |
| --------------------------------- | ----------- | --- | ----------- | ------------ |
| point_on_nurbs_curve              |             |  1  | No          | No           |
| insert_knot_nurbs_curve           |             |  1  | No          | No           |
| elevate_degree_nurbs_curve        |             |  1  | No          | No           |
| translate_nurbs_curve             |             |  1  | No          | No           |
| rotate_nurbs_curve                |             |  1  | No          | No           |
| homothetic_nurbs_curve            |             |  1  | No          | No           |
| point_on_nurbs_surface            |             |  2  | No          | No           |
| insert_knot_nurbs_surface         |             |  2  | No          | No           |
| elevate_degree_nurbs_surface      |             |  2  | No          | No           |
| translate_nurbs_surface           |             |  2  | No          | No           |
| rotate_nurbs_surface              |             |  2  | No          | No           |
| homothetic_nurbs_surface          |             |  2  | No          | No           |
|                                   |             |     |             |              |
|                                   |             |     |             |              |


## Algorithms to timplement

A4.1  Compute point on rational B-spline curve 
A4.2  Compute C(u) derivatives from Cw(u) derivatives
A4.3  Compute point on rational B-spline surface 
A4.4  Compute S(u,v) derivatives from Sw(u,v) derivatives 
A5.1  Compute new curve from knot insertion 
A5.2  Compute point on rational B-spline curve
A5.3  Surface knot insertion
A5.4   Refine curve knot vector
A5.5   Refine surface knot vector
A5.6   Decompose curve into Bezier segments
A5.7   Decompose surface into Bezier patches
A5.8   Remove knot u (index r) num times
A5.9   Degree elevate a curve t times
A5.10   Degree elevate a surface t times 
A5.11   Degree reduce a curve from p to p-1
A6.1  Compute pth degree Bezier matrix 
A6.2   Compute inverse of pth-degree Bezier matrix
A7.1   Create arbitrary NURBS circular arc
A7.2  Create one Bezier conic arc 
A7.3   Construct open conic arc in 3D
A8.1   Create NURBS surface of revolution
A8.2    Create NURBS corner fillet surface
A9.1   Global interpolation through n+1 points
A9.2    Solve tridiagonal system for C2 cubic spline
A9.3  Compute parameters for global surface interpolation 
A9.4   Global surface interpolation
A9.5   Local surface interpolation through (n+1)(m+1) points
A9.6   Weighted & constrained least squares curve fit
A9.7   Global surface approx with fixed num of ctrl pts
A9.8   Get knot removal error bound (nonrational)
A9.9   Remove knots from curve, bounded
A9.10   Global curve approximation to within bound E
A9.11    Fit to tolerance E with conic segment
A9.12    Fit to tolerance E with cubic segment
A10.1   Swept surface. Trajectory interpolated
A10.2    Swept surface. Trajectory not interpolated
A10.3   Create Gordon surface
A10.4   Create bicubica11y blended Coons surface.
