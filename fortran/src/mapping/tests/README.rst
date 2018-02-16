Examples
********

This file describes all the tests of the current sub-package **mapping**.

Read and Export mapping
^^^^^^^^^^^^^^^^^^^^^^^

**mapping_1d_ex0.F90**: Creates a linear map, exports it and then reads it and exports it again.

**mapping_2d_ex0.F90**: Creates a bilinear map, exports it and then reads it and exports it again.

**mapping_3d_ex0.F90**: Creates a trilinear map, exports it and then reads it and exports it again.

**mapping_1d_ex00.F90**: Reads a mapping then exports it.

**mapping_2d_ex00.F90**: Reads a mapping then exports it.

**mapping_3d_ex00.F90**: Reads a mapping then exports it.

**mapping_1d_ex1.F90**: Creates a linear map with **degree=1** and **n_elements=2** .

**mapping_2d_ex1.F90**: Creates a bilinear map with **degree=(2,2)** and **n_elements=(2,2)** .

**mapping_3d_ex1.F90**: Creates a trilinear map with **degree=(2,2,2)** and **n_elements=(2,2,2)** .

**mapping_2d_ex15.F90**: Creates a Collela map 

**mapping_2d_ex17.F90**: Creates an eccentric annulus map 

Clamp/Unclamp
^^^^^^^^^^^^^

**mapping_1d_ex2.F90**: Creates a linear map **unclamp** it and then **clamp** it again.

**mapping_2d_ex2.F90**: Creates a bilinear map **unclamp** it and then **clamp** it again.

**mapping_3d_ex2.F90**: Creates a bilinear map **unclamp** it and then **clamp** it again.

Evaluation
^^^^^^^^^^

**mapping_1d_ex3.F90**: Creates a linear map and evaluate it on given sites.

**mapping_2d_ex3.F90**: Creates a bilinear map and evaluate it on given sites.

**mapping_3d_ex3.F90**: Creates a trilinear map and evaluate it on given sites.

**mapping_1d_ex4.F90**: Creates a linear map and evaluate it and its derivatives on given sites.

**mapping_2d_ex4.F90**: Creates a bilinear map and evaluate it and its derivatives on given sites.

**mapping_3d_ex4.F90**: Creates a trilinear map and evaluate it and its derivatives on given sites.


**mapping_1d_ex8.F90**: Creates an arc map and evaluate it and its derivatives on given sites.

**mapping_2d_ex8.F90**: Creates an annulus map and evaluate it and its derivatives on given sites.

**mapping_2d_ex9.F90**: Creates a circular map and evaluate it and its derivatives on given sites.

**mapping_3d_ex9.F90**: Creates a circular map, extrudes it to have a cylinder and evaluate it and its derivatives on given sites.


Computing mapping breaks
^^^^^^^^^^^^^^^^^^^^^^^^

**mapping_1d_ex5.F90**: Creates a linear map and computes its breaks. 

**mapping_2d_ex5.F90**: Creates a bilinear map and computes its breaks.

**mapping_3d_ex5.F90**: Creates a trilinear map and computes its breaks.


Conversion to uniform B-Splines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**mapping_1d_ex6.F90**: Creates a linear map and computes its uniform B-Spline representation. 

**mapping_2d_ex6.F90**: Creates a bilinear map and computes its uniform B-Spline representation. 

**mapping_3d_ex6.F90**: Creates a trilinear map and computes its uniform B-Spline representation. 

Conversion to pp-form 
^^^^^^^^^^^^^^^^^^^^^

**mapping_1d_ex7.F90**: Creates a linear map and computes its pp-form. 

**mapping_2d_ex7.F90**: Creates a bilinear map and computes its pp-form. 

**mapping_3d_ex7.F90**: Creates a trilinear map and computes its pp-form.

Mapping transformations
^^^^^^^^^^^^^^^^^^^^^^^

Translation
___________

**mapping_1d_ex10.F90**: Creates a linear map and translates it. 

**mapping_2d_ex10.F90**: Creates a bilinear map and translates it. 

**mapping_3d_ex10.F90**: Creates a trilinear map and translates it. 

Rotation
________

**mapping_2d_ex19.F90**: Creates a bilinear map and rotates it. 

Extrude
_______

**mapping_1d_ex11.F90**: Creates a linear map then extrudes it to a 2d map. 

**mapping_2d_ex11.F90**: Creates a bilinear map and extrudes it to a 3d map. 

Transposition
_____________

**mapping_2d_ex16.F90**: Creates an annulus map then transpose it. 

Refinement Matrix
^^^^^^^^^^^^^^^^^

**mapping_1d_ex12.F90**: Creates a linear map an the refinement matrix that corresponds to the insertion of **one** knot using Cox-Deboor formula. 

**mapping_1d_ex13.F90**: Creates a linear map an the refinement matrix that corresponds to the insertion of many knots using Cox-Deboor formula. 

Greville Abscissae
^^^^^^^^^^^^^^^^^^

**mapping_1d_ex14.F90**: Creates a linear map, refine/elevate it and compute its greville abscissae 

**mapping_2d_ex14.F90**: Creates a bilinear map, refine/elevate it and compute its greville abscissae 

**mapping_3d_ex14.F90**: Creates a trilinear map, refine/elevate it and compute its greville abscissae 

Composition
___________

**mapping_2d_ex20.F90**: Creates a bilinear map and composr it another one.

