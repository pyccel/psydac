.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)
.. _mapping:

Mapping
=======



.. graphviz::

   digraph spl_mapping {
      graph [ splines = true ]

      "mapping_1d" -> "mapping_2d" [label="extrude",   fontsize=10, dir="both"];
      "mapping_2d" -> "mapping_3d" [label="extrude",   fontsize=10, dir="both"];

      "mapping_1d" -> "mapping_1d" [label="refine/elevate\nclamp/unclamp\ntranslate", fontsize=10, dir="both"];
      "mapping_2d" -> "mapping_2d" [label="refine/elevate\nclamp/unclamp\ntranslate", fontsize=10, dir="both"];
      "mapping_3d" -> "mapping_3d" [label="refine/elevate\nclamp/unclamp\ntranslate", fontsize=10, dir="both"];
   }

.. The available operations are: **refine**, **elevate**, **clamp**, **unclamp** which are methods of the 1d, 2d and 3d mappings, in addition to **translate** that is provided by the **mapping_cad object**.



spl_t_mapping_abstract
**********************

Methods
^^^^^^^

Attributs
^^^^^^^^^

n_elements
__________

number of elements. **Default 0**

p_dim
_____

parametric dimension. **Default 0**  

d_dim
_____

physical dimension. **Default 1** 

n_points
________

number of control points. **Default 0** 

rationalize
___________

rationalize B-Splines. **Default False** 

spl_t_mapping_1d
****************

Methods
^^^^^^^

breaks
______

Computes the breaks of the knot vector. If :math:`T = \left(  t_i \right)_{1 \leq i \leq n + p +1}` is a non-decreasing sequence of real numbers. The breaks associated to this knots vector is the array containing unique entries of :math:`T`. 

clamp
_____

Converts the B-Spline mapping in order to have a clamped knot vector. 

More details can be found in :cite:`piegl` Chapter 12.

create
______

Creates a B-Spline curve given a degree **p**,a knots vector **knots** and a **control points** array. The user can also specify the **optional** array containing the **weights**. 

duplicate
_________

Duplicates the current object. Memory for the output object is allocated here.

elevate
_______

elevates the polynomial degree (number of elevation = times). 
We refeer to :cite:`piegl` Chapter 5 section 5, for more details.

evaluate
________

Evaluates a B-Spline curve on **1D** array sites.

evaluate_deriv
______________

Evaluates a B-Spline curve and its derivatives on **1D** array sites.

export
______

Exports a B-Spline curve. Only the **namelist** format is available. 

free
____

Deallocates memory.

insert_knot
___________

Insert a new knot in the knots vector. 
We refer to :cite:`piegl` Chapter 5, section 2, for more details.

print_info
__________

Prints information about the current object.

read_from_file
______________

Reads a B-Spline curve from a file. Only the **namelist** format is available for the moment.

refine
______

Refines a B-Spline curve. Refinment is done given the degree, the number of elements. The user can also specify if this subroutine is inplace. 

Refining a B-Spline curve means either Degree elevation or knot insertion.

set_control_points
__________________

Sets new values for the control points.

set_weights
___________

Sets new values for the weights.

to_pp
_____

Converts the control points associated to the B-Spline family to a piecewise polynomial form. The result is an array corresponding to the pp-form on each element.

For more details, we refer the user to :cite:`DeBoor_Book2001`, Chapter 9.

to_us
_____

Gets the coefficients associated to the uniform clamped description of the current B-Spline curve.

unclamp
_______

Unclamping a B-Spline curve. 

More details can be found in :cite:`piegl` Chapter 12.

Attributs
^^^^^^^^^

n_u
___

number of control points

p_u
___

spline degree            

n_elements_u
____________

number of elements

knots_u
_______

array of knots of size n_u+p_u+1 

control_points
______________

array of control points in :math:`\mathbb{R}^d`

weights
_______

array of weights 

grid_u
______

corresponding grid 

i_spans_u
_________

knots indices corresponding to the grid

spl_t_mapping_2d
****************

Methods
^^^^^^^

breaks
______

Computes the breaks of the knot vector for a given direction (or both) [optional]. If :math:`T = \left(  t_i \right)_{1 \leq i \leq n + p +1}` is a non-decreasing sequence of real numbers. The breaks associated to this knots vector is the array containing unique entries of :math:`T`. 

clamp
_____

Converts the B-Spline mapping in order to have a clamped knot vector. 

More details can be found in :cite:`piegl` Chapter 12.

create
______

Creates a B-Spline surface given degrees **p_u** and **p_v**, knots vectors **knots_u** and **knots_v** and a **control points** array. The user can also specify the **optional** array containing the **weights**. 

duplicate
_________

Duplicates the current object. Memory for the output object is allocated here.

elevate
_______

elevates the polynomial degree (number of elevation = times). 
We refeer to :cite:`piegl` Chapter 5 section 5, for more details.

evaluate
________

Evaluates a B-Spline surface on **1D** array sites for every direction. The result is the evaluation of the B-Spline surface over the correspongin **2D** grid.

evaluate_deriv
______________

Evaluates a B-Spline surface and its derivatives on **1D** array sites. The result is the evaluation of the B-Spline surface over the correspongin **2D** grid. 

export
______

Exports a B-Spline surface. Only the **namelist** format is available. 

extract
_______

Extracts a B-Spline curve from a B-Spline surface

free
____

Deallocates memory.

insert_knot
___________

Insert a new knot in the knots vector of a given **axis**. 
We refer to :cite:`piegl` Chapter 5, section 2, for more details.

print_info
__________

Prints information about the current object.

read_from_file
______________

Reads a B-Spline surface from a file. Only the **namelist** format is available for the moment.

refine
______

Refines a B-Spline surface. Refinment is done given the degrees, the number of elements in each direction. The user can also specify if this subroutine is inplace. 

Refining a B-Spline surface means either Degree elevation or knot insertion.

set_control_points
__________________

Sets new values for the control points.

set_weights
___________

Sets new values for the weights.

to_pp
_____

Converts the control points associated to the B-Spline families to a piecewise polynomial form. The result is an array corresponding to the pp-form on each element.

For more details, we refer the user to :cite:`DeBoor_Book2001`, Chapter 9.

to_us
_____

Gets the coefficients associated to the uniform clamped description of the current B-Spline surface.

unclamp
_______

Unclamping a B-Spline surface. 

More details can be found in :cite:`piegl` Chapter 12.


Attributs
^^^^^^^^^

n_u
___

number of control points in the u-direction

p_u
___

spline degree in the u-direction            

n_elements_u
____________

number of elements in the u-direction 

knots_u
_______

array of knots of size n_u+p_u+1 in the u-direction  

n_v
___

number of control points in the v-direction

p_v
___

spline degree in the v-direction            

n_elements_v
____________

number of elements in the v-direction 

knots_v
_______

array of knots of size n_v+p_v+1 in the v-direction  

control_points
______________

array of control points in :math:`\mathbb{R}^d`

weights
_______

array of weights 

grid_u
______

corresponding grid in the u-direction 

i_spans_u
_________

knots indices corresponding to the grid in the u-direction 

grid_v
______

corresponding grid in the v-direction 

i_spans_v
_________

knots indices corresponding to the grid in the v-direction 

spl_t_mapping_3d
****************

Methods
^^^^^^^

breaks
______

Computes the breaks of the knot vector for a given direction (or both) [optional]. If :math:`T = \left(  t_i \right)_{1 \leq i \leq n + p +1}` is a non-decreasing sequence of real numbers. The breaks associated to this knots vector is the array containing unique entries of :math:`T`. 

clamp
_____

Converts the B-Spline mapping in order to have a clamped knot vector. 

More details can be found in :cite:`piegl` Chapter 12.

create
______

Creates a B-Spline volume given degrees **p_u**, **p_v** and **p_w**, knots vectors **knots_u** **knots_v** and **knots_w** and a **control points** array. The user can also specify the **optional** array containing the **weights**. 

duplicate
_________

Duplicates the current object. Memory for the output object is allocated here.

elevate
_______

elevates the polynomial degree (number of elevation = times). 
We refeer to :cite:`piegl` Chapter 5 section 5, for more details.

evaluate
________

Evaluates a B-Spline volume on **1D** array sites for every direction. The result is the evaluation of the B-Spline volume over the correspongin **3D** grid.

evaluate_deriv
______________

Evaluates a B-Spline volume and its derivatives on **1D** array sites. The result is the evaluation of the B-Spline volume over the correspongin **3D** grid. 

export
______

Exports a B-Spline volume. Only the **namelist** format is available. 

extract
_______

Extracts a B-Spline surface from a B-Spline volume

free
____

Deallocates memory.

insert_knot
___________

Insert a new knot in the knots vector of a given **axis**. 
We refer to :cite:`piegl` Chapter 5, section 2, for more details.

print_info
__________

Prints information about the current object.

read_from_file
______________

Reads a B-Spline volume from a file. Only the **namelist** format is available for the moment.

refine
______

Refines a B-Spline volume. Refinment is done given the degrees, the number of elements in each direction. The user can also specify if this subroutine is inplace. 

Refining a B-Spline volume means either Degree elevation or knot insertion.

set_control_points
__________________

Sets new values for the control points.

set_weights
___________

Sets new values for the weights.

to_pp
_____

Converts the control points associated to the B-Spline families to a piecewise polynomial form. The result is an array corresponding to the pp-form on each element.

For more details, we refer the user to :cite:`DeBoor_Book2001`, Chapter 9.

to_us
_____

Gets the coefficients associated to the uniform clamped description of the current B-Spline volume.

unclamp
_______

Unclamping a B-Spline volume. 

More details can be found in :cite:`piegl` Chapter 12.


Attributs
^^^^^^^^^

n_u
___

number of control points in the u-direction

p_u
___

spline degree in the u-direction            

n_elements_u
____________

number of elements in the u-direction 

knots_u
_______

array of knots of size n_u+p_u+1 in the u-direction  

n_v
___

number of control points in the v-direction

p_v
___

spline degree in the v-direction            

n_elements_v
____________

number of elements in the v-direction 

knots_v
_______

array of knots of size n_v+p_v+1 in the v-direction  

n_w
___

number of control points in the w-direction

p_w
___

spline degree in the w-direction            

n_elements_w
____________

number of elements in the w-direction 

knots_w
_______

array of knots of size n_w+p_w+1 in the w-direction  

control_points
______________

array of control points in :math:`\mathbb{R}^d`

weights
_______

array of weights 

grid_u
______

corresponding grid in the u-direction 

i_spans_u
_________

knots indices corresponding to the grid in the u-direction 

grid_v
______

corresponding grid in the v-direction 

i_spans_v
_________

knots indices corresponding to the grid in the v-direction 

grid_w
______

corresponding grid in the w-direction 

i_spans_w
_________

knots indices corresponding to the grid in the w-direction 


spl_t_mapping_cad
*****************

The object mapping_cad allows the user to apply some common construction algorithms in Computer Aided Design.
More details can be found in :cite:`piegl`, Chapter 8. 

Methods
^^^^^^^

translate
_________

translates a mapping given a displacements array [inplace] 

extrude
_______

Construct a NURBS surface/volume by extruding a NURBS curve/surface.

spl_t_mapping_gallery
*********************

Subroutines
^^^^^^^^^^^

spl_mapping_linear
__________________

creates a 1D mapping line between two points :math:`P_1` and :math:`P_2` in :math:`\mathbb{R}^d`. The user can specify the final **degree** and number of elements **n_elements** [optional]. 

spl_mapping_arc
_______________

creates a 1D mapping arc.

.. todo:: Not yet finished: add angle, center and radius as optional args

spl_mapping_bilinear
____________________

creates a 2D bilinear mapping using the points :math:`P_{11}, P_{12}, P_{21}` and :math:`P_{22}` in :math:`\mathbb{R}^d`. The user can specify the final **degrees** and number of elements **n_elements** arrays [optional].   

spl_mapping_annulus
___________________

creates a 2D mapping annulus of a minimal radius **r_min** and maximal radius **r_max**. The user can also specify the **center** as an array [optional]. 

spl_mapping_circle
__________________

creates a 2D circular mapping of a radius **radius** [optional] and a center **center** as an array [optional]. 

spl_mapping_trilinear
_____________________

creates a 3D trilinear mapping given the points  :math:`P_{ijk}, ~ i,j,k \in \{ 1,2 \}` in :math:`\mathbb{R}^d`. The user can specify the final **degrees** and number of elements **n_elements** arrays [optional].   


.. Local Variables:
.. mode: rst
.. End:
