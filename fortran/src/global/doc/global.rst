.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)
.. _global:

Global Variables
================

Subroutines
***********

spl_initialize
^^^^^^^^^^^^^^

Initializating **SPL** by calling **plf_initialize**. 

spl_finalize
^^^^^^^^^^^^

Finalizating **SPL** by calling **plf_finalize**. 

spl_time
^^^^^^^^

Runs **plf_time**.

Constants
*********
    
Mapping boundary ids 
^^^^^^^^^^^^^^^^^^^^

spl_mapping_boundary_min=-1
___________________________

id for the **min** boundary

spl_mapping_boundary_max=1
__________________________

id for the **max** boundary

Boundary Conditions
^^^^^^^^^^^^^^^^^^^

spl_bc_dirichlet_homogen=0
__________________________

Homogeneous Dirichlet Boundary Condition 

spl_bc_periodic=1
_________________

Periodic Boundary condition	

spl_bc_none=2
_____________

No Boundary Condition

spl_bc_unclamped=3
__________________

Unclamped B-Splines with no Boundary Condition

Mapping format output
^^^^^^^^^^^^^^^^^^^^^

spl_mapping_format_nml=0
________________________

Mapping in nml format 


.. Local Variables:
.. mode: rst
.. End:
