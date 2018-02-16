TODO
====

Pre-RELEASE
***********

general
^^^^^^^

- make sure that every class has the method print_info                     
- setup: remove SPL_DIR, ... and clean make_project.sh
- bug linear_operator_diagonal when n_block_rows <> n_block_cols. the bug is in the dot operation
- dot kron does not work if n_blocks > 1
- resets pointers to null whenver they are used in SPL
+ make -j2  
- add make doc
- update README.md for every project SPL, SPL, and all our libraries 
- have 1 argument per line, when defining a function/subroutine
- check that pointers to classes are always initialized with null()
+ make free deferred in all objects 
+ add spl_t_abstract as the abstract class that all other objects extend
- use is_allocated in all objects
- use  GCC_COMPILING when needed
- recompile plaf with agmg

Documentation
^^^^^^^^^^^^^

- doxygen    (ahmed)
- slides     (ahmed)

inputs
^^^^^^^^^^^^^^^
- inputs folder for tests parameters   

REALSE
******

general
^^^^^^^

Documentation
^^^^^^^^^^^^^

- user guide (ahmed)



