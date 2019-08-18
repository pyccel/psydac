TODO
====

* mv psydac/feec/tests/todo_test_derivatives.py to psydac/feec/tests/test_derivatives.py once it is fixed

* Add GMRES solver to 'psydac.linalg.solvers'

* Add unit tests for Cart subcommunicators in 'psydac.ddm.cart'

* Create parallel 'KroneckerLinearOperator' in 'psydac.linalg.kronecker' using Cart subcommunicators

* Create object of type 'psydac.fem.psydacines.SplineSpace' with Cart

* Extend functionality of 'SplineSpace' class:
  . add (private?) methods 'init_fem' and 'init_collocation'
  . call methods above when required
  . add method 'compute_interpolant( self, values, field )'
  . implement 'eval_field_gradient' method

* Create 'SplineMapping' class in module 'psydac.mapping.discrete'

* add a section in documentation about hdf5 installation (serial/parallel) + h5-tools (install+usage)


Core
****

* interface to *psydac_eval_psydacines_ders*

API
***

- reorganize code in api (no change to codgen)

- additional tests and unit tests in api/tests

- for the moment the codegen is not using the support from space:

  * allow *fem_context* to be called with grid and degrees, and return a discrete space and a None for mapping.

  * use support in codegen/interface

- normal vector in 3d

- periodic bc in codegen

- init_fem is called everytime wa call discretize: maybe we should first do some checks (nderiv is ok norder etc)

- hessian not yet implemented in sympde and api

- add other solvers to the solver_driver (only cg is available now)

- remove psydac/run_tests.sh. however, we need to clear the cache of sympy after 1d, 2d and 3d tests, otherwise pytest will crash.

- add sympde and pyccel install procedure; maybe wget to download the requierements files as requirements_pyccel.txt etc then call pip3

- api tests should be moved to examples

- shall we add the Boundayr object as an attribut of SplineMapping? what to do if we don't use a mapping? we can also add the notion of a patch ... 

- DiscreteModel is not up to date; changes need to be done because of the SumForm concept that was introduced lately. in fact, things will become even easier. a Model will be like a namespace or a Module.

- add BoundaryDomain to define the whole boundary of a domain
