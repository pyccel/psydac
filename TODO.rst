TODO
====

* Add GMRES solver to 'spl.linalg.solvers'

* Add unit tests for Cart subcommunicators in 'spl.ddm.cart'

* Create parallel 'KroneckerLinearOperator' in 'spl.linalg.kronecker' using Cart subcommunicators

* Create object of type 'spl.fem.splines.SplineSpace' with Cart

* Extend functionality of 'SplineSpace' class:
  . add (private?) methods 'init_fem' and 'init_collocation'
  . call methods above when required
  . add method 'compute_interpolant( self, values, field )'
  . implement 'eval_field_gradient' method

* Create 'SplineMapping' class in module 'spl.mapping.discrete'


Core
****

* interface to *spl_eval_splines_ders*
