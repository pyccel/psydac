TODO
====

* 'StencilVector' class in 'spl.linalg.stencil':

  . in parallel version, toarray() accepts 'ghost' flag => if True, include ghost regions

* In directory 'spl/linalg/tests':

  . add extensive unit tests to 'test_stencil_vector.py'
  . add extensive unit tests to 'test_stencil_matrix.py'

* Update method 'tocoo()' of class 'spl.linalg.StencilMatrix' (parallel version)

* Add GMRES solver to 'spl.linalg.solvers'

* Add unit tests for Cart subcommunicators in 'spl.ddm.cart'

* Create parallel 'KroneckerLinearOperator' in 'spl.linalg.kronecker' using Cart subcommunicators

* Create object of type 'spl.fem.splines.SplineSpace' with Cart
