TODO
====

* Change class names in 'spl.linalg.stencil':

  . VectorSpace >> StencilVectorSpace
  . Vector      >> StencilVector
  . Matrix      >> StencilMatrix 

* In directory 'spl/linalg/tests':

  . add assert statements to 'test_matrix.py'
  . merge 'test_matrix.py' into 'test_stencil_matrix.py'
  . add extensive unit tests to 'test_stencil_matrix.py'

* Update method 'tocoo()' of class 'spl.linalg.StencilMatrix' (parallel version)

* Add GMRES solver to 'spl.linalg.solvers'

* Add unit tests for Cart subcommunicators in 'spl.ddm.cart'

* Create parallel 'KroneckerLinearOperator' in 'spl.linalg.kronecker' using Cart subcommunicators

* Create object of type 'spl.fem.splines.SplineSpace' with Cart
