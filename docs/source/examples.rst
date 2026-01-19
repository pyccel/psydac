Examples
========

.. +------------------------------------------------------------------------------------------------------------------------+
.. | Here you will find examples of how to use PSYDAC and explanations thereof as well as links to notebooks in the future. |
.. +------------------------------------------------------------------------------------------------------------------------+ 

.. The notebooks get copied into the source directory by the continuous integration pipeline. 
.. The notebooks should have all output cleared before being committed to the repository.

Notebooks
---------
For the documentation, we provide several Jupyter notebooks that illustrate how to use PSYDAC to solve different types of problems.
They can be found in the `notebooks directory <https://github.com/pyccel/psydac/tree/devel/examples/notebooks>`_, 
but are also generated as part of the documentation and can be accessed here:


.. toctree::
   :maxdepth: 1
   :caption: Notebooks:

   examples/poisson_2d_square
   examples/Poisson_non_periodic
   examples/Helmholtz_non_periodic
   examples/fem_L2_projection
   examples/regularized_curlcurl_3D_VTK
   examples/petsc_eigenvalues_regularized_curlcurl
   examples/feec_curlcurl_eigenvalue
   examples/feec_time_harmonic_Maxwell
   examples/feec_vector_potential_torus

FEEC Examples
-------------

In the `FEEC examples directory <https://github.com/pyccel/psydac/tree/devel/examples/feec>`_, 
you will find several examples of how to use PSYDAC to solve problems arising within the Finite Element Exterior Calculus (FEEC) framework.
These examples include:


* Poisson problems
* General curl-curl eigenvalue problems
* Time-harmonic Maxwell equations with source terms
* Time-dependent Maxwell equations


Performance
-----------
There are also some examples in the `performance directory <https://github.com/pyccel/psydac/tree/devel/examples/performance>`_ explaining the assembly algorithms used in PSYDAC.
