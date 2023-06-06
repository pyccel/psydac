.. PSYDAC documentation master file, created by
   sphinx-quickstart on Fri Apr 21 11:24:31 2023.

   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PSYDAC's documentation!

Contents
--------
.. toctree::
   :maxdepth: 1
   
   modules
   examples
   maintenance

.. _note:
.. note::

   .. image:: wip.gif
      :width: 400

   This documentation is still under construction.
   For the time being, its purpose is to assist the developers.
   Find our github `here <https://github.com/pyccel/psydac/tree/devel/>`_.

+--------------------------------------------------------------------------------------------------------------------------------+
|Psydac is a high-level finite-element library in Python 3, that uses high-order splines, mapped domains and MPI parallelization.|
|                                                                                                                                |
|In order to use Psydac, the user provides a geometry analytically or through an input file, and then defines the model equations|
|in symbolic form (weak formulation) using Sympde, which provides the mathematical expressions and checks the semantic validity  |
|of the model.                                                                                                                   |   
|                                                                                                                                |                              
|Once a finite element discretization has been chosen, Psydac maps the abstract concepts to concrete objects, the basic building |
|blocks being MPI-distributed vectors and matrices.                                                                              |
|For all the computationally intensive operations (matrix and vector assembly, matrix-vector products, etc.), Psydac generates   |
|ad-hoc Python code which is accelerated using either Numba or Pyccel.                                                           |
+--------------------------------------------------------------------------------------------------------------------------------+

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
