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

.. _note:
.. note::

   This documentation is still under construction.
   For the time being, its purpose is to assist the developers.
   Find our GitHub repository `here <https://github.com/pyccel/psydac/tree/devel/>`_.

Psydac is a Python 3 library for isogeometric analysis (IGA). It uses high-order tensor-product splines, multi-patch mapped
domains, and hybrid MPI-OpenMP parallelization.

In order to use Psydac, the user provides a geometry analytically or through an input file, and then defines the model equations
in symbolic form (weak formulation) using `SymPDE <https://github.com/pyccel/sympde>`_, which provides the mathematical
expressions and checks the semantic validity of the model.

Once a finite element discretization has been chosen, Psydac maps the abstract concepts to concrete objects, the basic building
blocks being MPI-distributed vectors and matrices.
For all the computationally intensive operations (assembly of matrices, vectors, and norms, etc.), Psydac generates ad-hoc
Python code which is then accelerated to Fortran speed using `Pyccel <https://github.com/pyccel/pyccel>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
