Docs Maintenance
================

Even though the content of this documentation is generated automatically, its structure is handwritten and, unless we use custom sphinx templates in the future, must be maintained.

This section shall serve as a short guide on how to maintain this documentation.

   * :ref:`The structure <structure>`
   * :ref:`Adding or deleting a class or a (public) function to a module <simplechange>`
   * :ref:`Adding or deleting a module <heavychange>`
   * :ref:`Changing a class or a function within a module <nochange>`

.. _structure:

The structure
-------------

All of Psydac's documentation is contained in the ``/docs`` folder.
This /docs folder consists of one directory, ``/source``, as well as an empty ``.nojekyll`` file, a ``Makefile`` and ``make.bat``.

For the sake of maintenance, only the ``/source`` folder is of interest.

docs/source
###########

This folder gives structure to the documentation. As this structure is self made, it needs to be maintained as Psydac grows.

   * Every Sphinx documentation must contain an ``index.rst``. Sphinx will generate ``index.html`` from this file, the index page of this documentation.
   * This ``index.rst`` file must contain a toctree (``.. toctree::``) containing all "content" that is not included on the index page itself. As of now, 06.06.2023, this amounts to three sections, namely modules, examples and maintenance. Each of these sections has its own .rst file, ``modules.rst``, ``examples.rst`` and ``maintenance.rst`` respectively. All .rst files which don't have a path to this root toctree will not be part of the documentation.
   * Further, you can find ``conf.py`` and as of now ``wip.gif``. Both are not relevant for the sake of maintenance.

As of now, maintaining the documentation is equivalent to maintaining the developers documentation, i.e., the modules section.

The modules section
###################

The modules section, for the sake of tidiness, consists of one modules.rst file in ``docs/source`` and the directory ``docs/source/modules``. 

   * The ``modules.rst`` file consists only of a toctree linking to all the .rst files in ``docs/source/modules``. Thus, ``modules.rst`` must only be changed if we wish to document more or less subpackages than, as of now, psydac.api/ddm/feec/fem/linalg/mapping/polar/utilities.
   * The ``docs/source/modules`` directory consists of, as of now, eight .rst files and eight folders, one of each for each of psydacs subpackages api/ddm/feec/fem/linalg/mapping/polar/utilities which are currently part of the documentation. If we also wanted to document the subpackage ``psydac.cad`` for example, we'd have to add the respective structure here as well as add ``psydac/cad`` to the toctree of the ``modules.rst`` file as described above.

Individual modules
##################

Unless major changes take place, we only have to update existing files in the ``docs/source/modules`` directory, not add or remove any.
As it was the case for the ``modules.rst`` file, each subpackage .rst file (``api.rst``, ``ddm.rst``, ...) consists only of a (hidden) toctree containing 
all of its modules (``api.basic``, ``api.discretization``, ...) as well as a visually more pleasing table which also highlights module docstrings.
Hence, ``api.rst`` for example must only be updated if modules are being added to or deleted from the ``psydac.api`` subpackage.

More regularly though, pull requests will only add or delete classes and functions of modules.
Such changes are quickly applied to the documentation, as such changes must be addressed in one file only, see :ref:`below <simplechange>`. 
The file in question is the .rst file of the corresponding module and it contains all classes and (public) functions of such a module.

.. _simplechange:

Adding or deleting a class or a (public) function to a module
-------------------------------------------------------------

1) Adding a class

With a coming PR, the class ``GMRES`` will be added to the ``psydac.linalg.solvers`` module.
This module, prior to the PR, consisted of one function, ``inverse``, as well as six classes, ``ConjugateGradient``, ``PConjugateGradient``, ... and ``LSMR``. 
Upon adding ``GMRES``, the ``docs/source/modules/linalg/solvers.rst`` file has to be changed as follows:

.. code-block:: rst

   linalg.solvers
   ==============

      * :ref:`inverse <inverse>`
      * :ref:`ConjugateGradient <conjugategradient>`
      * :ref:`PConjugateGradient <pconjugategradient>`
      * :ref:`BiConjugateGradient <biconjugategradient>`  
      * :ref:`BiConjugateGradientStabilized <biconjugategradientstabilized>`
      * :ref:`MinimumResidual <minimumresidual>`
      * :ref:`LSMR <lsmr>`
      * :ref:`GMRES <gmres>`				# this line is new

   .. inheritance-diagram:: psydac.linalg.solvers

   .. _inverse:

   inverse
   -------

   .. autofunction:: psydac.linalg.solvers.inverse

   .. _conjugategradient:

   ConjugateGradient
   -----------------

   .. autoclass:: psydac.linalg.solvers.ConjugateGradient
      :members:
      
   ...   					        # five further classes later, add:
   
   .. _gmres:						# this line is new
   
   GMRES						# this line is new
   -----						# this line is new
   
   .. autoclass:: psydac.linalg.solvers.GMRES		# this line is new
      :members:						# this line is new
      
2) Adding a (public) function

A good example would have been the addition of the ``inverse`` function to the ``psydac.linalg.solvers`` module a few months ago. 
Adding such a function to the documentation is very similar to adding a new class, with the exception that we have to use the 
``.. autofunction::`` and not the ``.. autoclass::`` directive as well as not include ``:members:``, see above.

3) Deleting

Remove the few lines above.

.. _heavychange:

Adding or deleting a module
---------------------------

1) Adding a module

Say we want to add the (already in this documentation existing) module ``psydac.fem.basic``.
This amounts to:

   * add ``fem/basic`` to the toctree in ``docs/source/modules/fem.rst``
   * ``mkdir basic.rst`` in ``docs/source/modules/fem``
   * Copy-Paste the structure of such a module .rst file and fit to ``psydac.fem.basic``.

As ``psydac.fem.basic`` consists of two classes only, ``FemSpace`` and ``FemField``, the ``basic.rst`` file should look like this:

.. code-block:: rst

   fem.basic
   =========

      * :ref:`FemSpace <femspace>`
      * :ref:`FemField <femfield>`

   .. inheritance-diagram:: psydac.fem.basic

   .. _femspace:

   FemSpace
   --------

   .. autoclass:: psydac.fem.basic.FemSpace
      :members:

   .. _femfield:

   FemField
   --------

   .. autoclass:: psydac.fem.basic.FemField
      :members:
      
Note that if a module consists of functions only and no class, removing the ``inheritance-diagram`` line is necessary.

2) Deleting

Undo the above.

.. _nochange:

Changing a class or a (public) function within a module
-------------------------------------------------------

No changes must be made to the documentation if no changes to the underlying code structure are made. That is, if we only change the way the 
function ``inverse`` in ``psydac.linalg.solvers`` works, or if we only change local variables or the solve-algorithm within the ``ConjugateGradient``
class in ``psydac.linalg.solvers`` for example, no changes have to be made to the documentation.
