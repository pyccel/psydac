SPL
===

**SPL** is a Fortran 2003/2008 library for B-Splines/NURBS and Computer Aided Design Algorithms. 

Getting Started
***************

First, make sure you have

* **git** *optional*
* **cmake** (version 2.8 or higher)
* **Fortran** 2003 Compiler (gfortran version 4.7 or higher, or appropriate ifort)

Compilers
^^^^^^^^^

**SPL** was tested with the following compilers

* gcc: 4.7, 4.8.4, 4.8.5, 4.9.3, 5.4
* intel: 15.0.4, 16.0.3. mpiifort 4.1.3, 5.0, 5.1
* pgi

Dependencies
^^^^^^^^^^^^

* CLAPPIO (mandatory). CLAPPIO_ is a part from the CLAPP_ framework.
* PLAF (mandatory). PLAF_ is a part from the CLAPP_ framework.
* LAPACK (optional)
* PETSC (optional)
* AGMG (optional)
* MPI (optional)
* MPICH2

.. _CLAPP: https://gitlab.mpcdf.mpg.de/groups/clapp
.. _CLAPPIO: https://gitlab.mpcdf.mpg.de/clapp/clappio
.. _SPL: https://gitlab.mpcdf.mpg.de/clapp/spl
.. _PLAF: https://gitlab.mpcdf.mpg.de/clapp/plaf


Tested configurations
^^^^^^^^^^^^^^^^^^^^^

TODO

.. note:: Don't forget to put your ssh public key in your gitlab account

Getting the library
^^^^^^^^^^^^^^^^^^^

In your terminal, run the following commands::

  git clone  git@gitlab.mpcdf.mpg.de:clapp/spl.git
  cd spl 
  # checkout the stable branch
  git checkout stable

You can also go to the devel branch, if you are willing to have access to our last developments::

  # checkout the devel branch
  git checkout devel

Compiling SPL
^^^^^^^^^^^^^

In your terminal, run::

  mkdir build && cd build
  cmake ..
  make
  make test

If you want to install **SPL**, you will need to define the prefix during the configuration step of **ccmake** or using the flag **CMAKE_INSTALL_PREFIX** for the **cmake** command line. Then run::

  make install

If the prefix is not specified then **cmake** will first look for the variable **CLAPP_DIR**, if not found, then the path **../usr** will be used as **default** path.

Now, you only need to export the variable **CLAPP_DIR** if it's not already defined. You can also add it to your *bashrc/bach_profile* file::

  export CLAPP_DIR=PATH_TO_SPL_SRC/usr


Building documentation
^^^^^^^^^^^^^^^^^^^^^^

Sphinx
______

Sphinx is a tool that makes it easy to create intelligent and beautiful documentation, written by Georg Brandl and licensed under the BSD license.

Sphinx can be installed using:

* Debian/Ubuntu: Install Sphinx using packaging system::

    apt-get install python-sphinx

* Mac OS X: Install Sphinx using MacPorts::

    sudo port install py27-sphinx 

You should also download the rtd sphinx theme, using for example::
  
  pip install sphinx_rtd_theme

Then, you can build the Html SPL documentation with::
  
  cd PATH-TO-SPL-DIR/doc/sphinx
  
  python initialize.py

  make html

Finnaly, the html documentation will be available in the following directory::

  PATH-TO-SPL-DIR/doc/sphinx/_build/html

Latex
_____

LaTeX is a high-quality typesetting system; it includes features designed for the production of technical and scientific documentation. LaTeX is available as free software for all OS distributions.

You can build the Pdf SPL documentation with::

  cd PATH-TO-SPL-DIR/doc/sphinx
  
  python initialize.py

  make latex

Finnaly, the pdf documentation will be available in the following directory::
  
  PATH-TO-SPL-DIR/doc/sphinx/_build/latex

**Remark:** full SPL documentation is available in the SPL_ repository
    
