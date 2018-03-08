Welcome to SPL
==============

|build-status| |docs|

**SPL** is a Python/Fortran 2003 library for B-Splines/NURBS and Computer Aided Design Algorithms. 

Install
*******

In your terminal, define your installation path (where you want to put the fortran/python libraries/packages) and run::

  export PREFIX=__ADD_YOUR_INSTALLATION_PATH__
  mkdir build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=$PREFIX  ..
  make && make install
  cd ..

For Python users
^^^^^^^^^^^^^^^^

If you intend to use the *Python* package, do not forget to use the **-fPIC** flag. 
In the build directory, run::

  cmake -DCMAKE_Fortran_FLAGS="-fPIC" -DCMAKE_INSTALL_PREFIX=$PREFIX  ..
  make && make install
  cd ..

then::

  python setup.py install --prefix=$PREFIX

Note that in **setup.py** we check if **PREFIX** is an environment variable, if not we assume that **spl** has been installed in **__PATH_TO_SPL_PROJECT__/usr**.

More information
^^^^^^^^^^^^^^^^

First, make sure you have

* **git** *optional*
* **cmake** (version 2.8 or higher)
* **Fortran** 2003 Compiler (gfortran version 4.7 or higher, or appropriate ifort)

Compilers
_________

**SPL** was tested with the following compilers

* gcc: 4.7, 4.8.4, 4.8.5, 4.9.3, 5.4
* intel: 15.0.4, 16.0.3. mpiifort 4.1.3, 5.0, 5.1
* pgi

Dependencies
____________

* LAPACK (optional)
* MPI (optional)
* MPICH2

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

  make html

Finnaly, the html documentation will be available in the following directory::

  PATH-TO-SPL-DIR/doc/sphinx/_build/html

Whenever needed, pleases use **sphinx-apidoc** in order to update **rtd** website::

  cd doc/sphinx
  sphinx-apidoc -o api-python/ ../../spl

To update the *Fortran* api, you will need to use *Doxygen*::

  cd doc/doxygen
  doxygen Doxyfile

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
    


.. |build-status| image:: https://travis-ci.org/pyccel/spl.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/pyccel/spl

.. |docs| image:: https://readthedocs.org/projects/spl/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: http://spl.readthedocs.io/en/latest/?badge=latest


