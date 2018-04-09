Welcome to SPL
==============

|build-devel| |docs|

**SPL** is a Python/Fortran 2003 library for B-Splines/NURBS and Computer Aided Design Algorithms. 

Requirements
************

- **Python3**

- **pip3**

- **cmake** (version 2.8 or higher)

- **Fortran** 2003 Compiler (gfortran version 4.7 or higher, or appropriate ifort)

- all *Python* dependencies can be installed using::

    sudo -H pip3 install -r requirements.txt


Install
*******

We first install the Fortran library/headers using *cmake* (by default the installation path
is **$PWD/../usr**)::

  mkdir build && cd build
  cmake -DCMAKE_Fortran_FLAGS="-fPIC" ..
  make && make install && cd ..

For the *Python* package, you can install it using *pip*::

  sudo -H FORTRAN_INSTALL_DIR=$PWD/../usr pip3 install .

For Python3 users, use *pip3* instead of *pip*.

For developers only
********************

Whenever you change the fortran code (inside **fortran/src**), you need to recompile the fortran library, using::

  cd build && make clean && make install && cd ..

then use **pip3** as previously to install the python package.

.. note:: even if only change the python source, you need to reinstall it to take the modifications into account.

  This can be done with pip3 with an upgrade option:

    sudo -H FORTRAN_INSTALL_DIR=$PWD/../usr pip3 install --upgrade .


.. note:: for some reasons, you may need to clean your installation using::

    sudo rm -rf /usr/local/lib/python3.5/dist-packages/spl*

  then run the install command::

    sudo -H FORTRAN_INSTALL_DIR=$PWD/../usr pip3 install .

.. More information
.. ^^^^^^^^^^^^^^^^
.. 
.. Compilers
.. _________
.. 
.. **SPL** was tested with the following compilers
.. 
.. * gcc: 4.7, 4.8.4, 4.8.5, 4.9.3, 5.4
.. * intel: 15.0.4, 16.0.3. mpiifort 4.1.3, 5.0, 5.1
.. * pgi


    
.. |build-devel| image:: https://travis-ci.org/pyccel/spl.svg?branch=devel
    :alt: devel status
    :scale: 100%
    :target: https://travis-ci.org/pyccel/spl

.. |docs| image:: https://readthedocs.org/projects/spl/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: http://spl.readthedocs.io/en/latest/?badge=latest


