Welcome to SPL
==============

|build-devel| |docs|

**SPL** is a Python/Fortran 2003 library for B-Splines/NURBS and Computer Aided Design Algorithms. 

Requirements
************

- **Python3**::

    sudo apt-get install python3 python3-dev

- **pip3**::

    sudo apt-get install python-pip3

- **Fortran** compiler

- all *Python* dependencies can be installed using::

    sudo -H pip3 install -r requirements.txt

Install
*******
run::

    python3 -m pip install

- **Development mode**::

    python3 -m pip install --user -e .
    
Uninstall
*********

- **run**::

    python3 -m pip uninstall spl
    
.. |build-devel| image:: https://travis-ci.org/pyccel/spl.svg?branch=devel
    :alt: devel status
    :scale: 100%
    :target: https://travis-ci.org/pyccel/spl

.. |docs| image:: https://readthedocs.org/projects/spl/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: http://spl.readthedocs.io/en/latest/?badge=latest

  

