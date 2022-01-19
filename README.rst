Welcome to PSYDAC
=================

|build-devel| |docs|

**PSYDAC** is a Python 3 library for isogeometric analysis. 

Requirements
************

- **Python3**::

    sudo apt-get install python3 python3-dev

- **pip3**::

    sudo apt-get install python3-pip

- All *Python* dependencies can be installed using::

    export CC="mpicc"
    export HDF5_MPI="ON"
    export HDF5_DIR=/path/to/hdf5/openmpi 
    python3 -m pip  install -r requirements.txt

Installing library
******************

- **Standard mode**::

    python3 -m pip install .

- **Development mode**::

    python3 -m pip install --user -e .
    
Uninstall
*********

- **Whichever the install mode**::

    python3 -m pip uninstall psydac

Running tests
*********
.. codeblock:: bash

   export PSYDAC_MESH_DIR=/path/to/psydac/mesh/
   python -m pytest --pyargs psydac -m "not parallel"
   python /path/to/psydac/mpi_tester.py --pyargs psydac -m "parallel"
   

   
 .. |build-devel| image:: https://travis-ci.com/pyccel/psydac.svg?branch=devel
    :alt: devel status
    :scale: 100%
    :target: https://travis-ci.com/pyccel/psydac

.. |docs| image:: https://readthedocs.org/projects/spl/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: http://spl.readthedocs.io/en/latest/?badge=latest

Mesh Generation
***************

After installation, a command line **psydac-mesh** will be available.


Example of usage
^^^^^^^^^^^^^^^^

.. code-block:: bash

  psydac-mesh -n='16,16' -d='3,3' square mesh.h5
