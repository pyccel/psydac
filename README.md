# Welcome to PSYDAC

[![build-devel](https://travis-ci.com/pyccel/psydac.svg?branch=devel)](https://travis-ci.com/pyccel/psydac) [![docs](https://readthedocs.org/projects/spl/badge/?version=latest)](http://spl.readthedocs.io/en/latest/?badge=latest)

**PSYDAC** is a Python 3 Library for isogeometric analysis.

### Requirements

* **Python3**:
  ```bash
  sudo apt-get install python3 python3-dev
  ```
* **pip3**:
  ```bash
  sudo apt-get install python3-pip
  ```
* All *Python* dependencies can be installed using:
  ```bash
  export CC="mpicc"
  export HDF5_MPI="ON"
  export HDF5_DIR=/path/to/hdf5/openmpi
  python3 -m pip install -r requirements.txt
  ```
### Installing the library

* **Standard mode**:
  ```bash
  python3 -m pip install .
  ```
* **Development mode**:
  ```bash
  python3 -m pip install --user -e .
  ```

### Uninstall

* **Whichever the install mode**:
  ```bash
  python3 -m pip uninstall psydac
  ```

### Running tests
```bash
export PSYDAC_MESH_DIR=/path/to/psydac/mesh/
python3 -m pytest --pyargs psydac -m "not parallel"
python3 /path/to/psydac/mpi_tester.py --pyargs psydac -m "parallel"
```

### Mesh Generation

After installation, a command `psydac-mesh` will be available.

##### Example of usage  
```bash 
psydac-mesh -n='16,16' -d='3,3' square mesh.h5
```


