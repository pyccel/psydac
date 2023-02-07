# Welcome to PSYDAC

[![build-devel](https://travis-ci.com/pyccel/psydac.svg?branch=devel)](https://travis-ci.com/pyccel/psydac) [![docs](https://readthedocs.org/projects/spl/badge/?version=latest)](http://spl.readthedocs.io/en/latest/?badge=latest)

**PSYDAC** is a Python 3 Library for isogeometric analysis.

## Table of contents

-   [Requirements](#requirements)
-   [Python setup and dependencies](#python-setup-and-dependencies)
-   [Installing the library](#installing-the-library)
-   [Uninstall](#uninstall)
-   [Running tests](#running-tests)
-   [Speeding up Psydac's core](#speeding-up-psydacs-core)
-   [User Documentation](#user-documentation)

## Requirements

Psydac requires a certain number of components to be installed on the machine:

- Fortran and C compilers with OpenMP support
- OpenMP library
- BLAS and LAPACK libraries
- MPI library
- HDF5 library with MPI support

The installations instructions depend on the operating system and on the packaging manager used.

### Linux Debian-Ubuntu-Mint

To install all requirements on a Linux Ubuntu operating system, just use APT, the Advanced Packaging Tool:
```sh
sudo apt update
sudo apt install python3 python3-dev python3-pip
sudo apt install gcc gfortran
sudo apt install libblas-dev liblapack-dev
sudo apt install libopenmpi-dev openmpi-bin
sudo apt install libomp-dev libomp5
sudo apt install libhdf5-openmpi-dev
```

### macOS

To install all the requirements on a macOS operating system we recommend using [Homebrew](https://brew.sh/):

```eh
brew update
brew install gcc
brew install openblas
brew install lapack
brew install open-mpi
brew install libomp
brew install hdf5-mpi
```

### Other operating systems

Please see the [instructions for the pyccel library](https://github.com/pyccel/pyccel#Requirements) for further details.


## Python setup and dependencies

We recommend creating a clean Python virtual environment using [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment):
```sh
python3 -m venv <ENV-PATH>
```
where `<ENV-PATH>` is the location to create the virtual environment.
(A new directory will be created at the required location.)

In order to activate the environment from a new terminal session just run the command
```sh
source <ENV-PATH>/bin/activate
```

One can clone the Psydac repository at any location `<ROOT-PATH>` in the filesystem which does not require administrator privileges, using either
```sh
git clone https://github.com/pyccel/psydac.git
```
or
```sh
git clone git@github.com:pyccel/psydac.git
```
The latter command requires a GitHub account.

At this point please take note of the **installation point** of your parallel HDF5 library, which can be found with
```sh
h5pcc -showconfig -echo || true
```
The absolute path to this folder should be stored in the `HDF5_DIR` environment variable for use in the next step.
```sh
export HDF5_DIR=/path/to/parallel/hdf5
```

Psydac depends on several Python packages, which should be installed in the newly created virtual environment.
These dependencies can be installed from the cloned directory `<ROOT-PATH>/psydac` using
```sh
export CC="mpicc" HDF5_MPI="ON"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_extra.txt --no-build-isolation
```

## Installing the library

*   **Standard mode**:
    ```bash
    python3 -m pip install .
    ```

*   **Development mode**:
    ```bash
    python3 -m pip install --user -e .
    ```

## Uninstall

*   **Whichever the install mode**:
    ```bash
    python3 -m pip uninstall psydac
    ```

## Running tests

```bash
export PSYDAC_MESH_DIR=/path/to/psydac/mesh/
python3 -m pytest --pyargs psydac -m "not parallel"
python3 /path/to/psydac/mpi_tester.py --pyargs psydac -m "parallel"
```

## Speeding up **Psydac**'s core

Some of the low-level functions in psydac are written in python in a way that can be accelerated by pyccel. Currently, all of those are in `psydac/core/kernels.py`, `psydac/core/bsplines_pyccel.py` and `psydac/linalg/kernels.py`.
```bash
cd path/to/psydac/core
pyccel kernels.py --language fortran
pyccel bsplines_pyccel.py --language fortran

cd ../linalg
pyccel kernels.py --language fortran
```

## User documentation

-   [Output formats](./output.md)
-   [Notebook examples](./examples/notebooks/)
-   [Other examples](./examples/)

## Mesh Generation

After installation, a command `psydac-mesh` will be available.

### Example of usage

```bash
psydac-mesh -n='16,16' -d='3,3' square mesh.h5
```
