# Welcome to PSYDAC

[![devel_tests](https://github.com/pyccel/psydac/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/continuous-integration.yml) [![docs](https://github.com/pyccel/psydac/actions/workflows/documentation.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/documentation.yml)

**PSYDAC** is a Python 3 Library for isogeometric analysis.

## Table of contents

-   [Requirements](#requirements)
-   [Python setup and project download](#python-setup-and-project-download)
-   [Installing the library](#installing-the-library)
-   [Uninstall](#uninstall)
-   [Running tests](#running-tests)
-   [Speeding up Psydac's core](#speeding-up-psydacs-core)
-   [User Documentation](#user-documentation)
-   [Code Documentation](#code-documentation)

## Requirements

Psydac requires a certain number of components to be installed on the machine:

-   Fortran and C compilers with OpenMP support
-   OpenMP library
-   BLAS and LAPACK libraries
-   MPI library
-   HDF5 library with MPI support

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
`PETSc` has to be installed from source with configuration for complex numbers:
```sh
./configure --with-scalar-type=complex --with-fortran-bindings=0 --have-numpy=1 
make PETSC_DIR=$(pwd) PETSC_ARCH=petsc-complex-c-debug all check
```
Lastly, `petsc4py` has to be installed:
```sh
python -m pip install wheel Cython numpy
PETSC_DIR=$(pwd) PETSC_ARCH=petsc-complex-c-debug pip install src/binding/petsc4py
```
This installation is also valid for macOS.

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

## Python setup and project download

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

## Installing the library

Psydac depends on several Python packages, which should be installed in the newly created virtual environment.
These dependencies can be installed from the cloned directory `<ROOT-PATH>/psydac` using
```sh
export CC="mpicc"
export HDF5_MPI="ON"
export HDF5_DIR=/path/to/parallel/hdf5

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_extra.txt --no-build-isolation
```
where the `HDF5_DIR` environment variable should store the absolute path to the **installation point** of your parallel HDF5 library, which can be found with
```sh
h5pcc -showconfig
```
or (on macOS)
```sh
brew info hdf5-mpi
```

At this point the Psydac library may be installed in **standard mode**, which copies the relevant files to the correct locations of the virtual environment, or in **development mode**, which only installs symbolic links to the Psydac directory. The latter mode allows one to effect the behavior of Psydac by modifying the source files.

-   **Standard mode**:
    ```bash
    python3 -m pip install .
    ```

-   **Development mode**:
    ```bash
    python3 -m pip install --editable .
    ```

## Uninstall

-   **Whichever the install mode**:
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

Many of Psydac's low-level Python functions can be translated to a compiled language using the [Pyccel](https://github.com/pyccel/pyccel) transpiler. Currently, all of those functions are collected in modules which follow the name pattern `[module]_kernels.py`.

The classical installation translates all kernel files to Fortran without user intervention. This does not happen in the case of an editable install, but the command `psydac-accelerate` is made available to the user instead. This command applies Pyccel to all the kernel files in the source directory, and the C language may be selected instead of Fortran (which is the default).

-   **Only in development mode**:
    ```bash
    python3 /path/to/psydac/psydac_accelerate.py [--language LANGUAGE] [--openmp]
    ```

## User documentation

-   [Output formats](./output.md)
-   [Notebook examples](./examples/notebooks/)
-   [Other examples](./examples/)

## Code documentation

Find our latest code documentation [here](https://pyccel.github.io/psydac/).

## Mesh Generation

After installation, a command `psydac-mesh` will be available.

### Example of usage

```bash
psydac-mesh -n='16,16' -d='3,3' square mesh.h5
```
