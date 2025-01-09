# Welcome to PSYDAC

This is a fork of psydac which contains functionality needed for struphy which should be kept close to the original code.

Over time, features from psydac-for-struphy will be merged with psydac/devel.

Eventually, the goals is that struphy uses the real psydac/devel instead of this fork.

## Important

* Some pytests have been modified due to the requirement that we use a domain length of 1, see [this assert statement](https://github.com/max-models/psydac-for-struphy/blob/76f039ac9406675548ffd8c753b0292e2d0596b4/psydac/feec/global_projectors.py#L734).
* All of test_feec_maxwell_multipatch_2d.py has been commented out since it is not needed for Struphy, and the domain sizes are not 1
* Skip `test_maxwell_2d_dirichlet_spline_mapping` since `spline_mapping=True` gives incompatible shapes for `F.jacobian(eta1, eta2)` in `pull_2d_hcurl` (for example: `F.jacobian(eta1, eta2).shape = (2, 36, 10, 2)`), this is not sliced correctly with `[..., 0, 0]`.

[![devel_tests](https://github.com/pyccel/psydac/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/continuous-integration.yml) [![docs](https://github.com/pyccel/psydac/actions/workflows/documentation.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/documentation.yml)

**PSYDAC** is a Python 3 Library for isogeometric analysis.

## Table of contents

-   [Requirements](#requirements)
-   [Python setup and project download](#python-setup-and-project-download)
-   [Installing the library](#installing-the-library)
-   [Optional PETSc installation](#optional-petsc-installation)
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
These dependencies can be installed from the cloned directory `<ROOT-PATH>/psydac` with the following steps.

First, set an environment variable with the path to the parallel HDF5 library.
This path can be obtained with a command which depends on your system.

-   **Ubuntu/Debian**:
    ```sh
    export HDF5_DIR=$(dpkg -L libhdf5-openmpi-dev | grep "libhdf5.so" | xargs dirname)
    ```

-   **macOS**:
    ```sh
    export HDF5_DIR=$(brew list hdf5-mpi | grep "libhdf5.dylib" | xargs dirname | xargs dirname)
    ```

Next, install the Python dependencies using `pip`:
```sh
export CC="mpicc"
export HDF5_MPI="ON"

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_extra.txt --no-build-isolation
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

## Optional PETSc installation

Although Psydac provides several iterative linear solvers which work with our native matrices and vectors, it is often useful to access a dedicated library like [PETSc](https://petsc.org). To this end, our matrices and vectors have the method `topetsc()`, which converts them to the corresponding `petsc4py` objects.
(`petsc4py` is a Python package which provides Python bindings to PETSc.) After solving the linear system with a PETSc solver, the function `petsc_to_psydac` allows converting the solution vector back to the Psydac format.

In order to use these additional feature, PETSc and petsc4py must be installed as follows.
First, we download the latest release of PETSc from its [official Git repository](https://gitlab.com/petsc/petsc):
```sh
git clone --depth 1 --branch v3.22.2 https://gitlab.com/petsc/petsc.git
```
Next, we specify a configuration for complex numbers, and install PETSc in a local directory:
```sh
cd petsc

export PETSC_DIR=$(pwd)
export PETSC_ARCH=petsc-cmplx

./configure --with-scalar-type=complex --with-fortran-bindings=0 --have-numpy=1

make all check

cd -
```
Finally, we install the Python package `petsc4py` which is included in the `PETSc` source distribution:
```sh
python3 -m pip install wheel Cython numpy
python3 -m pip install petsc/src/binding/petsc4py
```

## Uninstall

-   **Whichever the install mode**:
    ```bash
    python3 -m pip uninstall psydac
    ```
-   **If PETSc was installed**:
    ```bash
    python3 -m pip uninstall petsc4py
    ```

The non-Python dependencies can be uninstalled manually using the package manager.
In the case of PETSc, it is sufficient to remove the cloned source directory given that the installation has been performed locally.

## Running tests

Let `<PSYDAC-PATH>` be the installation directory of Psydac.
In order to run all serial and parallel tests which do not use PETSc, just type:
```bash
export PSYDAC_MESH_DIR=<PSYDAC-PATH>/mesh/
python3 -m pytest --pyargs psydac -m "not parallel and not petsc"
python3 <PSYDAC-PATH>/mpi_tester.py --pyargs psydac -m "parallel and not petsc"
```

If PETSc and petsc4py were installed, some additional tests can be run:
```bash
python3 -m pytest --pyargs psydac -m "not parallel and petsc"
python3 <PSYDAC-PATH>/mpi_tester.py --pyargs psydac -m "parallel and petsc"
```

## Speeding up **Psydac**'s core

Many of Psydac's low-level Python functions can be translated to a compiled language using the [Pyccel](https://github.com/pyccel/pyccel) transpiler. Currently, all of those functions are collected in modules which follow the name pattern `[module]_kernels.py`.

The classical installation translates all kernel files to Fortran without user intervention. This does not happen in the case of an editable install, but the command `psydac-accelerate` is made available to the user instead. This command applies Pyccel to all the kernel files in the source directory. The default language is currently Fortran, C should also be supported in a near future.

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
