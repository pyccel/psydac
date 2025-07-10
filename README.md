# Welcome to PSYDAC

[![devel_tests](https://github.com/pyccel/psydac/actions/workflows/testing.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/testing.yml) [![docs](https://github.com/pyccel/psydac/actions/workflows/documentation.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/documentation.yml)

**PSYDAC** is a Python 3 Library for isogeometric analysis.

## Installation

PSYDAC requires a certain number of components to be installed on the machine:

-   Fortran and C compilers with OpenMP support
-   OpenMP library
-   BLAS and LAPACK libraries
-   MPI library
-   HDF5 library with MPI support

The installation instructions depend on the operating system and on the packaging manager used.
It is particularly important to determine the **HDF5 root folder**, as this will be needed to install the [`h5py`](https://docs.h5py.org/en/latest/build.html#source-installation) package in parallel mode.
Detailed instructions can be found in the [documentation](installation.md).

Once those components are installed, we recommend using [`venv`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) to set up a fresh Python virtual environment at a location `<ENV-PATH>`:
```bash
python3 -m venv <ENV-PATH>
source <ENV-PATH>/bin/activate
```

PSYDAC and its Python dependencies can now be installed in the virtual environment using [`pip`](https://pip.pypa.io/en/stable/), the Python package manager:
```bash
git clone https://github.com/pyccel/psydac.git

export CC="mpicc"
export HDF5_MPI="ON"
export HDF5_DIR=<HDF5-PATH>

pip install --upgrade pip
pip install h5py --no-cache-dir --no-binary h5py
pip install psydac
```
Here `<HDF5-PATH>` is the path to the HDF5 root folder, such that `<HDF5-PATH>/lib/` contains the HDF5 dynamic libraries with MPI support.
For an editable install, the `-e/--editable` flag should be provided to the last command above.

Again, for more details we refer to our [documentation](./installation.md).

> [!TIP]
> PSYDAC provides the functionality to convert its MPI-parallel matrices and vectors to their [PETSc](https://petsc.org) equivalent, and back.
> This gives the user access to a wide variety of linear solvers and other algorithms.
> Instructions for installing [PETSc](https://petsc.org) and `petsc4py` can be found in our [documentation](installation.md#optional-petsc-installation).

## Running tests

The test suite of PSYDAC is based on [`pytest`](https://docs.pytest.org/en/stable/), which should be installed in the same virtual environment:
```bash
source <ENV-PATH>/bin/activate
pip install pytest
```

Let `<PSYDAC-PATH>` be the installation directory of PSYDAC.
In order to run all serial and parallel tests which do not use PETSc, just type:
```bash
export PSYDAC_MESH_DIR=<PSYDAC-PATH>/mesh/
pytest --pyargs psydac -m "not parallel and not petsc"
python <PSYDAC-PATH>/mpi_tester.py --pyargs psydac -m "parallel and not petsc"
```

If PETSc and petsc4py were installed, some additional tests can be run:
```bash
pytest --pyargs psydac -m "not parallel and petsc"
python <PSYDAC-PATH>/mpi_tester.py --pyargs psydac -m "parallel and petsc"
```

## Speeding up **Psydac**'s core

Many of PSYDAC's low-level Python functions can be translated to a compiled language using the [Pyccel](https://github.com/pyccel/pyccel) transpiler. Currently, all of those functions are collected in modules which follow the name pattern `[module]_kernels.py`.

The classical installation translates all kernel files to Fortran without user intervention. This does not happen in the case of an editable install, but the command `psydac-accelerate` is made available to the user instead. This command applies Pyccel to all the kernel files in the source directory. The default language is currently Fortran, C should also be supported in a near future.

-   **Only in development mode**:
    ```bash
    python /path/to/psydac/psydac_accelerate.py [--language LANGUAGE] [--openmp]
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
