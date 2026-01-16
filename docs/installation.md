# Installation details

-   [Requirements](#requirements)
-   [Python setup](#python-setup)
-   [Installing the library](#installing-the-library)
-   [Optional PETSc installation](#optional-petsc-installation)
-   [Running tests and examples](#running-tests-and-examples)
-   [Uninstall](#uninstall)

## Requirements

PSYDAC requires a certain number of components to be installed on the machine:

-   Fortran and C compilers with OpenMP support
-   OpenMP library
-   BLAS and LAPACK libraries
-   MPI library
-   HDF5 library with MPI support

The installation instructions depend on the operating system and on the packaging manager used.

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

```sh
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

### High-performance computers using Environment Modules

Many high-performance computers use [Environment Modules](https://modules.sourceforge.net/).
On those systems one typically needs to load the correct versions (i.e. compatible with each other) of the modules `gcc`, `openmpi`, and `hdf5-mpi`, e.g.

```sh
module load gcc/15
module load openmpi/5.0
module load hdf5-mpi/1.14.1
```
OpenMP instructions should work out of the box.
For access to BLAS and LAPACK routines there are usually different options, we refer therefore to any documentation provided by the supercomputer's maintainers.

## Python setup

We recommend creating a clean Python virtual environment using [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment):
```sh
python3 -m venv <ENV-PATH>
```
where `<ENV-PATH>` is the location to create the virtual environment.
(A new directory will be created at the required location.)
In order to activate the environment just run the command
```sh
source <ENV-PATH>/bin/activate
```
At this point the commands `python` and [`pip`](https://pip.pypa.io/en/stable/) will refer to the Python 3 interpreter and package manager of the virtual environment, respectively.
Additionally, the command `deactivate` closes the environment.
It is good practice to keep `pip` up to date with
```sh
pip install --upgrade pip
```

## Installing the library

PSYDAC depends on several Python packages, which should be installed in the newly created virtual environment.
Almost all of these dependencies will be automatically installed by `pip` at the time of installing the `psydac` package later on.

The single exception is the `h5py` package, which needs to be installed in parallel mode.
This means that a wheel will be built from sources and linked to the local parallel HDF5 library.

To this end, we first set the environment variable `HDF5_DIR` s.t. the path `$HDF5_DIR/lib/` will correspond to the folder containing the dynamic library `libhdf5.so` (on Ubuntu/Debian) or `libhdf5.dylib` (on macOS).
This path can be obtained with a command which depends on your system.

-   **Ubuntu/Debian**:
    ```sh
    export HDF5_DIR=$(dpkg -L libhdf5-openmpi-dev | grep "libhdf5.so" | xargs dirname)
    ```

-   **macOS**:
    ```sh
    export HDF5_DIR=$(brew list hdf5-mpi | grep "libhdf5.dylib" | xargs dirname | xargs dirname)
    ```

- **High-performance computers using [Environment Modules](https://modules.sourceforge.net/)**:

    The correct location of the HDF5 library can be found using the `module show` command, which reveals any environment variables after the `setenv` keyword.
    For example, on this system both `HDF5_HOME` and `HDF5_ROOT` contain the information we need:

    ```sh
    > module show hdf5-mpi/1.14.1

    -------------------------------------------------------------------
    /mpcdf/soft/SLE_15/sub/gcc_15/sub/openmpi_5_0/modules/libs/hdf5-mpi/1.14.1:

    module-whatis   {HDF5 library 1.14.1 with MPI support, built for openmpi_5_0_7_gcc_15_1}
    conflict        hdf5-serial
    conflict        hdf5-mpi
    setenv          HDF5_HOME /mpcdf/soft/SLE_15/packages/skylake/hdf5/gcc_15-15.1.0-openmpi_5.0-5.0.7/1.14.1
    setenv          HDF5_ROOT /mpcdf/soft/SLE_15/packages/skylake/hdf5/gcc_15-15.1.0-openmpi_5.0-5.0.7/1.14.1
    prepend-path    PATH /mpcdf/soft/SLE_15/packages/skylake/hdf5/gcc_15-15.1.0-openmpi_5.0-5.0.7/1.14.1/bin
    -------------------------------------------------------------------
    ```

    Therefore it is sufficient to set

    ```sh
    export HDF5_DIR=$HDF5_HOME
    ```

Next, install `h5py` in parallel mode using `pip`:
```sh
export CC="mpicc"
export HDF5_MPI="ON"

pip install h5py --no-cache-dir --no-binary h5py
```

At this point the PSYDAC library may be installed from PyPI in **standard mode**, which copies the relevant files to the correct locations of the virtual environment, or it may be installed from a cloned directory in **development mode**, which only installs symbolic links. The latter mode allows one to affect the behavior of PSYDAC by modifying the source files.

-   **Standard mode** from PyPI:
    ```bash
    pip install "psydac[test]"
    ```

-   **Development mode** from GitHub:
    ```bash
    git clone --recurse-submodules https://github.com/pyccel/psydac.git
    cd psydac

    pip install meson-python "pyccel>=2.1.0"
    pip install --no-build-isolation --editable ".[test]"
    ```
    An equivalent repository address for the `clone` command is `git@github.com:pyccel/psydac.git`, which requires a GitHub account.

## Optional PETSc installation

Although PSYDAC provides several iterative linear solvers which work with our native matrices and vectors, it is often useful to access a dedicated library like [PETSc](https://petsc.org). To this end, our matrices and vectors have the method `topetsc()`, which converts them to the corresponding `petsc4py` objects.
(`petsc4py` is a Python package which provides Python bindings to PETSc.) After solving the linear system with a PETSc solver, the function `petsc_to_psydac` allows converting the solution vector back to the PSYDAC format.

In order to use these additional feature, PETSc and petsc4py must be installed as follows.
First, we download the latest release of PETSc from its [official Git repository](https://gitlab.com/petsc/petsc):
```sh
git clone --depth 1 --branch v3.24.2 https://gitlab.com/petsc/petsc.git
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
pip install wheel Cython numpy
pip install petsc/src/binding/petsc4py
```

## Running tests and examples

After installing the library, the test suite may be run by following these [instructions](https://pyccel.github.io/psydac/index.html#running-tests).
Users may also run complete examples, which can be found [here](https://pyccel.github.io/psydac/examples.html).

## Uninstall

-   **Whichever the install mode**:
    ```bash
    pip uninstall psydac
    ```
-   **If PETSc was installed**:
    ```bash
    pip uninstall petsc4py
    ```

The non-Python dependencies can be uninstalled manually using the package manager.
In the case of PETSc, it is sufficient to remove the cloned source directory given that the installation has been performed locally.
