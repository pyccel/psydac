<h1 align="center">
<img src="docs/source/logo/psydac_banner.svg" width="500" alt="Shows the psydac logo.">
</h1><br>

[![devel_tests](https://github.com/pyccel/psydac/actions/workflows/testing.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/testing.yml) [![docs](https://github.com/pyccel/psydac/actions/workflows/documentation.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/documentation.yml)

PSYDAC is a Python 3 library for isogeometric analysis.
It is an academic, open-source project created by numerical mathematicians at the [Max Planck Institute for Plasma Physics](https://www.ipp.mpg.de/en) ([NMPP](https://www.ipp.mpg.de/ippcms/eng/for/bereiche/numerik) division, [FEM](https://www.ipp.mpg.de/5150531/fem) group).

PSYDAC can solve general systems of partial differential equations in weak form, which users define using the domain-specific language provided by [SymPDE](https://github.com/pyccel/sympde).
It supports finite element exterior calculus ([FEEC](https://en.wikipedia.org/wiki/Finite_element_exterior_calculus)) with tensor-product spline spaces and handles multi-patch geometries in various ways.

PSYDAC automatically generates Python code for the assembly of user-defined functionals and linear and bilinear forms from the weak formulation of the problem.
This Python code is then accelerated to C/Fortran speed using [Pyccel](https://github.com/pyccel/pyccel).
The library also enables large parallel computations on distributed-memory supercomputers using [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) and [OpenMP](https://en.wikipedia.org/wiki/OpenMP).

> [!NOTE]
> The name PSYDAC stands for "Python Spline librarY for Differential equations with Automatic Code generation".
> It is pronounced like the famous Pokémon character, from which the developers draw inspiration for its psychic powers.

## Citing

If PSYDAC has been significant in your research, and you would like to acknowledge the project in your academic publication, we would ask that you cite the following paper:

Güçlü, Y., S. Hadjout, and A. Ratnani. “PSYDAC: A High-Performance IGA Library in Python.” In 8th European Congress on Computational Methods in Applied Sciences and Engineering. CIMNE, 2022. https://doi.org/10.23967/eccomas.2022.227.

The associated BibTeX file can be found [here](./CITATION.bib).

## Installation

PSYDAC requires a certain number of components to be installed on the machine:

-   Fortran and C compilers with OpenMP support
-   OpenMP library
-   BLAS and LAPACK libraries
-   MPI library
-   HDF5 library with MPI support

The installation instructions depend on the operating system and on the packaging manager used.
It is particularly important to determine the **HDF5 root folder**, as this will be needed to install the [`h5py`](https://docs.h5py.org/en/latest/build.html#source-installation) package in parallel mode.
Detailed instructions can be found in the [documentation](./docs/installation.md).

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
pip install ./psydac
```
Here `<HDF5-PATH>` is the path to the HDF5 root folder, such that `<HDF5-PATH>/lib/` contains the HDF5 dynamic libraries with MPI support.
For an editable install, the `-e/--editable` flag should be provided to the last command above.

Again, for more details we refer to our [documentation](./docs/installation.md).

> [!TIP]
> PSYDAC provides the functionality to convert its MPI-parallel matrices and vectors to their [PETSc](https://petsc.org) equivalent, and back.
> This gives the user access to a wide variety of linear solvers and other algorithms.
> Instructions for installing [PETSc](https://petsc.org) and `petsc4py` can be found in our [documentation](.docs/installation.md#optional-petsc-installation).

## Running Tests

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

## Speeding up PSYDAC's core

Many of PSYDAC's low-level Python functions can be translated to a compiled language using the [Pyccel](https://github.com/pyccel/pyccel) transpiler. Currently, all of those functions are collected in modules which follow the name pattern `[module]_kernels.py`.

The classical installation translates all kernel files to Fortran without user intervention. This does not happen in the case of an editable install, but the command `psydac-accelerate` is made available to the user instead. This command applies Pyccel to all the kernel files in the source directory. The default language is currently Fortran, C should also be supported in a near future.

-   **Only in development mode**:
    ```bash
    python /path/to/psydac/psydac_accelerate.py [--language LANGUAGE] [--openmp]
    ```

## Examples and Tutorials

A [tutorial](https://pyccel.github.io/IGA-Python/intro.html) on isogeometric analysis, with many example notebooks where various PDEs are solved with PSYDAC, is under construction in the [IGA-Python](https://github.com/pyccel/IGA-Python) repository.
Some other examples can be found [here](./examples/).

## Library Documentation

-   [Output formats](./docs/output.md)
-   [Mesh generation](./docs/psydac-mesh.md)
-   [Library reference](https://pyccel.github.io/psydac/)

## Contributing

There are several ways to contribute to this project!

If you find a problem, please check if this is already discussed in one of [our issues](https://github.com/pyccel/psydac/issues) and feel free to add your opinion; if not, please create a [new issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue).
If you want to fix an issue, improve our notebooks, or add a new example, please [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) our Git repository, make and commit your changes, and create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) (PRs).
All PRs are reviewed by the project maintainers.
During the PR review, GitHub workflows are triggered on various platforms.

We keep an up-to-date list of maintainers and contributors in our [AUTHORS](./AUTHORS) file.
Thank you!
