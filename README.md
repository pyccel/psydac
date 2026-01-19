# <img src="https://raw.githubusercontent.com/pyccel/psydac/devel/docs/source/logo/psydac_banner.svg" width="600" style="display: block; margin: 0 auto" alt="PSYDAC logo." class="dark-light">

[![devel_tests](https://github.com/pyccel/psydac/actions/workflows/testing.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/testing.yml)
[![docs](https://github.com/pyccel/psydac/actions/workflows/documentation.yml/badge.svg)](https://github.com/pyccel/psydac/actions/workflows/documentation.yml)

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

The associated BibTeX file can be found [here](https://github.com/pyccel/psydac/blob/devel/CITATION.bib).

## Installation

PSYDAC requires a certain number of components to be installed on the machine:

-   Fortran and C compilers with OpenMP support
-   OpenMP library
-   BLAS and LAPACK libraries
-   MPI library
-   HDF5 library with MPI support

The installation instructions depend on the operating system and on the packaging manager used.
It is particularly important to determine the **HDF5 root folder**, as this will be needed to install the [`h5py`](https://docs.h5py.org/en/latest/build.html#source-installation) package in parallel mode.
Detailed instructions can be found in the [documentation](https://pyccel.github.io/psydac/installation.html).

Once those components are installed, we recommend using [`venv`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) to set up a fresh Python virtual environment at a location `<ENV-PATH>`:
```bash
python3 -m venv <ENV-PATH>
source <ENV-PATH>/bin/activate
```

PSYDAC and its Python dependencies can now be installed in the virtual environment using [`pip`](https://pip.pypa.io/en/stable/), the Python package manager:
```bash
export CC="mpicc"
export HDF5_MPI="ON"
export HDF5_DIR=<HDF5-PATH>

pip install --upgrade pip
pip install h5py --no-cache-dir --no-binary h5py
pip install "psydac[test]"
```
Here `<HDF5-PATH>` is the path to the HDF5 root folder, such that `<HDF5-PATH>/lib/` contains the HDF5 dynamic libraries with MPI support.

The last command above installs the latest version of PSYDAC found on [PyPI](https://pypi.org), the Python Package Index, together with some optional packages needed for running the unit tests.
A developer wanting to modify the source code should skip that command, and instead clone the PSYDAC repository to perform an **editable install**:

```bash
git clone --recurse-submodules https://github.com/pyccel/psydac.git
cd psydac

pip install meson-python "pyccel>=2.1.0"
pip install --no-build-isolation --editable ".[test]"
```

Again, for more details we refer to our [documentation](https://pyccel.github.io/psydac/installation.html).

> [!TIP]
> PSYDAC provides the functionality to convert its MPI-parallel matrices and vectors to their [PETSc](https://petsc.org) equivalent, and back.
> This gives the user access to a wide variety of linear solvers and other algorithms.
> Instructions for installing [PETSc](https://petsc.org) and `petsc4py` can be found in our [documentation](https://pyccel.github.io/psydac/installation.html#id9).

## Running Tests

We strongly advice users and developers to run the test suite of PSYDAC to verify the correct installation on their machine (possibly a supercomputer).
All unit tests are based on [`pytest`](https://docs.pytest.org/en/stable/) and are installed together with the library.
For convenience, PSYDAC provides the `psydac test` command as shown below.

In order to run all serial and parallel tests which do not use PETSc, just type:
```bash
psydac test
psydac test --mpi
```

If PETSc and petsc4py were installed, additional serial and parallel tests can be run:
```bash
psydac test --petsc
psydac test --petsc --mpi
```

## Speeding up PSYDAC's core

Many of PSYDAC's low-level Python functions can be translated to a compiled language using the [Pyccel](https://github.com/pyccel/pyccel) transpiler.
Currently, all of those functions are collected in modules which follow the name pattern `[module]_kernels.py`.

For both classical and editable installations, *all kernel files are translated to Fortran __without user intervention__*.
If the user adds or edits a kernel file within an editable install, they should use the command `psydac compile` in order to be able to see the changes at runtime.
This command applies Pyccel to all the kernel files in the source directory.
The default language is Fortran, and C is also available.

-   **Only in development mode**:
    ```bash
    psydac compile [--language {fortran, c}]
    ```

## Examples and Tutorials

Our [documentation](https://pyccel.github.io/psydac/examples.html) provides Jupyter notebooks that present many aspects of this library. 
Additional [tutorials](https://pyccel.github.io/IGA-Python/intro.html) on isogeometric analysis, with many example notebooks where various PDEs are solved with PSYDAC, is under construction in the [IGA-Python](https://github.com/pyccel/IGA-Python) repository.
Some other examples can be found [here](https://github.com/pyccel/psydac/blob/devel/examples).

## Library Documentation

-   [Output formats](https://pyccel.github.io/psydac/output.html)
-   [Mesh generation](https://pyccel.github.io/psydac/psydac-mesh.html)
-   [Modules](https://pyccel.github.io/psydac/modules.html)

## Contributing

There are several ways to contribute to this project!

If you find a problem, please check if this is already discussed in one of [our issues](https://github.com/pyccel/psydac/issues) and feel free to add your opinion; if not, please create a [new issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue).
If you want to fix an issue, improve our notebooks, or add a new example, please [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) our Git repository, make and commit your changes, and create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) (PRs).
All PRs are reviewed by the project maintainers.
During the PR review, GitHub workflows are triggered on various platforms.

We keep an up-to-date list of maintainers and contributors in our [AUTHORS](https://github.com/pyccel/psydac/blob/devel/AUTHORS) file.
Thank you!
