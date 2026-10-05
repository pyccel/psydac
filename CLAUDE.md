# Project: PSYDAC

## Overview
Python 3 library for isogeometric analysis (IGA). Solve general systems of partial
differential equations (PDEs) in weak form, defined using the domain-specific
language provided by SymPDE. Supports finite element exterior calculus (FEEC)
with tensor-product spline spaces. Handles multi-patch geometries in various ways;
usually broken-FEEC a.k.a. CONGA (conforming/non-conforming Galerkin).
Python code is automatically generated for the assembly of user-defined functionals,
linear forms, and bilinear forms. This Python code is then accelerated to C/Fortran
speed using Pyccel. The library enables large parallel computations on distributed-
memory supercomputers using MPI and OpenMP.

## Tech stack
- Python 3.10+, type hints required everywhere
- meson-python + mesonpy for building
- pip with submodules (for igakit)
- pytest + pytest-cov + pytest-mpi + pytest-xdist for testing
- sympde for symbolic definition of weak formulations
- pyccel for translation of Python kernels to Fortran or C
- mpi4py for MPI parallelization
- h5py for parallel I/O
- petsc4py for direct linear solvers (optional install)

## Code standards
- Follow PEP 8 style guide whenever possible
- Docstrings of public functions and classes follow Numpydoc conventions
- pylint + black + isort for linting/typing
- Computational kernels to be accelerated with Pyccel follow `<module>_kernels.py`
- Minimize code duplication. Never add code that already exists
- Strive for concise, clean, and human-readable code. Avoid useless verbosity
- Self-explanatory names for variables, functions, and classes
- Do not reassign value to an existing variable, especially if the type changes
- Use short comments (one-liners or inline) to explain "why" rather than "what" or "how"

## Testing conventions
- Each library subpackage contains a `tests/` folder with an `__init__.py` file
- Unit tests are part of the library and shipped with it
- Unit tests can be run with `psydac test` CLI command (see `README.md`)
- Test file names follow `test_<module>.py`
- Test function names follow `test_<function>` or `test_<class>_<method>`
- Parametrize unit tests with `@pytest.parametrize` to minimize code duplication
- Keep run time of tests at a minimum
- Aim for 100% coverage on newly committed code

## File structure
- psydac/api - high-level Python interface
- psydac/cad - computer-aided design (CAD) functionality
- psydac/cmd - command-line interface (CLI) commands
- psydac/core - splines functionality
- psydac/ddm - MPI decomposition of domain, vectors of spline coefficients, and matrices
- psydac/feec - finite element exterior calculus (FEEC)
- psydac/fem - "finite element method" middle-level Python interface (spaces and fields)
- psydac/linalg - "linear algebra" low-level Python interface (vectors, linear operators, iterative solvers)
- psydac/polar - implementation of polar splines for H1 spaces
- psydac/pyccel - distillation of an old version of Pyccel for Python code generation (to be removed)
- psydac/utilities - various generic functions used across the library
- docs - Sphinx documentation
- examples - Various examples of library usage (mostly Jupyter notebooks)
- scripts - Various scripts for library maintenance
- subprojects - Git submodules

## Always do
- Before claiming done:
  . Run `python -m black --check $(git ls-files "*.py")`
  . Run `python -m isort --check $(git ls-files "*.py" | grep -v "FOUND_DUPLICATED_IMPORT.py")`
  . Run `pylint --disable=all --enable=unused-import ./psydac --ignore-paths="__pyccel*" --ignore-paths="__epyccel*"`
- Unless specified differently, assume process with `rank=0` in MPI communicator is root
- Use root MPI process to perform serial operations (e.g. open/save image)

## Never do
- Commit secrets or `.env` files
- Add new dependencies without team discussion
- Use `eval` function together with `input`
