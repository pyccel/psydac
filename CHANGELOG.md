# Change Log

All notable changes to this project will be documented in this file.

## Unreleased

### Added

-   #576 : Add a module `psydac.feec.polar.conga_projections` with broken FEEC polar projections in 2D
-   #576 : Add 2D Poisson and TE Maxwell examples on polar mapped domains in `psydac/feec/polar/examples`
-   #577 : Add an installation configuration option to choose the backend language
-   [DEVELOPER] Create action `install_petsc4py` to install PETSc & `petsc4py` w/ complex support

### Fixed

-   #576 : Require `sympde==0.19.3` which fixes a bug in the linearity checks
-   #579 : Require `h5py>=3.16` which installs correctly with `setuptools>=81.0`
-   #579 : Don't run postprocessing unit tests with `pytest-xdist` because `h5py` is not thread-safe
-   #579 : Return error code on failure of the `psydac test` and `psydac compile` commands
-   #577 : Fix installation following release of Pyccel 2.2
-   #571 : Fix correct application of the sum factorization algorithm
-   #570 : Optimize PSYDAC logo
-   #565 : Expand editable install info in `README.md`
-   #566 : Fix command `psydac test --mpi` on Ubuntu machines
-   [DEVELOPER] Add missing 'description' properties (required!) to our GitHub actions
-   [DEVELOPER] Update CI installation of `petsc4py` after release of `setuptools` 81.0
-   [DEVELOPER] Check correct reporting of failure for `psydac test` command in CI testing
-   [DEVELOPER] Use correct configuration file in coverage CI tests

### Changed

-   #576 : Use latest version of Igakit (commit dalcinl/igakit@92ee097 of 2026/07/24) which supports NumPy >= 2.4
-   #580 : Use PETSc 3.25.0 whose Python bindings `petsc4py` install correctly with `setuptools>=81.0`
-   #579 : Require `pyccel>=2.2.3` which can compile all kernels with C
-   #579 : Require `numpy>=2.1` to support Python >= 3.10
-   #579 : Require `pytest>=9.0` and use `pytest.toml` instead of `pytest.ini` for Pytest configuration
-   #579 : Move coverage configuration from `pyproject.toml` to `psydac/pytest.toml`
-   [DEVELOPER] Update GitHub Actions for repository checkout and Python setup
-   [DEVELOPER] Rename actions: `macos/ubuntu_install` -> `macos/ubuntu_installations`
-   [DEVELOPER] Do not check file changes to trigger testing workflow on PRs
-   [DEVELOPER] Run documentation workflow on pushes to `devel` whenever `README.md` is modified
-   [DEVELOPER] Run testing and documentation workflows on PRs only when set to "ready for review"

### Deprecated

### Removed

## [1.0.0] - 2026-01-19

The first official release on PyPI. A complete overhaul since version 0.1.

## [0.1] - 2020-01-14

The first beta version, not on PyPI.
