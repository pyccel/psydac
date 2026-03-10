# Change Log

All notable changes to this project will be documented in this file.

## Unreleased

### Added

-   #577 : Add an installation configuration option to choose the backend language

### Fixed

-   #577 : Fix installation following release of Pyccel 2.2
-   #571 : Fix correct application of the sum factorization algorithm
-   #570 : Optimize PSYDAC logo
-   #565 : Expand editable install info in `README.md`
-   #566 : Fix command `psydac test --mpi` on Ubuntu machines
-   [DEVELOPER] Update CI installation of `h5py` and `petsc4py` after release of `setuptools` 81.0

### Changed

-   [DEVELOPER] Do not check file changes to trigger testing workflow on PRs
-   [DEVELOPER] Run documentation workflow on pushes to `devel` whenever `README.md` is modified
-   [DEVELOPER] Run testing and documentation workflows on PRs only when set to "ready for review"

### Deprecated

### Removed

## [1.0.0] - 2026-01-19

The first official release on PyPI. A complete overhaul since version 0.1.

## [0.1] - 2020-01-14

The first beta version, not on PyPI.
